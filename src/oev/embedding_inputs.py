"""Validate inline media before handing decoded inputs to a model processor."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import math
import warnings


MAX_MEDIA_BYTES = 20 * 1024 * 1024
MAX_BATCH_MEDIA_BYTES = 64 * 1024 * 1024
MAX_MEDIA_ITEMS = 16
MAX_IMAGE_PIXELS = 16_000_000
MAX_AUDIO_SECONDS = 330
MAX_AUDIO_SAMPLES = 32_000_000
AUDIO_SAMPLE_RATE = 16_000


def configure_audio_processor(processor):
    # Some saved processors cap placeholder counts at 280/750 audio tokens.
    # The embedding encoder supports the full 8192-token window; count every
    # audio token so a long recording is rejected by budget rather than clipped.
    if hasattr(processor, "audio_seq_length"):
        processor.audio_seq_length = max(processor.audio_seq_length, 8192)


def media_bytes(value, modality):
    if not isinstance(value, str):
        raise ValueError(f"{modality} must be a base64 data URL")
    header, separator, encoded = value.partition(",")
    if not separator or not header.startswith(f"data:{modality}/") or not header.endswith(";base64"):
        raise ValueError(f"{modality} must be a base64 data URL; remote URLs and server file paths are unsupported")
    if not encoded or len(encoded) > 4 * ((MAX_MEDIA_BYTES + 2) // 3):
        raise ValueError(f"{modality} must contain 1 to {MAX_MEDIA_BYTES} decoded bytes")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"{modality} contains invalid base64") from exc
    if not data or len(data) > MAX_MEDIA_BYTES:
        raise ValueError(f"{modality} must contain 1 to {MAX_MEDIA_BYTES} decoded bytes")
    return data


def validate_inputs(inputs):
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= 128:
        raise ValueError("input must contain 1 to 128 nonempty text strings or modality objects")
    total_bytes = 0
    for item in inputs:
        if isinstance(item, str):
            if not item.strip():
                raise ValueError("input text must be nonempty")
            continue
        if not isinstance(item, dict) or not item or item.keys() - {"text", "image", "audio"}:
            raise ValueError("input objects must contain text, image, or audio")
        if "text" in item and (not isinstance(item["text"], str) or not item["text"].strip()):
            raise ValueError("input text must be nonempty")
        count = 0
        for modality in ("image", "audio"):
            if modality not in item:
                continue
            values = item[modality] if isinstance(item[modality], list) else [item[modality]]
            if not values:
                raise ValueError(f"{modality} must be nonempty")
            count += len(values)
            if count > MAX_MEDIA_ITEMS:
                raise ValueError(f"each input may contain at most {MAX_MEDIA_ITEMS} media items")
            for value in values:
                total_bytes += len(media_bytes(value, modality))
                if total_bytes > MAX_BATCH_MEDIA_BYTES:
                    raise ValueError(f"input media exceeds the {MAX_BATCH_MEDIA_BYTES}-byte batch limit")


def decode_image(value):
    from PIL import Image, ImageOps

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(media_bytes(value, "image"))) as image:
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"image exceeds {MAX_IMAGE_PIXELS} pixels")
                return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError(f"invalid image: {exc}") from exc


def decode_audio(value):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        with sf.SoundFile(BytesIO(media_bytes(value, "audio"))) as audio:
            if audio.frames < 1 or audio.samplerate < 1 or audio.frames / audio.samplerate > MAX_AUDIO_SECONDS:
                raise ValueError(f"audio must be nonempty and at most {MAX_AUDIO_SECONDS} seconds long")
            if audio.channels > 8:
                raise ValueError("audio may have at most 8 channels")
            if audio.samplerate > 192_000 or audio.frames * audio.channels > MAX_AUDIO_SAMPLES:
                raise ValueError("audio exceeds the decoded sample budget (32 million samples, at most 192 kHz)")
            samples = audio.read(dtype="float32", always_2d=True).mean(axis=1)
            sample_rate = audio.samplerate
        if not np.isfinite(samples).all():
            raise ValueError("audio samples must be finite")
        if sample_rate != AUDIO_SAMPLE_RATE:
            divisor = math.gcd(sample_rate, AUDIO_SAMPLE_RATE)
            samples = resample_poly(samples, AUDIO_SAMPLE_RATE // divisor, sample_rate // divisor)
        return {"array": samples, "sampling_rate": AUDIO_SAMPLE_RATE}
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid audio: {exc}") from exc


def decode_inputs(inputs):
    decoded = []
    for item in inputs:
        # Explicit text keys keep text resembling a file/URL from being loaded by
        # Sentence Transformers' automatic modality detection.
        if isinstance(item, str):
            decoded.append({"text": item})
            continue
        result = {}
        for key, value in item.items():
            if key == "text":
                result[key] = value
            else:
                decode = decode_image if key == "image" else decode_audio
                result[key] = [decode(part) for part in value] if isinstance(value, list) else decode(value)
        decoded.append(result)
    return decoded
