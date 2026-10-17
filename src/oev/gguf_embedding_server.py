"""Own a native llama-server for architectures newer than the Python binding."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import weakref

from .embedding_inputs import decode_inputs


def _stop(process, log):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log.close()


class GGUFServerEmbeddingBackend:
    """One model, one local native worker; no public auxiliary listener."""

    def __init__(self, source, executable, *, context_length, gpu_layers, flash_attn, pooling, mmproj_path=None):
        binary = Path(executable).expanduser().resolve()
        if not binary.is_file():
            raise ValueError("gguf_server_path must point to a llama-server executable")
        projector = Path(mmproj_path).expanduser().resolve() if mmproj_path is not None else None
        if projector is not None and not projector.is_file():
            raise ValueError("mmproj_path must point to a multimodal projector GGUF file")
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.context_window = context_length
        self.prompts = {}
        self.modalities = frozenset({"text"})
        # Avoid proxying localhost and accidentally forwarding model inputs.
        self._http = build_opener(ProxyHandler({}))
        command = [str(binary), "--model", str(Path(source).expanduser().resolve()),
                   "--embedding", "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", str(context_length), "--batch-size", str(context_length),
                   "--ubatch-size", str(context_length), "--parallel", "1",
                   "--n-gpu-layers", str(gpu_layers), "--flash-attn", "on" if flash_attn else "off"]
        if pooling != "auto":
            command.extend(["--pooling", pooling])
        if projector is not None:
            command.extend(["--mmproj", str(projector)])
        log = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen(
                command, cwd=binary.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except Exception:
            log.close()
            raise
        # Close on app shutdown, startup failure, garbage collection, and interpreter exit.
        self._cleanup = weakref.finalize(self, _stop, self._process, log)
        try:
            deadline = time.monotonic() + 120
            while True:
                if self._process.poll() is not None or time.monotonic() >= deadline:
                    log.seek(0)
                    detail = log.read()[-2000:].decode("utf-8", errors="replace")
                    raise RuntimeError(f"native GGUF embedding worker did not start: {detail}")
                try:
                    self._request("/health", timeout=1)
                    break
                except (HTTPError, URLError, TimeoutError):
                    time.sleep(0.1)
            # Read actual projected output width, rather than the transformer's hidden
            # width (EmbeddingGemma2 is 512 hidden dimensions -> 768 output dimensions).
            probe = self._request("/v1/embeddings", {"input": "warmup", "encoding_format": "float"})
            self.dimension = len(probe["data"][0]["embedding"])
            if self.dimension < 1:
                raise RuntimeError("native GGUF embedding worker returned no pooled vector")
            if projector is not None:
                supported = self._request("/props")["modalities"]
                self.modalities = frozenset({"text"} | ({"image"} if supported.get("vision") else set())
                                            | ({"audio"} if supported.get("audio") else set()))
                if self.modalities == {"text"}:
                    raise ValueError("native GGUF worker did not load an image or audio encoder")
        except Exception:
            self.close()
            raise

    def _request(self, path, body=None, *, timeout=120):
        request = Request(self.base_url + path,
                          data=None if body is None else json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with self._http.open(request, timeout=timeout) as response:
            return json.load(response)

    def embed(self, texts, max_input_tokens):
        from .embedding_backends import _check_tokens

        if self._process.poll() is not None:
            raise RuntimeError("native GGUF embedding worker has exited")
        counts = [len(self._request("/tokenize", {"content": text, "add_special": True})["tokens"]) for text in texts]
        _check_tokens(counts, min(max_input_tokens, self.context_window))
        body = self._request("/v1/embeddings", {"input": texts, "encoding_format": "float"})
        items = sorted(body["data"], key=lambda item: item["index"])
        if [item["index"] for item in items] != list(range(len(texts))):
            raise RuntimeError("native GGUF worker returned invalid embedding indices")
        return [item["embedding"] for item in items], body["usage"]["prompt_tokens"]

    def close(self):
        self._cleanup()

    def embed_multimodal(self, inputs, max_input_tokens, *, prompt=""):
        if self._process.poll() is not None:
            raise RuntimeError("native GGUF embedding worker has exited")
        # The native server counts expanded media tokens itself and rejects context
        # overflow. Its context must enforce the service's configured token limit.
        if max_input_tokens < self.context_window:
            raise ValueError("GGUF multimodal worker context exceeds max_input_tokens; reload it with the smaller context")
        decoded = decode_inputs(inputs)
        payload = []
        for item in decoded:
            content = [{"type": "text", "text": prompt}] if prompt else []
            for key, value in item.items():
                for part in value if isinstance(value, list) else [value]:
                    if key == "text":
                        content.append({"type": "text", "text": part})
                        continue
                    buffer = BytesIO()
                    if key == "image":
                        part.save(buffer, format="PNG")
                        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
                        content.append({"type": "image_url", "image_url": {"url": url}})
                    else:
                        import soundfile as sf

                        sf.write(buffer, part["array"], part["sampling_rate"], format="WAV", subtype="FLOAT")
                        data = base64.b64encode(buffer.getvalue()).decode("ascii")
                        content.append({"type": "input_audio", "input_audio": {"data": data, "format": "wav"}})
            payload.append({"content": content})
        try:
            body = self._request("/v1/embeddings", {"input": payload, "encoding_format": "float"})
        except HTTPError as exc:
            detail = json.load(exc).get("error", {}).get("message", "native GGUF embedding request failed")
            if 400 <= exc.code < 500 or "too large" in detail or "context size" in detail:
                raise ValueError(detail) from exc
            raise RuntimeError(detail) from exc
        items = sorted(body["data"], key=lambda item: item["index"])
        if [item["index"] for item in items] != list(range(len(inputs))):
            raise RuntimeError("native GGUF worker returned invalid embedding indices")
        return [item["embedding"] for item in items], body["usage"]["prompt_tokens"]
