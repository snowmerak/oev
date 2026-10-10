from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from oev.hf import HuggingFaceBackend
from oev.hybrid import HybridGemma4Backend


IDS = [2, 4, 8, 12]
SLOTS = [5, 7]


def gemma4_model():
    text = transformers.Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, hidden_size_per_layer_input=32,
        max_position_embeddings=128, sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
        final_logit_softcapping=2.0,
    )
    vision = transformers.Gemma4VisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        position_embedding_size=128,
    )
    audio = transformers.Gemma4AudioConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=32,
    )
    return transformers.Gemma4ForConditionalGeneration(
        transformers.Gemma4Config(text_config=text, vision_config=vision, audio_config=audio)
    ).eval()


def unified_model(class_name, cap=2.0):
    text = transformers.Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, sliding_window=32,
        layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=cap,
    )
    config = (
        text if class_name == "Gemma4UnifiedForCausalLM"
        else transformers.Gemma4UnifiedConfig(text_config=text)
    )
    return getattr(transformers, class_name)(config).eval()


def compress(model):
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import (
        QuantizationArgs, QuantizationConfig, QuantizationScheme, apply_quantization_config,
    )

    config = QuantizationConfig(
        config_groups={"weights": QuantizationScheme(
            targets=["Linear"], format="pack-quantized",
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True, strategy="group", group_size=32,
            ),
        )},
        ignore=["lm_head"], format="pack-quantized", quantization_status="frozen",
    )
    apply_quantization_config(model, config)
    for module in model.modules():
        if hasattr(module, "weight_scale"):
            module.weight_scale.data.fill_(0.01)
    compressor = ModelCompressor(quantization_config=config)
    compressor.compress_model(model)
    return compressor


def full_logits(model):
    with torch.inference_mode():
        return model(
            input_ids=torch.tensor([IDS]), use_cache=False, logits_to_keep=1,
        ).logits[0, -1, SLOTS].float().tolist()


def no_full_projection(*args, **kwargs):
    raise AssertionError("full-vocabulary projection was called")


@pytest.mark.parametrize("class_name", [
    "Gemma4UnifiedForCausalLM", "Gemma4UnifiedForConditionalGeneration",
])
@pytest.mark.parametrize("cap", [None, 2.0])
def test_unified_selective_logits_match_full_projection(class_name, cap, monkeypatch):
    model = unified_model(class_name, cap)
    expected = full_logits(model)
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-unified")
    monkeypatch.setattr(model.lm_head, "forward", no_full_projection)
    assert backend.selected_logits(IDS, SLOTS) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("unified", [False, True])
def test_concurrent_ct_initialization_runs_once_then_allows_overlap(unified, monkeypatch):
    model = unified_model("Gemma4UnifiedForConditionalGeneration") if unified else gemma4_model()
    compressor = compress(model)
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-ct")
    original = compressor.decompress_model
    entered, second_started, competing, release = Event(), Event(), Event(), Event()
    counter_lock = Lock()
    calls = 0

    def delayed_decompress(model):
        nonlocal calls
        with counter_lock:
            calls += 1
            if calls > 1:
                competing.set()
        entered.set()
        assert release.wait(timeout=10)
        return original(model)

    def second_inference():
        second_started.set()
        return backend.selected_logits(IDS, SLOTS)

    monkeypatch.setattr(compressor, "decompress_model", delayed_decompress)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(backend.selected_logits, IDS, SLOTS)
        try:
            assert entered.wait(timeout=10)
            second = pool.submit(second_inference)
            assert second_started.wait(timeout=10)
            assert not competing.wait(timeout=0.1)
        finally:
            release.set()
        actual = [first.result(timeout=10), second.result(timeout=10)]
    assert calls == 1
    assert not hasattr(model, "ct_decompress_hook")
    expected = full_logits(model)
    for scores in actual:
        assert scores == pytest.approx(expected, abs=1e-6)

    # Once weights are stable, a lock must not serialize decoder forwards.
    barrier = Barrier(2)
    decoder_forward = model.model.forward

    def concurrent_decoder(*args, **kwargs):
        barrier.wait(timeout=10)
        return decoder_forward(*args, **kwargs)

    monkeypatch.setattr(model.model, "forward", concurrent_decoder)
    monkeypatch.setattr(model.lm_head, "forward", no_full_projection)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(backend.selected_logits, IDS, SLOTS) for _ in range(2)]
        for future in futures:
            assert future.result(timeout=10) == pytest.approx(expected, abs=1e-6)


def test_hybrid_decompresses_on_cpu_before_cuda_placement(monkeypatch):
    model = gemma4_model().to(torch.bfloat16)
    compress(model)
    original_to = torch.nn.Module.to
    placements = []

    def observe_placement(module, *args, **kwargs):
        if args == ("cuda",):
            assert not hasattr(model, "ct_decompress_hook")
            for layer in model.modules():
                assert not hasattr(layer, "weight_packed")
            assert all(parameter.device.type == "cpu" for parameter in model.parameters())
            assert all(not torch.is_inference(parameter) for parameter in model.parameters())
            placements.append(module)
            return module
        return original_to(module, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Module, "to", observe_placement)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    # Initialization must expand weights without running a dummy inference.
    monkeypatch.setattr(model, "forward", no_full_projection)
    backend = HybridGemma4Backend(model, tokenizer=None, model_name="tiny-hybrid-ct")
    assert placements
    assert not hasattr(model, "ct_decompress_hook")
    with torch.inference_mode():
        hidden = backend.text_model(input_ids=torch.tensor([IDS]), use_cache=False).last_hidden_state
    assert hidden.shape == (1, len(IDS), 32)
    assert torch.isfinite(hidden).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires CUDA with bfloat16 support",
)
@pytest.mark.parametrize("quantized", [False, True])
def test_hybrid_cuda_logits_match_cpu_full_projection(quantized):
    model = gemma4_model().to(torch.bfloat16)
    if quantized:
        compress(model)
        # Use an independent model for the CT reference so the hybrid model
        # retains its initial decompression hook until backend construction.
        reference = gemma4_model().to(torch.bfloat16)
        compress(reference)
        reference.load_state_dict(model.state_dict())
    else:
        reference = model
    expected = full_logits(reference)
    backend = HybridGemma4Backend(model, tokenizer=None, model_name="tiny-hybrid")
    assert not hasattr(model, "ct_decompress_hook")
    assert backend.text_model.embed_tokens.weight.device.type == "cpu"
    assert model.lm_head.weight.device.type == "cpu"
    assert next(backend.text_model.layers.parameters()).device.type == "cuda"
    for _ in range(2):
        assert backend.selected_logits(IDS, SLOTS) == pytest.approx(expected, abs=0.005)
