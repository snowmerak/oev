import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("compressed_tensors")

from compressed_tensors.compressors import ModelCompressor, PackedQuantizationCompressor
from compressed_tensors.quantization import (
    QuantizationArgs, QuantizationConfig, QuantizationScheme, apply_quantization_config,
)
from compressed_tensors.utils import get_direct_state_dict

spec = importlib.util.spec_from_file_location(
    "qat_probe", Path(__file__).parents[1] / "scripts" / "probe_ct_w4a16.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def compressed_linear(dtype, group_size=32):
    model = torch.nn.Sequential(torch.nn.Linear(64, 32, bias=True, dtype=dtype)).eval()
    config = QuantizationConfig(
        config_groups={"weights": QuantizationScheme(
            targets=["Linear"], format="pack-quantized",
            weights=QuantizationArgs(num_bits=4, type="int", symmetric=True,
                                     strategy="group", group_size=group_size),
        )},
        format="pack-quantized", quantization_status="frozen",
    )
    apply_quantization_config(model, config)
    model[0].weight_scale.data.uniform_(0.01, 0.05)
    ModelCompressor(quantization_config=config).compress_model(model)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_on_demand_matches_ct_decompression_and_preserves_packed_weights(dtype):
    model = compressed_linear(dtype)
    layer = model[0]
    packed = layer.weight_packed.clone()
    reference = PackedQuantizationCompressor.decompress(
        get_direct_state_dict(layer), layer.quantization_scheme,
    )["weight"]
    inputs = torch.randn(2, 3, 64, dtype=dtype)
    expected = torch.nn.functional.linear(inputs, reference, layer.bias)
    assert probe.enable_on_demand(model) == 1
    assert not hasattr(model, "ct_decompress_hook")
    with torch.inference_mode():
        for _ in range(2):
            torch.testing.assert_close(model(inputs), expected, rtol=0, atol=0)
    assert torch.equal(layer.weight_packed, packed)
    assert not hasattr(layer, "weight")


def test_on_demand_rejects_unsupported_group_before_removing_hook():
    model = compressed_linear(torch.float32, group_size=64)
    with pytest.raises(ValueError, match="expected symmetric W4A16 group 32"):
        probe.enable_on_demand(model)
    assert hasattr(model, "ct_decompress_hook")


def test_unified_on_demand_inference_preserves_packed_weights_with_selective_head(monkeypatch):
    transformers = pytest.importorskip("transformers")
    from oev.hf import HuggingFaceBackend

    config = transformers.Gemma4UnifiedTextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=128, sliding_window=32,
        layer_types=["sliding_attention", "full_attention"], final_logit_softcapping=2.0,
    )
    model = transformers.Gemma4UnifiedForConditionalGeneration(
        transformers.Gemma4UnifiedConfig(text_config=config)
    ).eval()
    quantization = QuantizationConfig(
        config_groups={"weights": QuantizationScheme(
            targets=["Linear"], format="pack-quantized",
            weights=QuantizationArgs(
                num_bits=4, type="int", symmetric=True, strategy="group", group_size=32,
            ),
        )},
        ignore=["lm_head"], format="pack-quantized", quantization_status="frozen",
    )
    apply_quantization_config(model, quantization)
    for layer in model.modules():
        if hasattr(layer, "weight_scale"):
            layer.weight_scale.data.fill_(0.01)
    ModelCompressor(quantization_config=quantization).compress_model(model)
    packed = {
        name: layer.weight_packed.clone()
        for name, layer in model.named_modules() if hasattr(layer, "weight_packed")
    }
    assert probe.enable_on_demand(model) == len(packed)
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-unified-packed")
    ids, slots = [2, 4, 8, 12], [5, 7]
    with torch.inference_mode():
        expected = model(
            input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1,
        ).logits[0, -1, slots].tolist()

    def no_full_projection(*args, **kwargs):
        raise AssertionError("full-vocabulary projection was called")

    monkeypatch.setattr(model.lm_head, "forward", no_full_projection)
    for _ in range(2):
        assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)
    for name, layer in model.named_modules():
        if name in packed:
            assert torch.equal(layer.weight_packed, packed[name])
            assert not hasattr(layer, "weight")
