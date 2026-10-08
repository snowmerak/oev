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
