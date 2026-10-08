import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from oev.hf import HuggingFaceBackend


def test_hf_backend_reads_only_requested_final_position_logits():
    config = transformers.GPT2Config(
        vocab_size=64, n_positions=32, n_ctx=32, n_embd=32, n_layer=1, n_head=2
    )
    model = transformers.GPT2LMHeadModel(config).eval()
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-gpt2")
    ids = [1, 4, 8, 12]
    slots = [2, 7, 11]
    with torch.inference_mode():
        expected = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1, slots].tolist()
    actual = backend.selected_logits(ids, slots)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_gemma4_last_position_optimization_matches_full_logits():
    config = transformers.Gemma4TextConfig(
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        hidden_size_per_layer_input=8,
        max_position_embeddings=128,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma4ForCausalLM(config).eval()
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-gemma4")
    assert backend._keep_last_only
    ids = [2, 4, 8, 12]
    slots = [5, 7]
    with torch.inference_mode():
        expected = model(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            use_cache=False,
            logits_to_keep=0,
        ).logits[0, -1, slots].tolist()
    assert backend._selective_gemma4
    assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)

    # The optimized path never invokes the full-vocabulary lm_head forward.
    def no_full_projection(*args, **kwargs):
        raise AssertionError("full-vocabulary projection was called")

    model.lm_head.forward = no_full_projection
    assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)


def test_compressed_gemma4_runs_decompression_hook_before_selective_projection():
    pytest.importorskip("compressed_tensors")
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationConfig,
        QuantizationScheme,
        apply_quantization_config,
    )

    config = transformers.Gemma4TextConfig(
        vocab_size=64, vocab_size_per_layer_input=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, hidden_size_per_layer_input=32,
        max_position_embeddings=128, sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
    )
    model = transformers.Gemma4ForCausalLM(config).eval()
    quantization = QuantizationConfig(
        config_groups={
            "weights": QuantizationScheme(
                targets=["Linear"], format="pack-quantized",
                weights=QuantizationArgs(
                    num_bits=4, type="int", symmetric=True,
                    strategy="group", group_size=32,
                ),
            )
        },
        ignore=["lm_head"], format="pack-quantized", quantization_status="frozen",
    )
    apply_quantization_config(model, quantization)
    for module in model.modules():
        if hasattr(module, "weight_scale"):
            module.weight_scale.data.fill_(0.01)
    ModelCompressor(quantization_config=quantization).compress_model(model)
    assert hasattr(model, "ct_decompress_hook")
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-ct-gemma4")
    ids, slots = [2, 4, 8, 12], [5, 7]

    # The first call must reach the model's root hook, not its decoder directly.
    actual = backend.selected_logits(ids, slots)
    assert not hasattr(model, "ct_decompress_hook")
    with torch.inference_mode():
        expected = model(
            input_ids=torch.tensor([ids]), use_cache=False, logits_to_keep=1,
        ).logits[0, -1, slots].tolist()
    assert actual == pytest.approx(expected, abs=1e-6)

    # Once decompressed, subsequent calls can skip full-vocabulary projection.
    def no_full_projection(*args, **kwargs):
        raise AssertionError("full-vocabulary projection was called")

    model.lm_head.forward = no_full_projection
    assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)


def test_gemma4_checkpoint_class_uses_selective_projection_and_softcap():
    text_config = transformers.Gemma4TextConfig(
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        hidden_size_per_layer_input=8,
        max_position_embeddings=128,
        sliding_window=32,
        layer_types=["sliding_attention", "full_attention"],
        final_logit_softcapping=2.0,
    )
    vision_config = transformers.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        position_embedding_size=128,
    )
    audio_config = transformers.Gemma4AudioConfig(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=32
    )
    config = transformers.Gemma4Config(
        text_config=text_config, vision_config=vision_config, audio_config=audio_config
    )
    model = transformers.Gemma4ForConditionalGeneration(config).eval()
    backend = HuggingFaceBackend(model, tokenizer=None, model_name="tiny-gemma4-checkpoint-class")
    assert backend._selective_gemma4
    ids = [2, 4, 8, 12]
    slots = [5, 7]
    with torch.inference_mode():
        expected = model(
            input_ids=torch.tensor([ids]),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
            use_cache=False,
            logits_to_keep=0,
        ).logits[0, -1, slots].tolist()
    assert backend.selected_logits(ids, slots) == pytest.approx(expected, abs=1e-6)
