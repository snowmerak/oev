import json
from types import SimpleNamespace

import pytest

from oev.hf_embedding_metadata import local_pooling_defaults, pooling_loader


POOLING = "sentence_transformers.models.Pooling"


def checkpoint(tmp_path, name="Qwen3-Embedding-0.6B", modules=None):
    path = tmp_path / name
    path.mkdir(parents=True)
    (path / "modules.json").write_text(json.dumps(modules or [{"type": POOLING, "path": "1_Pooling"}]))
    return path


@pytest.mark.parametrize("name,dimension", [("Qwen3-Embedding-0.6B", 1024), ("Qwen3-Embedding-4B", 2560), ("Qwen3-Embedding-8B", 4096)])
def test_missing_qwen_pooling_is_recovered_without_writing(tmp_path, name, dimension, caplog):
    path = checkpoint(tmp_path, name)
    config = SimpleNamespace(model_type="qwen3", hidden_size=dimension)
    assert local_pooling_defaults(path, config) == {
        POOLING: {"embedding_dimension": dimension, "pooling_mode": "lasttoken", "include_prompt": True}
    }
    assert not (path / "1_Pooling").exists()
    assert "in memory" in caplog.text


def test_hf_cache_directory_identifies_qwen(tmp_path):
    path = checkpoint(tmp_path / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots")
    path.rename(path.parent / "snapshot-sha")
    path = path.parent / "snapshot-sha"
    assert local_pooling_defaults(path, SimpleNamespace(model_type="qwen3", hidden_size=1024))[POOLING]["pooling_mode"] == "lasttoken"


def test_gemma_pooling_uses_projection_width_not_hidden_width(tmp_path):
    path = checkpoint(tmp_path, "local-gemma")
    config = SimpleNamespace(model_type="embedding_gemma2", text_config=SimpleNamespace(hidden_size=512, embedding_dim=768))
    assert local_pooling_defaults(path, config)[POOLING] == {
        "embedding_dimension": 768, "pooling_mode": "mean", "include_prompt": True
    }


def test_existing_pooling_is_preserved_even_for_unknown_model(tmp_path):
    path = checkpoint(tmp_path, "custom-model")
    folder = path / "1_Pooling"
    folder.mkdir()
    original = b'{"embedding_dimension": 7, "pooling_mode": "cls"}'
    (folder / "config.json").write_bytes(original)
    assert local_pooling_defaults(path, SimpleNamespace(model_type="custom")) == {}
    assert (folder / "config.json").read_bytes() == original


@pytest.mark.parametrize("architecture,name", [("qwen3", "renamed-checkpoint"), ("bert", "Qwen3-Embedding-0.6B")])
def test_unknown_model_has_actionable_error(tmp_path, architecture, name):
    path = checkpoint(tmp_path, name)
    with pytest.raises(ValueError, match="cannot infer.*pooling safely"):
        local_pooling_defaults(path, SimpleNamespace(model_type=architecture, hidden_size=1024))


@pytest.mark.parametrize("dimension", [None, 0, -1, True])
def test_missing_dimension_is_not_guessed(tmp_path, dimension):
    path = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="does not declare.*dimension"):
        local_pooling_defaults(path, SimpleNamespace(model_type="qwen3", hidden_size=dimension))


def test_multiple_pooling_modules_are_not_guessed(tmp_path):
    path = checkpoint(tmp_path, modules=[{"type": POOLING, "path": "1_Pooling"}, {"type": POOLING, "path": "2_Pooling"}])
    with pytest.raises(ValueError, match="restore.*metadata"):
        local_pooling_defaults(path, SimpleNamespace(model_type="qwen3", hidden_size=1024))


def test_pooling_path_must_stay_in_checkpoint(tmp_path):
    path = checkpoint(tmp_path, modules=[{"type": POOLING, "path": "../../outside"}])
    with pytest.raises(ValueError, match="escapes model directory"):
        local_pooling_defaults(path, SimpleNamespace(model_type="qwen3", hidden_size=1024))


def test_nonlocal_or_no_modules_uses_original_loader(tmp_path):
    assert local_pooling_defaults(tmp_path / "absent", None) == {}
    assert local_pooling_defaults(tmp_path, None) == {}
    original = object()
    assert pooling_loader(original, {}) is original


def test_defaults_bridge_preserves_other_module_defaults():
    class Base:
        def _get_module_init_defaults(self, class_ref):
            return {"other": 42}

    loader = pooling_loader(Base, {POOLING: {"embedding_dimension": 1024}})
    model = loader()
    assert model._get_module_init_defaults(POOLING) == {"other": 42, "embedding_dimension": 1024}
    assert model._get_module_init_defaults("custom.Dense") == {"other": 42}


def test_real_pooling_load_accepts_defaults_and_saved_config_wins(tmp_path):
    pytest.importorskip("sentence_transformers")
    from sentence_transformers.sentence_transformer.modules import Pooling

    defaults = {"embedding_dimension": 1024, "pooling_mode": "lasttoken", "include_prompt": True}
    recovered = Pooling.load(str(tmp_path), local_files_only=True, init_defaults=defaults)
    assert recovered.embedding_dimension == 1024 and recovered.pooling_mode == "lasttoken"
    (tmp_path / "config.json").write_text(json.dumps({"word_embedding_dimension": 3, "pooling_mode_cls_token": True}))
    saved = Pooling.load(str(tmp_path), local_files_only=True, init_defaults=defaults)
    assert saved.embedding_dimension == 3 and saved.pooling_mode == "cls"
