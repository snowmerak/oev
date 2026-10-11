"""Recover known pooling metadata without changing a local model directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)

_POOLING_CLASSES = {
    "sentence_transformers.models.Pooling",
    "sentence_transformers.sentence_transformer.modules.Pooling",
    "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
}
_QWEN_NAMES = {"Qwen3-Embedding-0.6B", "Qwen3-Embedding-4B", "Qwen3-Embedding-8B"}


def local_pooling_defaults(path: Path, config) -> dict[str, dict]:
    """Supply only absent pooling configs for identifiable native checkpoints."""
    modules_path = path / "modules.json"
    if not path.is_dir() or not modules_path.is_file():
        return {}
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    defaults = {}
    for module in modules:
        class_ref = module.get("type")
        if class_ref not in _POOLING_CLASSES:
            continue
        config_path = (path / module["path"] / "config.json").resolve()
        if not config_path.is_relative_to(path.resolve()):
            raise ValueError(f"pooling module path escapes model directory: {module['path']}")
        if config_path.exists():
            continue
        # A class-wide default cannot safely repair two different pooling modules.
        if sum(item.get("type") in _POOLING_CLASSES for item in modules) != 1:
            raise ValueError(f"missing {config_path}; restore the original model metadata")
        if config.model_type == "embedding_gemma2":
            # The transformer exposes projected token embeddings, not hidden_size.
            dimension = getattr(getattr(config, "text_config", None), "embedding_dim", None)
            mode = "mean"
        elif config.model_type == "qwen3" and any(
            part in _QWEN_NAMES or part in {f"models--Qwen--{name}" for name in _QWEN_NAMES}
            for part in path.parts
        ):
            dimension = getattr(config, "hidden_size", None)
            mode = "lasttoken"
        else:
            raise ValueError(
                f"missing {config_path}; cannot infer this model's pooling safely. "
                "Download the complete original checkpoint, including modules.json and its subdirectories."
            )
        if type(dimension) is not int or dimension < 1:
            raise ValueError(f"missing {config_path}; model config does not declare the pooling dimension")
        defaults[class_ref] = {"embedding_dimension": dimension, "pooling_mode": mode, "include_prompt": True}
        logger.warning("Missing %s; using native %s pooling with dimension %d in memory", config_path, mode, dimension)
    return defaults


def pooling_loader(base, defaults):
    """Use Sentence Transformers' module defaults; saved metadata has priority."""
    if not defaults:
        return base

    class LocalPoolingSentenceTransformer(base):
        def _get_module_init_defaults(self, class_ref):
            return {**super()._get_module_init_defaults(class_ref), **defaults.get(class_ref, {})}

    return LocalPoolingSentenceTransformer
