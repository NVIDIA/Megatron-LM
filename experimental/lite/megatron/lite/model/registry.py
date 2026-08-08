"""Model and protocol registry."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry tables (populated by register_model)
# ---------------------------------------------------------------------------

# model_name → model package module path
MODEL_PACKAGES: dict[str, str] = {}

# HF model_type string -> Megatron Lite model_name
_HF_MODEL_TYPE_MAP: dict[str, str] = {}

# (model_name, impl) → runtime_model_name
_IMPL_TO_RUNTIME_MODEL: dict[tuple[str, str], str] = {}

# runtime_model_name → protocol module path
TRAIN_RUNTIME_MODULES: dict[str, str] = {}

# ---------------------------------------------------------------------------
# Registration API
# ---------------------------------------------------------------------------


def register_model(
    model_name: str,
    *,
    package: str,
    hf_model_types: list[str] | None = None,
    impls: dict[str, str] | None = None,
) -> None:
    """Register a model and all its implementations in one call.

    Args:
        model_name: Megatron Lite model name (e.g. ``"qwen3"``).
        package: Model package module path (e.g. ``"megatron.lite.model.qwen3_moe"``).
        hf_model_types: HF ``model_type`` strings that map to this model.
        impls: ``{impl_name: protocol_module_path}``.
            The first impl is also registered as the bare ``model_name`` runtime key.

    Example::

        register_model(
            "qwen3",
            package="megatron.lite.model.qwen3_moe",
            hf_model_types=["qwen3_moe", "qwen2_moe"],
            impls={
                "lite": "megatron.lite.model.qwen3_moe.lite.protocol",
            },
        )
    """
    MODEL_PACKAGES[model_name] = package

    if hf_model_types:
        for hf_type in hf_model_types:
            _HF_MODEL_TYPE_MAP[hf_type] = model_name

    if impls:
        for i, (impl_name, proto_module) in enumerate(impls.items()):
            runtime_key = model_name if i == 0 else f"{model_name}_{impl_name}"
            _IMPL_TO_RUNTIME_MODEL[(model_name, impl_name)] = runtime_key
            TRAIN_RUNTIME_MODULES[runtime_key] = proto_module


# ---------------------------------------------------------------------------
# Built-in models
# ---------------------------------------------------------------------------

register_model(
    "qwen3",
    package="megatron.lite.model.qwen3_moe",
    hf_model_types=["qwen3_moe", "qwen2_moe"],
    impls={
        "lite": "megatron.lite.model.qwen3_moe.lite.protocol",
    },
)

register_model(
    "qwen3_moe",
    package="megatron.lite.model.qwen3_moe",
    hf_model_types=None,
    impls={
        "lite": "megatron.lite.model.qwen3_moe.lite.protocol",
    },
)

register_model(
    "qwen3_5",
    package="megatron.lite.model.qwen3_5",
    hf_model_types=["qwen3_5_moe"],
    impls={
        "lite": "megatron.lite.model.qwen3_5.lite.protocol",
    },
)


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_model_package(model_name: str):
    if model_name not in MODEL_PACKAGES:
        raise ValueError(
            f"Unknown model: {model_name!r}. Available: {list(MODEL_PACKAGES)}"
        )
    return importlib.import_module(MODEL_PACKAGES[model_name])


def get_train_runtime_module(model_name: str):
    if model_name in TRAIN_RUNTIME_MODULES:
        return importlib.import_module(TRAIN_RUNTIME_MODULES[model_name])
    raise ValueError(f"No protocol module for: {model_name!r}")


def resolve_runtime_model_name(model_name: str, impl: str) -> str:
    key = (model_name, impl)
    if key not in _IMPL_TO_RUNTIME_MODEL:
        raise ValueError(
            f"No runtime for ({model_name!r}, {impl!r}). "
            f"Known: {list(_IMPL_TO_RUNTIME_MODEL)}"
        )
    return _IMPL_TO_RUNTIME_MODEL[key]


def resolve_model_type_from_hf(source: str | Path | dict) -> str:
    """Resolve Megatron Lite model_name from an HF source.

    Args:
        source: One of:
            - Directory path (str/Path) containing ``config.json``
            - Path to a ``config.json`` file directly
            - A dict / HF config object with a ``model_type`` key
    """
    if isinstance(source, dict):
        hf_config = source
    elif hasattr(source, "model_type"):
        # HF PretrainedConfig object
        hf_config = {"model_type": source.model_type}
    else:
        p = Path(source)
        if p.is_file():
            config_path = p
        elif p.is_dir():
            config_path = p / "config.json"
        else:
            raise FileNotFoundError(f"Not a file or directory: {source}")
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found at {source}")
        with open(config_path) as f:
            hf_config = json.load(f)

    hf_model_type = hf_config.get("model_type", "")
    model_name = _HF_MODEL_TYPE_MAP.get(hf_model_type)
    if model_name is not None:
        return model_name
    raise ValueError(
        f"Cannot resolve model_type={hf_model_type!r}. "
        f"Known: {list(_HF_MODEL_TYPE_MAP)}. Set model_name explicitly."
    )


__all__ = [
    "get_model_package",
    "get_train_runtime_module",
    "register_model",
    "resolve_model_type_from_hf",
    "resolve_runtime_model_name",
]
