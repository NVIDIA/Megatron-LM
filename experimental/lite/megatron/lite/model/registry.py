# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Model and implementation registry."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelRegistration:
    """Registry entry for one public model family."""

    package: str
    hf_model_types: list[str] = field(default_factory=list)
    impls: dict[str, str] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelRegistration] = {}
HF_MODEL_TYPE_MAP: dict[str, str] = {}
TRAIN_RUNTIME_MODULES: dict[str, str] = {}
IMPL_TO_RUNTIME_MODEL: dict[tuple[str, str], str] = {}


def register_model(
    model_name: str,
    *,
    package: str,
    hf_model_types: list[str] | None = None,
    impls: dict[str, str] | None = None,
) -> None:
    """Register a model family and its runtime protocol modules."""

    if not model_name:
        raise ValueError("Model name must be non-empty.")
    if not package:
        raise ValueError("Model package must be non-empty.")

    entry = ModelRegistration(
        package=package,
        hf_model_types=list(hf_model_types or []),
        impls=dict(impls or {}),
    )
    MODEL_REGISTRY[model_name] = entry

    for hf_model_type in entry.hf_model_types:
        HF_MODEL_TYPE_MAP[hf_model_type] = model_name

    for index, (impl_name, module_path) in enumerate(entry.impls.items()):
        runtime_key = model_name if index == 0 else f"{model_name}_{impl_name}"
        IMPL_TO_RUNTIME_MODEL[(model_name, impl_name)] = runtime_key
        TRAIN_RUNTIME_MODULES[runtime_key] = module_path


def list_models() -> list[str]:
    """Return registered model names."""

    return sorted(MODEL_REGISTRY)


def get_model_package(model_name: str):
    """Import the package registered for ``model_name``."""

    try:
        package = MODEL_REGISTRY[model_name].package
    except KeyError as exc:
        raise ValueError(f"Unknown model {model_name!r}. Known: {list_models()}") from exc
    return importlib.import_module(package)


def get_train_runtime_module(runtime_model_name: str):
    """Import the protocol module registered for a runtime model key."""

    try:
        module_path = TRAIN_RUNTIME_MODULES[runtime_model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown runtime model {runtime_model_name!r}. "
            f"Known: {sorted(TRAIN_RUNTIME_MODULES)}"
        ) from exc
    return importlib.import_module(module_path)


def resolve_runtime_model_name(model_name: str, impl: str) -> str:
    """Resolve ``(model_name, impl)`` to the runtime registry key."""

    try:
        return IMPL_TO_RUNTIME_MODEL[(model_name, impl)]
    except KeyError as exc:
        known = sorted(f"{model}:{implementation}" for model, implementation in IMPL_TO_RUNTIME_MODEL)
        raise ValueError(f"No runtime registered for ({model_name!r}, {impl!r}). Known: {known}") from exc


def resolve_model_type_from_hf(source: str | Path | dict[str, Any] | Any) -> str:
    """Resolve a Lite model name from a HuggingFace-style config source."""

    if isinstance(source, dict):
        hf_config = source
    elif hasattr(source, "model_type"):
        hf_config = {"model_type": source.model_type}
    else:
        path = Path(source)
        config_path = path if path.is_file() else path / "config.json"
        with open(config_path, encoding="utf-8") as handle:
            hf_config = json.load(handle)

    hf_model_type = hf_config.get("model_type", "")
    try:
        return HF_MODEL_TYPE_MAP[hf_model_type]
    except KeyError as exc:
        raise ValueError(
            f"Cannot resolve HF model_type {hf_model_type!r}. "
            f"Known: {sorted(HF_MODEL_TYPE_MAP)}"
        ) from exc


register_model(
    "toy_dense",
    package="megatron.lite.model.toy_dense",
    hf_model_types=["mlite_toy_dense"],
    impls={"torch": "megatron.lite.model.toy_dense"},
)


__all__ = [
    "HF_MODEL_TYPE_MAP",
    "IMPL_TO_RUNTIME_MODEL",
    "MODEL_REGISTRY",
    "TRAIN_RUNTIME_MODULES",
    "get_model_package",
    "get_train_runtime_module",
    "list_models",
    "register_model",
    "resolve_model_type_from_hf",
    "resolve_runtime_model_name",
]
