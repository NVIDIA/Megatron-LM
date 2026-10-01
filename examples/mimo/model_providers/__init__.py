# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO model-provider descriptors consumed by the generic entry and builder."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import argparse


@dataclass(frozen=True)
class MimoProvider:
    """Model-specific wiring the generic MIMO entry and builder consume.

    encoder_module_names: modality-encoder module names this provider defines.
    language_spec / encoder_specs: ``(args, pg_collection, grid) -> ModuleSpec`` factories.
    language_input_projection_specs: optional projector factories for the first language stage.
    special_token_ids: ``(args) -> {module_name: token_id}``.
    build_communicator: ``(args, topology) -> MultiModulePipelineCommunicator``.

    Providers are built by (args) -> MimoProvider factories registered in MODEL_PROVIDERS.
    """

    encoder_module_names: Sequence[str]
    language_spec: Callable
    encoder_specs: Mapping[str, Callable]
    special_token_ids: Callable
    build_communicator: Callable
    language_input_projection_specs: Mapping[str, Callable] = field(default_factory=dict)


MODEL_PROVIDERS: Mapping[str, str] = {
    "nemotron-moe-vlm": "examples.mimo.model_providers.nemotron_moe_vlm:nemotron_provider",
    "rope2d-vit-vlm": "examples.mimo.model_providers.rope2d_vit_vlm:rope2d_vit_vlm_provider",
}
DEFAULT_MODEL_PROVIDER = "nemotron-moe-vlm"


def resolve_provider(args: "argparse.Namespace") -> MimoProvider:
    """Return the :class:`MimoProvider` selected by ``--model-provider``."""
    name = getattr(args, "model_provider", DEFAULT_MODEL_PROVIDER)
    if name not in MODEL_PROVIDERS:
        raise ValueError(f"unknown --model-provider {name!r}; known: {sorted(MODEL_PROVIDERS)}")
    module_name, factory_name = MODEL_PROVIDERS[name].split(":")
    return getattr(importlib.import_module(module_name), factory_name)(args)
