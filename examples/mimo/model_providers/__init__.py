# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO model-provider descriptors consumed by the generic entry and builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

if TYPE_CHECKING:
    import argparse


@dataclass(frozen=True)
class MimoProvider:
    """Model-specific wiring the generic MIMO entry and builder consume.

    encoder_module_names: modality-encoder module names this provider defines.
    language_spec / encoder_specs: ``(args, pg_collection, grid) -> ModuleSpec`` factories.
    special_token_ids: ``(args) -> {module_name: token_id}``.
    build_communicator: ``(args, topology) -> MultiModulePipelineCommunicator``.
    """

    encoder_module_names: Sequence[str]
    language_spec: Callable
    encoder_specs: Mapping[str, Callable]
    special_token_ids: Callable
    build_communicator: Callable


def resolve_provider(args: "argparse.Namespace") -> MimoProvider:
    """Return the :class:`MimoProvider` selected by ``--model-provider``."""
    # Imported lazily: nemotron_moe_vlm imports MimoProvider from this package.
    from examples.mimo.model_providers.nemotron_moe_vlm import (
        NEMOTRON_MODEL_PROVIDER,
        nemotron_provider,
    )

    providers = {NEMOTRON_MODEL_PROVIDER: nemotron_provider}
    name = getattr(args, "model_provider", NEMOTRON_MODEL_PROVIDER)
    if name not in providers:
        raise ValueError(f"unknown --model-provider {name!r}; known: {sorted(providers)}")
    return providers[name]()
