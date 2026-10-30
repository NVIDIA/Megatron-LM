# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO model-provider descriptors consumed by the generic entry and builder."""

from __future__ import annotations

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
    train_valid_test_data_provider: optional ``(args, topology) -> iterators`` for packed input.
    prepare_source: optional ``(source, args, topology) -> (samples, slices, inputs)``.
    initialize_model: optional ``(model, args)`` callback before DDP wrapping.
    """

    encoder_module_names: Sequence[str]
    language_spec: Callable
    encoder_specs: Mapping[str, Callable]
    special_token_ids: Callable
    build_communicator: Callable
    language_input_projection_specs: Mapping[str, Callable] = field(default_factory=dict)
    train_valid_test_data_provider: Callable | None = None
    prepare_source: Callable | None = None
    initialize_model: Callable | None = None


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
