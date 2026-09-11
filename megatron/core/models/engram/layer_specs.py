# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compose Engram into existing GPT layer specifications."""

from __future__ import annotations

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import BaseTransformerLayer

from .config import EngramConfig
from .engram import Engram


def _attach_to_layer_spec(layer_spec: ModuleSpec, engram_spec: ModuleSpec) -> None:
    if not issubclass(layer_spec.module, BaseTransformerLayer):
        raise TypeError(f"Engram expected a TransformerLayer ModuleSpec, got {layer_spec.module}.")
    if layer_spec.submodules is None or not hasattr(layer_spec.submodules, "engram"):
        raise TypeError("Transformer layer spec does not expose the Engram composition point.")
    layer_spec.submodules.engram = engram_spec


def apply_engram_to_layer_spec(spec, engram_config: EngramConfig):
    """Attach one Engram ModuleSpec without copying a GPT layer-spec builder."""
    engram_spec = ModuleSpec(module=Engram, params={"engram_config": engram_config})
    if isinstance(spec, TransformerBlockSubmodules):
        for layer_spec in spec.layer_specs:
            _attach_to_layer_spec(layer_spec, engram_spec)
        return spec
    if isinstance(spec, ModuleSpec) and issubclass(spec.module, TransformerBlock):
        if not isinstance(spec.submodules, TransformerBlockSubmodules):
            raise TypeError("TransformerBlock ModuleSpec must provide TransformerBlockSubmodules.")
        if spec.submodules.layer_specs is None:
            raise TypeError("TransformerBlockSubmodules must provide layer_specs.")
        for layer_spec in spec.submodules.layer_specs:
            _attach_to_layer_spec(layer_spec, engram_spec)
        return spec
    if isinstance(spec, ModuleSpec):
        _attach_to_layer_spec(spec, engram_spec)
        return spec
    raise TypeError(f"Unsupported GPT transformer spec for Engram: {type(spec).__name__}.")
