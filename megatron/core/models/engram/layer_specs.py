# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compose Engram into existing GPT and hybrid layer specifications."""

from __future__ import annotations

import copy

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import BaseTransformerLayer

from .config import EngramConfig
from .engram import Engram


def _accepts_engram(spec) -> bool:
    """Whether ``spec`` describes a transformer layer exposing the Engram composition point."""
    return (
        isinstance(spec, ModuleSpec)
        and isinstance(spec.module, type)
        and issubclass(spec.module, BaseTransformerLayer)
        and spec.submodules is not None
        and hasattr(spec.submodules, "engram")
    )


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


def hybrid_layer_types(hybrid_layer_pattern: str | None) -> list[str]:
    """Return the hybrid pattern's ordered layer-type symbols, one per global layer."""
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols, parse_hybrid_pattern

    main_pattern = parse_hybrid_pattern(hybrid_layer_pattern).main_pattern
    if not main_pattern:
        raise ValueError("Engram requires --hybrid-layer-pattern to place its memory layers.")
    # Pipe symbols are pipeline-stage boundaries, not layers, so they must not shift numbering.
    return [symbol for symbol in main_pattern if symbol != Symbols.PIPE]


def apply_engram_to_hybrid_stack_spec(
    spec: ModuleSpec,
    engram_config: EngramConfig,
    hybrid_layer_pattern: str | None,
    transformer_config,
) -> ModuleSpec:
    """Attach one Engram ModuleSpec to the hybrid layer types the configuration selects.

    A hybrid stack picks a per-layer submodule spec by pattern symbol, so the memory is attached
    per layer type rather than to a single layer spec. Every configured layer ID is resolved
    through the pattern to the submodule field that will actually be built, and a field that
    cannot carry the memory - a Mamba layer, or a layer type this stack spec leaves as
    IdentityOp - is rejected here. Checking the spec rather than the pattern alone is what keeps
    a mis-selected layer from silently building no memory at all.

    The spec is copied before mutation: stack specs are module-level singletons that several
    builders share, so attaching in place would carry a stale EngramConfig into unrelated models.
    Layer selection itself stays in TransformerLayer, which builds the module only for the
    configured global layer numbers.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    if not isinstance(spec, ModuleSpec) or spec.submodules is None:
        raise TypeError(f"Unsupported hybrid stack spec for Engram: {type(spec).__name__}.")

    # Hyper connections are supported: HybridStack wraps every layer in
    # HyperConnectionHybridLayer, whose inner paths bypass TransformerLayer._forward_attention,
    # so the wrapper itself adds the memory residual to the n-stream tensor before its read gate
    # and tells the wrapped layer to skip its own injection (skip_engram).

    # Multi-token prediction is supported: the nested MTP stack is built from these same
    # submodules, and HybridStack passes is_mtp_layer for every layer type, so
    # TransformerLayer never builds a memory at an MTP-local layer number.

    # HybridStackSubmodules field that carries each layer symbol. C/H/W all share
    # csa_layer (the compress ratio comes from the layer config, not the spec).
    field_for_symbol = {
        Symbols.MAMBA: "mamba_layer",
        Symbols.GDN: "gdn_layer",
        Symbols.ATTENTION: "attention_layer",
        Symbols.DS_ATTENTION: "dsa_layer",
        Symbols.MLA: "mla_layer",
        Symbols.CSA: "csa_layer",
        Symbols.HCA: "csa_layer",
        Symbols.WINDOW: "csa_layer",
        Symbols.MLP: "mlp_layer",
        Symbols.MOE: "moe_layer",
    }
    layer_types = hybrid_layer_types(hybrid_layer_pattern)
    spec = copy.deepcopy(spec)

    engram_spec = ModuleSpec(module=Engram, params={"engram_config": engram_config})
    for layer_id in engram_config.layer_ids:
        if layer_id > len(layer_types):
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, but the hybrid layer pattern "
                f"describes only {len(layer_types)} layers."
            )
        symbol = layer_types[layer_id - 1]
        if symbol not in field_for_symbol:
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, whose hybrid symbol "
                f"'{symbol}' has no known HybridStackSubmodules field."
            )
        layer_spec = getattr(spec.submodules, field_for_symbol[symbol], None)
        if not _accepts_engram(layer_spec):
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, which the hybrid layer pattern "
                f"assigns to a '{symbol}' layer. That layer type does not accept Engram in this "
                "stack spec; choose a layer of a type that does."
            )
        if layer_spec.submodules.engram is not engram_spec:
            _attach_to_layer_spec(layer_spec, engram_spec)
    return spec
