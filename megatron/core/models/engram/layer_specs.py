# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Compose Engram into existing hybrid layer specifications."""

from __future__ import annotations

import copy

from megatron.core.transformer.spec_utils import ModuleSpec
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
    through the pattern to the submodule fields the stack may build that layer from, and a field
    that cannot carry the memory - a Mamba layer, or a layer type this stack spec does not
    provide - is rejected here. Checking the spec rather than the pattern alone is what keeps
    a mis-selected layer from silently building no memory at all.

    Some symbols map to more than one field: HybridStack chooses among them from the layer
    config (``experimental_attention_variant='gdn2'``, ``mla_down_proj_fusion``,
    ``qk_layernorm``), and H and W layers fall back to the C fields when the stack spec has no
    dedicated ``hca_layer`` or ``window_layer``. The memory is attached to every field the
    stack spec provides for the symbol, so it follows whichever one is built; at least one must
    exist.

    MoE ('E') layers are rejected under ``cuda_graph_impl='transformer_engine'``: the MoE
    capture scopes graph MoE layers, and the Engram lookup has data-dependent shapes.

    The spec is copied before mutation: stack specs are module-level singletons that several
    builders share, so attaching in place would carry a stale EngramConfig into unrelated models.
    Layer selection itself stays in TransformerLayer, which builds the module only for the
    configured global layer numbers, so attaching to a field that also builds other layers is
    harmless.
    """
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    if not isinstance(spec, ModuleSpec) or spec.submodules is None:
        raise TypeError(f"Unsupported hybrid stack spec for Engram: {type(spec).__name__}.")

    # Hyper-connection (mHC) and multi-token-prediction stacks need no handling here: the mHC
    # wrapper adds the memory residual to its n-stream tensor itself and its inner layer skips
    # its own injection, and the MTP layers built from these same submodules, numbered
    # locally, never build a memory.

    # HybridStackSubmodules fields that HybridStack may build each layer symbol from.
    csa_fields = ("csa_layer", "csa_qk_layernorm_layer")
    fields_for_symbol = {
        Symbols.MAMBA: ("mamba_layer",),
        Symbols.GDN: ("gdn_layer", "gdn2_layer"),
        Symbols.KDA: ("kda_layer",),
        Symbols.ATTENTION: ("attention_layer",),
        Symbols.DS_ATTENTION: ("dsa_layer",),
        Symbols.MLA: ("mla_layer", "mla_fused_down_proj_layer"),
        Symbols.CSA: csa_fields,
        Symbols.HCA: ("hca_layer",),
        Symbols.WINDOW: ("window_layer",),
        Symbols.MLP: ("mlp_layer",),
        Symbols.MOE: ("moe_layer",),
    }
    # HybridStack builds H and W layers from the C fields when the dedicated field is unset.
    fallback_fields_for_symbol = {Symbols.HCA: csa_fields, Symbols.WINDOW: csa_fields}
    layer_types = hybrid_layer_types(hybrid_layer_pattern)
    spec = copy.deepcopy(spec)

    def _provided_fields(field_names: tuple[str, ...]) -> list[str]:
        return [name for name in field_names if getattr(spec.submodules, name, None) is not None]

    engram_spec = ModuleSpec(module=Engram, params={"engram_config": engram_config})
    for layer_id in engram_config.layer_ids:
        if layer_id > len(layer_types):
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, but the hybrid layer pattern "
                f"describes only {len(layer_types)} layers."
            )
        symbol = layer_types[layer_id - 1]
        if symbol not in fields_for_symbol:
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, whose hybrid symbol "
                f"'{symbol}' has no known HybridStackSubmodules field."
            )
        if (
            symbol == Symbols.MOE
            and getattr(transformer_config, "cuda_graph_impl", "none") == "transformer_engine"
        ):
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, which the hybrid layer pattern "
                f"assigns to a '{symbol}' layer. With cuda_graph_impl='transformer_engine' the "
                "MoE capture scopes graph MoE layers, and the Engram lookup's hashed gather and "
                "all-to-all have data-dependent shapes; choose a non-MoE layer."
            )
        field_names = _provided_fields(fields_for_symbol[symbol]) or _provided_fields(
            fallback_fields_for_symbol.get(symbol, ())
        )
        if not field_names:
            raise ValueError(
                f"engram_layer_ids contains layer {layer_id}, which the hybrid layer pattern "
                f"assigns to a '{symbol}' layer, but this stack spec provides no layer spec for "
                "that layer type."
            )
        for field_name in field_names:
            layer_spec = getattr(spec.submodules, field_name)
            if not _accepts_engram(layer_spec):
                raise ValueError(
                    f"engram_layer_ids contains layer {layer_id}, which the hybrid layer pattern "
                    f"assigns to a '{symbol}' layer. That layer type does not accept Engram in "
                    f"this stack spec ({field_name}); choose a layer of a type that does."
                )
            if layer_spec.submodules.engram is not engram_spec:
                _attach_to_layer_spec(layer_spec, engram_spec)
    return spec
