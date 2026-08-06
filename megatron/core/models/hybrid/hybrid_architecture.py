# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""First-class architecture descriptions for :class:`HybridModel`.

This module deliberately keeps the public surface small.  A layer occurrence is
an existing :class:`~megatron.core.transformer.spec_utils.ModuleSpec` paired with
the :class:`~megatron.core.transformer.transformer_config.TransformerConfig`
that should be passed to that layer.  Nested Python lists provide composition,
and :class:`PipelineSplit` provides explicit PP/VPP chunk boundaries.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence, TypeAlias

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    Symbols,
    parse_hybrid_pattern,
    validate_segment_layers,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

HYBRID_LAYER_TYPE = "hybrid_layer_type"
VALID_HYBRID_LAYER_TYPES = frozenset({"mamba", "gdn", "attention", "dsa", "mla", "mlp", "moe"})


@dataclass(frozen=True, slots=True)
class PipelineSplit:
    """A boundary between two physical or virtual pipeline model chunks."""


@dataclass(frozen=True, slots=True)
class HybridLayerSpec:
    """An existing layer spec paired with its per-occurrence configuration.

    Args:
        module_spec: Specification for an existing Megatron Core layer class.
        config: Complete transformer configuration to pass to this occurrence.
    """

    module_spec: ModuleSpec
    config: TransformerConfig

    def __post_init__(self) -> None:
        if not isinstance(self.module_spec, ModuleSpec):
            raise TypeError(
                f"module_spec must be a ModuleSpec, got {type(self.module_spec).__name__}."
            )
        if not isinstance(self.config, TransformerConfig):
            raise TypeError(
                f"config must be a TransformerConfig, got {type(self.config).__name__}."
            )
        # Resolve eagerly so malformed public descriptors fail at recipe construction.
        _get_layer_type(self.module_spec)

    @property
    def layer_type(self) -> str:
        """Return the stable semantic type carried by ``ModuleSpec.metainfo``."""

        return _get_layer_type(self.module_spec)


HybridLayerPattern: TypeAlias = Sequence["HybridLayerSpec | PipelineSplit | HybridLayerPattern"]


@dataclass(frozen=True, slots=True)
class ResolvedHybridArchitecture:
    """A validated global hybrid architecture shared by every PP/VPP chunk."""

    segments: tuple[tuple[HybridLayerSpec, ...], ...]
    mtp_layers: tuple[HybridLayerSpec, ...] = ()
    mtp_num_layers: int = 0
    source: str = "direct"

    @property
    def main_layers(self) -> tuple[HybridLayerSpec, ...]:
        """Return decoder occurrences in global model order."""

        return tuple(layer for segment in self.segments for layer in segment)

    @property
    def metric_num_layers(self) -> int:
        """Return the number of explicit layer slots used by MoE metrics."""

        return len(self.main_layers) + self.mtp_num_layers * len(self.mtp_layers)

    def select_segment(
        self, *, pp_rank: int, pp_size: int, vp_stage: int | None
    ) -> tuple[tuple[HybridLayerSpec, ...], int]:
        """Select a local chunk and its global decoder-layer offset.

        Segments use the same VPP-major ordering as the legacy ``|`` syntax:
        ``segment_index = vp_stage * pp_size + pp_rank``.
        """

        vp_rank = 0 if vp_stage is None else vp_stage
        segment_index = vp_rank * pp_size + pp_rank
        if segment_index >= len(self.segments):
            raise ValueError(
                f"Pipeline segment index {segment_index} is out of range for "
                f"{len(self.segments)} resolved segments (pp_rank={pp_rank}, "
                f"pp_size={pp_size}, vp_stage={vp_stage})."
            )
        offset = sum(len(segment) for segment in self.segments[:segment_index])
        return self.segments[segment_index], offset

    @property
    def has_heterogeneous_inference_shapes(self) -> bool:
        """Whether dynamic inference would need non-uniform cache/buffer shapes."""

        layers = self.main_layers + self.mtp_layers
        signatures: dict[str, set[tuple[Any, ...]]] = {
            "mamba": set(),
            "attention": set(),
            "moe": set(),
        }
        for layer in layers:
            config = layer.config
            if layer.layer_type == "mamba":
                signatures["mamba"].add(
                    (
                        config.mamba_state_dim,
                        config.mamba_head_dim,
                        config.mamba_num_heads,
                        config.mamba_num_groups,
                    )
                )
            elif layer.layer_type == "attention":
                signatures["attention"].add(
                    (config.num_query_groups or config.num_attention_heads, config.kv_channels)
                )
            elif layer.layer_type == "moe":
                signatures["moe"].add((config.moe_router_topk,))
        return any(len(values) > 1 for values in signatures.values())

    def has_incompatible_dynamic_inference_shapes(self, base_config: TransformerConfig) -> bool:
        """Whether dynamic inference's model-global buffers fit every occurrence.

        Dynamic inference currently allocates attention KV and MoE routing buffers
        from the model-wide config.  Even a family-uniform direct override is
        incompatible when it differs from that allocation source.  Mamba state
        shapes are discovered from the built layer, so only differing Mamba
        occurrences are rejected.
        """

        if self.has_heterogeneous_inference_shapes:
            return True

        base_attention_signature = (
            base_config.num_query_groups or base_config.num_attention_heads,
            base_config.kv_channels,
        )
        layers = self.main_layers + self.mtp_layers
        for layer in layers:
            if layer.layer_type == "attention":
                occurrence_signature = (
                    layer.config.num_query_groups or layer.config.num_attention_heads,
                    layer.config.kv_channels,
                )
                if occurrence_signature != base_attention_signature:
                    return True
            elif (
                layer.layer_type == "moe"
                and layer.config.moe_router_topk != base_config.moe_router_topk
            ):
                return True
        return False


def flatten_hybrid_layer_pattern(
    pattern: HybridLayerPattern, *, allow_splits: bool = True
) -> tuple[tuple[HybridLayerSpec, ...], ...]:
    """Recursively flatten a direct Python pattern while preserving split nodes.

    Args:
        pattern: Nested lists or tuples containing layer descriptors and split nodes.
        allow_splits: If false, encountering :class:`PipelineSplit` is an error.

    Returns:
        A tuple of flat pipeline segments. Empty segments are intentionally retained.
    """

    segments: list[list[HybridLayerSpec]] = [[]]

    def visit(node: Any, path: tuple[int, ...]) -> None:
        if isinstance(node, HybridLayerSpec):
            segments[-1].append(node)
            return
        if isinstance(node, PipelineSplit):
            if not allow_splits:
                raise ValueError(
                    f"PipelineSplit at path {list(path)} is not allowed in an MTP pattern."
                )
            segments.append([])
            return
        if isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                visit(child, path + (index,))
            return
        raise TypeError(
            f"Hybrid layer pattern leaf at path {list(path)} has unsupported type "
            f"{type(node).__name__}; expected HybridLayerSpec, PipelineSplit, list, or tuple."
        )

    visit(pattern, ())
    return tuple(tuple(segment) for segment in segments)


def resolve_hybrid_architecture(
    *,
    config: TransformerConfig,
    hybrid_stack_spec: ModuleSpec,
    layer_specs: HybridLayerPattern | None = None,
    mtp_layer_specs: HybridLayerPattern | None = None,
    hybrid_layer_pattern: str | None = None,
) -> ResolvedHybridArchitecture:
    """Resolve direct descriptors or a legacy string into one global architecture.

    Direct split nodes are authoritative for PP/VPP chunking.  Legacy strings retain
    their existing pipe-free even/uneven PP compatibility behavior.
    """

    if layer_specs is not None and hybrid_layer_pattern is not None:
        raise ValueError("layer_specs and hybrid_layer_pattern are mutually exclusive.")
    if layer_specs is None and hybrid_layer_pattern is None:
        raise ValueError("Exactly one of layer_specs or hybrid_layer_pattern must be provided.")
    if mtp_layer_specs is not None and layer_specs is None:
        raise ValueError("mtp_layer_specs requires direct layer_specs.")

    if layer_specs is not None:
        if config.mtp_standalone:
            raise ValueError("Direct hybrid architectures do not support standalone MTP placement.")
        raw_segments = flatten_hybrid_layer_pattern(layer_specs)
        _validate_direct_pipeline(config, raw_segments)
        raw_mtp_segments = (
            flatten_hybrid_layer_pattern(mtp_layer_specs, allow_splits=False)
            if mtp_layer_specs is not None
            else ((),)
        )
        raw_mtp_layers = raw_mtp_segments[0]
        mtp_num_layers = config.mtp_num_layers or 0
        if raw_mtp_layers and mtp_num_layers <= 0:
            raise ValueError("mtp_layer_specs requires config.mtp_num_layers > 0.")
        if mtp_num_layers > 0 and not raw_mtp_layers:
            raise ValueError("config.mtp_num_layers > 0 requires mtp_layer_specs.")

        segments = tuple(
            tuple(_materialize_layer(layer, config) for layer in segment)
            for segment in raw_segments
        )
        mtp_layers = tuple(_materialize_layer(layer, config) for layer in raw_mtp_layers)
        architecture = ResolvedHybridArchitecture(
            segments=segments, mtp_layers=mtp_layers, mtp_num_layers=mtp_num_layers, source="direct"
        )
        _validate_mtp_placement(architecture)
        _validate_uniform_expert_count(architecture, config)
        return architecture

    return _resolve_legacy_architecture(config, hybrid_stack_spec, hybrid_layer_pattern)


def _validate_direct_pipeline(
    config: TransformerConfig, segments: tuple[tuple[HybridLayerSpec, ...], ...]
) -> None:
    pp_size = config.pipeline_model_parallel_size
    num_segments = len(segments)
    num_layers = sum(len(segment) for segment in segments)

    if num_layers != config.num_layers:
        raise ValueError(
            f"Direct layer_specs contains {num_layers} decoder layers, but "
            f"TransformerConfig.num_layers is {config.num_layers}."
        )
    if pp_size <= 0:
        raise ValueError(f"pipeline_model_parallel_size must be positive, got {pp_size}.")
    if pp_size > 1 and num_segments == 1:
        raise ValueError(
            "Direct layer_specs with pipeline_model_parallel_size > 1 must contain "
            "explicit PipelineSplit() boundaries."
        )
    if num_segments % pp_size != 0:
        raise ValueError(
            f"Direct layer_specs defines {num_segments} pipeline segments, which is not "
            f"divisible by pipeline_model_parallel_size={pp_size}."
        )

    inferred_vp_size = num_segments // pp_size
    if pp_size == 1 and inferred_vp_size > 1:
        raise ValueError("Virtual pipeline parallelism requires pipeline_model_parallel_size > 1.")
    configured_vp_size = config.virtual_pipeline_model_parallel_size
    if configured_vp_size is not None and configured_vp_size != inferred_vp_size:
        raise ValueError(
            f"PipelineSplit() nodes imply virtual_pipeline_model_parallel_size="
            f"{inferred_vp_size}, but the config specifies {configured_vp_size}."
        )

    if num_segments > 1:
        conflicts = {
            "pipeline_model_parallel_layout": config.pipeline_model_parallel_layout,
            "num_layers_in_first_pipeline_stage": config.num_layers_in_first_pipeline_stage,
            "num_layers_in_last_pipeline_stage": config.num_layers_in_last_pipeline_stage,
            "account_for_embedding_in_pipeline_split": (
                config.account_for_embedding_in_pipeline_split
            ),
            "account_for_loss_in_pipeline_split": config.account_for_loss_in_pipeline_split,
        }
        active_conflicts = [name for name, value in conflicts.items() if value not in (None, False)]
        if active_conflicts:
            raise ValueError(
                "PipelineSplit() already defines pipeline ownership and cannot be combined with: "
                + ", ".join(active_conflicts)
            )

    config.virtual_pipeline_model_parallel_size = inferred_vp_size if inferred_vp_size > 1 else None


def _resolve_legacy_architecture(
    config: TransformerConfig, hybrid_stack_spec: ModuleSpec, hybrid_layer_pattern: str | None
) -> ResolvedHybridArchitecture:
    parsed = parse_hybrid_pattern(hybrid_layer_pattern)
    submodules = hybrid_stack_spec.submodules
    symbol_fields = {
        Symbols.MAMBA: ("mamba", "mamba_layer"),
        Symbols.GDN: ("gdn", "gdn_layer"),
        Symbols.ATTENTION: ("attention", "attention_layer"),
        Symbols.DS_ATTENTION: ("dsa", "dsa_layer"),
        Symbols.MLA: ("mla", "mla_layer"),
        Symbols.MLP: ("mlp", "mlp_layer"),
        Symbols.MOE: ("moe", "moe_layer"),
    }

    def descriptor(symbol: str) -> HybridLayerSpec:
        layer_type, field_name = symbol_fields[symbol]
        return HybridLayerSpec(
            module_spec=_legacy_module_spec(getattr(submodules, field_name), layer_type),
            config=config,
        )

    main_pattern = parsed.main_pattern or ""
    # This is a read-only summary of the legacy pattern, not a replacement for
    # select_pipeline_segment().  In particular, do not infer or mutate PP/VPP
    # settings, or add validation that existing HybridModel callers did not see.
    pattern_segments = main_pattern.split(Symbols.PIPE)

    segments = tuple(
        tuple(descriptor(symbol) for symbol in validate_segment_layers(segment))
        for segment in pattern_segments
    )

    mtp_layers: tuple[HybridLayerSpec, ...] = ()
    if parsed.mtp_pattern:
        mtp_layers = tuple(
            descriptor(symbol) for symbol in validate_segment_layers(parsed.mtp_pattern)
        )
    return ResolvedHybridArchitecture(
        segments=segments,
        mtp_layers=mtp_layers,
        mtp_num_layers=parsed.mtp_num_depths,
        source="legacy",
    )


_UNIFORM_CONFIG_FIELDS = (
    "hidden_size",
    "params_dtype",
    "pipeline_dtype",
    "fp16",
    "bf16",
    "fp32_residual_connection",
    "enable_autocast",
    "autocast_dtype",
    "apply_query_key_layer_scaling",
    "attention_softmax_in_fp32",
    "disable_bf16_reduced_precision_matmul",
    "dsa_indexer_k_norm_fp32",
    "fp8",
    "fp8_recipe",
    "fp8_param",
    "fp8_quantizer_factory",
    "fp8_margin",
    "fp8_interval",
    "fp8_amax_history_len",
    "fp8_amax_compute_algo",
    "fp8_wgrad",
    "fp8_output_proj",
    "fp8_dot_product_attention",
    "fp8_multi_head_attention",
    "tp_only_amax_red",
    "activation_func_fp8_input_store",
    "first_last_layers_bf16",
    "num_layers_at_start_in_bf16",
    "num_layers_at_end_in_bf16",
    "moe_router_dtype",
    "mamba_training_ssm_states_dtype",
    "fp4",
    "fp4_recipe",
    "fp4_param",
    "fp4_quantizer_factory",
    "quant_recipe",
    "normalization",
    "layernorm_epsilon",
    "layernorm_zero_centered_gamma",
    "tensor_model_parallel_size",
    "tensor_parallel_num_weight_shards",
    "gtp_weight_remat_size",
    "pipeline_model_parallel_size",
    "pipeline_model_parallel_comm_backend",
    "context_parallel_size",
    "hierarchical_context_parallel_sizes",
    "max_seqlen_per_dp_cp_rank",
    "hybrid_context_parallel",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "expert_tensor_parallel_num_weight_shards",
    "expert_gtp_weight_remat_size",
    "sequence_parallel",
)

_TOPOLOGY_CONFIG_FIELDS = (
    "num_layers",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "pipeline_model_parallel_layout",
    "num_layers_in_first_pipeline_stage",
    "num_layers_in_last_pipeline_stage",
    "account_for_embedding_in_pipeline_split",
    "account_for_loss_in_pipeline_split",
    "mtp_num_layers",
    "mtp_use_repeated_layer",
    "mtp_standalone",
)


def _materialize_layer(layer: HybridLayerSpec, base_config: TransformerConfig) -> HybridLayerSpec:
    for field_name in _UNIFORM_CONFIG_FIELDS:
        layer_value = getattr(layer.config, field_name)
        base_value = getattr(base_config, field_name)
        if layer_value != base_value:
            raise ValueError(
                f"Per-layer config for {layer.layer_type!r} changes model-wide field "
                f"{field_name!r}: {layer_value!r} != {base_value!r}."
            )
    return HybridLayerSpec(
        module_spec=layer.module_spec, config=_materialize_config(layer.config, base_config)
    )


def _materialize_config(
    layer_config: TransformerConfig, base_config: TransformerConfig
) -> TransformerConfig:
    materialized = copy.deepcopy(layer_config)
    for field_name in _TOPOLOGY_CONFIG_FIELDS:
        setattr(materialized, field_name, getattr(base_config, field_name))
    return materialized


def _validate_mtp_placement(architecture: ResolvedHybridArchitecture) -> None:
    """Reject a prediction block on a logical chunk with no decoder layers."""

    if (
        architecture.mtp_layers
        and architecture.mtp_num_layers > 0
        and not architecture.segments[-1]
    ):
        raise ValueError(
            "MTP must share the final logical PP/VPP chunk with at least one decoder layer; "
            "standalone MTP placement is not supported."
        )


def _validate_uniform_expert_count(
    architecture: ResolvedHybridArchitecture, base_config: TransformerConfig
) -> None:
    expert_counts = {
        layer.config.num_moe_experts
        for layer in architecture.main_layers + architecture.mtp_layers
        if layer.layer_type == "moe"
    }
    if len(expert_counts) > 1:
        raise ValueError(
            "All MoE occurrences must use one uniform num_moe_experts value; got "
            f"{sorted(expert_counts)}."
        )
    if expert_counts and next(iter(expert_counts)) != base_config.num_moe_experts:
        occurrence_count = next(iter(expert_counts))
        raise ValueError(
            "MoE occurrence num_moe_experts must match the model-wide config; "
            f"got {occurrence_count!r} != {base_config.num_moe_experts!r}."
        )


def _get_layer_type(module_spec: ModuleSpec) -> str:
    layer_type = module_spec.metainfo.get(HYBRID_LAYER_TYPE)
    if layer_type not in VALID_HYBRID_LAYER_TYPES:
        raise ValueError(
            f"ModuleSpec.metainfo[{HYBRID_LAYER_TYPE!r}] must be one of "
            f"{sorted(VALID_HYBRID_LAYER_TYPES)}, got {layer_type!r}."
        )
    return layer_type


def _with_layer_type(module_spec: ModuleSpec, layer_type: str) -> ModuleSpec:
    if not isinstance(module_spec, ModuleSpec):
        raise TypeError(
            f"Hybrid stack entry for {layer_type!r} must be a ModuleSpec, got "
            f"{type(module_spec).__name__}."
        )
    if module_spec.metainfo.get(HYBRID_LAYER_TYPE) == layer_type:
        return module_spec
    tagged = copy.copy(module_spec)
    tagged.metainfo = dict(module_spec.metainfo)
    tagged.metainfo[HYBRID_LAYER_TYPE] = layer_type
    return tagged


def _legacy_module_spec(module_spec: ModuleSpec | type, layer_type: str) -> ModuleSpec:
    """Tag a legacy stack entry without narrowing its historical accepted types."""

    if isinstance(module_spec, ModuleSpec):
        return _with_layer_type(module_spec, layer_type)
    return ModuleSpec(module=module_spec, metainfo={HYBRID_LAYER_TYPE: layer_type})


__all__ = [
    "HYBRID_LAYER_TYPE",
    "HybridLayerPattern",
    "HybridLayerSpec",
    "PipelineSplit",
    "ResolvedHybridArchitecture",
    "flatten_hybrid_layer_pattern",
    "resolve_hybrid_architecture",
]
