# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Module boundaries over the canonical layout plan (no alternate communication path)."""

from copy import copy
from dataclasses import dataclass

import torch

from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group, resolve_tp_cp_group

from .layout import CPLayout, THDCPLayoutPlan, convert_cp_partition_mode


def get_cp_boundary_layout(config) -> CPLayout:
    """Read the global CP layout already resolved during configuration."""
    layout = config.global_cp_layout
    if layout not in ("contiguous", "zigzag"):
        raise ValueError("Resolve global_cp_layout before constructing the model")
    return layout


@dataclass(frozen=True)
class CPLayoutViews:
    """Batch-owned route and leaf metadata. Leaves never refer back to this object."""

    contiguous: PackedSeqParams
    zigzag: PackedSeqParams
    plan: THDCPLayoutPlan

    def get(self, layout: CPLayout) -> PackedSeqParams:
        """Return the precomputed consumer metadata without copying or synchronizing."""
        if layout not in ("contiguous", "zigzag"):
            raise ValueError(f"Expected a concrete CP layout, got {layout!r}")
        return self.contiguous if layout == "contiguous" else self.zigzag


def attach_cp_layout_views(params_by_layout, plan):
    """Attach an acyclic context once at the producer boundary.

    Public views have a context; its leaf views do not. Reuse tensor metadata and the
    existing plan, without rerunning PackedSeqParams.__post_init__ or constructing routes.
    """
    if plan is None:
        return params_by_layout
    leaves = {}
    for mode in ("contiguous", "zigzag"):
        leaf = copy(params_by_layout[mode])
        leaf.cp_layout_views = None
        leaves[mode] = leaf
    views = CPLayoutViews(leaves["contiguous"], leaves["zigzag"], plan)
    result = {}
    for mode, params in params_by_layout.items():
        params = copy(params)
        params.cp_layout_views = views
        result[mode] = params
    return result


@dataclass(frozen=True)
class CpPartitionModeConverter:
    """Call-local reverse edge; safe for interleaved two-stage execution."""

    source_partition_mode: CPLayout
    target_partition_mode: CPLayout
    cp_group: torch.distributed.ProcessGroup
    tp_group: torch.distributed.ProcessGroup | None
    tp_cp_group: torch.distributed.ProcessGroup | None
    sequence_parallel: bool
    thd_plan: THDCPLayoutPlan | None

    def convert(self, value, *, seq_dim=0):
        """Restore a sequence tensor using the same plan as the entry edge."""
        if value is None:
            return None
        return convert_cp_partition_mode(
            value,
            source_partition_mode=self.source_partition_mode,
            target_partition_mode=self.target_partition_mode,
            seq_dim=seq_dim,
            cp_group=self.cp_group,
            tp_group=self.tp_group,
            tp_cp_group=self.tp_cp_group,
            sequence_parallel=self.sequence_parallel,
            thd_plan=self.thd_plan,
        )

    def restore_output(self, output_with_bias):
        """Restore activations; the token-independent output bias is not redistributed."""
        output, bias = output_with_bias
        return self.convert(output), bias


@dataclass(frozen=True)
class CPLayoutIntermediate:
    """Pair one split-call result with its own reverse edge, never module-global state."""

    value: object
    converter: CpPartitionModeConverter


def prepare_module_cp_input(
    module,
    hidden_states,
    packed_seq_params,
    *,
    target_layout,
    attention_mask=None,
    attention_bias=None,
    inference_context=None,
):
    """Prepare a self-attention/mixer input and its matching metadata.

    Called after the residual branch has been read and before its compute module.
    The reverse edge must run before BDA, residual writes, and MLP/MoE execution.
    Bridge callers can use this same boundary with producer-prepared metadata.
    """
    config = module.config
    # MLP-only TransformerLayer specs have no compute consumer at this boundary.
    from megatron.core.transformer.identity_op import IdentityOp

    if isinstance(getattr(module, "self_attention", None), IdentityOp):
        return hidden_states, packed_seq_params, None
    source = getattr(packed_seq_params, "cp_partition_mode", None)
    if source is None:
        source = get_cp_boundary_layout(config)
    if source == target_layout:
        return hidden_states, packed_seq_params, None
    groups = module.pg_collection
    cp_group = resolve_cp_group(groups.cp, packed_seq_params)
    if cp_group is None or cp_group.size() == 1:
        return hidden_states, packed_seq_params, None
    cross_attention = getattr(module, "cross_attention", None)
    if cross_attention is not None and not isinstance(cross_attention, IdentityOp):
        raise ValueError("CP module layout conversion with cross-attention is not supported")
    if inference_context is not None:
        raise ValueError("CP module layout conversion is a training-only boundary")
    if attention_mask is not None or attention_bias is not None:
        raise ValueError(
            "CP module layout conversion requires implicit attention masks and no bias"
        )

    plan = None
    target_params = packed_seq_params
    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        views = packed_seq_params.cp_layout_views
        if views is None:
            raise ValueError(
                "THD module conversion requires producer-prebuilt CPLayoutViews; request both "
                "layouts through get_batches_on_this_cp_rank before model forward"
            )
        plan, target_params = views.plan, views.get(target_layout)
    graph_impl = getattr(config, "cuda_graph_impl", "none")
    scopes = getattr(config, "cuda_graph_modules", ())
    scopes = (scopes,) if isinstance(scopes, str) else scopes
    captures_attention = not scopes or any(
        getattr(scope, "name", scope) in ("attn", "mamba", "full", "full_iteration")
        for scope in scopes
    )
    if (
        (plan is not None and graph_impl != "none")
        or graph_impl == "full_iteration"
        or (graph_impl != "none" and captures_attention)
    ):
        raise ValueError(
            "CP module conversion with THD views or inside attention/full CUDA graphs is not "
            "supported; use eager execution (SBHD also permits MLP-only graphs)"
        )
    tp_cp_group = resolve_tp_cp_group(
        getattr(groups, "tp_cp", None),
        packed_seq_params,
        sequence_parallel=config.sequence_parallel,
        tp_group=groups.tp,
    )
    converted = convert_cp_partition_mode(
        hidden_states,
        source_partition_mode=source,
        target_partition_mode=target_layout,
        cp_group=cp_group,
        tp_group=groups.tp,
        tp_cp_group=tp_cp_group,
        sequence_parallel=config.sequence_parallel,
        thd_plan=plan,
    )
    reverse = CpPartitionModeConverter(
        target_layout, source, cp_group, groups.tp, tp_cp_group, config.sequence_parallel, plan
    )
    return converted, target_params, reverse


def attention_cp_layout(module):
    """Choose the compute consumer's layout, separately from the residual policy."""
    consumer_layout = getattr(getattr(module, "self_attention", None), "cp_input_layout", None)
    if consumer_layout is not None:
        return consumer_layout
    config = module.config
    if getattr(config, "experimental_attention_variant", None) in (
        "gdn",
        "gdn2",
        "gated_delta_net",
    ):
        return config.linear_attention_cp_layout
    return getattr(config, "attention_cp_layout", "zigzag")


def finalize_packed_seq_params(
    packed_seq_params,
    cp_group=None,
    *,
    tp_group=None,
    tp_cp_group=None,
    sequence_parallel=False,
    needs_layout_conversion=True,
):
    """Thin producer adapter for Bridge callers that already have packed metadata.

    Call once during batch preparation and use the returned metadata. This never mutates
    the input or installs a lazy per-layer builder. New raw-batch producers should use
    ContextParallelBatch.prepare_module_layouts instead.
    """
    if (
        packed_seq_params is None
        or packed_seq_params.qkv_format != "thd"
        or not needs_layout_conversion
        or packed_seq_params.cp_layout_views is not None
    ):
        return packed_seq_params
    cp_group = resolve_cp_group(cp_group, packed_seq_params)
    if cp_group is None or cp_group.size() == 1:
        return packed_seq_params
    from .utils import get_batches_on_this_cp_rank

    params = packed_seq_params
    if params.max_seqlen_q is None:
        raise ValueError("Precompute max_seqlen_q at the packed batch producer")
    if params.cu_seqlens_kv is not None and not torch.equal(
        params.cu_seqlens_q, params.cu_seqlens_kv
    ):
        raise ValueError("Module CP conversion requires matching self-attention Q/KV boundaries")
    mode = params.cp_partition_mode or "zigzag"
    physical = params.cu_seqlens_q_padded
    batch = {
        "cu_seqlens": params.cu_seqlens_q.unsqueeze(0),
        "cu_seqlens_padded": None if physical is None else physical.unsqueeze(0),
        "max_seqlen": params.cu_seqlens_q.new_tensor([params.max_seqlen_q]),
    }
    dynamic = params.local_cp_size is not None
    if dynamic:
        batch["local_cp_size"] = params.cu_seqlens_q.new_tensor(params.local_cp_size)
        batch["dynamic_cp_group"] = cp_group
    tp_cp_group = resolve_tp_cp_group(
        tp_cp_group, params, sequence_parallel=sequence_parallel, tp_group=tp_group
    )
    views = get_batches_on_this_cp_rank(
        batch,
        boundary_layout=mode,
        dynamic_cp=dynamic,
        cp_group=cp_group,
        additional_layouts=("contiguous", "zigzag"),
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
        tokens_per_sample=params.tokens_per_sample,
    )
    if mode == "zigzag" and (
        views.thd_plan.zigzag_local_token_count != views.thd_plan.contiguous_local_token_count
    ):
        raise ValueError(
            "An already-sharded zigzag input must include its physical padding metadata"
        )
    return views.prepare_module_layouts().get_packed_seq_params()
