# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed-sequence metadata helpers for CP partition-mode tracking."""

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from megatron.core.packed_seq_params import PackedSeqParams


def get_packed_seq_params_cp_partition_cu_seqlens(
    packed_seq_params: Optional["PackedSeqParams"],
) -> Optional[torch.Tensor]:
    """Return THD cumulative sequence lengths used for CP layout conversion.

    ``packed_seq_params=None`` represents the ordinary SBHD path. Only THD
    metadata carries global packed-token boundaries.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def finalize_packed_seq_params(
    packed_seq_params: Optional["PackedSeqParams"], *, sequence_parallel: bool = False
) -> Optional["PackedSeqParams"]:
    """Resolve CP metadata and prebuild the THD layout routes for a microbatch.

    The CP-only route is always prebuilt; the fused TP x CP route next to it only under
    sequence parallelism (see ``prebuild_thd_cp_partition_routes``).

    Args:
        packed_seq_params: Metadata for this microbatch, or None for unpacked input.
        sequence_parallel: Whether the model shards the sequence over tensor parallelism.
            Only then are the module inputs sequence-parallel shards whose layout
            conversion exchanges them over the TP x CP group, so the TP groups are passed
            to the prebuild only when set and it builds the fused TP x CP route as well.
            A route the prebuild did not anticipate is built lazily from the cached host
            ``cu_seqlens`` without an extra device-to-host copy.
    """
    if packed_seq_params is None:
        return None

    # Keep these imports local: routes depends on this module for metadata access.
    from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
    from megatron.core.packed_seq_params import resolve_cp_group
    from megatron.core.parallel_state import (
        get_context_parallel_group,
        get_dynamic_tensor_data_context_parallel_group_map,
        get_tensor_and_context_parallel_group,
        get_tensor_model_parallel_group,
    )

    # Query MPU's global CP group only when the metadata carries no runtime or caller-owned
    # group: callers that own their process groups may leave the global group uninitialized.
    static_cp_group = None
    if packed_seq_params.local_cp_size is None and packed_seq_params.cp_group is None:
        static_cp_group = get_context_parallel_group()
    cp_group = resolve_cp_group(static_cp_group, packed_seq_params)
    if packed_seq_params.local_cp_size == 1:
        packed_seq_params.cp_singleton_group = cp_group
        packed_seq_params.cp_group = None
    else:
        packed_seq_params.cp_group = cp_group
    # The TP x CP group that pairs with this microbatch's CP group. Under dynamic CP the
    # CP group is a sub-group, whose TP x sub-group counterpart exists next to it; store
    # it on the metadata so modules pick it up through resolve_tp_cp_group when converting
    # sequence-parallel shards. Static CP leaves the field unset, and modules keep using
    # their own process-group collection. The map snapshot is keyed by the runtime CP
    # size and is empty when dynamic CP is off, so a ``local_cp_size`` without a matching
    # group (for example a static-CP caller that fills the field) resolves to None without
    # touching the TP parallel state.
    if getattr(packed_seq_params, "local_cp_size", None) is not None:
        tp_cp_group = get_dynamic_tensor_data_context_parallel_group_map().get(
            packed_seq_params.local_cp_size
        )
        packed_seq_params.tp_cp_group = tp_cp_group
    else:
        tp_cp_group = get_tensor_and_context_parallel_group(check_initialized=False)
    # Sequence-parallel THD layout conversion exchanges shards directly over that TP x CP
    # group. Prebuild the route while the CUDA queue is still shallow: the fused TP x CP
    # route for sequence-parallel shards, otherwise the CP-only route. Pass the resolved
    # runtime CP group explicitly: the singleton contract above leaves
    # ``packed_seq_params.cp_group`` unset when CP is off for this microbatch.
    tp_group = (
        get_tensor_model_parallel_group(check_initialized=False) if sequence_parallel else None
    )
    prebuild_thd_cp_partition_routes(
        packed_seq_params,
        cp_group,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group if sequence_parallel else None,
    )
    return packed_seq_params
