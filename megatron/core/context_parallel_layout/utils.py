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
    packed_seq_params: Optional["PackedSeqParams"],
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Optional["PackedSeqParams"]:
    """Resolve CP metadata and prebuild the THD layout route for a microbatch.

    Args:
        packed_seq_params: Packed-sequence metadata for the microbatch.
        cp_group: Caller-provided static context-parallel process group. A
            per-microbatch group stored in ``packed_seq_params`` takes precedence.

    Returns:
        The finalized packed-sequence metadata, or ``None`` when no metadata was provided.
    """
    if packed_seq_params is None:
        return None

    # Keep these imports local: routes depends on this module for metadata access.
    from megatron.core.context_parallel_layout.routes import prebuild_thd_cp_partition_routes
    from megatron.core.packed_seq_params import resolve_cp_group

    cp_group = resolve_cp_group(static_cp_group=cp_group, packed_seq_params=packed_seq_params)
    prebuild_thd_cp_partition_routes(packed_seq_params=packed_seq_params, cp_group=cp_group)
    return packed_seq_params
