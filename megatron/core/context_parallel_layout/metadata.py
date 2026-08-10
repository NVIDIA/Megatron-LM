# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Packed-sequence metadata helpers for CP partition-mode tracking."""

from typing import Any, Optional

import torch

from megatron.core.context_parallel_layout import CpPartitionMode


def get_packed_seq_params_cp_partition_cu_seqlens(
    packed_seq_params: Optional[Any],
) -> Optional[torch.Tensor]:
    """Return THD cumulative sequence lengths used for CP layout conversion.

    SBHD callers may still pass synthetic ``PackedSeqParams`` for CP layout
    annotation. Only THD metadata carries global packed-token boundaries.
    """
    if packed_seq_params is None or getattr(packed_seq_params, "qkv_format", None) != "thd":
        return None
    return (
        packed_seq_params.cu_seqlens_q_padded
        if packed_seq_params.cu_seqlens_q_padded is not None
        else packed_seq_params.cu_seqlens_q
    )


def replace_packed_seq_params_cp_partition_mode(
    packed_seq_params: Optional[Any], cp_partition_mode: Optional[CpPartitionMode]
) -> Optional[Any]:
    """Annotate packed-sequence metadata with the current CP partition mode."""
    if packed_seq_params is None:
        return packed_seq_params
    if getattr(packed_seq_params, "cp_partition_mode", None) == cp_partition_mode:
        return packed_seq_params
    packed_seq_params.cp_partition_mode = cp_partition_mode
    return packed_seq_params
