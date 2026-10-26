# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dispatch for cuDNN's aligned HCA backward."""

from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams


@lru_cache(maxsize=1)
def _get_aligned_hca_backward():
    """Load the optional API only for eligible calls."""
    try:
        from cudnn import AlignedHCABackward, aligned_hca_backward_wrapper
    except ImportError as exc:
        raise ImportError(
            "hca_aligned_backward requires cuDNN Frontend with "
            "aligned_hca_backward_wrapper (NVIDIA/cudnn-frontend#1198)."
        ) from exc
    return AlignedHCABackward, aligned_hca_backward_wrapper


def prepare_aligned_hca_metadata(packed_seq_params: PackedSeqParams) -> None:
    """Refresh the eager packing verdict once per microbatch, before model execution."""
    packed_seq_params._aligned_hca_sequence = None
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    sequence_length = packed_seq_params.max_seqlen_q
    if (
        packed_seq_params.qkv_format != "thd"
        or cu is None
        or cu.ndim != 1
        or cu.numel() < 2
        or not isinstance(sequence_length, int)
        or cu.is_inference()
        or (cu.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        return
    single_sequence = ((cu[0] == 0) & (cu[1:] == sequence_length).all()).item()
    packed_seq_params._aligned_hca_sequence = (cu, cu._version, sequence_length, single_sequence)


def _is_single_sequence(
    packed_seq_params: PackedSeqParams, cu: Tensor, sequence_length: int
) -> bool:
    """Reuse prepared metadata only while its tensor and contents are unchanged."""
    prepared = getattr(packed_seq_params, "_aligned_hca_sequence", None)
    if (
        prepared is not None
        and prepared[0] is cu
        and not cu.is_inference()
        and prepared[1] == cu._version
        and prepared[2] == sequence_length
    ):
        return prepared[3]
    return ((cu[0] == 0) & (cu[1:] == sequence_length).all()).item()


def aligned_hca_cp_rank(
    query: Tensor,
    kv: Tensor,
    packed_seq_params: PackedSeqParams,
    cp_group: torch.distributed.ProcessGroup,
    *,
    window_size: int,
    compress_ratio: int,
    boundary_rows: int,
) -> Optional[int]:
    """Select a single-sequence aligned layout supported by cuDNN."""
    if query.ndim != 3:
        return None
    local_tokens, cp_size = query.shape[0], cp_group.size()
    sequence_length = local_tokens * cp_size
    compressed_rows = cp_size * (local_tokens // 128 + 1)
    if (
        cp_size not in (4, 8, 16)
        or not 8192 <= sequence_length <= 131072
        or local_tokens % 128 != 0
        or packed_seq_params.qkv_format != "thd"
        or packed_seq_params.cp_partition_mode != "contiguous"
        or packed_seq_params.max_seqlen_q != sequence_length
        or window_size != 128
        or compress_ratio != 128
        or boundary_rows != 128
        or tuple(query.shape[1:]) != (128, 512)
        or tuple(kv.shape) != (local_tokens + 128 + compressed_rows, 512)
        or query.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or not query.is_cuda
        or kv.device != query.device
    ):
        return None
    cu = packed_seq_params.cu_seqlens_q_padded
    if cu is None:
        cu = packed_seq_params.cu_seqlens_q
    if cu is None or cu.ndim != 1 or cu.numel() < 2 or cu.device != query.device:
        return None
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing and not _is_single_sequence(packed_seq_params, cu, sequence_length):
        return None
    api, _ = _get_aligned_hca_backward()
    if not api.supports_configuration(local_tokens, cp_size, query.device):
        return None
    if capturing:
        # Recheck device metadata on every replay, including reused input buffers.
        torch._assert_async(
            (cu[0] == 0) & (cu[1:] == sequence_length).all(),
            "hca_aligned_backward CUDA graphs require one aligned padded sequence; "
            "disable the flag for changing pack boundaries.",
        )
    return cp_group.rank()


def aligned_hca_backward(
    q: Tensor,
    kv: Tensor,
    out: Tensor,
    dout: Tensor,
    lse: Tensor,
    attn_sink: Tensor,
    *,
    cp_rank: int,
    softmax_scale: float,
    cp_size: int = 16,
):
    """Normalize strides after output RoPE and invoke the optional cuDNN API."""
    _, backward = _get_aligned_hca_backward()
    return backward(
        q.contiguous(),
        kv.contiguous(),
        out.contiguous(),
        dout.contiguous(),
        lse.contiguous(),
        attn_sink.contiguous(),
        cp_rank=cp_rank,
        softmax_scale=softmax_scale,
        cp_size=cp_size,
    )
