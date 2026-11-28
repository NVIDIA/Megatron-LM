# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed metadata factories shared by attention correctness tests."""

import torch

from megatron.core.packed_seq_params import PackedSeqParams


def make_test_packed_seq_params(sequence_length=None, cu_seqlens=None):
    if cu_seqlens is None:
        assert sequence_length is not None
        cu_seqlens = [0, 6, 19, 22, sequence_length]
    cu_seqlens = torch.IntTensor(cu_seqlens).cuda()
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = seqlens.max().item()
    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params


def make_test_packed_seq_params_with_padding(
    sequence_length=None, cu_seqlens=None, cu_seqlens_padded=None
):
    """Create PackedSeqParams with both regular and padded cu_seqlens for testing padded sequences."""
    if cu_seqlens is None:
        assert sequence_length is not None
        cu_seqlens = [
            0,
            6,
            19,
            22,
            sequence_length - 8,
        ]  # Actual sequence lengths (with some padding removed)
    if cu_seqlens_padded is None:
        assert sequence_length is not None
        cu_seqlens_padded = [0, 8, 22, 28, sequence_length]  # Padded sequence lengths

    cu_seqlens = torch.IntTensor(cu_seqlens).cuda()
    cu_seqlens_padded = torch.IntTensor(cu_seqlens_padded).cuda()

    # Use padded lengths for max_seqlen calculation
    seqlens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    max_seqlen, _ = seqlens_padded.max(dim=0, keepdim=True)
    max_seqlen = max_seqlen.tolist()[0]

    packed_seq_params = PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format='thd',
    )
    return packed_seq_params
