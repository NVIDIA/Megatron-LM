# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.packed_seq_helpers import build_packed_seq_idx


def test_build_packed_seq_idx_covers_sequences_and_trailing_padding():
    packed_seq_params = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32)
    )

    seq_idx = build_packed_seq_idx(packed_seq_params, total_tokens=8)

    assert torch.equal(seq_idx, torch.tensor([[0, 0, 1, 1, 1, 2, 2, 2]], dtype=torch.int32))


def test_build_packed_seq_idx_prefers_padded_boundaries():
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )

    seq_idx = build_packed_seq_idx(packed_seq_params, total_tokens=8)

    assert torch.equal(seq_idx, torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.int32))
