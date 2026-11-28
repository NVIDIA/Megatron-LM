# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_build_packed_seq_idx_is_capturable_and_follows_refreshed_boundaries():
    """A chunk-wise CUDA graph rebuilds PackedSeqParams from its static cu_seqlens buffers, so a
    Mamba mixer derives seq_idx inside the capture: the helper must not upload host values, and
    a replay has to follow the boundaries copied into the buffers for that microbatch."""
    total_tokens = 16
    packings = [[0, 5, 7, 11], [0, 3, 3, 16], [0, 16, 16, 16], [0, 1, 2, 3]]

    def packed(boundaries):
        cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        return PackedSeqParams(
            qkv_format="thd", cu_seqlens_q=cu_seqlens, cu_seqlens_q_padded=cu_seqlens.clone()
        )

    static_params = packed(packings[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = build_packed_seq_idx(static_params, total_tokens)

    for boundaries in packings[1:] + packings[:1]:
        refreshed = torch.tensor(boundaries, dtype=torch.int32, device="cuda")
        static_params.cu_seqlens_q.copy_(refreshed)
        static_params.cu_seqlens_q_padded.copy_(refreshed)
        graph.replay()
        eager = build_packed_seq_idx(packed(boundaries), total_tokens)
        assert captured.dtype == eager.dtype == torch.int32
        assert torch.equal(captured, eager), boundaries
    # The first packing: [0,0,0,0,0, 1,1, 2,2,2,2, 3,3,3,3,3].
    assert captured.tolist() == [[0] * 5 + [1] * 2 + [2] * 4 + [3] * 5]
