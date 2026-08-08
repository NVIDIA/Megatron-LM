# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import torch

from megatron.lite.primitive.parallel import (
    ParallelState,
    build_pipeline_chunk_layout,
    reconstruct_packed_from_cp_parts,
    roll_packed_thd_left,
    split_packed_to_cp_local,
    zigzag_position_ids_for_cp,
    zigzag_reconstruct_from_cp_parts,
    zigzag_slice_for_cp,
    zigzag_split_for_cp,
)


def test_cp_zigzag_split_slice_and_reconstruct_match():
    tensor = torch.arange(16).reshape(1, 8, 2)
    parts = [zigzag_split_for_cp(tensor, rank, cp_size=2, seq_dim=1) for rank in range(2)]

    assert torch.equal(parts[0], tensor[:, [0, 1, 6, 7], :])
    assert torch.equal(parts[1], tensor[:, [2, 3, 4, 5], :])
    assert torch.equal(zigzag_slice_for_cp(tensor, 0, cp_size=2, seq_dim=1), parts[0])
    assert torch.equal(zigzag_slice_for_cp(tensor, 1, cp_size=2, seq_dim=1), parts[1])
    assert torch.equal(zigzag_reconstruct_from_cp_parts(parts, seq_dim=1), tensor)


def test_cp_position_ids_follow_zigzag_order():
    assert torch.equal(
        zigzag_position_ids_for_cp(8, cp_rank=0, cp_size=2, device=torch.device("cpu")),
        torch.tensor([[0, 1, 6, 7]]),
    )
    assert torch.equal(
        zigzag_position_ids_for_cp(8, cp_rank=1, cp_size=2, device=torch.device("cpu")),
        torch.tensor([[2, 3, 4, 5]]),
    )


def test_pp_layout_marks_stage_boundaries_and_vpp_chunks():
    rank0 = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    rank1 = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)

    assert build_pipeline_chunk_layout(8, rank0).layer_indices == [0, 1, 2, 3]
    assert build_pipeline_chunk_layout(8, rank0).has_embed is True
    assert build_pipeline_chunk_layout(8, rank0).has_head is False
    assert build_pipeline_chunk_layout(8, rank1).layer_indices == [4, 5, 6, 7]
    assert build_pipeline_chunk_layout(8, rank1).has_embed is False
    assert build_pipeline_chunk_layout(8, rank1).has_head is True

    vpp_rank0_chunk1 = build_pipeline_chunk_layout(8, rank0, vpp=2, vpp_chunk_id=1)
    vpp_rank1_chunk1 = build_pipeline_chunk_layout(8, rank1, vpp=2, vpp_chunk_id=1)
    assert vpp_rank0_chunk1.layer_indices == [4, 5]
    assert vpp_rank0_chunk1.has_head is False
    assert vpp_rank1_chunk1.layer_indices == [6, 7]
    assert vpp_rank1_chunk1.has_head is True


def test_thd_roll_keeps_sequence_boundaries():
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    rolled, token_sum = roll_packed_thd_left(torch.arange(8), cu_seqlens_padded=cu_seqlens, dims=0)

    assert torch.equal(rolled, torch.tensor([1, 2, 3, 0, 5, 6, 7, 0]))
    assert token_sum.item() == 24


def test_thd_cp_split_and_reconstruct_roundtrip():
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    tensor = torch.arange(8)
    parts = [
        split_packed_to_cp_local(
            tensor, cu_seqlens_padded=cu_seqlens, cp_size=2, cp_rank=rank, dim=0
        )
        for rank in range(2)
    ]

    assert torch.equal(parts[0], torch.tensor([0, 1, 6, 7]))
    assert torch.equal(parts[1], torch.tensor([2, 3, 4, 5]))
    assert torch.equal(
        reconstruct_packed_from_cp_parts(parts, cu_seqlens_padded=cu_seqlens, cp_size=2, dim=0),
        tensor,
    )
