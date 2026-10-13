# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the DSA-over-GQA context-parallel helpers.

Covers the pieces that need no process group: the scalar that carries a tile's global position,
the tiling validation, and the gather helper's guards and zigzag reorder. These run on any GPU
count, which matters because the real cp=2 configuration needs ranks this cluster allocates in
groups of four alongside TP.

End-to-end parity -- output, gradients, and the collectives themselves -- lives in
test_dsa_gqa_context_parallel_dist.py, which runs under torch.distributed against a live group.
Emulating a post-gather rank single-process is not possible for the learned-K router: its keys
come from hidden_states, so a local hidden slice yields an index covering only this rank.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    _gather_kv_for_context_parallel,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_zigzag_allgather_cp_key_reorder,
    build_zigzag_cp_local_positions,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _tile_global_start,
    _validate_query_positions,
)

SEQLEN, BATCH, HIDDEN = 16, 2, 12
HEADS, HEAD_DIM = 2, 4
CP_SIZE = 2


def _inputs(seed=0, requires_grad=False):
    torch.manual_seed(seed)
    q = torch.randn(SEQLEN, BATCH, HEADS, HEAD_DIM, requires_grad=requires_grad)
    k = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, requires_grad=requires_grad)
    v = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, requires_grad=requires_grad)
    h = torch.randn(SEQLEN, BATCH, HIDDEN)
    return q, k, v, h


class TestPositionHelpers:
    """The scalar plumbing that carries a tile's global position."""

    def test_tile_global_start_is_identity_without_cp(self):
        assert _tile_global_start(None, 0) == 0
        assert _tile_global_start(None, 7) == 7

    def test_tile_global_start_reads_the_positions_tensor(self):
        pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, 1, torch.device("cpu"))
        # rank 1 of 2 holds chunks 1 and 2 of 4, i.e. global 4..7 then 8..11
        assert pos.tolist() == [4, 5, 6, 7, 8, 9, 10, 11]
        assert _tile_global_start(pos, 0) == 4
        assert _tile_global_start(pos, 4) == 8

    def test_validate_accepts_tiles_that_stay_within_a_chunk(self):
        pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, 0, torch.device("cpu"))
        _validate_query_positions(pos, pos.numel(), 4)
        _validate_query_positions(pos, pos.numel(), 2)

    def test_validate_rejects_a_tile_straddling_two_chunks(self):
        """A tile spanning both zigzag chunks cannot be described by one scalar start."""
        pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, 0, torch.device("cpu"))
        with pytest.raises(ValueError, match="spans a discontinuity"):
            _validate_query_positions(pos, pos.numel(), 8)

    def test_validate_is_a_noop_without_cp(self):
        _validate_query_positions(None, SEQLEN, 8)


class TestGatherHelper:
    """Shape and ordering contract of the all-gather, without a live process group."""

    def test_single_rank_is_a_noop(self):
        q, k, v, _ = _inputs()
        gk, gv, pos = _gather_kv_for_context_parallel(k, v, None, "allgather")
        assert gk is k and gv is v and pos is None

    def test_non_allgather_comm_type_is_rejected(self):
        q, k, v, _ = _inputs()
        fake_group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
        with pytest.raises(NotImplementedError, match="allgather"):
            _gather_kv_for_context_parallel(k, v, fake_group, "p2p")

    def test_reorder_restores_global_position_order(self):
        """Gathered tensors arrive in rank order; the reorder must sort them by position."""
        device = torch.device("cpu")
        gathered_positions = torch.cat(
            [build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, r, device) for r in range(CP_SIZE)]
        )
        reorder = build_zigzag_allgather_cp_key_reorder(SEQLEN // CP_SIZE, CP_SIZE, device)
        assert gathered_positions[reorder].tolist() == list(range(SEQLEN))
