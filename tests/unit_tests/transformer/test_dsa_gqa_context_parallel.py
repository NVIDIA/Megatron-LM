# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parity tests for DSA-over-GQA context parallelism.

Context parallelism is exercised here without a process group. After the all-gather a rank's
state is exactly "full K and V, this rank's slice of Q, this rank's global query positions", and
that state can be constructed directly. Emulating it single-process tests the part that is
genuinely new -- threading global positions through the tiling loops -- and it runs on any GPU
count, which matters because the real cp=2 configuration needs ranks this cluster allocates in
groups of four alongside TP.

What this does *not* cover is the collective itself: whether the gather's backward reduce-scatters
rather than splits, and whether the learned-K reduction spans the group. Those need a live process
group and live in test_dsa_gqa_context_parallel_dist.py, which runs under torch.distributed.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_gqa import (
    _gather_kv_for_context_parallel,
    _simplified_indexer_norm_spec,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_zigzag_allgather_cp_key_reorder,
    build_zigzag_cp_local_positions,
)
from megatron.core.transformer.experimental_attention_variant.dsa_min_memory import (
    _simplified_topk_index_tile,
    _tile_global_start,
    _validate_query_positions,
    dsa_min_memory_gqa,
)
from tests.unit_tests.transformer.test_attention_variant_dsa_gqa import (
    _simplified_test_indexer,
)

SEQLEN, BATCH, HIDDEN = 16, 2, 12
HEADS, HEAD_DIM, TOPK = 2, 4, 3
CP_SIZE = 2
# chunk_len = SEQLEN // (2 * CP_SIZE) = 4, so a tile of 4 never straddles the two zigzag chunks.
Q_CHUNK = 4


def _inputs(seed=0, requires_grad=False):
    torch.manual_seed(seed)
    q = torch.randn(SEQLEN, BATCH, HEADS, HEAD_DIM, requires_grad=requires_grad)
    k = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, requires_grad=requires_grad)
    v = torch.randn(SEQLEN, BATCH, 1, HEAD_DIM, requires_grad=requires_grad)
    h = torch.randn(SEQLEN, BATCH, HIDDEN)
    return q, k, v, h


def _input_norm():
    """The indexer input-norm spec the min-memory backward path expects."""
    linear_qkv = SimpleNamespace(
        layer_norm_weight=torch.randn(HIDDEN),
        layer_norm_bias=None,
        eps=1.0e-5,
        skip_norm_and_all_gather=False,
    )
    norm_config = SimpleNamespace(
        normalization="RMSNorm", layernorm_epsilon=1.0e-5, layernorm_zero_centered_gamma=False
    )
    return _simplified_indexer_norm_spec(linear_qkv, norm_config)


def _run(
    query, key, value, hidden, indexer, query_positions=None, input_norm=None, q_chunk=Q_CHUNK
):
    """Invoke the min-memory sparse path the way _forward_min_memory does."""
    return dsa_min_memory_gqa(
        query,
        key,
        value,
        hidden.detach(),
        indexer,
        HEAD_DIM**-0.5,
        0.1,
        False,
        query_chunk_size=q_chunk,
        key_chunk_size=SEQLEN,
        use_triton=False,
        simplified_input_norm=input_norm,
        query_positions=query_positions,
    )


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


class TestForwardParity:
    """cp=2 emulated against a full-sequence cp=1 reference."""

    def test_topk_selection_agrees_with_cp1(self):
        """Top-k is discrete, so exact agreement is the sharpest signal of correct masking."""
        q, k, v, h = _inputs()
        indexer = _simplified_test_indexer(HIDDEN, HEAD_DIM, TOPK, learned_k=False)
        w = indexer.linear_q.weight

        for rank in range(CP_SIZE):
            pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, rank, q.device)
            for q_lo in range(0, pos.numel(), Q_CHUNK):
                q_hi = q_lo + Q_CHUNK
                g_start = int(pos[q_lo].item())
                _, local_idx, _ = _simplified_topk_index_tile(
                    h[pos],
                    k,
                    q_lo,
                    q_hi,
                    w,
                    TOPK,
                    HEAD_DIM,
                    0,
                    None,
                    False,
                    False,
                    HEAD_DIM**-0.5,
                    SEQLEN,
                    q_pos_start=g_start,
                )
                _, ref_idx, _ = _simplified_topk_index_tile(
                    h,
                    k,
                    g_start,
                    g_start + Q_CHUNK,
                    w,
                    TOPK,
                    HEAD_DIM,
                    0,
                    None,
                    False,
                    False,
                    HEAD_DIM**-0.5,
                    SEQLEN,
                )
                assert torch.equal(
                    local_idx, ref_idx
                ), f"rank {rank} tile {q_lo}:{q_hi} (global {g_start}) selected different keys"

    def test_output_matches_cp1(self):
        q, k, v, h = _inputs()
        indexer = _simplified_test_indexer(HIDDEN, HEAD_DIM, TOPK, learned_k=False)
        ref_out, ref_loss = _run(q, k, v, h, indexer)

        for rank in range(CP_SIZE):
            pos = build_zigzag_cp_local_positions(SEQLEN, CP_SIZE, rank, q.device)
            # Post-gather state: K and V full length, Q and hidden_states local.
            out, _ = _run(q[pos], k, v, h[pos], indexer, query_positions=pos)
            torch.testing.assert_close(out, ref_out[pos], atol=1e-5, rtol=1e-5)
