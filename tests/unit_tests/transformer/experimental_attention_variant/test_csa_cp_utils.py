# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_TWO_CHUNK,
    apply_thd_cp_local_rope_fused,
    build_cp_compressor_prep_compact_fused,
    compute_cp_indexer_topk_logical_fused,
    exchange_cp_boundary_hidden,
    local_kv_cp_chunk_ranges,
    local_q_cp_chunk_ranges,
    thd_cp_local_row_indices,
)
from tests.unit_tests.test_utilities import Utils


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CUDA utility tests require CUDA.")


def test_csa_cp_partition_mode_contract():
    """Validate the public CSA CP partition-mode names.

    Expected: omitted mode defaults to the original contiguous partition,
    supported strings round-trip exactly, and unknown values fail before any CP
    layout helper can silently choose the wrong row order.
    """
    assert local_q_cp_chunk_ranges(None, 4, 1, 0) == ((0, 4),)
    assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_CONTIGUOUS, 4, 1, 0) == ((0, 4),)
    assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 4, 1, 0) == ((0, 4),)

    with pytest.raises(RuntimeError, match="Unsupported CSA CP partition mode"):
        local_q_cp_chunk_ranges("invalid_mode", 4, 1, 0)


@pytest.mark.parametrize(
    "partition_mode, uses_two_chunk",
    [(DSV4_CP_PARTITION_CONTIGUOUS, False), (DSV4_CP_PARTITION_TWO_CHUNK, True)],
    ids=["contiguous", "two_chunk"],
)
def test_thd_cp_partition_mode_selects_expected_row_order(partition_mode, uses_two_chunk):
    """Validate the CP row order selected by each partition mode.

    Expected: ``contiguous`` returns this rank's single local range, while
    ``two_chunk`` returns chunk ``rank`` followed by chunk
    ``2*cp_size-1-rank``. A failure means CP data slicing and layer-side layout
    helpers could disagree about local packed-token order.
    """
    local_rows = 16
    cp_size = 4

    for cp_rank in range(cp_size):
        ranges = local_q_cp_chunk_ranges(partition_mode, local_rows, cp_size, cp_rank)
        if uses_two_chunk:
            chunk_len = local_rows // 2
            total_chunks = 2 * cp_size
            chunk_ids = (cp_rank, total_chunks - 1 - cp_rank)
            expected = tuple(
                (chunk_id * chunk_len, (chunk_id + 1) * chunk_len) for chunk_id in chunk_ids
            )
        else:
            expected = ((cp_rank * local_rows, (cp_rank + 1) * local_rows),)
        assert ranges == expected


def test_two_chunk_cp_ranges_match_expected_order():
    """Validate the two-chunk CP partition contract.

    Expected: rank r owns global chunk r followed by chunk 2*cp_size-1-r.
    A failure here means DSv4 chunk-aware CP helpers would disagree with the
    two-chunk local row order.
    """
    assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 16, 1, 0) == ((0, 16),)
    expected_by_rank = {
        0: ((0, 2), (14, 16)),
        1: ((2, 4), (12, 14)),
        2: ((4, 6), (10, 12)),
        3: ((6, 8), (8, 10)),
    }
    for rank, expected in expected_by_rank.items():
        assert local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 4, 4, rank) == expected

    with pytest.raises(RuntimeError, match="even local_rows"):
        local_q_cp_chunk_ranges(DSV4_CP_PARTITION_TWO_CHUNK, 5, 4, 0)


def test_cp_chunk_ranges_match_partition_mode():
    """Validate local and left-boundary ranges for each CP partition mode.

    Expected: each mode returns the current rank's local rows and matching
    left-boundary rows in global packed-token coordinates.
    """
    assert local_q_cp_chunk_ranges(None, local_rows=4, cp_size=4, cp_rank=2) == ((8, 12),)
    assert local_kv_cp_chunk_ranges(
        DSV4_CP_PARTITION_CONTIGUOUS, local_rows=4, boundary_rows=2, cp_size=4, cp_rank=2
    ) == ((6, 8), (8, 12))

    assert local_q_cp_chunk_ranges(
        DSV4_CP_PARTITION_TWO_CHUNK, local_rows=4, cp_size=4, cp_rank=0
    ) == ((0, 2), (14, 16))
    assert local_kv_cp_chunk_ranges(
        DSV4_CP_PARTITION_TWO_CHUNK, local_rows=4, boundary_rows=4, cp_size=4, cp_rank=0
    ) == ((-2, 0), (12, 14), (0, 2), (14, 16))

    with pytest.raises(RuntimeError, match="boundary rows must be divisible"):
        local_kv_cp_chunk_ranges(
            DSV4_CP_PARTITION_TWO_CHUNK, local_rows=4, boundary_rows=3, cp_size=4, cp_rank=0
        )


def test_thd_cp_local_row_indices_match_chunk_ranges():
    """Validate THD batch row selection for each CSA CP partition mode.

    Expected: data loading selects the same global packed-token rows that the
    attention layer later uses for RoPE, boundary exchange, and index lowering.
    """
    assert torch.equal(
        thd_cp_local_row_indices(DSV4_CP_PARTITION_CONTIGUOUS, 16, 4, 2, torch.device("cpu")),
        torch.tensor([8, 9, 10, 11]),
    )
    assert torch.equal(
        thd_cp_local_row_indices(DSV4_CP_PARTITION_TWO_CHUNK, 16, 4, 0, torch.device("cpu")),
        torch.tensor([0, 1, 14, 15]),
    )
    assert torch.equal(
        thd_cp_local_row_indices(DSV4_CP_PARTITION_TWO_CHUNK, 16, 4, 3, torch.device("cpu")),
        torch.tensor([6, 7, 8, 9]),
    )

    with pytest.raises(RuntimeError, match="divisible by cp_size"):
        thd_cp_local_row_indices(DSV4_CP_PARTITION_CONTIGUOUS, 15, 4, 0, torch.device("cpu"))


def test_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP boundary exchange contract.

    Expected: with cp_group=None, the fixed left boundary is zero-filled and its
    backward path contributes no gradient to the local tensor. A failure here
    means the CP path could invent boundary tokens or bogus local gradients.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_cp_boundary_hidden(local, [], 2, DSV4_CP_PARTITION_CONTIGUOUS, None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_two_chunk_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP two-chunk boundary exchange contract.

    Expected: with cp_group=None, the helper degenerates to one zero-filled
    boundary and contributes no local gradient. A failure here means chunk-aware
    CP boundary plumbing could perturb disabled-CP paths.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_cp_boundary_hidden(local, [], 2, DSV4_CP_PARTITION_TWO_CHUNK, None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_thd_cp_left_boundary_exchange_forward_backward():
    """Validate distributed CP boundary exchange forward/backward.

    Expected: forward receives the previous rank's tail window, and backward
    sends gradient to this rank's tail only when the next rank consumed it as a
    left boundary.
    """
    _require_cuda()
    if Utils.world_size < 2:
        pytest.skip("Distributed CP boundary exchange requires at least 2 ranks.")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=Utils.world_size,
    )
    try:
        cp_group = ProcessGroupCollection.use_mpu_process_groups().cp
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_size = parallel_state.get_context_parallel_world_size()
        d_window = 2
        local_len = 4
        width = 3
        local_numel = local_len * width
        local_start = cp_rank * local_numel
        values = torch.arange(
            local_start, local_start + local_numel, device='cuda', dtype=torch.float32
        ).reshape(local_len, width)
        local = values.detach().clone().requires_grad_(True)

        boundary = exchange_cp_boundary_hidden(
            local, [], d_window, DSV4_CP_PARTITION_CONTIGUOUS, cp_group
        )
        if cp_rank == 0:
            expected_boundary = torch.zeros_like(boundary)
        else:
            left_rank_start = (cp_rank - 1) * local_numel
            expected_boundary = torch.arange(
                left_rank_start + (local_len - d_window) * width,
                left_rank_start + local_numel,
                device='cuda',
                dtype=torch.float32,
            ).reshape(d_window, width)
        assert torch.equal(boundary, expected_boundary)

        boundary.sum().backward()
        expected_grad = torch.zeros_like(local)
        if cp_rank + 1 < cp_size:
            expected_grad[-d_window:] = 1
        assert torch.equal(local.grad, expected_grad)
    finally:
        Utils.destroy_model_parallel()


def test_apply_thd_cp_local_rope_fused_dispatches_equal_and_ragged_chunks(monkeypatch):
    """Validate the public local-RoPE wrapper dispatch contract.

    Expected: equal one/two-chunk layouts call the fused kernel once with both
    chunk starts, while ragged multi-chunk layouts call once per chunk and
    concatenate the chunk results in local row order.
    """
    calls = []

    def fake_apply(
        x,
        cos,
        sin,
        cu,
        chunk0_start,
        chunk_len,
        nope_dim,
        pos_dim,
        chunk1_start,
        inverse,
        clamp_to_valid_token,
    ):
        calls.append(
            (
                tuple(x.shape),
                int(chunk0_start),
                int(chunk_len),
                int(nope_dim),
                int(pos_dim),
                int(chunk1_start),
                bool(inverse),
                bool(clamp_to_valid_token),
            )
        )
        return x + len(calls) * 10

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels.ThdLocalRope, "apply", staticmethod(fake_apply)
    )
    cos = sin = torch.empty(8, 2)
    cu = torch.tensor([0, 8], dtype=torch.int32)

    equal = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    out = apply_thd_cp_local_rope_fused(equal, cos, sin, 1, 1, cu, ((2, 4), (10, 12)), inverse=True)
    assert torch.equal(out, equal + 10)
    assert calls[-1] == ((4, 2), 2, 2, 1, 1, 10, True, False)

    ragged = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    out = apply_thd_cp_local_rope_fused(
        ragged, cos, sin, 1, 1, cu, ((2, 4), (10, 13)), clamp_to_valid_token=True
    )
    assert torch.equal(out, torch.cat((ragged[:2] + 20, ragged[2:] + 30), dim=0))
    assert calls[-2:] == [
        ((2, 2), 2, 2, 1, 1, 0, False, True),
        ((3, 2), 10, 3, 1, 1, 0, False, True),
    ]

    with pytest.raises(RuntimeError, match="expects x rows"):
        apply_thd_cp_local_rope_fused(equal[:3], cos, sin, 1, 1, cu, ((2, 4), (10, 12)))


def test_compressor_prep_compact_wrapper_merges_two_chunk_outputs(monkeypatch):
    """Validate public compressor-prep wrapper splitting, merge, and padding.

    Expected: two chunks are compacted independently, local compact group ids
    are concatenated in local chunk order, and the wrapper pads to the shared
    aligned capacity used by later fixed-shape collectives.
    """
    calls = []

    def fake_apply(hidden, boundary, cu, global_start, rows, ratio, d_comp, d_window, c_cap):
        calls.append((int(global_start), int(rows), int(c_cap), tuple(boundary.shape)))
        marker = len(calls)
        hidden_compact = torch.full((int(c_cap) * int(ratio), 1), marker, dtype=hidden.dtype)
        comp_ids = torch.arange(int(c_cap), dtype=torch.int32) + marker * 10
        return hidden_compact, comp_ids

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels.CompressorInputCompact, "apply", staticmethod(fake_apply)
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    boundary = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    cu = torch.tensor([0, 32], dtype=torch.int32)
    cu_comp = torch.tensor([0, 8], dtype=torch.int32)

    hidden_compact, comp_ids, rank_rows = build_cp_compressor_prep_compact_fused(
        hidden, boundary, cu, cu_comp, ((0, 8), (24, 32)), cp_size=2, ratio=4, d_comp=8, d_window=2
    )

    assert calls == [(0, 8, 4, (2, 1)), (24, 8, 4, (2, 1))]
    assert hidden_compact.shape == (32, 1)
    assert torch.equal(comp_ids, torch.tensor([10, 11, 12, 13, 20, 21, 22, 23]))
    assert torch.equal(rank_rows, torch.tensor([0, 1, 10, 11, 14, 15, 6, 7], dtype=torch.int32))


def test_compute_cp_indexer_topk_logical_fused_splits_chunks_and_passes_visible_lengths(
    monkeypatch,
):
    """Validate public CP indexer-topk wrapper chunk splitting.

    Expected: each local chunk gets its own trapezoid metadata and indexer-topk
    call, visible K lengths are passed through, and chunk outputs are
    concatenated in local row order.
    """
    topk_calls = []

    def fake_indexer_topk(
        q,
        k,
        weights,
        *,
        topk,
        ratio,
        indexer_softmax_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        max_seqlen_q,
        max_seqlen_kv,
        visible_k_lengths,
    ):
        topk_calls.append(
            (
                tuple(q.shape),
                int(topk),
                int(ratio),
                float(indexer_softmax_scale),
                int(max_seqlen_q),
                int(max_seqlen_kv),
                k.clone(),
                cu_seqlens_q.clone(),
                cu_seqlens_kv.clone(),
                visible_k_lengths.clone(),
            )
        )
        return torch.full((q.shape[0], int(topk)), len(topk_calls), dtype=torch.int32), None

    monkeypatch.setattr(csa_cp_utils, "indexer_topk", fake_indexer_topk)

    q = torch.randn(5, 2)
    weights = torch.randn(5, 1)
    k_seq = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    cu_q = torch.tensor([0, 10], dtype=torch.int32)
    cu_comp = torch.tensor([0, 3], dtype=torch.int32)

    out = compute_cp_indexer_topk_logical_fused(
        q,
        weights,
        k_seq,
        cu_q,
        cu_comp,
        ((2, 4), (8, 11)),
        ratio=4,
        topk_width=2,
        indexer_softmax_scale=0.5,
        max_seqlen_q=10,
        max_seqlen_kv=3,
    )

    assert torch.equal(out, torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2]]))
    assert torch.equal(topk_calls[0][-4][0], k_seq[0])
    assert torch.count_nonzero(topk_calls[0][-4][1:]) == 0
    assert torch.equal(topk_calls[0][-3], torch.tensor([0, 2, 2], dtype=torch.int32))
    assert torch.equal(topk_calls[0][-2], torch.tensor([0, 1, 1], dtype=torch.int32))
    assert torch.equal(topk_calls[0][-1], torch.ones(2, dtype=torch.int32))
    assert torch.equal(topk_calls[1][-4][:2], k_seq[:2])
    assert torch.count_nonzero(topk_calls[1][-4][2:]) == 0
    assert torch.equal(topk_calls[1][-3], torch.tensor([0, 2, 3], dtype=torch.int32))
    assert torch.equal(topk_calls[1][-2], torch.tensor([0, 2, 2], dtype=torch.int32))
    assert torch.equal(topk_calls[1][-1], torch.tensor([2, 2, 0], dtype=torch.int32))
    assert (
        compute_cp_indexer_topk_logical_fused(
            q, weights, k_seq[:0], cu_q, cu_comp, ((2, 4),), 4, 2, 1.0, 10, 3
        )
        is None
    )
