# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_TWO_CHUNK,
    all_gather_fixed_cp_tensor,
    apply_thd_cp_compressed_rope_fused,
    apply_thd_cp_local_rope_fused,
    build_cp_attention_indices_fused,
    build_cp_compressor_prep_compact_fused,
    build_cp_indexer_loss_indices_fused,
    build_cp_rank_major_compressed_metadata_fused,
    build_global_compressed_cu_seqlens,
    compute_cp_indexer_topk_logical_fused,
    exchange_cp_boundary_hidden,
    local_kv_cp_chunk_ranges,
    local_q_cp_chunk_ranges,
    pack_cp_kv_full_fused,
    repack_rank_major_compressed_to_seq_major_fused,
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


def test_global_compressed_cu_seqlens_matches_reference():
    """Validate compressed-prefix metadata for ragged padded sequences.

    Expected: each sequence contributes ``padded_seq_len // ratio`` compressed
    rows to the global seq-major prefix. A failure here means rank-major
    compressed rows could be repacked with the wrong sequence offsets.
    """
    _require_cuda()
    cu_seqlens_cpu = torch.tensor([0, 7, 128, 381, 512], dtype=torch.int32)
    ratio = 4

    actual = build_global_compressed_cu_seqlens(cu_seqlens_cpu.cuda(), ratio)

    assert torch.equal(actual.cpu(), torch.tensor([0, 1, 31, 94, 126], dtype=torch.int32))


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


def test_apply_thd_cp_compressed_rope_fused_forwards_kernel_arguments(monkeypatch):
    """Validate the public compressed-RoPE wrapper is a stable typed facade.

    Expected: the wrapper forwards tensor arguments, dimensions, ratio, and
    inverse flag unchanged to the fused kernel entry point.
    """
    captured = {}

    def fake_apply(x, cos, sin, comp_ids, ratio, nope_dim, pos_dim, inverse):
        captured.update(
            ratio=int(ratio),
            nope_dim=int(nope_dim),
            pos_dim=int(pos_dim),
            inverse=bool(inverse),
            comp_ids=comp_ids,
        )
        return x + 1

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels.ThdCompressedRope, "apply", staticmethod(fake_apply)
    )
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    comp_ids = torch.tensor([0, 2, -1], dtype=torch.int32)
    out = apply_thd_cp_compressed_rope_fused(
        x, torch.empty(16, 1), torch.empty(16, 1), comp_ids, 4, 1, 1, inverse=True
    )

    assert torch.equal(out, x + 1)
    assert captured["ratio"] == 4
    assert captured["nope_dim"] == 1
    assert captured["pos_dim"] == 1
    assert captured["inverse"] is True
    assert captured["comp_ids"] is comp_ids


def test_rank_major_compressed_metadata_wrapper_computes_capacities(monkeypatch):
    """Validate public compressed-metadata capacity and partition dispatch.

    Expected: contiguous and two-chunk modes compute aligned capacities before
    delegating to the kernel wrapper, so all ranks use fixed-size all-gather
    rows independent of the number of valid compressed groups.
    """
    calls = []

    def fake_build(cu, cp_size, l_local, ratio, d_comp, c_cap, **kwargs):
        calls.append(
            (
                int(cp_size),
                int(l_local),
                int(ratio),
                int(d_comp),
                int(c_cap),
                kwargs.get("c_cap_per_rank"),
                kwargs.get("use_two_chunk", False),
            )
        )
        rows = int(cp_size) * int(kwargs.get("c_cap_per_rank", c_cap))
        return (
            torch.full((rows,), -1, dtype=torch.int32),
            torch.full((rows,), -1, dtype=torch.int32),
            torch.zeros((rows,), dtype=torch.bool),
        )

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels, "build_compressed_row_metadata", fake_build
    )
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)

    build_cp_rank_major_compressed_metadata_fused(cu, ((16, 32),), 4, 4, 8)
    build_cp_rank_major_compressed_metadata_fused(cu, ((0, 4), (28, 32)), 4, 4, 8)

    assert calls == [(4, 16, 4, 8, 8, None, False), (4, 4, 4, 8, 3, 8, True)]


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
        cu_compact = torch.tensor([0, int(c_cap) * int(ratio)], dtype=cu.dtype)
        seq_ids = torch.full((int(c_cap),), marker, dtype=torch.int32)
        comp_ids = torch.arange(int(c_cap), dtype=torch.int32) + marker * 10
        valid = torch.ones((int(c_cap),), dtype=torch.bool)
        return hidden_compact, cu_compact, seq_ids, comp_ids, valid

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels.CompressorInputCompact, "apply", staticmethod(fake_apply)
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(6, 1)
    boundary = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    cu = torch.tensor([0, 6], dtype=torch.int32)

    hidden_compact, cu_compact, comp_ids = build_cp_compressor_prep_compact_fused(
        hidden, boundary, cu, ((0, 3), (10, 13)), ratio=2, d_comp=2, d_window=2
    )

    assert calls == [(0, 3, 2, (2, 1)), (10, 3, 2, (2, 1))]
    assert hidden_compact.shape == (32, 1)
    assert torch.equal(cu_compact, torch.tensor([0, 8], dtype=torch.int32))
    assert torch.equal(comp_ids[:4], torch.tensor([10, 11, 20, 21], dtype=torch.int32))
    assert torch.equal(comp_ids[4:], torch.full((12,), -1, dtype=torch.int32))


def test_all_gather_fixed_cp_tensor_single_rank_preserves_autograd():
    """Validate the public fixed CP all-gather no-CP fallback.

    Expected: with cp_group=None, the helper returns the local tensor and
    backward is the identity, matching the disabled-CP execution path.
    """
    x = torch.arange(6, dtype=torch.float32).reshape(3, 2).requires_grad_(True)

    out = all_gather_fixed_cp_tensor(x, None)

    assert torch.equal(out, x)
    out.square().sum().backward()
    assert torch.equal(x.grad, 2 * x.detach())


def test_pack_cp_kv_full_fused_wrapper_selects_contiguous_and_two_chunk_layouts(monkeypatch):
    """Validate public KV-pack wrapper argument lowering for both layouts.

    Expected: contiguous CP passes the physical local range through directly,
    while two-chunk CP switches to shared-window layout and returns the shared
    compressed base used by final index lowering.
    """
    calls = []

    def fake_apply(*args):
        calls.append(args)
        kv_local = args[0]
        capacity = int(args[13])
        return kv_local.new_full((capacity,) + tuple(kv_local.shape[1:]), len(calls))

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels.ThdFullKvPack, "apply", staticmethod(fake_apply)
    )
    kv_local = torch.ones(4, 2)
    boundary = torch.ones(2, 2)
    compressed = torch.ones(6, 2)
    ids = torch.zeros(6, dtype=torch.int32)
    valid = torch.ones(6, dtype=torch.bool)
    cu = torch.tensor([0, 8], dtype=torch.int32)
    rank_by_seq = torch.arange(2, dtype=torch.int32)
    cu_comp = torch.tensor([0, 2], dtype=torch.int32)

    out, shared = pack_cp_kv_full_fused(
        kv_local,
        boundary,
        compressed,
        ids,
        ids,
        valid,
        cu,
        ((4, 8),),
        d_window=2,
        ratio=4,
        rank_major_by_seq_major=rank_by_seq,
        cu_seqlens_compressed=cu_comp,
    )
    assert shared is None
    assert out.shape == (12, 2)
    assert calls[-1][9:19] == (4, 4, 2, 4, 12, 0, 0, 0, 0, 0)

    out, shared = pack_cp_kv_full_fused(
        kv_local,
        torch.ones(4, 2),
        compressed,
        ids,
        ids,
        valid,
        cu,
        ((0, 2), (14, 16)),
        d_window=2,
        ratio=4,
    )
    assert shared == 8
    assert out.shape == (14, 2)
    assert calls[-1][9:19] == (0, 0, 2, 0, 14, 0, 14, 2, 2, 4)


def test_repack_rank_major_compressed_to_seq_major_fused_forwards_arguments(monkeypatch):
    """Validate the public compressed-KV repack wrapper.

    Expected: the wrapper forwards rank-major tensors and requested seq-major
    capacity unchanged to the fused repack kernel.
    """
    captured = {}

    def fake_repack(rank_major, seq_ids, comp_ids, valid, cu_comp, rows):
        captured.update(rows=int(rows), rank_major=rank_major, seq_ids=seq_ids)
        return rank_major[:rows].clone(), torch.arange(rows, dtype=torch.int32)

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels, "repack_compressed_kv_to_seq_major", fake_repack
    )
    rank_major = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    seq_ids = torch.zeros(6, dtype=torch.int32)
    comp_ids = torch.arange(6, dtype=torch.int32)
    valid = torch.ones(6, dtype=torch.bool)
    cu_comp = torch.tensor([0, 3], dtype=torch.int32)

    out, rank_by_seq = repack_rank_major_compressed_to_seq_major_fused(
        rank_major, seq_ids, comp_ids, valid, cu_comp, 3
    )

    assert captured["rows"] == 3
    assert captured["rank_major"] is rank_major
    assert captured["seq_ids"] is seq_ids
    assert torch.equal(out, rank_major[:3])
    assert torch.equal(rank_by_seq, torch.arange(3, dtype=torch.int32))


def test_compute_cp_indexer_topk_logical_fused_splits_chunks_and_passes_visible_lengths(
    monkeypatch,
):
    """Validate public CP indexer-topk wrapper chunk splitting.

    Expected: each local chunk gets its own trapezoid metadata and indexer-topk
    call, visible K lengths are passed through, and chunk outputs are
    concatenated in local row order.
    """
    metadata_calls = []
    topk_calls = []

    def fake_metadata(k_seq, cu_q, cu_comp, global_start, length, ratio):
        metadata_calls.append((int(global_start), int(length), int(ratio)))
        marker = len(metadata_calls)
        return (
            k_seq + marker,
            torch.tensor([0, int(length)], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.full((int(length),), marker, dtype=torch.int32),
        )

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
                visible_k_lengths.clone(),
            )
        )
        return torch.full((q.shape[0], int(topk)), len(topk_calls), dtype=torch.int32), None

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels, "build_indexer_topk_metadata", fake_metadata
    )
    monkeypatch.setattr(csa_cp_utils, "indexer_topk", fake_indexer_topk)

    q = torch.randn(5, 2)
    weights = torch.randn(5, 1)
    k_seq = torch.randn(4, 2)
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

    assert metadata_calls == [(2, 2, 4), (8, 3, 4)]
    assert torch.equal(out, torch.tensor([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2]]))
    assert torch.equal(topk_calls[0][-1], torch.ones(2, dtype=torch.int32))
    assert torch.equal(topk_calls[1][-1], torch.full((3,), 2, dtype=torch.int32))
    assert (
        compute_cp_indexer_topk_logical_fused(
            q, weights, k_seq[:0], cu_q, cu_comp, ((2, 4),), 4, 2, 1.0, 10, 3
        )
        is None
    )


def test_final_index_wrappers_lower_two_chunk_arguments(monkeypatch):
    """Validate public final attention and indexer-loss index wrappers.

    Expected: two-chunk wrappers require shared compressed metadata and pass
    chunk starts, chunk length, shared base, and local row count to the fused
    kernel entry points.
    """
    captured = {}

    def fake_attention(*args, **kwargs):
        captured["attention"] = (args, kwargs)
        l_local = int(args[2])
        width = int(args[6])
        return torch.zeros(l_local, width, dtype=torch.int32), torch.zeros(
            l_local, dtype=torch.int32
        )

    def fake_loss(*args, **kwargs):
        captured["loss"] = (args, kwargs)
        l_local = int(args[3])
        width = args[7].shape[1] + int(args[5])
        return torch.zeros(l_local, width, dtype=torch.int32), torch.zeros_like(args[7])

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels, "build_attention_indices", fake_attention
    )
    monkeypatch.setattr(csa_cp_utils.csa_cp_layout_kernels, "build_indexer_loss_indices", fake_loss)

    cu = torch.tensor([0, 16], dtype=torch.int32)
    cu_comp = torch.tensor([0, 4], dtype=torch.int32)
    rank_by_seq = torch.arange(4, dtype=torch.int32)
    logical = torch.zeros(4, 2, dtype=torch.int32)
    chunks = ((0, 2), (14, 16))

    with pytest.raises(RuntimeError, match="requires a shared compressed base"):
        build_cp_attention_indices_fused(cu, chunks, 2, 2, 4, logical, 2)

    build_cp_attention_indices_fused(
        cu,
        chunks,
        d_window=2,
        window_size=2,
        ratio=4,
        indexer_topk_compressed_logical_ids=logical,
        max_n_compressed=2,
        rank_major_by_seq_major=rank_by_seq,
        cu_seqlens_compressed=cu_comp,
        shared_compressed_base=8,
    )
    args, kwargs = captured["attention"]
    assert args[0] is cu
    assert args[1:7] == (0, 4, 2, 2, 4, 2)
    assert kwargs["chunk_starts"] == (0, 14)
    assert kwargs["chunk_len"] == 2
    assert kwargs["shared_compressed_base"] == 8

    build_cp_indexer_loss_indices_fused(
        cu,
        cu_comp,
        chunks,
        d_window=2,
        window_size=2,
        ratio=4,
        indexer_topk_compressed_logical_ids=logical,
        rank_major_by_seq_major=rank_by_seq,
        shared_compressed_base=8,
    )
    args, kwargs = captured["loss"]
    assert args[0] is cu
    assert args[1] is cu_comp
    assert args[2:7] == (0, 4, 2, 2, 4)
    assert args[7] is logical
    assert kwargs["chunk_starts"] == (0, 14)
    assert kwargs["chunk_len"] == 2
    assert kwargs["shared_compressed_base"] == 8
