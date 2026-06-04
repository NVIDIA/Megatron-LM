# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    build_compressor_prep_compact,
    build_cp_flat_idxs,
    build_cp_flat_idxs_for_indexer_loss,
    build_cp_flat_idxs_for_indexer_loss_fused,
    build_cp_flat_idxs_fused,
    build_cp_indexer_topk_inputs,
    can_use_csa_cp_fused_kernels,
    contiguous_cp_partition,
    exchange_left_boundary_tensor,
    pack_cp_kv_full,
    pack_cp_kv_full_fused,
)


def _require_cute_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CSA CP CuTe kernels require CUDA.")
    probe = torch.empty(1, device="cuda")
    if not can_use_csa_cp_fused_kernels(probe):
        pytest.skip("CSA CP CuTe kernels are not available in this environment.")


def test_contiguous_cp_partition_requires_padded_total_divisible_by_cp_size():
    """Validate the fixed-size padded contiguous CP partition contract.

    Expected: a single CP rank owns the full padded range, while a padded token
    count that is not divisible by cp_size raises before CP layout construction builds
    local layouts. A failure here means CP could silently proceed with uneven
    rank padded-token ranges.
    """
    cu_seqlens_padded = torch.tensor([0, 7], dtype=torch.int32)

    assert contiguous_cp_partition(cu_seqlens_padded, cp_size=1, cp_rank=0) == (0, 7)

    with pytest.raises(RuntimeError, match="padded_total_tokens % cp_size"):
        contiguous_cp_partition(cu_seqlens_padded, cp_size=2, cp_rank=0)


def test_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP boundary exchange contract.

    Expected: with cp_group=None, the fixed left boundary is zero-filled and its
    backward path contributes no gradient to the local tensor. A failure here
    means the CP path could invent boundary tokens or bogus local gradients.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_left_boundary_tensor(local, d_window=2, cp_group=None)

    assert torch.equal(boundary, torch.zeros(2, 3))
    boundary.sum().backward()
    assert torch.equal(local.grad, torch.zeros_like(local))


def test_compressor_prep_compact_uses_only_complete_boundary_groups():
    """Validate compressor-prep compacting across a sequence boundary.

    Expected: for ratio=128, only the complete visible group [320, 448) is
    compacted; the partial boundary group [192, 320) is skipped and unused
    fixed capacity remains padded. A failure here means compressor prep may read
    outside the legal boundary window or emit wrong compressed metadata.
    """
    cu_seqlens = torch.tensor([0, 192, 512], dtype=torch.int32)
    global_start = 384
    l_local = 128
    d_window = 128
    ratio = 128
    d_comp = 128
    hidden_size = 2

    local_positions = torch.arange(global_start, global_start + l_local, dtype=torch.float32)
    hidden_local = local_positions[:, None].repeat(1, hidden_size)
    boundary_positions = torch.arange(global_start - d_window, global_start, dtype=torch.float32)
    boundary_hidden = boundary_positions[:, None].repeat(1, hidden_size)

    hidden_compact, cu_compact, seq_ids, comp_ids, valid, c_cap = build_compressor_prep_compact(
        hidden_local,
        boundary_hidden,
        cu_seqlens,
        global_start,
        l_local,
        ratio,
        d_comp,
        d_window,
    )

    expected_positions = torch.arange(320, 448, dtype=torch.float32)[:, None].repeat(
        1, hidden_size
    )
    assert c_cap == 2
    assert torch.equal(cu_compact.cpu(), torch.tensor([0, 0, 128], dtype=torch.int32))
    assert torch.equal(hidden_compact[:128].cpu(), expected_positions)
    assert torch.equal(hidden_compact[128:].cpu(), torch.zeros_like(hidden_compact[128:]).cpu())
    assert torch.equal(seq_ids.cpu(), torch.tensor([1, -1], dtype=torch.int32))
    assert torch.equal(comp_ids.cpu(), torch.tensor([1, -1], dtype=torch.int32))
    assert torch.equal(valid.cpu(), torch.tensor([True, False]))


def test_kv_full_pack_keeps_per_sequence_window_then_compressed_layout():
    """Validate the post-all-gather KV full packing order.

    Expected: each active sequence writes its local/boundary window first, then
    valid compressed entries, while invalid compressed entries are skipped and
    tail capacity stays zero. A failure here means final lowered indices could
    point at the wrong KV rows.
    """
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    global_start = 3
    l_local = 3
    d_window = 2

    kv_local = torch.tensor([[103.0], [104.0], [105.0]])
    boundary_kv = torch.tensor([[101.0], [102.0]])
    compressed_rank_major = torch.tensor([[900.0], [901.0], [902.0]])
    seq_ids_rank_major = torch.tensor([0, 1, 1], dtype=torch.int32)
    comp_ids_rank_major = torch.tensor([0, 0, 1], dtype=torch.int32)
    valid_rank_major = torch.tensor([True, True, False])

    kv_full, window_map, compressed_map = pack_cp_kv_full(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        global_start,
        l_local,
        d_window,
    )

    assert window_map == {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    assert compressed_map == {(0, 0): 3, (1, 0): 6}
    expected_prefix = torch.tensor([101.0, 102.0, 103.0, 900.0, 104.0, 105.0, 901.0])
    assert torch.equal(kv_full[:7].squeeze(-1), expected_prefix)
    assert torch.equal(kv_full[7:], torch.zeros_like(kv_full[7:]))


def test_build_cp_indexer_topk_inputs_uses_local_trapezoid_cu_seqlens():
    """Validate CP-local indexer top-k input construction for a split sequence.

    Expected: the helper keeps only this rank's Q rows but builds a longer
    compressed-KV prefix for the same sequence, producing different Q/KV
    cu_seqlens. A failure here means the THD top-k kernel cannot receive the
    local trapezoid mask shape needed after CP partitioning.
    """
    ratio = 4
    cu_seqlens_q = torch.tensor([0, 32], dtype=torch.int32)
    cu_seqlens_compressed = torch.tensor([0, 8], dtype=torch.int32)
    q_indexer_local = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2)
    weights_indexer_local = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    k_indexer_seq_major = torch.arange(16, dtype=torch.float32).reshape(8, 2)

    (
        q_topk,
        k_topk,
        weights_topk,
        cu_q_topk,
        cu_k_topk,
        max_q,
        max_k,
        local_row_ids,
    ) = build_cp_indexer_topk_inputs(
        q_indexer_local,
        weights_indexer_local,
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
        global_start=16,
        l_local=8,
        ratio=ratio,
    )

    assert torch.equal(q_topk, q_indexer_local)
    assert torch.equal(weights_topk, weights_indexer_local)
    assert torch.equal(k_topk, k_indexer_seq_major[:6])
    assert torch.equal(cu_q_topk, torch.tensor([0, 8], dtype=torch.int32))
    assert torch.equal(cu_k_topk, torch.tensor([0, 6], dtype=torch.int32))
    assert max_q == 8
    assert max_k == 6
    assert torch.equal(local_row_ids, torch.arange(8, dtype=torch.long))


def test_build_cp_flat_idxs_respect_sequence_local_compressed_ids():
    """Validate final idx lowering for the normal sparse-attention path.

    Expected: window ids and visible compressed ids lower to kv_full flat ids in
    the same sequence only, and topk_length counts valid entries per row. A
    failure here means sparse attention could attend across sequence boundaries
    or use an incorrect compact length.
    """
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    window_map = {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    compressed_map = {(0, 0): 3, (1, 0): 6}

    topk_idxs, topk_length = build_cp_flat_idxs(
        cu_seqlens,
        global_start=3,
        l_local=3,
        window_size=2,
        ratio=2,
        device=torch.device("cpu"),
        window_map=window_map,
        compressed_map=compressed_map,
        max_n_compressed=2,
    )

    expected = torch.tensor(
        [
            [1, 2, 3, -1],
            [4, -1, -1, -1],
            [4, 5, 6, -1],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(topk_idxs, expected)
    assert torch.equal(topk_length, torch.tensor([3, 1, 3], dtype=torch.int32))


def test_build_cp_flat_idxs_for_indexer_loss_preserves_rank_major_ids():
    """Validate final idx lowering for the sparse indexer-loss path.

    Expected: compressed top-k columns are lowered before window columns, and
    the rank-major compressed ids stay aligned with the selected compressed
    entries. A failure here means the fused sparse-loss kernel could compare
    indexer scores against the wrong compressed KV entries.
    """
    cu_seqlens = torch.tensor([0, 4, 8], dtype=torch.int32)
    window_map = {1: 0, 2: 1, 3: 2, 4: 4, 5: 5}
    compressed_map = {(0, 0): 3, (1, 0): 6}
    logical_ids = torch.tensor([[0, -1], [-1, -1], [0, -1]], dtype=torch.int32)
    rank_major_ids = torch.tensor([[5, -1], [-1, -1], [7, -1]], dtype=torch.int32)

    topk_idxs, lowered_rank_major_ids = build_cp_flat_idxs_for_indexer_loss(
        cu_seqlens,
        global_start=3,
        l_local=3,
        window_size=2,
        device=torch.device("cpu"),
        window_map=window_map,
        compressed_map=compressed_map,
        indexer_topk_compressed_logical_ids=logical_ids,
        indexer_topk_rank_major_ids=rank_major_ids,
    )

    expected = torch.tensor(
        [
            [3, -1, 1, 2],
            [-1, -1, 4, -1],
            [6, -1, 4, 5],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(topk_idxs, expected)
    assert torch.equal(lowered_rank_major_ids, rank_major_ids)


def test_cute_compressor_prep_compact_matches_reference_forward_and_backward():
    """Validate the fused compressor-prep compact kernel against the reference path.

    Expected: fused forward emits the same compact rows, cu_seqlens, metadata,
    and validity as the PyTorch reference; fused backward scatters gradients to
    the same local and boundary rows. A failure here means the replacement
    kernel would feed the compressor different tokens or route gradients to the
    wrong owner.
    """
    _require_cute_cuda()
    cu_seqlens_cpu = torch.tensor([0, 192, 512], dtype=torch.int32)
    global_start = 384
    l_local = 128
    d_window = 128
    ratio = 128
    d_comp = 128
    hidden_size = 3

    local_positions = torch.arange(global_start, global_start + l_local, dtype=torch.float32)
    hidden_local_cpu = (
        local_positions[:, None].repeat(1, hidden_size).to(torch.bfloat16).requires_grad_(True)
    )
    boundary_positions = torch.arange(global_start - d_window, global_start, dtype=torch.float32)
    boundary_hidden_cpu = (
        boundary_positions[:, None].repeat(1, hidden_size).to(torch.bfloat16).requires_grad_(True)
    )
    ref = build_compressor_prep_compact(
        hidden_local_cpu,
        boundary_hidden_cpu,
        cu_seqlens_cpu,
        global_start,
        l_local,
        ratio,
        d_comp,
        d_window,
    )
    ref_hidden, ref_cu, ref_seq, ref_comp, ref_valid, ref_c_cap = ref
    grad_cpu = torch.randn_like(ref_hidden)
    ref_hidden.backward(grad_cpu)

    hidden_local_cuda = hidden_local_cpu.detach().cuda().requires_grad_(True)
    boundary_hidden_cuda = boundary_hidden_cpu.detach().cuda().requires_grad_(True)
    cu_seqlens_cuda = cu_seqlens_cpu.cuda()
    fused = build_compressor_prep_compact(
        hidden_local_cuda,
        boundary_hidden_cuda,
        cu_seqlens_cuda,
        global_start,
        l_local,
        ratio,
        d_comp,
        d_window,
    )
    fused_hidden, fused_cu, fused_seq, fused_comp, fused_valid, fused_c_cap = fused
    fused_hidden.backward(grad_cpu.cuda())

    assert fused_c_cap == ref_c_cap
    torch.testing.assert_close(fused_hidden.cpu(), ref_hidden)
    assert torch.equal(fused_cu.cpu(), ref_cu)
    assert torch.equal(fused_seq.cpu(), ref_seq)
    assert torch.equal(fused_comp.cpu(), ref_comp)
    assert torch.equal(fused_valid.cpu(), ref_valid)
    torch.testing.assert_close(hidden_local_cuda.grad.cpu(), hidden_local_cpu.grad)
    torch.testing.assert_close(boundary_hidden_cuda.grad.cpu(), boundary_hidden_cpu.grad)


def test_cute_kv_pack_and_final_idx_match_reference_layout():
    """Validate fused KV packing and normal final idx lowering together.

    Expected: fused KV full uses the same prefix layout as the reference pack,
    and fused final idx lowers window plus all-compressed ids to the same flat
    rows. A failure here means sparse attention would read different KV rows
    after replacing the Python map-based lowering.
    """
    _require_cute_cuda()
    cu_seqlens_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)
    global_start = 3
    l_local = 3
    d_window = 2
    ratio = 2
    window_size = 2
    max_n_compressed = 2

    kv_local_cpu = torch.tensor(
        [[103.0], [104.0], [105.0]], dtype=torch.bfloat16, requires_grad=True
    )
    boundary_kv_cpu = torch.tensor([[101.0], [102.0]], dtype=torch.bfloat16, requires_grad=True)
    compressed_cpu = torch.tensor(
        [[900.0], [901.0], [902.0], [903.0]], dtype=torch.bfloat16, requires_grad=True
    )
    seq_ids_cpu = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    comp_ids_cpu = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
    valid_cpu = torch.tensor([True, True, True, True])
    ref_kv, ref_window_map, ref_compressed_map = pack_cp_kv_full(
        kv_local_cpu,
        boundary_kv_cpu,
        compressed_cpu,
        seq_ids_cpu,
        comp_ids_cpu,
        valid_cpu,
        cu_seqlens_cpu,
        global_start,
        l_local,
        d_window,
    )
    ref_topk, ref_len = build_cp_flat_idxs(
        cu_seqlens_cpu,
        global_start,
        l_local,
        window_size,
        ratio,
        torch.device("cpu"),
        ref_window_map,
        ref_compressed_map,
        max_n_compressed=max_n_compressed,
    )

    kv_local_cuda = kv_local_cpu.detach().cuda().requires_grad_(True)
    boundary_kv_cuda = boundary_kv_cpu.detach().cuda().requires_grad_(True)
    compressed_cuda = compressed_cpu.detach().cuda().requires_grad_(True)
    fused_kv = pack_cp_kv_full_fused(
        kv_local_cuda,
        boundary_kv_cuda,
        compressed_cuda,
        seq_ids_cpu.cuda(),
        comp_ids_cpu.cuda(),
        valid_cpu.cuda(),
        cu_seqlens_cpu.cuda(),
        global_start,
        l_local,
        d_window,
        ratio,
    )
    fused_topk, fused_len = build_cp_flat_idxs_fused(
        cu_seqlens_cpu.cuda(),
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        max_n_compressed=max_n_compressed,
    )

    torch.testing.assert_close(fused_kv[: ref_kv.shape[0]].cpu(), ref_kv)
    assert torch.equal(fused_topk.cpu(), ref_topk)
    assert torch.equal(fused_len.cpu(), ref_len)

    grad = torch.randn_like(ref_kv)
    ref_kv.backward(grad)
    fused_kv[: ref_kv.shape[0]].backward(grad.cuda())
    torch.testing.assert_close(kv_local_cuda.grad.cpu(), kv_local_cpu.grad)
    torch.testing.assert_close(boundary_kv_cuda.grad.cpu(), boundary_kv_cpu.grad)
    torch.testing.assert_close(compressed_cuda.grad.cpu(), compressed_cpu.grad)


def test_cute_indexer_loss_final_idx_matches_reference_lowering():
    """Validate fused Path-B final idx and rank-major id lowering.

    Expected: compressed top-k ids stay in fixed compressed columns, window ids
    occupy the following columns, and rank-major ids match the selected
    compressed rows. A failure here means sparse indexer loss would compare
    indexer scores with the wrong compressed KV rows.
    """
    _require_cute_cuda()
    cu_seqlens_cpu = torch.tensor([0, 4, 8], dtype=torch.int32)
    global_start = 3
    l_local = 3
    d_window = 2
    window_size = 2
    ratio = 2
    window_map = {1: 0, 2: 1, 3: 2, 4: 5, 5: 6}
    compressed_map = {(0, 0): 3, (0, 1): 4, (1, 0): 7, (1, 1): 8}
    logical_ids = torch.tensor([[0, 1], [-1, -1], [0, 1]], dtype=torch.int32)
    rank_major_ids = torch.tensor([[0, 1], [-1, -1], [2, 3]], dtype=torch.int32)
    ref_topk, ref_rank_major = build_cp_flat_idxs_for_indexer_loss(
        cu_seqlens_cpu,
        global_start,
        l_local,
        window_size,
        torch.device("cpu"),
        window_map,
        compressed_map,
        logical_ids,
        rank_major_ids,
    )

    cu_seqlens_compressed = torch.tensor([0, 2, 4], dtype=torch.int32)
    rank_by_seq_major = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    fused_topk, fused_rank_major = build_cp_flat_idxs_for_indexer_loss_fused(
        cu_seqlens_cpu.cuda(),
        cu_seqlens_compressed.cuda(),
        global_start,
        l_local,
        d_window,
        window_size,
        ratio,
        logical_ids.cuda(),
        rank_by_seq_major.cuda(),
    )

    assert torch.equal(fused_topk.cpu(), ref_topk)
    assert torch.equal(fused_rank_major.cpu(), rank_major_ids)
    assert torch.equal(fused_rank_major.cpu(), ref_rank_major)
