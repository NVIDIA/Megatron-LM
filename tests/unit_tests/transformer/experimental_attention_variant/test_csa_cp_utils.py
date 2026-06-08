# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa_cp_kernels
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    DSV4_CP_PARTITION_CONTIGUOUS,
    DSV4_CP_PARTITION_MODES,
    DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK,
    apply_cp_compressed_rope_fused,
    apply_thd_chunked_cp_rope_fused,
    apply_thd_cp_chunks_rope_fused,
    apply_thd_overlap_transform_fused,
    build_compressor_prep_compact,
    build_cp_flat_idxs,
    build_cp_flat_idxs_for_indexer_loss,
    build_cp_flat_idxs_for_indexer_loss_fused,
    build_cp_flat_idxs_fused,
    build_cp_indexer_topk_inputs,
    build_cp_indexer_topk_inputs_fused,
    build_chunked_compressor_prep_compact_fused,
    build_chunked_cp_flat_idxs_for_indexer_loss_fused,
    build_chunked_cp_flat_idxs_fused,
    build_chunked_rank_major_compressed_metadata_fused,
    build_global_compressed_cu_seqlens,
    build_global_compressed_cu_seqlens_fused,
    build_compressor_prep_compact_fused,
    can_use_csa_cp_fused_kernels,
    chunked_cp_partition,
    compute_chunked_cp_indexer_topk_logical_fused,
    contiguous_cp_partition,
    exchange_chunked_left_boundary_tensor,
    exchange_left_boundary_tensor,
    pack_cp_kv_full,
    pack_cp_kv_full_fused,
    pack_chunked_cp_kv_full_fused,
    pad_indexer_topk_to_fixed_width_fused,
    normalize_dsv4_cp_partition_mode,
)
from megatron.core.transformer.experimental_attention_variant.dsa_kernels import indexer_topk


def _require_cute_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CSA CP CuTe kernels require CUDA.")
    probe = torch.empty(1, device="cuda")
    if not can_use_csa_cp_fused_kernels(probe):
        pytest.skip("CSA CP CuTe kernels are not available in this environment.")


def _rope_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    nope_dim: int,
    pos_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    x_nope, x_pos = torch.split(x, [nope_dim, pos_dim], dim=-1)
    cos_flat = cos.reshape(cos.shape[0], -1).index_select(0, positions.to(torch.long))
    sin_flat = sin.reshape(sin.shape[0], -1).index_select(0, positions.to(torch.long))
    view_shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (pos_dim,)
    cos_pos = cos_flat[:, :pos_dim].view(view_shape)
    sin_pos = sin_flat[:, :pos_dim].view(view_shape)
    if inverse:
        sin_pos = -sin_pos

    half = pos_dim // 2
    x1 = x_pos[..., 0::2]
    x2 = x_pos[..., 1::2]
    left = x1 * cos_pos[..., :half] - x2 * sin_pos[..., :half]
    right = x2 * cos_pos[..., half:] + x1 * sin_pos[..., half:]
    x_rot = torch.stack((left, right), dim=-1).flatten(-2)
    return torch.cat((x_nope, x_rot), dim=-1)


def _seq_positions_from_global_rows(
    cu_seqlens: torch.Tensor,
    global_row_base: int,
    rows: int,
    padded_total_tokens: int,
    clamp_to_valid_token: bool = False,
) -> torch.Tensor:
    global_rows = torch.arange(global_row_base, global_row_base + rows, dtype=torch.long)
    if clamp_to_valid_token:
        global_rows = global_rows.clamp(min=0, max=padded_total_tokens - 1)
    seq_ids = torch.searchsorted(cu_seqlens.to(torch.long), global_rows, right=True) - 1
    seq_ids = seq_ids.clamp(min=0, max=cu_seqlens.numel() - 2)
    return global_rows - cu_seqlens.to(torch.long)[seq_ids]


def _overlap_transform_thd_reference(
    tensor: torch.Tensor,
    is_first_in_seg: torch.Tensor,
    head_dim: int,
    fill_value: float,
) -> torch.Tensor:
    n_groups, ratio, b_dim, _ = tensor.shape
    out = tensor.new_full((n_groups, 2 * ratio, b_dim, head_dim), fill_value)
    out[:, ratio:] = tensor[:, :, :, head_dim : 2 * head_dim]
    prev = torch.roll(tensor[:, :, :, :head_dim], shifts=1, dims=0)
    prev[is_first_in_seg] = fill_value
    out[:, :ratio] = prev
    return out


def test_dsv4_cp_partition_mode_contract():
    """Validate the public DSv4 CP partition-mode names.

    Expected: omitted mode defaults to the original contiguous partition,
    supported strings round-trip exactly, and unknown values fail before any CP
    layout helper can silently choose the wrong row order.
    """
    assert DSV4_CP_PARTITION_MODES == (
        DSV4_CP_PARTITION_CONTIGUOUS,
        DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK,
    )
    assert normalize_dsv4_cp_partition_mode(None) == DSV4_CP_PARTITION_CONTIGUOUS
    assert (
        normalize_dsv4_cp_partition_mode(DSV4_CP_PARTITION_CONTIGUOUS)
        == DSV4_CP_PARTITION_CONTIGUOUS
    )
    assert (
        normalize_dsv4_cp_partition_mode(DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK)
        == DSV4_CP_PARTITION_PACKED_STREAM_TWO_CHUNK
    )

    with pytest.raises(RuntimeError, match="Unsupported DSv4 CP partition mode"):
        normalize_dsv4_cp_partition_mode("chunked")


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


def test_chunked_cp_partition_matches_packed_stream_two_chunk_order():
    """Validate the packed-stream two-chunk CP partition contract.

    Expected: rank r owns global chunk r followed by chunk 2*cp_size-1-r.
    A failure here means DSv4 chunk-aware CP helpers would disagree with the
    packed-stream two-chunk local row order.
    """
    assert chunked_cp_partition(16, cp_size=1, cp_rank=0) == ((0, 16),)
    assert chunked_cp_partition(16, cp_size=4, cp_rank=0) == ((0, 2), (14, 16))
    assert chunked_cp_partition(16, cp_size=4, cp_rank=1) == ((2, 4), (12, 14))
    assert chunked_cp_partition(16, cp_size=4, cp_rank=2) == ((4, 6), (10, 12))
    assert chunked_cp_partition(16, cp_size=4, cp_rank=3) == ((6, 8), (8, 10))

    with pytest.raises(RuntimeError, match=r"padded_total_tokens % \(2 \* cp_size\)"):
        chunked_cp_partition(18, cp_size=4, cp_rank=0)


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


def test_chunked_boundary_exchange_single_rank_returns_zero_boundary_and_zero_grad():
    """Validate the no-CP two-chunk boundary exchange contract.

    Expected: with cp_group=None, the helper degenerates to one zero-filled
    boundary and contributes no local gradient. A failure here means chunk-aware
    CP boundary plumbing could perturb disabled-CP paths.
    """
    local = torch.arange(12, dtype=torch.float32).reshape(4, 3).requires_grad_(True)

    boundary = exchange_chunked_left_boundary_tensor(local, d_window=2, cp_group=None)

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

    Expected: the helper keeps fixed L_local Q row capacity and builds a
    compressed-KV prefix for the same sequence, producing different fixed
    Q/KV cu_seqlens. A failure here means the THD top-k kernel cannot receive
    the local trapezoid mask shape needed after CP partitioning.
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
    assert torch.equal(k_topk[:6], k_indexer_seq_major[:6])
    assert torch.equal(k_topk[6:], torch.zeros_like(k_topk[6:]))
    assert torch.equal(cu_q_topk, torch.tensor([0, 8, 8], dtype=torch.int32))
    assert torch.equal(cu_k_topk, torch.tensor([0, 6, 6], dtype=torch.int32))
    assert max_q == 8
    assert max_k == 6
    assert torch.equal(local_row_ids, torch.arange(8, dtype=torch.long))


def test_build_cp_indexer_topk_inputs_keeps_tail_padding_rows():
    """Validate fixed-capacity top-k input construction with padded tail rows.

    Expected: tail padding rows stay in Q/weights with a zero-length K segment,
    so downstream top-k can return invalid ids without changing tensor shapes.
    A failure here means the CP indexer path could reintroduce rank-dependent
    shapes when only the last sequence has padding.
    """
    ratio = 4
    cu_seqlens_q = torch.tensor([0, 6], dtype=torch.int32)
    cu_seqlens_compressed = torch.tensor([0, 1], dtype=torch.int32)
    q_indexer_local = torch.arange(16, dtype=torch.float32).reshape(8, 1, 2)
    weights_indexer_local = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    k_indexer_seq_major = torch.arange(8, dtype=torch.float32).reshape(4, 2)

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
        global_start=0,
        l_local=8,
        ratio=ratio,
    )

    assert torch.equal(q_topk, q_indexer_local)
    assert torch.equal(weights_topk, weights_indexer_local)
    assert torch.equal(k_topk[:1], k_indexer_seq_major[:1])
    assert torch.equal(k_topk[1:], torch.zeros_like(k_topk[1:]))
    assert torch.equal(cu_q_topk, torch.tensor([0, 6, 8], dtype=torch.int32))
    assert torch.equal(cu_k_topk, torch.tensor([0, 1, 1], dtype=torch.int32))
    assert max_q == 6
    assert max_k == 1
    assert torch.equal(local_row_ids, torch.arange(8, dtype=torch.long))


def test_cute_chunked_cp_indexer_topk_matches_explicit_chunk_calls():
    """Validate the two-chunk CP indexer-topk scheduling contract.

    Expected: the chunked helper is exactly equivalent to calling the current
    THD indexer forward/top-k once per contiguous global chunk and concatenating
    the resulting logical compressed ids in local row order. A failure here means
    Packed-stream two-chunk CP could accidentally feed a mixed row space to a kernel that
    can only build one contiguous causal/trapezoid mask per call.
    """
    _require_cute_cuda()
    torch.manual_seed(1234)
    ratio = 4
    topk_width = 3
    chunk_ranges = chunked_cp_partition(32, cp_size=4, cp_rank=1)
    l_local = sum(end - start for start, end in chunk_ranges)
    cu_seqlens_q = torch.tensor([0, 32], dtype=torch.int32, device="cuda")
    cu_seqlens_compressed = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    q_indexer_local = torch.randn(l_local, 2, 8, dtype=torch.bfloat16, device="cuda")
    weights_indexer_local = torch.randn(l_local, 2, dtype=torch.bfloat16, device="cuda")
    k_indexer_seq_major = torch.randn(8, 8, dtype=torch.bfloat16, device="cuda")

    actual = compute_chunked_cp_indexer_topk_logical_fused(
        q_indexer_local,
        weights_indexer_local,
        k_indexer_seq_major,
        cu_seqlens_q,
        cu_seqlens_compressed,
        chunk_ranges,
        ratio,
        topk_width,
        indexer_softmax_scale=0.125,
        max_seqlen_q=32,
        max_seqlen_kv=8,
    )

    expected_chunks = []
    local_offset = 0
    for global_start, global_end in chunk_ranges:
        l_chunk = global_end - global_start
        (
            q_for_topk,
            k_for_topk,
            weights_for_topk,
            cu_q_topk,
            cu_k_topk,
            max_q_topk,
            max_k_topk,
            seq_lens_topk,
        ) = build_cp_indexer_topk_inputs_fused(
            q_indexer_local.narrow(0, local_offset, l_chunk),
            weights_indexer_local.narrow(0, local_offset, l_chunk),
            k_indexer_seq_major,
            cu_seqlens_q,
            cu_seqlens_compressed,
            global_start,
            l_chunk,
            ratio,
            max_seqlen_q=32,
            max_seqlen_kv=8,
        )
        chunk_topk, _ = indexer_topk(
            q_for_topk,
            k_for_topk,
            weights_for_topk,
            topk=min(topk_width, max_k_topk),
            ratio=ratio,
            indexer_softmax_scale=0.125,
            cu_seqlens_q=cu_q_topk,
            cu_seqlens_kv=cu_k_topk,
            max_seqlen_q=max_q_topk,
            max_seqlen_kv=max_k_topk,
            fixed_topk_width=topk_width,
            compute_topk_length=False,
            precomputed_seq_lens=seq_lens_topk,
        )
        expected_chunks.append(chunk_topk)
        local_offset += l_chunk
    expected = torch.cat(expected_chunks, dim=0)

    assert torch.equal(actual.cpu(), expected.cpu())


def test_cute_chunked_compressor_prep_matches_explicit_chunk_calls():
    """Validate chunked compressor-prep compact against per-chunk fused calls.

    Expected: local rows ordered as packed-stream two-chunk CP produce the same compact
    rows, compact cu_seqlens, compressed ids, and backward gradients as explicit
    contiguous calls on each chunk followed by GPU-side concatenation. A failure
    here means the production compressor input would not match the local row
    order used by the layer.
    """
    _require_cute_cuda()
    torch.manual_seed(2468)
    cu_seqlens = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
    chunk_ranges = chunked_cp_partition(64, cp_size=4, cp_rank=1)
    ratio = 4
    d_comp = 8
    d_window = 8
    hidden_size = 4

    hidden_local = torch.randn(16, hidden_size, dtype=torch.bfloat16, device="cuda").requires_grad_(
        True
    )
    boundary_hidden = torch.randn(
        2 * d_window, hidden_size, dtype=torch.bfloat16, device="cuda"
    ).requires_grad_(True)
    actual = build_chunked_compressor_prep_compact_fused(
        hidden_local,
        boundary_hidden,
        cu_seqlens,
        chunk_ranges,
        ratio,
        d_comp,
        d_window,
    )

    compact_parts = []
    cu_parts = []
    seq_parts = []
    comp_parts = []
    valid_parts = []
    c_caps = []
    local_offset = 0
    for chunk_id, (global_start, global_end) in enumerate(chunk_ranges):
        l_chunk = global_end - global_start
        part = build_compressor_prep_compact_fused(
            hidden_local.narrow(0, local_offset, l_chunk),
            boundary_hidden.narrow(0, chunk_id * d_window, d_window),
            cu_seqlens,
            global_start,
            l_chunk,
            ratio,
            d_comp,
            d_window,
        )
        compact, cu_compact, seq_ids, comp_ids, valid, c_cap = part
        compact_parts.append(compact)
        cu_parts.append(cu_compact)
        seq_parts.append(seq_ids)
        comp_parts.append(comp_ids)
        valid_parts.append(valid)
        c_caps.append(c_cap)
        local_offset += l_chunk

    cu_deltas = torch.stack([cu[1:] - cu[:-1] for cu in cu_parts], dim=0).sum(dim=0)
    expected_cu = torch.cat(
        (torch.zeros((1,), dtype=cu_seqlens.dtype, device="cuda"), torch.cumsum(cu_deltas, dim=0))
    )
    expected = (
        torch.cat(compact_parts, dim=0),
        expected_cu,
        torch.cat(seq_parts, dim=0),
        torch.cat(comp_parts, dim=0),
        torch.cat(valid_parts, dim=0),
        sum(c_caps),
        c_caps[0],
    )

    for actual_tensor, expected_tensor in zip(actual[:5], expected[:5]):
        assert torch.equal(actual_tensor.cpu(), expected_tensor.cpu())
    assert actual[5:] == expected[5:]

    grad = torch.randn_like(actual[0])
    actual[0].backward(grad, retain_graph=True)
    actual_hidden_grad = hidden_local.grad.detach().clone()
    actual_boundary_grad = boundary_hidden.grad.detach().clone()
    hidden_local.grad.zero_()
    boundary_hidden.grad.zero_()
    expected[0].backward(grad)
    assert torch.equal(hidden_local.grad.cpu(), actual_hidden_grad.cpu())
    assert torch.equal(boundary_hidden.grad.cpu(), actual_boundary_grad.cpu())


def test_cute_chunked_rank_major_metadata_matches_explicit_chunk_metadata():
    """Validate rank-major compressed metadata for the two-chunk CP order.

    Expected: the metadata kernel reconstructs each rank's two global chunks and
    emits rows in the same rank-major order as the compressed KV all-gather. A
    failure here means later seq-major repack or KV full packing could read the
    wrong compressed rows without any communication mismatch.
    """
    _require_cute_cuda()
    cu_seqlens = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
    cp_size = 4
    chunk_len = 8
    ratio = 4
    d_comp = 8
    d_window = 8
    c_cap_per_chunk = (chunk_len + d_comp) // ratio

    actual = build_chunked_rank_major_compressed_metadata_fused(
        cu_seqlens,
        cp_size,
        chunk_len,
        ratio,
        d_comp,
        c_cap_per_chunk,
    )

    seq_parts = []
    comp_parts = []
    valid_parts = []
    dummy_hidden = torch.zeros(chunk_len, 4, dtype=torch.bfloat16, device="cuda")
    dummy_boundary = torch.zeros(d_window, 4, dtype=torch.bfloat16, device="cuda")
    for cp_rank in range(cp_size):
        for global_start, global_end in chunked_cp_partition(64, cp_size, cp_rank):
            _compact, _cu, seq_ids, comp_ids, valid, c_cap = build_compressor_prep_compact_fused(
                dummy_hidden,
                dummy_boundary,
                cu_seqlens,
                global_start,
                global_end - global_start,
                ratio,
                d_comp,
                d_window,
            )
            assert c_cap == c_cap_per_chunk
            seq_parts.append(seq_ids)
            comp_parts.append(comp_ids)
            valid_parts.append(valid)

    expected = (torch.cat(seq_parts), torch.cat(comp_parts), torch.cat(valid_parts))
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.equal(actual_tensor.cpu(), expected_tensor.cpu())


def test_cute_chunked_kv_pack_and_final_idx_use_one_sparse_row_space():
    """Validate chunked KV packing and final idx lowering for one sparse-attn call.

    Expected: the helper packs per-chunk window rows, stores shared compressed
    rows once, and lowers every compressed top-k id into that shared section.
    The indexer-loss rank-major ids still point to the separate all-gathered
    compressed-K tensor. A failure here means one sparse-attention call would
    index a different row space from the packed KV tensor.
    """
    _require_cute_cuda()
    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    cu_seqlens_compressed = torch.tensor([0, 4], dtype=torch.int32, device="cuda")
    chunk_ranges = ((0, 4), (12, 16))
    d_window = 2
    window_size = 2
    ratio = 4

    kv_local = torch.arange(8 * 4, dtype=torch.float32, device="cuda").reshape(8, 4)
    boundary_kv = (100 + torch.arange(4 * 4, dtype=torch.float32, device="cuda")).reshape(4, 4)
    compressed_rank_major = (
        900 + torch.arange(4 * 4, dtype=torch.float32, device="cuda")
    ).reshape(4, 4)
    seq_ids_rank_major = torch.zeros((4,), dtype=torch.int32, device="cuda")
    comp_ids_rank_major = torch.arange(4, dtype=torch.int32, device="cuda")
    valid_rank_major = torch.ones((4,), dtype=torch.bool, device="cuda")
    rank_major_by_seq_major = torch.arange(4, dtype=torch.int32, device="cuda")
    indexer_logical = torch.tensor(
        [[0, -1], [0, -1], [0, -1], [0, -1], [2, 3], [2, 3], [2, 3], [2, 3]],
        dtype=torch.int32,
        device="cuda",
    )

    kv_full, offsets = pack_chunked_cp_kv_full_fused(
        kv_local,
        boundary_kv,
        compressed_rank_major,
        seq_ids_rank_major,
        comp_ids_rank_major,
        valid_rank_major,
        cu_seqlens,
        chunk_ranges,
        d_window,
        ratio,
        rank_major_by_seq_major=rank_major_by_seq_major,
        cu_seqlens_compressed=cu_seqlens_compressed,
    )

    empty_compressed = compressed_rank_major[:0]
    empty_seq_ids = seq_ids_rank_major[:0]
    empty_comp_ids = comp_ids_rank_major[:0]
    empty_valid = valid_rank_major[:0]
    first_window = pack_cp_kv_full_fused(
        kv_local[:4],
        boundary_kv[:2],
        empty_compressed,
        empty_seq_ids,
        empty_comp_ids,
        empty_valid,
        cu_seqlens,
        0,
        4,
        d_window,
        ratio,
    )
    second_window = pack_cp_kv_full_fused(
        kv_local[4:],
        boundary_kv[2:],
        empty_compressed,
        empty_seq_ids,
        empty_comp_ids,
        empty_valid,
        cu_seqlens,
        12,
        4,
        d_window,
        ratio,
    )
    shared_compressed_base = first_window.shape[0] + second_window.shape[0]
    expected_kv_full = torch.cat([first_window, second_window, compressed_rank_major], dim=0)
    assert offsets == (0, first_window.shape[0])
    assert shared_compressed_base == kv_full.shape[0] - compressed_rank_major.shape[0]
    assert torch.equal(kv_full.cpu(), expected_kv_full.cpu())

    topk_idxs, topk_length = build_chunked_cp_flat_idxs_fused(
        cu_seqlens,
        chunk_ranges,
        offsets,
        d_window,
        window_size,
        ratio,
        indexer_logical,
        rank_major_by_seq_major=rank_major_by_seq_major,
        cu_seqlens_compressed=cu_seqlens_compressed,
        shared_compressed_base=shared_compressed_base,
    )
    expected_topk = torch.tensor(
        [
            [0, shared_compressed_base + 0, -1, -1],
            [0, 1, shared_compressed_base + 0, -1],
            [1, 2, shared_compressed_base + 0, -1],
            [2, 3, shared_compressed_base + 0, -1],
            [7, 8, shared_compressed_base + 2, shared_compressed_base + 3],
            [8, 9, shared_compressed_base + 2, shared_compressed_base + 3],
            [9, 10, shared_compressed_base + 2, shared_compressed_base + 3],
            [10, 11, shared_compressed_base + 2, shared_compressed_base + 3],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_length = torch.tensor([2, 3, 3, 3, 4, 4, 4, 4], dtype=torch.int32, device="cuda")
    assert torch.equal(topk_idxs.cpu(), expected_topk.cpu())
    assert torch.equal(topk_length.cpu(), expected_length.cpu())

    loss_topk, loss_rank_major = build_chunked_cp_flat_idxs_for_indexer_loss_fused(
        cu_seqlens,
        cu_seqlens_compressed,
        chunk_ranges,
        offsets,
        d_window,
        window_size,
        ratio,
        indexer_logical,
        rank_major_by_seq_major,
        shared_compressed_base=shared_compressed_base,
    )
    expected_loss_topk = torch.tensor(
        [
            [shared_compressed_base + 0, -1, 0, -1],
            [shared_compressed_base + 0, -1, 0, 1],
            [shared_compressed_base + 0, -1, 1, 2],
            [shared_compressed_base + 0, -1, 2, 3],
            [shared_compressed_base + 2, shared_compressed_base + 3, 7, 8],
            [shared_compressed_base + 2, shared_compressed_base + 3, 8, 9],
            [shared_compressed_base + 2, shared_compressed_base + 3, 9, 10],
            [shared_compressed_base + 2, shared_compressed_base + 3, 10, 11],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_loss_rank_major = torch.tensor(
        [[0, -1], [0, -1], [0, -1], [0, -1], [2, 3], [2, 3], [2, 3], [2, 3]],
        dtype=torch.int32,
        device="cuda",
    )
    assert torch.equal(loss_topk.cpu(), expected_loss_topk.cpu())
    assert torch.equal(loss_rank_major.cpu(), expected_loss_rank_major.cpu())


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


def test_cute_global_compressed_cu_seqlens_matches_reference():
    """Validate fused compressed-prefix metadata for ragged padded sequences.

    Expected: each sequence contributes ``padded_seq_len // ratio`` compressed
    rows to the global seq-major prefix. A failure here means rank-major
    compressed rows could be repacked with the wrong sequence offsets.
    """
    _require_cute_cuda()
    cu_seqlens_cpu = torch.tensor([0, 7, 128, 381, 512], dtype=torch.int32)
    ratio = 4

    ref = build_global_compressed_cu_seqlens(cu_seqlens_cpu, ratio)
    fused = build_global_compressed_cu_seqlens_fused(cu_seqlens_cpu.cuda(), ratio)

    assert torch.equal(fused.cpu(), ref)


def test_cute_thd_overlap_transform_matches_reference_forward_and_backward():
    """Validate fused THD compressor overlap transform.

    Expected: fused output copies the current group's second half and the
    previous group's first half with segment-boundary fill, and backward routes
    gradients to the same input positions. A failure here means the compressor
    could pool the wrong overlapped tokens or lose gradients at segment starts.
    """
    _require_cute_cuda()
    torch.manual_seed(3456)
    is_first = torch.tensor([True, False, True, False, False], dtype=torch.bool)
    head_dim = 3

    for fill_value in (0.0, float("-inf")):
        x_cpu = torch.randn(5, 4, 2, 2 * head_dim, dtype=torch.float32, requires_grad=True)
        ref = _overlap_transform_thd_reference(x_cpu, is_first, head_dim, fill_value)
        grad_cpu = torch.randn_like(ref)
        ref.backward(grad_cpu)

        x_cuda = x_cpu.detach().cuda().requires_grad_(True)
        fused = apply_thd_overlap_transform_fused(
            x_cuda,
            is_first.cuda(),
            head_dim,
            fill_value,
        )
        fused.backward(grad_cpu.cuda())

        torch.testing.assert_close(fused.cpu(), ref)
        torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad)


def test_cute_pad_indexer_topk_to_fixed_width_fills_invalid_tail():
    """Validate fused CP indexer top-k padding to the static production width.

    Expected: existing visible top-k ids are copied unchanged and the remaining
    fixed-width columns are filled with -1, including the zero-visible-K case. A
    failure here means the CP indexer path may reintroduce PyTorch full/cat
    padding or feed non-invalid tail ids into final idx lowering.
    """
    _require_cute_cuda()

    visible = torch.tensor([[3, 1], [-1, 0], [5, -1]], dtype=torch.int32, device="cuda")
    padded = pad_indexer_topk_to_fixed_width_fused(visible, 5)

    expected = torch.tensor(
        [[3, 1, -1, -1, -1], [-1, 0, -1, -1, -1], [5, -1, -1, -1, -1]],
        dtype=torch.int32,
    )
    assert torch.equal(padded.cpu(), expected)

    empty_visible = torch.empty((3, 0), dtype=torch.int32, device="cuda")
    empty_padded = pad_indexer_topk_to_fixed_width_fused(empty_visible, 4)

    assert torch.equal(empty_padded.cpu(), torch.full((3, 4), -1, dtype=torch.int32))


def test_cute_filter_indexer_topk_scores_removes_nonfinite_ids():
    """Validate fused THD indexer top-k post-filtering.

    Expected: ids whose selected score is finite are preserved, while ids that
    are already invalid, out of range, NaN, or -inf are replaced with -1 and the
    per-row valid lengths are counted. A failure here means production top-k may
    either keep causally masked candidates or reintroduce PyTorch gather/where
    filtering in the CP hot path.
    """
    _require_cute_cuda()

    scores = torch.tensor(
        [
            [0.5, float("-inf"), 2.0, float("nan")],
            [float("-inf"), 1.0, 3.0, 4.0],
            [float("nan"), float("-inf"), -2.0, 7.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    topk = torch.tensor(
        [
            [0, 1, 3, -1],
            [2, 0, 5, 1],
            [0, 1, 2, 3],
        ],
        dtype=torch.int32,
        device="cuda",
    )

    fused_topk, fused_length = csa_cp_kernels.filter_indexer_topk_scores(scores, topk)
    fused_padded_topk, fused_padded_length = csa_cp_kernels.filter_indexer_topk_scores(
        scores, topk, output_width=6
    )

    safe_topk = topk.clamp(min=0, max=scores.shape[1] - 1).to(torch.long)
    selected = torch.gather(scores, dim=-1, index=safe_topk)
    valid = (topk >= 0) & (topk < scores.shape[1]) & torch.isfinite(selected)
    expected_topk = torch.where(valid, topk, torch.full_like(topk, -1))
    expected_length = valid.sum(dim=-1).to(torch.int32)
    expected_padded_topk = torch.cat(
        [expected_topk, torch.full((topk.shape[0], 2), -1, dtype=torch.int32, device="cuda")],
        dim=-1,
    )

    assert torch.equal(fused_topk.cpu(), expected_topk.cpu())
    assert torch.equal(fused_length.cpu(), expected_length.cpu())
    assert torch.equal(fused_padded_topk.cpu(), expected_padded_topk.cpu())
    assert torch.equal(fused_padded_length.cpu(), expected_length.cpu())


def test_cute_thd_chunked_cp_rope_matches_reference_forward_and_backward():
    """Validate THD CP RoPE position reconstruction in kernel.

    Expected: fused THD CP RoPE maps ``global_row_base + row`` through
    ``cu_seqlens_padded`` to the same sequence-local positions as the PyTorch
    reference, and backward applies the inverse rotation. A failure here means
    local Q/K, boundary K, or inverse output RoPE would use wrong CP positions.
    """
    _require_cute_cuda()
    torch.manual_seed(1234)
    cu_seqlens = torch.tensor([0, 5, 11, 16], dtype=torch.int32)
    rows = 4
    global_row_base = 8
    padded_total_tokens = 16
    nope_dim = 4
    pos_dim = 4

    x_cpu = torch.randn(rows, 2, nope_dim + pos_dim, dtype=torch.float32, requires_grad=True)
    cos_cpu = torch.randn(16, 1, 1, pos_dim, dtype=torch.float32)
    sin_cpu = torch.randn(16, 1, 1, pos_dim, dtype=torch.float32)
    positions = _seq_positions_from_global_rows(
        cu_seqlens, global_row_base, rows, padded_total_tokens
    )
    ref = _rope_reference(x_cpu, cos_cpu, sin_cpu, positions, nope_dim, pos_dim)
    grad_cpu = torch.randn_like(ref)
    ref.backward(grad_cpu)

    x_cuda = x_cpu.detach().cuda().requires_grad_(True)
    fused = apply_thd_chunked_cp_rope_fused(
        x_cuda,
        cos_cpu.cuda(),
        sin_cpu.cuda(),
        nope_dim,
        pos_dim,
        cu_seqlens.cuda(),
        cp_rank=2,
        cp_size=4,
    )
    fused.backward(grad_cpu.cuda())

    torch.testing.assert_close(fused.cpu(), ref)
    torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad)

    x_cuda_explicit = x_cpu.detach().cuda().requires_grad_(True)
    fused_explicit = apply_thd_chunked_cp_rope_fused(
        x_cuda_explicit,
        cos_cpu.cuda(),
        sin_cpu.cuda(),
        nope_dim,
        pos_dim,
        cu_seqlens.cuda(),
        cp_rank=0,
        cp_size=4,
        global_row_base=global_row_base,
    )
    fused_explicit.backward(grad_cpu.cuda())

    torch.testing.assert_close(fused_explicit.cpu(), ref)
    torch.testing.assert_close(x_cuda_explicit.grad.cpu(), x_cpu.grad)


def test_cute_thd_cp_chunks_rope_matches_reference_forward_and_backward():
    """Validate the two-chunk THD CP RoPE fast path.

    Expected: local rows are split by ``chunk_ranges`` and each row uses the
    sequence-local RoPE position corresponding to its chunk's global token. A
    failure here means packed-stream two-chunk CP would rotate Q/K or output gradients
    with positions from the wrong half of the global token stream.
    """
    _require_cute_cuda()
    torch.manual_seed(4321)
    cu_seqlens = torch.tensor([0, 7, 18, 32], dtype=torch.int32)
    chunk_ranges = ((3, 9), (22, 28))
    rows = sum(end - start for start, end in chunk_ranges)
    padded_total_tokens = 32
    nope_dim = 4
    pos_dim = 4

    x_cpu = torch.randn(rows, 2, nope_dim + pos_dim, dtype=torch.float32, requires_grad=True)
    cos_cpu = torch.randn(32, 1, 1, pos_dim, dtype=torch.float32)
    sin_cpu = torch.randn(32, 1, 1, pos_dim, dtype=torch.float32)
    positions = torch.cat(
        [
            _seq_positions_from_global_rows(
                cu_seqlens,
                start,
                end - start,
                padded_total_tokens,
            )
            for start, end in chunk_ranges
        ],
        dim=0,
    )
    ref = _rope_reference(x_cpu, cos_cpu, sin_cpu, positions, nope_dim, pos_dim)
    grad_cpu = torch.randn_like(ref)
    ref.backward(grad_cpu)

    x_cuda = x_cpu.detach().cuda().requires_grad_(True)
    fused = apply_thd_cp_chunks_rope_fused(
        x_cuda,
        cos_cpu.cuda(),
        sin_cpu.cuda(),
        nope_dim,
        pos_dim,
        cu_seqlens.cuda(),
        chunk_ranges,
    )
    fused.backward(grad_cpu.cuda())

    torch.testing.assert_close(fused.cpu(), ref)
    torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad)


def test_cute_boundary_rope_clamps_global_rows_before_reference_lookup():
    """Validate rank-0 left-boundary RoPE clamp semantics.

    Expected: negative boundary global rows clamp to the first valid padded
    token before sequence-local position lookup. A failure here means rank 0
    boundary K could read invalid RoPE coordinates.
    """
    _require_cute_cuda()
    torch.manual_seed(5678)
    cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)
    rows = 3
    global_row_base = -2
    padded_total_tokens = 12
    nope_dim = 2
    pos_dim = 4

    x_cpu = torch.randn(rows, 1, nope_dim + pos_dim, dtype=torch.float32, requires_grad=True)
    cos_cpu = torch.randn(12, 1, 1, pos_dim, dtype=torch.float32)
    sin_cpu = torch.randn(12, 1, 1, pos_dim, dtype=torch.float32)
    positions = _seq_positions_from_global_rows(
        cu_seqlens, global_row_base, rows, padded_total_tokens, clamp_to_valid_token=True
    )
    ref = _rope_reference(x_cpu, cos_cpu, sin_cpu, positions, nope_dim, pos_dim)
    grad_cpu = torch.randn_like(ref)
    ref.backward(grad_cpu)

    x_cuda = x_cpu.detach().cuda().requires_grad_(True)
    fused = apply_thd_chunked_cp_rope_fused(
        x_cuda,
        cos_cpu.cuda(),
        sin_cpu.cuda(),
        nope_dim,
        pos_dim,
        cu_seqlens.cuda(),
        cp_rank=0,
        cp_size=4,
        row_offset=global_row_base,
        clamp_to_valid_token=True,
    )
    fused.backward(grad_cpu.cuda())

    torch.testing.assert_close(fused.cpu(), ref)
    torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad)


def test_cute_compressed_rope_uses_compressed_group_ids_forward_and_backward():
    """Validate compressed-row RoPE position reconstruction in kernel.

    Expected: fused compressed RoPE uses ``max(comp_id, 0) * ratio`` for each
    row and backward applies the inverse rotation. A failure here means
    compressor output K would use wrong positions after compacting.
    """
    _require_cute_cuda()
    torch.manual_seed(9012)
    comp_ids = torch.tensor([0, 2, -1, 3], dtype=torch.int32)
    ratio = 4
    nope_dim = 3
    pos_dim = 4

    x_cpu = torch.randn(4, 1, 2, nope_dim + pos_dim, dtype=torch.float32, requires_grad=True)
    cos_cpu = torch.randn(16, 1, 1, pos_dim, dtype=torch.float32)
    sin_cpu = torch.randn(16, 1, 1, pos_dim, dtype=torch.float32)
    positions = comp_ids.clamp(min=0).to(torch.long) * ratio
    ref = _rope_reference(x_cpu, cos_cpu, sin_cpu, positions, nope_dim, pos_dim)
    grad_cpu = torch.randn_like(ref)
    ref.backward(grad_cpu)

    x_cuda = x_cpu.detach().cuda().requires_grad_(True)
    fused = apply_cp_compressed_rope_fused(
        x_cuda,
        cos_cpu.cuda(),
        sin_cpu.cuda(),
        comp_ids.cuda(),
        ratio,
        nope_dim,
        pos_dim,
    )
    fused.backward(grad_cpu.cuda())

    torch.testing.assert_close(fused.cpu(), ref)
    torch.testing.assert_close(x_cuda.grad.cpu(), x_cpu.grad)


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
