# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import List, Tuple

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    thd_indexer_kernels,
    thd_layout_kernels,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils.cp_utils import (
    prepare_cp_compressor_input,
)

# This file guards only DSv4 THD layout/metadata kernels. Layer-level CUDA graph
# tests guard graph capture/replay behavior.

_E2E_RAGGED_PADDED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 200)
_E2E_CP_SIZE = 4


def _require_cute_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CuTe kernels require CUDA.")
    if not thd_layout_kernels._CUTE_AVAILABLE:
        pytest.skip("DSv4 CP CuTe kernels are not available in this environment.")


def _require_triton_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 THD indexer glue kernels require CUDA.")
    if not thd_indexer_kernels._TRITON_AVAILABLE:
        pytest.skip("DSv4 THD indexer Triton kernels are not available in this environment.")


def _make_e2e_like_cu_seqlens(device: str = "cuda") -> torch.Tensor:
    """Return a representative ragged THD prefix pattern.

    The lengths sum to 4096, so CP4 owns 1024 local rows. They contain
    short, long, boundary-crossing, and padded-tail sequences without allocating
    full e2e hidden sizes in these focused kernel tests.
    """
    return torch.tensor(
        [0] + list(torch.tensor(_E2E_RAGGED_PADDED_SEG_LENS).cumsum(0).tolist()),
        dtype=torch.int32,
        device=device,
    )


def _compressed_cu_seqlens(cu_seqlens: torch.Tensor, ratio: int) -> torch.Tensor:
    """Return floor-compressed sequence prefixes for the supplied THD layout."""
    return torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=cu_seqlens.device),
            torch.cumsum(
                torch.div(cu_seqlens[1:] - cu_seqlens[:-1], int(ratio), rounding_mode="floor"),
                dim=0,
                dtype=torch.int32,
            ),
        )
    )


def _e2e_like_local_range() -> Tuple[int, int]:
    """Return rank 1's CP4 global start and local row count."""
    total = sum(_E2E_RAGGED_PADDED_SEG_LENS)
    local = total // _E2E_CP_SIZE
    return local, local


def _compressed_groups(
    cu_seqlens: torch.Tensor, global_start: int, l_local: int, ratio: int, d_comp: int
) -> List[Tuple[int, int]]:
    cu = [int(x) for x in cu_seqlens.cpu().tolist()]
    groups: List[Tuple[int, int]] = []
    global_end = int(global_start) + int(l_local)
    for seq, (seq_start, seq_end) in enumerate(zip(cu[:-1], cu[1:])):
        local_start = max(seq_start, int(global_start))
        local_end = min(seq_end, global_end)
        if local_start >= local_end:
            continue
        n_full_groups = (seq_end - seq_start) // int(ratio)
        first_numer = max(0, int(global_start) - int(d_comp) - seq_start)
        first_group = (first_numer + int(ratio) - 1) // int(ratio) if first_numer > 0 else 0
        stop_group = min((local_end - seq_start) // int(ratio), n_full_groups)
        for comp_id in range(first_group, max(first_group, stop_group)):
            groups.append((seq, comp_id))
    return groups


def _native_compressor_input_compact(
    hidden_local: torch.Tensor,
    boundary_hidden: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
    d_comp: int,
    d_window: int,
    c_cap: int,
):
    groups = _compressed_groups(cu_seqlens, global_start, l_local, ratio, d_comp)
    cu = [int(x) for x in cu_seqlens.cpu().tolist()]
    pieces = []
    for seq, comp_id in groups:
        seq_start = cu[seq]
        for token in range(ratio):
            src_global = seq_start + comp_id * ratio + token
            if src_global < global_start:
                pieces.append(
                    boundary_hidden[
                        src_global
                        - (global_start - d_window) : src_global
                        - (global_start - d_window)
                        + 1
                    ]
                )
            else:
                pieces.append(
                    hidden_local[src_global - global_start : src_global - global_start + 1]
                )
    compact_len = int(c_cap) * int(ratio)
    if pieces:
        hidden_compact = torch.cat(pieces, dim=0)
    else:
        hidden_compact = hidden_local.new_empty((0,) + tuple(hidden_local.shape[1:]))
    if hidden_compact.shape[0] < compact_len:
        hidden_compact = torch.cat(
            (
                hidden_compact,
                hidden_local.new_zeros(
                    (compact_len - hidden_compact.shape[0],) + tuple(hidden_local.shape[1:])
                ),
            ),
            dim=0,
        )

    comp_ids = torch.full((c_cap,), -1, dtype=torch.int32, device=hidden_local.device)
    for slot, (_, comp_id) in enumerate(groups[:c_cap]):
        comp_ids[slot] = comp_id
    return hidden_compact, comp_ids


def _native_attention_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    seq_to_rank_row: torch.Tensor,
    compressed_base: int,
    indexer_topk: torch.Tensor = None,
):
    """Reference final-index lowering for ragged THD CP rows."""
    device = cu_seqlens.device
    total_width = int(window_size) + int(compressed_width)
    topk = torch.full((int(l_local), total_width), -1, dtype=torch.int32, device=device)
    lengths = torch.zeros((int(l_local),), dtype=torch.int32, device=device)
    cu = [int(x) for x in cu_seqlens.cpu().tolist()]
    cu_comp = [int(x) for x in cu_seqlens_compressed.cpu().tolist()]

    for row in range(int(l_local)):
        global_q = int(global_start) + row
        seq_id = next(
            (
                seq
                for seq, (start, end) in enumerate(zip(cu[:-1], cu[1:]))
                if start <= global_q < end
            ),
            None,
        )
        if seq_id is None:
            if total_width > 0:
                topk[row, 0] = 0
                lengths[row] = 1
            continue
        seq_start = cu[seq_id]
        seq_comp_len = cu_comp[seq_id + 1] - cu_comp[seq_id]
        write_col = 0
        window_start_for_q = max(global_q - int(window_size) + 1, seq_start)
        window_count = max(0, global_q - window_start_for_q + 1)
        for w in range(int(window_size)):
            pos = window_start_for_q + w
            if w < window_count:
                topk[row, write_col] = pos - (int(global_start) - int(d_window))
                write_col += 1
        if int(ratio) > 1 and int(compressed_width) > 0:
            pos_in_seq = global_q - seq_start
            for j in range(int(compressed_width)):
                if indexer_topk is not None:
                    comp_id = int(indexer_topk[row, j])
                else:
                    n_visible = min((pos_in_seq + 1) // int(ratio), int(compressed_width))
                    comp_id = j if j < n_visible else -1
                if 0 <= comp_id < seq_comp_len:
                    rank_id = int(seq_to_rank_row[cu_comp[seq_id] + comp_id])
                    if rank_id >= 0:
                        topk[row, write_col] = int(compressed_base) + rank_id
                        write_col += 1
        lengths[row] = write_col
    return topk, lengths


def _native_indexer_loss_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    logical_ids: torch.Tensor,
    seq_to_rank_row: torch.Tensor,
    compressed_base: int,
):
    """Reference compressed-first indexer-loss lowering for ragged THD rows."""
    device = logical_ids.device
    compressed_width = logical_ids.shape[1]
    total_width = compressed_width + int(window_size)
    topk = torch.full((int(l_local), total_width), -1, dtype=torch.int32, device=device)
    rank_major = torch.full((int(l_local), compressed_width), -1, dtype=torch.int32, device=device)
    cu = [int(x) for x in cu_seqlens.cpu().tolist()]
    cu_comp = [int(x) for x in cu_seqlens_compressed.cpu().tolist()]
    for row in range(int(l_local)):
        global_q = int(global_start) + row
        seq_id = next(
            (
                seq
                for seq, (start, end) in enumerate(zip(cu[:-1], cu[1:]))
                if start <= global_q < end
            ),
            None,
        )
        if seq_id is None:
            continue
        seq_start = cu[seq_id]
        comp_len = cu_comp[seq_id + 1] - cu_comp[seq_id]

        for col in range(compressed_width):
            comp_id = int(logical_ids[row, col])
            if 0 <= comp_id < comp_len:
                seq_major = cu_comp[seq_id] + comp_id
                rank_id = int(seq_to_rank_row[seq_major])
                if rank_id >= 0:
                    topk[row, col] = int(compressed_base) + rank_id
                    rank_major[row, col] = rank_id

        window_start_for_q = max(global_q - int(window_size) + 1, seq_start)
        window_count = global_q - window_start_for_q + 1
        for window_col in range(int(window_size)):
            pos = window_start_for_q + window_col
            if window_col < window_count:
                topk[row, compressed_width + window_col] = pos - (int(global_start) - int(d_window))

    return topk, rank_major


@pytest.mark.parametrize("with_causal_offsets", [False, True])
def test_build_indexer_seq_lens_matches_native(with_causal_offsets):
    _require_triton_cuda()
    cu_q = torch.tensor([0, 5, 5, 9], dtype=torch.int32, device="cuda")
    cu_kv = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device="cuda")
    causal_offsets = (
        torch.tensor([3, 0, 1], dtype=torch.int32, device="cuda") if with_causal_offsets else None
    )
    total_q = 12
    ratio = 4

    actual = thd_indexer_kernels.build_seq_lens(cu_q, cu_kv, total_q, ratio, causal_offsets)
    expected = thd_indexer_kernels._build_seq_lens_fallback(
        cu_q, cu_kv, total_q, ratio, causal_offsets
    )
    assert torch.equal(actual, expected)


def test_sanitize_indexer_topk_matches_native():
    _require_triton_cuda()
    scores = torch.tensor(
        [[0.0, 1.0, -torch.inf, 3.0], [0.0, torch.nan, 2.0, 3.0], [0.0, 1.0, torch.inf, 3.0]],
        dtype=torch.float32,
        device="cuda",
    )
    candidates = torch.tensor(
        [[0, 1, 2, 4, -1], [1, 2, 3, 0, -1], [2, 3, 1, 0, -1]], dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.tensor([4, 3, 4], dtype=torch.int32, device="cuda")

    output_width = candidates.shape[1] + 2
    actual = thd_indexer_kernels.sanitize_topk(
        candidates, scores, seq_lens, output_width=output_width
    )
    expected = thd_indexer_kernels._sanitize_topk_fallback(
        candidates, scores, seq_lens, output_width=output_width
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_compressor_input_compact_matches_native_forward_backward():
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    global_start, l_local = _e2e_like_local_range()
    ratio = 128
    d_comp = 128
    d_window = 128
    c_cap = (l_local + d_comp) // ratio
    hidden = (
        torch.arange(global_start, global_start + l_local, dtype=torch.float32, device="cuda")[
            :, None
        ]
        .repeat(1, 3)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )
    boundary = (
        torch.arange(global_start - d_window, global_start, dtype=torch.float32, device="cuda")[
            :, None
        ]
        .repeat(1, 3)
        .to(torch.bfloat16)
        .requires_grad_(True)
    )

    ref = _native_compressor_input_compact(
        hidden, boundary, cu, global_start, l_local, ratio, d_comp, d_window, c_cap
    )
    grad = torch.randn_like(ref[0])
    ref[0].backward(grad)
    ref_hidden_grad = hidden.grad.detach().clone()
    ref_boundary_grad = boundary.grad.detach().clone()
    hidden.grad.zero_()
    boundary.grad.zero_()

    fused = thd_layout_kernels.CompressorInputCompact.apply(
        hidden, boundary, cu, global_start, ratio, d_comp, c_cap
    )
    fused[0].backward(grad)
    assert torch.equal(fused[0], ref[0])
    assert torch.equal(fused[1], ref[1])
    assert torch.equal(hidden.grad, ref_hidden_grad)
    assert torch.equal(boundary.grad, ref_boundary_grad)


def test_build_attention_indices_matches_native():
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    global_start, l_local = _e2e_like_local_range()
    d_window = 128
    window = 16
    ratio = 4
    compressed_width = 16
    cu_comp = _compressed_cu_seqlens(cu, ratio)
    seq_to_rank_row = torch.arange(int(cu_comp[-1]), dtype=torch.int32, device="cuda").flip(0)
    compressed_base = d_window + l_local
    fused = thd_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        cu_seqlens_compressed=cu_comp,
        seq_to_rank_row=seq_to_rank_row,
        compressed_rows=seq_to_rank_row.shape[0],
    )
    expected = _native_attention_indices(
        cu,
        cu_comp,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        seq_to_rank_row,
        compressed_base,
    )
    assert torch.equal(fused[0], expected[0])
    assert torch.equal(fused[1], expected[1])

    logical_ids = torch.tensor([2, 0, -1], dtype=torch.int32, device="cuda").expand(l_local, -1)
    fused_indexer = thd_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        logical_ids.shape[1],
        logical_ids,
        cu_seqlens_compressed=cu_comp,
        seq_to_rank_row=seq_to_rank_row,
        compressed_rows=seq_to_rank_row.shape[0],
    )
    expected_indexer = _native_attention_indices(
        cu,
        cu_comp,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        logical_ids.shape[1],
        seq_to_rank_row,
        compressed_base,
        logical_ids,
    )
    assert torch.equal(fused_indexer[0], expected_indexer[0])
    assert torch.equal(fused_indexer[1], expected_indexer[1])

    padded = thd_layout_kernels.build_attention_indices(
        torch.tensor([0, 8], dtype=torch.int32, device="cuda"), 0, 10, 2, 2, 0, 0
    )
    # Padded THD query rows still need one in-range dummy KV id for fused DSA.
    assert torch.equal(
        padded[0][8:10], torch.tensor([[0, -1], [0, -1]], dtype=torch.int32, device="cuda")
    )
    assert torch.equal(padded[1][8:10], torch.tensor([1, 1], dtype=torch.int32, device="cuda"))


def test_build_attention_indices_writes_aligned_width():
    _require_cute_cuda()
    cu = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
    cu_comp = torch.tensor([0, 2], dtype=torch.int32, device="cuda")
    logical_ids = torch.tensor([[0, 1, -1], [1, 0, -1]], dtype=torch.int32, device="cuda").repeat(
        4, 1
    )

    exact = thd_layout_kernels.build_attention_indices(
        cu,
        0,
        8,
        0,
        3,
        4,
        logical_ids.shape[-1],
        logical_ids,
        cu_seqlens_compressed=cu_comp,
        for_indexer_loss=True,
        compressed_base=8,
        compressed_rows=2,
        compressed_is_sequence_major=True,
    )
    aligned = thd_layout_kernels.build_attention_indices(
        cu,
        0,
        8,
        0,
        3,
        4,
        logical_ids.shape[-1],
        logical_ids,
        cu_seqlens_compressed=cu_comp,
        for_indexer_loss=True,
        compressed_base=8,
        compressed_rows=2,
        compressed_is_sequence_major=True,
        output_alignment=8,
    )

    logical_width = 3 + logical_ids.shape[-1]
    assert exact[0].shape == (8, logical_width)
    assert aligned[0].shape == (8, 8)
    assert torch.equal(aligned[0][:, :logical_width], exact[0])
    assert torch.all(aligned[0][:, logical_width:] == -1)
    assert torch.equal(aligned[2], exact[2])


def test_build_attention_indices_indexer_loss_mode_matches_native():
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    ratio = 4
    cu_comp = _compressed_cu_seqlens(cu, ratio)
    global_start, l_local = _e2e_like_local_range()
    d_window = 128
    window = 16
    compressed_width = 8
    logical_ids = (
        torch.arange(compressed_width, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .repeat(l_local, 1)
    )
    logical_ids[1::5, -1] = -1
    seq_to_rank_row = torch.arange(int(cu_comp[-1]), dtype=torch.int32, device="cuda").flip(0)
    compressed_base = d_window + l_local
    fused = thd_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        logical_ids,
        cu_seqlens_compressed=cu_comp,
        seq_to_rank_row=seq_to_rank_row,
        for_indexer_loss=True,
        compressed_rows=seq_to_rank_row.shape[0],
    )
    expected = _native_indexer_loss_indices(
        cu,
        cu_comp,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        logical_ids,
        seq_to_rank_row,
        compressed_base,
    )
    assert torch.equal(fused[0], expected[0])
    assert torch.equal(fused[2], expected[1])


def test_build_attention_indices_emits_padding_mask():
    _require_cute_cuda()
    cu_padded = torch.tensor([0, 4, 8, 12], dtype=torch.int32, device="cuda")
    cu_unpadded = torch.tensor([0, 3, 7, 7], dtype=torch.int32, device="cuda")
    cu_compressed = _compressed_cu_seqlens(cu_padded, ratio=4)
    total_q = 14
    logical_ids = torch.zeros((total_q, 1), dtype=torch.int32, device="cuda")

    actual = thd_layout_kernels.build_attention_indices(
        cu_padded,
        0,
        total_q,
        2,
        2,
        4,
        1,
        logical_ids,
        cu_seqlens_compressed=cu_compressed,
        for_indexer_loss=True,
        compressed_base=total_q + 2,
        compressed_rows=int(cu_compressed[-1]),
        compressed_is_sequence_major=True,
        cu_seqlens_unpadded=cu_unpadded,
    )
    expected_mask = torch.tensor(
        [False, False, False, True, False, False, False, False, True, True, True, True, True, True],
        dtype=torch.bool,
        device="cuda",
    )
    assert actual[3] is not None
    assert torch.equal(actual[3], expected_mask)


def test_build_attention_indices_sequence_major_matches_identity_mapping():
    """CP=1 sequence-major lowering must not need a materialized identity map."""
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    ratio = 4
    cu_comp = _compressed_cu_seqlens(cu, ratio)
    l_local = int(cu[-1]) + 2
    window = 16
    compressed_width = 8
    compressed_rows = int(cu_comp[-1]) + 7
    compressed_base = l_local
    identity_map = torch.arange(compressed_rows, dtype=torch.int32, device="cuda")
    logical_ids = (
        torch.arange(compressed_width, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .repeat(l_local, 1)
    )
    logical_ids[1::5, -1] = -1

    common_kwargs = dict(
        cu_seqlens_compressed=cu_comp,
        compressed_base=compressed_base,
        compressed_rows=compressed_rows,
    )
    for compressed_topk, for_indexer_loss in (
        (None, False),
        (logical_ids, False),
        (logical_ids, True),
    ):
        expected = thd_layout_kernels.build_attention_indices(
            cu,
            0,
            l_local,
            0,
            window,
            ratio,
            compressed_width,
            compressed_topk,
            seq_to_rank_row=identity_map,
            for_indexer_loss=for_indexer_loss,
            **common_kwargs,
        )
        actual = thd_layout_kernels.build_attention_indices(
            cu,
            0,
            l_local,
            0,
            window,
            ratio,
            compressed_width,
            compressed_topk,
            for_indexer_loss=for_indexer_loss,
            compressed_is_sequence_major=True,
            **common_kwargs,
        )
        assert torch.equal(actual[0], expected[0])
        if for_indexer_loss:
            assert actual[1] is None
            assert torch.equal(actual[2], expected[2])
        else:
            assert torch.equal(actual[1], expected[1])
            assert actual[2] is None


def test_build_attention_indices_many_short_sequences_match_native():
    _require_cute_cuda()
    cu = torch.arange(0, 4097, 32, dtype=torch.int32, device="cuda")
    ratio = 4
    cu_comp = _compressed_cu_seqlens(cu, ratio)
    global_start = 1024
    l_local = 1024
    d_window = 128
    window = 16
    compressed_width = 128
    seq_to_rank_row = torch.arange(int(cu_comp[-1]), dtype=torch.int32, device="cuda").flip(0)
    compressed_base = d_window + l_local

    actual = thd_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        cu_seqlens_compressed=cu_comp,
        seq_to_rank_row=seq_to_rank_row,
        compressed_rows=seq_to_rank_row.shape[0],
    )
    expected = _native_attention_indices(
        cu,
        cu_comp,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        seq_to_rank_row,
        compressed_base,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])

    logical_ids = torch.arange(compressed_width, dtype=torch.int32, device="cuda").expand(
        l_local, -1
    )
    actual = thd_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        logical_ids,
        cu_seqlens_compressed=cu_comp,
        seq_to_rank_row=seq_to_rank_row,
        for_indexer_loss=True,
        compressed_rows=seq_to_rank_row.shape[0],
    )
    expected = _native_indexer_loss_indices(
        cu,
        cu_comp,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        logical_ids,
        seq_to_rank_row,
        compressed_base,
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[2], expected[1])


@pytest.mark.parametrize(
    ("ratio", "lengths"),
    [(4, (3, 10, 20, 5, 33, 10, 27, 20)), (128, (3, 130, 170, 5, 260, 129, 200, 127))],
    ids=["ratio4", "ratio128"],
)
@pytest.mark.parametrize("cp_size", [2, 4])
def test_composed_cp_layout_maps_every_index_and_gradient_to_its_source(ratio, lengths, cp_size):
    """Compose compaction, rank-major gather, and final index lowering."""
    _require_cute_cuda()
    total = sum(lengths)
    local_rows = total // cp_size
    d_window = 8 if ratio == 4 else ratio
    window_size = 4
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device="cuda"
    )
    cu_compressed = _compressed_cu_seqlens(cu, ratio)
    hidden = (
        torch.arange(1, total + 1, dtype=torch.float32, device="cuda")
        .unsqueeze(1)
        .requires_grad_(True)
    )

    boundaries = []
    locals_ = []
    compact_values = []
    compact_first_tokens = []
    compact_ids = []
    row_maps = []
    for rank in range(cp_size):
        start = rank * local_rows
        local = hidden[start : start + local_rows]
        boundary = torch.cat(
            (
                hidden.new_zeros((max(0, d_window - start), 1)),
                hidden[max(0, start - d_window) : start],
            )
        )
        compact, group_ids, row_map = prepare_cp_compressor_input(
            local, boundary, cu, cu_compressed, start, cp_size, ratio
        )
        grouped = compact.reshape(group_ids.shape[0], ratio, 1)
        boundaries.append(boundary)
        locals_.append(local)
        compact_values.append(grouped.sum(dim=1))
        compact_first_tokens.append(grouped[:, 0, 0])
        compact_ids.append(group_ids)
        row_maps.append(row_map)

    capacity = compact_ids[0].shape[0]
    expected_map = torch.full((total // ratio,), -1, dtype=torch.int32, device="cuda")
    physical_to_tokens = {}
    logical_values = []
    logical_row = 0
    seq_start = 0
    for seq_len in lengths:
        for group in range(seq_len // ratio):
            first_token = seq_start + group * ratio
            owner = (first_token + ratio - 1) // local_rows
            matches = torch.nonzero(
                compact_first_tokens[owner] == first_token + 1, as_tuple=False
            ).flatten()
            assert matches.numel() == 1
            slot = int(matches[0])
            assert int(compact_ids[owner][slot]) == group
            physical_row = owner * capacity + slot
            expected_map[logical_row] = physical_row
            physical_to_tokens[physical_row] = range(first_token, first_token + ratio)
            logical_values.append(sum(range(first_token + 1, first_token + ratio + 1)))
            logical_row += 1
        seq_start += seq_len

    for row_map in row_maps:
        assert torch.equal(row_map, expected_map)
    reachable = set(int(row) for row in expected_map[expected_map >= 0].cpu().tolist())
    all_ids = torch.cat(compact_ids)
    assert all(int(row) not in reachable for row in torch.nonzero(all_ids < 0).flatten().tolist())

    compressed_rank_major = torch.cat(compact_values)
    sequence_major = torch.index_select(
        compressed_rank_major, 0, expected_map[:logical_row].long()
    ).squeeze(1)
    assert torch.equal(
        sequence_major, torch.tensor(logical_values, dtype=torch.float32, device="cuda")
    )
    compressed_width = 3 if ratio == 4 else max(lengths) // ratio
    loss = hidden.new_zeros(())
    expected_grad = torch.zeros_like(hidden)
    cu_list = [int(value) for value in cu.cpu().tolist()]
    for rank in range(cp_size):
        start = rank * local_rows
        logical_topk = None
        if ratio == 4:
            logical_topk = torch.full(
                (local_rows, compressed_width), -1, dtype=torch.int32, device="cuda"
            )
            for row, global_row in enumerate(range(start, start + local_rows)):
                seq = next(i for i in range(len(lengths)) if global_row < cu_list[i + 1])
                visible = min((global_row - cu_list[seq] + 1) // ratio, lengths[seq] // ratio)
                selected = list(range(visible - 1, max(-1, visible - compressed_width - 1), -1))
                if selected:
                    logical_topk[row, : len(selected)] = torch.tensor(
                        selected, dtype=torch.int32, device="cuda"
                    )

        actual = thd_layout_kernels.build_attention_indices(
            cu,
            start,
            local_rows,
            d_window,
            window_size,
            ratio,
            compressed_width,
            logical_topk,
            cu_seqlens_compressed=cu_compressed,
            seq_to_rank_row=expected_map,
            compressed_rows=compressed_rank_major.shape[0],
        )
        expected = _native_attention_indices(
            cu,
            cu_compressed,
            start,
            local_rows,
            d_window,
            window_size,
            ratio,
            compressed_width,
            expected_map,
            d_window + local_rows,
            logical_topk,
        )
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])

        if logical_topk is not None:
            actual = thd_layout_kernels.build_attention_indices(
                cu,
                start,
                local_rows,
                d_window,
                window_size,
                ratio,
                compressed_width,
                logical_topk,
                cu_seqlens_compressed=cu_compressed,
                seq_to_rank_row=expected_map,
                for_indexer_loss=True,
                compressed_rows=compressed_rank_major.shape[0],
            )
            expected_loss_indices = _native_indexer_loss_indices(
                cu,
                cu_compressed,
                start,
                local_rows,
                d_window,
                window_size,
                ratio,
                logical_topk,
                expected_map,
                d_window + local_rows,
            )
            assert torch.equal(actual[0], expected_loss_indices[0])
            assert torch.equal(actual[2], expected_loss_indices[1])

        indices = actual[0]
        valid = indices >= 0
        kv = torch.cat((boundaries[rank], locals_[rank], compressed_rank_major))
        selected = torch.index_select(kv, 0, indices.clamp_min(0).long().flatten()).reshape(
            indices.shape
        )
        coefficients = (
            torch.arange(indices.numel(), dtype=torch.float32, device="cuda").reshape(indices.shape)
            + rank * indices.numel()
            + 1
        )
        loss = loss + (selected * coefficients * valid).sum()

        compressed_base = d_window + local_rows
        for row, index_row in enumerate(indices.cpu().tolist()):
            for column, index in enumerate(index_row):
                if index < 0:
                    continue
                coefficient = float(coefficients[row, column])
                if index < compressed_base:
                    token = start - d_window + index
                    assert 0 <= token <= start + row
                    expected_grad[token] += coefficient
                else:
                    physical_row = index - compressed_base
                    assert physical_row in physical_to_tokens
                    for token in physical_to_tokens[physical_row]:
                        expected_grad[token] += coefficient

    loss.backward()
    assert torch.equal(hidden.grad, expected_grad)
