# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
from typing import List, Tuple

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa_cp_layout_kernels

# This file guards only DSv4 CP layout/metadata kernels. Layer-level CUDA graph
# tests guard graph capture/replay behavior.

_E2E_RAGGED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 148)
_E2E_RAGGED_PADDED_SEG_LENS = (1, 127, 1000, 23, 129, 900, 55, 257, 800, 95, 509, 200)
_E2E_CP_SIZE = 4
_E2E_CP_RANK = 1


def _require_cute_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CuTe kernels require CUDA.")
    if not csa_cp_layout_kernels._CUTE_AVAILABLE:
        pytest.skip("DSv4 CP CuTe kernels are not available in this environment.")


def _make_e2e_like_cu_seqlens(device: str = "cuda", *, padded: bool = True) -> torch.Tensor:
    """Return the ragged THD prefix pattern used by DSv4 CP e2e tests.

    The padded variant sums to 4096, so CP4 owns 1024 local rows. It contains
    short, long, boundary-crossing, and padded-tail sequences without allocating
    full e2e hidden sizes in these focused kernel tests.
    """
    lengths = _E2E_RAGGED_PADDED_SEG_LENS if padded else _E2E_RAGGED_SEG_LENS
    return torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32, device=device
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


def _e2e_like_contiguous_range(cp_rank: int = _E2E_CP_RANK) -> Tuple[int, int]:
    """Return one CP4 contiguous local range for the e2e-like padded THD layout."""
    total = sum(_E2E_RAGGED_PADDED_SEG_LENS)
    local = total // _E2E_CP_SIZE
    start = int(cp_rank) * local
    return start, local


def _e2e_like_two_chunk_ranges(cp_rank: int = _E2E_CP_RANK) -> Tuple[Tuple[int, int], ...]:
    """Return one CP4 two-chunk local row order for the e2e-like THD layout."""
    total = sum(_E2E_RAGGED_PADDED_SEG_LENS)
    local = total // _E2E_CP_SIZE
    chunk_len = local // 2
    total_chunks = 2 * _E2E_CP_SIZE
    chunk_ids = (int(cp_rank), total_chunks - 1 - int(cp_rank))
    return tuple((chunk_id * chunk_len, (chunk_id + 1) * chunk_len) for chunk_id in chunk_ids)


def _cosine_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float().reshape(-1)
    b = expected.detach().float().reshape(-1)
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return float(torch.dot(a, b) / denom)


def _tensor_sim(actual: torch.Tensor, expected: torch.Tensor) -> float:
    a = actual.detach().float()
    b = expected.detach().float()
    denom = b.norm().clamp_min(1e-12)
    return float(1.0 - (a - b).norm() / denom)


def _assert_rope_close(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    cos = _cosine_sim(actual, expected)
    sim = _tensor_sim(actual, expected)
    assert cos > 0.9999, f"{name} cosine similarity too low: {cos}"
    assert sim > 0.9999, f"{name} tensor similarity too low: {sim}"


def _num_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _time_cuda(fn, *, warmup: int = 3, iters: int = 5) -> float:
    iters = int(os.environ.get("DSV4_CP_KERNEL_BENCH_ITERS", iters))
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / iters)


def _print_perf(name: str, cute_ms: float, torch_ms: float, effective_bytes: int) -> None:
    cute_gbps = effective_bytes / max(cute_ms, 1e-9) / 1.0e6
    torch_gbps = effective_bytes / max(torch_ms, 1e-9) / 1.0e6
    print(
        f"{name}: cute={cute_ms:.4f} ms ({cute_gbps:.2f} GB/s effective), "
        f"torch={torch_ms:.4f} ms ({torch_gbps:.2f} GB/s effective), "
        f"speedup={torch_ms / max(cute_ms, 1e-9):.2f}x"
    )


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


def _seq_positions_from_rows(
    cu_seqlens: torch.Tensor, global_rows: torch.Tensor, clamp_to_valid_token: bool = False
) -> torch.Tensor:
    cu = cu_seqlens.cpu().to(torch.long)
    rows = global_rows.cpu().to(torch.long)
    if clamp_to_valid_token:
        rows = rows.clamp(min=0, max=int(cu[-1]) - 1)
    seq_ids = torch.searchsorted(cu, rows, right=True) - 1
    seq_ids = seq_ids.clamp(min=0, max=cu.numel() - 2)
    return (rows - cu[seq_ids]).to(global_rows.device)


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


def _native_attention_indices_one_seq(
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    device: torch.device,
):
    rows = torch.arange(l_local, dtype=torch.int32, device=device).unsqueeze(1)
    window_cols = torch.arange(window_size, dtype=torch.int32, device=device).unsqueeze(0)
    window_start = (rows - int(window_size) + 1).clamp_min(0)
    window_count = rows - window_start + 1
    window_values = int(d_window) + window_start + window_cols
    window_values = torch.where(
        window_cols < window_count, window_values, torch.full_like(window_values, -1)
    )
    if ratio > 1 and compressed_width > 0:
        comp_cols = torch.arange(compressed_width, dtype=torch.int32, device=device).unsqueeze(0)
        visible = ((rows + 1) // int(ratio)).clamp(max=int(compressed_width))
        comp_values = int(d_window) + int(l_local) + comp_cols
        comp_values = torch.where(
            comp_cols < visible, comp_values, torch.full_like(comp_values, -1)
        )
        topk = torch.cat((window_values, comp_values.expand(l_local, -1)), dim=1)
        topk_length = window_count.squeeze(1) + visible.squeeze(1)
    else:
        topk = window_values
        topk_length = window_count.squeeze(1)
    return topk, topk_length


def _native_attention_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    compressed_width: int,
    rank_by_seq_major: torch.Tensor,
    compressed_base: int,
    indexer_topk: torch.Tensor = None,
):
    """Reference contiguous final-index lowering for ragged THD CP rows."""
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
                    rank_id = int(rank_by_seq_major[cu_comp[seq_id] + comp_id])
                    if rank_id >= 0:
                        topk[row, write_col] = int(compressed_base) + rank_id
                        write_col += 1
        lengths[row] = write_col
    return topk, lengths


def _native_indexer_loss_indices_one_seq(
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    logical_ids: torch.Tensor,
    rank_by_seq_major: torch.Tensor,
):
    device = logical_ids.device
    compressed_width = logical_ids.shape[1]
    attention_topk, _ = _native_attention_indices_one_seq(
        l_local, d_window, window_size, ratio, 0, device
    )
    compressed = torch.full((l_local, compressed_width), -1, dtype=torch.int32, device=device)
    rank_major = torch.full_like(compressed, -1)
    valid = (logical_ids >= 0) & (logical_ids < rank_by_seq_major.shape[0])
    if valid.any():
        lowered = int(d_window) + int(l_local) + logical_ids.clamp(min=0)
        rank_ids = rank_by_seq_major.index_select(0, logical_ids.clamp(min=0).reshape(-1)).view_as(
            logical_ids
        )
        compressed = torch.where(valid, lowered, compressed)
        rank_major = torch.where(valid, rank_ids, rank_major)
    return torch.cat((compressed, attention_topk), dim=1), rank_major


def _native_indexer_loss_indices(
    cu_seqlens: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    window_size: int,
    ratio: int,
    logical_ids: torch.Tensor,
    rank_by_seq_major: torch.Tensor,
    compressed_base: int,
):
    """Reference contiguous compressed-first indexer-loss lowering for ragged THD rows."""
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
                rank_id = int(rank_by_seq_major[seq_major])
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


def test_thd_local_rope_matches_native_forward_backward_and_reports_bandwidth():
    _require_cute_cuda()
    torch.manual_seed(11)
    cu = _make_e2e_like_cu_seqlens()
    chunk_ranges = _e2e_like_two_chunk_ranges()
    rows = sum(end - start for start, end in chunk_ranges)
    nope_dim = 4
    pos_dim = 4
    x = torch.randn(rows, 2, nope_dim + pos_dim, device="cuda", requires_grad=True)
    cos = torch.randn(max(_E2E_RAGGED_PADDED_SEG_LENS), 1, 1, pos_dim, device="cuda")
    sin = torch.randn(max(_E2E_RAGGED_PADDED_SEG_LENS), 1, 1, pos_dim, device="cuda")
    global_rows = torch.cat(
        [torch.arange(start, end, device="cuda") for start, end in chunk_ranges]
    )
    positions = _seq_positions_from_rows(cu, global_rows)
    ref_x = x.detach().clone().requires_grad_(True)
    ref = _rope_reference(ref_x, cos, sin, positions, nope_dim, pos_dim)
    grad = torch.randn_like(ref)
    ref.backward(grad)

    fused_x = x.detach().clone().requires_grad_(True)
    fused = csa_cp_layout_kernels.ThdLocalRope.apply(
        fused_x,
        cos,
        sin,
        cu,
        chunk_ranges[0][0],
        chunk_ranges[0][1] - chunk_ranges[0][0],
        nope_dim,
        pos_dim,
        chunk_ranges[1][0],
        False,
        False,
    )
    fused.backward(grad)
    _assert_rope_close(fused, ref, "thd_local_rope.forward")
    _assert_rope_close(fused_x.grad, ref_x.grad, "thd_local_rope.backward")

    bench_x = torch.randn(512, 8, nope_dim + pos_dim, device="cuda")
    bench_grad = torch.randn_like(bench_x)
    bench_cu = torch.tensor([0, 2048], dtype=torch.int32, device="cuda")
    bench_cos = torch.randn(2048, 1, 1, pos_dim, device="cuda")
    bench_sin = torch.randn(2048, 1, 1, pos_dim, device="cuda")
    bench_pos = torch.arange(512, device="cuda")

    def run_cute():
        bx = bench_x.detach().clone().requires_grad_(True)
        y = csa_cp_layout_kernels.ThdLocalRope.apply(
            bx, bench_cos, bench_sin, bench_cu, 0, 512, nope_dim, pos_dim, 0, False, False
        )
        y.backward(bench_grad)

    def run_torch():
        bx = bench_x.detach().clone().requires_grad_(True)
        y = _rope_reference(bx, bench_cos, bench_sin, bench_pos, nope_dim, pos_dim)
        y.backward(bench_grad)

    effective_bytes = 4 * _num_bytes(bench_x)
    _print_perf(
        "thd_local_rope_fwd_bwd", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )


def test_compressor_input_compact_matches_native_forward_backward_and_reports_bandwidth():
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    global_start, l_local = _e2e_like_contiguous_range()
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

    fused = csa_cp_layout_kernels.CompressorInputCompact.apply(
        hidden, boundary, cu, global_start, l_local, ratio, d_comp, d_window, c_cap
    )
    fused[0].backward(grad)
    assert torch.equal(fused[0], ref[0])
    assert torch.equal(fused[1], ref[1])
    assert torch.equal(hidden.grad, ref_hidden_grad)
    assert torch.equal(boundary.grad, ref_boundary_grad)

    bench_cu = torch.tensor([0, 16384], dtype=torch.int32, device="cuda")
    bench_global_start = 4096
    bench_l_local = 8192
    bench_ratio = 4
    bench_d_comp = 128
    bench_d_window = 128
    bench_c_cap = (bench_l_local + bench_d_comp) // bench_ratio
    bench_hidden = torch.randn(bench_l_local, 64, dtype=torch.bfloat16, device="cuda")
    bench_boundary = torch.randn(bench_d_window, 64, dtype=torch.bfloat16, device="cuda")
    bench_grad = torch.randn(bench_c_cap * bench_ratio, 64, dtype=torch.bfloat16, device="cuda")

    def run_cute():
        h = bench_hidden.detach().clone().requires_grad_(True)
        b = bench_boundary.detach().clone().requires_grad_(True)
        y = csa_cp_layout_kernels.CompressorInputCompact.apply(
            h,
            b,
            bench_cu,
            bench_global_start,
            bench_l_local,
            bench_ratio,
            bench_d_comp,
            bench_d_window,
            bench_c_cap,
        )[0]
        y.backward(bench_grad)

    def run_torch():
        h = bench_hidden.detach().clone().requires_grad_(True)
        b = bench_boundary.detach().clone().requires_grad_(True)
        y = _native_compressor_input_compact(
            h,
            b,
            bench_cu,
            bench_global_start,
            bench_l_local,
            bench_ratio,
            bench_d_comp,
            bench_d_window,
            bench_c_cap,
        )[0]
        y.backward(bench_grad)

    effective_bytes = 2 * (_num_bytes(bench_hidden, bench_boundary, bench_grad))
    _print_perf(
        "compressor_input_compact_fwd_bwd",
        _time_cuda(run_cute),
        _time_cuda(run_torch),
        effective_bytes,
    )


def test_build_attention_indices_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    global_start, l_local = _e2e_like_contiguous_range()
    d_window = 128
    window = 16
    ratio = 4
    compressed_width = 16
    cu_comp = _compressed_cu_seqlens(cu, ratio)
    rank_by_seq = torch.arange(int(cu_comp[-1]), dtype=torch.int32, device="cuda").flip(0)
    compressed_base = d_window + l_local
    fused = csa_cp_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        compressed_width,
        cu_seqlens_compressed=cu_comp,
        rank_row_for_seq_row=rank_by_seq,
        compressed_base=compressed_base,
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
        rank_by_seq,
        compressed_base,
    )
    assert torch.equal(fused[0], expected[0])
    assert torch.equal(fused[1], expected[1])

    logical_ids = torch.tensor([2, 0, -1], dtype=torch.int32, device="cuda").expand(l_local, -1)
    fused_indexer = csa_cp_layout_kernels.build_attention_indices(
        cu,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        logical_ids.shape[1],
        logical_ids,
        cu_seqlens_compressed=cu_comp,
        rank_row_for_seq_row=rank_by_seq,
        compressed_base=compressed_base,
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
        rank_by_seq,
        compressed_base,
        logical_ids,
    )
    assert torch.equal(fused_indexer[0], expected_indexer[0])
    assert torch.equal(fused_indexer[1], expected_indexer[1])

    padded = csa_cp_layout_kernels.build_attention_indices(
        torch.tensor([0, 8], dtype=torch.int32, device="cuda"), 0, 10, 2, 2, 0, 0
    )
    # Padded THD query rows still need one in-range dummy KV id for fused DSA.
    assert torch.equal(
        padded[0][8:10], torch.tensor([[0, -1], [0, -1]], dtype=torch.int32, device="cuda")
    )
    assert torch.equal(padded[1][8:10], torch.tensor([1, 1], dtype=torch.int32, device="cuda"))

    bench_l_local = 4096
    bench_window = 128
    bench_ratio = 4
    bench_compressed_width = bench_l_local // bench_ratio
    bench_cu = torch.tensor([0, bench_l_local], dtype=torch.int32, device="cuda")
    bench_cu_comp = torch.tensor([0, bench_compressed_width], dtype=torch.int32, device="cuda")
    bench_rank_by_seq = torch.arange(bench_compressed_width, dtype=torch.int32, device="cuda")
    bench_ref = _native_attention_indices_one_seq(
        bench_l_local,
        bench_window,
        bench_window,
        bench_ratio,
        bench_compressed_width,
        torch.device("cuda"),
    )

    def run_cute():
        csa_cp_layout_kernels.build_attention_indices(
            bench_cu,
            0,
            bench_l_local,
            bench_window,
            bench_window,
            bench_ratio,
            bench_compressed_width,
            cu_seqlens_compressed=bench_cu_comp,
            rank_row_for_seq_row=bench_rank_by_seq,
            compressed_base=bench_window + bench_l_local,
        )

    def run_torch():
        _native_attention_indices_one_seq(
            bench_l_local,
            bench_window,
            bench_window,
            bench_ratio,
            bench_compressed_width,
            torch.device("cuda"),
        )

    effective_bytes = _num_bytes(*bench_ref)
    _print_perf(
        "build_attention_indices", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )


def test_build_indexer_loss_indices_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    cu = _make_e2e_like_cu_seqlens()
    ratio = 4
    cu_comp = _compressed_cu_seqlens(cu, ratio)
    global_start, l_local = _e2e_like_contiguous_range()
    d_window = 128
    window = 16
    compressed_width = 8
    logical_ids = (
        torch.arange(compressed_width, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .repeat(l_local, 1)
    )
    logical_ids[1::5, -1] = -1
    rank_by_seq = torch.arange(int(cu_comp[-1]), dtype=torch.int32, device="cuda").flip(0)
    compressed_base = d_window + l_local
    fused = csa_cp_layout_kernels.build_indexer_loss_indices(
        cu,
        cu_comp,
        global_start,
        l_local,
        d_window,
        window,
        ratio,
        logical_ids,
        rank_by_seq,
        compressed_base=compressed_base,
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
        rank_by_seq,
        compressed_base,
    )
    assert torch.equal(fused[0], expected[0])
    assert torch.equal(fused[1], expected[1])

    bench_l_local = 4096
    bench_window = 128
    bench_ratio = 4
    bench_compressed_width = bench_l_local // bench_ratio
    bench_cu = torch.tensor([0, bench_l_local], dtype=torch.int32, device="cuda")
    bench_cu_comp = torch.tensor([0, bench_compressed_width], dtype=torch.int32, device="cuda")
    bench_logical = (
        torch.arange(bench_compressed_width, dtype=torch.int32, device="cuda")
        .unsqueeze(0)
        .expand(bench_l_local, -1)
    )
    bench_rank_by_seq = torch.arange(bench_compressed_width, dtype=torch.int32, device="cuda")
    bench_ref = _native_indexer_loss_indices_one_seq(
        bench_l_local, bench_window, bench_window, bench_ratio, bench_logical, bench_rank_by_seq
    )

    def run_cute():
        csa_cp_layout_kernels.build_indexer_loss_indices(
            bench_cu,
            bench_cu_comp,
            0,
            bench_l_local,
            bench_window,
            bench_window,
            bench_ratio,
            bench_logical,
            bench_rank_by_seq,
            compressed_base=bench_window + bench_l_local,
        )

    def run_torch():
        _native_indexer_loss_indices_one_seq(
            bench_l_local, bench_window, bench_window, bench_ratio, bench_logical, bench_rank_by_seq
        )

    effective_bytes = _num_bytes(bench_logical, bench_rank_by_seq, *bench_ref)
    _print_perf(
        "build_indexer_loss_indices", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )
