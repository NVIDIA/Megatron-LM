# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os
from typing import List, Tuple

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import csa_cp_kernels

# This file guards only DSv4 CP layout/metadata kernels. Layer-level CUDA graph
# tests guard graph capture/replay behavior.


def _require_cute_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CuTe kernels require CUDA.")
    if not csa_cp_kernels._CUTE_AVAILABLE:
        pytest.skip("DSv4 CP CuTe kernels are not available in this environment.")


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

    cu_compact = torch.zeros_like(cu_seqlens)
    running = 0
    for seq in range(len(cu) - 1):
        running += sum(1 for group_seq, _ in groups if group_seq == seq) * int(ratio)
        cu_compact[seq + 1] = running

    seq_ids = torch.full((c_cap,), -1, dtype=torch.int32, device=hidden_local.device)
    comp_ids = torch.full((c_cap,), -1, dtype=torch.int32, device=hidden_local.device)
    valid = torch.zeros((c_cap,), dtype=torch.bool, device=hidden_local.device)
    for slot, (seq, comp_id) in enumerate(groups[:c_cap]):
        seq_start = cu[seq]
        group_end = seq_start + (comp_id + 1) * int(ratio)
        seq_ids[slot] = seq
        comp_ids[slot] = comp_id
        valid[slot] = group_end - 1 >= global_start and group_end - 1 < global_start + l_local
    return hidden_compact, cu_compact, seq_ids, comp_ids, valid


def _native_build_compressed_row_metadata(
    cu_seqlens: torch.Tensor,
    cp_size: int,
    chunk_len: int,
    ratio: int,
    d_comp: int,
    c_cap_per_chunk: int,
    c_cap_per_rank: int,
    use_two_chunk: bool,
):
    total_rows = int(cp_size) * int(c_cap_per_rank)
    seq_ids = torch.full((total_rows,), -1, dtype=torch.int32, device=cu_seqlens.device)
    comp_ids = torch.full((total_rows,), -1, dtype=torch.int32, device=cu_seqlens.device)
    valid = torch.zeros((total_rows,), dtype=torch.bool, device=cu_seqlens.device)
    for row in range(total_rows):
        rank = row // c_cap_per_rank
        rank_slot = row - rank * c_cap_per_rank
        local_chunk = 0
        slot = rank_slot
        chunk_id = rank
        if use_two_chunk:
            local_chunk = rank_slot // c_cap_per_chunk
            if local_chunk >= 2:
                continue
            slot = rank_slot - local_chunk * c_cap_per_chunk
            chunk_id = rank if local_chunk == 0 else cp_size * 2 - 1 - rank
        rank_start = chunk_id * chunk_len
        groups = _compressed_groups(cu_seqlens, rank_start, chunk_len, ratio, d_comp)
        if slot < len(groups):
            seq, comp = groups[slot]
            seq_start = int(cu_seqlens[seq].item())
            group_end = seq_start + (comp + 1) * ratio
            seq_ids[row] = seq
            comp_ids[row] = comp
            valid[row] = group_end - 1 >= rank_start and group_end - 1 < rank_start + chunk_len
    return seq_ids, comp_ids, valid


def _native_repack_compressed_kv_to_seq_major(
    compressed_rank_major: torch.Tensor,
    seq_ids: torch.Tensor,
    comp_ids: torch.Tensor,
    valid: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    seq_major_rows: int,
):
    out = compressed_rank_major.new_zeros(
        (seq_major_rows,) + tuple(compressed_rank_major.shape[1:])
    )
    rank_by_seq = torch.full(
        (seq_major_rows,), -1, dtype=torch.int32, device=compressed_rank_major.device
    )
    for rank_row in range(compressed_rank_major.shape[0]):
        if bool(valid[rank_row]):
            seq = int(seq_ids[rank_row])
            comp = int(comp_ids[rank_row])
            seq_major = int(cu_seqlens_compressed[seq]) + comp
            if 0 <= seq_major < seq_major_rows:
                out[seq_major] = compressed_rank_major[rank_row]
                rank_by_seq[seq_major] = rank_row
    return out, rank_by_seq


def _native_indexer_topk_metadata(
    k_seq_major: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_compressed: torch.Tensor,
    global_start: int,
    l_local: int,
    ratio: int,
):
    n_seq = cu_seqlens_q.shape[0] - 1
    k_topk = torch.zeros_like(k_seq_major)
    cu_q = torch.zeros((n_seq + 2,), dtype=torch.int32, device=k_seq_major.device)
    cu_k = torch.zeros((n_seq + 2,), dtype=torch.int32, device=k_seq_major.device)
    seq_lens = torch.empty((l_local,), dtype=torch.int32, device=k_seq_major.device)
    q_prefix = 0
    k_prefix = 0
    for seq in range(n_seq):
        seq_start = int(cu_seqlens_q[seq])
        seq_end = int(cu_seqlens_q[seq + 1])
        local_start = max(seq_start, int(global_start))
        local_end = min(seq_end, int(global_start) + int(l_local))
        q_len = 0
        k_len = 0
        if local_start < local_end:
            q_len = local_end - local_start
            seq_comp_start = int(cu_seqlens_compressed[seq])
            seq_comp_end = int(cu_seqlens_compressed[seq + 1])
            k_len = min((local_end - seq_start) // int(ratio), seq_comp_end - seq_comp_start)
            if k_len:
                k_topk[k_prefix : k_prefix + k_len] = k_seq_major[
                    seq_comp_start : seq_comp_start + k_len
                ]
        if q_len:
            seq_lens[q_prefix : q_prefix + q_len] = k_len
        q_prefix += q_len
        k_prefix += k_len
        cu_q[seq + 1] = q_prefix
        cu_k[seq + 1] = k_prefix
    actual_total = int(cu_seqlens_q[-1])
    padding_start = max(actual_total, int(global_start))
    padding_q = max(0, int(global_start) + int(l_local) - padding_start)
    if padding_q:
        seq_lens[q_prefix : q_prefix + padding_q] = 0
    cu_q[n_seq + 1] = q_prefix + padding_q
    cu_k[n_seq + 1] = k_prefix
    return k_topk, cu_q, cu_k, seq_lens


def _native_thd_full_kv_pack(
    kv_local: torch.Tensor,
    boundary_kv: torch.Tensor,
    compressed_rank_major: torch.Tensor,
    seq_ids: torch.Tensor,
    comp_ids: torch.Tensor,
    valid: torch.Tensor,
    cu_seqlens: torch.Tensor,
    global_start: int,
    l_local: int,
    d_window: int,
    ratio: int,
    capacity: int,
):
    rows = []
    n_seq = cu_seqlens.shape[0] - 1
    for seq in range(n_seq):
        seq_start = int(cu_seqlens[seq])
        seq_end = int(cu_seqlens[seq + 1])
        local_start = max(seq_start, int(global_start))
        local_end = min(seq_end, int(global_start) + int(l_local))
        if local_start >= local_end:
            continue
        window_start = max(seq_start, local_start - int(d_window))
        for src_global in range(window_start, local_end):
            if src_global < global_start:
                rows.append(
                    boundary_kv[
                        src_global
                        - (global_start - d_window) : src_global
                        - (global_start - d_window)
                        + 1
                    ]
                )
            else:
                rows.append(kv_local[src_global - global_start : src_global - global_start + 1])
        if ratio > 1:
            for comp_id in range((seq_end - seq_start) // int(ratio)):
                for rank_row in range(compressed_rank_major.shape[0]):
                    if (
                        bool(valid[rank_row])
                        and int(seq_ids[rank_row]) == seq
                        and int(comp_ids[rank_row]) == comp_id
                    ):
                        rows.append(compressed_rank_major[rank_row : rank_row + 1])
    if rows:
        out = torch.cat(rows, dim=0)
    else:
        out = kv_local.new_empty((0,) + tuple(kv_local.shape[1:]))
    if out.shape[0] < capacity:
        out = torch.cat(
            (out, kv_local.new_zeros((capacity - out.shape[0],) + tuple(kv_local.shape[1:]))), dim=0
        )
    return out


def _native_attention_indices_one_seq(
    l_local: int, window_size: int, ratio: int, compressed_width: int, device: torch.device
):
    rows = torch.arange(l_local, dtype=torch.int32, device=device).unsqueeze(1)
    window_cols = torch.arange(window_size, dtype=torch.int32, device=device).unsqueeze(0)
    window_start = (rows - int(window_size) + 1).clamp_min(0)
    window_count = rows - window_start + 1
    window_values = window_start + window_cols
    window_values = torch.where(
        window_cols < window_count, window_values, torch.full_like(window_values, -1)
    )
    if ratio > 1 and compressed_width > 0:
        comp_cols = torch.arange(compressed_width, dtype=torch.int32, device=device).unsqueeze(0)
        visible = ((rows + 1) // int(ratio)).clamp(max=int(compressed_width))
        comp_values = int(l_local) + comp_cols
        comp_values = torch.where(
            comp_cols < visible, comp_values, torch.full_like(comp_values, -1)
        )
        topk = torch.cat((window_values, comp_values.expand(l_local, -1)), dim=1)
        topk_length = window_count.squeeze(1) + visible.squeeze(1)
    else:
        topk = window_values
        topk_length = window_count.squeeze(1)
    return topk, topk_length


def _native_indexer_loss_indices_one_seq(
    l_local: int,
    window_size: int,
    ratio: int,
    logical_ids: torch.Tensor,
    rank_by_seq_major: torch.Tensor,
):
    device = logical_ids.device
    compressed_width = logical_ids.shape[1]
    attention_topk, _ = _native_attention_indices_one_seq(l_local, window_size, ratio, 0, device)
    compressed = torch.full((l_local, compressed_width), -1, dtype=torch.int32, device=device)
    rank_major = torch.full_like(compressed, -1)
    valid = (logical_ids >= 0) & (logical_ids < rank_by_seq_major.shape[0])
    if valid.any():
        lowered = int(l_local) + logical_ids.clamp(min=0)
        rank_ids = rank_by_seq_major.index_select(0, logical_ids.clamp(min=0).reshape(-1)).view_as(
            logical_ids
        )
        compressed = torch.where(valid, lowered, compressed)
        rank_major = torch.where(valid, rank_ids, rank_major)
    return torch.cat((compressed, attention_topk), dim=1), rank_major


def test_thd_local_rope_matches_native_forward_backward_and_reports_bandwidth():
    _require_cute_cuda()
    torch.manual_seed(11)
    cu = torch.tensor([0, 7, 18, 32], dtype=torch.int32, device="cuda")
    chunk_ranges = ((3, 9), (22, 28))
    rows = sum(end - start for start, end in chunk_ranges)
    nope_dim = 4
    pos_dim = 4
    x = torch.randn(rows, 2, nope_dim + pos_dim, device="cuda", requires_grad=True)
    cos = torch.randn(32, 1, 1, pos_dim, device="cuda")
    sin = torch.randn(32, 1, 1, pos_dim, device="cuda")
    global_rows = torch.cat(
        [torch.arange(start, end, device="cuda") for start, end in chunk_ranges]
    )
    positions = _seq_positions_from_rows(cu, global_rows)
    ref_x = x.detach().clone().requires_grad_(True)
    ref = _rope_reference(ref_x, cos, sin, positions, nope_dim, pos_dim)
    grad = torch.randn_like(ref)
    ref.backward(grad)

    fused_x = x.detach().clone().requires_grad_(True)
    fused = csa_cp_kernels.ThdLocalRope.apply(
        fused_x,
        cos,
        sin,
        cu,
        chunk_ranges[0][0],
        6,
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
        y = csa_cp_kernels.ThdLocalRope.apply(
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


def test_thd_compressed_rope_matches_native_forward_backward_and_reports_bandwidth():
    _require_cute_cuda()
    torch.manual_seed(22)
    comp_ids = torch.tensor([0, 2, -1, 3], dtype=torch.int32, device="cuda")
    ratio = 4
    nope_dim = 3
    pos_dim = 4
    x = torch.randn(4, 1, 2, nope_dim + pos_dim, device="cuda", requires_grad=True)
    cos = torch.randn(32, 1, 1, pos_dim, device="cuda")
    sin = torch.randn(32, 1, 1, pos_dim, device="cuda")
    positions = comp_ids.clamp(min=0).to(torch.long) * ratio
    ref_x = x.detach().clone().requires_grad_(True)
    ref = _rope_reference(ref_x, cos, sin, positions, nope_dim, pos_dim)
    grad = torch.randn_like(ref)
    ref.backward(grad)

    fused_x = x.detach().clone().requires_grad_(True)
    fused = csa_cp_kernels.ThdCompressedRope.apply(
        fused_x, cos, sin, comp_ids, ratio, nope_dim, pos_dim, False
    )
    fused.backward(grad)
    _assert_rope_close(fused, ref, "thd_compressed_rope.forward")
    _assert_rope_close(fused_x.grad, ref_x.grad, "thd_compressed_rope.backward")

    bench_x = torch.randn(1024, 4, nope_dim + pos_dim, device="cuda")
    bench_grad = torch.randn_like(bench_x)
    bench_ids = torch.arange(1024, dtype=torch.int32, device="cuda") % 8
    bench_pos = bench_ids.to(torch.long) * ratio

    def run_cute():
        bx = bench_x.detach().clone().requires_grad_(True)
        y = csa_cp_kernels.ThdCompressedRope.apply(
            bx, cos, sin, bench_ids, ratio, nope_dim, pos_dim, False
        )
        y.backward(bench_grad)

    def run_torch():
        bx = bench_x.detach().clone().requires_grad_(True)
        y = _rope_reference(bx, cos, sin, bench_pos, nope_dim, pos_dim)
        y.backward(bench_grad)

    effective_bytes = 4 * _num_bytes(bench_x)
    _print_perf(
        "thd_compressed_rope_fwd_bwd", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )


def test_compressor_input_compact_matches_native_forward_backward_and_reports_bandwidth():
    _require_cute_cuda()
    cu = torch.tensor([0, 192, 512], dtype=torch.int32, device="cuda")
    global_start = 384
    l_local = 128
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

    fused = csa_cp_kernels.CompressorInputCompact.apply(
        hidden, boundary, cu, global_start, l_local, ratio, d_comp, d_window, c_cap
    )
    fused[0].backward(grad)
    assert torch.equal(fused[0], ref[0])
    for actual, expected in zip(fused[1:], ref[1:]):
        assert torch.equal(actual, expected)
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
        y = csa_cp_kernels.CompressorInputCompact.apply(
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


def test_thd_full_kv_pack_matches_native_forward_backward_and_reports_bandwidth():
    _require_cute_cuda()
    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    global_start = 3
    l_local = 3
    d_window = 2
    ratio = 2
    kv_local = torch.tensor(
        [[103.0], [104.0], [105.0]], dtype=torch.bfloat16, device="cuda"
    ).requires_grad_(True)
    boundary = torch.tensor([[101.0], [102.0]], dtype=torch.bfloat16, device="cuda").requires_grad_(
        True
    )
    compressed = torch.tensor(
        [[900.0], [901.0], [902.0], [903.0]], dtype=torch.bfloat16, device="cuda"
    ).requires_grad_(True)
    seq_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device="cuda")
    comp_ids = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([True, True, True, True], device="cuda")
    dummy = torch.empty((1,), dtype=torch.int32, device="cuda")
    capacity = 11

    ref = _native_thd_full_kv_pack(
        kv_local,
        boundary,
        compressed,
        seq_ids,
        comp_ids,
        valid,
        cu,
        global_start,
        l_local,
        d_window,
        ratio,
        capacity,
    )
    grad = torch.randn_like(ref)
    ref.backward(grad)
    ref_grads = (
        kv_local.grad.detach().clone(),
        boundary.grad.detach().clone(),
        compressed.grad.detach().clone(),
    )
    kv_local.grad.zero_()
    boundary.grad.zero_()
    compressed.grad.zero_()

    fused = csa_cp_kernels.ThdFullKvPack.apply(
        kv_local,
        boundary,
        compressed,
        seq_ids,
        comp_ids,
        valid,
        cu,
        dummy,
        cu,
        global_start,
        l_local,
        d_window,
        ratio,
        capacity,
        0,
        0,
        0,
        0,
        0,
    )
    fused.backward(grad)
    assert torch.equal(fused, ref)
    assert torch.equal(kv_local.grad, ref_grads[0])
    assert torch.equal(boundary.grad, ref_grads[1])
    assert torch.equal(compressed.grad, ref_grads[2])

    bench_cu = torch.tensor([0, 8192], dtype=torch.int32, device="cuda")
    bench_global_start = 2048
    bench_l_local = 4096
    bench_d_window = 128
    bench_ratio = 128
    bench_compressed_rows = int(bench_cu[-1]) // bench_ratio
    bench_capacity = bench_d_window + bench_l_local + bench_compressed_rows
    bench_kv = torch.randn(bench_l_local, 64, dtype=torch.bfloat16, device="cuda")
    bench_boundary = torch.randn(bench_d_window, 64, dtype=torch.bfloat16, device="cuda")
    bench_compressed = torch.randn(bench_compressed_rows, 64, dtype=torch.bfloat16, device="cuda")
    bench_seq_ids = torch.zeros(bench_compressed_rows, dtype=torch.int32, device="cuda")
    bench_comp_ids = torch.arange(bench_compressed_rows, dtype=torch.int32, device="cuda")
    bench_valid = torch.ones(bench_compressed_rows, dtype=torch.bool, device="cuda")
    bench_dummy = torch.empty((1,), dtype=torch.int32, device="cuda")
    bench_grad = torch.randn(bench_capacity, 64, dtype=torch.bfloat16, device="cuda")

    def run_cute():
        k = bench_kv.detach().clone().requires_grad_(True)
        b = bench_boundary.detach().clone().requires_grad_(True)
        c = bench_compressed.detach().clone().requires_grad_(True)
        y = csa_cp_kernels.ThdFullKvPack.apply(
            k,
            b,
            c,
            bench_seq_ids,
            bench_comp_ids,
            bench_valid,
            bench_cu,
            bench_dummy,
            bench_cu,
            bench_global_start,
            bench_l_local,
            bench_d_window,
            bench_ratio,
            bench_capacity,
            0,
            0,
            0,
            0,
            0,
        )
        y.backward(bench_grad)

    def run_torch():
        k = bench_kv.detach().clone().requires_grad_(True)
        b = bench_boundary.detach().clone().requires_grad_(True)
        c = bench_compressed.detach().clone().requires_grad_(True)
        y = _native_thd_full_kv_pack(
            k,
            b,
            c,
            bench_seq_ids,
            bench_comp_ids,
            bench_valid,
            bench_cu,
            bench_global_start,
            bench_l_local,
            bench_d_window,
            bench_ratio,
            bench_capacity,
        )
        y.backward(bench_grad)

    effective_bytes = 2 * _num_bytes(bench_kv, bench_boundary, bench_compressed, bench_grad)
    _print_perf(
        "thd_full_kv_pack_fwd_bwd", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )


def test_repack_compressed_kv_to_seq_major_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    rank_major = torch.arange(8 * 4, dtype=torch.bfloat16, device="cuda").reshape(8, 4)
    seq_ids = torch.tensor([0, 1, 0, -1, 1, 2, 1, 1], dtype=torch.int32, device="cuda")
    comp_ids = torch.tensor([0, 0, 1, -1, 1, 0, 1, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([True, True, True, False, True, True, True, True], device="cuda")
    cu_comp = torch.tensor([0, 2, 4, 5], dtype=torch.int32, device="cuda")
    ref = _native_repack_compressed_kv_to_seq_major(
        rank_major, seq_ids, comp_ids, valid, cu_comp, 5
    )
    fused = csa_cp_kernels.repack_compressed_kv_to_seq_major(
        rank_major, seq_ids, comp_ids, valid, cu_comp, 5
    )
    assert torch.equal(fused[0], ref[0])
    assert torch.equal(fused[1], ref[1])

    bench_rows = 4096
    bench_width = 64
    bench_rank_major = torch.randn(bench_rows, bench_width, dtype=torch.bfloat16, device="cuda")
    bench_seq_ids = torch.zeros(bench_rows, dtype=torch.int32, device="cuda")
    bench_comp_ids = torch.arange(bench_rows, dtype=torch.int32, device="cuda")
    bench_valid = torch.ones(bench_rows, dtype=torch.bool, device="cuda")
    bench_cu_comp = torch.tensor([0, bench_rows], dtype=torch.int32, device="cuda")

    def run_cute():
        csa_cp_kernels.repack_compressed_kv_to_seq_major(
            bench_rank_major, bench_seq_ids, bench_comp_ids, bench_valid, bench_cu_comp, bench_rows
        )

    def run_torch():
        _native_repack_compressed_kv_to_seq_major(
            bench_rank_major, bench_seq_ids, bench_comp_ids, bench_valid, bench_cu_comp, bench_rows
        )

    effective_bytes = _num_bytes(bench_rank_major) * 2 + bench_rows * 4
    _print_perf(
        "repack_compressed_kv_to_seq_major",
        _time_cuda(run_cute),
        _time_cuda(run_torch),
        effective_bytes,
    )


def test_build_compressed_row_metadata_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    cu = torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda")
    cp_size = 4
    chunk_len = 8
    ratio = 4
    d_comp = 8
    c_cap_per_chunk = 4
    c_cap_per_rank = 10
    ref = _native_build_compressed_row_metadata(
        cu, cp_size, chunk_len, ratio, d_comp, c_cap_per_chunk, c_cap_per_rank, True
    )
    fused = csa_cp_kernels.build_compressed_row_metadata(
        cu,
        cp_size,
        chunk_len,
        ratio,
        d_comp,
        c_cap_per_chunk,
        c_cap_per_rank=c_cap_per_rank,
        use_two_chunk=True,
    )
    for actual, expected in zip(fused, ref):
        assert torch.equal(actual, expected)

    bench_cu = torch.tensor([0, 65536], dtype=torch.int32, device="cuda")
    bench_cp_size = 4
    bench_chunk_len = 8192
    bench_ratio = 128
    bench_d_comp = 128
    bench_c_cap = (bench_chunk_len + bench_d_comp) // bench_ratio
    bench_c_cap_per_rank = bench_c_cap * 2

    def run_cute():
        csa_cp_kernels.build_compressed_row_metadata(
            bench_cu,
            bench_cp_size,
            bench_chunk_len,
            bench_ratio,
            bench_d_comp,
            bench_c_cap,
            c_cap_per_rank=bench_c_cap_per_rank,
            use_two_chunk=True,
        )

    def run_torch():
        _native_build_compressed_row_metadata(
            bench_cu,
            bench_cp_size,
            bench_chunk_len,
            bench_ratio,
            bench_d_comp,
            bench_c_cap,
            bench_c_cap_per_rank,
            True,
        )

    effective_bytes = bench_cp_size * bench_c_cap_per_rank * (4 + 4 + 1)
    _print_perf(
        "build_compressed_row_metadata",
        _time_cuda(run_cute),
        _time_cuda(run_torch),
        effective_bytes,
    )


def test_build_indexer_topk_metadata_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    ratio = 4
    cu_q = torch.tensor([0, 6], dtype=torch.int32, device="cuda")
    cu_comp = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    k_seq = torch.arange(8, dtype=torch.bfloat16, device="cuda").reshape(4, 2)
    ref = _native_indexer_topk_metadata(k_seq, cu_q, cu_comp, 0, 8, ratio)
    fused = csa_cp_kernels.build_indexer_topk_metadata(k_seq, cu_q, cu_comp, 0, 8, ratio)
    for actual, expected in zip(fused, ref):
        assert torch.equal(actual, expected)

    bench_ratio = 128
    bench_l_local = 8192
    bench_rows = 512
    bench_k_seq = torch.randn(bench_rows, 64, dtype=torch.bfloat16, device="cuda")
    bench_cu_q = torch.tensor([0, bench_l_local], dtype=torch.int32, device="cuda")
    bench_cu_comp = torch.tensor([0, bench_rows], dtype=torch.int32, device="cuda")

    def run_cute():
        csa_cp_kernels.build_indexer_topk_metadata(
            bench_k_seq, bench_cu_q, bench_cu_comp, 0, bench_l_local, bench_ratio
        )

    def run_torch():
        _native_indexer_topk_metadata(
            bench_k_seq, bench_cu_q, bench_cu_comp, 0, bench_l_local, bench_ratio
        )

    effective_bytes = _num_bytes(bench_k_seq) * 2 + bench_l_local * 4
    _print_perf(
        "build_indexer_topk_metadata", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )


def test_build_attention_indices_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    fused = csa_cp_kernels.build_attention_indices(cu, 3, 3, 2, 2, 2, 2)
    expected_topk = torch.tensor(
        [[1, 2, 3, 4], [5, -1, -1, -1], [5, 6, 7, -1]], dtype=torch.int32, device="cuda"
    )
    expected_len = torch.tensor([4, 1, 3], dtype=torch.int32, device="cuda")
    assert torch.equal(fused[0], expected_topk)
    assert torch.equal(fused[1], expected_len)

    padded = csa_cp_kernels.build_attention_indices(
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
    bench_ref = _native_attention_indices_one_seq(
        bench_l_local, bench_window, bench_ratio, bench_compressed_width, torch.device("cuda")
    )

    def run_cute():
        csa_cp_kernels.build_attention_indices(
            bench_cu,
            0,
            bench_l_local,
            bench_window,
            bench_window,
            bench_ratio,
            bench_compressed_width,
        )

    def run_torch():
        _native_attention_indices_one_seq(
            bench_l_local, bench_window, bench_ratio, bench_compressed_width, torch.device("cuda")
        )

    effective_bytes = _num_bytes(*bench_ref)
    _print_perf(
        "build_attention_indices", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )


def test_build_indexer_loss_indices_matches_native_and_reports_bandwidth():
    _require_cute_cuda()
    cu = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
    cu_comp = torch.tensor([0, 2, 4], dtype=torch.int32, device="cuda")
    logical_ids = torch.tensor([[0, 1], [-1, -1], [0, 1]], dtype=torch.int32, device="cuda")
    rank_by_seq = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda")
    fused = csa_cp_kernels.build_indexer_loss_indices(
        cu, cu_comp, 3, 3, 2, 2, 2, logical_ids, rank_by_seq
    )
    expected_topk = torch.tensor(
        [[3, 4, 1, 2], [-1, -1, 5, -1], [7, 8, 5, 6]], dtype=torch.int32, device="cuda"
    )
    expected_rank = torch.tensor([[0, 1], [-1, -1], [2, 3]], dtype=torch.int32, device="cuda")
    assert torch.equal(fused[0], expected_topk)
    assert torch.equal(fused[1], expected_rank)

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
        bench_l_local, bench_window, bench_ratio, bench_logical, bench_rank_by_seq
    )

    def run_cute():
        csa_cp_kernels.build_indexer_loss_indices(
            bench_cu,
            bench_cu_comp,
            0,
            bench_l_local,
            bench_window,
            bench_window,
            bench_ratio,
            bench_logical,
            bench_rank_by_seq,
        )

    def run_torch():
        _native_indexer_loss_indices_one_seq(
            bench_l_local, bench_window, bench_ratio, bench_logical, bench_rank_by_seq
        )

    effective_bytes = _num_bytes(bench_logical, bench_rank_by_seq, *bench_ref)
    _print_perf(
        "build_indexer_loss_indices", _time_cuda(run_cute), _time_cuda(run_torch), effective_bytes
    )
