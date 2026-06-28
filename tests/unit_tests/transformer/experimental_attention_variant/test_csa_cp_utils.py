# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

import megatron.core.parallel_state as parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.experimental_attention_variant import csa_cp_utils
from megatron.core.transformer.experimental_attention_variant.csa_cp_utils import (
    apply_thd_cp_local_rope_fused,
    apply_thd_cp_local_rope_unfused,
    compute_cp_indexer_topk,
    exchange_cp_boundary_hidden,
    prepare_cp_compressor_input,
)
from tests.unit_tests.test_utilities import Utils


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("DSv4 CP CUDA utility tests require CUDA.")


def _rope_reference(x, cos, sin, positions, nope_dim, pos_dim, inverse=False):
    x_nope, x_pos = torch.split(x, [nope_dim, pos_dim], dim=-1)
    cos_pos = cos.index_select(0, positions).view(x.shape[0], 1, pos_dim)
    sin_pos = sin.index_select(0, positions).view(x.shape[0], 1, pos_dim)
    if inverse:
        sin_pos = -sin_pos
    half = pos_dim // 2
    x1, x2 = x_pos[..., 0::2], x_pos[..., 1::2]
    rotated = torch.stack(
        (
            x1 * cos_pos[..., :half] - x2 * sin_pos[..., :half],
            x2 * cos_pos[..., half:] + x1 * sin_pos[..., half:],
        ),
        dim=-1,
    ).flatten(-2)
    return torch.cat((x_nope, rotated), dim=-1)


def _sequence_positions(cu_seqlens, rows):
    seq_ids = torch.bucketize(rows, cu_seqlens[1:], right=True).clamp_max(cu_seqlens.shape[0] - 2)
    starts = cu_seqlens[seq_ids]
    ends = cu_seqlens[seq_ids + 1]
    return torch.where((rows >= starts) & (rows < ends), rows - starts, 0)


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

        boundary = exchange_cp_boundary_hidden(local, 0, d_window, cp_group)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_apply_thd_cp_local_rope_matches_reference_forward_backward(dtype, inverse):
    _require_cuda()
    torch.manual_seed(11)
    cu = torch.tensor([0, 4, 12], dtype=torch.int32, device="cuda")
    global_start = 2
    global_rows = torch.arange(2, 6, dtype=torch.int32, device="cuda")
    positions = _sequence_positions(cu, global_rows)
    nope_dim = pos_dim = 4
    x = torch.randn(4, 2, 8, dtype=dtype, device="cuda")
    cos = torch.randn(8, pos_dim, dtype=dtype, device="cuda")
    sin = torch.randn(8, pos_dim, dtype=dtype, device="cuda")

    ref_x = x.detach().clone().requires_grad_(True)
    expected = _rope_reference(ref_x, cos, sin, positions, nope_dim, pos_dim, inverse)
    grad = torch.randn_like(expected)
    expected.backward(grad)

    actual_x = x.detach().clone().requires_grad_(True)
    actual = apply_thd_cp_local_rope_fused(
        actual_x, cos, sin, nope_dim, pos_dim, cu, global_start, inverse=inverse
    )
    actual.backward(grad)
    rtol, atol = (1e-5, 1e-5) if dtype == torch.float32 else (2e-2, 5e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_x.grad, ref_x.grad, rtol=rtol, atol=atol)


def test_apply_thd_cp_local_rope_maps_invalid_boundary_rows_to_position_zero():
    _require_cuda()
    cu = torch.tensor([0, 4, 12], dtype=torch.int32, device="cuda")
    global_start = -2
    global_rows = torch.arange(-2, 14, dtype=torch.int32, device="cuda")
    positions = _sequence_positions(cu, global_rows)
    x = torch.randn(global_rows.shape[0], 1, 8, device="cuda")
    cos = torch.randn(8, 4, device="cuda")
    sin = torch.randn(8, 4, device="cuda")
    expected = _rope_reference(x, cos, sin, positions, 4, 4)
    actual = apply_thd_cp_local_rope_fused(x, cos, sin, 4, 4, cu, global_start)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("inverse", [False, True])
def test_apply_thd_cp_local_rope_unfused_matches_explicit_positions(inverse):
    _require_cuda()
    torch.manual_seed(17)
    cu = torch.tensor([0, 3, 9], dtype=torch.int32, device="cuda")
    global_start = 2
    global_rows = torch.arange(2, 7, dtype=torch.int32, device="cuda")
    positions = _sequence_positions(cu, global_rows)
    x = torch.randn(5, 2, 8, device="cuda", requires_grad=True)
    freqs = torch.randn(6, 1, 1, 4, device="cuda")
    config = SimpleNamespace(
        apply_rope_fusion=False, rotary_interleaved=False, multi_latent_attention=False
    )

    expected_x = x.detach().clone().requires_grad_(True)
    expected = _rope_reference(
        expected_x, torch.cos(freqs[:, 0, 0]), torch.sin(freqs[:, 0, 0]), positions, 4, 4, inverse
    )
    grad = torch.randn_like(expected)
    expected.backward(grad)

    actual_x = x.detach().clone().requires_grad_(True)
    actual = apply_thd_cp_local_rope_unfused(
        actual_x, freqs, 4, 4, cu, global_start, config, inverse=inverse
    )
    actual.backward(grad)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_x.grad, expected_x.grad)


def test_prepare_cp_compressor_input_builds_rank_row_map(monkeypatch):
    calls = []

    def fake_apply(hidden, boundary, cu, global_start, ratio, d_comp, c_cap):
        calls.append((int(global_start), hidden.shape[0], int(c_cap), tuple(boundary.shape)))
        marker = len(calls)
        hidden_compact = torch.full((int(c_cap) * int(ratio), 1), marker, dtype=hidden.dtype)
        comp_ids = torch.arange(int(c_cap), dtype=torch.int32) + marker * 10
        return hidden_compact, comp_ids

    monkeypatch.setattr(
        csa_cp_utils.csa_cp_layout_kernels.CompressorInputCompact, "apply", staticmethod(fake_apply)
    )
    hidden = torch.arange(16, dtype=torch.float32).reshape(16, 1)
    boundary = torch.arange(2, dtype=torch.float32).reshape(2, 1)
    cu = torch.tensor([0, 32], dtype=torch.int32)
    cu_comp = torch.tensor([0, 8], dtype=torch.int32)

    hidden_compact, comp_ids, rank_rows = prepare_cp_compressor_input(
        hidden, boundary, cu, cu_comp, 0, cp_size=2, ratio=4
    )

    assert calls == [(0, 16, 8, (2, 1))]
    assert hidden_compact.shape == (32, 1)
    assert torch.equal(comp_ids, torch.arange(10, 18, dtype=torch.int32))
    assert torch.equal(rank_rows, torch.tensor([0, 1, 2, 3, 10, 11, 12, 13], dtype=torch.int32))


def test_compute_cp_indexer_topk_passes_valid_lengths(monkeypatch):
    topk_calls = []

    def fake_indexer_topk(
        q, k, _weights, *, topk, cu_seqlens_q, cu_seqlens_kv, valid_k_lengths, **_
    ):
        topk_calls.append(
            (k.clone(), cu_seqlens_q.clone(), cu_seqlens_kv.clone(), valid_k_lengths.clone())
        )
        return torch.full((q.shape[0], int(topk)), len(topk_calls), dtype=torch.int32), None

    monkeypatch.setattr(csa_cp_utils, "indexer_topk", fake_indexer_topk)

    q = torch.randn(5, 2)
    weights = torch.randn(5, 1)
    k_seq = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    cu_q = torch.tensor([0, 10], dtype=torch.int32)
    cu_comp = torch.tensor([0, 3], dtype=torch.int32)

    out = compute_cp_indexer_topk(
        q,
        weights,
        k_seq,
        cu_q,
        cu_comp,
        8,
        ratio=4,
        topk_width=2,
        indexer_softmax_scale=0.5,
        max_seqlen_q=10,
        use_fused=True,
    )

    assert torch.equal(out, torch.ones(5, 2, dtype=torch.int32))
    assert torch.equal(topk_calls[0][0][:2], k_seq[:2])
    assert torch.count_nonzero(topk_calls[0][0][2:]) == 0
    assert torch.equal(topk_calls[0][1], torch.tensor([0, 2, 5], dtype=torch.int32))
    assert torch.equal(topk_calls[0][2], torch.tensor([0, 2, 2], dtype=torch.int32))
    assert torch.equal(topk_calls[0][3], torch.tensor([2, 2, 0, 0, 0], dtype=torch.int32))
    assert (
        compute_cp_indexer_topk(q, weights, k_seq[:0], cu_q, cu_comp, 2, 4, 2, 1.0, 10, True)
        is None
    )


def test_compute_cp_indexer_topk_unfused_uses_exact_global_positions(monkeypatch):
    def fail_if_fused(*_args, **_kwargs):
        raise AssertionError("unfused top-k must not call the fused kernel")

    monkeypatch.setattr(csa_cp_utils, "indexer_topk", fail_if_fused)
    torch.manual_seed(19)
    q = torch.randn(8, 2, 3)
    weights = torch.randn(8, 2)
    k = torch.randn(4, 3)
    cu_q = torch.tensor([0, 5, 13, 20], dtype=torch.int32)
    cu_k = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
    ratio = 4
    topk_width = 3
    scale = 0.7

    actual = compute_cp_indexer_topk(
        q,
        weights,
        k,
        cu_q,
        cu_k,
        global_start=7,
        ratio=ratio,
        topk_width=topk_width,
        indexer_softmax_scale=scale,
        max_seqlen_q=8,
        use_fused=False,
    )

    expected = torch.full((q.shape[0], topk_width), -1, dtype=torch.int32)
    for local_row, global_row in enumerate(range(7, 15)):
        seq = next(i for i in range(len(cu_q) - 1) if global_row < int(cu_q[i + 1]))
        visible = min((global_row - int(cu_q[seq]) + 1) // ratio, int(cu_k[seq + 1] - cu_k[seq]))
        candidates = []
        for local_k in range(visible):
            k_row = int(cu_k[seq]) + local_k
            score = 0.0
            for head in range(q.shape[1]):
                dot = torch.dot(q[local_row, head], k[k_row]).item()
                score += max(dot, 0.0) * weights[local_row, head].item() * scale
            candidates.append((score, local_k))
        candidates.sort(reverse=True)
        for column, (_, local_k) in enumerate(candidates[:topk_width]):
            expected[local_row, column] = local_k

    assert torch.equal(actual, expected)
