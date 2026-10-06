# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import (
    _compute_index_scores,
    _kpool_compress_keys,
    _kpool_compress_keys_per_seg,
    _kpool_fp8_input,
    fused_qk_topk_kpool,
    rotate_activation,
)
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    generate_varlen_mask_params_for_positions,
)

try:
    from fast_hadamard_transform import hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    hadamard_transform = None
    HAVE_HADAMARD = False


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return x * scale


@pytest.fixture(autouse=True)
def patch_hadamard_if_needed():
    if not HAVE_HADAMARD:
        with patch(
            "megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform",
            _mock_hadamard_transform,
        ):
            yield
    else:
        yield


@pytest.mark.parametrize("lengths", [(8,), (7,), (3,), (3, 6), (5, 9, 3)])
@pytest.mark.parametrize("query_stride", [1, 2])
@pytest.mark.parametrize("explicit_key_positions", [False, True])
@pytest.mark.parametrize("fp8_indexer", [False, True])
def test_kpool_preserves_each_query_causal_tail(
    lengths, query_stride, explicit_key_positions, fp8_indexer
):
    torch.manual_seed(123)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
    total = sum(lengths)
    positions = torch.arange(0, total, query_stride, device="cuda")
    starts, ends = generate_varlen_mask_params_for_positions(cu, positions)
    q = torch.randn(len(positions), 1, 2, 8, device="cuda")
    k = torch.randn(total, 1, 8, device="cuda")
    weights = torch.ones(len(positions), 1, 2, device="cuda")
    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=16,
        pool_size=4,
        gate_score=torch.zeros_like(k),
        ape=torch.zeros(4, 8, device="cuda"),
        varlen_starts=starts,
        varlen_ends=ends,
        key_positions=torch.arange(total, device="cuda") if explicit_key_positions else None,
        cu_seqlens_kv=cu,
        fp8_indexer=fp8_indexer,
    )
    assert indices.shape == (1, len(positions), 19)
    for row, start, end in zip(indices[0], starts.tolist(), ends.tolist()):
        # Below the pool budget, sparse attention must contain the full causal prefix.
        actual = row[row >= 0].sort().values
        expected = torch.arange(start, end, device="cuda", dtype=actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("input_scale", [0.0, 1e-5, 1.0, 1000.0])
def test_kpool_fp8_input_matches_hadamard_matrix_reference(input_scale):
    torch.manual_seed(456)
    x = (torch.randn(33, 128, device="cuda") * input_scale).to(torch.bfloat16)
    matrix = torch.ones(1, 1, device="cuda")
    for _ in range(7):
        matrix = torch.cat((torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0)
    rotated = (x.float() @ matrix / 128**0.5).to(torch.bfloat16).float()
    scale = torch.exp2(
        torch.ceil(torch.log2(rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    )
    expected = (rotated / scale).to(torch.float8_e4m3fn).float() * scale
    torch.testing.assert_close(_kpool_fp8_input(x), expected, rtol=0, atol=0)


def test_kpool_packed_compression_keeps_fixed_shape_and_matches_reference():
    torch.manual_seed(457)
    lengths = [7, 5, 9]
    total = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
    k = torch.randn(total, 1, 8, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(k)
    ape = torch.randn(4, 8, device="cuda", dtype=torch.float32)

    pooled, bases = _kpool_compress_keys_per_seg(k, gate, ape, 4, cu)
    expected = torch.cat(
        [_kpool_compress_keys(k[s:e], gate[s:e], ape, 4) for s, e in zip(cu[:-1], cu[1:]) if e > s],
        dim=0,
    )
    torch.testing.assert_close(pooled[: expected.size(0)], expected, rtol=0, atol=0)
    assert pooled.shape[0] == total // 4
    assert bases.tolist() == [0, 7, 12, 16, -1]


def test_kpool_explicit_mask_tail_uses_global_query_positions():
    torch.manual_seed(458)
    query_positions = [0, 1, 6, 7]
    q = torch.randn(len(query_positions), 1, 1, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 1, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(len(query_positions), 1, 1, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((len(query_positions), 8), float("-inf"), device="cuda")
    for row, position in enumerate(query_positions):
        mask[row, : position + 1] = 0

    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=4,
        pool_size=4,
        gate_score=torch.zeros_like(k),
        ape=torch.zeros(4, 8, device="cuda"),
        mask=mask,
        always_select_tail=True,
    )
    assert indices.shape == (1, len(query_positions), 7)
    torch.testing.assert_close(
        indices[0, 2], torch.arange(7, device="cuda", dtype=indices.dtype), rtol=0, atol=0
    )
    assert not torch.any(indices[0, :3] == 7)


@pytest.mark.parametrize("valid_range", [(1, 6), (2, 9), (5, 9)])
def test_kpool_explicit_mask_tail_handles_non_aligned_left_padding(valid_range):
    start, end = valid_range
    q = torch.randn(1, 1, 1, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(9, 1, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(1, 1, 1, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((1, 9), float("-inf"), device="cuda")
    mask[:, start:end] = 0

    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=8,
        pool_size=4,
        gate_score=torch.zeros_like(k),
        ape=torch.zeros(4, 8, device="cuda"),
        mask=mask,
        always_select_tail=True,
        rotate_activation_enabled=False,
    )
    selected = indices[0, 0]
    selected = selected[selected >= 0]
    expected = torch.arange(start, end, device="cuda", dtype=selected.dtype)
    torch.testing.assert_close(selected.sort().values, expected, rtol=0, atol=0)


def test_kpool_explicit_batched_mask_tail_handles_non_aligned_left_padding():
    q = torch.randn(1, 2, 1, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(9, 2, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(1, 2, 1, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((2, 1, 9), float("-inf"), device="cuda")
    mask[0, 0, 2:9] = 0
    mask[1, 0, 5:9] = 0

    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=8,
        pool_size=4,
        gate_score=torch.zeros_like(k),
        ape=torch.zeros(4, 8, device="cuda"),
        mask=mask,
        always_select_tail=True,
        rotate_activation_enabled=False,
    )
    for row, (start, end) in zip(indices[:, 0], ((2, 9), (5, 9))):
        selected = row[row >= 0]
        expected = torch.arange(start, end, device="cuda", dtype=selected.dtype)
        torch.testing.assert_close(selected.sort().values, expected, rtol=0, atol=0)


def test_kpool_explicit_batched_mask_filters_invalid_tokens():
    q = torch.randn(4, 2, 1, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 2, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(4, 2, 1, device="cuda", dtype=torch.bfloat16)
    mask = torch.full((2, 4, 8), float("-inf"), device="cuda")
    mask[:, :, :4] = 0
    mask[0, 2, 4:7] = 0
    mask[1, 3, 4:8] = 0

    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=4,
        pool_size=4,
        gate_score=torch.zeros_like(k),
        ape=torch.zeros(4, 8, device="cuda"),
        mask=mask,
    )
    selected_valid = torch.gather(torch.isfinite(mask), -1, indices.clamp_min(0).to(torch.long))
    assert torch.all((indices < 0) | selected_valid)


def test_kpool_rotation_applies_to_query_and_pooled_key():
    torch.manual_seed(459)
    q = torch.randn(4, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(8, 1, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(4, 1, 2, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(k)
    ape = torch.randn(4, 8, device="cuda", dtype=torch.float32)
    scores, _ = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=4,
        pool_size=4,
        gate_score=gate,
        ape=ape,
        use_relu=False,
        always_select_tail=False,
        rotate_activation_enabled=True,
    )
    pooled = _kpool_compress_keys(k, gate, ape, 4)
    expected = _compute_index_scores(
        rotate_activation(q), weights, rotate_activation(pooled), use_relu=False
    )
    torch.testing.assert_close(scores, expected, rtol=0, atol=0)


@pytest.mark.parametrize("mask_mode", ["none", "explicit_2d", "explicit_3d", "varlen", "packed"])
@pytest.mark.parametrize("use_relu", [False, True])
@pytest.mark.parametrize("always_select_tail", [False, True])
@pytest.mark.parametrize("input_mode", ["none", "fp8", "rotate"])
def test_kpool_chunked_topk_matches_full_score_reference(
    mask_mode, use_relu, always_select_tail, input_mode
):
    torch.manual_seed(461)
    sq, sk, batch, heads, dim, pool_size = 9, 16, 2, 3, 8, 4
    if mask_mode == "packed":
        batch = 1
    q = torch.randn(sq, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(sk, batch, dim, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(sq, batch, heads, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(k)
    ape = torch.randn(pool_size, dim, device="cuda")
    kwargs = {}
    if mask_mode.startswith("explicit"):
        mask = torch.full((batch, sq, sk), float("-inf"), device="cuda")
        for bi in range(batch):
            for qi in range(sq):
                start = (qi + bi) % pool_size
                end = min(sk, start + qi + 1)
                mask[bi, qi, start:end] = 0
        kwargs["mask"] = mask[0] if mask_mode == "explicit_2d" else mask
    elif mask_mode in ("varlen", "packed"):
        kwargs["varlen_starts"] = torch.tensor([0] * 7 + [7] * 2, device="cuda", dtype=torch.int64)
        kwargs["varlen_ends"] = torch.arange(1, sq + 1, device="cuda", dtype=torch.int64)
        if mask_mode == "packed":
            kwargs["cu_seqlens_kv"] = torch.tensor([0, 7, sk], device="cuda")

    args = (q, k, weights, 8, pool_size, gate, ape)
    with patch(
        "megatron.core.transformer.experimental_attention_variant.dsa._KPOOL_SCORE_CHUNK_BYTES", 128
    ):
        with torch.no_grad():
            scores, expected = fused_qk_topk_kpool(
                *args,
                use_relu=use_relu,
                always_select_tail=always_select_tail,
                fp8_indexer=input_mode == "fp8",
                rotate_activation_enabled=input_mode == "rotate",
                return_index_scores=True,
                **kwargs,
            )
            no_scores, actual = fused_qk_topk_kpool(
                *args,
                use_relu=use_relu,
                always_select_tail=always_select_tail,
                fp8_indexer=input_mode == "fp8",
                rotate_activation_enabled=input_mode == "rotate",
                return_index_scores=False,
                **kwargs,
            )
    assert scores is not None and no_scores is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_kpool_chunked_topk_with_no_complete_pool():
    q = torch.randn(3, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 1, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(3, 1, 2, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(k)
    ape = torch.randn(4, 8, device="cuda")
    args = (q, k, weights, 4, 4, gate, ape)
    _, expected = fused_qk_topk_kpool(*args)
    scores, actual = fused_qk_topk_kpool(*args, return_index_scores=False)
    assert scores is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_compute_index_scores_preserves_autograd_graph():
    torch.manual_seed(460)
    q = torch.randn(3, 1, 2, 4, requires_grad=True)
    k = torch.randn(5, 1, 4, requires_grad=True)
    weights = torch.randn(3, 1, 2, requires_grad=True)

    scores = _compute_index_scores(q, weights, k)
    assert scores.requires_grad
    scores.sum().backward()
    assert q.grad is not None
    assert weights.grad is not None
    assert k.grad is not None
