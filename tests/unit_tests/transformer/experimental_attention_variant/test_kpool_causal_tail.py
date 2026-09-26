# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa import (
    _compute_index_scores,
    _kpool_compress_keys,
    _kpool_fp8_input,
    rotate_activation,
    fused_qk_topk_kpool,
)
from megatron.core.transformer.experimental_attention_variant.dsa_masking import (
    generate_varlen_mask_params_for_positions,
)


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


def test_kpool_rotate_activation_rotates_q_and_compressed_k_together():
    torch.manual_seed(789)
    q = torch.randn(4, 1, 2, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(6, 1, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(4, 1, 2, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn_like(k)
    ape = torch.randn(2, 8, device="cuda", dtype=torch.float32)

    scores, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=4,
        pool_size=2,
        gate_score=gate,
        ape=ape,
        use_relu=False,
        always_select_tail=False,
        rotate_activation_enabled=True,
    )
    pooled_k = _kpool_compress_keys(k, gate, ape, pool_size=2)
    expected_scores = _compute_index_scores(
        rotate_activation(q), weights, rotate_activation(pooled_k), use_relu=False
    )
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    expected_pool_ids = expected_scores.topk(2, dim=-1).indices
    expected_indices = (
        expected_pool_ids.unsqueeze(-1) * 2 + torch.arange(2, device="cuda")
    ).reshape(1, 4, 4)
    torch.testing.assert_close(indices, expected_indices.to(indices.dtype))


@pytest.mark.parametrize("batched_mask", [False, True])
def test_kpool_explicit_token_mask_filters_pool_and_tail_tokens(batched_mask):
    torch.manual_seed(790)
    q = torch.randn(3, 2, 1, 8, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(6, 2, 8, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(3, 2, 1, device="cuda", dtype=torch.bfloat16)
    mask = torch.triu(
        torch.full((3, 6), float("-inf"), device="cuda", dtype=torch.float32), diagonal=1
    )
    mask[:, 1] = float("-inf")
    if batched_mask:
        mask = mask.unsqueeze(0).expand(2, -1, -1).contiguous()

    _, indices = fused_qk_topk_kpool(
        q,
        k,
        weights,
        index_topk=4,
        pool_size=2,
        gate_score=torch.zeros_like(k),
        ape=torch.zeros(2, 8, device="cuda"),
        mask=mask,
        use_relu=False,
        always_select_tail=True,
    )
    assert not torch.any(indices == 1)
    assert not torch.any(indices == 3)
    assert not torch.any(indices == 4)
    assert not torch.any(indices == 5)
