# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the PyTorch QSA indexer and sparse-GQA references."""

import math

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops.qsa_stable_topk import (
    qsa_stable_topk_indices,
)
from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSAIndexer,
    compute_block_scores,
    qsa_block_causal_mask,
    qsa_raw_block_logits,
    select_qsa_token_ids,
    unfused_qsa_gqa_attention,
)
from megatron.core.transformer.experimental_attention_variant.qsa_indexer_loss import (
    qsa_indexer_kl_loss,
    qsa_teacher_token_distribution,
)

_INDEXER_SETTINGS = dict(
    hidden_size=12,
    num_query_heads=3,
    num_key_heads=1,
    head_dim=4,
    compress_ratio=2,
    token_budget=4,
    norm_epsilon=1e-6,
    attention_scaling=0.75,
    params_dtype=torch.float32,
    query_chunk_size=3,
)


def _indexer(**overrides) -> QSAIndexer:
    settings = dict(_INDEXER_SETTINGS)
    settings.update(overrides)
    return QSAIndexer(**settings)


def _indexer_inputs(seed: int = 23) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    hidden_states = torch.randn(14, 2, _INDEXER_SETTINGS["hidden_size"])
    rotary_angles = torch.randn(14, 1, 1, _INDEXER_SETTINGS["head_dim"])
    return hidden_states, rotary_angles


def test_block_score_equation_and_scale() -> None:
    torch.manual_seed(31)
    batch, queries, heads, blocks, head_dim = 2, 5, 3, 4, 8
    index_query = torch.randn(batch, queries, heads, head_dim, dtype=torch.float64)
    compressed_key = torch.randn(batch, blocks, 1, head_dim, dtype=torch.float64)

    raw = qsa_raw_block_logits(index_query, compressed_key[:, :, 0])
    scaled = compute_block_scores(index_query, compressed_key)
    expected = torch.zeros_like(raw)
    for batch_idx in range(batch):
        for query_idx in range(queries):
            for block_idx in range(blocks):
                expected[batch_idx, query_idx, block_idx] = sum(
                    max(
                        0.0,
                        float(
                            index_query[batch_idx, query_idx, head_idx]
                            @ compressed_key[batch_idx, block_idx, 0]
                        ),
                    )
                    for head_idx in range(heads)
                )

    torch.testing.assert_close(raw, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(scaled, expected / math.sqrt(head_dim), rtol=1e-12, atol=1e-12)
    assert raw.dtype == scaled.dtype == torch.float64
    assert not torch.allclose(raw, scaled)


def test_stable_topk_breaks_equal_scores_by_ascending_block_id() -> None:
    scores = torch.tensor([[4.0, 2.0, 2.0, 3.0, 2.0]])
    expected = torch.tensor([[0, 3, 1, 2]])
    assert torch.equal(qsa_stable_topk_indices(scores, 4), expected)


def test_routes_are_causal_deterministic_and_respect_right_padding() -> None:
    sequence_length, ratio, token_budget = 20, 2, 4
    query = torch.zeros(2, sequence_length, 3, 4)
    key = torch.zeros(2, sequence_length // ratio, 1, 4)
    lengths = torch.tensor([sequence_length, 13])

    first = select_qsa_token_ids(
        query, key, lengths, token_budget=token_budget, compress_ratio=ratio, query_chunk_size=5
    )
    second = select_qsa_token_ids(
        query, key, lengths, token_budget=token_budget, compress_ratio=ratio, query_chunk_size=7
    )

    assert torch.equal(first, second)
    assert torch.equal(first[0, -1, :token_budget], torch.arange(token_budget, dtype=torch.int32))
    positions = torch.arange(sequence_length).view(1, -1, 1)
    assert not bool(((first >= 0) & (first > positions)).any())
    assert bool((first[1, lengths[1] :] == -1).all())


def test_block_causal_mask_excludes_incomplete_and_padded_blocks() -> None:
    mask = qsa_block_causal_mask(
        10, 2, torch.tensor([10, 7]), compress_ratio=4, device=torch.device("cpu")
    )
    assert not bool(mask[:, :3].any())
    assert bool(mask[0, 3:, 0].all())
    assert bool(mask[0, 7:, 1].all())
    assert not bool(mask[1, :, 1].any())
    assert not bool(mask[1, 7:].any())


def test_partial_rope_passes_suffix_and_rotates_block_at_start() -> None:
    head_dim, rotary_dim, ratio = 16, 8, 4
    sequence_length, hidden_size, heads = 8, 12, 3
    indexer = QSAIndexer(
        hidden_size=hidden_size,
        num_query_heads=heads,
        num_key_heads=1,
        head_dim=head_dim,
        compress_ratio=ratio,
        token_budget=8,
        params_dtype=torch.float32,
    )
    torch.manual_seed(37)
    hidden_states = torch.randn(sequence_length, 1, hidden_size)
    rotary_angles = torch.randn(sequence_length, 1, 1, rotary_dim)
    query, pooled_key = indexer.index_states(hidden_states, rotary_angles)

    projected = indexer.index_qk_proj(hidden_states)
    raw_query, raw_key = torch.split(projected, [heads * head_dim, head_dim], dim=-1)
    raw_query = indexer.q_layernorm(
        raw_query.view(sequence_length, 1, heads, head_dim).permute(1, 0, 2, 3)
    )
    raw_key = raw_key.view(sequence_length, 1, 1, head_dim).permute(1, 0, 2, 3)
    pooled = indexer.k_layernorm(raw_key.reshape(1, 2, ratio, head_dim).mean(dim=2).unsqueeze(2))

    angles = rotary_angles.reshape(sequence_length, rotary_dim)
    cos = torch.cos(angles) * indexer.attention_scaling
    sin = torch.sin(angles) * indexer.attention_scaling

    def rotate(states: torch.Tensor, cos_: torch.Tensor, sin_: torch.Tensor) -> torch.Tensor:
        first, second = states.chunk(2, dim=-1)
        return states * cos_ + torch.cat((-second, first), dim=-1) * sin_

    torch.testing.assert_close(query[..., rotary_dim:], raw_query[..., rotary_dim:])
    torch.testing.assert_close(pooled_key[..., rotary_dim:], pooled[..., rotary_dim:])
    starts = torch.arange(2) * ratio
    expected = rotate(
        pooled[..., :rotary_dim], cos[starts][None, :, None, :], sin[starts][None, :, None, :]
    )
    torch.testing.assert_close(pooled_key[..., :rotary_dim], expected)
    ends = starts + ratio - 1
    wrong = rotate(
        pooled[..., :rotary_dim], cos[ends][None, :, None, :], sin[ends][None, :, None, :]
    )
    assert not torch.allclose(pooled_key[..., :rotary_dim], wrong)


def test_unfused_gqa_matches_direct_per_head_reference_and_backpropagates() -> None:
    torch.manual_seed(41)
    query = torch.randn(1, 3, 4, 3, dtype=torch.float64, requires_grad=True)
    key = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 4, 2, 3, dtype=torch.float64, requires_grad=True)
    routes = torch.tensor([[[0, -1, -1], [0, 1, -1], [0, 1, 2]]], dtype=torch.int32)

    output = unfused_qsa_gqa_attention(query, key, value, routes)
    expected = torch.zeros_like(output)
    for query_idx in range(query.size(1)):
        ids = routes[0, query_idx]
        ids = ids[ids >= 0].long()
        for head_idx in range(query.size(2)):
            kv_head = head_idx // 2
            scores = query[0, query_idx, head_idx] @ key[0, ids, kv_head].T
            probabilities = torch.softmax(scores / math.sqrt(query.size(-1)), dim=-1)
            expected[0, query_idx, head_idx] = probabilities @ value[0, ids, kv_head]

    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert bool(tensor.grad.abs().sum() > 0)


def test_route_and_score_runs_one_projection_and_keeps_topk_detached() -> None:
    indexer = _indexer(qsa_indexer_loss_coeff=0.05)
    hidden_states, rotary_angles = _indexer_inputs()
    calls = 0

    def count_projection(module, args, output) -> None:
        nonlocal calls
        calls += 1

    handle = indexer.index_qk_proj.register_forward_hook(count_projection)
    output = indexer.route_and_score(hidden_states, rotary_angles)
    handle.remove()

    assert calls == 1
    assert output.block_logits is not None and output.block_logits.requires_grad
    assert output.support_mask is not None and not output.support_mask.requires_grad
    assert not output.routes.requires_grad

    teacher = qsa_teacher_token_distribution(torch.softmax(torch.randn(2, 3, 14, 14), dim=-1))
    loss = qsa_indexer_kl_loss(
        output.block_logits,
        teacher,
        output.support_mask,
        compress_ratio=indexer.compress_ratio,
        loss_coeff=indexer.qsa_indexer_loss_coeff,
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in indexer.parameters())


def test_zero_loss_coefficient_freezes_indexer_and_skips_loss_tensors() -> None:
    indexer = _indexer()
    hidden_states, rotary_angles = _indexer_inputs()
    output = indexer.route_and_score(hidden_states, rotary_angles)

    assert all(not parameter.requires_grad for parameter in indexer.parameters())
    assert output.block_logits is None
    assert output.support_mask is None
    assert not output.routes.requires_grad


def test_sparse_loss_flag_changes_only_support() -> None:
    hidden_states, rotary_angles = _indexer_inputs()
    sparse = _indexer(qsa_indexer_loss_coeff=0.05, qsa_indexer_use_sparse_loss=True)
    dense = _indexer(qsa_indexer_loss_coeff=0.05, qsa_indexer_use_sparse_loss=False)
    dense.load_state_dict(sparse.state_dict(), strict=True)

    sparse_output = sparse.route_and_score(hidden_states, rotary_angles)
    dense_output = dense.route_and_score(hidden_states, rotary_angles)
    assert torch.equal(sparse_output.routes, dense_output.routes)
    assert bool((sparse_output.support_mask & ~dense_output.support_mask).sum() == 0)
    assert int(sparse_output.support_mask.sum()) < int(dense_output.support_mask.sum())


@pytest.mark.parametrize("coefficient", [-1e-9, float("nan"), float("inf")])
def test_invalid_loss_coefficients_are_rejected(coefficient: float) -> None:
    with pytest.raises(ValueError, match="qsa_indexer_loss_coeff"):
        _indexer(qsa_indexer_loss_coeff=coefficient)


def test_boolean_loss_coefficient_is_rejected() -> None:
    with pytest.raises(TypeError, match="qsa_indexer_loss_coeff"):
        _indexer(qsa_indexer_loss_coeff=True)
