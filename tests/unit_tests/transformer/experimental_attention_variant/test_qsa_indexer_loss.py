# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Correctness tests for the QSA indexer KL objective."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.qsa import (
    QSAIndexer,
    compute_block_scores,
    qsa_block_causal_mask,
    qsa_topk_block_support,
)
from megatron.core.transformer.experimental_attention_variant.qsa_indexer_loss import (
    qsa_indexer_kl_loss,
    qsa_maxpool_teacher_to_blocks,
    qsa_teacher_token_distribution,
)


def _teacher(batch, heads, queries, keys, dtype=torch.float64, seed=5):
    torch.manual_seed(seed)
    return torch.softmax(torch.randn(batch, heads, queries, keys, dtype=dtype), dim=-1)


def test_teacher_pipeline_sums_heads_then_max_pools_blocks() -> None:
    """Eq. (17): head sum, token L1, then block MaxPool -- not sum pooling."""
    batch, heads, queries, keys, ratio = 2, 3, 6, 8, 4
    probabilities = _teacher(batch, heads, queries, keys)

    tokens = qsa_teacher_token_distribution(probabilities)
    expected_tokens = probabilities.sum(dim=1)
    expected_tokens = expected_tokens / expected_tokens.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(tokens, expected_tokens)
    torch.testing.assert_close(tokens.sum(-1), torch.ones(batch, queries, dtype=torch.float64))

    num_blocks = keys // ratio
    blocks = qsa_maxpool_teacher_to_blocks(tokens, compress_ratio=ratio, num_blocks=num_blocks)
    expected_blocks = tokens.unflatten(-1, (num_blocks, ratio)).amax(-1)
    torch.testing.assert_close(blocks, expected_blocks)

    # The whole point of MaxPool: it must not agree with sum pooling.
    sum_pooled = tokens.unflatten(-1, (num_blocks, ratio)).sum(-1)
    assert not torch.allclose(blocks, sum_pooled)


def test_incomplete_tail_tokens_never_reach_the_kl() -> None:
    """The tail block joins core attention but is not a scoring candidate."""
    batch, heads, queries, keys, ratio = 1, 2, 5, 10, 4
    num_blocks = keys // ratio  # 2 complete blocks; tokens 8 and 9 are the tail
    probabilities = _teacher(batch, heads, queries, keys)
    tokens = qsa_teacher_token_distribution(probabilities)

    torch.manual_seed(11)
    logits = torch.randn(batch, queries, num_blocks, dtype=torch.float64)
    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)
    baseline = qsa_indexer_kl_loss(logits, tokens, support, compress_ratio=ratio, loss_coeff=1.0)

    perturbed = tokens.clone()
    perturbed[..., num_blocks * ratio :] += 10.0  # tail mass only
    shifted = qsa_indexer_kl_loss(logits, perturbed, support, compress_ratio=ratio, loss_coeff=1.0)
    torch.testing.assert_close(baseline, shifted)


def test_kl_vanishes_when_student_matches_the_pooled_teacher() -> None:
    batch, heads, queries, keys, ratio = 2, 3, 6, 8, 4
    num_blocks = keys // ratio
    tokens = qsa_teacher_token_distribution(_teacher(batch, heads, queries, keys))
    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)
    pooled = qsa_maxpool_teacher_to_blocks(tokens, compress_ratio=ratio, num_blocks=num_blocks)
    target = pooled / pooled.sum(-1, keepdim=True)

    loss = qsa_indexer_kl_loss(target.log(), tokens, support, compress_ratio=ratio, loss_coeff=1.0)
    assert abs(float(loss)) < 1e-12


def test_stage_two_support_is_the_detached_indexer_topk() -> None:
    batch, queries, num_blocks, budget, ratio = 2, 12, 6, 2, 4
    torch.manual_seed(17)
    logits = torch.randn(batch, queries, num_blocks, dtype=torch.float64, requires_grad=True)
    lengths = torch.full((batch,), queries, dtype=torch.int64)
    causal = qsa_block_causal_mask(
        queries, num_blocks, lengths, compress_ratio=ratio, device=logits.device
    )
    support = qsa_topk_block_support(logits, causal, block_budget=budget)

    assert not support.requires_grad
    assert bool((support & ~causal).sum() == 0), "support must stay inside the causal mask"
    visible = causal.sum(-1)
    sparse = visible > budget
    assert bool(sparse.any()), "configuration must exercise a real top-k"
    assert torch.equal(support.sum(-1)[sparse], torch.full_like(support.sum(-1)[sparse], budget))
    # Rows within budget keep every visible block.
    assert torch.equal(support[~sparse], causal[~sparse])


def test_indexer_kl_gradcheck_in_float64() -> None:
    """Gradcheck the differentiable chain with a frozen, smooth configuration."""
    batch, queries, heads, num_blocks, ratio, head_dim = 2, 6, 3, 4, 4, 5
    keys = num_blocks * ratio
    torch.manual_seed(19)

    # Strictly positive q and k keep every dot product away from the ReLU kink.
    index_query = (
        torch.rand(batch, queries, heads, head_dim, dtype=torch.float64) + 1.0
    ).requires_grad_(True)
    compressed_key = (
        torch.rand(batch, num_blocks, 1, head_dim, dtype=torch.float64) + 1.0
    ).requires_grad_(True)

    # A teacher with a unique arg-max inside every block, so MaxPool is smooth.
    tokens = torch.zeros(batch, queries, keys, dtype=torch.float64)
    peaks = torch.arange(1, num_blocks + 1, dtype=torch.float64)
    for block in range(num_blocks):
        tokens[..., block * ratio] = peaks[block]
        for offset in range(1, ratio):
            tokens[..., block * ratio + offset] = peaks[block] * 0.1 * offset
    tokens = tokens / tokens.sum(-1, keepdim=True)

    # Freeze the support: gradcheck must never differentiate the discrete top-k.
    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)

    def objective(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        logits = compute_block_scores(query, key)
        return qsa_indexer_kl_loss(logits, tokens, support, compress_ratio=ratio, loss_coeff=1.0)

    assert torch.autograd.gradcheck(objective, (index_query, compressed_key), eps=1e-6, atol=1e-8)


def test_indexer_only_training_drives_the_kl_down() -> None:
    """Freeze everything but the indexer and overfit a fixed teacher."""
    torch.manual_seed(29)
    sequence_length, batch_size, hidden_size = 32, 2, 16
    ratio, budget_tokens, heads, head_dim = 4, 8, 4, 8
    indexer = QSAIndexer(
        hidden_size=hidden_size,
        num_query_heads=heads,
        num_key_heads=1,
        head_dim=head_dim,
        compress_ratio=ratio,
        token_budget=budget_tokens,
        params_dtype=torch.float32,
    )
    indexer.requires_grad_(True)  # the checkpoint contract freezes it by default

    hidden_states = torch.randn(sequence_length, batch_size, hidden_size)
    rotary_angles = torch.randn(sequence_length, 1, 1, head_dim // 2)
    lengths = torch.full((batch_size,), sequence_length, dtype=torch.int64)
    num_blocks = sequence_length // ratio
    tokens = qsa_teacher_token_distribution(
        _teacher(batch_size, 3, sequence_length, sequence_length, dtype=torch.float32, seed=41)
    )
    causal = qsa_block_causal_mask(
        sequence_length, num_blocks, lengths, compress_ratio=ratio, device=hidden_states.device
    )

    optimizer = torch.optim.Adam(indexer.parameters(), lr=1e-2)
    before = {name: p.detach().clone() for name, p in indexer.named_parameters()}
    history = []
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        query, pooled_key = indexer.index_states(hidden_states, rotary_angles)
        logits = compute_block_scores(query, pooled_key)
        support = qsa_topk_block_support(logits, causal, block_budget=budget_tokens // ratio)
        loss = qsa_indexer_kl_loss(logits, tokens, support, compress_ratio=ratio, loss_coeff=1.0)
        loss.backward()
        grads = [p.grad for p in indexer.parameters() if p.grad is not None]
        assert grads, "indexer parameters must receive gradients"
        assert any(bool(g.abs().sum() > 0) for g in grads), "gradients must be nonzero"
        optimizer.step()
        history.append(loss.detach().item())

    assert history[-1] < history[0] * 0.7, f"KL did not fall: {history[0]:.4f} -> {history[-1]:.4f}"
    assert all(
        not torch.equal(before[name], p.detach()) for name, p in indexer.named_parameters()
    ), "every indexer parameter must actually move"


def test_teacher_promotes_bf16_to_fp32_and_tracks_the_fp32_reference() -> None:
    """A BF16 teacher must reduce in FP32; BF16 rows have many small terms."""
    batch, heads, queries, keys = 2, 8, 6, 32
    torch.manual_seed(61)
    logits = torch.randn(batch, heads, queries, keys)
    bf16 = qsa_teacher_token_distribution(torch.softmax(logits.bfloat16(), dim=-1))
    fp32 = qsa_teacher_token_distribution(torch.softmax(logits, dim=-1))

    assert bf16.dtype == torch.float32, "BF16 input must reduce and return in FP32"
    assert fp32.dtype == torch.float32
    torch.testing.assert_close(bf16.sum(-1), torch.ones(batch, queries))
    torch.testing.assert_close(bf16, fp32, rtol=3e-2, atol=3e-3)

    doubles = qsa_teacher_token_distribution(torch.softmax(logits.double(), dim=-1))
    assert doubles.dtype == torch.float64, "FP64 must survive for gradcheck"


def test_kl_never_propagates_gradient_into_core_attention() -> None:
    """The teacher is a target, not a gradient path."""
    batch, heads, queries, keys, ratio = 1, 4, 8, 16, 4
    num_blocks = keys // ratio
    torch.manual_seed(67)
    attention_scores = torch.randn(batch, heads, queries, keys, requires_grad=True)
    teacher = qsa_teacher_token_distribution(torch.softmax(attention_scores, dim=-1))
    assert not teacher.requires_grad, "teacher must be detached"

    logits = torch.randn(batch, queries, num_blocks, requires_grad=True)
    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)
    qsa_indexer_kl_loss(logits, teacher, support, compress_ratio=ratio, loss_coeff=1.0).backward()

    assert attention_scores.grad is None, "no gradient may reach core attention"
    assert logits.grad is not None and bool(logits.grad.abs().sum() > 0)


def test_dense_stage1_teacher_rejects_head_sharded_tensor_parallelism() -> None:
    """The materialized Stage-1 helper has no process group for its required head sum."""
    probabilities = torch.softmax(torch.randn(1, 4, 3, 8), dim=-1)
    with pytest.raises(NotImplementedError, match="tensor_parallel_size=1"):
        qsa_teacher_token_distribution(probabilities, tensor_parallel_size=2)
    qsa_teacher_token_distribution(probabilities, tensor_parallel_size=1)


def test_fully_masked_rows_contribute_zero_and_stay_finite() -> None:
    """A query with no scorable block must not produce NaN or a gradient."""
    batch, heads, queries, keys, ratio = 1, 3, 6, 8, 4
    num_blocks = keys // ratio
    teacher = qsa_teacher_token_distribution(_teacher(batch, heads, queries, keys))
    torch.manual_seed(71)
    logits = torch.randn(batch, queries, num_blocks, dtype=torch.float64, requires_grad=True)

    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)
    support[:, 0, :] = False  # query 0 sees no complete block
    loss = qsa_indexer_kl_loss(logits, teacher, support, compress_ratio=ratio, loss_coeff=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[:, 0], torch.zeros_like(logits.grad[:, 0]))


def test_padded_query_rows_are_excluded_from_loss_and_gradient() -> None:
    batch, heads, queries, keys, ratio = 2, 3, 6, 8, 4
    num_blocks = keys // ratio
    teacher = qsa_teacher_token_distribution(_teacher(batch, heads, queries, keys))
    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)
    torch.manual_seed(73)
    base = torch.randn(batch, queries, num_blocks, dtype=torch.float64)

    valid_rows = torch.ones(batch, queries, dtype=torch.bool)
    valid_rows[:, 4:] = False  # right padding

    logits = base.clone().requires_grad_(True)
    loss = qsa_indexer_kl_loss(
        logits, teacher, support, compress_ratio=ratio, loss_coeff=1.0, query_valid_rows=valid_rows
    )
    loss.backward()
    torch.testing.assert_close(logits.grad[:, 4:], torch.zeros_like(logits.grad[:, 4:]))

    # Perturbing only the padded rows must not move the loss.
    shifted = base.clone()
    shifted[:, 4:] += 5.0
    other = qsa_indexer_kl_loss(
        shifted, teacher, support, compress_ratio=ratio, loss_coeff=1.0, query_valid_rows=valid_rows
    )
    torch.testing.assert_close(loss.detach(), other)


def test_query_valid_rows_are_normalized_or_rejected() -> None:
    teacher = qsa_teacher_token_distribution(_teacher(2, 3, 6, 8))
    support = torch.ones(2, 6, 2, dtype=torch.bool)
    logits = torch.randn(2, 6, 2, dtype=torch.float64)

    one_dimensional = qsa_indexer_kl_loss(
        logits,
        teacher,
        support,
        compress_ratio=4,
        loss_coeff=1.0,
        query_valid_rows=torch.tensor([True, True, True, False, False, False]),
    )
    expanded = qsa_indexer_kl_loss(
        logits,
        teacher,
        support,
        compress_ratio=4,
        loss_coeff=1.0,
        query_valid_rows=torch.tensor([[True, True, True, False, False, False]] * 2),
    )
    torch.testing.assert_close(one_dimensional, expanded)

    with pytest.raises(ValueError, match="query_valid_rows shape mismatch"):
        qsa_indexer_kl_loss(
            logits,
            teacher,
            support,
            compress_ratio=4,
            loss_coeff=1.0,
            query_valid_rows=torch.ones(2, 1),
        )


def test_loss_coefficient_scales_linearly_and_reductions_are_consistent() -> None:
    batch, heads, queries, keys, ratio = 2, 3, 8, 8, 4
    num_blocks = keys // ratio
    teacher = qsa_teacher_token_distribution(_teacher(batch, heads, queries, keys))
    support = torch.ones(batch, queries, num_blocks, dtype=torch.bool)
    torch.manual_seed(79)
    logits = torch.randn(batch, queries, num_blocks, dtype=torch.float64)

    single = qsa_indexer_kl_loss(logits, teacher, support, compress_ratio=ratio, loss_coeff=1.0)
    doubled = qsa_indexer_kl_loss(logits, teacher, support, compress_ratio=ratio, loss_coeff=2.0)
    torch.testing.assert_close(doubled, single * 2.0)

    summed = qsa_indexer_kl_loss(
        logits,
        teacher,
        support,
        compress_ratio=ratio,
        loss_coeff=1.0,
        calculate_per_token_loss=True,
    )
    torch.testing.assert_close(summed, single * (batch * queries))

    valid_rows = torch.ones(batch, queries, dtype=torch.bool)
    valid_rows[:, 6:] = False
    masked_mean = qsa_indexer_kl_loss(
        logits, teacher, support, compress_ratio=ratio, loss_coeff=1.0, query_valid_rows=valid_rows
    )
    masked_sum = qsa_indexer_kl_loss(
        logits,
        teacher,
        support,
        compress_ratio=ratio,
        loss_coeff=1.0,
        query_valid_rows=valid_rows,
        calculate_per_token_loss=True,
    )
    torch.testing.assert_close(masked_mean, masked_sum / float(valid_rows.sum()))
