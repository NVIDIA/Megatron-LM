# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fused sigmoid router chain (experiment line "fuse"): the math twin's closed-form backward
equals autograd of Megatron's eager routing + seq-aux-loss text (CPU), and on CUDA the Function's
Triton forward / backward match the math twin."""

import pytest
import torch

from megatron.core.transformer.moe import fused_router_chain as frc


def eager_chain(logits, bias, topk, groups, group_topk, scale, coeff, total):
    """``topk_routing_with_score_function`` (sigmoid, expert bias, group-limited top-k) and
    ``compute_routing_scores_for_aux_loss`` + ``switch_load_balancing_loss_func`` on one rank."""
    num_tokens, num_experts = logits.shape
    scores = torch.sigmoid(logits.float())
    scores_for_routing = scores + bias.float()
    group_scores = (
        scores_for_routing.view(num_tokens, groups, -1)
        .topk(topk // group_topk, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, groups, num_experts // groups)
        .reshape(num_tokens, -1)
    )
    masked = scores_for_routing.masked_fill(~score_mask.bool(), float('-inf'))
    _, top_indices = torch.topk(masked, k=topk, dim=-1)
    picked = torch.gather(scores, dim=1, index=top_indices)
    probs = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20) * scale
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs.type_as(logits))
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    aux_scores = torch.sigmoid(logits.float())
    aux_scores = aux_scores / (aux_scores.sum(dim=-1, keepdim=True) + 1e-20)
    _, aux_top = torch.topk(aux_scores, k=topk, dim=1)
    counts = torch.zeros_like(logits).int().scatter(1, aux_top, 1).bool().sum(dim=0)
    aux_loss = torch.sum(aux_scores.sum(dim=0) * counts) * (
        num_experts * coeff / (topk * total * total)
    )
    return routing_probs, routing_map, aux_loss, top_indices, counts


SHAPES = ((32, 8, 2, 1, 2), (256, 64, 8, 4, 8))


@pytest.mark.parametrize(("tokens", "experts", "groups", "group_topk", "topk"), SHAPES)
def test_math_twin_matches_autograd_of_the_eager_text(tokens, experts, groups, group_topk, topk):
    gen = torch.Generator().manual_seed(tokens + experts)
    logits = (torch.randn(tokens, experts, generator=gen) * 2).requires_grad_(True)
    bias = torch.randn(experts, generator=gen) * 0.1
    dense, routing_map, aux, top, counts = eager_chain(
        logits, bias, topk, groups, group_topk, 2.5, 1e-4, tokens
    )
    grad_probs = torch.randn(tokens, experts, generator=gen) * 1e-3 * routing_map
    grad_aux = torch.tensor(0.7)
    (want,) = torch.autograd.grad((dense * grad_probs).sum() + aux * grad_aux, (logits,))
    with torch.no_grad():
        probs_dense, indices, scores = frc.math_forward(logits, bias, topk, groups, group_topk, 2.5)
        local_counts, colsum = frc.math_aux_stats(scores, topk)
        loss = frc.math_aux_loss(colsum, local_counts, tokens, topk, experts, 1e-4)
        aux_scale = experts * 1e-4 / (topk * tokens * tokens)
        got = frc.math_backward(
            grad_probs, grad_aux, scores, indices, local_counts, 2.5, aux_scale
        )
    assert torch.equal(probs_dense, dense)
    assert torch.equal(torch.sort(indices, 1).values, torch.sort(top, 1).values)
    assert torch.equal(local_counts, counts.float())
    torch.testing.assert_close(loss, aux)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(frc.triton is None, reason="Triton not available")
@pytest.mark.parametrize(("tokens", "experts", "groups", "group_topk", "topk"), SHAPES)
@pytest.mark.parametrize("dense_indices", [False, True])
def test_function_matches_the_math_twin_on_cuda(
    tokens, experts, groups, group_topk, topk, dense_indices
):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(tokens * 3 + experts)
    logits = (torch.randn(tokens, experts, generator=gen, device=device) * 2).requires_grad_(True)
    bias = torch.randn(experts, generator=gen, device=device) * 0.1
    probs, routing_map, aux = frc.fused_sigmoid_router_chain(
        logits, bias, topk, groups, group_topk, 2.5, 1e-4, (), dense_indices
    )
    with torch.no_grad():
        want_probs, want_idx, scores = frc.math_forward(logits, bias, topk, groups, group_topk, 2.5)
        counts, colsum = frc.math_aux_stats(scores, topk)
        want_aux = frc.math_aux_loss(colsum, counts, tokens, topk, experts, 1e-4)
    torch.testing.assert_close(probs, want_probs)
    if dense_indices:
        assert routing_map.dtype == torch.int64 and routing_map.shape == (tokens, topk)
        assert torch.equal(torch.sort(routing_map, 1).values, torch.sort(want_idx, 1).values)
    else:
        assert routing_map.dtype == torch.bool
        assert torch.equal(routing_map, want_probs != 0)
    torch.testing.assert_close(aux, want_aux)
    grad_probs = torch.randn(tokens, experts, generator=gen, device=device) * 1e-3
    grad_aux = torch.tensor(0.7, device=device)
    (got,) = torch.autograd.grad((probs * grad_probs).sum() + aux * grad_aux, (logits,))
    with torch.no_grad():
        want = frc.math_backward(
            grad_probs * (want_probs != 0), grad_aux, scores, want_idx, counts, 2.5,
            experts * 1e-4 / (topk * tokens * tokens),
        )
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-9)
