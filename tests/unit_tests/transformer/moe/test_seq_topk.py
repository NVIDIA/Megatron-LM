# Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.

"""Unit tests for SeqTopK routing (``megatron/core/transformer/moe/seq_topk.py``).

These tests exercise the pure-tensor selection logic (no model parallelism, no dispatchers).
Run with: ``pytest tests/unit_tests/transformer/moe/test_seq_topk.py``.
"""

import pytest
import torch

from megatron.core.transformer.moe.seq_topk import (
    build_seq_idx,
    dense_to_padded_indices,
    seqtopk_routing,
)


def _make_logits(num_tokens: int, num_experts: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(num_tokens, num_experts, generator=gen)


def test_build_seq_idx_contiguous():
    cu = torch.tensor([0, 3, 5, 9])
    seq_idx = build_seq_idx(cu, 9, torch.device("cpu"))
    # Expected: 0,0,0,1,1,2,2,2,2
    assert seq_idx.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    # Contiguous & non-decreasing.
    assert (seq_idx[1:] >= seq_idx[:-1]).all()


@pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
def test_budget_and_cap(score_function):
    # 2 sequences: T0=4, T1=3. K=2, U=4. num_experts=8.
    cu = torch.tensor([0, 4, 7])
    seq_idx = build_seq_idx(cu, 7, torch.device("cpu"))
    logits = _make_logits(7, 8, seed=1)
    K, U = 2, 4
    probs, routing_map = seqtopk_routing(
        logits, topk=K, upper_bound=U, seq_idx=seq_idx, score_function=score_function
    )
    # Per-token cap.
    counts = routing_map.sum(dim=1)
    assert (counts <= U).all(), f"cap violated: max {counts.max()}"
    # Per-sequence total budget = T_s * K (budget fills when candidates available).
    for s, T_s in enumerate([4, 3]):
        tot = routing_map[seq_idx == s].sum().item()
        assert tot == T_s * K, f"seq {s}: expected budget {T_s * K}, got {tot}"


def test_total_budget_matches_topk():
    # Total selected across the whole micro-batch = T_total * K (same as standard TopK).
    cu = torch.tensor([0, 5, 11, 13])
    seq_idx = build_seq_idx(cu, 13, torch.device("cpu"))
    logits = _make_logits(13, 16, seed=2)
    K, U = 3, 6
    _, routing_map = seqtopk_routing(
        logits, topk=K, upper_bound=U, seq_idx=seq_idx, score_function="sigmoid"
    )
    assert routing_map.sum().item() == 13 * K


def _brute_force_seqtopk(scores: torch.Tensor, seq_idx: torch.Tensor, K: int, U: int) -> torch.Tensor:
    """Reference implementation: per sequence, greedily pick the top (T_s*K) (token,expert)
    pairs by score, capping each token at U selected experts. Returns a dense bool mask."""
    num_tokens, num_experts = scores.shape
    selected = torch.zeros((num_tokens, num_experts), dtype=torch.bool)
    for s in seq_idx.unique().tolist():
        tok_ids = (seq_idx == s).nonzero(as_tuple=True)[0]
        T_s = tok_ids.numel()
        budget = T_s * K
        # All (token, expert) pairs in this sequence, ranked by score desc.
        sub = scores[tok_ids]  # [T_s, num_experts]
        flat_scores, flat_idx = torch.sort(sub.flatten(), descending=True)
        used = [0] * T_s
        count = 0
        for val, idx in zip(flat_scores.tolist(), flat_idx.tolist()):
            if val == float("-inf"):
                break
            t_local = idx // num_experts
            e = idx % num_experts
            if used[t_local] >= U:
                continue
            selected[tok_ids[t_local], e] = True
            used[t_local] += 1
            count += 1
            if count >= budget:
                break
    return selected


@pytest.mark.parametrize("score_function", ["sigmoid", "softmax"])
def test_selection_matches_brute_force(score_function):
    # Without groups, SeqTopK = capped per-sequence top-(T*K). Compare to a brute-force reference.
    cu = torch.tensor([0, 5, 9, 13])
    seq_idx = build_seq_idx(cu, 13, torch.device("cpu"))
    logits = _make_logits(13, 12, seed=42)
    K, U = 2, 4
    _, routing_map = seqtopk_routing(
        logits, topk=K, upper_bound=U, seq_idx=seq_idx, score_function=score_function
    )
    # The ranking score: sigmoid scores, or raw logits for (post) softmax.
    if score_function == "sigmoid":
        rank_scores = torch.sigmoid(logits.float())
    else:
        rank_scores = logits.float()
    ref = _brute_force_seqtopk(rank_scores, seq_idx, K, U)
    assert torch.equal(routing_map, ref), "SeqTopK selection diverged from brute-force reference"


def test_dynamic_allocation_gives_hard_tokens_more_experts():
    # A token with uniformly high scores should receive more experts than easy tokens,
    # while the per-sequence total budget is preserved.
    cu = torch.tensor([0, 4])
    seq_idx = build_seq_idx(cu, 4, torch.device("cpu"))
    # token 0 is "hard" (high scores across experts); tokens 1-3 are "easy" (low everywhere).
    logits = torch.tensor(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],  # hard: many high scores
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        ]
    )
    K, U = 1, 3
    _, routing_map = seqtopk_routing(
        logits, topk=K, upper_bound=U, seq_idx=seq_idx, score_function="softmax"
    )
    counts = routing_map.sum(dim=1).tolist()
    # Hard token (token 0) hits the cap U=3; it grabs the top-3 pairs. Remaining budget (1)
    # goes to the next-highest easy token. Other easy tokens get 0.
    assert counts[0] == 3, f"hard token expected 3 experts, got {counts[0]}"
    assert counts[2] == 0 and counts[3] == 0
    # Total budget preserved (T*K = 4).
    assert sum(counts) == 4


def test_probs_renormalized_per_token():
    cu = torch.tensor([0, 3])
    seq_idx = build_seq_idx(cu, 3, torch.device("cpu"))
    logits = _make_logits(3, 8, seed=3)
    _, routing_map = seqtopk_routing(
        logits, topk=2, upper_bound=4, seq_idx=seq_idx, score_function="sigmoid"
    )
    probs, _ = seqtopk_routing(
        logits, topk=2, upper_bound=4, seq_idx=seq_idx, score_function="sigmoid"
    )
    for t in range(3):
        c = routing_map[t].sum().item()
        s = probs[t].sum().item()
        if c == 0:
            assert s == pytest.approx(0.0, abs=1e-6)
        else:
            assert s == pytest.approx(1.0, abs=1e-5), f"token {t} sum {s} != 1"
        # unselected experts have zero prob
        assert (probs[t][~routing_map[t]] == 0).all()


def test_gradient_flows_only_through_probs():
    cu = torch.tensor([0, 4])
    seq_idx = build_seq_idx(cu, 4, torch.device("cpu"))
    logits = _make_logits(4, 8, seed=4).requires_grad_(True)
    probs, routing_map = seqtopk_routing(
        logits, topk=2, upper_bound=4, seq_idx=seq_idx, score_function="sigmoid"
    )
    loss = probs.sum()
    loss.backward()
    assert logits.grad is not None
    # Gradients only at selected (token, expert) pairs (straight-through on selection).
    grad = logits.grad
    assert (grad[~routing_map] == 0).all(), "gradient leaked to unselected experts"
    assert (grad[routing_map] != 0).any(), "no gradient on selected experts"


def test_expert_bias_only_affects_selection():
    cu = torch.tensor([0, 3])
    seq_idx = build_seq_idx(cu, 3, torch.device("cpu"))
    logits = _make_logits(3, 6, seed=5)
    bias = torch.zeros(6)
    bias[5] = 100.0  # force expert 5 to be selected everywhere
    _, routing_map = seqtopk_routing(
        logits, topk=1, upper_bound=2, seq_idx=seq_idx, score_function="sigmoid",
        expert_bias=bias,
    )
    assert routing_map[:, 5].all(), "expert_bias should force expert 5 selection"


def test_group_candidate_filtering():
    # With group filtering, only candidate experts (in selected groups) can be chosen.
    cu = torch.tensor([0, 4])
    seq_idx = build_seq_idx(cu, 4, torch.device("cpu"))
    num_experts = 8
    num_groups = 4
    group_topk = 1  # only 1 group (2 experts) per token
    logits = _make_logits(4, num_experts, seed=6)
    _, routing_map = seqtopk_routing(
        logits, topk=2, upper_bound=4, seq_idx=seq_idx, score_function="sigmoid",
        num_groups=num_groups, group_topk=group_topk,
    )
    counts = routing_map.sum(dim=1)
    assert (counts <= 4).all()
    # Each token's selected experts must lie within a single group of 2 experts.
    for t in range(4):
        sel = routing_map[t].nonzero(as_tuple=True)[0].tolist()
        groups = {e // (num_experts // num_groups) for e in sel}
        assert len(groups) <= 1, f"token {t} selected across groups {groups}"


def test_dense_to_padded_indices_roundtrip():
    # Reconstruct (indices, probs) and verify the multihot view matches the original.
    cu = torch.tensor([0, 3, 6])
    seq_idx = build_seq_idx(cu, 6, torch.device("cpu"))
    logits = _make_logits(6, 10, seed=7)
    probs, routing_map = seqtopk_routing(
        logits, topk=2, upper_bound=4, seq_idx=seq_idx, score_function="sigmoid"
    )
    U = 4
    indices, padded_probs = dense_to_padded_indices(probs, routing_map, k=U)
    assert indices.shape == (6, U)
    # -1 padding count per row == U - selected_count
    pad = (indices == -1).sum(dim=1)
    sel = routing_map.sum(dim=1)
    assert (pad == U - sel).all()
    # Non-padded indices match the selected experts.
    for t in range(6):
        valid_idx = indices[t][indices[t] != -1]
        assert set(valid_idx.tolist()) == set(routing_map[t].nonzero(as_tuple=True)[0].tolist())


def test_padding_tokens_get_zero_experts():
    cu = torch.tensor([0, 4])
    seq_idx = build_seq_idx(cu, 4, torch.device("cpu"))
    logits = _make_logits(4, 6, seed=8)
    pad = torch.tensor([False, False, True, True])  # last two are padding
    _, routing_map = seqtopk_routing(
        logits, topk=2, upper_bound=4, seq_idx=seq_idx, score_function="sigmoid",
        padding_mask=pad,
    )
    assert routing_map[2].sum().item() == 0
    assert routing_map[3].sum().item() == 0
    # Valid tokens still get their budget from the (non-padded) sequence length.
    assert routing_map[:2].sum().item() <= 4 * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])