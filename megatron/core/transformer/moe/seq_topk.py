# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sequence-level TopK (SeqTopK) routing for MoE layers.

Reference: "Route Experts by Sequence, Not by Token" (arXiv 2511.06494).

Standard TopK allocates a fixed budget of K experts to *every* token. SeqTopK moves the
budget to the sequence level: the top (T * K) (token, expert) pairs are selected across all
T tokens within a single sequence. The total budget per sequence is preserved, but harder
tokens can receive more experts (up to a per-token cap ``upper_bound``) and easier tokens
fewer (possibly zero), enabling end-to-end learned dynamic allocation.

This module implements the selection logic only. It returns the same dense
``[num_tokens, num_experts]`` ``probs`` / ``routing_map`` contract that the rest of the
Megatron MoE router/dispatcher already consumes, so SeqTopK is a drop-in replacement for
the per-token top-K step. Sequence boundaries are taken from a contiguous per-token group
id (``seq_idx``, derived upstream from ``packed_seq_params.cu_seqlens_q``); competition is
strictly within a sequence (never across sequences / batches).
"""

from typing import Optional, Tuple

import torch


def build_seq_idx(
    cu_seqlens: torch.Tensor,
    num_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a contiguous per-token sequence id tensor from a cumulative-seqlens vector.

    Args:
        cu_seqlens (torch.Tensor): Cumulative sequence lengths, shape ``[num_seqs + 1]``,
            starting with 0 and ending with ``num_tokens``. Defines contiguous segments:
            tokens ``[cu_seqlens[s], cu_seqlens[s+1])`` belong to sequence ``s``.
        num_tokens (int): Total number of tokens (must equal ``cu_seqlens[-1]``).
        device (torch.device): Device for the result.

    Returns:
        torch.Tensor: ``int64`` tensor of shape ``[num_tokens]`` where entry ``t`` is the
        sequence id of token ``t``. Contiguous and non-decreasing (sorted).
    """
    cu = cu_seqlens.to(device=device, dtype=torch.long)
    if cu.numel() < 2:
        return torch.zeros(num_tokens, dtype=torch.long, device=device)
    seq_lengths = cu[1:] - cu[:-1]  # [num_seqs]
    seq_idx = torch.repeat_interleave(
        torch.arange(seq_lengths.numel(), device=device, dtype=torch.long), seq_lengths
    )
    return seq_idx[:num_tokens]


def _per_key_cumcount(keys: torch.Tensor) -> torch.Tensor:
    """1-based per-key cumulative count along the input order.

    For each position ``p`` returns the number of positions ``q <= p`` with ``keys[q] == keys[p]``
    (i.e. the 1-based rank of this element among earlier same-key elements). Vectorized via a
    stable sort by key: within each key group the elements retain their original order, so the
    within-group position equals the rank.

    Args:
        keys (torch.Tensor): ``[M]`` integer keys.

    Returns:
        torch.Tensor: ``[M]`` int64 1-based per-key cumulative count.
    """
    n = keys.numel()
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=keys.device)
    order = torch.argsort(keys, stable=True)
    k_sorted = keys[order]
    new_group = torch.ones(n, dtype=torch.bool, device=keys.device)
    new_group[1:] = k_sorted[1:] != k_sorted[:-1]
    group_id = torch.cumsum(new_group.long(), dim=0) - 1
    starts_idx = torch.nonzero(new_group, as_tuple=False).flatten()
    group_start = starts_idx[group_id]
    within_rank = torch.arange(n, device=keys.device) - group_start + 1  # 1-based, in input order
    cum = torch.zeros(n, dtype=torch.long, device=keys.device)
    cum[order] = within_rank
    return cum


def _group_candidate_mask(
    scores: torch.Tensor, topk: int, num_groups: Optional[int], group_topk: Optional[int]
) -> Optional[torch.Tensor]:
    """DeepSeek-style group-limited candidate mask.

    Mirrors ``group_limited_topk`` in ``moe_utils.py`` (group selection only) but returns a
    boolean candidate mask ``[num_tokens, num_experts]`` (True = candidate) instead of final
    top-k indices, so the subsequent per-sequence budget selection can operate on the candidate
    set. Within-group topk uses ``topk // group_topk`` exactly as in ``group_limited_topk``.

    Args:
        scores (torch.Tensor): ``[num_tokens, num_experts]`` routing scores.
        topk (int): Base per-token budget K (used for within-group topk = ``topk // group_topk``).
        num_groups (Optional[int]): Number of equal-sized expert groups.
        group_topk (Optional[int]): Number of groups selected per token.

    Returns:
        Optional[torch.Tensor]: boolean candidate mask, or ``None`` if group filtering is
        disabled (all experts are candidates).
    """
    if not group_topk or not num_groups:
        return None
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // num_groups
    within_group_topk = max(1, topk // group_topk)
    group_scores = (
        scores.view(num_tokens, num_groups, experts_per_group)
        .topk(within_group_topk, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, experts_per_group)
        .reshape(num_tokens, -1)
    )
    return score_mask.bool()


def _capped_per_sequence_budget_select(
    rank_scores: torch.Tensor,
    seq_idx: torch.Tensor,
    tokens_per_seq: torch.Tensor,
    topk: int,
    upper_bound: int,
) -> torch.Tensor:
    """Capped per-sequence budget selection.

    For each sequence ``s`` with ``T_s`` *valid* tokens, selects the top ``B_s = T_s * topk``
    ``(token, expert)`` pairs from the candidate ``rank_scores`` matrix, subject to a per-token
    cap of ``upper_bound`` selected experts. Equivalently: among all candidate pairs in a
    sequence (ranked by score), take the highest-scoring ``B_s`` pairs while skipping any pair
    whose token has already reached ``upper_bound`` selected experts.

    The selection operates globally (one sort over all ``num_tokens * num_experts`` candidate
    pairs) but competition is strictly per-sequence: a pair only competes against pairs in its
    own sequence, and each sequence's budget is ``T_s * topk``.

    The budget is sized from the *valid* (non-padding) token count per sequence, not the full
    segment length: padding tokens have already been masked to ``-inf`` in ``rank_scores`` (so
    they are never selected) and must not contribute to the budget, otherwise the surplus
    budget would be absorbed by padding tokens and they would be routed.

    Args:
        rank_scores (torch.Tensor): ``[num_tokens, num_experts]`` scores used for *selection*
            (e.g. sigmoid scores + expert_bias). Non-candidate / padding entries must already
            be ``-inf``.
        seq_idx (torch.Tensor): ``[num_tokens]`` sequence ids (need not be contiguous for this
            function; per-token grouping keys are the sequence ids).
        tokens_per_seq (torch.Tensor): ``[num_seqs]`` valid (non-padding) token count per
            sequence. Drives the per-sequence budget ``B_s = tokens_per_seq[s] * topk``.
        topk (int): Base per-token budget K; the per-sequence budget is ``T_s * topk``.
        upper_bound (int): Per-token cap U (>= topk). A token may receive 0..U experts.

    Returns:
        torch.Tensor: boolean ``[num_tokens, num_experts]`` selection mask.
    """
    num_tokens, num_experts = rank_scores.shape
    device = rank_scores.device
    selected = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device=device)
    if num_tokens == 0:
        return selected

    budget_per_seq = tokens_per_seq * topk                              # [num_seqs], B_s

    # Sort all candidate (token, expert) pairs globally by score, descending.
    flat_scores = rank_scores.reshape(-1)                               # [num_tokens * num_experts]
    sorted_scores, sorted_idx = torch.sort(flat_scores, descending=True)
    tok = sorted_idx // num_experts                                      # global token id per pair
    exp = sorted_idx % num_experts
    seq_of = seq_idx[tok]                                               # sequence id per pair

    # Per-token cap: a pair is cap-eligible if it is among this token's top-`upper_bound` pairs.
    tok_rank = _per_key_cumcount(tok)                                   # 1-based rank within token
    cap_eligible = tok_rank <= upper_bound

    # Per-sequence budget: among cap-eligible pairs (in score-desc order), take the first
    # `budget_per_seq[seq]` per sequence.
    elig_pos = torch.nonzero(cap_eligible, as_tuple=False).flatten()    # positions in sorted order
    if elig_pos.numel() == 0:
        return selected
    seq_e = seq_of[elig_pos]
    seq_rank = _per_key_cumcount(seq_e)                                 # 1-based rank among eligible per seq
    take = seq_rank <= budget_per_seq[seq_e]
    chosen_sorted_pos = elig_pos[take]

    chosen_tok = tok[chosen_sorted_pos]
    chosen_exp = exp[chosen_sorted_pos]
    selected[chosen_tok, chosen_exp] = True
    return selected


def seqtopk_routing(
    logits: torch.Tensor,
    topk: int,
    upper_bound: int,
    seq_idx: torch.Tensor,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    use_pre_softmax: bool = False,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequence-level TopK routing.

    Computes expert scores, optionally filters candidates via group-limited routing, then
    selects the top ``(T_s * topk)`` ``(token, expert)`` pairs per sequence with a per-token cap
    ``upper_bound``. Returns dense ``probs`` / ``routing_map`` (same contract as
    ``topk_routing_with_score_function`` with ``dense_output=False``).

    The selection (argTopK) is non-differentiable; gradients flow into the selected experts'
    logits/probs through the dense score computation (straight-through, matching standard
    Megatron TopK).

    Args:
        logits (torch.Tensor): ``[num_tokens, num_experts]`` router logits (fp32 recommended;
            intermediate scores are computed in fp32).
        topk (int): Base per-token budget K.
        upper_bound (int): Per-token cap U (>= topk).
        seq_idx (torch.Tensor): ``[num_tokens]`` contiguous sequence ids.
        num_groups (Optional[int]): Group-limited routing groups (DeepSeek style), optional.
        group_topk (Optional[int]): Groups selected per token for candidate filtering.
        use_pre_softmax (bool): If True (softmax only), normalize over all experts before
            selection and do not renormalize over the selected set.
        scaling_factor (Optional[float]): Pre-softmax prob scaling factor.
        score_function (str): "softmax", "sigmoid", or "sqrtsoftplus".
        expert_bias (Optional[torch.Tensor]): Per-expert bias added to scores for *selection*
            only (probs use raw scores). Shape ``[num_experts]``.
        padding_mask (Optional[torch.Tensor]): ``[num_tokens]`` boolean mask, True = padding.
            Padding tokens are excluded from selection (assigned 0 experts) and excluded from
            the per-sequence budget, which is sized from the valid (non-padding) token count so
            that surplus budget can never be absorbed by padding tokens. When there is no
            padding (as in THD packing) the valid count equals the full segment length.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(probs, routing_map)``, both dense
        ``[num_tokens, num_experts]``. ``probs`` is float (type_as logits), ``routing_map``
        is bool. For tokens with zero selected experts both rows are all-zero (the token
        then contributes only its residual).
    """
    num_tokens, num_experts = logits.shape
    orig_dtype = logits.dtype
    logits_f = logits.float()

    # 1) Score function (fp32).
    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits_f, dim=-1)
        else:
            scores = logits_f  # rank on raw logits; renormalize over selected after selection
    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits_f)
    elif score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits_f).sqrt()
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    # 2) Group-limited candidate filtering (DeepSeek style). Produces a candidate mask.
    cand_mask = _group_candidate_mask(scores, topk, num_groups, group_topk)
    if cand_mask is not None:
        rank_scores = scores.masked_fill(~cand_mask, float('-inf'))
    else:
        rank_scores = scores.clone()

    # Exclude padding tokens from selection (assign them 0 experts, do not consume budget).
    if padding_mask is not None:
        rank_scores = rank_scores.masked_fill(padding_mask.bool().unsqueeze(1), float('-inf'))

    # 3) Selection scores (add expert_bias for ranking only).
    select_scores = rank_scores
    if expert_bias is not None:
        select_scores = select_scores + expert_bias.float().unsqueeze(0)

    # Per-sequence valid (non-padding) token count drives the budget T_valid_s * topk, so padding
    # tokens neither consume budget nor get routed (no surplus overflow into padding rows).
    num_seqs = int(seq_idx.max().item()) + 1
    if padding_mask is not None:
        tokens_per_seq = torch.bincount(
            seq_idx[~padding_mask.bool()], minlength=num_seqs
        )
    else:
        tokens_per_seq = torch.bincount(seq_idx, minlength=num_seqs)

    # 4) Capped per-sequence budget selection -> boolean mask [num_tokens, num_experts].
    # The selection (argTopK) is non-differentiable: detach so no autograd graph is built through
    # the greedy max/sort. Gradients flow only through the dense probs below (straight-through).
    with torch.no_grad():
        selected = _capped_per_sequence_budget_select(
            select_scores.detach(), seq_idx, tokens_per_seq, topk, upper_bound
        )

    # 5) Dense probs (differentiable through scores/logits; selection mask is non-diff).
    # Use a large-negative finite value (not -inf) when masking so that rows with zero selected
    # experts do not produce NaNs from softmax(all -inf); such rows are zeroed out afterwards.
    neg_large = -1e9
    if score_function == "softmax":
        if use_pre_softmax:
            # probs = full-softmax at selected (subset, not renormalized), scaled.
            probs = scores * selected.float()
            if scaling_factor:
                probs = probs * scaling_factor
        else:
            # softmax over the selected experts (renormalized); unselected -> ~0 via large neg.
            masked_logits = logits_f.masked_fill(~selected, neg_large)
            probs = torch.softmax(masked_logits, dim=-1) * selected.float()
    else:
        # sigmoid / sqrtsoftplus: L1 normalize selected scores.
        sel_scores = scores * selected.float()
        denom = sel_scores.sum(dim=-1, keepdim=True)
        probs = sel_scores / (denom + 1e-20)
        # Tokens with zero selected experts -> all-zero probs (denom 0).

    probs = probs.type_as(orig_dtype)
    routing_map = selected.to(torch.bool)
    return probs, routing_map


def dense_to_padded_indices(
    probs: torch.Tensor, routing_map: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert dense multihot probs/routing_map to fixed ``[num_tokens, k]`` index/prob tensors
    with ``-1`` / ``0`` padding for slots beyond each token's selected count.

    Used when a downstream dispatcher (e.g. DeepEP) needs a fixed-slot representation and the
    default ``torch.topk(probs, k)`` path is unsuitable. Selection is taken from the boolean
    ``routing_map`` so no phantom experts are ever emitted; padding slots get index ``-1`` and
    prob ``0``.

    Args:
        probs (torch.Tensor): ``[num_tokens, num_experts]`` dense probs.
        routing_map (torch.Tensor): ``[num_tokens, num_experts]`` boolean mask.
        k (int): Fixed slot count (>= max per-token selected count, typically ``upper_bound``).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (token_indices, token_probs) of shape
        ``[num_tokens, k]``; padding index ``-1``, padding prob ``0``.
    """
    num_tokens, num_experts = probs.shape
    device = probs.device
    token_indices = torch.full((num_tokens, k), -1, dtype=torch.long, device=device)
    token_probs = torch.zeros((num_tokens, k), dtype=torch.float32, device=device)
    # For each token, the selected experts are known via routing_map. Place them left-to-right.
    # Use the per-row counts to Scatter: arg-sort of (probs desc) restricted to selected.
    row_count = routing_map.sum(dim=-1)  # [num_tokens]
    max_count = int(row_count.max().item()) if num_tokens > 0 else 0
    fill = min(k, max_count)
    if fill == 0:
        return token_indices, token_probs
    # For selected entries, pick indices ordered by score desc (stable); only need first `fill`.
    masked = probs.masked_fill(~routing_map, float('-inf'))
    top_vals, top_idx = torch.topk(masked, k=fill, dim=-1)  # [num_tokens, fill]
    col = torch.arange(fill, device=device).unsqueeze(0)
    valid = col < row_count.unsqueeze(1)  # [num_tokens, fill]
    token_indices[:, :fill] = torch.where(valid, top_idx, torch.full_like(top_idx, -1))
    token_probs[:, :fill] = torch.where(valid, top_vals.float(), torch.zeros_like(top_vals, dtype=torch.float32))
    return token_indices, token_probs