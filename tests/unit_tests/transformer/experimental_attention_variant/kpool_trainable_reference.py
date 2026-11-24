# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent reference for the trainable KPool indexer loss (sparse pool-level KL).

Written from the semantics alone, never from the production helpers: this module imports
nothing from ``dsa.py``. Every quantity is built from dense, token-level tensors in plain
PyTorch, one query row at a time, so the code reads like the definitions below.

Notation, for query row ``t`` of a sequence or of a packed document that starts at ``s``
(rows are global token positions; there is no context parallelism here):

- pool ``c`` covers the ``P = pool_size`` tokens ``base[c] .. base[c] + P - 1``. Pools restart
  at each packed document and never span two; unused capacity slots have ``base[c] = -1``;
- ``C(t)``: the complete pools of ``t``'s document whose last token is at or before ``t``;
- ``S(t) ⊆ C(t)``: the selected pools, at most ``G = index_topk // P`` of them;
- ``T(t)``: the query-local tail, the ``(t - s + 1) % P`` newest tokens ``.. t``;
- ``A(t) = tokens(S(t)) ∪ T(t)``: what the sparse attention attends.

Teacher, per main-attention head ``h`` with softmax scale ``σ``::

    lse_h(t)     = log Σ_{j ∈ A(t)} exp(σ Q_h(t)·K_j)              # tail included
    m_h(t, c)    = Σ_{j ∈ tokens(c)} exp(σ Q_h(t)·K_j − lse_h(t))   # c ∈ S(t)
    M(t, c)      = Σ_h m_h(t, c)                                    # over every head (all TP ranks)
    target(t, c) = M(t, c) / Σ_{c'} M(t, c')                        # exact-zero safe divide

The tail is in each head's denominator and never in the target. A row whose pool mass is
exactly zero (no selected pool, or every pool mass underflows in the arithmetic dtype) has
an all-zero target and no loss.

Student: the selector's own pool score. HF scoring:
``I(t, c) = Σ_h w_h · relu(q_h·k̃_c · D^-1/2)`` with ``w`` carrying ``H^-1/2``. Legacy scoring:
``I(t, c) = Σ_h w_h · relu(q_h·k̃_c)`` with ``w`` carrying both scales (ReLU optional).

Loss: ``coeff · Σ_t has_mass(t) · Σ_{c ∈ S(t)} target·(log target − log softmax_c I) / row_count``.

Compression of pool ``c`` from token keys ``k``, gate scores ``g`` and the slot bias ``ape``:

- ``'key_dtype'``: transcribes the Hugging Face reference (transformers ``fee0f2b9bd``,
  ``Glm5NextTextIndexer.get_pooled_states``): an FP32 softmax over the pool's slots of
  ``g + ape`` per channel, cast to the key dtype, then the product and the slot sum in the key
  dtype;
- ``'fp32'``: the documented FP32 profile: the same softmax weights, an FP32 weighted sum
  divided by the FP32 normalizer, and one BF16 rounding of the pooled key;
- ``'exact'``: the FP32 profile's math in a chosen dtype without any rounding, as the
  gradient reference.
"""

from types import SimpleNamespace
from typing import List, Optional, Sequence

import torch

COMPRESSION_PROFILES = ("key_dtype", "fp32", "exact")


# ---------------------------------------------------------------------------------------
# Pool partition, causality and the tail
# ---------------------------------------------------------------------------------------


def document_starts(seq_len: int, cu_seqlens: Optional[Sequence[int]] = None) -> List[int]:
    """The first token of each row's document (0 without packing)."""
    if cu_seqlens is None:
        return [0] * seq_len
    bounds = [int(value) for value in cu_seqlens]
    if bounds[0] != 0 or bounds[-1] != seq_len:
        raise ValueError(f"cu_seqlens must cover [0, {seq_len}), got {bounds}")
    starts = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        starts.extend([start] * (end - start))
    return starts


def pool_token_base(
    seq_len: int, pool_size: int, cu_seqlens: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """The first token of each of the ``seq_len // pool_size`` pool slots, ``-1`` if unused.

    Without packing, slot ``c`` is the pool ``[c·P, c·P + P)``. With packing, each document
    contributes its complete pools from its own first token, in document order, and the
    remaining slots are unused capacity.
    """
    slots = seq_len // pool_size
    if cu_seqlens is None:
        return torch.arange(slots, dtype=torch.int64) * pool_size
    bounds = [int(value) for value in cu_seqlens]
    bases = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        base = start
        while base + pool_size <= end:
            bases.append(base)
            base += pool_size
    assert len(bases) <= slots
    bases.extend([-1] * (slots - len(bases)))
    return torch.tensor(bases, dtype=torch.int64)


def visible_pools(row: int, base: Sequence[int], pool_size: int, doc_start: int) -> List[int]:
    """``C(t)``: complete pools of the row's document whose last token is at or before it."""
    return [
        pool
        for pool, first in enumerate(base)
        if first >= 0 and first >= doc_start and first + pool_size - 1 <= row
    ]


def tail_tokens(row: int, pool_size: int, doc_start: int) -> List[int]:
    """``T(t)``: the tokens after the row's last complete pool, up to and including the row."""
    count = (row - doc_start + 1) % pool_size
    return list(range(row - count + 1, row + 1))


def pool_tokens(pool: int, base: Sequence[int], pool_size: int) -> List[int]:
    """``tokens(c)``."""
    return list(range(base[pool], base[pool] + pool_size))


def random_pool_selection(
    generator: torch.Generator,
    *,
    seq_len: int,
    batch: int,
    pool_size: int,
    index_topk: int,
    base: torch.Tensor,
    cu_seqlens: Optional[Sequence[int]] = None,
    interleave_pads: bool = False,
) -> torch.Tensor:
    """A selection ``S(t) ⊆ C(t)`` per row: int64 ``[batch, seq_len, G]``, any slot order.

    Each row keeps ``min(G, |C(t)|)`` distinct visible pools in random order, as a selector
    may emit them, padded with ``-1``. ``interleave_pads`` scatters the pads among the valid
    slots instead of leaving them at the end.
    """
    budget = index_topk // pool_size
    base_list = base.tolist()
    starts = document_starts(seq_len, cu_seqlens)
    out = torch.full((batch, seq_len, budget), -1, dtype=torch.int64)
    for b in range(batch):
        for row in range(seq_len):
            candidates = visible_pools(row, base_list, pool_size, starts[row])
            count = min(budget, len(candidates))
            order = torch.randperm(len(candidates), generator=generator)[:count].tolist()
            chosen = [candidates[i] for i in order]
            slots = (
                torch.randperm(budget, generator=generator)[:count].tolist()
                if interleave_pads
                else list(range(count))
            )
            for slot, pool in zip(slots, chosen):
                out[b, row, slot] = pool
    return out


def expand_selection(
    pool_ids: torch.Tensor,
    base: torch.Tensor,
    pool_size: int,
    index_topk: int,
    cu_seqlens: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Token ids ``[batch, seq_len, index_topk + P - 1]`` (int64, ``-1`` pads).

    Pool slot ``j`` holds ``tokens(c)`` in columns ``j·P .. j·P + P - 1``; the last ``P - 1``
    columns hold ``T(t)``.
    """
    batch, seq_len, budget = pool_ids.shape
    assert budget * pool_size == index_topk
    base_list = base.tolist()
    starts = document_starts(seq_len, cu_seqlens)
    out = torch.full((batch, seq_len, index_topk + pool_size - 1), -1, dtype=torch.int64)
    ids = pool_ids.tolist()
    for b in range(batch):
        for row in range(seq_len):
            for slot, pool in enumerate(ids[b][row]):
                if pool >= 0:
                    out[b, row, slot * pool_size : (slot + 1) * pool_size] = torch.tensor(
                        pool_tokens(pool, base_list, pool_size)
                    )
            tail = tail_tokens(row, pool_size, starts[row])
            if tail:
                out[b, row, index_topk : index_topk + len(tail)] = torch.tensor(tail)
    return out


def canonical_slots(
    token_topk: torch.Tensor, pool_ids: torch.Tensor, *, index_topk: int, pool_size: int
):
    """Valid pools sorted by id (stable, ``-1`` last), each with its ``P`` token columns.

    Returns ``(pool_ids_c, token_topk_c)`` as new tensors in the input dtypes. The tail
    columns stay where they are.
    """
    batch, seq_len, budget = pool_ids.shape
    ids = pool_ids.tolist()
    tokens = token_topk.tolist()
    out_ids, out_tokens = [], []
    for b in range(batch):
        row_ids, row_tokens = [], []
        for row in range(seq_len):
            order = sorted(
                range(budget), key=lambda slot: (ids[b][row][slot] < 0, ids[b][row][slot])
            )
            row_ids.append([ids[b][row][slot] for slot in order])
            regrouped = []
            for slot in order:
                regrouped.extend(tokens[b][row][slot * pool_size : (slot + 1) * pool_size])
            row_tokens.append(regrouped + tokens[b][row][index_topk:])
        out_ids.append(row_ids)
        out_tokens.append(row_tokens)
    return (
        torch.tensor(out_ids, dtype=pool_ids.dtype).reshape(pool_ids.shape),
        torch.tensor(out_tokens, dtype=token_topk.dtype).reshape(token_topk.shape),
    )


# ---------------------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------------------


def compress_keys(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    base: torch.Tensor,
    pool_size: int,
    *,
    profile: str,
    exact_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Pooled keys ``[num_slots, batch, D]`` for ``k``/``gate_score`` ``[tokens, batch, D]``.

    Unused capacity slots (``base < 0``) are zero; nothing may select them. Differentiable in
    ``k``, ``gate_score`` and ``ape`` under the caller's grad mode.
    """
    if profile not in COMPRESSION_PROFILES:
        raise ValueError(f"unknown compression profile {profile!r}")
    tokens, batch, dim = k.shape
    slots = base.numel()
    valid = (base >= 0).view(slots, 1, 1)
    if slots == 0 or tokens == 0:
        dtype = {"key_dtype": k.dtype, "fp32": torch.bfloat16, "exact": exact_dtype}[profile]
        return k.new_zeros((0, batch, dim), dtype=dtype)
    positions = base.clamp_min(0).unsqueeze(-1) + torch.arange(pool_size)
    positions = positions.clamp_max(tokens - 1)
    keys = k[positions]  # [slots, P, batch, D]
    gates = gate_score[positions]
    if profile == "exact":
        logits = gates.to(exact_dtype) + ape.to(exact_dtype)[None, :, None, :]
        probabilities = torch.softmax(logits, dim=1)
        pooled = (probabilities * keys.to(exact_dtype)).sum(dim=1)
    elif profile == "key_dtype":
        logits = gates.float() + ape.float()[None, :, None, :]
        probabilities = torch.nan_to_num(logits.softmax(dim=1)).to(keys.dtype)
        pooled = (probabilities * keys).sum(dim=1)
    else:
        logits = gates.float() + ape.float()[None, :, None, :]
        shifted = torch.exp(logits - logits.amax(dim=1, keepdim=True).detach())
        numerator = (shifted * keys.float()).sum(dim=1)
        pooled = (numerator / shifted.sum(dim=1).clamp_min(1e-12)).to(torch.bfloat16)
    return torch.where(valid, pooled, torch.zeros_like(pooled))


# ---------------------------------------------------------------------------------------
# Student, teacher and loss
# ---------------------------------------------------------------------------------------


def selected_pool_scores(
    q: torch.Tensor,
    weights: torch.Tensor,
    k_pooled: torch.Tensor,
    pool_ids: torch.Tensor,
    *,
    hf_scoring: bool,
    use_relu: bool = True,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """``I [batch, seq_len, G]`` at the selected pools; ``0`` at unused (``-1``) slots.

    ``q`` is ``[seq_len, batch, H, D]``, ``weights`` ``[seq_len, batch, H]`` and ``k_pooled``
    ``[num_slots, batch, D]``.
    """
    seq_len, batch, heads, dim = q.shape
    budget = pool_ids.shape[-1]
    valid = pool_ids >= 0
    if seq_len == 0 or k_pooled.size(0) == 0:
        zero = (q.to(dtype).sum() + weights.to(dtype).sum()) * 0
        return zero + torch.zeros(batch, seq_len, budget, dtype=dtype)
    keys = k_pooled.to(dtype).permute(1, 0, 2)  # [batch, slots, D]
    gathered = keys[torch.arange(batch).view(batch, 1, 1), pool_ids.clamp_min(0)]
    dots = torch.einsum("sbhd,bsgd->bshg", q.to(dtype), gathered)
    if hf_scoring:
        dots = dots * dim**-0.5
    if use_relu:
        dots = torch.relu(dots)
    scores = torch.einsum("sbh,bshg->bsg", weights.to(dtype), dots)
    return scores.masked_fill(~valid, 0.0)


def sparse_pool_target(
    query: torch.Tensor,
    key: torch.Tensor,
    pool_ids: torch.Tensor,
    base: torch.Tensor,
    pool_size: int,
    *,
    softmax_scale: float,
    cu_seqlens: Optional[Sequence[int]] = None,
    heads: Optional[Sequence[int]] = None,
    tail_in_denominator: bool = True,
    dtype: torch.dtype = torch.float64,
) -> SimpleNamespace:
    """The teacher of every row, from the main attention's detached query and key.

    ``query`` is ``[seq_len, batch, np, hn]`` and ``key`` ``[seq_len, batch, nk, hn]`` with
    ``nk == 1`` (absorbed MLA latent key) or ``nk == np``. ``heads`` restricts the head sum to
    a subset (one tensor-parallel rank's heads); the normalization still runs over that sum.
    ``tail_in_denominator=False`` drops ``T(t)`` from ``A(t)``: a per-head renormalization over
    the selected pools, which is NOT the specified teacher (used as a contrast).

    Returns ``target [b, sq, G]``, ``row_has_mass [b, sq]``, ``tail_mass [b, sq]`` (``p_E``:
    the head-summed tail probability over the head-summed total) and the unnormalized
    head-summed pool mass ``pool_mass [b, sq, G]``, all in ``dtype``.
    """
    seq_len, batch, num_heads, _ = query.shape
    head_list = list(range(num_heads)) if heads is None else list(heads)
    budget = pool_ids.shape[-1]
    base_list = base.tolist()
    starts = document_starts(seq_len, cu_seqlens)
    ids = pool_ids.tolist()
    query_d = query.detach().to(dtype)
    key_d = key.detach().to(dtype)
    if key_d.size(2) == 1:
        key_d = key_d.expand(-1, -1, num_heads, -1)
    pool_mass = torch.zeros(batch, seq_len, budget, dtype=dtype)
    tail_mass = torch.zeros(batch, seq_len, dtype=dtype)
    for b in range(batch):
        for row in range(seq_len):
            groups = [
                (slot, pool_tokens(pool, base_list, pool_size))
                for slot, pool in enumerate(ids[b][row])
                if pool >= 0
            ]
            tail = tail_tokens(row, pool_size, starts[row]) if tail_in_denominator else []
            attended = [token for _, tokens in groups for token in tokens] + tail
            if not attended or not head_list:
                continue
            index = torch.tensor(attended)
            # [heads, |A|] logits of this row.
            q_row = query_d[row, b, head_list]
            k_rows = key_d[index, b][:, head_list]
            logits = softmax_scale * torch.einsum("hd,jhd->hj", q_row, k_rows)
            probabilities = torch.exp(logits - torch.logsumexp(logits, dim=-1, keepdim=True))
            column = 0
            for slot, tokens in groups:
                pool_mass[b, row, slot] = probabilities[:, column : column + len(tokens)].sum()
                column += len(tokens)
            total = probabilities.sum()
            if tail and total > 0:
                tail_mass[b, row] = probabilities[:, column:].sum() / total
    row_mass = pool_mass.sum(dim=-1, keepdim=True)
    target = pool_mass / torch.where(row_mass > 0, row_mass, torch.ones_like(row_mass))
    return SimpleNamespace(
        target=target,
        row_has_mass=row_mass.squeeze(-1) > 0,
        tail_mass=tail_mass,
        pool_mass=pool_mass,
    )


def sparse_kl_loss(
    target: torch.Tensor,
    scores: torch.Tensor,
    pool_ids: torch.Tensor,
    row_has_mass: torch.Tensor,
    *,
    row_count,
    loss_coeff: float,
) -> torch.Tensor:
    """``coeff · Σ_rows has_mass · KL(target ‖ softmax over the valid slots of I) / row_count``."""
    valid = pool_ids >= 0
    any_valid = valid.any(dim=-1, keepdim=True)
    masked = scores.masked_fill(~valid, float("-inf"))
    masked = torch.where(any_valid, masked, torch.zeros_like(masked))
    log_predict = torch.log_softmax(masked, dim=-1).masked_fill(~valid, 0.0)
    target = target.to(scores.dtype)
    positive = valid & (target > 0)
    safe_target = torch.where(positive, target, torch.ones_like(target))
    terms = torch.where(
        positive, target * (torch.log(safe_target) - log_predict), torch.zeros_like(target)
    )
    per_row = terms.sum(dim=-1) * row_has_mass.to(scores.dtype)
    if isinstance(row_count, torch.Tensor):
        row_count = row_count.to(scores.dtype)
    return loss_coeff * per_row.sum() / row_count


def sparse_indexer_loss(
    *,
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    pool_ids: torch.Tensor,
    pool_size: int,
    softmax_scale: float,
    hf_scoring: bool,
    use_relu: bool,
    profile: str,
    loss_coeff: float,
    row_count,
    cu_seqlens: Optional[Sequence[int]] = None,
    row_mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float64,
) -> SimpleNamespace:
    """The whole loss from token-level inputs: compression, student, teacher, KL.

    Differentiable in ``q``, ``weights``, ``k``, ``gate_score`` and ``ape``; the teacher is
    detached. ``row_mask`` ``[b, sq]`` drops padded rows (the token loss's
    ``query_valid_rows``); ``row_count`` is the caller's divisor. Returns the loss with the
    intermediate ``k_pooled``, ``scores`` and the teacher.
    """
    seq_len = k.size(0)
    base = pool_token_base(seq_len, pool_size, cu_seqlens)
    k_pooled = compress_keys(
        k, gate_score, ape, base, pool_size, profile=profile, exact_dtype=dtype
    )
    scores = selected_pool_scores(
        q, weights, k_pooled, pool_ids, hf_scoring=hf_scoring, use_relu=use_relu, dtype=dtype
    )
    teacher = sparse_pool_target(
        query, key, pool_ids, base, pool_size, softmax_scale=softmax_scale, cu_seqlens=cu_seqlens
    )
    has_mass = teacher.row_has_mass
    if row_mask is not None:
        has_mass = has_mass & row_mask
    loss = sparse_kl_loss(
        teacher.target.to(dtype),
        scores,
        pool_ids,
        has_mass,
        row_count=row_count,
        loss_coeff=loss_coeff,
    )
    return SimpleNamespace(
        loss=loss,
        k_pooled=k_pooled,
        scores=scores,
        teacher=teacher,
        base=base,
        row_has_mass=has_mass,
    )
