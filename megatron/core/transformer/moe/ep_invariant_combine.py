# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""EP-size-invariant MoE combine (``TransformerConfig.moe_ep_invariant_combine``).

The batch-invariant combine sums a token's expert outputs per EP rank, rounds that
partial to fp32 and adds the rank partials in rank order, so where the fp32 roundings
fall depends on the expert-parallel size. Here every EP size evaluates one fixed fp32
reduction tree over the expert ids instead:

- The E experts form G leaf blocks of m = E / G consecutive experts, where G is the
  largest power of two dividing E. A leaf is the fp32 sum of the token's rows from
  its block, added in expert order starting from +0.0.
- The output is a pairwise fp32 tree over the G leaves by block id: leaves 2j and
  2j + 1 are added, then their parents, and so on.

With a power-of-two EP size, rank r owns the experts [r * E / EP, (r + 1) * E / EP),
which is an aligned subtree. The inference engine evaluates each rank's subtree to an
fp32 partial and its reduce-scatter (``pairwise_tree=True``) finishes the tree over the
rank partials in the same pairwise order; training's AllToAll combine already has all
of a token's rows and evaluates the whole tree. Every EP size thus performs the same
fp32 additions and gives the same bits, at the cost of the existing fp32 combine.
Adding an empty (+0.0) subtree is exact, so empty leaves and ranks do not change the
result; starting each leaf from +0.0 also turns -0.0 into +0.0 consistently.
"""

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False

# Tree depth supported by _tree_combine_kernel's 32-bit occupancy mask: at most 2^10 leaves.
_MAX_LEVELS = 10


def num_leaf_blocks(num_experts: int) -> int:
    """G: the largest power of two dividing ``num_experts``."""
    return num_experts & -num_experts


def check_ep_size(num_experts: int, ep_size: int) -> None:
    """Raise unless every EP rank owns an aligned subtree of the combine tree."""
    if ep_size & (ep_size - 1) or num_experts % ep_size:
        raise ValueError(
            "moe_ep_invariant_combine requires a power-of-two expert_model_parallel_size "
            f"that divides num_moe_experts (got EP={ep_size}, experts={num_experts})."
        )
    if num_leaf_blocks(num_experts) > 1 << _MAX_LEVELS:
        raise ValueError(
            f"moe_ep_invariant_combine supports at most {1 << _MAX_LEVELS} leaf blocks "
            f"(got {num_leaf_blocks(num_experts)} for {num_experts} experts)."
        )


@triton.jit
def _replace(t, i: tl.constexpr, v):
    """Tuple t with element i replaced by v."""
    return t[:i] + (v,) + t[i + 1 :]


@triton.jit
def _fold_below(acc, pending, occupied, top, LEVELS: tl.constexpr):
    """Add the non-empty pending left subtrees at levels < top onto acc, lowest level first:
    each one is the left sibling of the subtree acc has grown into so far."""
    for level in tl.static_range(LEVELS):
        if level < top:
            if ((occupied >> level) & 1) == 1:
                acc = pending[level] + acc
    return acc


@triton.jit
def _add_to_leaf(acc, pending, occupied, cur_leaf, leaf, v, LEVELS: tl.constexpr):
    """Add row ``v`` of ``leaf`` (leaves visited in increasing order) to the tree walk.

    ``acc`` is the subtree grown from the current leaf; pending[l] (non-empty iff bit l of
    ``occupied``) is a left sibling at level l waiting for it. When the next routed leaf b
    follows leaf a, the two meet at height h = bit_length(a ^ b): the pending subtrees
    below level h - 1 fold onto acc, which becomes the pending left sibling at level
    h - 1, and b starts a new acc. A leaf adds its rows in column order from +0.0.
    """
    if leaf != cur_leaf:
        if cur_leaf >= 0:
            meet = 31 - libdevice.clz(cur_leaf ^ leaf)  # level h - 1
            acc = _fold_below(acc, pending, occupied, meet, LEVELS)
            for level in tl.static_range(LEVELS):
                if level == meet:
                    pending = _replace(pending, level, acc)
            occupied = (occupied & ~((1 << (meet + 1)) - 1)) | (1 << meet)
        acc = tl.zeros_like(acc)
        cur_leaf = leaf
    return acc + v, pending, occupied, cur_leaf


@triton.jit
def _tree_combine_kernel(
    rows_ptr,  # [R, H] expert output rows (bf16 or fp32)
    slots_ptr,  # [T, S] int32 row of each (token, column), -1 = none
    probs_ptr,  # [T, S] fp32 per-(token, column) weight (HAS_PROBS)
    valid_tokens_ptr,  # scalar int32: tokens >= this get zeros (HAS_VALID_TOKENS)
    out_ptr,  # [T, H] fp32
    hidden,
    HAS_PROBS: tl.constexpr,
    HAS_VALID_TOKENS: tl.constexpr,
    NUM_LEAVES: tl.constexpr,  # power of two
    LEVELS: tl.constexpr,  # log2(NUM_LEAVES)
    LEAF_SIZE: tl.constexpr,  # columns per leaf: S = NUM_LEAVES * LEAF_SIZE
    WORD: tl.constexpr,  # columns per occupancy bitmask: min(64, next power of two >= S)
    BLOCK_H: tl.constexpr,
):
    """One program per (token, hidden tile): the token's subtree over the NUM_LEAVES leaves.

    The pairwise tree over all leaves equals the tree over the token's non-empty leaves:
    empty subtrees are +0.0 and no node is ever -0.0, so adding one is an exact no-op.
    The kernel visits the token's routed columns in order (an occupancy bitmask per WORD
    columns, walked with ffs), so its work scales with the routed rows like the plain
    expert-order unpermute. ``acc`` is the subtree grown from the current
    leaf; pending[l] (non-empty iff bit l of ``occupied``) is a left sibling at level l
    waiting for it. When the next routed leaf b follows leaf a, the two meet at height
    h = bit_length(a ^ b): the pending subtrees below level h - 1 fold onto acc, which
    becomes the pending left sibling at level h - 1, and b starts a new acc.
    """
    tok = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs < hidden
    slot_base = tok * (NUM_LEAVES * LEAF_SIZE)
    zero = tl.zeros([BLOCK_H], dtype=tl.float32)
    pending = ()
    for _ in tl.static_range(LEVELS):
        pending = pending + (zero,)
    occupied = tl.full([], 0, tl.int32)
    acc = zero
    cur_leaf = tl.full([], -1, tl.int32)
    live = True
    if HAS_VALID_TOKENS:
        live = tok < tl.load(valid_tokens_ptr)
    if live:
        # Visit only the routed columns: a bitmask per WORD columns, walked with ffs.
        bit_ids = tl.arange(0, WORD)
        for word in tl.range(0, NUM_LEAVES * LEAF_SIZE, WORD):
            cols = word + bit_ids
            word_rows = tl.load(
                slots_ptr + slot_base + cols, mask=cols < NUM_LEAVES * LEAF_SIZE, other=-1
            )
            bits = tl.sum(tl.where(word_rows >= 0, tl.full([WORD], 1, tl.int64) << bit_ids, 0))
            while bits != 0:
                col = word + libdevice.ffs(bits) - 1
                bits = bits & (bits - 1)
                row = tl.load(slots_ptr + slot_base + col).to(tl.int64)
                v = tl.load(rows_ptr + row * hidden + offs, mask=mask_h, other=0.0)
                v = v.to(tl.float32)
                if HAS_PROBS:
                    # mul_rn: a separately rounded product, never contracted into an FMA.
                    v = libdevice.mul_rn(v, tl.load(probs_ptr + slot_base + col))
                leaf = (col // LEAF_SIZE).to(tl.int32)
                acc, pending, occupied, cur_leaf = _add_to_leaf(
                    acc, pending, occupied, cur_leaf, leaf, v, LEVELS
                )
        acc = _fold_below(acc, pending, occupied, LEVELS, LEVELS)
    tl.store(out_ptr + tok * hidden + offs, acc, mask=mask_h)


def tree_combine_forward(
    rows: torch.Tensor,
    slots: torch.Tensor,
    num_leaves: int,
    probs: Optional[torch.Tensor] = None,
    valid_tokens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """fp32 [T, H]: the combine subtree over ``slots`` [T, S] (row ids, -1 = none).

    Column j of ``slots`` belongs to leaf j // (S / num_leaves); columns must be in
    expert order. ``probs`` [T, S] multiplies each row in fp32 first. Rows of tokens at
    or beyond ``valid_tokens`` (a scalar int32 CUDA tensor) are zero.
    """
    assert HAVE_TRITON, "moe_ep_invariant_combine requires Triton."
    num_tokens, num_cols = slots.shape
    hidden = rows.shape[-1]
    assert num_leaves & (num_leaves - 1) == 0 and num_cols % num_leaves == 0
    assert num_leaves <= 1 << _MAX_LEVELS
    slots = slots.to(torch.int32).contiguous()
    if probs is not None:
        probs = probs.to(torch.float32).contiguous()
    if out is None:
        out = torch.empty(num_tokens, hidden, dtype=torch.float32, device=rows.device)
    if num_tokens == 0:
        return out
    block_h = min(triton.next_power_of_2(hidden), 1024)
    grid = (num_tokens, triton.cdiv(hidden, block_h))
    _tree_combine_kernel[grid](
        rows,
        slots,
        probs if probs is not None else slots,
        valid_tokens if valid_tokens is not None else slots,
        out,
        hidden,
        HAS_PROBS=probs is not None,
        HAS_VALID_TOKENS=valid_tokens is not None,
        NUM_LEAVES=num_leaves,
        LEVELS=num_leaves.bit_length() - 1,
        LEAF_SIZE=num_cols // num_leaves,
        WORD=min(64, triton.next_power_of_2(num_cols)),
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


def dense_slots(row_of_slot: torch.Tensor, expert_of_slot: torch.Tensor, num_experts: int):
    """[T, K] (row, expert) top-k slots -> [T, num_experts] row per expert (-1 = none)."""
    num_tokens = row_of_slot.shape[0]
    valid = (row_of_slot >= 0) & (expert_of_slot >= 0) & (expert_of_slot < num_experts)
    # Invalid slots scatter into a spare last column, which is dropped.
    cols = torch.where(valid, expert_of_slot, num_experts).long()
    dense = torch.full(
        (num_tokens, num_experts + 1), -1, dtype=torch.int32, device=row_of_slot.device
    )
    dense.scatter_(1, cols, torch.where(valid, row_of_slot, -1).to(torch.int32))
    return dense[:, :num_experts].contiguous()


@triton.jit
def _tree_combine_routes_kernel(
    rows_ptr,  # [R, H] expert output rows (bf16 or fp32)
    cols_ptr,  # [T, K] int32 routed column of each slot, ascending; unused slots last
    slot_rows_ptr,  # [T, K] int32 row of each slot, -1 = unused
    slot_probs_ptr,  # [T, K] fp32 weight of each slot (HAS_PROBS)
    out_ptr,  # [T, H] fp32
    hidden,
    HAS_PROBS: tl.constexpr,
    LEVELS: tl.constexpr,  # log2(number of leaves)
    LEAF_SIZE: tl.constexpr,  # columns per leaf
    K: tl.constexpr,  # slots per token
    BLOCK_H: tl.constexpr,
):
    """``_tree_combine_kernel``'s tree, for tokens whose routes are all listed (training).

    The K slot rows do not depend on each other, so all of them are loaded before the
    tree walk and are in flight together; the dense kernel's walk finds the next row
    only after the previous one. Same fp32 additions in the same order, so the same bits.
    """
    tok = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs < hidden
    zero = tl.zeros([BLOCK_H], dtype=tl.float32)
    values = ()
    for j in tl.static_range(K):
        row = tl.load(slot_rows_ptr + tok * K + j)
        v = tl.load(
            rows_ptr + tl.maximum(row, 0).to(tl.int64) * hidden + offs,
            mask=mask_h & (row >= 0),
            other=0.0,
        ).to(tl.float32)
        if HAS_PROBS:
            # mul_rn: a separately rounded product, never contracted into an FMA.
            v = libdevice.mul_rn(v, tl.load(slot_probs_ptr + tok * K + j))
        values = values + (v,)
    pending = ()
    for _ in tl.static_range(LEVELS):
        pending = pending + (zero,)
    occupied = tl.full([], 0, tl.int32)
    acc = zero
    cur_leaf = tl.full([], -1, tl.int32)
    for j in tl.static_range(K):
        if tl.load(slot_rows_ptr + tok * K + j) >= 0:
            leaf = (tl.load(cols_ptr + tok * K + j) // LEAF_SIZE).to(tl.int32)
            acc, pending, occupied, cur_leaf = _add_to_leaf(
                acc, pending, occupied, cur_leaf, leaf, values[j], LEVELS
            )
    acc = _fold_below(acc, pending, occupied, LEVELS, LEVELS)
    tl.store(out_ptr + tok * hidden + offs, acc, mask=mask_h)


@triton.jit
def _tree_combine_backward_kernel(
    grad_ptr,  # [T, H] fp32 output gradient
    rows_ptr,  # [R, H] expert output rows
    slot_rows_ptr,  # [T, K] int32 row of each slot, -1 = unused
    slot_probs_ptr,  # [T, K] fp32 weight of each slot (HAS_PROBS)
    d_rows_ptr,  # [R, H] row gradient (rows of no slot stay zero)
    d_slot_probs_ptr,  # [T, K] fp32 weight gradient (HAS_PROBS)
    hidden,
    HAS_PROBS: tl.constexpr,
    K: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Gradient of the plain weighted sum, one program per token: each slot's row gets
    grad * prob and its weight gets <grad, row>, reading grad and each row once."""
    tok = tl.program_id(0).to(tl.int64)
    for j in tl.static_range(K):
        row = tl.load(slot_rows_ptr + tok * K + j).to(tl.int64)
        if row >= 0:
            dot = tl.zeros([BLOCK_H], dtype=tl.float32)
            for start in tl.range(0, hidden, BLOCK_H):
                offs = start + tl.arange(0, BLOCK_H)
                mask = offs < hidden
                g = tl.load(grad_ptr + tok * hidden + offs, mask=mask, other=0.0)
                d_row = g
                if HAS_PROBS:
                    d_row = g * tl.load(slot_probs_ptr + tok * K + j)
                    r = tl.load(rows_ptr + row * hidden + offs, mask=mask, other=0.0)
                    dot += g * r.to(tl.float32)
                tl.store(
                    d_rows_ptr + row * hidden + offs,
                    d_row.to(d_rows_ptr.dtype.element_ty),
                    mask=mask,
                )
            if HAS_PROBS:
                tl.store(d_slot_probs_ptr + tok * K + j, tl.sum(dot, 0))


def _routes_block_h(hidden: int, num_slots: int) -> int:
    """Hidden tile for the route kernels: all K slot rows live in registers at once, so
    keep K * BLOCK_H / 128 threads at about 96 values per thread (powers of two)."""
    budget = max(128, 1 << ((96 * 128 // max(num_slots, 1)).bit_length() - 1))
    return min(triton.next_power_of_2(hidden), 1024, budget)


def sorted_routes(
    row_of_slot: torch.Tensor,
    expert_of_slot: torch.Tensor,
    num_experts: int,
    probs: Optional[torch.Tensor] = None,
):
    """[T, K] top-k slots -> each token's routes in expert order, unused slots last.

    Returns int32 (expert, row) [T, K] (unused: expert ``num_experts``, row -1) and, with
    ``probs`` [T, num_experts], the fp32 weight of each slot.
    """
    valid = (row_of_slot >= 0) & (expert_of_slot >= 0) & (expert_of_slot < num_experts)
    experts, order = torch.where(valid, expert_of_slot, num_experts).sort(dim=1, stable=True)
    rows = torch.where(experts < num_experts, row_of_slot.gather(1, order), -1)
    slot_probs = None
    if probs is not None:
        slot_probs = probs.gather(1, experts.clamp_max(num_experts - 1)).float().contiguous()
    return experts.to(torch.int32).contiguous(), rows.to(torch.int32).contiguous(), slot_probs


def tree_combine_routes_forward(
    rows: torch.Tensor,
    cols: torch.Tensor,
    slot_rows: torch.Tensor,
    num_cols: int,
    num_leaves: int,
    slot_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """fp32 [T, H]: ``tree_combine_forward`` over each token's listed routes.

    ``cols`` / ``slot_rows`` [T, K] (from ``sorted_routes``): each route's column in
    [0, ``num_cols``), ascending, and its row; ``slot_probs`` [T, K] weights each row.
    Bitwise equal to ``tree_combine_forward`` on the dense map of the same routes.
    """
    assert HAVE_TRITON, "moe_ep_invariant_combine requires Triton."
    assert num_leaves & (num_leaves - 1) == 0 and num_cols % num_leaves == 0
    assert num_leaves <= 1 << _MAX_LEVELS
    num_tokens, num_slots = cols.shape
    hidden = rows.shape[-1]
    out = torch.empty(num_tokens, hidden, dtype=torch.float32, device=rows.device)
    if num_tokens == 0 or num_slots == 0:
        return out.zero_()
    block_h = _routes_block_h(hidden, num_slots)
    _tree_combine_routes_kernel[(num_tokens, triton.cdiv(hidden, block_h))](
        rows,
        cols,
        slot_rows,
        slot_probs if slot_probs is not None else cols,
        out,
        hidden,
        HAS_PROBS=slot_probs is not None,
        LEVELS=num_leaves.bit_length() - 1,
        LEAF_SIZE=num_cols // num_leaves,
        K=num_slots,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return out


class _TreeCombine(torch.autograd.Function):
    """``tree_combine_forward`` over all experts, with the gradient of the plain sum."""

    @staticmethod
    def forward(ctx, rows, row_of_slot, expert_of_slot, num_experts, probs):
        # pylint: disable=missing-function-docstring
        rows = rows.contiguous()
        cols, slot_rows, slot_probs = sorted_routes(row_of_slot, expert_of_slot, num_experts, probs)
        out = tree_combine_routes_forward(
            rows, cols, slot_rows, num_experts, num_leaf_blocks(num_experts), slot_probs
        )
        ctx.save_for_backward(rows, cols, slot_rows, slot_probs)
        ctx.num_experts = num_experts
        ctx.probs_dtype = probs.dtype if probs is not None else None
        return out

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=missing-function-docstring
        rows, cols, slot_rows, slot_probs = ctx.saved_tensors
        num_tokens, num_slots = slot_rows.shape
        hidden = rows.shape[-1]
        d_rows = torch.zeros_like(rows)
        d_slot_probs = None
        if slot_probs is not None:
            d_slot_probs = torch.zeros(num_tokens, num_slots, device=rows.device)
        if num_tokens and num_slots:
            _tree_combine_backward_kernel[(num_tokens,)](
                grad.to(torch.float32).contiguous(),
                rows,
                slot_rows,
                slot_probs if slot_probs is not None else slot_rows,
                d_rows,
                d_slot_probs if d_slot_probs is not None else slot_rows,
                hidden,
                HAS_PROBS=slot_probs is not None,
                K=num_slots,
                BLOCK_H=min(triton.next_power_of_2(hidden), 1024),
                num_warps=4,
            )
        d_probs = None
        if d_slot_probs is not None:
            # Unused slots carry column num_experts: a spare column, dropped.
            d_probs = grad.new_zeros(num_tokens, ctx.num_experts + 1, dtype=torch.float32)
            d_probs.scatter_(1, cols.long(), d_slot_probs)
            d_probs = d_probs[:, :-1].to(ctx.probs_dtype)
        return d_rows, None, None, None, d_probs


def tree_combine(
    rows: torch.Tensor,
    row_of_slot: torch.Tensor,
    expert_of_slot: torch.Tensor,
    num_experts: int,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Autograd-aware combine of each token's top-k slots over the whole tree -> fp32 [T, H].

    ``row_of_slot`` / ``expert_of_slot`` [T, K]: permuted row and global expert of each
    slot (-1 = none). ``probs`` [T, num_experts] multiplies each row in fp32 first.
    """
    return _TreeCombine.apply(rows, row_of_slot, expert_of_slot, num_experts, probs)
