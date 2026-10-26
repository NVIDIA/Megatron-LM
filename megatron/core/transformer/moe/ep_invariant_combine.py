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
                if leaf != cur_leaf:
                    if cur_leaf >= 0:
                        meet = 31 - libdevice.clz(cur_leaf ^ leaf)  # level h - 1
                        acc = _fold_below(acc, pending, occupied, meet, LEVELS)
                        for level in tl.static_range(LEVELS):
                            if level == meet:
                                pending = _replace(pending, level, acc)
                        occupied = (occupied & ~((1 << (meet + 1)) - 1)) | (1 << meet)
                    acc = zero
                    cur_leaf = leaf
                # A leaf adds its rows in column order, starting from +0.0.
                acc = acc + v
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


def _per_row(values: torch.Tensor, slot_row: torch.Tensor, num_rows: int, fill) -> torch.Tensor:
    """[T, K] per-slot values -> [num_rows] per permuted row (``fill`` for rows of no slot).

    Each permuted row belongs to at most one valid slot; invalid slots (row -1) land in a
    spare last entry, which is dropped.
    """
    out = torch.full((num_rows + 1,), fill, dtype=values.dtype, device=values.device)
    out.scatter_(0, torch.where(slot_row >= 0, slot_row, num_rows).flatten(), values.flatten())
    return out[:num_rows]


class _TreeCombine(torch.autograd.Function):
    """``tree_combine_forward`` over all experts, with the gradient of the plain sum."""

    @staticmethod
    def forward(ctx, rows, row_of_slot, expert_of_slot, num_experts, probs):
        # pylint: disable=missing-function-docstring
        slots = dense_slots(row_of_slot, expert_of_slot, num_experts)
        out = tree_combine_forward(rows.contiguous(), slots, num_leaf_blocks(num_experts), probs)
        valid = (row_of_slot >= 0) & (expert_of_slot >= 0) & (expert_of_slot < num_experts)
        slot_row = torch.where(valid, row_of_slot, -1)
        ctx.save_for_backward(rows, slot_row, expert_of_slot.clamp(0, num_experts - 1), probs)
        return out

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=missing-function-docstring
        rows, slot_row, expert, probs = ctx.saved_tensors
        num_tokens, topk = slot_row.shape
        grad = grad.to(torch.float32)
        # d out[t] / d row = prob of the row's (token, expert) slot: one gather per row.
        token = torch.arange(num_tokens, device=grad.device)[:, None].expand(-1, topk)
        token_of_row = _per_row(token, slot_row, rows.shape[0], num_tokens)
        grad_ext = torch.cat([grad, grad.new_zeros(1, grad.shape[1])])
        d_rows = grad_ext.index_select(0, token_of_row)
        d_probs = None
        if probs is not None:
            slot_prob = probs.gather(1, expert).float()
            d_rows *= _per_row(slot_prob, slot_row, rows.shape[0], 0.0)[:, None]
            # d out[t] / d probs[t, e] = the (token, e) row; spare column for invalid slots.
            d_probs = grad.new_zeros(num_tokens, probs.shape[1] + 1)
            safe = slot_row.clamp_min(0)
            for k in range(topk):
                dot = (grad * rows.index_select(0, safe[:, k]).float()).sum(-1)
                col = torch.where(slot_row[:, k] >= 0, expert[:, k], probs.shape[1])
                d_probs.scatter_(1, col[:, None], dot[:, None])
            d_probs = d_probs[:, :-1].to(probs.dtype)
        return d_rows.to(rows.dtype), None, None, None, d_probs


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
