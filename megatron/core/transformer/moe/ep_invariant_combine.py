# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""EP-size-invariant MoE combine (``TransformerConfig.moe_ep_invariant_combine``).

The batch-invariant combine sums a token's expert outputs per EP rank and then adds
the rank partials in rank order, so the result depends on the expert-parallel size.
Here every expert output value v is mapped to two fp64 integer limbs

    hi = floor(v),  lo = round((v - hi) * 2^48),  v ~= hi + lo * 2^-48

and the token's limbs are summed. With at most ``MAX_TOPK`` terms per token and
``|hi| <= HI_LIMIT`` every partial sum is an integer below 2^53, so each fp64 addition
is exact: any grouping of a token's rows (per local expert, per EP rank, any
reduction order) gives the same two sums. ``finalize`` rounds hi + lo * 2^-48 once
to fp32 (correctly rounded, via round-to-odd in fp64).

v equals hi + lo * 2^-48 exactly when v is a multiple of 2^-48 (e.g. bf16 values
with |v| >= 2^-41, or outputs of GEMMs with fp16 operands); finer bits are rounded
to that grid, and |v| >= 2^47 saturates, both deterministically per row, so the
result stays EP-invariant. inf / nan rows propagate through ``hi``.
"""

import torch

FRAC_BITS = 48
MAX_TOPK = 32
# k <= MAX_TOPK rows with |hi| <= 2^47 keep |sum(hi)| <= 2^52; lo < 2^48 each keeps
# sum(lo) <= 2^53.
HI_LIMIT = 2.0**47
_SCALE = 2.0**FRAC_BITS


def check_topk(topk: int) -> None:
    """Raise if a token's limb sums could round (more than ``MAX_TOPK`` rows)."""
    if topk > MAX_TOPK:
        raise ValueError(
            f"moe_ep_invariant_combine supports moe_router_topk <= {MAX_TOPK} "
            f"(got {topk}): the fp64 limb sums must stay below 2^53."
        )


def to_limbs(x: torch.Tensor) -> torch.Tensor:
    """[..., H] values -> [..., 2H] fp64 limbs (hi | lo)."""
    v = x.to(torch.float64)
    finite = torch.isfinite(v)
    big = v.abs() >= HI_LIMIT
    hi = torch.where(big, torch.sign(v) * HI_LIMIT, torch.floor(v))
    lo = torch.where(big, torch.zeros_like(v), torch.round((v - hi) * _SCALE))
    hi = torch.where(finite, hi, v)
    lo = torch.where(finite, lo, torch.zeros_like(v))
    return torch.cat([hi, lo], dim=-1)


def finalize(limbs: torch.Tensor) -> torch.Tensor:
    """[..., 2H] fp64 limb sums -> [..., H] fp32, hi + lo * 2^-48 correctly rounded."""
    hi, lo = limbs.chunk(2, dim=-1)
    b = lo * (1.0 / _SCALE)  # exact
    s = hi + b
    bb = s - hi
    err = (hi - (s - bb)) + (b - bb)  # TwoSum: hi + b == s + err exactly
    # Round to odd in fp64, then round to nearest in fp32: correctly rounded.
    bits = s.view(torch.int64)
    nudge = (err != 0) & ((bits & 1) == 0)
    step = torch.where((err > 0) == (s > 0), 1, -1)
    s = torch.where(nudge, bits + step, bits).view(torch.float64)
    return torch.where(torch.isfinite(hi), s, hi).to(torch.float32)


class _Finalize(torch.autograd.Function):
    """``finalize`` with the gradient of hi + lo * 2^-48 w.r.t. the value (d/d hi = 1)."""

    @staticmethod
    def forward(ctx, limbs):  # pylint: disable=missing-function-docstring
        return finalize(limbs)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=missing-function-docstring
        g = grad.to(torch.float64)
        return torch.cat([g, torch.zeros_like(g)], dim=-1)


def finalize_with_grad(limbs: torch.Tensor) -> torch.Tensor:
    """Autograd-aware ``finalize``."""
    return _Finalize.apply(limbs)


class _SlotCombine(torch.autograd.Function):
    """out[t] = finalize(sum_k limbs(rows[slot_rows[t, k]] * slot_probs[t, k]))."""

    @staticmethod
    def forward(ctx, rows, slot_rows, slot_probs, num_tokens):
        # pylint: disable=missing-function-docstring
        hidden = rows.shape[-1]
        acc = torch.zeros(num_tokens, 2 * hidden, dtype=torch.float64, device=rows.device)
        valid = slot_rows >= 0
        safe = slot_rows.clamp_min(0)
        for k in range(slot_rows.shape[1]):  # one [T, 2H] limb buffer at a time
            v = rows.index_select(0, safe[:, k]).to(torch.float32)
            if slot_probs is not None:
                v = v * slot_probs[:, k : k + 1].to(torch.float32)
            v = v.masked_fill(~valid[:, k : k + 1], 0.0)
            acc += to_limbs(v)
        ctx.save_for_backward(rows, slot_rows, slot_probs)
        return finalize(acc)

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=missing-function-docstring
        rows, slot_rows, slot_probs = ctx.saved_tensors
        grad = grad.to(torch.float32)
        valid = slot_rows >= 0
        safe = slot_rows.clamp_min(0)
        d_rows = torch.zeros(rows.shape, dtype=torch.float32, device=rows.device)
        d_probs = None if slot_probs is None else torch.zeros_like(slot_probs)
        for k in range(slot_rows.shape[1]):
            g = grad.masked_fill(~valid[:, k : k + 1], 0.0)
            if slot_probs is not None:
                d_probs[:, k] = (g * rows.index_select(0, safe[:, k]).float()).sum(-1)
                g = g * slot_probs[:, k : k + 1].to(torch.float32)
            # Each permuted row belongs to at most one (token, slot).
            d_rows.index_add_(0, safe[valid[:, k], k], g[valid[:, k]])
        return d_rows.to(rows.dtype), None, d_probs, None


def slot_combine(rows, slot_rows, slot_probs, num_tokens):
    """Combine each token's top-k slot rows (``slot_rows`` [T, K], -1 = none) to fp32 [T, H].

    ``slot_probs`` ([T, K] or None) multiplies each row in fp32 first. The result does
    not depend on how the rows were distributed over EP ranks.
    """
    return _SlotCombine.apply(rows, slot_rows, slot_probs, num_tokens)


class _LimbPartial(torch.autograd.Function):
    """[R, H] rows -> [T, 2H] fp64 limb partial: out[token_of_row[r]] += limbs(rows[r])."""

    @staticmethod
    def forward(ctx, rows, token_of_row, num_tokens):
        # pylint: disable=missing-function-docstring
        acc = torch.zeros(num_tokens, 2 * rows.shape[-1], dtype=torch.float64, device=rows.device)
        acc.index_add_(0, token_of_row, to_limbs(rows))  # integer sums: exact in any order
        ctx.save_for_backward(token_of_row)
        ctx.dtype = rows.dtype
        return acc

    @staticmethod
    def backward(ctx, grad):  # pylint: disable=missing-function-docstring
        (token_of_row,) = ctx.saved_tensors
        hidden = grad.shape[-1] // 2
        return grad[:, :hidden].index_select(0, token_of_row).to(ctx.dtype), None, None


def limb_partial(rows: torch.Tensor, token_of_row: torch.Tensor, num_tokens: int):
    """Autograd-aware fp64 limb partial of this rank's expert rows per token."""
    return _LimbPartial.apply(rows, token_of_row.to(torch.long), num_tokens)
