# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The batch-invariant MoE combine as its contract defines it, for the rank-tree unpermute
oracle (not a test module).

The batch-invariant unpermute (``megatron/core/transformer/moe/batch_invariant.py``) must give
the bits generation's NVLS combine gives, so its contract fixes every rounding point
(``unpermute``'s docstring and the reference-loop comments in that file):

* each EP rank's partial adds the token's routes on that rank in slot order, from zero: in
  fp64 and rounded once to fp32 (the unit-weight path, ``probs=None``), or, for the FlashInfer
  finalize (``unpermute_with_bf16_rank_partials``), as fp32 fused multiply-adds of probability
  and row rounded to bf16;
* the ranks are added in ascending order: an fp32 chain for the ``ordered`` collective, an fp64
  chain rounded once to fp32 for ``multimem``;
* the fp32 result is rounded to the rows' dtype.

The functions here evaluate that definition with torch elementwise ops (IEEE, round to nearest
even) on whole token columns; they share no indexing or blocking with the Triton kernel.
"""

from typing import Optional

import torch


def fma_f32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Correctly rounded fp32 ``a * b + c`` (one rounding) for fp32 tensors.

    ``a * b`` is exact in fp64 (24 + 24 significant bits). ``s = fl64(a * b + c)`` and its
    rounding error ``err`` (TwoSum) represent the exact sum. Rounding ``s`` to fp32 is the
    correct result unless ``s`` lies exactly halfway between two fp32 values, where the sign of
    ``err`` decides (the exact sum cannot cross any other fp32 midpoint, which are fp64 values).
    """
    p = a.double() * b.double()
    c64 = c.double()
    s = p + c64
    t = s - p
    err = (p - (s - t)) + (c64 - t)
    r = s.float()
    d = s - r.double()
    toward = torch.where(d > 0, float("inf"), float("-inf")).to(torch.float32)
    n = torch.nextafter(r, toward)
    tie = (d != 0) & (s == (r.double() + n.double()) / 2)
    beyond = tie & (((d > 0) & (err > 0)) | ((d < 0) & (err < 0)))
    return torch.where(beyond, n, r)


def rank_tree_combine(
    rows: torch.Tensor,
    inverse_rows: torch.Tensor,
    inverse_experts: torch.Tensor,
    num_experts: int,
    ep_size: int,
    *,
    cross_rank_fp64: bool,
    probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[t]`` of the batch-invariant combine, in ``rows.dtype`` (``[T, H]``).

    ``rows`` ``[R, H]``: the expert outputs (rows no valid route names are never read: their
    values must not matter). ``inverse_rows`` / ``inverse_experts`` ``[T, topk]``: per token and
    slot, the row and the global expert, ``-1`` where there is none. A slot counts on rank ``r``
    when its row is ``>= 0`` and its expert lies in ``[r * E / ep, (r + 1) * E / ep)``. With
    ``probs`` (fp32 ``[T, topk]``) the rank partials are the bf16-rounded FMA chains; without,
    the fp64 sums rounded to fp32.
    """
    tokens, topk = inverse_rows.shape
    hidden = rows.shape[1]
    per_rank = num_experts // ep_size
    acc_dtype = torch.float64 if cross_rank_fp64 else torch.float32
    acc = torch.zeros(tokens, hidden, dtype=acc_dtype, device=rows.device)
    for rank in range(ep_size):
        lo, hi = rank * per_rank, (rank + 1) * per_rank
        part_dtype = torch.float64 if probs is None else torch.float32
        part = torch.zeros(tokens, hidden, dtype=part_dtype, device=rows.device)
        for k in range(topk):
            row, expert = inverse_rows[:, k], inverse_experts[:, k]
            take = ((row >= 0) & (expert >= lo) & (expert < hi))[:, None]
            values = rows[row.clamp(min=0)]
            if probs is None:
                summed = part + values.double()
            else:
                weight = probs[:, k : k + 1].float().expand(tokens, hidden)
                summed = fma_f32(values.float(), weight, part)
            part = torch.where(take, summed, part)
        part32 = part.float() if probs is None else part.bfloat16().float()
        acc = acc + part32.to(acc_dtype)
    return acc.float().to(rows.dtype)
