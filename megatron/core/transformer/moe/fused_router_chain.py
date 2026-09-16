# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The sigmoid group-limited top-k router chain as ONE autograd Function (experiment line "fuse").

Eager Megatron (``moe_router_fusion=False``) computes a router call through ~30 small torch ops
-- ``torch.sigmoid``, ``+ expert_bias``, ``group_limited_topk`` (view / topk / sum / topk /
scatter / expand / masked_fill / topk), ``gather``, the normalization, the scaling, two dense
``scatter``s, and for the sequence-wise balance loss a second sigmoid / normalization / topk /
scatter, the counts' column sum (all-reduced over the aux-loss group), ``probs.sum(0)`` and the
dot product -- and autograd derives their backward op by op (a comparable number of kernels).
TransformerEngine's ``moe_router_fusion`` replaces them by THREE Functions
(``fused_topk_with_score_function``, ``fused_compute_score_for_moe_aux_loss``,
``fused_moe_aux_loss``) that still meet at the logits through autograd's fanout accumulation.

This module owns the WHOLE chain in one ``torch.autograd.Function``: the forward is one Triton
kernel per token row (sigmoid, bias, group scores, group and expert ranks, normalization, the
dense probs and map or the dense top-k indices) plus the balance-loss statistics (a row-block
kernel for the per-expert counts and score column sums, the counts all-reduced over the aux-loss
groups, a finalize kernel for the loss); the backward is ONE kernel from the dense probs gradient
and the aux loss gradient straight to the logits gradient (the normalization's Jacobian on the
picked experts, the balance loss's Jacobian, their sum, the sigmoid's derivative).  Kernel text
adapted from the mtfga product's fused router ops (``src/mtfga/ops/router.py``, line "fuse").

Scope (fail closed, the caller falls back to the eager path): ``score_function == "sigmoid"``,
``moe_router_load_balancing_type == "seq_aux_loss"`` (the plain / global aux losses and the
z-loss off), no expert capacity / token dropping, no padding mask, no packed sequences, no hash
or replay routing, micro batch 1 per sequence-aux group, powers of two for the expert count, the
group count and topk (nano 8 / 2 / 2, medium 64 / 8 / 8).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - the Triton kernels are the CUDA path only
    triton = None
    tl = None

_AUX_BLOCK_R = 16
_AUX_BLOCK_P = 32


def _pow2(*values: int) -> bool:
    return all(v > 0 and (v & (v - 1)) == 0 for v in values)


def fused_router_chain_applicable(
    logits: torch.Tensor,
    topk: int,
    num_groups: Optional[int],
    group_topk: Optional[int],
    score_function: str,
    bsz: int,
) -> bool:
    """Whether the fused chain can take this call (shapes and settings; the caller checks the
    routing-type / padding / capacity settings)."""
    if triton is None or not logits.is_cuda or logits.dim() != 2:
        return False
    if score_function != "sigmoid" or bsz != 1:
        return False
    e = logits.shape[1]
    groups = num_groups or 1
    gt = group_topk or 1
    if not _pow2(e, groups, topk) or e % groups or topk % gt or topk > (e // groups) * gt:
        return False
    return True


# ---------------------------------------------------------------------------- the math twin
#
# The same closed forms in torch, on any device: the oracle of the kernels and the CPU check of
# the backward derivation against autograd of Megatron's eager text (tests/unit_tests/.../
# test_fused_router_chain.py).


def math_forward(logits, expert_bias, topk, num_groups, group_topk, scaling_factor):
    """(probs_dense [T, E] fp32, indices [T, k] i64, scores [T, E] fp32) as the eager text."""
    num_tokens, num_experts = logits.shape
    scores = torch.sigmoid(logits.float())
    bias = expert_bias.float() if expert_bias is not None else torch.zeros_like(scores[0])
    biased = scores + bias
    groups = num_groups or 1
    gt = group_topk or 1
    if groups > 1 or gt > 1:
        group_scores = (
            biased.view(num_tokens, groups, -1).topk(topk // gt, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=gt, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, groups, num_experts // groups)
            .reshape(num_tokens, -1)
        )
        masked = biased.masked_fill(~score_mask.bool(), float("-inf"))
        _, indices = torch.topk(masked, k=topk, dim=-1)
    else:
        _, indices = torch.topk(biased, k=topk, dim=-1)
    picked = torch.gather(scores, dim=1, index=indices)
    probs = picked / (picked.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else picked
    if scaling_factor:
        probs = probs * scaling_factor
    probs_dense = torch.zeros_like(scores).scatter(1, indices, probs)
    return probs_dense, indices, scores


def math_aux_stats(scores, topk):
    """(local counts [E] fp32, column sums of the normalized scores [E] fp32) of the eager
    ``compute_routing_scores_for_aux_loss`` + ``switch_load_balancing_loss_func`` inputs."""
    a = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    _, top = torch.topk(a, k=topk, dim=1)
    counts = torch.zeros_like(a).scatter(1, top, 1.0).sum(dim=0)
    return counts, a.sum(dim=0)


def math_aux_loss(colsum, counts_global, total_tokens, topk, num_experts, coeff):
    return torch.sum(colsum * counts_global) * (
        num_experts * coeff / (topk * total_tokens * total_tokens)
    )


def math_backward(
    grad_probs_dense, grad_aux, scores, indices, counts_global, scaling_factor, aux_scale
):
    """The closed-form logits gradient: the normalization's Jacobian on the picked experts
    (``group_topk_vjp``) + the balance loss's Jacobian (``seq_aux_loss_vjp``), through the
    sigmoid.  ``aux_scale`` = E * coeff / (topk * total * total)."""
    w = scores.gather(1, indices)
    denom = w.sum(-1, keepdim=True) + 1e-20
    g_norm = grad_probs_dense.gather(1, indices) * scaling_factor
    g_wsel = g_norm / denom - (g_norm * w).sum(-1, keepdim=True) / denom.pow(2)
    g_scores = torch.zeros_like(scores).scatter_(1, indices, g_wsel)
    if grad_aux is not None:
        denom_s = scores.sum(-1, keepdim=True) + 1e-20
        a = scores / denom_s
        factor = grad_aux * aux_scale
        g_scores = g_scores + factor / denom_s * (
            counts_global[None, :] - (a * counts_global[None, :]).sum(-1, keepdim=True)
        )
    return g_scores * scores * (1 - scores)


# ------------------------------------------------------------------------------ the kernels

if triton is not None:

    @triton.jit
    def _router_chain_fwd_kernel(
        X,
        B,
        S,
        P,
        M,
        IDX,
        stride_x,
        stride_s,
        stride_p,
        stride_m,
        stride_i,
        route_scale,
        E: tl.constexpr,
        G: tl.constexpr,
        GS: tl.constexpr,
        KG: tl.constexpr,
        GT: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # one token per program: the sigmoid scores (kept for the backward), biased scores, the
        # group scores (sum of the KG best biased scores of a group), the GT best groups, the K
        # best experts among them; ranks by pairwise comparison (descending, equal scores by the
        # lower index); out go the dense probs, the dense bool map and the top-k indices
        row = tl.program_id(0)
        cols = tl.arange(0, E)
        x = tl.load(X + row * stride_x + cols).to(tl.float32)
        s = tl.sigmoid(x)
        tl.store(S + row * stride_s + cols, s)
        if HAS_BIAS:
            biased = s + tl.load(B + cols).to(tl.float32)
        else:
            biased = s
        ci = cols[:, None]
        cj = cols[None, :]
        bi = biased[:, None]
        bj = biased[None, :]
        same_group = (ci // GS) == (cj // GS)
        before = same_group & ((bj > bi) | ((bj == bi) & (cj < ci)))
        grank = tl.sum(before.to(tl.int32), axis=1)
        top_in_group = tl.where(grank < KG, biased, 0.0)
        gids = tl.arange(0, G)
        member = gids[:, None] == (cols // GS)[None, :]
        gscore = tl.sum(tl.where(member, top_in_group[None, :], 0.0), axis=1)
        gi = gscore[:, None]
        gj = gscore[None, :]
        gci = gids[:, None]
        gcj = gids[None, :]
        gbefore = (gj > gi) | ((gj == gi) & (gcj < gci))
        grk = tl.sum(gbefore.to(tl.int32), axis=1)
        keep_g = (grk < GT).to(tl.int32)
        keep_e = tl.sum(tl.where(member, keep_g[:, None], 0), axis=0) > 0
        masked = tl.where(keep_e, biased, float("-inf"))
        mi = masked[:, None]
        mj = masked[None, :]
        ebefore = (mj > mi) | ((mj == mi) & (cj < ci))
        erank = tl.sum(ebefore.to(tl.int32), axis=1)
        sel = erank < K
        w = tl.where(sel, s, 0.0)
        denom = tl.sum(w, axis=0) + 1e-20
        weights = s / denom * route_scale
        tl.store(P + row * stride_p + cols, tl.where(sel, weights, 0.0))
        tl.store(M + row * stride_m + cols, sel.to(M.dtype.element_ty))
        tl.store(IDX + row * stride_i + erank, cols.to(tl.int64), mask=sel)

    @triton.jit(do_not_specialize=["rows"])
    def _aux_stats_kernel(
        S, PC, PS, rows, stride_s, E: tl.constexpr, K: tl.constexpr, BLOCK_R: tl.constexpr
    ):
        # row block pid: the scores normalized over the experts, a plain top-k of them; out go
        # the block's per-expert counts (exact integers in fp32) and normalized-score column sums
        pid = tl.program_id(0)
        cols = tl.arange(0, E)
        ci = cols[:, None]
        cj = cols[None, :]
        cnt = tl.zeros((E,), dtype=tl.float32)
        csum = tl.zeros((E,), dtype=tl.float32)
        for r in range(BLOCK_R):
            row = pid * BLOCK_R + r
            valid = row < rows
            row_c = tl.minimum(row, rows - 1)
            s = tl.load(S + row_c * stride_s + cols)
            a = s / (tl.sum(s, axis=0) + 1e-20)
            ai = a[:, None]
            aj = a[None, :]
            before = (aj > ai) | ((aj == ai) & (cj < ci))
            rank = tl.sum(before.to(tl.int32), axis=1)
            cnt += ((rank < K) & valid).to(tl.float32)
            csum += tl.where(valid, a, 0.0)
        tl.store(PC + pid * E + cols, cnt)
        tl.store(PS + pid * E + cols, csum)

    @triton.jit(do_not_specialize=["parts"])
    def _aux_loss_kernel(PS, COUNTS, OUT, parts, scale, E: tl.constexpr, BLOCK_P: tl.constexpr):
        # one program: the column sums over the blocks, dotted with the (global) counts, scaled
        cols = tl.arange(0, E)
        ps = tl.arange(0, BLOCK_P)
        csum = tl.zeros((BLOCK_P, E), dtype=tl.float32)
        for start in range(0, parts, BLOCK_P):
            rws = start + ps
            mask = (rws < parts)[:, None] & (cols < E)[None, :]
            csum += tl.load(PS + rws[:, None] * E + cols[None, :], mask=mask, other=0.0)
        colsum = tl.sum(csum, axis=0)
        counts = tl.load(COUNTS + cols)
        tl.store(OUT, tl.sum(colsum * counts, axis=0) * scale)

    @triton.jit
    def _router_chain_bwd_kernel(
        GP,
        GAUX,
        S,
        IDX,
        COUNTS,
        GX,
        stride_gp,
        stride_s,
        stride_i,
        stride_gx,
        route_scale,
        aux_scale,
        E: tl.constexpr,
        K: tl.constexpr,
        HAS_AUX: tl.constexpr,
    ):
        # one token per program: the dense probs gradient gathered at the picked experts, the
        # normalization's Jacobian scattered back, the balance loss's Jacobian against the global
        # counts, their sum through the sigmoid's derivative
        row = tl.program_id(0)
        cols = tl.arange(0, E)
        ks = tl.arange(0, K)
        s = tl.load(S + row * stride_s + cols)
        idx = tl.load(IDX + row * stride_i + ks)
        hit = cols[None, :].to(tl.int64) == idx[:, None]  # (K, E)
        gp_row = tl.load(GP + row * stride_gp + cols).to(tl.float32)
        w = tl.sum(tl.where(hit, s[None, :], 0.0), axis=1)  # (K,)
        gw = tl.sum(tl.where(hit, gp_row[None, :], 0.0), axis=1)  # (K,)
        denom = tl.sum(w, axis=0) + 1e-20
        g_norm = gw * route_scale
        dot = tl.sum(g_norm * w, axis=0)
        g_wsel = g_norm / denom - dot / (denom * denom)
        g_scores = tl.sum(tl.where(hit, g_wsel[:, None], 0.0), axis=0)  # (E,)
        if HAS_AUX:
            counts = tl.load(COUNTS + cols)
            denom_s = tl.sum(s, axis=0) + 1e-20
            a = s / denom_s
            factor = tl.load(GAUX).to(tl.float32) * aux_scale
            g_scores = g_scores + factor / denom_s * (counts - tl.sum(a * counts, axis=0))
        gx = g_scores * s * (1.0 - s)
        tl.store(GX + row * stride_gx + cols, gx.to(GX.dtype.element_ty))


# ----------------------------------------------------------------------------- the Function


class FusedSigmoidRouterChain(torch.autograd.Function):
    """logits [T, E] (+ expert bias) -> dense routing probs (logits dtype), the routing map
    (dense bool [T, E] or dense top-k indices [T, k] i64) and the sequence-wise balance loss
    (fp32 scalar; a zero scalar detached from nothing when ``aux_coeff`` is 0).  One Function:
    its backward turns the probs gradient and the aux loss gradient into the logits gradient."""

    @staticmethod
    def forward(
        ctx,
        logits: torch.Tensor,
        expert_bias: Optional[torch.Tensor],
        topk: int,
        num_groups: Optional[int],
        group_topk: Optional[int],
        scaling_factor: Optional[float],
        aux_coeff: float,
        aux_reduce_groups: Sequence[torch.distributed.ProcessGroup],
        dense_indices: bool,
    ):
        rows, e = logits.shape
        groups = num_groups or 1
        gt = group_topk or 1
        scale = float(scaling_factor) if scaling_factor else 1.0
        dev = logits.device
        scores = torch.empty((rows, e), dtype=torch.float32, device=dev)
        probs = torch.empty((rows, e), dtype=torch.float32, device=dev)
        map_u8 = torch.empty((rows, e), dtype=torch.uint8, device=dev)
        indices = torch.empty((rows, topk), dtype=torch.int64, device=dev)
        logits_c = logits if logits.stride(-1) == 1 else logits.contiguous()
        if rows:
            with torch.cuda.device(dev):
                _router_chain_fwd_kernel[(rows,)](
                    logits_c,
                    expert_bias if expert_bias is not None else scores,
                    scores,
                    probs,
                    map_u8,
                    indices,
                    logits_c.stride(0),
                    scores.stride(0),
                    probs.stride(0),
                    map_u8.stride(0),
                    indices.stride(0),
                    scale,
                    E=e,
                    G=groups,
                    GS=e // groups,
                    KG=topk // gt,
                    GT=gt,
                    K=topk,
                    HAS_BIAS=expert_bias is not None,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        world = 1
        for group in aux_reduce_groups:
            world *= torch.distributed.get_world_size(group)
        total = rows * world
        aux_scale = float(e * aux_coeff / (topk * total * total)) if aux_coeff and rows else 0.0
        counts = None
        if aux_coeff and rows:
            parts = triton.cdiv(rows, _AUX_BLOCK_R)
            pc = torch.empty((parts, e), dtype=torch.float32, device=dev)
            ps = torch.empty((parts, e), dtype=torch.float32, device=dev)
            with torch.cuda.device(dev):
                _aux_stats_kernel[(parts,)](
                    scores,
                    pc,
                    ps,
                    rows,
                    scores.stride(0),
                    E=e,
                    K=topk,
                    BLOCK_R=_AUX_BLOCK_R,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            counts = pc.sum(dim=0)  # exact: integers below 2^24
            for group in aux_reduce_groups:
                torch.distributed.all_reduce(counts, group=group)
            aux_loss = torch.empty((), dtype=torch.float32, device=dev)
            with torch.cuda.device(dev):
                _aux_loss_kernel[(1,)](
                    ps,
                    counts,
                    aux_loss,
                    parts,
                    aux_scale,
                    E=e,
                    BLOCK_P=_AUX_BLOCK_P,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
        else:
            aux_loss = torch.zeros((), dtype=torch.float32, device=dev)
        ctx.save_for_backward(scores, indices, counts)
        ctx.scale = scale
        ctx.aux_scale = aux_scale
        ctx.logits_dtype = logits.dtype
        routing_map = indices if dense_indices else map_u8.view(torch.bool)
        ctx.mark_non_differentiable(routing_map)
        return probs.to(logits.dtype), routing_map, aux_loss

    @staticmethod
    def backward(ctx, grad_probs, grad_map, grad_aux):
        scores, indices, counts = ctx.saved_tensors
        rows, e = scores.shape
        k = indices.shape[1]
        gx = torch.empty((rows, e), dtype=ctx.logits_dtype, device=scores.device)
        has_aux = counts is not None and grad_aux is not None
        if grad_probs is None:
            grad_probs = torch.zeros((rows, e), dtype=torch.float32, device=scores.device)
        gp = grad_probs if grad_probs.stride(-1) == 1 else grad_probs.contiguous()
        if rows:
            with torch.cuda.device(scores.device):
                _router_chain_bwd_kernel[(rows,)](
                    gp,
                    grad_aux if has_aux else scores,
                    scores,
                    indices,
                    counts if has_aux else scores,
                    gx,
                    gp.stride(0),
                    scores.stride(0),
                    indices.stride(0),
                    gx.stride(0),
                    ctx.scale,
                    ctx.aux_scale,
                    E=e,
                    K=k,
                    HAS_AUX=has_aux,
                    num_warps=1,
                    enable_fp_fusion=False,
                )
        return gx, None, None, None, None, None, None, None, None


def fused_sigmoid_router_chain(
    logits, expert_bias, topk, num_groups, group_topk, scaling_factor, aux_coeff,
    aux_reduce_groups, dense_indices,
):
    """``FusedSigmoidRouterChain.apply`` with the keyword-free argument order spelled once."""
    return FusedSigmoidRouterChain.apply(
        logits, expert_bias, topk, num_groups, group_topk, scaling_factor, aux_coeff,
        tuple(aux_reduce_groups), dense_indices,
    )
