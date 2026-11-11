# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Kimi Delta Attention on cudnn-frontend's native kernels, behind FLA's ``chunk_kda`` call.

Chunkwise context parallelism keeps FLA's framework (``fla/ops/cp/chunk_delta_h.py``) and swaps
cuDNN's kernels in:

    forward   summary of the last local sequence  -> one all_gather -> fold the predecessor
              ranks into the first local sequence's initial state -> KDA forward
    backward  summary of the first local sequence -> one all_gather -> fold the successor
              ranks into the last local sequence's final-state gradient -> KDA backward

A span summary is the affine map ``S -> S @ M + H`` that the span applies to the recurrent state
(``S``, ``H``: ``[HO, V, K]``, ``M``: ``[HO, K, K]``, fp32). Ranks exchange it packed as
``[HO, V + K, K]``, and the fold of the gathered maps is the one kernel this module adds.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from cudnn.linear_attention.ops.kda import (
    kda_bwd,
    kda_fwd,
    kimi_delta_attention,
    kimi_delta_attention_summary,
    kimi_delta_attention_summary_bwd,
)

# Value rows per fold program. The TF32 dot of a [16, 128] state block with a [128, 128]
# transition fits shared memory; [64, 128] does not.
_FOLD_BV = 16


@triton.jit(do_not_specialize=["first", "count", "step"])
def _fold_kernel(
    hm,  # [R, HO, V + K, K] gathered maps
    state,  # [HO, V, K]
    first,
    count,
    step,
    HO: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BV: tl.constexpr,
):
    """Compose ``S <- S @ M_r + H_r`` over ranks ``first, first + step, ...`` from ``S = 0``.

    As FLA's merge kernel: fp32 accumulation with the product on TF32 tensor cores.
    """
    i_v, i_h = tl.program_id(0), tl.program_id(1)
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    m_v = o_v < V
    b_s = tl.zeros([BV, K], dtype=tl.float32)
    for i in range(count):
        p_hm = hm + ((first + i * step) * HO + i_h) * (V + K) * K
        b_h = tl.load(p_hm + o_v[:, None] * K + o_k[None, :], mask=m_v[:, None], other=0.0)
        b_m = tl.load(p_hm + (V + o_k[:, None]) * K + o_k[None, :])
        b_s = tl.dot(b_s, b_m) + b_h
    tl.store(state + i_h * V * K + o_v[:, None] * K + o_k[None, :], b_s, mask=m_v[:, None])


def _fold(hm: torch.Tensor, state: torch.Tensor, first: int, count: int, step: int) -> None:
    """Fold the gathered maps of ranks ``first, first + step, ...`` (``count`` of them) into
    ``state``."""
    _, HO, VK, K = hm.shape
    V = VK - K
    grid = (triton.cdiv(V, _FOLD_BV), HO)
    _fold_kernel[grid](hm, state, first, count, step, HO=HO, V=V, K=K, BV=_FOLD_BV)


def _all_gather(x: torch.Tensor, group) -> torch.Tensor:
    out = x.new_empty(dist.get_world_size(group), *x.shape)
    dist.all_gather_into_tensor(out, x, group=group)
    return out


class _ChunkwiseCPKDA(torch.autograd.Function):
    """KDA on this rank's contiguous shard of the context-parallel token stream.

    ``cu_seqlens`` holds the rank-local sequence boundaries. ``context`` (``FLACPContext``) tells
    whether the first local sequence started on earlier ranks and whether the last one continues
    on later ranks; all the others are complete on this rank.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, a_log, dt_bias, cu_seqlens, scale, context, flags):
        """Seed the first local sequence with the state the earlier ranks leave, then run KDA."""
        HO, V, K = g.shape[1], v.shape[2], k.shape[2]
        local_cu = context.cu_seqlens
        if context.is_last_rank:
            hm = q.new_zeros(HO, V + K, K, dtype=torch.float32)
        else:
            # Only the last local sequence continues on later ranks. Its boundaries select it in
            # the whole shard, so that the plan shape does not depend on its length.
            h, m = kimi_delta_attention_summary(
                k, v, g, beta, local_cu[-2:], a_log=a_log, dt_bias=dt_bias, **flags
            )
            hm = torch.cat((h[0], m[0]), dim=1)
        hm = _all_gather(hm, context.group)

        initial_state = q.new_zeros(cu_seqlens.numel() - 1, HO, V, K, dtype=torch.float32)
        if not context.is_first_rank:
            rank = dist.get_rank(context.group)
            _fold(hm, initial_state[0], rank - context.pre_num_ranks, context.pre_num_ranks, 1)

        o = kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            a_log=a_log,
            dt_bias=dt_bias,
            **flags,
        )[0]
        # Only the first sequence has an initial state: keep that slot alone, as FLA does.
        ctx.save_for_backward(
            q, k, v, g, beta, a_log, dt_bias, cu_seqlens, initial_state[:1].clone()
        )
        ctx.scale, ctx.context, ctx.flags = scale, context, flags
        return o

    @staticmethod
    def backward(ctx, do):
        """Mirror of the forward, with the state gradient the later ranks send back."""
        q, k, v, g, beta, a_log, dt_bias, cu_seqlens, state0 = ctx.saved_tensors
        scale, context, flags = ctx.scale, ctx.context, ctx.flags
        HO, V, K = g.shape[1], v.shape[2], k.shape[2]
        local_cu = context.cu_seqlens
        do = do.contiguous()
        if context.is_first_rank:
            gm = q.new_zeros(HO, V + K, K, dtype=torch.float32)
        else:
            # Only the first local sequence started on earlier ranks.
            dh0, mt = kimi_delta_attention_summary_bwd(
                q,
                k,
                g,
                beta,
                do,
                local_cu[:2],
                scale=scale,
                output_transition=True,
                a_log=a_log,
                dt_bias=dt_bias,
                **flags,
            )
            gm = torch.cat((dh0[0], mt[0]), dim=1)
        gm = _all_gather(gm, context.group)

        d_final_state = q.new_zeros(cu_seqlens.numel() - 1, HO, V, K, dtype=torch.float32)
        if not context.is_last_rank:
            rank = dist.get_rank(context.group)
            _fold(
                gm,
                d_final_state[local_cu.numel() - 2],
                rank + context.post_num_ranks,
                context.post_num_ranks,
                -1,
            )
        initial_state = torch.zeros_like(d_final_state)
        initial_state[:1] = state0

        dq, dk, dv, dg, dbeta, _, d_a_log, d_dt_bias = kda_bwd(
            do,
            q,
            k,
            v,
            g,
            beta,
            cu_seqlens,
            scale,
            initial_state=initial_state,
            d_final_state=d_final_state,
            a_log=a_log,
            dt_bias=dt_bias,
            **flags,
        )
        return dq, dk, dv, dg, dbeta, d_a_log, d_dt_bias, None, None, None, None


def chunk_kda_cudnn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    cp_context=None,
) -> tuple[torch.Tensor, None]:
    """KDA with the arguments ``KimiDeltaAttention`` passes to FLA's ``chunk_kda``.

    Args:
        q, k: ``[B, T, H, K]`` in bf16 or fp16.
        v: ``[B, T, HV, V]``.
        g: ``[B, T, HV, K]`` raw gate logits.
        beta: ``[B, T, HV]`` after the sigmoid.
        A_log, dt_bias: ``[HV]`` and ``[HV * K]`` parameters of the bounded gate
            ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``, the only gate cuDNN fuses.
        safe_gate: selects a kernel variant in FLA and has no cuDNN counterpart.
        cu_seqlens: ``[N + 1]`` boundaries of the packed sequences (``B == 1``).
        cp_context: ``FLACPContext`` for chunkwise context parallelism. Its rank-local
            boundaries replace ``cu_seqlens``, as in FLA.

    Returns:
        ``(o, None)`` with ``o`` of shape ``[B, T, HV, V]``.
    """
    if initial_state is not None or output_final_state:
        raise NotImplementedError("chunk_kda_cudnn does not take or return recurrent states.")
    if not use_gate_in_kernel or lower_bound is None:
        raise ValueError(
            "chunk_kda_cudnn supports only the bounded gate computed in the kernel: pass raw "
            "gate logits with use_gate_in_kernel=True and a lower_bound."
        )
    B, T, HO, K = g.shape
    flags = {
        "use_qk_l2norm_in_kernel": bool(use_qk_l2norm_in_kernel),
        "safe_gate": True,
        "gate_lower_bound": float(lower_bound),
    }
    q, k, v, g, beta = (x.flatten(0, 1) for x in (q, k, v, g, beta))
    dt_bias = dt_bias.view(HO, K)
    if cu_seqlens is None:
        cu_seqlens = torch.arange(B + 1, device=q.device, dtype=torch.int32) * T

    if cp_context is None:
        o, _ = kimi_delta_attention(
            q, k, v, g, beta, cu_seqlens, a_log=A_log, dt_bias=dt_bias, **flags
        )
    else:
        # cuDNN builds one plan per sequence count: zero-length tail sequences round the
        # rank-local count up to a power of two.
        local_cu = cp_context.cu_seqlens
        num_seqs = 1 << (local_cu.numel() - 2).bit_length()
        cu_seqlens = torch.cat((local_cu, local_cu[-1:].expand(num_seqs + 1 - local_cu.numel())))
        o = _ChunkwiseCPKDA.apply(
            q, k, v, g, beta, A_log, dt_bias, cu_seqlens, K**-0.5, cp_context, flags
        )
    return o.unflatten(0, (B, T)), None
