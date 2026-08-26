# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_rule/fused_recurrent.py` in flash-linear-attention
# v0.5.1 (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""Fused recurrent Gated Delta Rule step, used by the decode path.

Gated Delta Product decode reaches this kernel by folding the `M` Householder
copies into the sequence dimension, so a single decode token becomes an
`M`-length sequence with the query placed on the last copy and the decay on the
first; the caller slices the answer back out.
"""

import torch

from .common import HAVE_TRITON, exp, tl, triton


@triton.heuristics(
    {
        'USE_G': lambda args: args['g'] is not None,
        'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
        'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
        'HAS_STATE_INDICES': lambda args: args['state_indices'] is not None,
        'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    }
)
@triton.jit(do_not_specialize=['T'])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    state_indices,
    state_slot_stride,
    state_head_stride,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    HAS_STATE_INDICES: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Walk one sequence token by token, carrying the `[K, V]` state."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
    # Dynamic batching addresses a persistent per-request cache by slot; a
    # padding request carries -1, reads no state and writes none. Static
    # batching keeps the dense layout, where request i owns row i.
    if HAS_STATE_INDICES:
        i_s = tl.load(state_indices + i_n).to(tl.int64)
        state_offset = i_s * state_slot_stride + i_hv * state_head_stride
    else:
        i_s = i_n
        state_offset = i_nh * K * V

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if USE_G:
        p_g = g + bos * HV + i_hv
    if IS_BETA_HEADWISE:
        p_beta = beta + bos * HV + i_hv
    else:
        p_beta = beta + (bos * HV + i_hv) * V + o_v

    p_o = o + (bos * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE and i_s >= 0:
        p_h0 = h0 + state_offset + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in tl.range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta).to(tl.float32)
        else:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)

        if USE_G:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)

        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        if USE_G:
            p_g += HV
        p_beta += HV * (1 if IS_BETA_HEADWISE else V)
        p_o += HV * V

    if STORE_FINAL_STATE and i_s >= 0:
        p_ht = ht + state_offset + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the recurrent Gated Delta Rule forward pass.

    Args:
        q: Queries `[B, T, H, K]`.
        k: Keys `[B, T, H, K]`.
        v: Values `[B, T, HV, V]`.
        g: Log decays `[B, T, HV]`, or `None`.
        beta: Betas `[B, T, HV]` (head-wise) or `[B, T, HV, V]`.
        scale: Score scale; defaults to `K ** -0.5`.
        initial_state: Starting state `[N, HV, K, V]`, or `None` for zeros.
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: Whether to L2-normalize `q` and `k` in-kernel.
        cu_seqlens: Sequence boundaries `[N+1]` for variable-length input.
        state: `[S, HV, K, V]` per-request state cache for dynamic batching,
            read and written in place at `state_indices`. Supersedes
            `initial_state` / `output_final_state`, which gather and scatter a
            dense state instead.
        state_indices: `[N]` cache slot per sequence; `-1` marks a padding
            request, whose output is zeroed and whose state is untouched.

    Returns `(o, final_state)` with `o` shaped like `v`. When `state` is given,
    `final_state` is that same cache tensor, updated in place.
    """
    assert HAVE_TRITON, "fused_recurrent_gated_delta_rule_update requires Triton"
    # The kernel indexes with raw pointer arithmetic and would read garbage from
    # a strided input, so force every tensor argument contiguous.
    q, k, v, beta = q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous()
    if g is not None:
        g = g.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V))
    NV = triton.cdiv(V, BV)
    if scale is None:
        scale = K**-0.5

    # Slot indices without a cache to index would leave the slot/head strides at
    # zero below, aliasing every request onto slot 0 while the padding mask still
    # runs -- wrong results behind well-formed output. The reverse is fine:
    # `state` with no indices is the static-batching identity mapping.
    assert (
        state_indices is None or state is not None
    ), "state_indices requires the state cache it indexes into"

    o = torch.empty_like(v)
    if state is not None:
        assert state.shape[1:] == (HV, K, V), (
            f"state is expected to have shape [num_slots, {HV}, {K}, {V}], "
            f"got {tuple(state.shape)}"
        )
        assert (
            state.stride(3) == 1 and state.stride(2) == V
        ), "the last two dimensions of the state cache must be contiguous"
        # One cache, read at the top of the step and written at the bottom.
        initial_state = final_state = state
    else:
        final_state = q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None

    fused_recurrent_gated_delta_rule_fwd_kernel[(NV, N * HV)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        state_indices=state_indices,
        state_slot_stride=state.stride(0) if state is not None else 0,
        state_head_stride=state.stride(1) if state is not None else 0,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )
    if state_indices is not None:
        # A padding row's recurrence ran over whatever the padded input buffer
        # held, so its output is overwritten rather than merely left unwritten:
        # the contract is zero, and a stale inf/NaN would survive a mask.
        assert cu_seqlens is None, "state_indices with cu_seqlens is not supported yet"
        o.masked_fill_((state_indices < 0).view(-1, *([1] * (o.ndim - 1))), 0)
    return o, final_state
