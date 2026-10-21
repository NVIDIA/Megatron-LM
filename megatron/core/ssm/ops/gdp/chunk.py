# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_product/chunk.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in this directory.

"""Chunked Gated Delta Product prefill.

The stage sequence:

1. Interleave the log decays across the Householder copies, then take the
   within-chunk cumulative sum on both the unexpanded and the expanded stream,
   converting to base 2 on the way (the kernels exponentiate with `exp2`).
2. Build the WY representation: `beta * K K^T`, then invert `I + A`, then
   recompute the `w` and `u` factors.
3. Sweep the inter-chunk state recurrence, emitting the chunk-boundary states.
4. Combine those with the within-chunk attention to get the outputs.

Two chunkings are in play throughout. The queries and the outputs live on the
token stream as written; the keys, values and betas live on the
Householder-expanded stream, whose sequences are `M` times longer. The second
chunking is not a rescaling of the first, because `ceil(L*M/64)` is not
`M*ceil(L/64)` unless `L` is a multiple of the chunk size.
"""

import torch

from .chunk_h import chunk_gated_delta_product_fwd_h
from .chunk_o import chunk_gated_delta_product_fwd_o
from .common import CHUNK_SIZE, RCP_LN2, l2norm_fwd, prepare_chunk_indices
from .cumsum import chunk_local_cumsum
from .scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .solve_tril import solve_tril
from .wy_fast import recompute_w_u_fwd


def chunk_gated_delta_product_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variable-length chunked Gated Delta Product forward pass.

    Args:
        q: Queries `[1, T, H, K]`.
        k: Keys `[1, T*M, H, K]` (Householder-expanded).
        v: Values `[1, T*M, H, V]` (Householder-expanded).
        g: Log decays `[1, T, H]`.
        beta: Betas `[1, T*M, H]`.
        num_householder: Number of Householder copies `M`.
        cu_seqlens: Sequence boundaries over the unexpanded stream, `[N+1]`.
        scale: Score scale; defaults to `K ** -0.5`.
        initial_state: Starting state `[N, H, K, V]`, or `None` for zeros.
        output_final_state: Whether to return the final state.
        use_qk_l2norm_in_kernel: Whether to L2-normalize `q` and `k` first.

    Returns `(o, final_state)` with `o` shaped `[1, T, H, V]`.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert q.dtype != torch.float32, "the chunked GDP kernels require bf16/fp16 inputs"
    assert B == 1, f"varlen prefill expects a single packed sequence, got batch {B}"
    assert k.shape == (B, T * num_householder, H, K), f"unexpected key shape {tuple(k.shape)}"
    assert v.shape == (B, T * num_householder, H, V), f"unexpected value shape {tuple(v.shape)}"
    assert beta.shape == (B, T * num_householder, H), f"unexpected beta shape {tuple(beta.shape)}"
    if g is not None:
        assert g.shape == (B, T, H), f"unexpected decay shape {tuple(g.shape)}"
    if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
        raise ValueError(
            "The number of initial states is expected to be equal to the number of input "
            f"sequences, i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
        )
    if scale is None:
        scale = K**-0.5

    # The kernels index with raw pointer arithmetic and would read garbage from
    # a strided input, so force every tensor argument contiguous.
    q, k, v, beta = q.contiguous(), k.contiguous(), v.contiguous(), beta.contiguous()
    if g is not None:
        g = g.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    cu_seqlens_dp = cu_seqlens * num_householder
    # Both chunkings are derived here, once, and threaded through every stage
    # so the sub-kernels do not each re-derive (and re-synchronize on) them.
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    chunk_indices_dp = prepare_chunk_indices(cu_seqlens_dp, CHUNK_SIZE)

    if g is not None:
        # The decay applies to the first Householder copy of each token; the
        # remaining copies are decay-free (zeros in log space).
        g_interleaved = g.new_zeros(B, T, num_householder, H, dtype=torch.float32)
        g_interleaved[:, :, 0] = g
        g_interleaved = g_interleaved.view(B, T * num_householder, H).contiguous()
        # The chunked kernels exponentiate in base 2, so the decays are
        # converted out of natural-log space here rather than in every kernel.
        g = chunk_local_cumsum(
            g,
            chunk_size=CHUNK_SIZE,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
            chunk_indices=chunk_indices,
        )
        g_interleaved = chunk_local_cumsum(
            g_interleaved,
            chunk_size=CHUNK_SIZE,
            scale=RCP_LN2,
            cu_seqlens=cu_seqlens_dp,
            output_dtype=torch.float32,
            chunk_indices=chunk_indices_dp,
        )
    else:
        g_interleaved = None

    # WY representation of the (inverse) transition matrix. u is the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g_interleaved,
        beta=beta,
        cu_seqlens=cu_seqlens_dp,
        chunk_size=CHUNK_SIZE,
        output_dtype=torch.float32,
        chunk_indices=chunk_indices_dp,
    )
    A = solve_tril(
        A=A, cu_seqlens=cu_seqlens_dp, chunk_indices=chunk_indices_dp, output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g_interleaved,
        cu_seqlens=cu_seqlens_dp,
        chunk_indices=chunk_indices_dp,
    )

    h, v_new, final_state = chunk_gated_delta_product_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g_interleaved,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens_dp,
        num_householder=num_householder,
        chunk_size=CHUNK_SIZE,
        chunk_indices=chunk_indices,
    )
    o = chunk_gated_delta_product_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
        num_householder=num_householder,
        chunk_indices=chunk_indices,
    )
    return o.to(q.dtype), final_state
