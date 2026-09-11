# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Forked from `fla/ops/gated_delta_product/chunk.py` in flash-linear-attention v0.5.1
# (https://github.com/fla-org/flash-linear-attention).
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

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
    chunk_indices: torch.Tensor | None = None,
    chunk_indices_dp: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    return_chunk_states: bool = False,
) -> tuple[torch.Tensor, ...]:
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
        chunk_indices: Chunk descriptors for the token stream as written.
        chunk_indices_dp: The same for the Householder-expanded stream, whose
            sequences are `M` times longer. Not a rescaling of `chunk_indices`:
            `ceil(L*M/64) != M*ceil(L/64)` in general.
        chunk_offsets: Per-sequence prefix sum of unexpanded chunk counts.
        state: `[S, H, K, V]` per-request state cache for dynamic batching,
            written in place at `state_indices` rather than returned densely.
        state_indices: `[N]` cache slot per sequence; `-1` marks padding.
        return_chunk_states: Also return the per-chunk states the scan passes
            through, `[NT, H, K, V]`. Row `chunk_offsets[i] + c` is sequence
            `i`'s state *entering* its chunk `c`, i.e. after its first `64 * c`
            tokens -- which is the mid-sequence state prefix caching snapshots.
            Note this differs from the Mamba2 chunk scan, whose raw states are
            indexed by the chunk they come *out* of.

    Returns `(o, final_state)` with `o` shaped `[1, T, H, V]`, or
    `(o, final_state, chunk_states)` when `return_chunk_states` is set.

    Passing the three descriptor arguments is what makes this capturable in a
    CUDA graph: deriving them here reads a device tensor on the host and yields
    a data-dependent length, which also sizes every launch grid below. Built
    once per step and padded to a fixed length by `metadata`, they keep the grids
    constant for a captured batch shape.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    # Slot-indexed state is the dynamic-batching path, which is also the only
    # caller that can be captured in a CUDA graph -- and capture additionally
    # requires the precomputed descriptors. Deriving them below would silently
    # fall back to a host sync and a data-dependent grid, so reject the
    # half-configured call rather than run a graph-unsafe kernel.
    assert (state_indices is None) or (
        chunk_indices is not None and chunk_indices_dp is not None and chunk_offsets is not None
    ), "slot-indexed state requires the precomputed chunk descriptors"
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

    # A device-side multiply of a fixed-size buffer: no sync, no shape change.
    cu_seqlens_dp = cu_seqlens * num_householder
    # Both chunkings are threaded through every stage so the sub-kernels do not
    # each re-derive (and re-synchronize on) them. The caller supplies them for
    # graph capture; otherwise they are derived here, once.
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    if chunk_indices_dp is None:
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

    # When the caller snapshots the per-chunk states for prefix caching, keep
    # `h` in the state-cache precision rather than the kernel's bf16 working
    # dtype: the recurrence accumulates in fp32 and only rounds on store, so a
    # bf16 `h` would snapshot a bf16-rounded state even into an fp32 cache and
    # the restored prefix diverges from an uncached run. MambaMixer passes
    # `state_dtype` to its scan for the same reason. The output path below still
    # consumes a bf16 view of `h`, so `o` is bit-for-bit unchanged.
    chunk_states_dtype = None
    if return_chunk_states:
        # Match the cache the caller will snapshot into. With neither cache nor
        # initial state to match, leave it to the kernel's default (input dtype)
        # rather than silently paying for fp32.
        if state is not None:
            chunk_states_dtype = state.dtype
        elif initial_state is not None:
            chunk_states_dtype = initial_state.dtype

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
        chunk_offsets=chunk_offsets,
        state=state,
        state_indices=state_indices,
        states_dtype=chunk_states_dtype,
    )
    o = chunk_gated_delta_product_fwd_o(
        q=q,
        k=k,
        v=v_new,
        # fp32 -> bf16 round-to-nearest matches the value the scan would have
        # stored directly in bf16, so the output kernel sees the same input.
        h=h if h.dtype == q.dtype else h.to(q.dtype),
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
        num_householder=num_householder,
        chunk_indices=chunk_indices,
    )
    if return_chunk_states:
        # h is [1, NT, H, K, V]; the extraction kernels index by chunk row.
        return o.to(q.dtype), final_state, h.squeeze(0)
    return o.to(q.dtype), final_state
