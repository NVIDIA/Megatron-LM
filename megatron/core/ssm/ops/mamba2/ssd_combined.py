# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from:
#   https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py
# Adapted from vLLM project (Apache-2.0).

import torch
import triton
from packaging import version

from .ssd_backend import cutedsl_ssd_available
from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .ssd_state_passing import _state_passing_fwd

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def is_int_pow_2(n):
    """Return True if n is a positive integer power of 2."""
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


def _mamba_chunk_scan_combined_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_chunk_seqlens=None,
    last_chunk_indices=None,
    return_raw_states=False,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    state_dtype=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    seqlen, nheads, headdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (
        seqlen,
        ngroups,
        dstate,
    ), f"B.shape={B.shape} != ({seqlen}, {ngroups}, {dstate})"
    assert dt.shape == (seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (cu_chunk_seqlens.shape[0] - 1,)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if x.stride(-1) != 1 and x.stride(0) != 1:  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (
        z is not None and z.stride(-1) != 1 and z.stride(0) != 1
    ):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    assert cu_chunk_seqlens is not None, "Assuming varlen input - must supply cu_chunk_seqlens"
    assert last_chunk_indices is not None, "last_chunk_indices must be provided"

    if initial_states is not None:
        num_seqs = last_chunk_indices.shape[0]
        assert initial_states.shape == (num_seqs, nheads, headdim, dstate)

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt,
        A,
        chunk_size,
        cu_chunk_seqlens,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
    )

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_chunk_seqlens, states_in_fp32=True)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states and
    #   ii) seq_idx to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    states = _state_passing_fwd(
        states.flatten(-2),  # ... p n -> ... (p n)
        dA_cumsum,  # (nheads, nchunks, chunk_size)
        cu_chunk_seqlens,
        initial_states=(
            initial_states.flatten(-2) if initial_states is not None else None
        ),  # (batch, nheads, headdim*dstate)
        seq_idx=seq_idx,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
    )
    states = states.unflatten(-1, (-1, dstate))

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        cu_chunk_seqlens,
        out,  # in-place update
        seq_idx,
        D=D,
        z=z,
        initial_states=initial_states,
    )

    final_states = states[last_chunk_indices]
    if return_raw_states:
        # Caller extracts any chunk-boundary states itself from the raw states.
        return final_states, states
    return final_states


def _mamba_chunk_scan_combined_varlen_triton(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_raw_states=False,
    state_dtype=None,
):
    """
    Argument:
        x: (seqlen, nheads, headdim)
        dt: (seqlen, nheads)
        A: (nheads)
        B: (seqlen, ngroups, dstate)
        C: (seqlen, ngroups, dstate)
        chunk_size: int
        cu_chunk_seqlens: (nchunks + 1,)
        last_chunk_indices: (batch,)
        seq_idx: (nchunks,)
        out: (seqlen, nheads, headdim) preallocated output tensor
        D: (nheads, headdim) or (nheads,)
        z: (seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        dt_softplus: Whether to apply softplus to dt
        return_raw_states: If True, returns ``(varlen_states, raw_states)`` where
            ``raw_states`` is the full ``(nchunks, nheads, headdim, dstate)``
            chunk-boundary state tensor for the caller to extract from directly.
        state_dtype: The data type of the ssm state
    Return:
        varlen_states: (batch, nheads, headdim, dstate), or
        (varlen_states, raw_states) if return_raw_states is True
    """

    assert seq_idx is not None

    varlen_states = _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        return_raw_states=return_raw_states,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        state_dtype=state_dtype,
    )

    return varlen_states


def mamba_chunk_scan_combined_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_raw_states=False,
    ssd_tiling=None,
    state_dtype=None,
):
    """Dispatch the varlen SSD scan to the CuteDSL (Blackwell) or Triton backend fallback.

    CuteDSL is opt-in on Blackwell (SM 10.0+) via
    `--inference-dynamic-batching-mamba-prefill-backend cutedsl`, which is what
    makes the caller pass an `ssd_tiling`: a faster drop-in covering the
    production prefill cases (arbitrary sequence
    lengths, chunked prefill via ``initial_states``, empty padded sequences, and
    ``return_raw_states`` for prefix caching). Eligibility is decided up front
    via :func:`cutedsl_unsupported_reason`; Triton is used on other platforms
    and for the argument combinations the CuteDSL kernel does not support
    (gating ``z``, and ``return_raw_states`` on a ragged batch, whose per-
    sequence chunk grid does not materialise the caller's chunk boundaries).
    """

    # The tiling is built only under the CuteDSL backend, so its presence IS the
    # backend selection; callers holding no per-step metadata (op-level tests,
    # benchmarks) simply get Triton. The availability check is a backstop.
    if ssd_tiling is not None and cutedsl_ssd_available():
        from .cutedsl_mamba2_ssd import (
            cutedsl_unsupported_reason,
            mamba_chunk_scan_combined_varlen_cutedsl_thd,
        )

        reason = cutedsl_unsupported_reason(
            x, chunk_size, ssd_tiling, z=z, return_raw_states=return_raw_states
        )
        if not reason:
            return mamba_chunk_scan_combined_varlen_cutedsl_thd(
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                chunk_size=chunk_size,
                cu_chunk_seqlens=cu_chunk_seqlens,
                last_chunk_indices=last_chunk_indices,
                tiling=ssd_tiling,
                seq_idx=seq_idx,
                out=out,
                D=D,
                z=z,
                dt_bias=dt_bias,
                initial_states=initial_states,
                dt_softplus=dt_softplus,
                dt_limit=dt_limit,
                return_raw_states=return_raw_states,
                state_dtype=state_dtype,
            )

    return _mamba_chunk_scan_combined_varlen_triton(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        cu_chunk_seqlens,
        last_chunk_indices,
        seq_idx,
        out,
        D=D,
        z=z,
        dt_bias=dt_bias,
        initial_states=initial_states,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        return_raw_states=return_raw_states,
        state_dtype=state_dtype,
    )
