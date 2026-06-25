# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""THD (token-packed, varlen) front-end for the Blackwell CuteDSL mamba2 SSD kernel.

The vendored CuteDSL kernel (:mod:`._mamba2_ssd_kernel`) consumes a dense
``(B, EH, D, C, L)`` layout: one persistent CTA per ``(sequence, head)`` that
walks ``C`` fixed-size chunks of ``L`` tokens, passing the SSM state across
chunks in-kernel. Triton's :func:`mamba_chunk_scan_combined_varlen` instead
consumes a token-packed ``(T, H, P)`` varlen layout.

Stage 1 of the port bridges the two on the host: each sequence is re-chunked at
the kernel's compile-time chunk size ``L`` and scattered into a zero-padded
dense buffer. Zero padding is mathematically exact for SSD because a padded
token has ``dt = 0`` -> no state contribution, and the within-chunk cumulative
decay naturally "holds" its last real value across the padded tail.

Preprocessing the kernel itself does NOT do (and that this wrapper performs on
the host, matching Triton semantics): ``dt = softplus(dt + dt_bias)`` clamped to
``dt_limit``, ``A`` already negated by the caller, and the per-chunk
``cumsum_delta = cumsum(dt * A)``.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_CUTE_AVAILABLE = None
_COMPILE_CACHE = {}
_MAX_ACTIVE_CLUSTERS = None


def is_cutedsl_ssd_available() -> bool:
    """Return True if the CuteDSL runtime can be imported on this system."""
    global _CUTE_AVAILABLE
    if _CUTE_AVAILABLE is None:
        try:
            import cutlass  # noqa: F401
            import cutlass.cute  # noqa: F401

            _CUTE_AVAILABLE = True
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning("CuteDSL not available: %s", e)
            _CUTE_AVAILABLE = False
    return _CUTE_AVAILABLE


def _torch_to_cute_dtype(dtype: torch.dtype):
    import cutlass

    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise ValueError(f"Unsupported io dtype for CuteDSL SSD kernel: {dtype}")


def _to_cute(torch_tensor, dynamic_modes):
    from cutlass.cute.runtime import from_dlpack

    ct = from_dlpack(torch_tensor, assumed_align=16)
    # dim_order() is a torch op (~18us); compute it ONCE, not per dynamic mode.
    # (A naive Python stride-sort mis-orders tied/size-1 dims, so use the real one.)
    stride_order = torch_tensor.dim_order()
    for mode in dynamic_modes:
        ct = ct.mark_compact_shape_dynamic(mode=mode, stride_order=stride_order)
    return ct


def _get_compiled(
    io_dtype,
    L,
    D,
    N,
    has_d,
    d_has_hdim,
    x_t,
    cumsum_t,
    delta_t,
    b_t,
    c_t,
    y_t,
    fstate_t,
    d_t,
    stream,
):
    """Compile (and cache) the SSD kernel for a fixed (dtype, L, D, N, D-fusion)."""
    import cutlass

    from ._mamba2_ssd_kernel import SSDKernel

    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        _MAX_ACTIVE_CLUSTERS = cutlass.utils.HardwareInfo().get_max_active_clusters(1)

    key = (io_dtype, L, D, N, has_d, d_has_hdim)
    compiled = _COMPILE_CACHE.get(key)
    if compiled is None:
        import cutlass.cute as cute

        ssd = SSDKernel(io_dtype, cutlass.Float32, cutlass.Float32, L, D, N, has_d, d_has_hdim)
        compiled = cute.compile(
            ssd, x_t, cumsum_t, delta_t, b_t, c_t, y_t, fstate_t, d_t, _MAX_ACTIVE_CLUSTERS, stream
        )
        _COMPILE_CACHE[key] = compiled
    return compiled


_COMPILE_CACHE_THD = {}
_THD_WS_CACHE = {}


def _thd_workspace(
    key, S, H, P, Cmax, L, G, N_pad, io_dtype, cute_io_dtype, has_d, d_has_hdim, stream
):
    """Get-or-create a reusable THD workspace (buffers + cute wrappers + compiled
    kernel) keyed by problem shape. Reusing buffers lets us cache the cute tensor
    descriptors (from_dlpack + mark_compact_shape_dynamic is ~40us/tensor) and
    avoid per-call re-allocation."""
    ws = _THD_WS_CACHE.get(key)
    if ws is not None:
        return ws
    device = "cuda"
    delta_buf = torch.empty(S, H, Cmax, L, device=device, dtype=io_dtype)
    cumsum_buf = torch.empty(S, H, Cmax, L, device=device, dtype=torch.float32)
    B_buf = torch.zeros(S, G, N_pad, Cmax, L, device=device, dtype=io_dtype)
    C_buf = torch.zeros(S, G, N_pad, Cmax, L, device=device, dtype=io_dtype)
    fstate_buf = torch.zeros(S, H, P, N_pad, device=device, dtype=io_dtype)
    d_buf = torch.zeros(H, P if d_has_hdim else 1, device=device, dtype=io_dtype) if has_d else None
    HP = H * P
    delta_t = _to_cute(delta_buf.permute(3, 2, 1, 0), [1, 2, 3])
    cumsum_t = _to_cute(cumsum_buf.permute(3, 2, 1, 0), [1, 2, 3])
    b_t = _to_cute(B_buf.permute(4, 2, 3, 1, 0), [2, 3, 4])
    c_t = _to_cute(C_buf.permute(4, 2, 3, 1, 0), [2, 3, 4])
    fstate_t = _to_cute(fstate_buf.permute(2, 3, 1, 0), [2, 3])
    d_t = _to_cute(d_buf.permute(1, 0), [1]) if has_d else None
    # x and y are fed as zero-copy THD views of fresh tensors each call, so they
    # are re-wrapped per call; compile with placeholders of the same THD strides.
    x_ph = torch.empty(S * Cmax * L, H, P, device=device, dtype=io_dtype)
    x_ph_t = _to_cute(
        x_ph.as_strided((P, L, Cmax, H, S), (1, HP, L * HP, P, Cmax * L * HP)), [2, 3, 4]
    )
    y_ph = torch.empty(S * Cmax * L, H, P, device=device, dtype=io_dtype)
    y_ph_t = _to_cute(
        y_ph.as_strided((L, P, Cmax, H, S), (HP, 1, L * HP, P, Cmax * L * HP)), [2, 3, 4]
    )
    compiled = _get_compiled_thd(
        cute_io_dtype,
        L,
        P,
        N_pad,
        has_d,
        d_has_hdim,
        x_ph_t,
        cumsum_t,
        delta_t,
        b_t,
        c_t,
        y_ph_t,
        fstate_t,
        d_t,
        stream,
    )
    ws = dict(
        delta_buf=delta_buf,
        cumsum_buf=cumsum_buf,
        B_buf=B_buf,
        C_buf=C_buf,
        fstate_buf=fstate_buf,
        d_buf=d_buf,
        delta_t=delta_t,
        cumsum_t=cumsum_t,
        b_t=b_t,
        c_t=c_t,
        fstate_t=fstate_t,
        d_t=d_t,
        compiled=compiled,
    )
    _THD_WS_CACHE[key] = ws
    return ws


_THD_VWS_CACHE = {}


def _thd_workspace_varlen(
    key, S, H, P, Cmax, L, G, N, N_pad, io_dtype, cute_io_dtype, has_d, d_has_hdim, stream
):
    """Reusable workspace for the varlen (unequal/non-divisible) path. All buffers
    are internal, so every cute descriptor is cacheable across calls with the
    same (S, Cmax, ...) shape; only data is refilled per call."""
    ws = _THD_VWS_CACHE.get(key)
    if ws is not None:
        return ws
    device = "cuda"
    SCL = S * Cmax * L
    HP = H * P
    x_pad = torch.zeros(SCL, H, P, device=device, dtype=io_dtype)
    delta_pad = torch.zeros(SCL, H, device=device, dtype=torch.float32)
    B_pad = torch.zeros(SCL, G, N, device=device, dtype=io_dtype)
    C_pad = torch.zeros(SCL, G, N, device=device, dtype=io_dtype)
    y_scratch = torch.empty(SCL, H, P, device=device, dtype=io_dtype)
    delta_d = torch.empty(S, H, Cmax, L, device=device, dtype=io_dtype)
    cumsum_d = torch.empty(S, H, Cmax, L, device=device, dtype=torch.float32)
    B_d = torch.zeros(S, G, N_pad, Cmax, L, device=device, dtype=io_dtype)
    C_d = torch.zeros(S, G, N_pad, Cmax, L, device=device, dtype=io_dtype)
    fstate_buf = torch.zeros(S, H, P, N_pad, device=device, dtype=io_dtype)
    d_buf = torch.zeros(H, P if d_has_hdim else 1, device=device, dtype=io_dtype) if has_d else None
    x_t = _to_cute(
        x_pad.as_strided((P, L, Cmax, H, S), (1, HP, L * HP, P, Cmax * L * HP)), [2, 3, 4]
    )
    y_t = _to_cute(
        y_scratch.as_strided((L, P, Cmax, H, S), (HP, 1, L * HP, P, Cmax * L * HP)), [2, 3, 4]
    )
    delta_t = _to_cute(delta_d.permute(3, 2, 1, 0), [1, 2, 3])
    cumsum_t = _to_cute(cumsum_d.permute(3, 2, 1, 0), [1, 2, 3])
    b_t = _to_cute(B_d.permute(4, 2, 3, 1, 0), [2, 3, 4])
    c_t = _to_cute(C_d.permute(4, 2, 3, 1, 0), [2, 3, 4])
    fstate_t = _to_cute(fstate_buf.permute(2, 3, 1, 0), [2, 3])
    d_t = _to_cute(d_buf.permute(1, 0), [1]) if has_d else None
    compiled = _get_compiled_thd(
        cute_io_dtype,
        L,
        P,
        N_pad,
        has_d,
        d_has_hdim,
        x_t,
        cumsum_t,
        delta_t,
        b_t,
        c_t,
        y_t,
        fstate_t,
        d_t,
        stream,
    )
    ws = dict(
        x_pad=x_pad,
        delta_pad=delta_pad,
        B_pad=B_pad,
        C_pad=C_pad,
        y_scratch=y_scratch,
        delta_d=delta_d,
        cumsum_d=cumsum_d,
        B_d=B_d,
        C_d=C_d,
        fstate_buf=fstate_buf,
        d_buf=d_buf,
        x_t=x_t,
        y_t=y_t,
        delta_t=delta_t,
        cumsum_t=cumsum_t,
        b_t=b_t,
        c_t=c_t,
        fstate_t=fstate_t,
        d_t=d_t,
        compiled=compiled,
    )
    _THD_VWS_CACHE[key] = ws
    return ws


def _get_compiled_thd(
    io_dtype,
    L,
    D,
    N,
    has_d,
    d_has_hdim,
    x_t,
    cumsum_t,
    delta_t,
    b_t,
    c_t,
    y_t,
    fstate_t,
    d_t,
    stream,
):
    """Compile (and cache) the THD SSD kernel variant (X headdim-major)."""
    import cutlass

    from ._mamba2_ssd_kernel_thd import SSDKernel

    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        _MAX_ACTIVE_CLUSTERS = cutlass.utils.HardwareInfo().get_max_active_clusters(1)

    key = (io_dtype, L, D, N, has_d, d_has_hdim)
    compiled = _COMPILE_CACHE_THD.get(key)
    if compiled is None:
        import cutlass.cute as cute

        ssd = SSDKernel(io_dtype, cutlass.Float32, cutlass.Float32, L, D, N, has_d, d_has_hdim)
        compiled = cute.compile(
            ssd, x_t, cumsum_t, delta_t, b_t, c_t, y_t, fstate_t, d_t, _MAX_ACTIVE_CLUSTERS, stream
        )
        _COMPILE_CACHE_THD[key] = compiled
    return compiled


_COMPILE_CACHE_VARLEN = {}
_DIV_WS_CACHE = {}


def _divisible_workspace(
    key, S, H, P, TC, L, G, N_pad, io_dtype, cute_io_dtype, has_d, d_has_hdim, stream
):
    """Reusable workspace for the divisible varlen path: cached dense B/C/delta/
    cumsum buffers + cute descriptors + compiled varlen kernel (x/y are zero-copy
    views supplied per call; cs/nc come from the metadata cache)."""
    ws = _DIV_WS_CACHE.get(key)
    if ws is not None:
        return ws
    device = "cuda"
    HP = H * P
    delta_df = torch.zeros(1, H, TC, L, device=device, dtype=torch.float32)
    delta_d = torch.zeros(1, H, TC, L, device=device, dtype=io_dtype)
    cumsum_d = torch.zeros(1, H, TC, L, device=device, dtype=torch.float32)
    B_d = torch.zeros(1, G, N_pad, TC, L, device=device, dtype=io_dtype)
    C_d = torch.zeros(1, G, N_pad, TC, L, device=device, dtype=io_dtype)
    fstate_base = torch.zeros(S, H, P, N_pad, device=device, dtype=io_dtype)
    d_buf = torch.zeros(H, P if d_has_hdim else 1, device=device, dtype=io_dtype) if has_d else None
    delta_t = _to_cute(delta_d.permute(3, 2, 1, 0), [1, 2, 3])
    cumsum_t = _to_cute(cumsum_d.permute(3, 2, 1, 0), [1, 2, 3])
    b_t = _to_cute(B_d.permute(4, 2, 3, 1, 0), [2, 3, 4])
    c_t = _to_cute(C_d.permute(4, 2, 3, 1, 0), [2, 3, 4])
    fstate_t = _to_cute(fstate_base.permute(2, 3, 1, 0), [2, 3])
    d_t = _to_cute(d_buf.permute(1, 0), [1]) if has_d else None
    x_ph = torch.empty(TC * L, H, P, device=device, dtype=io_dtype)
    x_ph_t = _to_cute(x_ph.as_strided((P, L, TC, H, 1), (1, HP, L * HP, P, TC * L * HP)), [2, 3, 4])
    y_ph = torch.empty(TC * L, H, P, device=device, dtype=io_dtype)
    y_ph_t = _to_cute(y_ph.as_strided((L, P, TC, H, 1), (HP, 1, L * HP, P, TC * L * HP)), [2, 3, 4])
    cs_ph = torch.zeros(S, device=device, dtype=torch.int32)
    cs_ph_t = _to_cute(cs_ph, [0])
    nc_ph_t = _to_cute(cs_ph, [0])
    compiled = _get_compiled_varlen(
        cute_io_dtype,
        L,
        P,
        N_pad,
        has_d,
        d_has_hdim,
        x_ph_t,
        cumsum_t,
        delta_t,
        b_t,
        c_t,
        y_ph_t,
        fstate_t,
        d_t,
        cs_ph_t,
        nc_ph_t,
        stream,
    )
    ws = dict(
        delta_df=delta_df,
        delta_d=delta_d,
        cumsum_d=cumsum_d,
        B_d=B_d,
        C_d=C_d,
        fstate_base=fstate_base,
        d_buf=d_buf,
        delta_t=delta_t,
        cumsum_t=cumsum_t,
        b_t=b_t,
        c_t=c_t,
        fstate_t=fstate_t,
        d_t=d_t,
        compiled=compiled,
    )
    _DIV_WS_CACHE[key] = ws
    return ws


def _get_compiled_varlen(
    io_dtype,
    L,
    D,
    N,
    has_d,
    d_has_hdim,
    x_t,
    cumsum_t,
    delta_t,
    b_t,
    c_t,
    y_t,
    fstate_t,
    d_t,
    cs_t,
    nc_t,
    stream,
):
    """Compile (and cache) the varlen tile-scheduler kernel (per-(seq,head) work
    items processing only their own chunks; no Cmax padding)."""
    import cutlass

    from ._mamba2_ssd_kernel_varlen import SSDKernel

    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        _MAX_ACTIVE_CLUSTERS = cutlass.utils.HardwareInfo().get_max_active_clusters(1)

    key = (io_dtype, L, D, N, has_d, d_has_hdim)
    compiled = _COMPILE_CACHE_VARLEN.get(key)
    if compiled is None:
        import cutlass.cute as cute

        ssd = SSDKernel(io_dtype, cutlass.Float32, cutlass.Float32, L, D, N, has_d, d_has_hdim)
        compiled = cute.compile(
            ssd,
            x_t,
            cumsum_t,
            delta_t,
            b_t,
            c_t,
            y_t,
            fstate_t,
            d_t,
            cs_t,
            nc_t,
            _MAX_ACTIVE_CLUSTERS,
            stream,
        )
        _COMPILE_CACHE_VARLEN[key] = compiled
    return compiled


def _current_cute_stream():
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


import collections as _collections

_META_CACHE = _collections.OrderedDict()
_META_CACHE_MAX = 64


def _chunk_meta(cu_chunk_seqlens, last_chunk_indices, L, T, device):
    """Derive per-call shape metadata (aligned-ness, Cmax, dest_idx, ...) and
    CACHE it keyed by the metadata tensors' identity.

    Computing this requires a host<->device sync (``.item()`` / ``.all()``),
    which — if done every call — serializes CPU dispatch against the GPU kernel
    and erases the kernel's speed advantage. By caching (and holding refs so the
    ids stay valid), repeated calls with the same chunk metadata (fixed-shape
    inference/training, or reused metadata buffers) skip the sync entirely and
    the CPU pipeline overlaps the GPU work.
    """
    k = (id(cu_chunk_seqlens), id(last_chunk_indices))
    m = _META_CACHE.get(k)
    if m is not None:
        _META_CACHE.move_to_end(k)
        return m
    ccs = cu_chunk_seqlens.to(torch.long)
    lci = last_chunk_indices.to(torch.long)
    cu_seqlens = torch.cat([ccs[:1], ccs[lci + 1]])
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    S = seq_lens.numel()
    seqlen0 = int(seq_lens[0])  # sync (once per unique metadata)
    aligned = (seqlen0 % L == 0) and bool((seq_lens == seqlen0).all())  # sync
    divisible = bool((seq_lens % L == 0).all())  # sync
    n_chunks_dev = chunk_start_dev = total_chunks = None
    if aligned:
        Cmax = seqlen0 // L
        dest_idx = None
    else:
        Cmax = int(((seq_lens + L - 1) // L).max().item())  # sync
        token_seq = torch.repeat_interleave(torch.arange(S, device=device), seq_lens, output_size=T)
        token_in_seq = torch.arange(T, device=device) - cu_seqlens[token_seq]
        dest_idx = token_seq * (Cmax * L) + token_in_seq
    if divisible:
        # Varlen tile-scheduler metadata: per-seq chunk count + exclusive-cumsum
        # start, plus the global chunk index per token (for the dense scatter).
        n_chunks_dev = (seq_lens // L).to(torch.int32)
        chunk_start_dev = torch.cumsum(n_chunks_dev, 0, dtype=torch.int32) - n_chunks_dev
        total_chunks = T // L
    m = dict(
        aligned=aligned,
        divisible=divisible,
        seqlen0=seqlen0,
        Cmax=Cmax,
        S=S,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        dest_idx=dest_idx,
        n_chunks_dev=n_chunks_dev,
        chunk_start_dev=chunk_start_dev,
        total_chunks=total_chunks,
        _refs=(cu_chunk_seqlens, last_chunk_indices),
    )
    _META_CACHE[k] = m
    if len(_META_CACHE) > _META_CACHE_MAX:
        _META_CACHE.popitem(last=False)
    return m


def mamba_chunk_scan_combined_varlen_cutedsl(
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
    return_intermediate_states=False,
    intermediate_chunk_indices=None,
    state_dtype=None,
    kernel_chunk_size=128,
):
    """CuteDSL-backed drop-in for ``mamba_chunk_scan_combined_varlen``.

    See :func:`megatron.core.ssm.ops.ssd_combined.mamba_chunk_scan_combined_varlen`
    for the argument contract. ``kernel_chunk_size`` is the internal chunk size
    ``L`` the kernel is compiled for (independent of ``chunk_size``, since the
    chunked scan result is chunk-size invariant).

    Raises:
        NotImplementedError: for argument combinations not yet supported by the
            Stage-1 port (gating ``z``, non-zero ``initial_states``,
            intermediate-state extraction). Callers should fall back to Triton.
    """
    if z is not None:
        raise NotImplementedError("CuteDSL SSD: z-gating not supported (use rmsnorm path)")
    if return_intermediate_states or intermediate_chunk_indices is not None:
        raise NotImplementedError("CuteDSL SSD: intermediate state extraction not supported")
    if initial_states is not None and torch.count_nonzero(initial_states) > 0:
        raise NotImplementedError("CuteDSL SSD: non-zero initial_states not supported")

    T, H, P = x.shape
    _, G, N = B.shape
    has_d = D is not None
    d_has_hdim = has_d and D.dim() == 2
    L = kernel_chunk_size
    # The kernel's MMA tile shapes are specialized for dstate == 128 (the
    # inter1 MMA uses dstate as its M-mode and the X/B smem atoms assume 128).
    # Zero-pad dstate up to a multiple of 128; padded state dims carry
    # B = C = 0 and are therefore mathematically inert.
    N_pad = ((N + 127) // 128) * 128
    device = x.device
    out_dtype = x.dtype
    # The CuteDSL kernel is a low-precision tensor-core kernel and only supports
    # fp16/bf16. fp32 inputs (e.g. from unit tests) are run through bf16.
    io_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    cute_io_dtype = _torch_to_cute_dtype(io_dtype)
    if state_dtype is None:
        state_dtype = out_dtype

    # ---- per-sequence boundaries (recover from chunk metadata) ----
    cu_chunk_seqlens = cu_chunk_seqlens.to(torch.long)
    last_chunk_indices = last_chunk_indices.to(torch.long)
    cu_seqlens = torch.cat([cu_chunk_seqlens[:1], cu_chunk_seqlens[last_chunk_indices + 1]])
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # (S,)
    S = seq_lens.numel()
    Cmax = int(((seq_lens + L - 1) // L).max().item())

    # ---- per-token (seq, chunk_in_seq, pos_in_chunk) at internal chunk size L ----
    arangeT = torch.arange(T, device=device)
    token_seq = torch.repeat_interleave(torch.arange(S, device=device), seq_lens)
    token_in_seq = arangeT - cu_seqlens[token_seq]
    token_cis = token_in_seq // L
    token_pos = token_in_seq % L

    # ---- dt preprocessing (matches Triton _chunk_cumsum_fwd) ----
    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    dt_min, dt_max = dt_limit
    if dt_min != 0.0 or dt_max != float("inf"):
        dt_f = torch.clamp(dt_f, dt_min, dt_max)

    # ---- scatter THD -> dense (B=S, EH=H, D=P, C=Cmax, L) ----
    x_d = torch.zeros(S, H, P, Cmax, L, device=device, dtype=io_dtype)
    x_d[token_seq, :, :, token_cis, token_pos] = x.to(io_dtype)

    delta_df = torch.zeros(S, H, Cmax, L, device=device, dtype=torch.float32)
    delta_df[token_seq, :, token_cis, token_pos] = dt_f
    delta_d = delta_df.to(io_dtype)
    cumsum_delta_d = torch.cumsum(delta_df * A.float().view(1, H, 1, 1), dim=-1)

    B_d = torch.zeros(S, G, N_pad, Cmax, L, device=device, dtype=io_dtype)
    B_d[token_seq, :, :N, token_cis, token_pos] = B.to(io_dtype)
    C_d = torch.zeros(S, G, N_pad, Cmax, L, device=device, dtype=io_dtype)
    C_d[token_seq, :, :N, token_cis, token_pos] = C.to(io_dtype)

    # Outputs: allocate in the (B, EH, D, C, L) contiguous order and hand the
    # kernel a permuted *view*. The kernel's TMA descriptor is built from the
    # view's strides, so these must match upstream run() exactly (which also
    # uses permuted views with the original [B,EH,D,C,L] strides).
    y_base = torch.zeros(S, H, P, Cmax, L, device=device, dtype=io_dtype)  # (B,EH,D,C,L)
    fstate_base = torch.zeros(S, H, P, N_pad, device=device, dtype=io_dtype)  # (B,EH,D,N)

    if not has_d:
        d_d = None
    elif d_has_hdim:
        d_d = D.to(io_dtype).contiguous()  # (H, P)
    else:
        d_d = D.to(io_dtype).view(H, 1).contiguous()  # (H, 1)

    # ---- build permuted cute tensor *views* (mirror upstream run()) ----
    # The dense tensors are contiguous in (B, EH, D, C, L) order; the kernel
    # consumes permuted *views* of them. The TMA descriptor is built from the
    # view's strides, which must equal upstream run()'s (permuted views over an
    # original [B,EH,D,C,L]-contiguous buffer). Do NOT call .contiguous() here.
    x_perm = x_d.permute(2, 4, 3, 1, 0)  # (D, L, C, EH, B)
    delta_perm = delta_d.permute(3, 2, 1, 0)  # (L, C, EH, B)
    cumsum_perm = cumsum_delta_d.permute(3, 2, 1, 0)
    b_perm = B_d.permute(4, 2, 3, 1, 0)  # (L, N, C, G, B)
    c_perm = C_d.permute(4, 2, 3, 1, 0)
    y_perm = y_base.permute(4, 2, 3, 1, 0)  # (L, D, C, EH, B)
    fstate_perm = fstate_base.permute(2, 3, 1, 0)  # (D, N, EH, B)

    x_t = _to_cute(x_perm, [2, 3, 4])
    delta_t = _to_cute(delta_perm, [1, 2, 3])
    cumsum_t = _to_cute(cumsum_perm, [1, 2, 3])
    b_t = _to_cute(b_perm, [2, 3, 4])
    c_t = _to_cute(c_perm, [2, 3, 4])
    y_t = _to_cute(y_perm, [2, 3, 4])
    fstate_t = _to_cute(fstate_perm, [2, 3])
    d_t = _to_cute(d_d.permute(1, 0), [1]) if has_d else None

    stream = _current_cute_stream()
    compiled = _get_compiled(
        cute_io_dtype,
        L,
        P,
        N_pad,
        has_d,
        d_has_hdim,
        x_t,
        cumsum_t,
        delta_t,
        b_t,
        c_t,
        y_t,
        fstate_t,
        d_t,
        stream,
    )
    compiled(x_t, cumsum_t, delta_t, b_t, c_t, y_t, fstate_t, d_t, stream)

    # ---- gather dense -> THD output, return per-seq final states ----
    # y_base / fstate_base are written in-place by the kernel (via the views).
    out.copy_(y_base[token_seq, :, :, token_cis, token_pos])
    return fstate_base[..., :N].to(state_dtype)


def mamba_chunk_scan_combined_varlen_cutedsl_thd(
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
    return_intermediate_states=False,
    intermediate_chunk_indices=None,
    state_dtype=None,
    kernel_chunk_size=128,
):
    """Stage-2 THD-native path: X is fed as a near-zero-copy headdim-contiguous
    view (the kernel's X operand is MN-major in this variant), avoiding the
    expensive X transpose. B/C/delta/cumsum stay dense (small); D-fusion is done
    on the host (kernel compiled with has_d=False). Falls back via
    NotImplementedError for unsupported argument combinations.
    """
    if z is not None:
        raise NotImplementedError("CuteDSL-THD SSD: z-gating not supported")
    if return_intermediate_states or intermediate_chunk_indices is not None:
        raise NotImplementedError("CuteDSL-THD SSD: intermediate states not supported")
    if initial_states is not None and torch.count_nonzero(initial_states) > 0:
        raise NotImplementedError("CuteDSL-THD SSD: non-zero initial_states not supported")

    T, H, P = x.shape
    _, G, N = B.shape
    d_has_hdim = D is not None and D.dim() == 2
    L = kernel_chunk_size
    N_pad = ((N + 127) // 128) * 128
    device = x.device
    out_dtype = x.dtype
    io_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    cute_io_dtype = _torch_to_cute_dtype(io_dtype)
    if state_dtype is None:
        state_dtype = out_dtype

    HP = H * P
    # Shape metadata (aligned-ness, Cmax, dest_idx) is cached to avoid a
    # per-call host<->device sync that would serialize CPU dispatch vs the GPU
    # kernel (see _chunk_meta).
    meta = _chunk_meta(cu_chunk_seqlens, last_chunk_indices, L, T, device)
    S, aligned, Cmax, seqlen0 = meta["S"], meta["aligned"], meta["Cmax"], meta["seqlen0"]

    dt_f = dt.float()
    if dt_bias is not None:
        dt_f = dt_f + dt_bias.float()
    if dt_softplus:
        dt_f = F.softplus(dt_f)
    dt_min, dt_max = dt_limit
    if dt_min != 0.0 or dt_max != float("inf"):
        dt_f = torch.clamp(dt_f, dt_min, dt_max)

    has_d = D is not None
    # Fast path: all sequences equal length and a multiple of L. Then the dense
    # (B,EH,D,C,L) tensors are pure views/transposes of the THD inputs (no
    # scatter), X is a zero-copy headdim-contiguous view, and we can reuse a
    # cached workspace (buffers + cute descriptors + compiled kernel).
    if aligned:
        stream = _current_cute_stream()
        key = (S, H, P, Cmax, L, G, N_pad, io_dtype, has_d, d_has_hdim)
        ws = _thd_workspace(
            key, S, H, P, Cmax, L, G, N_pad, io_dtype, cute_io_dtype, has_d, d_has_hdim, stream
        )
        ws["delta_buf"].copy_(dt_f.view(S, Cmax, L, H).permute(0, 3, 1, 2))
        torch.cumsum(
            dt_f.view(S, Cmax, L, H).permute(0, 3, 1, 2) * A.float().view(1, H, 1, 1),
            dim=-1,
            out=ws["cumsum_buf"],
        )
        ws["B_buf"][:, :, :N].copy_(B.view(S, Cmax, L, G, N).permute(0, 3, 4, 1, 2))
        ws["C_buf"][:, :, :N].copy_(C.view(S, Cmax, L, G, N).permute(0, 3, 4, 1, 2))
        if has_d:
            ws["d_buf"].copy_(D.to(io_dtype) if d_has_hdim else D.to(io_dtype).view(H, 1))
        x_v = x.as_strided((P, L, Cmax, H, S), (1, HP, L * HP, P, seqlen0 * HP))
        x_t = _to_cute(x_v, [2, 3, 4])
        # Y is stored in-kernel directly into a THD-strided view (zero-copy when
        # `out` is already the kernel io dtype; otherwise via a bf16 scratch).
        y_target = out if out.dtype == io_dtype else torch.empty_like(x)
        y_v = y_target.as_strided((L, P, Cmax, H, S), (HP, 1, L * HP, P, seqlen0 * HP))
        y_t = _to_cute(y_v, [2, 3, 4])
        ws["compiled"](
            x_t,
            ws["cumsum_t"],
            ws["delta_t"],
            ws["b_t"],
            ws["c_t"],
            y_t,
            ws["fstate_t"],
            ws["d_t"],
            stream,
        )
        if y_target is not out:
            out.copy_(y_target)
        return ws["fstate_buf"][..., :N].to(state_dtype)

    if meta["divisible"]:
        # Varlen tile-scheduler path: every sequence's length is a multiple of L,
        # so chunks pack contiguously (total_chunks = T/L). x/y are zero-copy
        # chunk-major THD views; B/C/delta are dense chunk-major (cached
        # workspace); each (seq,head) work-item processes ONLY its own chunks
        # (no Cmax padding waste).
        TC = meta["total_chunks"]
        has_d = D is not None
        stream = _current_cute_stream()
        key = (S, H, P, TC, L, G, N_pad, io_dtype, has_d, d_has_hdim)
        ws = _divisible_workspace(
            key, S, H, P, TC, L, G, N_pad, io_dtype, cute_io_dtype, has_d, d_has_hdim, stream
        )
        # Fast build (no scatter): for divisible packing, token = chunk*L + pos,
        # so a transpose (T,...)->(...,TC,L) places everything; copy into the
        # cached buffers. (N_pad tail stays zero from allocation.)
        ws["delta_df"].copy_(dt_f.permute(1, 0).reshape(1, H, TC, L))
        ws["delta_d"].copy_(ws["delta_df"])
        torch.cumsum(ws["delta_df"] * A.float().view(1, H, 1, 1), dim=-1, out=ws["cumsum_d"])
        ws["B_d"][:, :, :N].copy_(B.permute(1, 2, 0).reshape(1, G, N, TC, L))
        ws["C_d"][:, :, :N].copy_(C.permute(1, 2, 0).reshape(1, G, N, TC, L))
        if has_d:
            ws["d_buf"].copy_(D.to(io_dtype) if d_has_hdim else D.to(io_dtype).view(H, 1))
        x_v = x.as_strided((P, L, TC, H, 1), (1, HP, L * HP, P, T * HP))
        x_t = _to_cute(x_v, [2, 3, 4])
        y_target = out if out.dtype == io_dtype else torch.empty_like(x)
        y_v = y_target.as_strided((L, P, TC, H, 1), (HP, 1, L * HP, P, T * HP))
        y_t = _to_cute(y_v, [2, 3, 4])
        cs_t = _to_cute(meta["chunk_start_dev"], [0])
        nc_t = _to_cute(meta["n_chunks_dev"], [0])
        ws["compiled"](
            x_t,
            ws["cumsum_t"],
            ws["delta_t"],
            ws["b_t"],
            ws["c_t"],
            y_t,
            ws["fstate_t"],
            ws["d_t"],
            cs_t,
            nc_t,
            stream,
        )
        if y_target is not out:
            out.copy_(y_target)
        return ws["fstate_base"][..., :N].to(state_dtype)

    # General varlen path (cached workspace; index_copy + cheap transpose).
    # Cmax/dest_idx come from the cached metadata (no per-call sync).
    dest_idx = meta["dest_idx"]

    stream = _current_cute_stream()
    vkey = (S, Cmax, H, P, G, N_pad, io_dtype, has_d, d_has_hdim)
    ws = _thd_workspace_varlen(
        vkey, S, H, P, Cmax, L, G, N, N_pad, io_dtype, cute_io_dtype, has_d, d_has_hdim, stream
    )
    # Zero the maskers (delta/B/C padding must be 0 since dest_idx changes per
    # call; x/y padding is harmless, masked by delta=0). Fill via fast
    # contiguous-row index_copy, then cheap transposes into the cached buffers.
    # index_copy_ (dim-0 scatter of contiguous rows) is much faster than
    # advanced-index assignment (index_put_).
    ws["delta_pad"].zero_()
    ws["B_pad"].zero_()
    ws["C_pad"].zero_()
    ws["x_pad"].index_copy_(0, dest_idx, x.to(io_dtype))
    ws["delta_pad"].index_copy_(0, dest_idx, dt_f)
    ws["B_pad"].index_copy_(0, dest_idx, B.to(io_dtype))
    ws["C_pad"].index_copy_(0, dest_idx, C.to(io_dtype))
    dp = ws["delta_pad"].view(S, Cmax, L, H).permute(0, 3, 1, 2)
    ws["delta_d"].copy_(dp)
    torch.cumsum(dp * A.float().view(1, H, 1, 1), dim=-1, out=ws["cumsum_d"])
    ws["B_d"][:, :, :N].copy_(ws["B_pad"].view(S, Cmax, L, G, N).permute(0, 3, 4, 1, 2))
    ws["C_d"][:, :, :N].copy_(ws["C_pad"].view(S, Cmax, L, G, N).permute(0, 3, 4, 1, 2))
    if has_d:
        ws["d_buf"].copy_(D.to(io_dtype) if d_has_hdim else D.to(io_dtype).view(H, 1))
    ws["compiled"](
        ws["x_t"],
        ws["cumsum_t"],
        ws["delta_t"],
        ws["b_t"],
        ws["c_t"],
        ws["y_t"],
        ws["fstate_t"],
        ws["d_t"],
        stream,
    )
    out.copy_(ws["y_scratch"].index_select(0, dest_idx))
    return ws["fstate_buf"][..., :N].to(state_dtype)
