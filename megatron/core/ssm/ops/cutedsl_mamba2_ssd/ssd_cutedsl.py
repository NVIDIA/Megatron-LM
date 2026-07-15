# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""THD (token-packed, varlen) front-end for the Blackwell CuteDSL mamba2 SSD kernel.

The CuteDSL varlen kernel consumes a dense chunk-major layout: one persistent
CTA per ``(sequence, head)`` that walks that sequence's own fixed-size chunks
of ``L`` tokens, passing the SSM state across chunks in-kernel. Triton's
:func:`mamba_chunk_scan_combined_varlen` instead consumes a token-packed
``(T, H, P)`` varlen layout, which is what Megatron inference always produces.

This front-end feeds the kernel natively from the token-packed input when
every sequence length is a multiple of ``L`` (then chunks pack contiguously:
X/Y are zero-copy chunk-major THD views and B/C/delta are strided copies into
cached workspace buffers). Non-divisible (ragged) batches raise
``NotImplementedError`` so the dispatcher falls back to Triton — host-side
repacking was measured slower than Triton's native varlen handling.

Preprocessing the kernel itself does NOT do (and that this wrapper performs,
matching Triton semantics): ``dt = softplus(dt + dt_bias)`` clamped to
``dt_limit``, ``A`` already negated by the caller, and the per-chunk
``cumsum_delta = cumsum(dt * A)`` (fused into a single Triton launch; see
:mod:`._fused_cumsum`).
"""

import collections
import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from ._bc_repack import repack_bc_chunk_major
from ._fused_cumsum import fused_softplus_cumsum
from ._mamba2_ssd_kernel_varlen import SSDKernel as SSDKernelVarlen

logger = logging.getLogger(__name__)

_MAX_ACTIVE_CLUSTERS = None


def is_cutedsl_ssd_available() -> bool:
    """Return True if the CuteDSL runtime is importable on this system."""
    return True


def _torch_to_cute_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float16:
        return cutlass.Float16
    raise ValueError(f"Unsupported io dtype for CuteDSL SSD kernel: {dtype}")


def _to_cute(torch_tensor, dynamic_modes):
    """Make a torch.Tensor to cute.Tensor via dlpack, then mark dynamic modes"""
    ct = from_dlpack(torch_tensor, assumed_align=16)
    stride_order = torch_tensor.dim_order()
    for mode in dynamic_modes:
        ct = ct.mark_compact_shape_dynamic(mode=mode, stride_order=stride_order)
    return ct


_COMPILE_CACHE_VARLEN = {}
_DIV_WS_CACHE = {}
# Prefix-caching intermediate-output buffers, keyed by (num_inter, H, P, N_pad, dtype).
# num_inter varies per call, so these live in their own cache (not the shape workspace).
_INTER_OUT_CACHE = {}


def _inter_out(num_inter, H, P, N_pad, io_dtype):
    """Get-or-create the cached intermediate-state buffers and cute descriptor."""
    key = (num_inter, H, P, N_pad, io_dtype)
    entry = _INTER_OUT_CACHE.get(key)
    if entry is None:
        raw = torch.empty(num_inter, H, P, N_pad, device="cuda", dtype=io_dtype)
        final = torch.empty(num_inter, H, P, N_pad, device="cuda", dtype=io_dtype)
        entry = (raw, final, _to_cute(raw.permute(2, 3, 1, 0), [2, 3]))
        _INTER_OUT_CACHE[key] = entry
    return entry


def _get_workspace(
    key,
    S,
    H,
    P,
    TC,
    L,
    G,
    N_pad,
    io_dtype,
    cute_io_dtype,
    has_d,
    d_has_hdim,
    has_initial,
    has_intermediate,
    stream,
):
    """Reusable workspace for the divisible varlen path: cached dense B/C/delta/
    cumsum buffers + cute descriptors + compiled varlen kernel (x/y are zero-copy
    views supplied per call; cs/nc come from the metadata cache). When has_initial,
    a cached (S,H,P,N_pad) buffer holds the per-call initial SSM state."""
    ws = _DIV_WS_CACHE.get(key)
    if ws is not None:
        return ws
    device = "cuda"
    HP = H * P
    delta_d = torch.zeros(1, H, TC, L, device=device, dtype=torch.float32)
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
    # initial_states buffer: (S,H,P,N_pad) -> (D,N_pad,EH,S) cute view, like fstate.
    # When has_initial is False the kernel ignores it, but compile still needs a
    # tensor arg, so reuse the fstate descriptor as a placeholder.
    init_base = torch.zeros(S, H, P, N_pad, device=device, dtype=io_dtype) if has_initial else None
    init_t = _to_cute(init_base.permute(2, 3, 1, 0), [2, 3]) if has_initial else fstate_t
    # intermediate_out (prefix caching): the real (num_inter,H,P,N) buffer is per-call
    # (num_inter varies), so compile with a size-1 placeholder; emit_slot is (TC,).
    # When has_intermediate is False the kernel ignores both, but compile needs valid
    # tensor args, so reuse fstate as the inter placeholder + a cheap zeros emit_slot.
    inter_ph = torch.zeros(1, H, P, N_pad, device=device, dtype=io_dtype)
    inter_ph_t = _to_cute(inter_ph.permute(2, 3, 1, 0), [2, 3]) if has_intermediate else fstate_t
    es_ph = torch.zeros(TC, device=device, dtype=torch.int32)
    es_ph_t = _to_cute(es_ph, [0])
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
        has_initial,
        has_intermediate,
        x_ph_t,
        cumsum_t,
        delta_t,
        b_t,
        c_t,
        y_ph_t,
        fstate_t,
        d_t,
        init_t,
        inter_ph_t,
        es_ph_t,
        cs_ph_t,
        nc_ph_t,
        stream,
    )
    ws = dict(
        delta_d=delta_d,
        cumsum_d=cumsum_d,
        B_d=B_d,
        C_d=C_d,
        fstate_base=fstate_base,
        d_buf=d_buf,
        init_base=init_base,
        delta_t=delta_t,
        cumsum_t=cumsum_t,
        b_t=b_t,
        c_t=c_t,
        fstate_t=fstate_t,
        d_t=d_t,
        init_t=init_t,
        inter_ph_t=inter_ph_t,
        # emit_slot buffer + descriptor are cached (fixed (TC,) shape); only the
        # content is refilled per call, avoiding a per-call descriptor.
        emit_slot_buf=es_ph if has_intermediate else None,
        es_ph_t=es_ph_t,
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
    has_initial,
    has_intermediate,
    x_t,
    cumsum_t,
    delta_t,
    b_t,
    c_t,
    y_t,
    fstate_t,
    d_t,
    init_t,
    inter_t,
    es_t,
    cs_t,
    nc_t,
    stream,
):
    """Compile (and cache) the varlen tile-scheduler kernel (per-(seq,head) work
    items processing only their own chunks; no Cmax padding). has_initial seeds the
    SSM state from initial_states (chunked prefill); has_intermediate emits the
    running state at flagged chunks (prefix caching)."""
    global _MAX_ACTIVE_CLUSTERS
    if _MAX_ACTIVE_CLUSTERS is None:
        _MAX_ACTIVE_CLUSTERS = cutlass.utils.HardwareInfo().get_max_active_clusters(1)

    key = (io_dtype, L, D, N, has_d, d_has_hdim, has_initial, has_intermediate)
    compiled = _COMPILE_CACHE_VARLEN.get(key)
    if compiled is None:
        ssd = SSDKernelVarlen(
            io_dtype,
            cutlass.Float32,
            cutlass.Float32,
            L,
            D,
            N,
            has_d,
            d_has_hdim,
            has_initial,
            has_intermediate,
        )
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
            init_t,
            inter_t,
            es_t,
            cs_t,
            nc_t,
            _MAX_ACTIVE_CLUSTERS,
            stream,
        )
        _COMPILE_CACHE_VARLEN[key] = compiled
    return compiled


def _current_cute_stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


_META_CACHE = collections.OrderedDict()
_META_CACHE_MAX = 64


def _chunk_meta(cu_chunk_seqlens, last_chunk_indices, kernel_chunk_size):
    """Derive per-call shape metadata (divisibility, chunk counts, ...) and
    CACHE it keyed by the metadata tensors' identity.

    Computing this requires a host<->device sync (``.item()`` / ``.all()``),
    which — if done every call — serializes CPU dispatch against the GPU kernel
    and erases the kernel's speed advantage. By caching (and holding refs so the
    ids stay valid), repeated calls with the same chunk metadata (fixed-shape
    inference/training, or reused metadata buffers) skip the sync entirely and
    the CPU pipeline overlaps the GPU work.
    """
    # kernel_chunk_size must be part of the key: the derived fields (divisible,
    # chunk counts, cs/nc descriptors) all depend on it, and eligibility checks
    # may probe the same metadata tensors with a different L than the wrapper.
    k = (id(cu_chunk_seqlens), id(last_chunk_indices), kernel_chunk_size)
    m = _META_CACHE.get(k)
    if m is not None:
        _META_CACHE.move_to_end(k)
        return m
    ccs = cu_chunk_seqlens.to(torch.long)
    lci = last_chunk_indices.to(torch.long)
    cu_seqlens = torch.cat([ccs[:1], ccs[lci + 1]])
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    S = seq_lens.numel()
    # Real token count (sum of seq_lens). The caller's ``x`` may be a fixed-size
    # CUDA-graph buffer with trailing PADDING (T = x.shape[0] > real tokens), so
    # everything below must key off the real count, never the padded ``T``.
    n_real_tokens = int(cu_seqlens[-1] - cu_seqlens[0])  # sync (once per metadata)
    divisible = bool((seq_lens % kernel_chunk_size == 0).all())  # sync
    n_chunks_dev = chunk_start_dev = total_chunks = None
    cs_t = nc_t = None
    real_seq_idx = None
    S_real = S
    has_empty = False
    real_is_prefix = True
    if divisible:
        # Varlen tile-scheduler metadata: per-seq chunk count + exclusive-cumsum
        # start, plus the global chunk index per token (for the dense scatter).
        n_chunks_dev = (seq_lens // kernel_chunk_size).to(torch.int32)
        chunk_start_dev = torch.cumsum(n_chunks_dev, 0, dtype=torch.int32) - n_chunks_dev
        total_chunks = n_real_tokens // kernel_chunk_size
        # The dynamic engine pads the batch to a fixed slot count with EMPTY
        # sequences (seq_len == 0). The varlen tile scheduler makes one work-item
        # per (seq, head); an empty seq yields a 0-chunk work-item that DEADLOCKS
        # the persistent pipeline (producer/consumer mbarriers never satisfied).
        # Compact the empty seqs out: run the kernel over only the non-empty
        # sequences (chunks pack contiguously, so total_chunks is unchanged), then
        # the wrapper scatters per-seq final states back to full batch shape.
        real_seq_idx = torch.nonzero(n_chunks_dev, as_tuple=True)[0]  # sync (cached)
        S_real = int(real_seq_idx.numel())
        has_empty = S_real != S
        # nonzero() returns sorted indices; the real seqs form a contiguous prefix
        # iff the last real index == S_real - 1 (trailing empties, what the dynamic
        # engine produces). Only then does the packed real-chunk numbering used by
        # the emit map match the caller's cu_chunk chunk numbering; interleaved
        # empties + intermediate emission fall back to Triton (see wrapper guard).
        real_is_prefix = (not has_empty) or (S_real > 0 and int(real_seq_idx[-1]) == S_real - 1)
        if has_empty:
            nc_real = n_chunks_dev[real_seq_idx]
            cs_real = torch.cumsum(nc_real, 0, dtype=torch.int32) - nc_real
        else:
            nc_real, cs_real = n_chunks_dev, chunk_start_dev
        # These tensors are cached (stable address), so cache their cute
        # descriptors too — per-call from_dlpack/mark is ~20us each.
        cs_t = _to_cute(cs_real, [0])
        nc_t = _to_cute(nc_real, [0])
    m = dict(
        divisible=divisible,
        n_real_tokens=n_real_tokens,
        S=S,
        cu_seqlens=cu_seqlens,
        seq_lens=seq_lens,
        n_chunks_dev=n_chunks_dev,
        chunk_start_dev=chunk_start_dev,
        total_chunks=total_chunks,
        cs_t=cs_t,
        nc_t=nc_t,
        real_seq_idx=real_seq_idx,
        S_real=S_real,
        has_empty=has_empty,
        real_is_prefix=real_is_prefix,
        _refs=(cu_chunk_seqlens, last_chunk_indices),
    )
    _META_CACHE[k] = m
    if len(_META_CACHE) > _META_CACHE_MAX:
        _META_CACHE.popitem(last=False)
    return m


def cutedsl_unsupported_reason(
    x: torch.Tensor,
    chunk_size: int,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    *,
    z: torch.Tensor | None = None,
    return_intermediate_states: bool = False,
    intermediate_chunk_indices: torch.Tensor | None = None,
    kernel_chunk_size: int = 128,
) -> str | None:
    """Given a inference batch, check whether CuTe DSL is applicable"""
    if z is not None:
        return "CuteDSL THD SSD: z-gating not supported"
    if return_intermediate_states:
        return "CuteDSL THD SSD: return_intermediate_states (all-chunk) not supported"
    has_intermediate = intermediate_chunk_indices is not None
    # Intermediate-state indices are chunk indices at the caller's chunk_size; the
    # kernel chunks at kernel_chunk_size, so they only line up when equal (the
    # mamba mixer default).
    if has_intermediate and chunk_size != kernel_chunk_size:
        return "CuteDSL THD SSD: intermediate states need chunk_size == kernel L"
    meta = _chunk_meta(cu_chunk_seqlens, last_chunk_indices, kernel_chunk_size)
    if not meta["divisible"]:
        return "CuteDSL THD SSD: sequence lengths must be multiples of the kernel chunk size"
    # Intermediate emit numbers chunks over the packed real chunks; this matches
    # the caller's chunk numbering only when the real seqs form a contiguous
    # prefix (trailing empties, what the dynamic engine produces).
    if has_intermediate and not meta["real_is_prefix"]:
        return "CuteDSL THD SSD: intermediate states with interleaved empty sequences"
    return None


def mamba_chunk_scan_combined_varlen_cutedsl_thd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    cu_chunk_seqlens: torch.Tensor,
    last_chunk_indices: torch.Tensor,
    seq_idx: torch.Tensor | None,
    out: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = (0.0, float("inf")),
    return_intermediate_states: bool = False,
    intermediate_chunk_indices: torch.Tensor | None = None,
    state_dtype: torch.dtype | None = None,
    kernel_chunk_size: int = 128,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Host launcher for CuTe DSL SSD kernel"""
    has_initial = initial_states is not None
    has_intermediate = intermediate_chunk_indices is not None

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
    if x.dtype != io_dtype:
        x = x.to(io_dtype)
    HP = H * P

    meta = _chunk_meta(cu_chunk_seqlens, last_chunk_indices, L)
    S = meta["S"]

    # Dynamic inference hands fixed-size CUDA-graph token buffers with trailing
    # padding (x.shape[0] > real tokens). All CuteDSL paths assume token-packed
    # input with no tail, so trim x/B/C/dt (and the output view) to the real
    # tokens; padded rows carry no state (they are outside cu_chunk_seqlens) and
    # their outputs are ignored by the caller.
    n_real_tokens = meta["n_real_tokens"]
    if n_real_tokens != T:
        x = x[:n_real_tokens]
        B = B[:n_real_tokens]
        C = C[:n_real_tokens]
        dt = dt[:n_real_tokens]
        out = out[:n_real_tokens]
        T = n_real_tokens

    has_d = D is not None
    # Varlen tile-scheduler path: every sequence's length is a multiple of L,
    # so chunks pack contiguously (total_chunks = T/L). x/y are zero-copy
    # chunk-major THD views; B/C/delta are dense chunk-major (cached
    # workspace); each (seq,head) work-item processes ONLY its own chunks.
    total_chunks = meta["total_chunks"]
    # Empty (padded) sequences are compacted out of the kernel launch: run over
    # only the S_real non-empty seqs, then scatter final states back to full S.
    S_kernel = meta["S_real"]
    has_empty = meta["has_empty"]
    real_seq_idx = meta["real_seq_idx"]
    stream = _current_cute_stream()
    key = (
        S_kernel,
        H,
        P,
        total_chunks,
        L,
        G,
        N,
        N_pad,
        io_dtype,
        has_d,
        d_has_hdim,
        has_initial,
        has_intermediate,
    )
    ws = _get_workspace(
        key,
        S_kernel,
        H,
        P,
        total_chunks,
        L,
        G,
        N_pad,
        io_dtype,
        cute_io_dtype,
        has_d,
        d_has_hdim,
        has_initial,
        has_intermediate,
        stream,
    )

    fused_softplus_cumsum(
        dt, A, dt_bias, dt_softplus, dt_limit, ws["delta_d"], ws["cumsum_d"], 1, H, total_chunks
    )
    if B.stride(2) == 1 and C.stride(2) == 1:
        # Tiled transpose: coalesced on both the token-major loads and the
        # chunk-major stores, B and C in one launch (~5x the strided copy_).
        repack_bc_chunk_major(B, C, ws["B_d"], ws["C_d"], N, total_chunks, L)
    else:
        GN = G * N
        ws["B_d"][:, :, :N].copy_(B.as_strided((1, G, N, total_chunks, L), (0, N, 1, L * GN, GN)))
        ws["C_d"][:, :, :N].copy_(C.as_strided((1, G, N, total_chunks, L), (0, N, 1, L * GN, GN)))
    if has_d:
        ws["d_buf"].copy_(D.to(io_dtype) if d_has_hdim else D.to(io_dtype).view(H, 1))
    if has_initial:
        init_src = initial_states[real_seq_idx] if has_empty else initial_states
        ws["init_base"][..., :N].copy_(init_src.to(io_dtype))
    if has_intermediate:
        num_inter = intermediate_chunk_indices.shape[0]
        emit_slot = ws["emit_slot_buf"]
        emit_slot.fill_(-1)
        emit_slot[intermediate_chunk_indices] = torch.arange(
            num_inter, dtype=torch.int32, device=device
        )
        inter_raw, inter_final, inter_t = _inter_out(num_inter, H, P, N_pad, io_dtype)
        es_t = ws["es_ph_t"]
    else:
        inter_t = ws["inter_ph_t"]
        es_t = ws["es_ph_t"]

    x_v = x.as_strided((P, L, total_chunks, H, 1), (1, HP, L * HP, P, T * HP))
    x_t = _to_cute(x_v, [2, 3, 4])
    y_target = out if out.dtype == io_dtype else torch.empty_like(x)
    y_v = y_target.as_strided((L, P, total_chunks, H, 1), (HP, 1, L * HP, P, T * HP))
    y_t = _to_cute(y_v, [2, 3, 4])
    # cs/nc descriptors are cached in the metadata (stable tensors).
    cs_t = meta["cs_t"]
    nc_t = meta["nc_t"]

    compiled_ssd_kernel = ws["compiled"]
    compiled_ssd_kernel(
        x_t,
        ws["cumsum_t"],
        ws["delta_t"],
        ws["b_t"],
        ws["c_t"],
        y_t,
        ws["fstate_t"],
        ws["d_t"],
        ws["init_t"],
        inter_t,
        es_t,
        cs_t,
        nc_t,
        stream,
    )

    if y_target is not out:
        out.copy_(y_target)
    fstate = ws["fstate_base"][..., :N].to(state_dtype)
    if has_empty:
        # Scatter the compacted real-seq states back to full batch shape.
        # Empty seqs processed no tokens -> final state == their initial state
        # (unchanged), or zeros when no initial state was provided.
        full = torch.zeros(S, H, P, N, device=device, dtype=state_dtype)
        if has_initial:
            full.copy_(initial_states.to(state_dtype))
        full[real_seq_idx] = fstate
        fstate = full
    if has_intermediate:
        # Duplicate emit indices collide in the emit_slot scatter above (the
        # dynamic engine pads intermediate_chunk_indices to a fixed size with
        # chunk 0), so the kernel writes only one winning slot per unique chunk
        # and the losing slots hold stale data. Gather every requested slot
        # from its chunk's winning row (emit_slot records the winner) to match
        # Triton's states[indices] gather semantics for any duplicate pattern,
        # without a host sync.
        winners = emit_slot[intermediate_chunk_indices].long()
        torch.index_select(inter_raw, 0, winners, out=inter_final)
        return fstate, inter_final[..., :N].to(state_dtype)
    return fstate
