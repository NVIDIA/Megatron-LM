# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from ._y_scatter import scatter_y_ragged

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
    TC_X,
    L,
    G,
    N_pad,
    io_dtype,
    cute_io_dtype,
    has_d,
    d_has_hdim,
    has_initial,
    has_intermediate,
    ragged,
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
    # return_raw_states emits EVERY chunk, i.e. a fixed identity emit map. It is
    # constant for the shape, so cache the buffer and its descriptor instead of
    # refilling per call.
    es_all = torch.arange(TC, device=device, dtype=torch.int32) if has_intermediate else None
    es_all_t = _to_cute(es_all, [0]) if has_intermediate else None
    # X is read from the caller's token stream on the GLOBAL chunk grid (TC_X
    # chunks); the workspace and Y use the per-sequence grid (TC chunks), which
    # exceeds TC_X only when the batch is ragged.
    x_ph = torch.empty(TC_X * L, H, P, device=device, dtype=io_dtype)
    x_ph_t = _to_cute(
        x_ph.as_strided((P, L, TC_X, H, 1), (1, HP, L * HP, P, TC_X * L * HP)), [2, 3, 4]
    )
    # A ragged batch cannot write Y in place: two sequences share the chunk
    # their boundary falls in and their TMA stores would clobber each other. The
    # kernel therefore targets a persistent scratch that is scattered back after
    # the launch. Being persistent, its descriptor is built once here rather
    # than per call.
    y_scratch = torch.empty(TC * L, H, P, device=device, dtype=io_dtype) if ragged else None
    y_ph = y_scratch if ragged else torch.empty(TC * L, H, P, device=device, dtype=io_dtype)
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
        y_scratch=y_scratch,
        y_scratch_t=y_ph_t if ragged else None,
        # emit_slot buffer + descriptor are cached (fixed (TC,) shape); only the
        # content is refilled per call, avoiding a per-call descriptor.
        emit_slot_buf=es_ph if has_intermediate else None,
        es_ph_t=es_ph_t,
        es_all_t=es_all_t,
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
    xs_t,
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
            xs_t,
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
    remainders = seq_lens % kernel_chunk_size
    divisible = bool((remainders == 0).all())  # sync
    # TAIL-RAGGED: every non-empty sequence except the LAST non-empty one is a
    # multiple of L. Then sequence starts are still L-aligned, so chunks keep
    # packing contiguously on the global grid (chunk c == tokens [c*L, (c+1)*L))
    # and only the final chunk is partially filled. That keeps the zero-copy
    # chunk-major X/Y views valid; the partial chunk is handled by zeroing
    # delta (and hence holding cumsum flat) on the pad lanes, which makes the
    # pad tokens contribute exactly nothing to the scan -- see the wrapper.
    # Interior-ragged batches would need per-sequence chunk grids AND a
    # predicated Y store (a TMA store of the tail chunk would clobber the next
    # sequence's output rows), so they still fall back to Triton.
    tail_ragged = False
    if not divisible:
        nonempty = torch.nonzero(seq_lens > 0, as_tuple=True)[0]  # sync
        ragged = torch.nonzero((seq_lens > 0) & (remainders != 0), as_tuple=True)[0]
        tail_ragged = (
            nonempty.numel() > 0 and ragged.numel() == 1 and int(ragged[0]) == int(nonempty[-1])
        )
    chunk_aligned = divisible or tail_ragged
    n_chunks_dev = chunk_start_dev = total_chunks = None
    n_padded_tokens = n_real_tokens
    ragged_meta = None
    cs_t = nc_t = xs_t = None
    real_seq_idx = None
    S_real = S
    has_empty = False
    real_is_prefix = True
    if chunk_aligned:
        # Varlen tile-scheduler metadata: per-seq chunk count + exclusive-cumsum
        # start, plus the global chunk index per token (for the dense scatter).
        # Chunk counts are CEIL so a ragged tail gets its own (partial) chunk;
        # this is exact for the divisible case (remainder 0).
        n_chunks_dev = ((seq_lens + kernel_chunk_size - 1) // kernel_chunk_size).to(torch.int32)
        chunk_start_dev = torch.cumsum(n_chunks_dev, 0, dtype=torch.int32) - n_chunks_dev
        total_chunks = (n_real_tokens + kernel_chunk_size - 1) // kernel_chunk_size
        n_padded_tokens = total_chunks * kernel_chunk_size
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
        # X reads the same (chunk-aligned) grid as the workspace here.
        xs_t = cs_t
    else:
        # GENERAL RAGGED: sequences start mid-chunk. Anchor each sequence's own
        # chunk grid at the L-aligned position at or below its start,
        # ``base_s = floor(cu[s] / L) * L``, so:
        #   * X still reads the GLOBAL aligned grid (chunk ``base_s / L + c``)
        #     and stays zero-copy;
        #   * delta/cumsum/B/C live in an EXPANDED workspace grid
        #     (``ws_start[s] + c``) so the boundary chunks that two sequences
        #     share get one masked copy EACH -- delta is zeroed outside
        #     ``[cu[s], cu[s+1])``, which removes the foreign tokens from the
        #     scan exactly (both the leading and the trailing ones);
        #   * Y is written to an expanded scratch on the same workspace grid and
        #     scattered back afterwards, because the two owners of a shared
        #     chunk must not write the same output rows.
        x_start_dev = (cu_seqlens[:-1] // kernel_chunk_size).to(torch.int32)
        span = cu_seqlens[1:] - x_start_dev.to(torch.long) * kernel_chunk_size
        n_chunks_dev = ((span + kernel_chunk_size - 1) // kernel_chunk_size).to(torch.int32)
        n_chunks_dev = torch.where(seq_lens > 0, n_chunks_dev, torch.zeros_like(n_chunks_dev))
        chunk_start_dev = torch.cumsum(n_chunks_dev, 0, dtype=torch.int32) - n_chunks_dev
        ws_total_chunks = int(n_chunks_dev.sum())  # sync (once per metadata)
        total_chunks = (n_real_tokens + kernel_chunk_size - 1) // kernel_chunk_size
        n_padded_tokens = total_chunks * kernel_chunk_size
        # Empty sequences own no chunk and are compacted out of the launch.
        real_seq_idx = torch.nonzero(n_chunks_dev, as_tuple=True)[0]  # sync (cached)
        S_real = int(real_seq_idx.numel())
        has_empty = S_real != S
        real_is_prefix = (not has_empty) or (S_real > 0 and int(real_seq_idx[-1]) == S_real - 1)
        if has_empty:
            nc_real = n_chunks_dev[real_seq_idx]
            cs_real = torch.cumsum(nc_real, 0, dtype=torch.int32) - nc_real
            xs_real = x_start_dev[real_seq_idx]
        else:
            nc_real, cs_real, xs_real = n_chunks_dev, chunk_start_dev, x_start_dev
        cs_t = _to_cute(cs_real, [0])
        nc_t = _to_cute(nc_real, [0])
        xs_t = _to_cute(xs_real, [0])
        # Per-workspace-chunk descriptors for the masking kernels and the Y
        # scatter: the first token of the chunk, and the real-token window.
        seq_of_chunk = torch.repeat_interleave(
            torch.arange(S, device=seq_lens.device), n_chunks_dev.to(torch.long)
        )
        idx_in_seq = (
            torch.arange(ws_total_chunks, device=seq_lens.device)
            - chunk_start_dev.to(torch.long)[seq_of_chunk]
        )
        ws_token_base = (
            x_start_dev.to(torch.long)[seq_of_chunk] * kernel_chunk_size
            + idx_in_seq * kernel_chunk_size
        ).to(torch.int32)
        ws_valid_lo = cu_seqlens[:-1][seq_of_chunk].to(torch.int32)
        ws_valid_hi = cu_seqlens[1:][seq_of_chunk].to(torch.int32)
        ragged_meta = dict(
            ws_total_chunks=ws_total_chunks,
            ws_token_base=ws_token_base,
            ws_valid_lo=ws_valid_lo,
            ws_valid_hi=ws_valid_hi,
        )
    m = dict(
        divisible=divisible,
        tail_ragged=tail_ragged,
        chunk_aligned=chunk_aligned,
        general_ragged=not chunk_aligned,
        ragged_meta=ragged_meta,
        xs_t=xs_t,
        n_padded_tokens=n_padded_tokens,
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
    return_raw_states: bool = False,
    intermediate_chunk_indices: torch.Tensor | None = None,
    kernel_chunk_size: int = 128,
) -> str | None:
    """Given a inference batch, check whether CuTe DSL is applicable"""
    if z is not None:
        return "CuteDSL THD SSD: z-gating not supported"
    has_intermediate = intermediate_chunk_indices is not None
    # Emitted states are indexed by chunk. The caller chunks at ``chunk_size``
    # while the kernel chunks at ``kernel_chunk_size``, so the two numberings
    # line up only when they are equal (the mamba mixer default).
    if (has_intermediate or return_raw_states) and chunk_size != kernel_chunk_size:
        return "CuteDSL THD SSD: emitted states need chunk_size == kernel L"
    meta = _chunk_meta(cu_chunk_seqlens, last_chunk_indices, kernel_chunk_size)
    if return_raw_states and not meta["chunk_aligned"]:
        # Raw states must be the state at the CALLER's chunk boundaries. On the
        # ragged path each sequence's grid is shifted to the L-aligned position
        # below its start, so our chunk boundaries fall inside the caller's
        # chunks and the states it wants are never materialised.
        return "CuteDSL THD SSD: return_raw_states needs chunk-aligned sequences"
    if return_raw_states and not meta["real_is_prefix"]:
        # Zero-length chunks of empty sequences occupy rows in the caller's
        # numbering; we can only append them when the real sequences form a
        # contiguous prefix.
        return "CuteDSL THD SSD: return_raw_states with interleaved empty sequences"
    if not meta["divisible"]:
        # Any ragged batch processes pad-masked chunks, so the caller's token
        # buffers must physically cover the padded chunk grid: the TMA
        # loads/stores whole L-token chunks, and the trailing rows of `out` are
        # overwritten (they are outside cu_chunk_seqlens, so undefined anyway).
        if x.shape[0] < meta["n_padded_tokens"]:
            return (
                "CuteDSL THD SSD: ragged batches need the token buffer padded to "
                "a multiple of the kernel chunk size"
            )
    if meta["general_ragged"] and has_intermediate:
        # Emission indexes the EXPANDED per-sequence chunk grid, which no longer
        # matches the caller's chunk numbering.
        return "CuteDSL THD SSD: intermediate states with interior-ragged sequences"
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
    return_raw_states: bool = False,
    intermediate_chunk_indices: torch.Tensor | None = None,
    state_dtype: torch.dtype | None = None,
    kernel_chunk_size: int = 128,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Host launcher for CuTe DSL SSD kernel.

    Both state-emission modes ride the same kernel machinery (a per-chunk
    ``emit_slot`` map plus a TMA store of the running state):

    * ``return_raw_states`` returns ``(final_states, raw_states)`` with one row
      per CALLER chunk, matching Triton. It emits every chunk via a cached
      identity emit map.
    * ``intermediate_chunk_indices`` emits only the flagged chunks (sparse
      prefix caching). Upstream no longer asks for this, but the tests cover it
      and it shares all of the raw-states plumbing.
    """
    has_initial = initial_states is not None
    has_intermediate = intermediate_chunk_indices is not None or return_raw_states

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
    # For a tail-ragged batch the final chunk is only partly real, so the views
    # must span the PADDED chunk boundary (n_padded_tokens >= n_real_tokens);
    # they coincide for the divisible case. The eligibility check guarantees the
    # caller's buffers are large enough.
    n_real_tokens = meta["n_real_tokens"]
    n_tokens = meta["n_padded_tokens"]
    if n_tokens != T:
        x = x[:n_tokens]
        B = B[:n_tokens]
        C = C[:n_tokens]
        dt = dt[:n_tokens]
        out = out[:n_tokens]
        T = n_tokens
    if n_tokens != n_real_tokens:
        # Pad lanes of the final chunk get delta == 0, so they contribute
        # nothing to the scan -- but only if their x is FINITE (0 * NaN = NaN).
        # These rows lie outside cu_chunk_seqlens (undefined by the op
        # contract, and `out`'s matching rows are overwritten anyway), so
        # zeroing them is safe and makes the path robust to uninitialized
        # token buffers.
        x[n_real_tokens:].zero_()

    has_d = D is not None
    # Varlen tile-scheduler path: every sequence's length is a multiple of L,
    # so chunks pack contiguously (total_chunks = T/L). x/y are zero-copy
    # chunk-major THD views; B/C/delta are dense chunk-major (cached
    # workspace); each (seq,head) work-item processes ONLY its own chunks.
    total_chunks = meta["total_chunks"]
    # Ragged batches run on an EXPANDED chunk grid (a chunk shared by two
    # sequences is materialised once per owner), so the workspace and Y scratch
    # are sized by ws_chunks while X keeps reading the global grid.
    ragged = meta["general_ragged"]
    rmeta = meta["ragged_meta"]
    ws_chunks = rmeta["ws_total_chunks"] if ragged else total_chunks
    ragged_chunks = (
        (rmeta["ws_token_base"], rmeta["ws_valid_lo"], rmeta["ws_valid_hi"]) if ragged else None
    )
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
        ws_chunks,
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
        ragged,
    )
    ws = _get_workspace(
        key,
        S_kernel,
        H,
        P,
        ws_chunks,
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
        ragged,
        stream,
    )

    # n_real_tokens masks the ragged tail: delta (and hence cumsum) is zeroed on
    # the pad lanes, which removes them from the scan exactly.
    fused_softplus_cumsum(
        dt,
        A,
        dt_bias,
        dt_softplus,
        dt_limit,
        ws["delta_d"],
        ws["cumsum_d"],
        1,
        H,
        ws_chunks,
        n_real_tokens,
        ragged_chunks,
    )
    if B.stride(2) == 1 and C.stride(2) == 1:
        # Tiled transpose: coalesced on both the token-major loads and the
        # chunk-major stores, B and C in one launch (~5x the strided copy_).
        repack_bc_chunk_major(
            B, C, ws["B_d"], ws["C_d"], N, ws_chunks, L, n_real_tokens, ragged_chunks
        )
    else:
        assert not ragged, "ragged path needs the tiled repack (B/C must be n-contiguous)"
        GN = G * N
        ws["B_d"][:, :, :N].copy_(B.as_strided((1, G, N, total_chunks, L), (0, N, 1, L * GN, GN)))
        ws["C_d"][:, :, :N].copy_(C.as_strided((1, G, N, total_chunks, L), (0, N, 1, L * GN, GN)))
        if n_tokens != n_real_tokens:
            # Keep the pad lanes finite (stale workspace content could be
            # anything); delta == 0 already removes their contribution.
            ws["B_d"][:, :, :, -1, n_real_tokens - n_tokens :] = 0
            ws["C_d"][:, :, :, -1, n_real_tokens - n_tokens :] = 0
    if has_d:
        ws["d_buf"].copy_(D.to(io_dtype) if d_has_hdim else D.to(io_dtype).view(H, 1))
    if has_initial:
        init_src = initial_states[real_seq_idx] if has_empty else initial_states
        ws["init_base"][..., :N].copy_(init_src.to(io_dtype))
    if return_raw_states:
        # One row per CALLER chunk. The real chunks are emitted by the kernel
        # (identity emit map, cached); the trailing rows belong to the
        # zero-length chunks of empty sequences and are filled below.
        n_caller_chunks = cu_chunk_seqlens.shape[0] - 1
        inter_raw, _, inter_t = _inter_out(n_caller_chunks, H, P, N_pad, io_dtype)
        es_t = ws["es_all_t"]
    elif has_intermediate:
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
    if ragged:
        # Kernel writes the expanded scratch (cached descriptor); the real
        # output is filled by the scatter below.
        y_target = None
        y_t = ws["y_scratch_t"]
    else:
        y_target = out if out.dtype == io_dtype else torch.empty_like(x)
        y_v = y_target.as_strided((L, P, total_chunks, H, 1), (HP, 1, L * HP, P, T * HP))
        y_t = _to_cute(y_v, [2, 3, 4])
    # cs/nc/xs descriptors are cached in the metadata (stable tensors).
    cs_t = meta["cs_t"]
    nc_t = meta["nc_t"]
    xs_t = meta["xs_t"]

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
        xs_t,
        stream,
    )

    if ragged:
        scatter_y_ragged(
            ws["y_scratch"],
            out,
            rmeta["ws_token_base"],
            rmeta["ws_valid_lo"],
            rmeta["ws_valid_hi"],
            L,
        )
    elif y_target is not out:
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
    if return_raw_states:
        # The kernel filled rows [0, total_chunks). The remaining rows are the
        # zero-length chunks the caller appends for empty sequences; Triton
        # reports the state passed into them, i.e. their initial state.
        raw = inter_raw[..., :N]
        if n_caller_chunks != total_chunks:
            tail = raw[total_chunks:]
            if has_initial:
                empty_seq_idx = torch.nonzero(meta["n_chunks_dev"] == 0, as_tuple=True)[0]
                tail.copy_(initial_states[empty_seq_idx].to(io_dtype))
            else:
                tail.zero_()
        return fstate, raw.to(state_dtype)
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
