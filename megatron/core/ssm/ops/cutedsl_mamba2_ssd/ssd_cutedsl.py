# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import logging

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from megatron.core.utils import round_up_to_nearest_multiple

from ._bc_repack import repack_bc_chunk_major
from ._fused_cumsum import fused_softplus_cumsum
from ._mamba2_ssd_kernel_varlen import MMA_N_GRANULARITY
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
_WORKSPACE_CACHE = {}
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
    workspace = _WORKSPACE_CACHE.get(key)
    if workspace is not None:
        return workspace
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
    workspace = dict(
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
    _WORKSPACE_CACHE[key] = workspace
    return workspace


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


class SSDTiling:
    """How a varlen batch is tiled, built from what MambaMetadata publishes.

    The device arrays are int32 views into the engine's per-step buffers, so an
    instance is only valid for the step that produced it. The scalars are host
    values the op cannot derive from those arrays without a device->host sync.
    Everything else the launcher needs is arithmetic on these.

    Attributes:
        seq_chunk_start: Per active sequence, its first chunk in workspace order.
        seq_chunk_count: Per active sequence, how many chunks it owns.
        seq_chunk_base: Per active sequence, its first chunk as a GLOBAL index.
        active_seq_idx: Batch slots that carry tokens.
        empty_seq_idx: Batch slots that do not.
        chunk_token_base: Per workspace chunk, its first token.
        chunk_valid_start: Per workspace chunk, the owner's first real token.
        chunk_valid_end: Per workspace chunk, one past the owner's last real token.
        chunk_size: The granularity the arrays were built at. Must equal the
            kernel's L, or the tiling describes a different grid than the kernel
            walks.
        num_slots: Batch slots including the empty ones.
        num_real_tokens: Tokens actually covered by the sequences.
        starts_aligned: True when no sequence starts mid-chunk, so no chunk is
            shared and Y can be written in place.
        active_is_prefix: True when the active slots come first, so the packed
            chunk numbering matches the caller's.
    """

    __slots__ = (
        "seq_chunk_start",
        "seq_chunk_count",
        "seq_chunk_base",
        "active_seq_idx",
        "empty_seq_idx",
        "chunk_token_base",
        "chunk_valid_start",
        "chunk_valid_end",
        "chunk_size",
        "num_slots",
        "num_real_tokens",
        "starts_aligned",
        "active_is_prefix",
    )

    def __init__(self, metadata):
        """Read the per-step tiling off the batch metadata.

        Args:
            metadata: The step's ``MambaMetadata``. Only read through attribute
                access -- the op library imports nothing from the inference
                layer, so any object exposing the same ``ssd_*`` fields,
                ``mamba_chunk_size``, ``cu_seqlens`` and
                ``real_prefill_token_count`` works (tests pass a stand-in).
        """
        self.seq_chunk_start = metadata.ssd_seq_chunk_start
        self.seq_chunk_count = metadata.ssd_seq_chunk_count
        self.seq_chunk_base = metadata.ssd_seq_chunk_base
        self.active_seq_idx = metadata.ssd_active_seq_idx
        self.empty_seq_idx = metadata.ssd_empty_seq_idx
        self.chunk_token_base = metadata.ssd_chunk_token_base
        self.chunk_valid_start = metadata.ssd_chunk_valid_start
        self.chunk_valid_end = metadata.ssd_chunk_valid_end
        self.chunk_size = metadata.mamba_chunk_size
        self.num_slots = metadata.cu_seqlens.shape[0] - 1
        self.num_real_tokens = metadata.real_prefill_token_count
        self.starts_aligned = metadata.ssd_starts_aligned
        self.active_is_prefix = metadata.ssd_active_is_prefix


def cutedsl_unsupported_reason(
    x: torch.Tensor,
    chunk_size: int,
    tiling: SSDTiling,
    *,
    z: torch.Tensor | None = None,
    return_raw_states: bool = False,
    intermediate_chunk_indices: torch.Tensor | None = None,
    kernel_chunk_size: int = 128,
) -> str | None:
    """Why this batch cannot run on the CuteDSL SSD kernel, or None if it can.

    Sequence lengths are unconstrained; what is left are the argument
    combinations the kernel does not implement. Callers must consult this BEFORE
    calling the wrapper, which assumes eligibility and does not re-check.

    Args:
        x: Token-packed input, used only for its buffer length.
        chunk_size: The caller's chunk size.
        tiling: How this batch is tiled, from ``MambaMetadata``.
        z: Gating input, if any.
        return_raw_states: Whether the caller wants every chunk's state.
        intermediate_chunk_indices: Sparse emission map, if any.
        kernel_chunk_size: The kernel's L.

    Returns:
        A human-readable reason, or None when the batch is eligible.
    """
    if z is not None:
        return "CuteDSL THD SSD: z-gating not supported"

    L = kernel_chunk_size
    # The kernel walks its own L-sized grid, so the tiling has to describe that
    # same grid. The CALLER's chunk_size is irrelevant here -- SSD results do not
    # depend on how the caller chunked -- except for state emission below, which
    # is numbered per caller chunk.
    if tiling.chunk_size != L:
        return "CuteDSL THD SSD: tiling chunk size does not match the kernel L"

    emits_states = return_raw_states or intermediate_chunk_indices is not None
    if emits_states and chunk_size != L:
        return "CuteDSL THD SSD: emitted states need chunk_size == kernel L"

    # Partial chunks are pad-masked, so the caller's token buffers must
    # physically cover the padded chunk grid: the TMA moves whole L-token chunks,
    # and the trailing rows of `out` get overwritten (they are outside the
    # sequences, hence undefined by the op contract anyway).
    if x.shape[0] < round_up_to_nearest_multiple(tiling.num_real_tokens, L):
        return (
            "CuteDSL THD SSD: ragged batches need the token buffer padded to "
            "a multiple of the kernel chunk size"
        )

    if emits_states:
        # Emission is numbered over the kernel's own chunk grid. That matches the
        # caller's numbering only when no sequence starts mid-chunk (else the
        # grid is expanded per owner) and the active slots form a contiguous
        # prefix (else the packed numbering skips the empty slots).
        if not tiling.starts_aligned:
            return "CuteDSL THD SSD: emitted states need chunk-aligned sequence starts"
        if not tiling.active_is_prefix:
            return "CuteDSL THD SSD: emitted states with interleaved empty sequences"
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
    tiling: SSDTiling,
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
    # dstate is the MMA M-mode of the kernel's inter1 tile, which tcgen05
    # only supports at this granularity.
    N_pad = round_up_to_nearest_multiple(N, MMA_N_GRANULARITY)
    device = x.device
    out_dtype = x.dtype
    io_dtype = x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
    cute_io_dtype = _torch_to_cute_dtype(io_dtype)
    if state_dtype is None:
        state_dtype = out_dtype
    if x.dtype != io_dtype:
        x = x.to(io_dtype)
    HP = H * P

    S = tiling.num_slots
    # Advanced indexing needs int64; the engine publishes int32 to keep its
    # bookkeeping buffer uniform.
    real_seq_idx = tiling.active_seq_idx.long()
    empty_seq_idx = tiling.empty_seq_idx.long()

    # Dynamic inference hands fixed-size CUDA-graph token buffers with trailing
    # padding (x.shape[0] > real tokens). All CuteDSL paths assume token-packed
    # input with no tail, so trim x/B/C/dt (and the output view) to the real
    # tokens; padded rows carry no state (they are outside cu_chunk_seqlens) and
    # their outputs are ignored by the caller.
    # For a tail-ragged batch the final chunk is only partly real, so the views
    # must span the PADDED chunk boundary (n_padded_tokens >= n_real_tokens);
    # they coincide for the divisible case. The eligibility check guarantees the
    # caller's buffers are large enough.
    n_real_tokens = tiling.num_real_tokens
    total_chunks = -(-n_real_tokens // L)
    n_tokens = total_chunks * L
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
    # Varlen tile-scheduler path: each (seq, head) work-item walks only its own
    # chunks. X is always a zero-copy chunk-major THD view on the global grid;
    # B/C/delta live in the (cached) dense workspace on the per-sequence grid,
    # which is larger than the global one only when a chunk is shared.
    workspace_chunks = tiling.chunk_token_base.numel()
    # Sharing is exactly what makes an in-place Y store unsafe: both owners
    # would TMA-store the same output rows, so Y goes via a scratch instead.
    ragged = not tiling.starts_aligned
    ragged_chunks = (
        (tiling.chunk_token_base, tiling.chunk_valid_start, tiling.chunk_valid_end)
        if ragged
        else None
    )
    # Empty (padded) sequences are compacted out of the kernel launch: run over
    # only the S_real non-empty seqs, then scatter final states back to full S.
    S_kernel = real_seq_idx.numel()
    has_empty = S_kernel != S
    stream = _current_cute_stream()
    key = (
        S_kernel,
        H,
        P,
        workspace_chunks,
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
    workspace = _get_workspace(
        key,
        S_kernel,
        H,
        P,
        workspace_chunks,
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
        workspace["delta_d"],
        workspace["cumsum_d"],
        1,
        H,
        workspace_chunks,
        n_real_tokens,
        ragged_chunks,
    )
    if B.stride(2) == 1 and C.stride(2) == 1:
        # Tiled transpose: coalesced on both the token-major loads and the
        # chunk-major stores, B and C in one launch (~5x the strided copy_).
        repack_bc_chunk_major(
            B,
            C,
            workspace["B_d"],
            workspace["C_d"],
            N,
            workspace_chunks,
            L,
            n_real_tokens,
            ragged_chunks,
        )
    else:
        assert not ragged, "ragged path needs the tiled repack (B/C must be n-contiguous)"
        GN = G * N
        workspace["B_d"][:, :, :N].copy_(
            B.as_strided((1, G, N, total_chunks, L), (0, N, 1, L * GN, GN))
        )
        workspace["C_d"][:, :, :N].copy_(
            C.as_strided((1, G, N, total_chunks, L), (0, N, 1, L * GN, GN))
        )
        if n_tokens != n_real_tokens:
            # Keep the pad lanes finite (stale workspace content could be
            # anything); delta == 0 already removes their contribution.
            workspace["B_d"][:, :, :, -1, n_real_tokens - n_tokens :] = 0
            workspace["C_d"][:, :, :, -1, n_real_tokens - n_tokens :] = 0
    if has_d:
        workspace["d_buf"].copy_(D.to(io_dtype) if d_has_hdim else D.to(io_dtype).view(H, 1))
    if has_initial:
        init_src = initial_states[real_seq_idx] if has_empty else initial_states
        workspace["init_base"][..., :N].copy_(init_src.to(io_dtype))
    if return_raw_states:
        # One row per CALLER chunk. The real chunks are emitted by the kernel
        # (identity emit map, cached); the trailing rows belong to the
        # zero-length chunks of empty sequences and are filled below.
        n_caller_chunks = cu_chunk_seqlens.shape[0] - 1
        inter_raw, _, inter_t = _inter_out(n_caller_chunks, H, P, N_pad, io_dtype)
        es_t = workspace["es_all_t"]
    elif has_intermediate:
        num_inter = intermediate_chunk_indices.shape[0]
        emit_slot = workspace["emit_slot_buf"]
        emit_slot.fill_(-1)
        emit_slot[intermediate_chunk_indices] = torch.arange(
            num_inter, dtype=torch.int32, device=device
        )
        inter_raw, inter_final, inter_t = _inter_out(num_inter, H, P, N_pad, io_dtype)
        es_t = workspace["es_ph_t"]
    else:
        inter_t = workspace["inter_ph_t"]
        es_t = workspace["es_ph_t"]

    x_v = x.as_strided((P, L, total_chunks, H, 1), (1, HP, L * HP, P, T * HP))
    x_t = _to_cute(x_v, [2, 3, 4])
    if ragged:
        # Kernel writes the expanded scratch (cached descriptor); the real
        # output is filled by the scatter below.
        y_target = None
        y_t = workspace["y_scratch_t"]
    else:
        y_target = out if out.dtype == io_dtype else torch.empty_like(x)
        y_v = y_target.as_strided((L, P, total_chunks, H, 1), (HP, 1, L * HP, P, T * HP))
        y_t = _to_cute(y_v, [2, 3, 4])
    # cs/nc/xs descriptors are cached in the metadata (stable tensors).
    # Built per launch: the tiling arrays are views into the engine's per-step
    # buffers, so a descriptor must not outlive the step that produced them.
    cs_t = _to_cute(tiling.seq_chunk_start, [0])
    nc_t = _to_cute(tiling.seq_chunk_count, [0])
    xs_t = _to_cute(tiling.seq_chunk_base, [0])

    compiled_ssd_kernel = workspace["compiled"]
    compiled_ssd_kernel(
        x_t,
        workspace["cumsum_t"],
        workspace["delta_t"],
        workspace["b_t"],
        workspace["c_t"],
        y_t,
        workspace["fstate_t"],
        workspace["d_t"],
        workspace["init_t"],
        inter_t,
        es_t,
        cs_t,
        nc_t,
        xs_t,
        stream,
    )

    if ragged:
        scatter_y_ragged(
            workspace["y_scratch"],
            out,
            tiling.chunk_token_base,
            tiling.chunk_valid_start,
            tiling.chunk_valid_end,
            L,
        )
    elif y_target is not out:
        out.copy_(y_target)
    fstate = workspace["fstate_base"][..., :N].to(state_dtype)
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
