"""Host launcher for the SM100 fused KDA backward.

The token, chunk and sequence counts are dynamic in the compiled kernels, so one
compilation per (heads, device) of each kernel serves every packing of a
training run; without a tape, ``split`` changes only when the sequence count
crosses the one-wave bound.
"""

import threading

import torch

_CACHE = {}
_LOCK = threading.Lock()
_BT = 64
_D = 128


def compile_cache_size() -> int:
    """Number of compiled fused KDA backward specializations in this process."""
    return len(_CACHE)


def chunk_offsets_for(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Per-sequence prefix of ``ceil(len / 64)`` chunk counts, without a host sync."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    chunks = (lengths + _BT - 1) // _BT
    return torch.cat([chunks.new_zeros(1), chunks.cumsum(0)]).to(torch.int32)


def default_split(num_sequences: int, heads: int, device: torch.device) -> int:
    """CTAs per (sequence, head): two while both fit on the device in one wave.

    Each CTA walks the whole dS recursion but runs the intra-chunk and dq/dk
    work for every ``split``-th chunk only, so a second CTA roughly halves that
    part of the per-chunk latency when SMs would otherwise stay idle.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return 2 if 2 * num_sequences * heads <= sms else 1


def prefer_tape(num_sequences: int, heads: int, device: torch.device) -> bool:
    """Whether to run the dS recurrence separately (``kda_dhu``) and the gradients from its tape.

    The fused kernel walks each (sequence, head) with one or two CTAs, so its
    parallelism is the pair count; the tape path spreads every chunk over the
    device and overlaps the recurrence with the gradients, at the cost of the
    extra prep and recurrence passes and of a transient tape as large as the
    forward's ``h``.  On GB200 the tape path is about 3% faster at eight pairs
    per SM and within 1% of the fused kernel beyond, so it is used up to
    eight pairs per SM.
    """
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return num_sequences * heads <= 8 * sms


# Chunks per work item of the persistent tape-fed kernel.
TAPE_RUN = 2


def tape_flags(num_chunks: int, heads: int, device: torch.device) -> torch.Tensor:
    """Zeroed int32 chunk ready flags (``[C * H]``) plus the work counter of the
    tape-fed gradient kernel, for ``kda_dhu(..., flags=...)`` and
    ``fused_kda_bwd(..., flags=...)``."""
    return torch.zeros(num_chunks * heads + 1, dtype=torch.int32, device=device)


def _require(name, tensor, dtype, shape, device):
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate(q, k, v, gc, beta, aqk, akk, h, do, dht, cu_seqlens, chunk_offsets):
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[3] != _D:
        raise ValueError(f"q must be [1, T, H, {_D}], got {tuple(q.shape)}")
    if h.ndim != 5:
        raise ValueError(f"h must be [1, C, H, {_D}, {_D}], got {tuple(h.shape)}")
    _, tokens, heads, _ = q.shape
    num_sequences = cu_seqlens.numel() - 1
    device = q.device
    io_shape = (1, tokens, heads, _D)
    for name, tensor in (("q", q), ("k", k), ("v", v), ("do", do)):
        _require(name, tensor, torch.bfloat16, io_shape, device)
    _require("gc", gc, torch.float32, io_shape, device)
    _require("beta", beta, torch.float32, (1, tokens, heads), device)
    _require("aqk", aqk, torch.bfloat16, (1, tokens, heads, _BT), device)
    _require("akk", akk, torch.bfloat16, (1, tokens, heads, _BT), device)
    _require("h", h, torch.bfloat16, (1, h.shape[1], heads, _D, _D), device)
    _require("dht", dht, torch.float32, (num_sequences, heads, _D, _D), device)
    if num_sequences < 1:
        raise ValueError("cu_seqlens must contain at least one sequence")
    _require("cu_seqlens", cu_seqlens, torch.int32, (num_sequences + 1,), device)
    _require("chunk_offsets", chunk_offsets, torch.int32, (num_sequences + 1,), device)


def _dynamic(tensor, mode, align=16):
    from cutlass.cute.runtime import from_dlpack

    descriptor = from_dlpack(tensor, assumed_align=align)
    descriptor.mark_compact_shape_dynamic(
        mode=mode, stride_order=tuple(range(tensor.ndim)), divisibility=1
    )
    return descriptor


def fused_kda_bwd(
    q,
    k,
    v,
    gc,
    beta,
    aqk,
    akk,
    h,
    do,
    dht,
    cu_seqlens,
    scale,
    *,
    chunk_offsets=None,
    split=None,
    tape=None,
    flags=None,
    debug=False,
    trace=False,
):
    """Run the fused KDA backward on packed inputs.

    With T packed tokens, H heads, C chunks of 64 tokens and S sequences:
    ``q``, ``k``, ``v`` and ``do`` are BF16 ``[1, T, H, 128]``; ``gc`` is FP32
    ``[1, T, H, 128]``, FLA's chunk-local log2 gate cumsum; ``beta`` is FP32
    ``[1, T, H]``; ``aqk`` (scaled, causal) and ``akk`` (``(I + L)^-1``) are
    BF16 ``[1, T, H, 64]``; ``h`` holds the BF16 chunk-start states
    ``[1, C, H, 128, 128]`` in ``[K, V]`` layout; ``dht`` is the FP32
    final-state gradient ``[S, H, 128, 128]``; ``cu_seqlens`` is int32
    ``[S + 1]``.  ``chunk_offsets`` is the per-sequence chunk prefix of ``cu_seqlens``
    (computed on device when omitted). ``tape`` is ``(dh, dv_new)`` from
    ``kda_dhu``: the kernel then skips the recurrence and persistent CTAs take
    runs of ``split`` (default ``TAPE_RUN``) chunks from a global counter; without a
    tape ``split`` is 1 or 2 (default ``default_split``) and dh0 is computed here.
    ``flags`` are the chunk ready flags ``kda_dhu`` raised for this tape: the
    kernel then launches as its programmatic dependent and starts on the chunks
    already written, so nothing may be enqueued between the two calls.
    Returns ``(dq, dk, dv, dg, dbeta, dh0)``
    and, with ``debug=True``, a sixth FP32 tensor of per-chunk intermediates
    ``[C, H, 10, 64, 128]``. With ``trace=True`` the last element is an int64
    ``[4 roles, 8 chunks, 16]`` tensor of ``%globaltimer`` phase stamps from CTA 0.
    """
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from .kernel import DEBUG_SLOTS, FusedKdaBwdKernel

    if chunk_offsets is None:
        chunk_offsets = chunk_offsets_for(cu_seqlens)
    _validate(q, k, v, gc, beta, aqk, akk, h, do, dht, cu_seqlens, chunk_offsets)
    heads = q.shape[2]
    if tape is not None:
        # (dh, dv_new) from a separate dS recurrence: BF16 [1, C, H, 128, 128]
        # (the dS of every chunk) and BF16 [1, T, H, 128].
        dh_tape, dv_new = tape
        _require("dh", dh_tape, torch.bfloat16, tuple(h.shape), q.device)
        _require("dv_new", dv_new, torch.bfloat16, tuple(v.shape), q.device)
        if split is None:
            split = TAPE_RUN
    else:
        dh_tape, dv_new = h, v
        if split is None:
            split = default_split(cu_seqlens.numel() - 1, heads, q.device)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dg = torch.empty_like(gc)
    dbeta = torch.empty_like(beta)
    dh0 = torch.empty_like(dht)
    # Two-CTA split: per (sequence, head) and CTA, the FP32 dS handed to the other CTA.
    xfer_rows = (cu_seqlens.numel() - 1) * heads * 2 if split == 2 and tape is None else 1
    xfer = torch.empty((xfer_rows, _D, _D), device=q.device, dtype=torch.float32)
    signal = flags is not None
    if signal and tape is None:
        raise ValueError("flags need a tape")
    ctas = 0
    if tape is not None:
        # Persistent CTAs take runs of ``split`` chunks from a global counter
        # (the last element of the flags, zeroed with them).
        items = (cu_seqlens.numel() - 1) * heads * -(-h.shape[1] // split)
        ctas = max(1, min(torch.cuda.get_device_properties(q.device).multi_processor_count, items))
    if signal:
        _require("flags", flags, torch.int32, (h.shape[1] * heads + 1,), q.device)
        work = flags[-1:]
    else:
        flags = torch.empty((1,), device=q.device, dtype=torch.int32)
        work = torch.zeros((1,), device=q.device, dtype=torch.int32)
    # Unused buffers stay uninitialized: with flags, no kernel may run between
    # the recurrence and this launch.
    alloc = torch.zeros if debug else torch.empty
    debug_shape = (h.shape[1], heads, len(DEBUG_SLOTS), _BT, _D) if debug else (1, 1, 1, 1, 1)
    debug_out = alloc(debug_shape, device=q.device, dtype=torch.float32)
    alloc = torch.zeros if trace else torch.empty
    trace_out = alloc((4, 8, 16) if trace else (1, 1, 1), device=q.device, dtype=torch.int64)

    token_inputs = (q, k, v, gc, beta, aqk, akk)
    token_outputs = (dq, dk, dv, dg, dbeta)
    # Token, chunk and sequence counts stay dynamic; the debug buffers are only
    # for development and keep static shapes.
    args = (
        *(_dynamic(t, 1) for t in token_inputs),
        _dynamic(h, 1),
        _dynamic(dh_tape, 1),
        _dynamic(dv_new, 1),
        _dynamic(do, 1),
        _dynamic(dht, 0),
        _dynamic(cu_seqlens, 0, align=4),
        _dynamic(chunk_offsets, 0, align=4),
        *(_dynamic(t, 1) for t in token_outputs),
        _dynamic(dh0, 0),
        _dynamic(xfer, 0),
        _dynamic(flags, 0, align=4),
        _dynamic(work, 0, align=4),
        from_dlpack(debug_out, assumed_align=16),
        from_dlpack(trace_out, assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    # With a tape the run length and the persistent CTA count are runtime arguments.
    compiled_split = 1 if tape is not None else split
    key = (heads, compiled_split, tape is not None, signal, debug, trace, device_index)
    runtime = (cutlass.Float32(scale), cutlass.Int32(split), cutlass.Int32(ctas), stream)
    with _LOCK:
        compiled = _CACHE.get(key)
        if compiled is None:
            kernel = FusedKdaBwdKernel(
                heads=heads,
                split=compiled_split,
                tape=tape is not None,
                signal=signal,
                debug=debug,
                trace=trace,
            )
            compiled = cute.compile(kernel, *args, *runtime)
            _CACHE[key] = compiled
    compiled(*args, *runtime)
    outputs = (dq, dk, dv, dg, dbeta, dh0)
    if debug:
        outputs += (debug_out,)
    if trace:
        outputs += (trace_out,)
    return outputs


def kda_dhu(
    q,
    k,
    gc,
    beta,
    aqk,
    akk,
    do,
    dht,
    cu_seqlens,
    scale,
    *,
    chunk_offsets=None,
    num_chunks=None,
    flags=None,
    keepalive=None,
    trace=False,
):
    """KDA's backward state recurrence from the forward residuals.

    Computes FLA's ``recompute_w_u_fwd`` (w, qg, kg), ``chunk_kda_bwd_dAv``
    (dv) and ``chunk_gated_delta_rule_bwd_dhu`` with ``gk``: a chunk-parallel
    Triton prep (``prep``) followed by the CuTe DSL recurrence (``dhu``).
    ``q``, ``k`` and ``do`` are BF16 ``[1, T, H, 128]``, ``gc`` the FP32
    chunk-local log2 gate cumsum, ``beta`` FP32 ``[1, T, H]``, ``aqk`` and
    ``akk`` BF16 ``[1, T, H, 64]`` and ``dht`` the FP32 ``[S, H, 128, 128]``
    final-state gradient; ``num_chunks`` (default: an upper bound) sizes the
    tape like the forward's ``h``.  ``flags`` (from ``tape_flags``) makes
    the recurrence raise a ready flag per chunk for ``fused_kda_bwd(...,
    flags=flags)`` launched right after it; that kernel overlaps this one, so
    the intermediate buffers this one reads are appended to ``keepalive``, which
    the caller must hold until it has launched ``fused_kda_bwd``.  Returns ``(dh, dh0, dv2)``: the
    BF16 per-chunk dS ``[1, C, H, 128, 128]``, the FP32 initial-state gradient
    and BF16 dv_new.
    """
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import from_dlpack

    from .dhu import KdaDhuKernel
    from .prep import kda_dhu_prep

    if chunk_offsets is None:
        chunk_offsets = chunk_offsets_for(cu_seqlens)
    _, tokens, heads, _ = q.shape
    device = q.device
    for name, tensor in (("q", q), ("k", k), ("do", do)):
        _require(name, tensor, torch.bfloat16, (1, tokens, heads, _D), device)
    _require("gc", gc, torch.float32, (1, tokens, heads, _D), device)
    _require("beta", beta, torch.float32, (1, tokens, heads), device)
    _require("aqk", aqk, torch.bfloat16, (1, tokens, heads, _BT), device)
    _require("akk", akk, torch.bfloat16, (1, tokens, heads, _BT), device)
    num_sequences = cu_seqlens.numel() - 1
    _require("dht", dht, torch.float32, (num_sequences, heads, _D, _D), device)
    if num_chunks is None:
        num_chunks = (tokens + _BT - 1) // _BT + num_sequences
    qg, kg, w = kda_dhu_prep(q, k, gc, beta, akk, cu_seqlens, chunk_offsets, num_chunks)
    if flags is not None:
        if keepalive is None:
            raise ValueError("flags need a keepalive list")
        keepalive.extend((qg, kg, w))
    dh = torch.empty((1, num_chunks, heads, _D, _D), device=device, dtype=torch.bfloat16)
    dv2 = torch.empty_like(do)
    dh0 = torch.empty_like(dht)
    trace_out = torch.zeros((8, 16) if trace else (1, 1), device=device, dtype=torch.int64)
    signal = flags is not None
    if signal:
        _require("flags", flags, torch.int32, (num_chunks * heads + 1,), device)
    else:
        flags = torch.empty((1,), device=device, dtype=torch.int32)
    args = (
        *(_dynamic(t, 1) for t in (kg, w, qg, do, aqk, gc)),
        _dynamic(dht, 0),
        _dynamic(cu_seqlens, 0, align=4),
        _dynamic(chunk_offsets, 0, align=4),
        _dynamic(dh, 1),
        _dynamic(dv2, 1),
        _dynamic(dh0, 0),
        _dynamic(flags, 0, align=4),
        from_dlpack(trace_out, assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    key = ("dhu", heads, signal, trace, device_index)
    with _LOCK:
        compiled = _CACHE.get(key)
        if compiled is None:
            compiled = cute.compile(
                KdaDhuKernel(heads=heads, signal=signal, trace=trace),
                *args,
                cutlass.Float32(scale),
                stream,
            )
            _CACHE[key] = compiled
    compiled(*args, cutlass.Float32(scale), stream)
    if trace:
        return dh, dh0, dv2, trace_out
    return dh, dh0, dv2
