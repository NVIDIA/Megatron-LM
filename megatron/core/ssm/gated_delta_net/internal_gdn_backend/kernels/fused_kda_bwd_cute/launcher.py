"""Host launcher for the SM100 fused KDA backward.

The token, chunk and sequence counts are dynamic in the compiled kernel, so one
compilation per (heads, split, device) serves every packing of a training run;
``split`` changes only when the sequence count crosses the one-wave bound.
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
    (computed on device when omitted). Returns ``(dq, dk, dv, dg, dbeta, dh0)``
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
    if split is None:
        split = default_split(cu_seqlens.numel() - 1, heads, q.device)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dg = torch.empty_like(gc)
    dbeta = torch.empty_like(beta)
    dh0 = torch.empty_like(dht)
    debug_shape = (h.shape[1], heads, len(DEBUG_SLOTS), _BT, _D) if debug else (1, 1, 1, 1, 1)
    debug_out = torch.zeros(debug_shape, device=q.device, dtype=torch.float32)
    trace_out = torch.zeros((4, 8, 16) if trace else (1, 1, 1), device=q.device, dtype=torch.int64)

    token_inputs = (q, k, v, gc, beta, aqk, akk)
    token_outputs = (dq, dk, dv, dg, dbeta)
    # Token, chunk and sequence counts stay dynamic; the debug buffers are only
    # for development and keep static shapes.
    args = (
        *(_dynamic(t, 1) for t in token_inputs),
        _dynamic(h, 1),
        _dynamic(do, 1),
        _dynamic(dht, 0),
        _dynamic(cu_seqlens, 0, align=4),
        _dynamic(chunk_offsets, 0, align=4),
        *(_dynamic(t, 1) for t in token_outputs),
        _dynamic(dh0, 0),
        from_dlpack(debug_out, assumed_align=16),
        from_dlpack(trace_out, assumed_align=16),
    )
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    device_index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    key = (heads, split, debug, trace, device_index)
    with _LOCK:
        compiled = _CACHE.get(key)
        if compiled is None:
            kernel = FusedKdaBwdKernel(heads=heads, split=split, debug=debug, trace=trace)
            compiled = cute.compile(kernel, *args, cutlass.Float32(scale), stream)
            _CACHE[key] = compiled
    compiled(*args, cutlass.Float32(scale), stream)
    outputs = (dq, dk, dv, dg, dbeta, dh0)
    if debug:
        outputs += (debug_out,)
    if trace:
        outputs += (trace_out,)
    return outputs
