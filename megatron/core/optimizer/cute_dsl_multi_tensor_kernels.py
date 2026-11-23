# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Persistent-grid multi-tensor Adam and L2-norm kernels written in CuTe DSL.

They replace Transformer Engine's ``multi_tensor_adam`` (``AdamFunctor<float, float,
float, float>``) and ``multi_tensor_l2norm`` (``L2NormFunctor<float>`` + ``cleanup``) for
fp32 main params, grads and Adam moments. TE's ``multi_tensor_apply`` caps a launch at 320
blocks of one 65,536-element chunk each, so one optimizer call over a few billion elements
issues hundreds of launches. Here one launch covers a whole tensor list: a persistent grid
of ``CTAS_PER_SM x SMs`` CTAs pulls chunks from a global atomic counter until the list is
exhausted.

Terms and dtypes follow TE (``adam.cu``, ``l2norm.cu``):

* Adam (``ADAM_MODE_1``, decoupled weight decay), fp32 g / p / m / v::

      m = beta1 * m + (1 - beta1) * g
      v = beta2 * v + (1 - beta2) * g * g
      p = p - lr * ((m / bc1) / (sqrt(v / bc2) + eps) + wd * p)

  with ``bc1 = 1 - beta1**step``, ``bc2 = 1 - beta2**step`` computed on the host in double
  from the fp32 betas and rounded to fp32, as TE's launcher does.
* L2 norm of fp32 tensors: fp32 per-chunk partial sums of squares, then one reduction of
  the partials and ``sqrt``; the no-op flag is set when the norm is not finite. The
  partials are reduced in chunk order by the last CTA to finish, so the result is
  deterministic. Only the fp32 summation order differs from TE.

The tensor list reaches the kernel as an int64 table: per tensor the addresses of its
``D`` lists and its numel, then per chunk ``(tensor << 32) | chunk``. The table stays on
the device between calls: Adam's is keyed by the identity of its moment tensors (whose
addresses it keeps), the L2 norm's by the numels, and each call re-reads and uploads only
the other lists' addresses.

Registered as ``torch.ops.megatron_multi_tensor.adam_table`` and
``torch.ops.megatron_multi_tensor.l2norm_table``; ``multi_tensor_adam`` and
``multi_tensor_l2norm`` build the table from tensor lists and call them.

Launch configuration (512 threads, 2 CTAs/SM, 2 vectors in flight per thread) from a
one-GPU sweep on GB300 over 20k chunks / 1.29 G fp32 elements: Adam 5.30 ms (6.8 TB/s)
vs TE 6.52 ms; L2 norm 0.735 ms (7.0 TB/s) vs TE 1.60 ms.
"""

import itertools
import weakref
from functools import lru_cache
from typing import Callable

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.experimental import primitives as prims

CHUNK = 65536
"""Elements per chunk (TE's ``multi_tensor_applier`` chunk size)."""
THREADS = 512
CTAS_PER_SM = 2
UNROLL = 2
_WARP = 32
_NW = THREADS // _WARP


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@cute.jit
def _chunk_base(
    mp, num_tensors, c, D: cutlass.Constexpr, d: cutlass.Constexpr, VEC: cutlass.Constexpr
):
    """Return (pointer to chunk ``c`` of list ``d``, elements in the chunk)."""
    packed = (mp + (D + 1) * num_tensors + c).load()
    t = cutlass.Int32(packed >> 32)
    lc = packed & cutlass.Int64(0xFFFFFFFF)
    off = lc * CHUNK
    n = (mp + D * num_tensors + t).load() - off
    if n > CHUNK:
        n = cutlass.Int64(CHUNK)
    addr = (mp + d * num_tensors + t).load() + off * 4
    ptr = cute.make_ptr(cutlass.Float32, addr, cute.AddressSpace.gmem, assumed_align=4 * VEC)
    return ptr.raw_ptr(), cutlass.Int32(n)


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------


@cute.jit
def _adam_vec(
    gp, pp, mp_, vp, idx, lr, b1, b2, omb1, omb2, bc1, bc2, eps, wd, VEC: cutlass.Constexpr
):
    g = (gp + idx).load(count=VEC, alignment=4 * VEC)
    p = (pp + idx).load(count=VEC, alignment=4 * VEC)
    m = (mp_ + idx).load(count=VEC, alignment=4 * VEC)
    v = (vp + idx).load(count=VEC, alignment=4 * VEC)
    m = m * b1 + g * omb1
    v = v * b2 + g * omb2 * g
    denom = cute.math.sqrt(v / bc2) + eps
    upd = (m / bc1) / denom + p * wd
    p = p - upd * lr
    (pp + idx).store(p, alignment=4 * VEC)
    (mp_ + idx).store(m, alignment=4 * VEC)
    (vp + idx).store(v, alignment=4 * VEC)


@cute.kernel
def _adam_kernel(
    meta: cute.Tensor,  # int64 table
    ws: cute.Tensor,  # int32 [2]: chunk counter, done counter (zero between launches)
    num_tensors: cutlass.Int32,
    num_chunks: cutlass.Int32,
    lr: cutlass.Float32,
    b1: cutlass.Float32,
    b2: cutlass.Float32,
    bc1: cutlass.Float32,
    bc2: cutlass.Float32,
    eps: cutlass.Float32,
    wd: cutlass.Float32,
    VEC: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    nctas, _, _ = cute.arch.grid_dim()
    slot = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem)
    sp = slot.data_ptr()
    mp = meta.iterator.raw_ptr()
    wsp = ws.iterator.raw_ptr()
    omb1 = cutlass.Float32(1.0) - b1
    omb2 = cutlass.Float32(1.0) - b2

    if tid == 0:
        sp.store(cute.arch.atomic_add(wsp, cutlass.Int32(1), scope="gpu"))
    prims.barrier_cta_sync(0)
    parity = cutlass.Int32(0)
    c = sp.load()
    while c < num_chunks:
        if tid == 0:
            (sp + 1 - parity).store(cute.arch.atomic_add(wsp, cutlass.Int32(1), scope="gpu"))
        gp, n = _chunk_base(mp, num_tensors, c, 4, 0, VEC)
        pp, _ = _chunk_base(mp, num_tensors, c, 4, 1, VEC)
        mmp, _ = _chunk_base(mp, num_tensors, c, 4, 2, VEC)
        vp, _ = _chunk_base(mp, num_tensors, c, 4, 3, VEC)
        nv = n // VEC
        groups = nv // (UNROLL * THREADS)
        for gi in range(groups):
            for u in cutlass.range_constexpr(UNROLL):
                idx = ((gi * UNROLL + u) * THREADS + tid) * VEC
                _adam_vec(gp, pp, mmp, vp, idx, lr, b1, b2, omb1, omb2, bc1, bc2, eps, wd, VEC)
        for i in range(groups * UNROLL * THREADS + tid, nv, THREADS):
            _adam_vec(gp, pp, mmp, vp, i * VEC, lr, b1, b2, omb1, omb2, bc1, bc2, eps, wd, VEC)
        if cutlass.const_expr(VEC > 1):
            if tid < n - nv * VEC:
                _adam_vec(
                    gp, pp, mmp, vp, nv * VEC + tid, lr, b1, b2, omb1, omb2, bc1, bc2, eps, wd, 1
                )
        prims.barrier_cta_sync(0)
        parity = 1 - parity
        c = (sp + parity).load()

    if tid == 0:
        done = cute.arch.atomic_add(wsp + 1, cutlass.Int32(1), sem="acq_rel", scope="gpu")
        if done == nctas - 1:
            wsp.store(cutlass.Int32(0))
            (wsp + 1).store(cutlass.Int32(0))


@cute.jit
def _adam_host(
    meta: cute.Tensor,
    ws: cute.Tensor,
    num_tensors: cutlass.Int32,
    num_chunks: cutlass.Int32,
    lr: cutlass.Float32,
    b1: cutlass.Float32,
    b2: cutlass.Float32,
    bc1: cutlass.Float32,
    bc2: cutlass.Float32,
    eps: cutlass.Float32,
    wd: cutlass.Float32,
    nctas: cutlass.Int32,
    stream,
    VEC: cutlass.Constexpr,
) -> None:
    _adam_kernel(meta, ws, num_tensors, num_chunks, lr, b1, b2, bc1, bc2, eps, wd, VEC).launch(
        grid=(nctas, 1, 1), block=(THREADS, 1, 1), min_blocks_per_mp=CTAS_PER_SM, stream=stream
    )


# ---------------------------------------------------------------------------
# L2 norm
# ---------------------------------------------------------------------------


@cute.jit
def _block_sum(val, red_ptr, tid):
    """Sum ``val`` over the CTA into thread 0 (needs a barrier before red reuse)."""
    for sh in [16, 8, 4, 2, 1]:
        val = val + cute.arch.shuffle_sync_bfly(val, sh)
    if tid % _WARP == 0:
        (red_ptr + tid // _WARP).store(val)


@cute.kernel
def _l2norm_kernel(
    meta: cute.Tensor,  # int64 table
    ws: cute.Tensor,  # int32 [2]
    partial: cute.Tensor,  # fp32 [num_chunks]
    ret: cute.Tensor,  # fp32 [1]
    noop: cute.Tensor,  # int32 [1]
    num_tensors: cutlass.Int32,
    num_chunks: cutlass.Int32,
    VEC: cutlass.Constexpr,
) -> None:
    tid, _, _ = cute.arch.thread_idx()
    nctas, _, _ = cute.arch.grid_dim()
    slot = cutlass.Array(cutlass.Int32, 3, space=cutlass.AddressSpace.smem)
    red = cutlass.Array(cutlass.Float32, 2 * _NW, space=cutlass.AddressSpace.smem)
    sp = slot.data_ptr()
    rp = red.data_ptr()
    mp = meta.iterator.raw_ptr()
    wsp = ws.iterator.raw_ptr()
    pp = partial.iterator.raw_ptr()

    if tid == 0:
        sp.store(cute.arch.atomic_add(wsp, cutlass.Int32(1), scope="gpu"))
    prims.barrier_cta_sync(0)
    parity = cutlass.Int32(0)
    c = sp.load()
    while c < num_chunks:
        if tid == 0:
            (sp + 1 - parity).store(cute.arch.atomic_add(wsp, cutlass.Int32(1), scope="gpu"))
        xp, n = _chunk_base(mp, num_tensors, c, 1, 0, VEC)
        nv = n // VEC
        groups = nv // (UNROLL * THREADS)
        s = cutlass.Float32(0.0)
        for gi in range(groups):
            xs = []
            for u in cutlass.range_constexpr(UNROLL):
                idx = ((gi * UNROLL + u) * THREADS + tid) * VEC
                xs.append((xp + idx).load(count=VEC, alignment=4 * VEC))
            for u in cutlass.range_constexpr(UNROLL):
                s = s + (xs[u] * xs[u]).reduce("add")
        for i in range(groups * UNROLL * THREADS + tid, nv, THREADS):
            x = (xp + i * VEC).load(count=VEC, alignment=4 * VEC)
            s = s + (x * x).reduce("add")
        if cutlass.const_expr(VEC > 1):
            if tid < n - nv * VEC:
                x1 = (xp + nv * VEC + tid).load()
                s = s + x1 * x1
        _block_sum(s, rp + parity * _NW, tid)
        prims.barrier_cta_sync(0)
        if tid == 0:
            tot = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(_NW):
                tot = tot + (rp + parity * _NW + w).load()
            (pp + c).store(tot)
        parity = 1 - parity
        c = (sp + parity).load()

    if tid == 0:
        done = cute.arch.atomic_add(wsp + 1, cutlass.Int32(1), sem="acq_rel", scope="gpu")
        (sp + 2).store(cutlass.Int32(done == nctas - 1))
    prims.barrier_cta_sync(0)
    if (sp + 2).load() != 0:
        s = cutlass.Float32(0.0)
        for i in range(tid, num_chunks, THREADS):
            s = s + (pp + i).load()
        _block_sum(s, rp, tid)
        prims.barrier_cta_sync(0)
        if tid == 0:
            tot = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(_NW):
                tot = tot + (rp + w).load()
            ret.iterator.raw_ptr().store(cute.math.sqrt(tot))
            if cutlass.Int32(tot < cutlass.Float32(float("inf"))) == 0:
                noop.iterator.raw_ptr().store(cutlass.Int32(1))
            wsp.store(cutlass.Int32(0))
            (wsp + 1).store(cutlass.Int32(0))


@cute.jit
def _l2norm_host(
    meta: cute.Tensor,
    ws: cute.Tensor,
    partial: cute.Tensor,
    ret: cute.Tensor,
    noop: cute.Tensor,
    num_tensors: cutlass.Int32,
    num_chunks: cutlass.Int32,
    nctas: cutlass.Int32,
    stream,
    VEC: cutlass.Constexpr,
) -> None:
    _l2norm_kernel(meta, ws, partial, ret, noop, num_tensors, num_chunks, VEC).launch(
        grid=(nctas, 1, 1), block=(THREADS, 1, 1), min_blocks_per_mp=CTAS_PER_SM, stream=stream
    )


# ---------------------------------------------------------------------------
# Compile factories
# ---------------------------------------------------------------------------


def _fake(dtype, align=16):
    return make_fake_compact_tensor(dtype, (cute.sym_int64(),), assumed_align=align)


@lru_cache(maxsize=None)
def compile_adam(VEC: int = 4) -> Callable:
    """Compile the persistent Adam kernel (``VEC`` fp32 per access)."""
    f = cutlass.Float32(0.0)
    return cute.compile(
        _adam_host,
        _fake(cutlass.Int64),
        _fake(cutlass.Int32, 4),
        cutlass.Int32(1),
        cutlass.Int32(1),
        f,
        f,
        f,
        f,
        f,
        f,
        f,
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        VEC,
        options="--enable-tvm-ffi",
    )


@lru_cache(maxsize=None)
def compile_l2norm(VEC: int = 4) -> Callable:
    """Compile the persistent L2-norm kernel (``VEC`` fp32 per access)."""
    return cute.compile(
        _l2norm_host,
        _fake(cutlass.Int64),
        _fake(cutlass.Int32, 4),
        _fake(cutlass.Float32, 4),
        _fake(cutlass.Float32, 4),
        _fake(cutlass.Int32, 4),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        VEC,
        options="--enable-tvm-ffi",
    )


# ---------------------------------------------------------------------------
# Host-side tensor table
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


_WS: dict[tuple[int, int], torch.Tensor] = {}


def _workspace(device: torch.device, stream: int) -> torch.Tensor:
    """Zeroed int32 [2] counters for ``stream``; kernels leave them zeroed."""
    key = (device.index, stream)
    ws = _WS.get(key)
    if ws is None:
        ws = torch.zeros(2, dtype=torch.int32, device=device)
        _WS[key] = ws
    return ws


class _Table:
    """A tensor list's device table, reused by every call with the same key.

    ``meta`` holds the full table. The chunk map, the numels and the addresses
    of the ``fixed`` trailing lists are written once; the addresses of the other
    (fresh) lists are re-read on every call and copied into the head of ``meta``
    on the stream that then runs the kernel.
    """

    __slots__ = (
        "meta",
        "num_tensors",
        "num_chunks",
        "fixed_aligned",
        "probes",
        "watch",
        "__weakref__",
    )


_TABLES: dict[tuple, _Table] = {}
_MAX_TABLES = 64


def _f32(lst: list[torch.Tensor]) -> bool:
    return all(t.dtype is torch.float32 for t in lst)


def _evict(key: tuple, entry_ref: weakref.ref) -> Callable:
    def cb(_):
        if _TABLES.get(key) is entry_ref():
            del _TABLES[key]

    return cb


def _build(key, tensor_lists, fixed: int, dev: torch.device) -> _Table | None:
    first = tensor_lists[0]
    n = len(first)
    lists = len(tensor_lists)
    fixed_lists = tensor_lists[lists - fixed :]
    if not all(_f32(lst) for lst in fixed_lists):
        return None
    numel = np.array([t.numel() for t in first], dtype=np.int64)
    chunks = (numel + CHUNK - 1) // CHUNK
    num_chunks = int(chunks.sum())
    h = np.zeros((lists + 1) * n + num_chunks, dtype=np.int64)
    for d in range(lists - fixed, lists):
        h[d * n : (d + 1) * n] = [t.data_ptr() for t in tensor_lists[d]]
    h[lists * n : (lists + 1) * n] = numel
    starts = np.cumsum(chunks) - chunks
    tensor_of_chunk = np.repeat(np.arange(n, dtype=np.int64), chunks)
    local = np.arange(num_chunks, dtype=np.int64) - np.repeat(starts, chunks)
    h[(lists + 1) * n :] = (tensor_of_chunk << 32) | local
    e = _Table()
    e.meta = torch.from_numpy(h).to(dev)
    e.num_tensors = n
    e.num_chunks = num_chunks
    e.fixed_aligned = not np.any(h[(lists - fixed) * n : lists * n] % 16)
    # A fixed list is keyed by its tensors' identity, so the entry dies with any of
    # them (an id is unique only while its object lives); the first and last
    # addresses are re-checked on every call.
    e.probes = [(lst[0].data_ptr(), lst[-1].data_ptr()) for lst in fixed_lists]
    cb = _evict(key, weakref.ref(e))
    e.watch = [weakref.ref(t, cb) for lst in fixed_lists for t in lst]
    if len(_TABLES) >= _MAX_TABLES:
        _TABLES.clear()
    _TABLES[key] = e
    return e


def table(tensor_lists: list[list[torch.Tensor]], fixed: int = 0):
    """Return the kernel's int64 tensor table on the lists' device.

    The last ``fixed`` lists must be the same tensor objects, with the same
    storage, on every call that passes them (FusedAdam's fp32 moments): their
    addresses are cached, keyed by object identity. The other lists' addresses are
    re-read on every call; with ``fixed == 0`` the table is keyed by the numels.

    Returns ``(meta, num_tensors, num_chunks, vec, stream)``, or None when a tensor
    is not fp32 or the lists are not on one CUDA device (the caller then falls
    back). Like TE's ``multi_tensor_apply``, a tensor is read as ``numel``
    contiguous elements from its data pointer.
    """
    first = tensor_lists[0]
    n = len(first)
    dev = first[0].device
    if dev.type != "cuda":
        return None
    for lst in tensor_lists:
        if len(lst) != n or lst[0].device != dev:
            return None
    lists = len(tensor_lists)
    fresh = lists - fixed
    for lst in tensor_lists[:fresh]:
        if not _f32(lst):
            return None
    stream = torch.cuda.current_stream(dev).cuda_stream
    if fixed:
        ids = tuple(map(id, itertools.chain.from_iterable(tensor_lists[fresh:])))
        key = (dev.index, stream, lists, ids)
    else:
        key = (dev.index, stream, lists, tuple(t.numel() for t in first))
    e = _TABLES.get(key)
    if e is not None and not all(
        lst[0].data_ptr() == pa and lst[-1].data_ptr() == pb
        for (pa, pb), lst in zip(e.probes, tensor_lists[fresh:])
    ):
        del _TABLES[key]
        e = None
    if e is None:
        e = _build(key, tensor_lists, fixed, dev)
        if e is None:
            return None
    # A fresh pinned block per call: torch's host allocator recycles it only after
    # its copy has run, so the host never waits on the stream.
    host = torch.empty(fresh * n, dtype=torch.int64, pin_memory=True)
    h = host.numpy()
    for d in range(fresh):
        h[d * n : (d + 1) * n] = [t.data_ptr() for t in tensor_lists[d]]
    vec = 4 if e.fixed_aligned and not np.any(h % 16) else 1
    e.meta[: fresh * n].copy_(host, non_blocking=True)
    return e.meta, n, e.num_chunks, vec, stream


# ---------------------------------------------------------------------------
# torch.library registration
#
# The ops take the tensor table, not the tensor lists: boxing thousands of tensors
# through the dispatcher costs more host time than the launch it replaces. The Adam
# op updates in place the p / m / v tensors whose addresses the table holds.
# ---------------------------------------------------------------------------


def _nctas(nc: int, dev: torch.device) -> int:
    return max(1, min(nc, _num_sms(dev.index) * CTAS_PER_SM))


def _adam_table_cuda(meta, ws, num_tensors, num_chunks, vec, lr, beta1, beta2, bc1, bc2, eps, wd):
    compile_adam(vec)(
        meta,
        ws,
        num_tensors,
        num_chunks,
        lr,
        beta1,
        beta2,
        bc1,
        bc2,
        eps,
        wd,
        _nctas(num_chunks, meta.device),
    )


def _l2norm_table_cuda(meta, ws, noop, num_tensors, num_chunks, vec):
    ret = torch.empty(1, dtype=torch.float32, device=meta.device)
    partial = torch.empty(num_chunks, dtype=torch.float32, device=meta.device)
    compile_l2norm(vec)(
        meta, ws, partial, ret, noop, num_tensors, num_chunks, _nctas(num_chunks, meta.device)
    )
    return ret


def _l2norm_table_meta(meta, ws, noop, num_tensors, num_chunks, vec):
    return meta.new_empty((1,), dtype=torch.float32)


_LIB = torch.library.Library("megatron_multi_tensor", "DEF")
_LIB.define(
    "adam_table(Tensor meta, Tensor(a!) ws, int num_tensors, int num_chunks, int vec, "
    "float lr, float beta1, float beta2, float bc1, float bc2, float eps, float weight_decay) -> ()"
)
_LIB.define(
    "l2norm_table(Tensor meta, Tensor(a!) ws, Tensor(b!) noop, int num_tensors, "
    "int num_chunks, int vec) -> Tensor"
)
_LIB.impl("adam_table", _adam_table_cuda, "CUDA")
_LIB.impl("l2norm_table", _l2norm_table_cuda, "CUDA")
_LIB.impl("l2norm_table", _l2norm_table_meta, "Meta")


def multi_tensor_adam(g, p, m, v, lr, beta1, beta2, eps, step, mode, bias_correction, weight_decay):
    """TE ``multi_tensor_adam`` over fp32 lists in one launch; False if not covered."""
    if mode != 1:
        return False
    tab = table([g, p, m, v], fixed=2)
    if tab is None:
        return False
    meta, nt, nc, vec, stream = tab
    b1 = float(np.float32(beta1))
    b2 = float(np.float32(beta2))
    bc1 = float(np.float32(1.0 - b1**step)) if bias_correction else 1.0
    bc2 = float(np.float32(1.0 - b2**step)) if bias_correction else 1.0
    torch.ops.megatron_multi_tensor.adam_table(
        meta, _workspace(meta.device, stream), nt, nc, vec, lr, b1, b2, bc1, bc2, eps, weight_decay
    )
    return True


def multi_tensor_l2norm(x, noop):
    """TE ``multi_tensor_l2norm`` (global norm) of fp32 tensors; None if not covered."""
    tab = table([x])
    if tab is None:
        return None
    meta, nt, nc, vec, stream = tab
    return torch.ops.megatron_multi_tensor.l2norm_table(
        meta, _workspace(meta.device, stream), noop, nt, nc, vec
    )
