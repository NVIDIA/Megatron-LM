# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fused CuTe-DSL forward+backward kernels for the CSA/HCA ``Compressor`` gated pooling.

This module provides an additive fast path for the gated-softmax pooling region of
:meth:`megatron.core.transformer.experimental_attention_variant.csa.Compressor._forward_thd`
(THD packed layout, non-pre-grouped path): the chain

    gather-index build -> gather -> ``+ APE`` -> overlap-window transform (``coff == 2``)
    -> fp32 softmax over the window -> gated weighted sum -> bf16 cast

is replaced by ONE forward kernel and ONE backward kernel written in the CuTe Python DSL
(``nvidia-cutlass-dsl``, JIT-compiled on first use). The eager PyTorch implementation
remains the semantic reference and the fallback for every unsupported configuration.
See https://github.com/NVIDIA/Megatron-LM/issues/5968 for measurements and numerics.

Semantics (ground truth = the eager region in ``Compressor._forward_thd``):
  For each segment ``s`` (``cu_seqlens[s]..cu_seqlens[s+1]``) and each output block ``b``
  of ``ratio`` tokens, the ``2 * ratio`` window (``coff == 2``) is

  - ``k in [0, ratio)``: previous block's token ``tok0 - ratio + k``, first-half projection
    column ``j``, APE row ``k`` -> invalid for the segment's first block (score ``-inf``,
    kv ``0``, exactly like ``_overlap_transform_thd``).
  - ``k in [ratio, 2 * ratio)``: own token ``tok0 + k - ratio``, second-half projection
    column ``d + j``, APE row ``k - ratio``.

  ``out[b, j] = sum_k kv_k * softmax_k(score_k + ape_k)`` (fp32, single final bf16
  rounding). ``coff == 1`` (no overlap): the window is the block's own ``ratio`` tokens,
  column ``j``, always valid. Per-segment tail tokens (``seqlen % ratio``) are dropped,
  as in the eager code.

Numerics:
  All arithmetic is fp32 with a single final bf16 rounding. The fp32 accumulation
  structure mirrors the eager ops (serial max, serial sum of exp, ``p = e / denom``,
  serial sum ``kv * p``), with ``mul.rn.f32``/``fma.rn.f32`` pinned in PTX so results do
  not depend on compiler FMA contraction. Note the eager path itself rounds the softmax
  weights to bf16 and multiplies in bf16, so the fused output is NOT bit-identical to
  eager; against an fp32-intermediate eager reference, ``dKV``/``dScore`` are bit-identical
  and the forward matches to within one bf16 rounding step (see the issue for data).

Backward:
  Atomic-free for ``dKV``/``dScore``: every consumed input element belongs to exactly one
  pooling window (for ``coff == 2``, first-half columns are consumed by the NEXT block's
  window and second-half columns by the OWN block's window), so gradient stores are
  disjoint. Gradients of elements never consumed (segment-tail tokens; for ``coff == 2``
  the first-half projection columns of each segment's last block) are exact zeros from
  the zero-initialized gradient buffers, matching autograd. ``dAPE`` is accumulated in
  registers over ``rows_per_cta`` blocks and then reduced with one fp32 atomic per
  ``(k, dim)`` per CTA; ``dAPE`` is therefore not
  bitwise run-to-run deterministic (forward, ``dKV`` and ``dScore`` are).

Static-capacity padding (``fixed_total_comp``):
  When ``total_comp`` exceeds the true compressed row count (``cu_seqlens_comp[-1]``),
  forward computes the padding rows exactly like the eager code: they gather the window
  from token 0 with first-in-segment semantics (requires ``total_tokens >= ratio``, like
  the eager gather). Backward ignores incoming gradients on padding rows (they are tail
  padding for CUDA-graph static shapes and are not consumed downstream).

CUDA graphs:
  The launch path is capture-compatible once the kernel for a given
  ``(ratio, head_dim, coff)`` configuration has been JIT-compiled; run one eager step per
  configuration before capture. A first call that would JIT under capture raises a
  ``RuntimeError`` instead of corrupting the capture.

Constraints / gating:
  - Requires ``nvidia-cutlass-dsl`` (import-guarded; ``_CUTE_AVAILABLE`` is False without
    it) and compute capability 10.0 (the only validated architecture so far).
  - bf16 ``kv``/``score``, fp32 ``ape``; int32 flat offsets, i.e.
    ``total_tokens * coff * head_dim < 2**31``.
  - The dispatch keeps the eager path under ``torch.use_deterministic_algorithms(True)``
    (``dAPE`` atomics) and under ``torch.compile`` tracing (raw-pointer launch path).
  - ``MCORE_CSA_FUSED_COMPRESSOR=0`` disables the dispatch (eager everywhere);
    ``MCORE_CSA_FUSED_COMPRESSOR_FAST_LAUNCH=0`` disables only the cached-launch
    optimization (see ``_FastLauncher``).
"""

import ctypes
import os
import threading
from typing import Optional

import torch

try:
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.base_dsl.typing as _cutlass_typing
    import cutlass.cute as cute
    import cutlass.cute.arch as cute_arch
    import cutlass.cute.math as cute_math
    from cutlass._mlir.dialects import llvm as _llvm
    from cutlass.cute.runtime import make_ptr

    _CUTE_AVAILABLE = True
    _CUTE_IMPORT_ERROR: Optional[Exception] = None
# A partially installed or ABI-incompatible DSL can raise more than ImportError;
# importing this module (and transitively ``csa.py``) must never fail because of it.
except Exception as _e:  # pragma: no cover - import guard  # pylint: disable=broad-except
    cuda_driver = None
    cutlass = None
    _cutlass_typing = None
    cute = None
    cute_arch = None
    cute_math = None
    _llvm = None
    make_ptr = None
    _CUTE_AVAILABLE = False
    _CUTE_IMPORT_ERROR = _e

_ENV_ENABLE = "MCORE_CSA_FUSED_COMPRESSOR"
_ENV_FAST_LAUNCH = "MCORE_CSA_FUSED_COMPRESSOR_FAST_LAUNCH"

# The only compute capability the kernels have been validated on so far. The kernels use
# no architecture-specific features (plain loads/stores, fp32 atomics, pinned mul/fma
# PTX), but wider coverage stays opt-in until validated per architecture.
_SUPPORTED_COMPUTE_CAPABILITY = (10, 0)

_DEVICE_SUPPORTED_CACHE = {}


def _dispatch_enabled() -> bool:
    """Return True unless the fused compressor is disabled via environment variable."""
    return os.environ.get(_ENV_ENABLE, "1").lower() not in ("0", "false", "off")


def fused_compressor_available(device: Optional[torch.device] = None) -> bool:
    """Return True when the fused kernels can run: DSL importable + supported device."""
    if not _CUTE_AVAILABLE:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        if device is not None and device.type != "cuda":
            return False
        if device is None or device.index is None:
            index = torch.cuda.current_device()
        else:
            index = device.index
    except (RuntimeError, AssertionError):  # pragma: no cover - no CUDA context
        return False
    supported = _DEVICE_SUPPORTED_CACHE.get(index)
    if supported is None:
        supported = torch.cuda.get_device_capability(index) == _SUPPORTED_COMPUTE_CAPABILITY
        _DEVICE_SUPPORTED_CACHE[index] = supported
    return supported


# =============================================================================
# Cached fast-path launcher
# =============================================================================
# Each steady-state CuTe-DSL wrapper call spends tens of microseconds of pure host Python
# (rebuilding pointer/scalar/stream argument objects, adapter lookups, fresh ctypes
# allocations, re-packing the void** array) to end at a ~3-4 us C launch call. For
# microsecond-scale kernels that overhead dominates the wall clock, so the launch state is
# snapshotted ONCE per (kernel, config, device, thread) and replayed with in-place
# mutation of the argument storages.
# =============================================================================

# torch's raw current-stream query (~0.5 us) vs `torch.cuda.current_stream` object
# construction (~2-3 us). Same handle the slow path ends up passing. Private API: guard
# the bind so module import survives torch builds that do not expose it.
_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)
if _raw_stream is None:  # pragma: no cover - older/stripped torch builds

    def _raw_stream(device_index=None):
        """Fallback raw-stream query via the public torch API."""
        return torch.cuda.current_stream(device_index).cuda_stream


def _fast_launch_enabled() -> bool:
    """Return True unless the cached-launch optimization is disabled via environment."""
    return _CUTE_AVAILABLE and os.environ.get(_ENV_FAST_LAUNCH, "1") == "1"


def _view_for_arg(arg, addr):
    """Build a ctypes view over the storage backing one execution-args slot."""
    if _CUTE_AVAILABLE and isinstance(arg, _cutlass_typing.Numeric):
        if isinstance(arg, _cutlass_typing.Boolean):
            return ctypes.c_bool.from_address(addr)
        if isinstance(arg, _cutlass_typing.Integer):
            width = type(arg).width
            signed = getattr(type(arg), "signed", True)
            ctype = getattr(ctypes, f"c_{'int' if signed else 'uint'}{width}")
            return ctype.from_address(addr)
        if isinstance(arg, _cutlass_typing.Float32):
            return ctypes.c_float.from_address(addr)
        if isinstance(arg, _cutlass_typing.Float64):
            return ctypes.c_double.from_address(addr)
        raise TypeError(f"unsupported numeric scalar {type(arg)!r}")
    # A cute runtime Pointer (make_ptr) stores its address in a c_void_p `_desc`;
    # CUstream's storage is its own pointer-sized handle.
    if hasattr(arg, "_desc") and isinstance(arg._desc, ctypes.c_void_p):
        return ctypes.c_void_p.from_address(addr)
    if type(arg).__name__ == "CUstream":
        return ctypes.c_void_p.from_address(addr)
    raise TypeError(f"unsupported argument type {type(arg)!r}")


class _FastLauncher:
    """Replayable launch state for one compiled CuTe-DSL function.

    ``slots[i]`` is a ctypes view over the storage feeding runtime argument ``i`` (same
    order as the tuple passed to ``fn(*args)``); write ``.value`` then call ``launch()``.

    Guards:
      - Only flat argument tuples of cute runtime pointers (``make_ptr``), cutlass
        scalars, and ``CUstream`` are eligible; anything else raises during construction
        and the wrapper stays on its slow path.
      - Construction introspects private-but-stable DSL internals
        (``_default_executor``, ``_get_invoke_packed_args``); any structural mismatch on
        a future ``nvidia-cutlass-dsl`` upgrade raises during construction, and the
        wrapper permanently falls back to the regular (slow) launch path rather than
        attempting a launch from a partially built snapshot.
    """

    __slots__ = ("slots", "_capi", "_packed", "_res", "_has_res", "_keep")

    def __init__(self, fn, args):
        # Must run after the wrapper's first real `fn(*args)` call so the default
        # executor (device context, loaded modules) exists.
        exe_args, adapted = fn.execution_args.generate_execution_args(args, {})
        executor = fn._default_executor
        if executor is None:
            raise RuntimeError("build _FastLauncher after the first fn() call")
        if len(exe_args) != len(args):
            # struct/dlpack args expand to multiple slots -> unsupported.
            raise TypeError(f"non-flat exe_args ({len(exe_args)} slots for {len(args)} args)")
        # Private copy of the packed void** array: the executor's own is a shared
        # thread-local scratch buffer that any interleaved slow-path call would repoint
        # to its (dead) per-call storages.
        tls_packed = executor._get_invoke_packed_args(exe_args)
        total = len(exe_args) + executor._num_extra_args
        packed = (ctypes.c_void_p * total)()
        for i in range(total):
            packed[i] = tls_packed[i]
        views = []
        for arg, exe_arg in zip(args, exe_args):
            addr = exe_arg.value if isinstance(exe_arg, ctypes.c_void_p) else int(exe_arg)
            views.append(_view_for_arg(arg, addr))
        self.slots = views
        self._capi = executor.capi_func
        self._res = executor.cuda_result
        self._has_res = executor._has_cuda_result
        self._packed = packed
        # Keep every object owning a storage referenced by `packed` alive.
        self._keep = (args, exe_args, adapted, fn, executor)

    def launch(self):
        """Replay the snapshotted launch with the current slot values."""
        self._capi(self._packed)
        if self._has_res:
            result = self._res.value
            if result != 0:
                raise RuntimeError(
                    f"CUDA error {result} in CuTe-DSL fast launch (set "
                    f"{_ENV_FAST_LAUNCH}=0 to fall back to the slow launch path)"
                )


class _FastCache:
    """Thread-local ``{key: _FastLauncher}`` with build-once semantics.

    ``get`` returns a launcher or None (not built / build failed / disabled). ``put``
    attempts to build; a failed build is remembered so the wrapper pays the (cheap)
    attempt exactly once per thread and stays on its slow path afterwards. The cache is
    thread-local because forward wrappers run on the main thread while backward wrappers
    run on the autograd thread, and slot mutation is not thread-safe.
    """

    def __init__(self):
        self._tls = threading.local()

    def _map(self):
        """Return this thread's key -> launcher map."""
        cache_map = getattr(self._tls, "m", None)
        if cache_map is None:
            cache_map = {}
            self._tls.m = cache_map
        return cache_map

    def get(self, key):
        """Return the cached launcher for ``key`` or None."""
        launcher = self._map().get(key)
        return launcher if launcher is not None and launcher is not False else None

    def put(self, key, fn, args):
        """Try to build and cache a launcher for ``key``; never fails the call."""
        if not _fast_launch_enabled():
            return
        cache_map = self._map()
        if key in cache_map:
            return
        try:
            cache_map[key] = _FastLauncher(fn, args)
        except Exception:  # pylint: disable=broad-except
            # Structural mismatch (DSL upgrade, exotic arg): remember and keep the
            # wrapper on its slow path. Never fail the call.
            cache_map[key] = False


_FAST = _FastCache()


# =============================================================================
# CuTe-DSL kernel definitions
# =============================================================================

if _CUTE_AVAILABLE:

    _NEG_INF = float("-inf")

    @cutlass.dsl_user_op
    def _fmul_rn(a, b, *, loc=None, ip=None):
        """fp32 multiply pinned to ``mul.rn.f32`` (opaque to FMA contraction).

        The eager ``(kv * weights).sum(dim=1)`` and the softmax-backward inner sum both
        accumulate ROUNDED products serially; letting the compiler contract mul+add into
        FMA breaks bit-exactness against the fp32 eager reference.
        """
        return cutlass.Float32(
            _llvm.inline_asm(
                cutlass.Float32.mlir_type,
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                ],
                "mul.rn.f32 $0, $1, $2;",
                "=f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @cutlass.dsl_user_op
    def _ffma_rn(a, b, c, *, loc=None, ip=None):
        """fp32 fma pinned to ``fma.rn.f32``.

        The eager softmax-backward epilogue is ``ds = fma(p, -S, round(p * dp))``; pinning
        removes any dependence on compiler contraction choices.
        """
        return cutlass.Float32(
            _llvm.inline_asm(
                cutlass.Float32.mlir_type,
                [
                    cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(b).ir_value(loc=loc, ip=ip),
                    cutlass.Float32(c).ir_value(loc=loc, ip=ip),
                ],
                "fma.rn.f32 $0, $1, $2, $3;",
                "=f,f,f,f",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=_llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @cute.kernel
    def _compressor_fwd_kernel(
        mKV: cute.Tensor,  # flat [T * W] bf16, W = coff * d
        mScore: cute.Tensor,  # flat [T * W] bf16
        mAPE: cute.Tensor,  # flat [ratio * W] fp32
        mCu: cute.Tensor,  # [n_seq + 1] int32 (token cu_seqlens)
        mCuComp: cute.Tensor,  # [n_seq + 1] int32 (block cu_seqlens)
        mOut: cute.Tensor,  # flat [nb_total * d] bf16
        nb_total: cutlass.Int32,
        n_seq: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d: cutlass.Constexpr,
        coff: cutlass.Constexpr,
        rows_per_cta: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        """Forward: one thread per (output block, head dim), ``rows_per_cta`` blocks/CTA."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        dim = bidy * threads + tidx
        W: cutlass.Constexpr = coff * d
        win: cutlass.Constexpr = 2 * ratio if coff == 2 else ratio

        if dim < d:
            # Hoist APE loads: constant per (k, dim) across all rows.
            ape_k = []
            for k in cutlass.range_constexpr(win):
                if cutlass.const_expr(coff == 2 and k < ratio):
                    col = dim
                else:
                    col = (d + dim) if cutlass.const_expr(coff == 2) else dim
                ape_k.append(mAPE[(k % ratio) * W + col])

            # True compressed row count; rows in [nb_valid, nb_total) are static-capacity
            # padding (fixed_total_comp) and gather the window from token 0 with
            # first-in-segment semantics, like the eager code.
            nb_valid = mCuComp[n_seq]

            for rr in cutlass.range_constexpr(rows_per_cta):
                bb = bidx * rows_per_cta + rr
                if bb < nb_total:
                    # Per-segment boundary scan (n_seq is small).
                    seq_idx = cutlass.Int32(0)
                    bis = cutlass.Int32(0)
                    if bb < nb_valid:
                        bis = cutlass.Int32(bb)
                        for s in cutlass.range(n_seq):
                            cs = mCuComp[s]
                            ce = mCuComp[s + 1]
                            if bb >= cs:
                                if bb < ce:
                                    seq_idx = s
                                    bis = bb - cs
                    tok0 = mCu[seq_idx] + bis * ratio

                    sv = []
                    kvv = []
                    for k in cutlass.range_constexpr(win):
                        if cutlass.const_expr(coff == 2 and k < ratio):
                            off = (tok0 - ratio + k) * W + dim
                            v = cutlass.Float32(_NEG_INF)
                            u = cutlass.Float32(0.0)
                            if bis > 0:
                                v = cutlass.Float32(mScore[off]) + ape_k[k]
                                u = cutlass.Float32(mKV[off])
                        else:
                            if cutlass.const_expr(coff == 2):
                                off = (tok0 + k - ratio) * W + d + dim
                            else:
                                off = (tok0 + k) * W + dim
                            v = cutlass.Float32(mScore[off]) + ape_k[k]
                            u = cutlass.Float32(mKV[off])
                        sv.append(v)
                        kvv.append(u)

                    mx = sv[0]
                    for k in cutlass.range_constexpr(1, win):
                        if sv[k] > mx:
                            mx = sv[k]
                    den = cutlass.Float32(0.0)
                    ex = []
                    for k in cutlass.range_constexpr(win):
                        e = cute_math.exp(sv[k] - mx)
                        den = den + e
                        ex.append(e)
                    acc = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(win):
                        acc = acc + _fmul_rn(kvv[k], ex[k] / den)
                    mOut[bb * d + dim] = cutlass.BFloat16(acc)

    @cute.kernel
    def _compressor_bwd_kernel(
        mKV: cute.Tensor,  # flat [T * W] bf16
        mScore: cute.Tensor,  # flat [T * W] bf16
        mAPE: cute.Tensor,  # flat [ratio * W] fp32
        mCu: cute.Tensor,  # [n_seq + 1] int32
        mCuComp: cute.Tensor,  # [n_seq + 1] int32
        mGO: cute.Tensor,  # flat [nb_total * d] bf16
        mGKV: cute.Tensor,  # flat [T * W] bf16 (zero-initialized)
        mGS: cute.Tensor,  # flat [T * W] bf16 (zero-initialized)
        mGAPE: cute.Tensor,  # flat [ratio * W] fp32 (zero-initialized)
        nb_total: cutlass.Int32,
        n_seq: cutlass.Int32,
        ratio: cutlass.Constexpr,
        d: cutlass.Constexpr,
        coff: cutlass.Constexpr,
        rows_per_cta: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        """Backward: recompute window probs, disjoint ``dKV``/``dScore`` stores, ``dAPE`` atomics.

        Rows in ``[cu_seqlens_comp[-1], nb_total)`` are static-capacity padding
        (``fixed_total_comp``); their incoming gradients are ignored.
        """
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        dim = bidy * threads + tidx
        W: cutlass.Constexpr = coff * d
        win: cutlass.Constexpr = 2 * ratio if coff == 2 else ratio

        if dim < d:
            ape_k = []
            dape = []
            for k in cutlass.range_constexpr(win):
                if cutlass.const_expr(coff == 2 and k < ratio):
                    col = dim
                else:
                    col = (d + dim) if cutlass.const_expr(coff == 2) else dim
                ape_k.append(mAPE[(k % ratio) * W + col])
                dape.append(cutlass.Float32(0.0))

            nb_valid = mCuComp[n_seq]

            for rr in cutlass.range_constexpr(rows_per_cta):
                bb = bidx * rows_per_cta + rr
                if bb < nb_valid:
                    seq_idx = cutlass.Int32(0)
                    bis = cutlass.Int32(bb)
                    for s in cutlass.range(n_seq):
                        cs = mCuComp[s]
                        ce = mCuComp[s + 1]
                        if bb >= cs:
                            if bb < ce:
                                seq_idx = s
                                bis = bb - cs
                    tok0 = mCu[seq_idx] + bis * ratio

                    # Recompute window probs (same order as forward).
                    sv = []
                    kvv = []
                    offs = []
                    for k in cutlass.range_constexpr(win):
                        if cutlass.const_expr(coff == 2 and k < ratio):
                            off = (tok0 - ratio + k) * W + dim
                            v = cutlass.Float32(_NEG_INF)
                            u = cutlass.Float32(0.0)
                            if bis > 0:
                                v = cutlass.Float32(mScore[off]) + ape_k[k]
                                u = cutlass.Float32(mKV[off])
                        else:
                            if cutlass.const_expr(coff == 2):
                                off = (tok0 + k - ratio) * W + d + dim
                            else:
                                off = (tok0 + k) * W + dim
                            v = cutlass.Float32(mScore[off]) + ape_k[k]
                            u = cutlass.Float32(mKV[off])
                        sv.append(v)
                        kvv.append(u)
                        offs.append(off)

                    mx = sv[0]
                    for k in cutlass.range_constexpr(1, win):
                        if sv[k] > mx:
                            mx = sv[k]
                    den = cutlass.Float32(0.0)
                    ex = []
                    for k in cutlass.range_constexpr(win):
                        e = cute_math.exp(sv[k] - mx)
                        den = den + e
                        ex.append(e)

                    go = cutlass.Float32(mGO[bb * d + dim])

                    # Same expression tree as torch's softmax_backward_data:
                    # dp_k = go * kv_k ; S = serial sum of ROUNDED dp_k * p_k ;
                    # ds_k = fma(p_k, -S, round(dp_k * p_k)) ; dkv_k = go * p_k.
                    p = []
                    dp = []
                    S = cutlass.Float32(0.0)
                    for k in cutlass.range_constexpr(win):
                        pk = ex[k] / den
                        dpk = go * kvv[k]
                        S = S + _fmul_rn(dpk, pk)
                        p.append(pk)
                        dp.append(dpk)

                    for k in cutlass.range_constexpr(win):
                        if cutlass.const_expr(coff == 2 and k < ratio):
                            if bis > 0:
                                ds = _ffma_rn(p[k], -S, _fmul_rn(dp[k], p[k]))
                                mGKV[offs[k]] = cutlass.BFloat16(go * p[k])
                                mGS[offs[k]] = cutlass.BFloat16(ds)
                                dape[k] = dape[k] + ds
                        else:
                            ds = _ffma_rn(p[k], -S, _fmul_rn(dp[k], p[k]))
                            mGKV[offs[k]] = cutlass.BFloat16(go * p[k])
                            mGS[offs[k]] = cutlass.BFloat16(ds)
                            dape[k] = dape[k] + ds

            # One fp32 atomic per (k, dim) per CTA (amortized over rows_per_cta rows).
            for k in cutlass.range_constexpr(win):
                if cutlass.const_expr(coff == 2 and k < ratio):
                    col = dim
                else:
                    col = (d + dim) if cutlass.const_expr(coff == 2) else dim
                cute_arch.atomic_add(mGAPE.iterator + ((k % ratio) * W + col), dape[k])

    _EXT = (1 << 31) - 1  # flat extent placeholder (int32 offsets, no bounds checks)

    @cute.jit
    def _compressor_fwd_launch(
        kv_ptr: cute.Pointer,
        score_ptr: cute.Pointer,
        ape_ptr: cute.Pointer,
        cu_ptr: cute.Pointer,
        cuc_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        nb_total: cutlass.Int32,
        n_seq: cutlass.Int32,
        stream: cuda_driver.CUstream,
        ratio: cutlass.Constexpr,
        d: cutlass.Constexpr,
        coff: cutlass.Constexpr,
        rows_per_cta: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        """JIT entry point that wraps raw pointers into tensors and launches forward."""
        lay = cute.make_layout(_EXT)
        mKV = cute.make_tensor(kv_ptr, lay)
        mScore = cute.make_tensor(score_ptr, lay)
        mAPE = cute.make_tensor(ape_ptr, lay)
        mCu = cute.make_tensor(cu_ptr, lay)
        mCuComp = cute.make_tensor(cuc_ptr, lay)
        mOut = cute.make_tensor(out_ptr, lay)
        gx = (nb_total + rows_per_cta - 1) // rows_per_cta
        gy = (d + threads - 1) // threads
        _compressor_fwd_kernel(
            mKV,
            mScore,
            mAPE,
            mCu,
            mCuComp,
            mOut,
            nb_total,
            n_seq,
            ratio,
            d,
            coff,
            rows_per_cta,
            threads,
        ).launch(grid=(gx, gy, 1), block=(threads, 1, 1), stream=stream)

    @cute.jit
    def _compressor_bwd_launch(
        kv_ptr: cute.Pointer,
        score_ptr: cute.Pointer,
        ape_ptr: cute.Pointer,
        cu_ptr: cute.Pointer,
        cuc_ptr: cute.Pointer,
        go_ptr: cute.Pointer,
        gkv_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        gape_ptr: cute.Pointer,
        nb_total: cutlass.Int32,
        n_seq: cutlass.Int32,
        stream: cuda_driver.CUstream,
        ratio: cutlass.Constexpr,
        d: cutlass.Constexpr,
        coff: cutlass.Constexpr,
        rows_per_cta: cutlass.Constexpr,
        threads: cutlass.Constexpr,
    ):
        """JIT entry point that wraps raw pointers into tensors and launches backward."""
        lay = cute.make_layout(_EXT)
        mKV = cute.make_tensor(kv_ptr, lay)
        mScore = cute.make_tensor(score_ptr, lay)
        mAPE = cute.make_tensor(ape_ptr, lay)
        mCu = cute.make_tensor(cu_ptr, lay)
        mCuComp = cute.make_tensor(cuc_ptr, lay)
        mGO = cute.make_tensor(go_ptr, lay)
        mGKV = cute.make_tensor(gkv_ptr, lay)
        mGS = cute.make_tensor(gs_ptr, lay)
        mGAPE = cute.make_tensor(gape_ptr, lay)
        gx = (nb_total + rows_per_cta - 1) // rows_per_cta
        gy = (d + threads - 1) // threads
        _compressor_bwd_kernel(
            mKV,
            mScore,
            mAPE,
            mCu,
            mCuComp,
            mGO,
            mGKV,
            mGS,
            mGAPE,
            nb_total,
            n_seq,
            ratio,
            d,
            coff,
            rows_per_cta,
            threads,
        ).launch(grid=(gx, gy, 1), block=(threads, 1, 1), stream=stream)

    _COMPILED = {}
    _FWD_ROWS, _FWD_THREADS = 4, 128
    _BWD_ROWS, _BWD_THREADS = 8, 128

    def _bf16_ptr(t):
        """Wrap a bf16 tensor's data pointer for the DSL."""
        return make_ptr(cutlass.BFloat16, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    def _f32_ptr(t):
        """Wrap an fp32 tensor's data pointer for the DSL."""
        return make_ptr(cutlass.Float32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    def _i32_ptr(t):
        """Wrap an int32 tensor's data pointer for the DSL."""
        return make_ptr(cutlass.Int32, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    def _run_fwd(kv, score, ape, cu_i, cuc_i, out, nb_total, ratio, d, coff):
        """Launch the forward kernel (cached fast path -> compiled slow path -> JIT)."""
        dev = kv.device.index
        key = ("fwd", ratio, d, coff, dev)
        launcher = _FAST.get(key)
        if launcher is not None:
            # Cached launch: mutate the snapshotted argument storages in place; this is
            # the same launch the slow path below performs.
            slots = launcher.slots
            slots[0].value = kv.data_ptr()
            slots[1].value = score.data_ptr()
            slots[2].value = ape.data_ptr()
            slots[3].value = cu_i.data_ptr()
            slots[4].value = cuc_i.data_ptr()
            slots[5].value = out.data_ptr()
            slots[6].value = nb_total
            slots[7].value = cu_i.numel() - 1
            slots[8].value = _raw_stream(dev)
            launcher.launch()
            return
        stream = cuda_driver.CUstream(torch.cuda.current_stream(kv.device).cuda_stream)
        args = (
            _bf16_ptr(kv),
            _bf16_ptr(score),
            _f32_ptr(ape),
            _i32_ptr(cu_i),
            _i32_ptr(cuc_i),
            _bf16_ptr(out),
            cutlass.Int32(nb_total),
            cutlass.Int32(cu_i.numel() - 1),
            stream,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused CSA compressor: first call for config {key} happened under "
                    "CUDA graph capture (JIT compilation is not capture-safe); run one "
                    "eager warmup step for this configuration before capturing."
                )
            fn = cute.compile(
                _compressor_fwd_launch, *args, ratio, d, coff, _FWD_ROWS, _FWD_THREADS
            )
            _COMPILED[key] = fn
        fn(*args)
        _FAST.put(key, fn, args)

    def _run_bwd(kv, score, ape, cu_i, cuc_i, go, gkv, gs, gape, nb_total, ratio, d, coff):
        """Launch the backward kernel (cached fast path -> compiled slow path -> JIT)."""
        dev = kv.device.index
        key = ("bwd", ratio, d, coff, dev)
        launcher = _FAST.get(key)
        if launcher is not None:
            slots = launcher.slots
            slots[0].value = kv.data_ptr()
            slots[1].value = score.data_ptr()
            slots[2].value = ape.data_ptr()
            slots[3].value = cu_i.data_ptr()
            slots[4].value = cuc_i.data_ptr()
            slots[5].value = go.data_ptr()
            slots[6].value = gkv.data_ptr()
            slots[7].value = gs.data_ptr()
            slots[8].value = gape.data_ptr()
            slots[9].value = nb_total
            slots[10].value = cu_i.numel() - 1
            slots[11].value = _raw_stream(dev)
            launcher.launch()
            return
        stream = cuda_driver.CUstream(torch.cuda.current_stream(kv.device).cuda_stream)
        args = (
            _bf16_ptr(kv),
            _bf16_ptr(score),
            _f32_ptr(ape),
            _i32_ptr(cu_i),
            _i32_ptr(cuc_i),
            _bf16_ptr(go),
            _bf16_ptr(gkv),
            _bf16_ptr(gs),
            _f32_ptr(gape),
            cutlass.Int32(nb_total),
            cutlass.Int32(cu_i.numel() - 1),
            stream,
        )
        fn = _COMPILED.get(key)
        if fn is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"fused CSA compressor: first call for config {key} happened under "
                    "CUDA graph capture (JIT compilation is not capture-safe); run one "
                    "eager warmup step for this configuration before capturing."
                )
            fn = cute.compile(
                _compressor_bwd_launch, *args, ratio, d, coff, _BWD_ROWS, _BWD_THREADS
            )
            _COMPILED[key] = fn
        fn(*args)
        _FAST.put(key, fn, args)


class _CompressThdFused(torch.autograd.Function):
    """Autograd wrapper around the fused forward/backward kernels.

    Incoming gradients on static-capacity padding rows (``total_comp`` beyond
    ``cu_seqlens_comp[-1]``, see ``fixed_total_comp``) are ignored by the backward
    kernel; padding rows are tail padding and are not consumed downstream.
    """

    @staticmethod
    def forward(
        ctx, kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, head_dim, coff, total_comp
    ):
        """Run the fused forward kernel; saves inputs for the fused backward."""
        kv = kv.contiguous()
        score = score.contiguous()
        ape_c = ape.contiguous()
        cu_i = cu_seqlens.to(dtype=torch.int32).contiguous()
        cuc_i = cu_seqlens_comp.to(dtype=torch.int32).contiguous()
        if total_comp is None:
            total_comp = int(cuc_i[-1].item())
        out = torch.empty((int(total_comp), head_dim), device=kv.device, dtype=kv.dtype)
        if out.numel() > 0:
            _run_fwd(kv, score, ape_c, cu_i, cuc_i, out, int(total_comp), ratio, head_dim, coff)
        ctx.save_for_backward(kv, score, ape_c, cu_i, cuc_i)
        ctx.dims = (ratio, head_dim, coff)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Run the fused backward kernel into zero-initialized gradient buffers."""
        if (
            torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
        ):
            raise RuntimeError(
                "fused CSA compressor backward accumulates dAPE with fp32 atomics and is "
                "not deterministic; torch.use_deterministic_algorithms(True) is set. Use "
                "the eager path instead (the Compressor dispatch already falls back to "
                f"eager under deterministic mode; see {_ENV_ENABLE})."
            )
        kv, score, ape, cu_i, cuc_i = ctx.saved_tensors
        ratio, head_dim, coff = ctx.dims
        grad_kv = torch.zeros_like(kv)
        grad_score = torch.zeros_like(score)
        grad_ape = torch.zeros_like(ape, dtype=torch.float32)
        go = grad_out.contiguous()
        if go.numel() > 0:
            _run_bwd(
                kv,
                score,
                ape,
                cu_i,
                cuc_i,
                go,
                grad_kv,
                grad_score,
                grad_ape,
                go.shape[0],
                ratio,
                head_dim,
                coff,
            )
        return grad_kv, grad_score, grad_ape, None, None, None, None, None, None


def compress_thd_fused(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    ratio: int,
    head_dim: int,
    coff: int,
    total_comp: Optional[int] = None,
) -> torch.Tensor:
    """Fused gated-pooling region of ``Compressor._forward_thd`` (explicit op API).

    Args:
        kv: ``(total_tokens, coff * head_dim)`` bf16 gate values (``linear_wkv`` output).
        score: ``(total_tokens, coff * head_dim)`` bf16 gate scores (``linear_wgate``
            output).
        ape: ``(ratio, coff * head_dim)`` fp32 additive position embedding.
        cu_seqlens: ``(B + 1,)`` int32 cumulative token counts per segment.
        cu_seqlens_comp: ``(B + 1,)`` int32 cumulative compressed-block counts,
            ``cu_seqlens_comp[b + 1] - cu_seqlens_comp[b] == seqlen_b // ratio``.
        ratio: compression ratio (tokens per output block).
        head_dim: output feature dimension ``d``.
        coff: 2 for the overlapping window (``ratio == 4``), 1 for non-overlapping.
        total_comp: output row count. Defaults to ``cu_seqlens_comp[-1]`` (synchronizes);
            pass it explicitly (e.g. a ``fixed_total_comp`` static capacity, which must be
            ``>= cu_seqlens_comp[-1]``) to stay CUDA-graph-safe.

    Returns:
        ``(total_comp, head_dim)`` bf16 pooled output (pre-RMSNorm).
    """
    if not _CUTE_AVAILABLE:
        raise RuntimeError(f"fused CSA compressor unavailable: {_CUTE_IMPORT_ERROR!r}")
    if not fused_compressor_available(kv.device):
        raise RuntimeError(
            "fused CSA compressor requires a CUDA device with compute capability "
            f"{_SUPPORTED_COMPUTE_CAPABILITY}"
        )
    if kv.dtype != torch.bfloat16 or score.dtype != torch.bfloat16:
        raise TypeError("compress_thd_fused expects bf16 kv/score")
    if coff not in (1, 2):
        raise ValueError(f"coff must be 1 or 2, got {coff}")
    if kv.dim() != 2 or kv.shape[1] != coff * head_dim:
        raise ValueError(f"kv must be (total_tokens, coff * head_dim), got {tuple(kv.shape)}")
    if score.shape != kv.shape:
        raise ValueError(f"score shape {tuple(score.shape)} != kv shape {tuple(kv.shape)}")
    if ape.shape != (ratio, coff * head_dim):
        raise ValueError(f"ape must be (ratio, coff * head_dim), got {tuple(ape.shape)}")
    if cu_seqlens.dim() != 1 or cu_seqlens_comp.dim() != 1:
        raise ValueError("cu_seqlens and cu_seqlens_comp must be 1-D")
    if cu_seqlens.numel() != cu_seqlens_comp.numel() or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens and cu_seqlens_comp must both have B + 1 entries")
    for name, t in (
        ("score", score),
        ("ape", ape),
        ("cu_seqlens", cu_seqlens),
        ("cu_seqlens_comp", cu_seqlens_comp),
    ):
        if t.device != kv.device:
            raise ValueError(f"{name} is on {t.device}, expected {kv.device}")
    if total_comp is not None and total_comp > 0 and kv.shape[0] < ratio:
        # The eager gather would also fail here (padding rows read tokens [0, ratio)).
        raise ValueError(
            f"total_comp={total_comp} > 0 requires at least ratio={ratio} tokens, "
            f"got {kv.shape[0]}"
        )
    if kv.shape[0] * coff * head_dim >= 2**31:
        raise ValueError("compress_thd_fused requires total_tokens * coff * head_dim < 2**31")
    if ape.dtype != torch.float32:
        ape = ape.float()
    return _CompressThdFused.apply(
        kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, head_dim, coff, total_comp
    )


def maybe_compress_thd_fused(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_comp: torch.Tensor,
    total_comp: int,
    ratio: int,
    head_dim: int,
    coff: int,
) -> Optional[torch.Tensor]:
    """Dispatch helper for ``Compressor._forward_thd``: fused result or None (use eager).

    Returns the pooled ``(total_comp, 1, head_dim)`` bf16 tensor when the fused fast path
    supports the configuration, or None when the caller should keep the eager
    implementation. Initial dispatch gating (see issue #5968): THD non-pre-grouped path,
    ``compress_ratio == 4`` / ``coff == 2`` only (``compress_ratio == 128`` is functionally
    supported by :func:`compress_thd_fused` but not yet a wall-clock win, so it stays on
    eager), bf16 inputs, compute capability 10.0, int32 flat offsets.
    """
    if not _dispatch_enabled():
        return None
    if not _CUTE_AVAILABLE or kv.device.type != "cuda":
        return None
    if not fused_compressor_available(kv.device):
        return None
    if ratio != 4 or coff != 2:
        return None
    if kv.dtype != torch.bfloat16 or score.dtype != torch.bfloat16:
        return None
    if ape.dtype != torch.float32:
        return None
    # The backward is not deterministic for dAPE (fp32 atomics); respect torch's
    # deterministic mode by keeping the (deterministic) eager path.
    if torch.are_deterministic_algorithms_enabled():
        return None
    # The launch path uses raw pointers and a private stream query; keep eager (which
    # torch.compile can trace and fuse itself) when compiling.
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is not None and is_compiling():
        return None
    total = kv.shape[0]
    if kv.dim() != 3 or kv.shape[1] != 1 or kv.shape[2] != coff * head_dim:
        return None
    if score.shape != kv.shape:
        return None
    if total_comp <= 0 or total < ratio:
        return None
    if total * coff * head_dim >= 2**31:
        return None
    out = _CompressThdFused.apply(
        kv.reshape(total, coff * head_dim),
        score.reshape(total, coff * head_dim),
        ape,
        cu_seqlens,
        cu_seqlens_comp,
        ratio,
        head_dim,
        coff,
        total_comp,
    )
    return out.unsqueeze(1)
