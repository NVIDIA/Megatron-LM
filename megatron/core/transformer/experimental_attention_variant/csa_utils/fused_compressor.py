# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dispatch shim for the fused CSA/HCA ``Compressor`` gated-pooling kernels.

The fused forward+backward kernels live in the cudnn-frontend Python package
(``cudnn.csa.compressor``, added in https://github.com/NVIDIA/cudnn-frontend/pull/427,
following maintainer guidance on https://github.com/NVIDIA/Megatron-LM/pull/5984).
This module contains only the framework-side wiring:

  - an import guard that probes for the frontend's CSA compressor API by importing the
    concrete entry points (capability detection, no version comparisons — installs that
    predate the API simply lack ``cudnn.csa`` and keep the eager path);
  - a ``torch.autograd.Function`` connecting the frontend's forward/backward wrappers
    to autograd;
  - the dispatch helper :func:`maybe_compress_thd_fused` used by
    ``Compressor._forward_thd``: the fused result when the configuration is supported,
    or ``None`` (the caller keeps its eager implementation, which remains the semantic
    reference and the fallback everywhere).

What the fused path replaces (see the frontend's ``docs/fe-oss-apis/csa.md`` for kernel
details): for a THD packed path, the chain gather-index build -> gather -> ``+ APE`` ->
overlap-window transform (``coff == 2``) -> fp32 softmax -> gated weighted sum -> bf16
cast, i.e. one fused compute kernel per direction (backward adds a small ``dAPE``
zero-init) instead of ~40 forward / ~50 backward eager launches. CP pre-grouped buffers
reuse the same API with local token/group prefixes emitted by their compaction kernel;
static-capacity halo and padding rows remain noncanonical and their output gradients are
ignored by the frontend contract.
Numerics are fp32 with a single final bf16 rounding: not
bit-identical to the eager region (which rounds the softmax weights to bf16 and
multiplies in bf16) but at least as accurate against an fp64 oracle; forward, ``dKV``
and ``dScore`` are bitwise run-to-run deterministic, ``dAPE`` is accumulated with fp32
atomics and is not — the dispatch therefore keeps the eager path under
``torch.use_deterministic_algorithms(True)``. The two ratio families carry different
contracts against that fp32-intermediate eager reference: at ``compress_ratio == 4``
``dKV``/``dScore`` are bit-identical to it, while at ``compress_ratio == 128`` they are
faithful within the r128 gate tolerances (differing elements <= max(1, 0.1%), max_abs
<= 1.6e-2 on the frontend gate's documented input distribution). Keeping the softmax
weights in fp32 also makes the fused output measurably closer to an fp64 oracle than the
region it replaces: on a 4096-token pack the fused forward's max absolute error against
that oracle is 1.4-2.6x smaller than the eager region's across ``compress_ratio in
{4, 128}`` x ``coff in {1, 2}`` x ``head_dim in {128, 512}``, and matches an
fp32-intermediate eager reference (what remains is the single final bf16 rounding).
Measurements and numerics analysis: cudnn-frontend PR #427 and
https://github.com/NVIDIA/Megatron-LM/issues/5968.

Dispatch gating (everything else keeps eager):
  - the caller's ``use_fused_dsa_kernels(config)`` decision, i.e. the same switch that
    gates the other optional CSA/DSA fused kernels (``Compressor.use_fused_compressor``);
  - cudnn-frontend with the CSA compressor API importable, CUDA device with
    compute-capability major >= 10 (SM100+, the frontend's validated envelope);
  - ``compress_ratio in {4, 128}`` and ``coff in {1, 2}`` (the frontend's validated
    envelope; ``Compressor`` itself only produces ``(4, 2)`` and ``(128, 1)``), with
    ``compress_ratio == 128`` additionally restricted to ``head_dim in {128, 512}``
    (the r128 kernels' validated head dims); bf16 ``kv``/``score``, fp32 ``ape``, int32
    flat offsets (``total_tokens * coff * head_dim < 2**31``);
  - eager under deterministic mode (``dAPE`` atomics, above) and under
    ``torch.compile`` tracing (the frontend launch path takes raw pointers; eager lets
    the compiler fuse the region itself).

CUDA graphs: the frontend launch path is capture-compatible once the kernel for a given
``(ratio, head_dim, coff)`` configuration has been JIT-compiled; run one eager step per
configuration before capture (a first call that would JIT under capture raises a
``RuntimeError`` instead of corrupting the capture). The dispatch passes the caller's
static ``total_comp`` capacity through, so no device synchronization is introduced.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# The frontend kernels use no architecture-specific features beyond the SM100 baseline,
# so admit SM100 and newer GPU families. The dispatch pre-filters older devices so they
# keep eager silently. Mirrors ``cudnn.csa.compressor``'s envelope -- widen together
# with it.
_MINIMUM_COMPUTE_CAPABILITY_MAJOR = 10

# Mirrors ``cudnn.csa.compressor``'s validated envelope: ``ratio in {4, 128}`` x
# ``coff in {1, 2}``. ``Compressor`` currently only produces (4, 2) and (128, 1) --
# ``overlap`` is derived from ``compress_ratio`` -- but the gate follows the kernels
# rather than that derivation, so a future overlap-policy change keeps the fast path
# instead of silently falling back to eager. Widen together with the frontend.
_SUPPORTED_RATIOS = frozenset({4, 128})
_SUPPORTED_COFF = frozenset({1, 2})
# ratio 128 is served by the frontend's dedicated r128 kernels, validated for these
# head dims.
_R128_HEAD_DIMS = frozenset({128, 512})

# One-shot guard so a missing frontend warns once per process, not once per layer-call.
_warned_missing_frontend = False

_UNINITIALIZED = object()
# Lazily resolved ``cudnn.csa.compressor`` module: ``_UNINITIALIZED`` -> not probed yet,
# ``None`` -> probed and unavailable (reason kept in ``_frontend_error``).
_frontend = _UNINITIALIZED
_frontend_error: Optional[Exception] = None

_DEVICE_SUPPORTED_CACHE = {}


def _get_frontend():
    """Return cudnn-frontend's ``cudnn.csa.compressor`` module, or None if unavailable.

    Probed lazily (on the first dispatch call, not at Megatron import) by importing the
    concrete entry points rather than comparing version numbers: any install that has
    them can serve the dispatch, and ``nvidia-cudnn-frontend`` releases that predate the
    CSA compressor API (cudnn-frontend #427) lack ``cudnn.csa`` and fall back to eager
    naturally. Any import-time failure — missing package, missing ``nvidia-cutlass-dsl``
    (the frontend's ``cutedsl`` extra), a partially installed or ABI-incompatible
    frontend — also degrades to eager instead of breaking ``csa.py``'s import chain; the
    error is kept in ``_frontend_error`` for debugging.
    """
    global _frontend, _frontend_error
    if _frontend is _UNINITIALIZED:
        try:
            from cudnn.csa import compressor as fe_compressor

            fe_compressor.csa_compressor_forward_wrapper  # pylint: disable=pointless-statement
            fe_compressor.csa_compressor_backward_wrapper  # pylint: disable=pointless-statement
            _frontend = fe_compressor
        # A partially installed frontend or DSL can raise more than ImportError.
        except Exception as e:  # pylint: disable=broad-except
            _frontend = None
            _frontend_error = e
    return _frontend


def fused_compressor_available(device: Optional[torch.device] = None) -> bool:
    """Return True when the fused kernels can run: supported device + frontend importable.

    The device is checked first so the missing-frontend warning below is only emitted
    where the kernels could actually have run (compute-capability major >= 10); on every
    other device the eager path is the expected outcome, not a misconfiguration.
    """
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
        capability = torch.cuda.get_device_capability(index)
        supported = capability[0] >= _MINIMUM_COMPUTE_CAPABILITY_MAJOR
        _DEVICE_SUPPORTED_CACHE[index] = supported
    if not supported:
        return False
    if _get_frontend() is None:
        global _warned_missing_frontend
        if not _warned_missing_frontend:
            _warned_missing_frontend = True
            logger.warning(
                "CSA fused compressor is enabled and this device is supported, but "
                "cudnn-frontend's CSA compressor API is unavailable (%s: %s); keeping the "
                "eager compressor. Install nvidia-cudnn-frontend (with the 'cutedsl' "
                "extra) to enable the fused path.",
                type(_frontend_error).__name__,
                _frontend_error,
            )
        return False
    return True


class _CompressThdFused(torch.autograd.Function):
    """Autograd wiring around the cudnn-frontend forward/backward wrappers.

    The wrappers allocate their outputs and validate their inputs; gradient semantics
    (exact zeros for never-consumed elements, incoming gradients on static-capacity
    padding rows ignored) match autograd on the eager region — see the frontend docs.
    The backward wrapper raises under strict ``torch.use_deterministic_algorithms(True)``
    (``dAPE`` fp32 atomics); the dispatch already keeps eager in that mode, so this only
    triggers when deterministic mode is enabled between forward and backward.
    """

    @staticmethod
    def forward(
        ctx, kv, score, ape, cu_seqlens, cu_seqlens_comp, ratio, head_dim, coff, total_comp
    ):
        """Run the frontend forward wrapper; saves inputs for the fused backward."""
        kv = kv.contiguous()
        score = score.contiguous()
        ape_c = ape.contiguous()
        cu_i = cu_seqlens.to(dtype=torch.int32).contiguous()
        cuc_i = cu_seqlens_comp.to(dtype=torch.int32).contiguous()
        out = _get_frontend().csa_compressor_forward_wrapper(
            kv,
            score,
            ape_c,
            cu_i,
            cuc_i,
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
            total_comp=total_comp,
        )["out"]
        ctx.save_for_backward(kv, score, ape_c, cu_i, cuc_i)
        ctx.dims = (ratio, head_dim, coff)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Run the frontend backward wrapper; returns (dKV, dScore, dAPE, None...)."""
        kv, score, ape, cu_i, cuc_i = ctx.saved_tensors
        ratio, head_dim, coff = ctx.dims
        grad_kv, grad_score, grad_ape = _get_frontend().csa_compressor_backward_wrapper(
            kv,
            score,
            ape,
            cu_i,
            cuc_i,
            grad_out.contiguous(),
            ratio=ratio,
            head_dim=head_dim,
            coff=coff,
        )
        return grad_kv, grad_score, grad_ape, None, None, None, None, None, None


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
    enabled: bool = True,
) -> Optional[torch.Tensor]:
    """Dispatch helper for ``Compressor._forward_thd``: fused result or None (use eager).

    Returns the pooled ``(total_comp, 1, head_dim)`` bf16 tensor when the cudnn-frontend
    fused fast path supports the configuration, or None when the caller should keep the
    eager implementation. See the module docstring for the gating rules; the frontend's
    ``check_support`` re-validates the envelope and raises (instead of falling back) on
    anything the gates below let through.
    """
    if not enabled:
        return None
    if kv.device.type != "cuda":
        return None
    if not fused_compressor_available(kv.device):
        return None
    if ratio not in _SUPPORTED_RATIOS or coff not in _SUPPORTED_COFF:
        return None
    if ratio == 128 and head_dim not in _R128_HEAD_DIMS:
        return None
    if kv.dtype != torch.bfloat16 or score.dtype != torch.bfloat16:
        return None
    if ape.dtype != torch.float32:
        return None
    # The backward is not deterministic for dAPE (fp32 atomics); respect torch's
    # deterministic mode by keeping the (deterministic) eager path.
    if torch.are_deterministic_algorithms_enabled():
        return None
    # The frontend launch path uses raw pointers; keep eager (which torch.compile can
    # trace and fuse itself) when compiling.
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
