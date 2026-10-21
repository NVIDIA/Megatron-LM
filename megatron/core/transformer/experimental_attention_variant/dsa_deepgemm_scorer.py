# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional DeepGEMM MXFP8 selection scorer for the DSA indexer.

``dsa_indexer_scorer_backend='deepgemm'`` (with ``dsa_indexer_precision='mxfp8'``) replaces
the cuDNN MXFP8 dense scorer of the general packed THD path by DeepGEMM's
``fp8_fp4_mqa_logits`` (https://github.com/deepseek-ai/DeepGEMM, version 2.8.1 or newer). Its
BF16 logits feed DeepSelect's BF16 top-k (or cuDNN's BF16 radix top-k below DeepSelect's
width threshold); see :func:`dsa_cudnn_kernels._indexer_topk_packed_thd_deepgemm`.

Mapping. Every packed THD query row ``r`` (any CP size, zigzag CP included) attends to the
contiguous key range ``[starts[r], ends[r])`` of the original, globally ordered K. DeepGEMM
takes exactly such per-row ranges as ``cu_seq_len_k_start`` / ``cu_seq_len_k_end`` (``ks``,
``ke``) over a ``[Sk, D]`` K, so the call uses the original K (no duplicated-prefix K pool)
and the selected column ``t`` of row ``r`` is key ``ks[r] + t``. On SM100 it writes BF16
"compressed" logits ``[rows, max_seqlen_k]``: column ``t`` holds key ``ks[r] + t`` for
``t < ke[r] - ks[r]``; the other columns are left unwritten. The row stride is padded to a
multiple of 1024 bytes and the allocation covers whole padded rows, so
:func:`full_row_view` gives the compact ``[rows, stride]`` view cuDNN's top-k needs. One call
takes at most :func:`max_rows_per_call` query rows (116208 with 32 heads, 58104 with 64): the
scheduler-metadata kernel keeps two words per query block in one CTA's shared memory. Rows
are independent, so callers split larger calls by rows.

Measured on GB300 (DeepGEMM 2.8.2 ``5081f85``, JIT ``sm_103f``; CUDA graph p50, K 2048,
GLM-5.2 row lengths; quantize + score + top-k + index restore), per 1M CP64 rank-0 call:
FE 1.31 BF16 + DeepSelect FP32 58.1 ms, FE MXFP8 + DeepSelect FP32 43.0 ms, DeepGEMM MXFP8 +
DeepSelect BF16 28.8 ms; at 128K CP32 rank 0 2.38 / 1.84 / 1.04 ms. The transient score
buffer halves (64 -> 31 GiB at 1M rank 0). On ternary inputs the DeepGEMM logits equal FE
BF16's bit for bit on all 8.07e9 valid elements. Selection equals FE MXFP8's within 0.1%
recall (all-row overlap 0.992): recall is set by MXFP8 quantization (0.96 at 1M, 0.971-0.977
at 16K on untrained-indexer inputs), so MXFP8 selection needs its own training evidence
before production use. MXFP4 recall (0.72-0.90) is too low; it is not exposed.

Quantization: Transformer Engine ``MXFP8Quantizer`` (rowwise E4M3, logical UE8M0 1x32 scales)
on Q ``[T, H, 128]`` and K ``[Sk, 128]``; the 4 logical scale bytes of a row are one int32 in
DeepGEMM's packed UE8M0 layout (byte-identical to ``deep_gemm.utils.per_token_cast_to_fp8(
gran_k=32, use_packed_ue8m0=True)``), see
:func:`indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0`.

JIT and CUDA graphs. DeepGEMM compiles its kernels at first use (2.4 s cold for the
metadata and logits kernels; a warm call takes 0.7 ms) into ``DG_JIT_CACHE_DIR`` (deep_jit's
default is ``$HOME/.dj``). Point it at a persistent directory shared by all ranks of an image
(deep_jit commits entries atomically; a ``:``-separated list is searched front to back and a
miss compiles into the first entry), pre-populate it with one single-GPU :func:`warmup`, and
keep it apart from ``CUTE_DSL_CACHE_DIR`` and ``TRITON_CACHE_DIR``; nvcc must be in the
container at run time (else set ``DG_JIT_NVCC_COMPILER``). The specialization depends on the
device, the head count and dim, the precision and DeepGEMM's SM count
(``deep_gemm.get_num_sms()``, a template argument of both kernels), not on shapes. The first
eager call per specialization runs :func:`warmup`; a JIT failure there (no nvcc, an unwritable
cache) is recorded and the call falls back to the cuDNN MXFP8 scorer (logged once), while a
CUDA out-of-memory error propagates. :func:`mqa_logits` raises instead of
compiling or loading a kernel inside CUDA graph capture, so a capture needs at least one eager
step (``cuda_graph_warmup_steps >= 1``) or an explicit :func:`warmup`. That guard reads only
this module's record of the (device, head count, head dim) specializations that ran eagerly,
before any ``deep_gemm`` call: the first call of a process, even ``get_num_sms()``, builds
DeepGEMM's runtime (a cuBLASLt handle, device properties), which a capture cannot record.
Nothing else reads a tensor on the host: the scheduler metadata is built per call from that
call's ``(ks, ke)`` by a captured kernel and never cached across calls, and every buffer is
allocated in the call.

One device per process. deep_jit caches each kernel's ``CUfunction`` per process, keyed by
its source and not by device or CUDA context, and DeepGEMM's runtime (SM count, device
properties) is per process as well: a kernel loaded on one device fails on another with
``CUDA_ERROR_INVALID_HANDLE`` (seen on GB300 when ranks warmed up on ``cuda:0`` and then ran on
their own device). Megatron runs one device per rank, so the first device that calls DeepGEMM
owns it for the process; a call on any other device falls back to the cuDNN MXFP8 scorer with
the reason logged, and a direct :func:`mqa_logits` or :func:`warmup` on it raises.

Import. ``deep_gemm`` is imported lazily, once, at the first DeepGEMM-scored call (or reused
from ``sys.modules`` when another module, e.g. the batch-invariant kernels, imported it), and
probed: version >= 2.8.1, both functions with the named arguments used here (parsed from the
pybind signature line), no ``clean_logits`` or ``logits_dtype`` argument, and ``get_num_sms``.
2.8.1 made the SM100 logits BF16 and removed those two arguments; 2.8.0 (e.g. ``ea2b780``, the
build Megatron's ``batch_invariant`` extra pins) still defaults them to cleaned FP32 logits,
which this call's BF16 weights and ``max_seqlen_k`` reject. A missing or older build, or one that
fails to import (torch ABI mismatch), falls back to the cuDNN MXFP8 scorer with one warning;
check for the "DSA indexer scorer uses DeepGEMM" log line.
"""

from __future__ import annotations

import functools
import gc
import importlib
import inspect
import logging
import os
import re
import sys
from typing import Optional, Tuple

import torch
from torch import Tensor

from megatron.core.quantization import indexer_quantization

logger = logging.getLogger(__name__)

SCORER_BACKENDS = ("cudnn", "deepgemm")
PROVIDER_MODULE = "deep_gemm"
MIN_VERSION = (2, 8, 1)
# SM 10.x devices DeepGEMM's SM100 MQA logits dispatch on, and the one they have run on.
# SM 10.0 (GB200) falls back with a warning until test_dsa_deepgemm_scorer_gpu.py passes there.
KNOWN_SM = frozenset({(10, 0), (10, 3)})
SUPPORTED_SM = frozenset({(10, 3)})
HEAD_COUNTS = frozenset({32, 64})
HEAD_DIM = 128
LOGITS_ROW_ALIGNMENT_BYTES = 1024
JIT_CACHE_ENV = "DG_JIT_CACHE_DIR"
_REQUIRED_ARGUMENTS = {
    "fp8_fp4_mqa_logits": (
        "q",
        "kv",
        "weights",
        "cu_seq_len_k_start",
        "cu_seq_len_k_end",
        "max_seqlen_k",
        "schedule_meta",
    ),
    "get_mqa_logits_metadata": (
        "cu_seq_len_k_start",
        "cu_seq_len_k_end",
        "num_kv_tokens",
        "num_heads",
    ),
}
# Arguments of the 2.8.0 binding that 2.8.1 removed (see the module doc).
_REMOVED_ARGUMENTS = {"fp8_fp4_mqa_logits": ("clean_logits", "logits_dtype")}
_WARMUP_ROWS = 128
_WARMUP_KEYS = 256
# DeepGEMM's MQA-logits scheduler (``get_mqa_logits_metadata``, DeepGEMM 2.8.2) runs in one
# CTA of 256 threads that keeps two int32 words per query block of ``128 // heads`` rows plus
# one word per warp in shared memory, and host-asserts that this fits SM100's 232448 bytes.
_SM100_SMEM_CAPACITY_BYTES = 232448
_METADATA_THREADS = 256
CAPTURE_GUARD_MESSAGE = (
    "DeepGEMM would JIT-compile or load its MQA logits kernels inside CUDA graph capture: run "
    ">= 1 eager step before capture (cuda_graph_warmup_steps >= 1) or call "
    "dsa_deepgemm_scorer.warmup() for this device and indexer head count first."
)

_IMPORT_STATE: dict = {}
_LOGGED_REASONS: set = set()
# Specializations run eagerly in this process, and the ones whose JIT warm-up failed.
_WARM: set = set()
_JIT_FAILURES: dict = {}
# The same per (device, head count, head dim), known without calling deep_gemm: the capture
# guard reads only these before any deep_gemm call.
_WARM_SPECS: set = set()
_FAILED_SPECS: set = set()
# The one device this process runs DeepGEMM on ({"index": device index}, see the module doc).
_PROCESS_DEVICE: dict = {}


class _Provider:
    """The imported ``deep_gemm`` module and its parsed version."""

    def __init__(self, module, version: Tuple[int, ...]):
        self.module = module
        self.version = version

    @property
    def description(self) -> str:
        """Version, location and JIT cache, for the backend log line."""
        cache = os.environ.get(JIT_CACHE_ENV) or "unset: $HOME/.dj"
        return (
            f"deep_gemm {getattr(self.module, '__version__', '?')} from "
            f"{getattr(self.module, '__file__', '?')}, {JIT_CACHE_ENV} {cache}"
        )

    def num_sms(self) -> int:
        """DeepGEMM's current SM count, a template argument of its kernels."""
        return int(self.module.get_num_sms())


def _parse_version(text) -> Optional[Tuple[int, ...]]:
    """``(major, minor, patch)`` of a version string; a missing patch reads as 0."""
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3) or 0))


def _argument_names(function, name: str) -> Optional[frozenset]:
    """Named arguments of ``function``: its signature, else its pybind11 signature line."""
    try:
        return frozenset(inspect.signature(function).parameters)
    except (TypeError, ValueError):
        pass
    names = set()
    for line in (getattr(function, "__doc__", None) or "").splitlines():
        match = re.match(rf"\s*(?:\d+\.\s*)?{re.escape(name)}\((.*)\)\s*->", line)
        if match is not None:
            names.update(re.findall(r"(?:^|,\s*)(\w+)\s*:", match.group(1)))
    return frozenset(names) or None


def _api_problem(module) -> Tuple[Optional[str], Optional[Tuple[int, ...]]]:
    """Why ``module`` cannot serve as the scorer (or ``None``), and its parsed version."""
    raw_version = getattr(module, "__version__", None)
    version = _parse_version(raw_version)
    if version is None:
        return f"deep_gemm has no parsable __version__ ({raw_version!r})", None
    if version < MIN_VERSION:
        return (
            f"deep_gemm {raw_version} is older than {'.'.join(map(str, MIN_VERSION))} (SM100 "
            "BF16 MQA logits without clean_logits)",
            version,
        )
    missing = []
    for name, arguments in _REQUIRED_ARGUMENTS.items():
        function = getattr(module, name, None)
        if function is None:
            missing.append(name)
            continue
        names = _argument_names(function, name)
        if names is None:
            missing.append(f"{name} argument names")
            continue
        missing += [f"{name}({argument}=)" for argument in arguments if argument not in names]
        removed = [argument for argument in _REMOVED_ARGUMENTS.get(name, ()) if argument in names]
        if removed:
            return (
                f"deep_gemm {raw_version} has the pre-2.8.1 {name}("
                f"{'=, '.join(removed)}=) API (cleaned FP32 logits by default)",
                version,
            )
    if not callable(getattr(module, "get_num_sms", None)):
        missing.append("get_num_sms")
    if missing:
        return f"deep_gemm {raw_version} lacks {', '.join(missing)}", version
    return None, version


def max_rows_per_call(num_heads: int) -> int:
    """Most query rows one DeepGEMM MQA-logits call takes (its scheduler's shared memory).

    116208 rows with 32 heads, 58104 with 64; DeepGEMM raises from a host assert above it.
    """
    block_q = 128 // num_heads
    max_blocks = (_SM100_SMEM_CAPACITY_BYTES // 4 - _METADATA_THREADS // 32) // 2
    return max_blocks * block_q


def _capturing() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _load_provider() -> Optional[_Provider]:
    """Import and probe ``deep_gemm`` once; ``None`` with the reason recorded."""
    if "provider" not in _IMPORT_STATE:
        provider, error = None, None
        try:
            module = sys.modules.get(PROVIDER_MODULE)
            if module is None:
                module = importlib.import_module(PROVIDER_MODULE)
                # Like _ensure_dsa_namespace: collect the import's reference cycles now so a
                # first import from inside a step pins no caller frame.
                if not _capturing():
                    gc.collect()
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            error = f"cannot import deep_gemm: {type(exc).__name__}: {exc}"
        else:
            error, version = _api_problem(module)
            if error is None:
                provider = _Provider(module, version)
        _IMPORT_STATE["provider"] = provider
        _IMPORT_STATE["error"] = error
    return _IMPORT_STATE["provider"]


@functools.lru_cache(maxsize=None)
def _device_capability(device_index: int) -> tuple:
    return tuple(torch.cuda.get_device_capability(device_index))


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    return device.index if device.index is not None else torch.cuda.current_device()


def _device_problem(device: torch.device) -> Optional[str]:
    """Why ``device`` cannot run the DeepGEMM scorer, or ``None``."""
    if device.type != "cuda":
        return f"DeepGEMM needs a CUDA device, got {device}"
    sm = _device_capability(_device_index(device))
    if sm not in KNOWN_SM:
        return f"DeepGEMM's MQA logits need SM 10.x, got SM {sm[0]}.{sm[1]}"
    if sm not in SUPPORTED_SM:
        return (
            f"DeepGEMM is not validated on SM {sm[0]}.{sm[1]} (validated: "
            f"{', '.join(f'SM {a}.{b}' for a, b in sorted(SUPPORTED_SM))})"
        )
    return None


def _process_device_problem(device_index: int) -> Optional[str]:
    """Why this process cannot run DeepGEMM on ``device_index``, or ``None``.

    The first device that calls DeepGEMM owns it for the process (module docstring: deep_jit
    keeps one kernel handle per kernel and process, not per device).
    """
    owner = _PROCESS_DEVICE.get("index")
    if owner is None or owner == device_index:
        return None
    return (
        f"DeepGEMM already runs on cuda:{owner} in this process and supports one device per "
        f"process (deep_jit caches its kernel handles per process, not per device), so "
        f"cuda:{device_index} cannot use it"
    )


def _claim_process_device(device_index: int) -> None:
    """Make ``device_index`` this process's DeepGEMM device before its first deep_gemm call."""
    _PROCESS_DEVICE.setdefault("index", device_index)


def static_fallback_reason(
    *,
    device: torch.device,
    num_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    w_dtype: torch.dtype,
    deterministic: bool,
) -> Optional[str]:
    """Why a call of this geometry cannot use the DeepGEMM scorer, or ``None`` if it can.

    Reads no tensor value. The reason strings are stable per cause and double as the log-once
    key. The caller adds the JIT state (:func:`jit_problem`).
    """
    if deterministic:
        return "deterministic mode keeps the cuDNN MXFP8 scorer and the tie_break=1 top-k"
    problem = _device_problem(device)
    if problem is not None:
        return problem
    if num_heads not in HEAD_COUNTS or head_dim != HEAD_DIM:
        return (
            f"DeepGEMM selection needs 32 or 64 indexer heads of dim {HEAD_DIM}, got "
            f"{num_heads} heads of dim {head_dim}"
        )
    if (q_dtype, k_dtype, w_dtype) != (torch.bfloat16,) * 3:
        return (
            f"DeepGEMM selection needs BF16 indexer q/k/weights, got {q_dtype}/{k_dtype}/{w_dtype}"
        )
    if not indexer_quantization.HAVE_TE_MXFP8:
        return "Transformer Engine MXFP8 quantization is unavailable"
    if _load_provider() is None:
        return _IMPORT_STATE["error"]
    return None


def log_scorer_backend_once(reason: Optional[str]) -> None:
    """Log once per outcome which scorer a ``dsa_indexer_scorer_backend='deepgemm'`` run uses."""
    if reason in _LOGGED_REASONS:
        return
    _LOGGED_REASONS.add(reason)
    if reason is None:
        logger.info("DSA indexer scorer uses DeepGEMM (%s).", _load_provider().description)
        if not os.environ.get(JIT_CACHE_ENV):
            logger.warning(
                "%s is unset: DeepGEMM JIT-compiles into $HOME/.dj on every node; point it at a "
                "persistent directory shared by all ranks and pre-populate it with "
                "dsa_deepgemm_scorer.warmup().",
                JIT_CACHE_ENV,
            )
    elif reason.startswith("deterministic"):
        logger.info(
            "dsa_indexer_scorer_backend='deepgemm' uses the cuDNN MXFP8 scorer: %s.", reason
        )
    else:
        logger.warning(
            "dsa_indexer_scorer_backend='deepgemm' is unavailable, the cuDNN MXFP8 scorer runs: %s",
            reason,
        )


def _spec(device: torch.device, num_heads: int, head_dim: int) -> tuple:
    """The part of a specialization known without calling deep_gemm."""
    return (_device_index(device), num_heads, head_dim)


def _warm_key(device: torch.device, num_heads: int, head_dim: int) -> tuple:
    """The full specialization; calls ``deep_gemm.get_num_sms()``, so never cold in capture."""
    return (*_spec(device, num_heads, head_dim), "mxfp8", "scheduled", _load_provider().num_sms())


def jit_problem(device: torch.device, num_heads: int, head_dim: int) -> Optional[str]:
    """``None`` once this specialization ran eagerly; else warm it up now or report why not.

    Inside CUDA graph capture a specialization that never ran eagerly on this device raises
    (:data:`CAPTURE_GUARD_MESSAGE`) before any deep_gemm call. A warm-up that fails (no nvcc,
    an unwritable cache) is recorded and returned as the reason, so later calls (captured ones
    too) fall back without retrying. A CUDA out-of-memory error is not a JIT failure: it
    propagates and nothing is recorded. A device other than this process's DeepGEMM device
    returns that reason (:func:`_process_device_problem`) without calling deep_gemm, in
    capture too, so a capture falls back like its eager steps did.
    """
    spec = _spec(device, num_heads, head_dim)
    problem = _process_device_problem(spec[0])
    if problem is not None:
        return problem
    capturing = _capturing()
    if capturing and spec not in _WARM_SPECS and spec not in _FAILED_SPECS:
        raise RuntimeError(CAPTURE_GUARD_MESSAGE)
    _claim_process_device(spec[0])
    key = _warm_key(device, num_heads, head_dim)
    if key in _WARM:
        return None
    if key in _JIT_FAILURES:
        return _JIT_FAILURES[key]
    if capturing:  # warm at another DeepGEMM SM count only
        raise RuntimeError(CAPTURE_GUARD_MESSAGE)
    try:
        warmup(device, num_heads, head_dim)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, OSError) as exc:
        _JIT_FAILURES[key] = f"DeepGEMM JIT warm-up failed: {type(exc).__name__}: {exc}"
        _FAILED_SPECS.add(spec)
        return _JIT_FAILURES[key]
    return None


def _check_logits(logits: Tensor, rows: int, width: int) -> None:
    """The output contract this module relies on; raise naming the build otherwise."""
    problems = []
    if logits.dtype != torch.bfloat16:
        problems.append(f"dtype {logits.dtype} (expected BF16 on SM100)")
    if tuple(logits.shape) != (rows, width):
        problems.append(f"shape {tuple(logits.shape)} (expected {(rows, width)})")
    elif logits.stride(1) != 1 or logits.stride(0) < width:
        problems.append(f"strides {logits.stride()}")
    elif (logits.stride(0) * logits.element_size()) % LOGITS_ROW_ALIGNMENT_BYTES:
        problems.append(f"row stride {logits.stride(0)} not a multiple of 1024 bytes")
    elif (
        logits.storage_offset() + rows * logits.stride(0)
    ) * logits.element_size() > logits.untyped_storage().nbytes():
        problems.append("an allocation that does not cover whole padded rows")
    if problems:
        raise RuntimeError(
            f"{_load_provider().description}: fp8_fp4_mqa_logits returned {'; '.join(problems)}"
        )


def mqa_logits(
    q_fp8: Tensor,
    q_scale: Tensor,
    k_fp8: Tensor,
    k_scale: Tensor,
    weights: Tensor,
    ks: Tensor,
    ke: Tensor,
    *,
    max_seqlen_k: int,
) -> Tensor:
    """BF16 compressed logits ``[rows, max_seqlen_k]`` (row stride padded to 1024 bytes).

    ``q_fp8`` E4M3 ``[rows, H, 128]`` with int32 packed UE8M0 ``q_scale [rows, H]``, ``k_fp8``
    E4M3 ``[Sk, 128]`` with ``k_scale [Sk]``, BF16 ``weights [rows, H]``, int32 contiguous
    ``ks``/``ke`` with ``0 < ke - ks <= max_seqlen_k`` or an empty row. Column ``t`` of row
    ``r`` holds key ``ks[r] + t`` for ``t < ke[r] - ks[r]``; later columns are unwritten.
    Builds the scheduler metadata from ``(ks, ke)`` in the call (a captured kernel, no host
    read). Raises inside CUDA graph capture unless the specialization ran eagerly on this
    device before (checked before any deep_gemm call), on a device other than this process's
    DeepGEMM device, and for more than :func:`max_rows_per_call` rows.
    """
    rows, num_heads, head_dim = q_fp8.shape
    spec = _spec(q_fp8.device, num_heads, head_dim)
    capturing = _capturing()
    if capturing and spec not in _WARM_SPECS:
        raise RuntimeError(CAPTURE_GUARD_MESSAGE)
    provider = _load_provider()
    if provider is None:
        raise RuntimeError(f"the DeepGEMM scorer is unavailable: {_IMPORT_STATE['error']}")
    if rows > max_rows_per_call(num_heads):
        raise ValueError(
            f"one DeepGEMM MQA-logits call takes at most {max_rows_per_call(num_heads)} query "
            f"rows with {num_heads} heads, got {rows}; split the rows"
        )
    problem = _process_device_problem(spec[0])
    if problem is not None:
        raise RuntimeError(problem)
    _claim_process_device(spec[0])
    key = _warm_key(q_fp8.device, num_heads, head_dim)
    if capturing and key not in _WARM:
        raise RuntimeError(CAPTURE_GUARD_MESSAGE)
    # Keyword arguments only: the positional order changed between DeepGEMM releases.
    schedule_meta = provider.module.get_mqa_logits_metadata(
        cu_seq_len_k_start=ks, cu_seq_len_k_end=ke, num_kv_tokens=k_fp8.size(0), num_heads=num_heads
    )
    logits = provider.module.fp8_fp4_mqa_logits(
        q=(q_fp8, q_scale),
        kv=(k_fp8, k_scale),
        weights=weights,
        cu_seq_len_k_start=ks,
        cu_seq_len_k_end=ke,
        max_seqlen_k=max_seqlen_k,
        schedule_meta=schedule_meta,
    )
    _check_logits(logits, rows, max_seqlen_k)
    if not capturing:
        _WARM.add(key)
        _WARM_SPECS.add(spec)
    return logits


def full_row_view(logits: Tensor) -> Tensor:
    """The compact ``[rows, row_stride]`` view of logits checked by :func:`mqa_logits`."""
    rows, row_stride = logits.size(0), logits.stride(0)
    return logits.as_strided((rows, row_stride), (row_stride, 1))


def warmup(
    device: Optional[torch.device] = None, num_heads: int = 32, head_dim: int = HEAD_DIM
) -> None:
    """Compile (or load from ``DG_JIT_CACHE_DIR``) and run the kernels of one specialization.

    One tiny eager call (128 rows x 256 keys); call it outside CUDA graph capture, e.g. in a
    single-GPU job that pre-populates a shared JIT cache, and on the device this process
    scores on (after ``torch.cuda.set_device``): the first device that runs DeepGEMM is the
    only one the process can use (module docstring).
    """
    if _capturing():
        raise RuntimeError("dsa_deepgemm_scorer.warmup() must run outside CUDA graph capture")
    device = torch.device("cuda") if device is None else torch.device(device)
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((_WARMUP_ROWS, num_heads, head_dim), generator=generator, device=device)
    k = torch.randn((_WARMUP_KEYS, head_dim), generator=generator, device=device)
    w = torch.randn((_WARMUP_ROWS, num_heads), generator=generator, device=device)
    q_fp8, q_scale = indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0(q.bfloat16())
    k_fp8, k_scale = indexer_quantization.quantize_indexer_mxfp8_packed_ue8m0(k.bfloat16())
    ks = torch.zeros((_WARMUP_ROWS,), dtype=torch.int32, device=device)
    ke = torch.arange(1, _WARMUP_ROWS + 1, dtype=torch.int32, device=device)
    mqa_logits(q_fp8, q_scale, k_fp8, k_scale, w.bfloat16(), ks, ke, max_seqlen_k=_WARMUP_KEYS)


__all__ = [
    "CAPTURE_GUARD_MESSAGE",
    "SCORER_BACKENDS",
    "full_row_view",
    "jit_problem",
    "log_scorer_backend_once",
    "max_rows_per_call",
    "mqa_logits",
    "static_fallback_reason",
    "warmup",
]
