# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Padding-aware dense GEMMs: linear layers that compute only the live rows of a THD buffer.

Packed-sequence (THD) training with a fixed per-rank capacity (``pad_packed_seq_alignment``,
CUDA graphs) carries a padding tail at the end of every rank's token buffer; on long-sequence
recipes it is a third of the rows. Attention skips it through ``cu_seqlens`` and the MoE router
drops its routes, but every dense GEMM still multiplies it. This module runs such GEMMs through
Transformer Engine's graph-safe grouped GEMM with a single group whose row count is a device-side
scalar (``first_dims``), so the number of live rows can change from microbatch to microbatch
without a host synchronization and inside a CUDA graph replay. Rows at and beyond the live-row
bound of every produced tensor are zero-filled so that downstream row-wise kernels (and the
weight-gradient reductions over the token dimension) never see uninitialized memory.

Two consumers:

* the bf16 LM head (``LinearCrossEntropyModule``): ``PaddingAwareLinearFunction``;
* Transformer Engine linear layers (``TELinear`` and its column/row subclasses, bf16 or MXFP8):
  ``te_linear_padding_aware_forward`` reuses TE's grouped-tensor forward/backward (grouped
  quantization, cached quantized weights) with one group;
* Transformer Engine fused norm + linear layers (``TELayerNormColumnParallelLinear``):
  ``te_layernorm_linear_padding_aware_forward`` runs TE's norm kernels over the buffer and the
  same one-group GEMMs over the live rows.

Not covered: the op-fuser MLPs. ``TEFusedMLP`` runs ``BasicLinear`` dense GEMMs, and the
``GroupedLinear(num_groups=1)`` variants (``TEFusedMLPWithGroupedLinear``, ``FusedSharedExpertMLP``)
take Transformer Engine's single-group specialisation whose kernels derive the row count from the
tensor shape and ignore the split size, so a live-row split cannot shorten them.

Kernel choices are made once per layer from the model shape (weight shape, configured per-rank
capacity), never from the runtime row count: see ``dense_linear_policy``.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core.fusions.padding_aware_switches import (  # the MCORE_PADDING_AWARE_SKIP switch
    padding_aware_feature_disabled,
)

try:
    import fused_weight_gradient_mlp_cuda  # apex: the dense fp32-accumulate wgrad kernel mcore uses

    HAVE_APEX_WGRAD = True
except ImportError:  # pragma: no cover
    HAVE_APEX_WGRAD = False

try:
    import transformer_engine.pytorch  # noqa: F401 pylint: disable=unused-import
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm_for_grouped_tensor
    from transformer_engine.pytorch.tensor.storage.grouped_tensor_storage import (
        GroupedTensorStorage,
    )

    HAVE_PADDING_AWARE_GEMM = True
except ImportError:  # pragma: no cover - depends on the installed TE
    HAVE_PADDING_AWARE_GEMM = False

# Keyword arguments this module passes to ``_GroupedLinear._forward_grouped_tensor``. The
# installed Transformer Engine must take exactly these (plus the distributed-weight pair where it
# exists) and nothing else without a default; otherwise the TE-linear path stays off.
_TE_GROUPED_FORWARD_KWARGS = frozenset(
    {
        "inp",
        "m_splits",
        "use_bias",
        "is_first_microbatch",
        "fp8",
        "wgrad_store",
        "input_quantizers",
        "weight_quantizers",
        "grad_input_quantizers",
        "grad_weight_quantizers",
        "grad_output_quantizers",
        "fuse_wgrad_accumulation",
        "activation_dtype",
        "is_grad_enabled",
        "weight_workspaces",
        "cache_weight",
        "skip_fp8_weight_update",
        "save_original_input",
        "single_grouped_weight",
        "single_grouped_bias",
        "weights",
        "biases",
    }
)


def _te_grouped_forward_signature_supported(params) -> bool:
    """True when TE's grouped forward accepts the arguments passed here and requires no other:
    an older (LTS) or newer release with a different signature falls back to the dense path
    instead of failing at the first call."""
    passed = set(_TE_GROUPED_FORWARD_KWARGS)
    if "origin_weights" in params:
        passed |= {"origin_weights", "is_dist_weight"}
    if not passed <= set(params):
        return False
    for param in list(params.values())[1:]:  # the first parameter is the autograd ctx
        if param.name in passed or param.default is not inspect.Parameter.empty:
            continue
        if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            return False
    return True


try:
    from transformer_engine.pytorch.constants import TE_DType
    from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled
    from transformer_engine.pytorch.module.grouped_linear import _GroupedLinear
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

    try:
        from transformer_engine.pytorch.module.grouped_linear import (
            is_module_grouped_tensor_path_supported,
        )
    except ImportError:  # pragma: no cover - older TE without the predicate
        is_module_grouped_tensor_path_supported = None

    HAVE_TE_GROUPED_LINEAR_PATH = HAVE_PADDING_AWARE_GEMM and hasattr(
        _GroupedLinear, "_forward_grouped_tensor"
    )
    _TE_GROUPED_FORWARD_HAS_ORIGIN_WEIGHTS = False
    if HAVE_TE_GROUPED_LINEAR_PATH:
        _grouped_forward_params = inspect.signature(
            _GroupedLinear._forward_grouped_tensor
        ).parameters
        # Newer Transformer Engine (2.21+) takes the distributed-weight arguments explicitly.
        _TE_GROUPED_FORWARD_HAS_ORIGIN_WEIGHTS = "origin_weights" in _grouped_forward_params
        HAVE_TE_GROUPED_LINEAR_PATH = _te_grouped_forward_signature_supported(
            _grouped_forward_params
        )
except ImportError:  # pragma: no cover
    HAVE_TE_GROUPED_LINEAR_PATH = False
    _TE_GROUPED_FORWARD_HAS_ORIGIN_WEIGHTS = False

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

# Grouped tensors whose members are strided views of one buffer (``leading_dim``): needed for the
# grouped output projection of DSv4 attention; newer Transformer Engine only.
HAVE_TE_GROUPED_LEADING_DIM = HAVE_PADDING_AWARE_GEMM and (
    "leading_dim" in inspect.signature(GroupedTensorStorage._initialize_storage_fields).parameters
)

logger = logging.getLogger(__name__)


LIVE_ROW_ALIGNMENT = 128  # MXFP8 group quantization needs 128-row groups; bf16 has no constraint.


# --------------------------------------------------------------------------------------------
# Live-row bound and the context that publishes it
# --------------------------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _live_rows_bound_kernel(
        mask_ptr, out_ptr, n_rows, alignment, BLOCK: tl.constexpr  # True = padding
    ):
        # Single program: max index of a real (non-padding) row, +1, rounded up to ``alignment``,
        # clamped to n_rows. All-padding buffers map to n_rows (conservative).
        last = tl.full([BLOCK], -1, tl.int32)
        offs = tl.arange(0, BLOCK)
        for start in range(0, n_rows, BLOCK):
            idx = start + offs
            pad = tl.load(mask_ptr + idx, mask=idx < n_rows, other=1)
            real = (pad == 0) & (idx < n_rows)
            last = tl.where(real, idx, last)
        last_real = tl.max(last, axis=0)
        bound = (last_real + 1 + alignment - 1) // alignment * alignment
        # All-padding buffer: one alignment block (nothing downstream needs more than finite rows).
        bound = tl.where(last_real < 0, tl.minimum(alignment, n_rows), tl.minimum(bound, n_rows))
        tl.store(out_ptr, bound.to(tl.int64))


def live_rows_bound(
    padding_mask: torch.Tensor, alignment: int = LIVE_ROW_ALIGNMENT
) -> torch.Tensor:
    """Number of leading rows of a THD buffer that hold real tokens, as a device int64 ``(1,)``.

    ``padding_mask`` is the per-rank mask (True = padding) in any layout that flattens to the
    buffer's row order. The bound is the last real row plus one, rounded up to ``alignment`` and
    clamped to the buffer length, so the capacity tail is excluded while the small alignment gaps
    between sequences stay inside the computed range. All-padding buffers map to one alignment
    block (the GEMMs still run, over a finite zero block, and every tail is zero-filled).
    No host synchronization; one kernel launch.
    """
    mask = padding_mask.reshape(-1)
    total = mask.numel()
    if HAVE_TRITON and mask.is_cuda and mask.dtype in (torch.bool, torch.int8, torch.uint8):
        out = torch.empty(1, dtype=torch.int64, device=mask.device)
        _live_rows_bound_kernel[(1,)](
            mask.view(torch.int8) if mask.dtype == torch.bool else mask,
            out,
            total,
            alignment,
            BLOCK=4096,
        )
        return out
    valid = (~mask.to(torch.bool)).to(torch.int8)
    # argmax of the reversed mask = distance from the end to the last real row.
    from_end = torch.argmax(valid.flip(0), dim=0)
    last_real_plus_one = total - from_end
    bound = (last_real_plus_one + alignment - 1) // alignment * alignment
    bound = torch.where(valid.any(), bound, torch.full_like(bound, alignment))
    return bound.clamp(max=total).reshape(1).to(torch.int64)


@dataclass
class _LiveRows:
    bound: torch.Tensor  # int64 (1,) on the device
    total_rows: Optional[int]  # rows of the buffer the bound refers to (None = unchecked)


_live_rows: Optional[_LiveRows] = None


def _single_sequence_mask(padding_mask: torch.Tensor) -> bool:
    """Only a ``[1, T]`` / ``[T]`` (packed THD) mask flattens to the buffer's row order; a
    ``[b > 1, s]`` mask is batch-major while the ``[s, b, h]`` rows are sequence-major."""
    return padding_mask.dim() == 1 or padding_mask.shape[0] == 1


@contextlib.contextmanager
def padding_aware_live_rows(m_rows: Optional[torch.Tensor], total_rows: Optional[int] = None):
    """Make ``m_rows`` (from ``live_rows_bound``) the live-row bound for GEMMs run inside.

    ``total_rows`` is the row count of the buffer the bound was derived from; a GEMM whose input
    has a different number of rows (e.g. a tensor-parallel shard) ignores the bound.
    """
    global _live_rows
    previous = _live_rows
    _live_rows = None if m_rows is None else _LiveRows(m_rows, total_rows)
    try:
        yield
    finally:
        _live_rows = previous


def suspend_padding_aware_bound():
    """Context in which no live-row bound is published (rows not in the THD buffer order)."""
    return padding_aware_live_rows(None)


def current_live_rows() -> Optional[torch.Tensor]:
    """The live-row bound set by the enclosing ``padding_aware_live_rows`` context, if any."""
    return None if _live_rows is None else _live_rows.bound


def current_live_rows_for(rows: int) -> Optional[torch.Tensor]:
    """The live-row bound if one is set and it refers to a buffer of ``rows`` rows."""
    if _live_rows is None:
        return None
    if _live_rows.total_rows is not None and _live_rows.total_rows != rows:
        return None
    return _live_rows.bound


_log_budget = [256]  # MCORE_PADDING_AWARE_LOG=1: print the first bounds seen (debug; synchronizes)


def _maybe_log_bound(where: str, bound: torch.Tensor, total_rows: int) -> None:
    if _log_budget[0] <= 0 or os.environ.get("MCORE_PADDING_AWARE_LOG", "0") != "1":
        return
    if torch.cuda.is_current_stream_capturing():
        return  # bound.item() would abort the capture
    _log_budget[0] -= 1
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    live = int(bound.item())
    logger.info(
        "[padding-aware] rank %d %s: live rows %d/%d (padding %.1f%%)",
        rank,
        where,
        live,
        total_rows,
        100.0 * (1 - live / total_rows),
    )


def padding_aware_lm_head_context(config, padding_mask: Optional[torch.Tensor]):
    """Context for a model's post-process: padding-aware LM head GEMMs when enabled and possible."""
    if (
        not getattr(config, "padding_aware_lm_head", False)
        or padding_mask is None
        or not HAVE_PADDING_AWARE_GEMM
        or not HAVE_TRITON  # the bound and the tail zero-fill are Triton kernels
        or not _single_sequence_mask(padding_mask)
    ):
        return contextlib.nullcontext()
    bound = live_rows_bound(padding_mask)
    _maybe_log_bound("lm head", bound, padding_mask.numel())
    return padding_aware_live_rows(bound, padding_mask.numel())


def padding_aware_dense_linear_context(config, padding_mask: Optional[torch.Tensor]):
    """Context for a transformer layer: padding-aware TE linears when enabled and possible."""
    if (
        not getattr(config, "padding_aware_dense_linear", False)
        or padding_mask is None
        or not HAVE_TE_GROUPED_LINEAR_PATH
        or not _single_sequence_mask(padding_mask)
    ):
        return contextlib.nullcontext()
    bound = live_rows_bound(padding_mask)
    _maybe_log_bound("layer", bound, padding_mask.numel())
    return padding_aware_live_rows(bound, padding_mask.numel())


def with_padding_aware_dense_linears(fn):
    """Decorator for layer methods taking ``padding_mask``: publishes the live-row bound of that
    mask to the TE linears called inside (no-op unless ``config.padding_aware_dense_linear``).

    The bound is recomputed on every call (one small kernel) so that captured layer forwards
    read it from the graph's own static ``padding_mask`` input.
    """
    params = list(inspect.signature(fn).parameters)
    mask_index = params.index("padding_mask") - 1 if "padding_mask" in params else None

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        padding_mask = kwargs.get("padding_mask")
        if padding_mask is None and mask_index is not None and len(args) > mask_index:
            padding_mask = args[mask_index]
        with padding_aware_dense_linear_context(self.config, padding_mask):
            return fn(self, *args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------------------
# Tail zero-fill and grouped-tensor wrappers
# --------------------------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _zero_tail_rows_kernel(ptr, m_ptr, n_cols, stride, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)  # row * stride can exceed 2^31
        if row >= tl.load(m_ptr):
            offs = tl.arange(0, BLOCK)
            for start in range(0, n_cols, BLOCK):
                idx = start + offs
                tl.store(
                    ptr + row * stride + idx,
                    tl.zeros([BLOCK], dtype=ptr.dtype.element_ty),
                    mask=idx < n_cols,
                )


def zero_tail_rows(t: torch.Tensor, m_rows: torch.Tensor) -> None:
    """Zero rows ``>= m_rows`` of a 2-D tensor in place; the device scalar is read in-kernel."""
    assert HAVE_TRITON, "padding-aware GEMM needs Triton for the tail zero-fill"
    assert t.dim() == 2 and t.stride(1) == 1
    _zero_tail_rows_kernel[(t.shape[0],)](t, m_rows, t.shape[1], t.stride(0), BLOCK=1024)


def _grouped_rows(data2d: torch.Tensor, m_rows: torch.Tensor, offsets1: torch.Tensor):
    """Wrap a contiguous ``[rows, cols]`` tensor as one group of ``m_rows`` live rows."""
    rows, cols = data2d.shape
    return GroupedTensorStorage(
        shape=(rows, cols),
        dtype=data2d.dtype,
        num_tensors=1,
        quantizer=None,
        data=data2d.reshape(-1),
        first_dims=m_rows,
        tensor_offsets=offsets1 * cols,
    )


def _grouped_weight(weight: torch.Tensor):
    out_features, in_features = weight.shape
    return GroupedTensorStorage(
        shape=(out_features, in_features),
        dtype=weight.dtype,
        num_tensors=1,
        shapes=[(out_features, in_features)],
        quantizer=None,
        data=weight.reshape(-1),
    )


_wgrad_arenas: dict = {}  # (dtype, device) -> flat buffer shared by every weight shape
_retired_wgrad_arenas: list = []  # superseded arenas kept alive for captured graphs
_wgrad_scratches: dict = {}  # (shape, dtype, device) -> view into the arena (or own buffer)


def _wgrad_scratch(shape, dtype, device) -> torch.Tensor:
    """Persistent fp32 scratch for the weight gradient of one weight shape (see the policy).

    The weight gradients of different layers are computed one after the other, so all shapes
    share one arena sized for the largest; a layer gets a view into it. The arena only grows while
    no CUDA graph is being captured (a captured graph keeps the pointer it recorded); a shape that
    first appears during capture gets its own buffer.
    """
    key = (tuple(shape), dtype, device)
    scratch = _wgrad_scratches.get(key)
    if scratch is not None:
        return scratch
    if padding_aware_feature_disabled("arena"):
        scratch = torch.empty(shape, dtype=dtype, device=device)
        _wgrad_scratches[key] = scratch
        return scratch
    numel = 1
    for dim in shape:
        numel *= int(dim)
    arena_key = (dtype, device)
    arena = _wgrad_arenas.get(arena_key)
    if arena is None or arena.numel() < numel:
        if torch.cuda.is_current_stream_capturing():
            scratch = torch.empty(shape, dtype=dtype, device=device)
            _wgrad_scratches[key] = scratch
            return scratch
        if arena is not None:
            # A CUDA graph captured earlier may still address the old arena: keep it alive.
            _retired_wgrad_arenas.append(arena)
        arena = torch.empty(numel, dtype=dtype, device=device)
        _wgrad_arenas[arena_key] = arena
        # Earlier views pointed into the old arena; rebuild them on the new one.
        for other_key in list(_wgrad_scratches):
            other_shape, other_dtype, other_device = other_key
            if (other_dtype, other_device) == arena_key:
                _wgrad_scratches[other_key] = arena[: _wgrad_scratches[other_key].numel()].view(
                    other_shape
                )
    scratch = arena[:numel].view(shape)
    _wgrad_scratches[key] = scratch
    return scratch


# --------------------------------------------------------------------------------------------
# Kernel policy: decided once per layer from the model shape, never from the runtime row count
# --------------------------------------------------------------------------------------------
#
# Measurements (GB200, cuBLAS 13.3; see docs/user-guide/features/padding_aware_kernels.md):
#
# * Forward / dgrad / wgrad through the single-group grouped GEMM cost about the live-row
#   fraction of the dense GEMM once the GEMM is big enough (N >= ~1024, capacity >= 8192); for
#   tiny N (<= 128) the grouped kernel is slower than the dense one at any size.
# * The grouped path has fixed costs per linear: the cuBLASLt setup kernels and the split
#   offsets (~20 us), the zero-fill of the padded tail of the output and of dgrad (bandwidth
#   bound), and - when the weight gradient goes through the fp32 scratch - one add over the
#   whole fp32 weight gradient (~3x its bytes at HBM speed). At a per-rank capacity of 4096 rows
#   no DSv4 layer shape amortises them (the proxy); at 16384 rows (256K / CP16) every layer with
#   N >= 1024 does, by about the padding fraction.
# * Weight gradient, accumulation into fp32 main_grad: cuBLASLt's grouped kernels are
#   runtime-beta variants whose beta != 0 epilogue reads C at ~1 TB/s effective, a penalty
#   proportional to the fp32 weight bytes (2-2.6 ms on the 2.1 GB LM head) that does not exist
#   for the dense compile-time ``badd`` kernel. A beta = 0 GEMM into a persistent fp32 scratch
#   plus one add is cheaper for weights above WGRAD_SCRATCH_MIN_BYTES; tiny weights keep the
#   in-kernel accumulation (launch bound).
# * bf16 LM head (mcore's own linear): the alternative to the grouped wgrad is the dense fused
#   apex kernel over the full capacity. Dividing both costs by N*K leaves a rule in the capacity
#   alone: grouped(beta 0) + add wins from LM_HEAD_WGRAD_MIN_ROWS on at >= 20 % padding (break-even
#   9-12 % padding at 8192 rows; at 4096 rows dense wins unless padding > 33 %).
#
# ``dense_linear_policy`` therefore evaluates a small cost model on the static shape (weight
# shape, configured capacity, recipe): predicted saving = assumed padding fraction x dense GEMM
# time; predicted overhead = setup + tail zero-fill + (scratch add). Overrides:
# MCORE_PADDING_AWARE_FORCE=1 enables every layer that passes the hard constraints;
# MCORE_PADDING_AWARE_WGRAD = ``grouped`` | ``dense`` (LM head) / ``scratch`` | ``accumulate``.

MIN_CAPACITY_ROWS = 4096
LM_HEAD_WGRAD_MIN_ROWS = 8192
WGRAD_SCRATCH_MIN_BYTES = 8 << 20
# Cost model constants (GB200 measurements, bench_te_linear_pa.py): achieved GEMM rate for
# these shapes, achieved bandwidth of the fill/add kernels, fixed per-linear overhead, assumed
# padding fraction and the margin the saving must beat.
GEMM_FLOPS_PER_S = {"fp8": 2.5e15, "bf16": 1.5e15}
FILL_BYTES_PER_S = 4.0e12
FIXED_OVERHEAD_S = 20e-6
ASSUMED_PADDING = 0.2
# Margin the predicted saving must beat (calibrated against bench_te_linear_pa.py so that no
# shape measured as a loss at p = 0.2 is enabled); MCORE_PADDING_AWARE_MARGIN overrides it.
try:
    POLICY_MARGIN = float(os.environ.get("MCORE_PADDING_AWARE_MARGIN", "1.3"))
except ValueError:  # pragma: no cover
    logger.warning("MCORE_PADDING_AWARE_MARGIN is not a number; using 1.3")
    POLICY_MARGIN = 1.3


def static_capacity_rows(config) -> Optional[int]:
    """Per-rank row capacity of the THD buffer, from the config (None if not configured)."""
    rows = getattr(config, "max_seqlen_per_dp_cp_rank", None)
    if rows is None:
        return None
    if getattr(config, "sequence_parallel", False):
        rows //= max(1, getattr(config, "tensor_model_parallel_size", 1))
    return int(rows)


def _wgrad_override() -> str:
    return os.environ.get("MCORE_PADDING_AWARE_WGRAD", "auto")


def training_uses_cuda_graphs(config) -> bool:
    """Whether the transformer layers are replayed from CUDA graphs during training."""
    impl = getattr(config, "cuda_graph_impl", "none")
    return bool(getattr(config, "enable_cuda_graph", False)) or impl not in (None, "none")


def lm_head_wgrad_uses_grouped(capacity_rows: int) -> bool:
    """bf16 LM head: live-row grouped wgrad into the fp32 scratch (+ add) vs dense fused kernel."""
    override = _wgrad_override()
    if override == "grouped":
        return True
    if override == "dense":
        return False
    return capacity_rows >= LM_HEAD_WGRAD_MIN_ROWS


_logged_policies: set = set()


def log_policy_decision(
    owner: str, out_features: int, in_features: int, enabled: bool, reason: str
) -> None:
    """Log one line per distinct (module class, weight shape, decision) at construction, on rank 0
    only, so a run's log shows which linears run padding-aware and why the others do not."""
    key = (owner, out_features, in_features, enabled, reason)
    if key in _logged_policies:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    _logged_policies.add(key)
    logger.info(
        "padding-aware policy: %s [%d x %d] -> %s (%s)",
        owner,
        out_features,
        in_features,
        "padding-aware" if enabled else "dense",
        reason,
    )


@dataclass(frozen=True)
class DenseLinearPolicy:
    """Per-layer decision for a TE linear (see ``dense_linear_policy``)."""

    enabled: bool
    wgrad_scratch: bool  # beta-0 GEMM into the fp32 scratch + add (True) vs in-kernel beta-1
    reason: str = ""
    # The layer passes the structural constraints (bias, TP communication, dims, capacity);
    # ``enabled`` additionally requires the cost model to predict a gain.
    eligible: bool = False


DISABLED_POLICY = DenseLinearPolicy(False, False, "disabled")


def predicted_padding_aware_gain(
    out_features: int,
    in_features: int,
    capacity: int,
    fp8: bool,
    wgrad_scratch: bool,
    assumed_padding: float = ASSUMED_PADDING,
) -> tuple[float, float]:
    """(predicted saving, predicted overhead) in seconds per forward+backward of one linear at
    the assumed padding fraction; the padding-aware path pays off when saving > margin x overhead.
    """
    flops = 3 * 2.0 * capacity * out_features * in_features  # fwd + dgrad + wgrad
    dense_gemm = flops / GEMM_FLOPS_PER_S["fp8" if fp8 else "bf16"]
    saving = assumed_padding * dense_gemm
    tail_rows = assumed_padding * capacity
    overhead = FIXED_OVERHEAD_S + tail_rows * (out_features + in_features) * 2 / FILL_BYTES_PER_S
    if wgrad_scratch:
        overhead += 3 * 4 * out_features * in_features / FILL_BYTES_PER_S
    return saving, overhead


def _is_distributed_weight(tensor) -> bool:
    """TE's ``DistributedWeight`` capability marker (e.g. GTP-sharded parameters)."""
    return bool(getattr(tensor, "is_distributed_weight", False))


def _owns_distributed_weight(module) -> bool:
    """Whether any parameter the module itself owns is a distributed weight."""
    if isinstance(module, torch.nn.Module):
        return any(_is_distributed_weight(p) for p in module.parameters(recurse=False))
    return _is_distributed_weight(getattr(module, "weight", None))


def dense_linear_policy(module, config) -> DenseLinearPolicy:
    """Decide once, at construction, whether a TE linear runs padding-aware and how its weight
    gradient is accumulated. Inputs are the layer's weight shape, the configured capacity and
    the recipe; nothing about the runtime input."""
    if not getattr(config, "padding_aware_dense_linear", False):
        return DISABLED_POLICY
    if not HAVE_TE_GROUPED_LINEAR_PATH or not HAVE_TRITON:
        return DenseLinearPolicy(False, False, "transformer engine grouped-tensor path missing")
    out_features, in_features = int(module.out_features), int(module.in_features)
    if getattr(module, "use_bias", False):
        return DenseLinearPolicy(False, False, "bias")
    if getattr(module, "is_expert", False):
        return DenseLinearPolicy(False, False, "expert linear (rows are routed tokens)")
    if _owns_distributed_weight(module):
        # GTP-sharded weights are all-gathered inside TE's own forward; the grouped call here
        # would see the shard.
        return DenseLinearPolicy(False, False, "GTP-sharded weight")
    if module.parallel_mode is not None and module.tp_size > 1:
        return DenseLinearPolicy(False, False, "tensor-parallel communication inside TE")
    if getattr(module, "symmetric_ar_type", None) is not None:
        return DenseLinearPolicy(False, False, "symmetric all-reduce")
    if any(
        getattr(module, name, False)
        for name in (
            "ub_overlap_ag_fprop",
            "ub_overlap_rs_fprop",
            "ub_overlap_ag_dgrad",
            "ub_overlap_rs_dgrad",
            "ub_bulk_dgrad",
            "ub_bulk_wgrad",
        )
    ):
        return DenseLinearPolicy(False, False, "userbuffers overlap")
    if getattr(config, "delay_wgrad_compute", False):
        return DenseLinearPolicy(False, False, "delayed wgrad")
    if in_features % 32 or out_features % 32:
        return DenseLinearPolicy(False, False, "weight dims not multiples of 32")
    capacity = static_capacity_rows(config)
    if capacity is None:
        return DenseLinearPolicy(False, False, "max_seqlen_per_dp_cp_rank not set")
    if capacity < MIN_CAPACITY_ROWS:
        return DenseLinearPolicy(False, False, f"capacity {capacity} < {MIN_CAPACITY_ROWS}")
    override = _wgrad_override()
    if override == "scratch":
        scratch = True
    elif override == "accumulate":
        scratch = False
    else:
        scratch = out_features * in_features * 4 >= WGRAD_SCRATCH_MIN_BYTES
    if os.environ.get("MCORE_PADDING_AWARE_FORCE", "0") == "1":
        return DenseLinearPolicy(True, scratch, "forced", eligible=True)
    if (
        not training_uses_cuda_graphs(config)
        and os.environ.get("MCORE_PADDING_AWARE_EAGER", "0") != "1"
    ):
        # Eager steps pay the grouped path's extra CPU time per call on the critical path
        # (about twice TE's dense linear); only graph replay hides it.
        return DenseLinearPolicy(False, scratch, "eager mode (needs CUDA graphs)", eligible=True)
    assumed_padding = float(getattr(config, "padding_aware_assumed_padding", ASSUMED_PADDING))
    saving, overhead = predicted_padding_aware_gain(
        out_features,
        in_features,
        capacity,
        bool(getattr(config, "fp8", None)),
        scratch,
        assumed_padding,
    )
    if saving < POLICY_MARGIN * overhead:
        return DenseLinearPolicy(
            False,
            scratch,
            f"predicted saving {saving * 1e6:.0f} us < {POLICY_MARGIN} x overhead "
            f"{overhead * 1e6:.0f} us at {assumed_padding:.0%} padding",
            eligible=True,
        )
    return DenseLinearPolicy(True, scratch, "ok", eligible=True)


# --------------------------------------------------------------------------------------------
# bf16 LM head (mcore linear)
# --------------------------------------------------------------------------------------------


def _dense_wgrad_accumulate(x2d, dy2d, main_grad) -> None:
    """``main_grad += dy2d^T @ x2d`` over all rows, as mcore's fused path does (apex kernel)."""
    if HAVE_APEX_WGRAD and main_grad.dtype == torch.float32:
        fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(x2d, dy2d, main_grad)
    else:
        main_grad.add_(torch.matmul(dy2d.t(), x2d).to(main_grad.dtype))


class PaddingAwareLinearFunction(torch.autograd.Function):
    """``y = x @ W^T`` over the first ``m_rows`` rows of ``x`` (device scalar); the rest of ``y``
    is zero. Backward mirrors mcore's ``LinearWithGradAccumulationAndAsyncCommunication`` for the
    bf16 LM head: optional fused accumulation of the weight gradient into ``weight.main_grad``
    (grouped GEMM with fp32 accumulation over the live rows only, or the dense fused kernel over
    the full capacity, chosen by ``lm_head_wgrad_uses_grouped``) and the TP dgrad all-reduce.
    """

    @staticmethod
    def forward(
        ctx,
        inp,
        weight,
        m_rows,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        tp_group,
        wgrad_grouped,
    ):
        """GEMM over the live rows; the padding tail of the output is zero."""
        x2d = inp.reshape(-1, inp.shape[-1])
        assert x2d.is_contiguous() and weight.is_contiguous()
        offsets1 = tex.splits_to_offsets(m_rows, 1)
        out = torch.empty(*inp.shape[:-1], weight.shape[0], dtype=inp.dtype, device=inp.device)
        out2d = out.view(-1, weight.shape[0])
        general_grouped_gemm_for_grouped_tensor(
            _grouped_weight(weight),
            _grouped_rows(x2d, m_rows, offsets1),
            _grouped_rows(out2d, m_rows, offsets1),
            layout="TN",
        )
        zero_tail_rows(out2d, m_rows)
        ctx.save_for_backward(inp, weight, m_rows, offsets1)
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        ctx.wgrad_grouped = wgrad_grouped
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """dgrad over the live rows (tail zero) and wgrad reduced over the live rows only."""
        inp, weight, m_rows, offsets1 = ctx.saved_tensors
        x2d = inp.reshape(-1, inp.shape[-1])
        dy2d = grad_output.contiguous().view(-1, weight.shape[0])
        grouped_dy = _grouped_rows(dy2d, m_rows, offsets1)
        # dgrad: dX = dY W over the live rows.
        grad_input = torch.empty_like(inp)
        dx2d = grad_input.view(-1, inp.shape[-1])
        general_grouped_gemm_for_grouped_tensor(
            _grouped_weight(weight), grouped_dy, _grouped_rows(dx2d, m_rows, offsets1), layout="NN"
        )
        zero_tail_rows(dx2d, m_rows)
        handle = None
        if ctx.allreduce_dgrad:
            handle = torch.distributed.all_reduce(grad_input, group=ctx.tp_group, async_op=True)
        # wgrad: dW = dY^T X, reduction over the live rows only (padded rows contribute nothing).
        grouped_x = _grouped_rows(x2d, m_rows, offsets1)
        if ctx.gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
            if ctx.wgrad_grouped:
                # Live rows only; fp32 result through the scratch (see the policy note above).
                scratch = _wgrad_scratch(weight.shape, main_grad.dtype, main_grad.device)
                general_grouped_gemm_for_grouped_tensor(
                    grouped_x, grouped_dy, [scratch], layout="NT"
                )
                main_grad.add_(scratch)
            else:
                # Dense over the full capacity: the padded rows of x and dy are zero, so the result
                # is the same and this is exactly mcore's fused kernel.
                _dense_wgrad_accumulate(x2d, dy2d, main_grad)
            if hasattr(weight, "grad_added_to_main_grad"):
                grad_weight = torch.empty(
                    weight.shape, dtype=inp.dtype, device=inp.device, requires_grad=False
                )
                if getattr(weight, "zero_out_wgrad", False):
                    grad_weight.zero_()
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            grad_weight = torch.empty_like(weight)
            general_grouped_gemm_for_grouped_tensor(
                grouped_x, grouped_dy, [grad_weight], layout="NT"
            )
        if handle is not None:
            handle.wait()
        return grad_input, grad_weight, None, None, None, None, None


def padding_aware_linear(
    inp,
    weight,
    m_rows,
    *,
    gradient_accumulation_fusion,
    allreduce_dgrad,
    tp_group,
    wgrad_grouped: Optional[bool] = None,
):
    """Functional entry: see ``PaddingAwareLinearFunction``. ``wgrad_grouped`` defaults to the
    static rule on the buffer's row count."""
    if wgrad_grouped is None:
        wgrad_grouped = lm_head_wgrad_uses_grouped(inp.shape[:-1].numel())
    return PaddingAwareLinearFunction.apply(
        inp, weight, m_rows, gradient_accumulation_fusion, allreduce_dgrad, tp_group, wgrad_grouped
    )


# --------------------------------------------------------------------------------------------
# Transformer Engine linear layers (bf16 / MXFP8)
# --------------------------------------------------------------------------------------------


class PaddingAwareTELinearFunction(torch.autograd.Function):
    """One-group version of TE's ``_GroupedLinear`` grouped-tensor path for a ``te.Linear``.

    Forward and backward are TE's own (grouped MXFP8 quantization of the live rows, cached
    quantized weight, cuBLASLt grouped GEMM); this wrapper adds the tail zero-fill of the output
    and of the input gradient, and routes the weight gradient through the fp32 scratch when the
    layer's policy says so (TE would otherwise accumulate in-kernel with the slow beta = 1
    epilogue).
    """

    @staticmethod
    def forward(ctx, inp, m_rows, non_tensor_args, weight):
        """GEMM over the live rows; the padding tail of the output is zero."""
        (
            is_first_microbatch,
            fp8,
            input_quantizer,
            weight_quantizer,
            grad_output_quantizer,
            fuse_wgrad_accumulation,
            activation_dtype,
            is_grad_enabled,
            weight_workspace,
            cache_weight,
            skip_fp8_weight_update,
            save_original_input,
            wgrad_scratch,
        ) = non_tensor_args
        out, new_workspaces = _GroupedLinear._forward_grouped_tensor(
            ctx,
            inp=inp,
            m_splits=m_rows,
            use_bias=False,
            is_first_microbatch=is_first_microbatch,
            fp8=fp8,
            wgrad_store=None,
            input_quantizers=[input_quantizer],
            weight_quantizers=[weight_quantizer],
            grad_input_quantizers=[None],
            grad_weight_quantizers=[None],
            grad_output_quantizers=[grad_output_quantizer],
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            activation_dtype=activation_dtype,
            is_grad_enabled=is_grad_enabled,
            weight_workspaces=[weight_workspace],
            cache_weight=cache_weight,
            skip_fp8_weight_update=skip_fp8_weight_update,
            save_original_input=save_original_input,
            single_grouped_weight=False,
            single_grouped_bias=False,
            weights=(weight,),
            biases=(None,),
            **_te_grouped_forward_extra_kwargs(weight),
        )
        zero_tail_rows(out.view(-1, out.shape[-1]), m_rows)
        if is_grad_enabled:
            ctx.pa_m_rows = m_rows
            ctx.pa_wgrad_scratch = wgrad_scratch
        return out, new_workspaces

    @staticmethod
    def backward(ctx, grad_output, _grad_workspaces):
        """TE's grouped backward; dgrad tail zeroed; wgrad via the fp32 scratch if configured."""
        m_rows = ctx.pa_m_rows
        use_scratch = (
            ctx.pa_wgrad_scratch and ctx.fuse_wgrad_accumulation and ctx.weights_requires_grad
        )
        origin_weight = main_grad = scratch = None
        if use_scratch:
            origin_weight = ctx.origin_weight_refs[0]()
            assert origin_weight is not None, "weight was removed while fuse_wgrad_accumulation"
            main_grad = ctx.main_grad_funcs[0]()
            # TE accumulates (beta 1) unless this is the first microbatch or the weight asks for
            # an overwrite; reproduce that with the scratch: GEMM with beta 0, then add or copy.
            te_accumulates = not getattr(ctx, "origin_weights_overwrite_main_grad", False) and (
                ctx.is_first_microbatch is None or not ctx.is_first_microbatch
            )
            scratch = _wgrad_scratch(main_grad.shape, main_grad.dtype, main_grad.device)
            ctx.main_grad_funcs = [lambda: scratch]
            ctx.origin_weights_overwrite_main_grad = True  # => beta 0 into the scratch
        result = _GroupedLinear._backward_grouped_tensor(ctx, grad_output)
        grad_input, grad_weight = result[0], result[5]
        if grad_input is not None:
            zero_tail_rows(grad_input.view(-1, grad_input.shape[-1]), m_rows)
        if use_scratch:
            origin_weight.main_grad = main_grad  # TE pointed it at the scratch
            if te_accumulates:
                main_grad.add_(scratch)
            else:
                main_grad.copy_(scratch)
        return grad_input, None, None, grad_weight


def _te_grouped_forward_extra_kwargs(weight):
    """Arguments newer Transformer Engine versions require from ``_forward_grouped_tensor``."""
    if _TE_GROUPED_FORWARD_HAS_ORIGIN_WEIGHTS:
        return {"origin_weights": (weight,), "is_dist_weight": False}
    return {}


def _graph_capture_flags(is_first_microbatch):
    """TE skips the weight-cache update under graph capture via a device flag."""
    skip_fp8_weight_update = None
    if FP8GlobalStateManager.fp8_graph_capturing():
        skip_fp8_weight_update = (
            FP8GlobalStateManager.quantization_state.skip_fp8_weight_update_tensor
        )
    if skip_fp8_weight_update is not None:
        is_first_microbatch = False
    return is_first_microbatch, skip_fp8_weight_update


def _activation_dtype_for(inp: torch.Tensor) -> torch.dtype:
    """The dtype TE's ``set_activation_dtype`` will pick: autocast's, else the input's."""
    try:
        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except TypeError:  # pragma: no cover - torch < 2.4
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return inp.dtype


def _te_grouped_path_possible(inp: torch.Tensor) -> bool:
    """The recipe and dtype conditions of ``_te_grouped_call``, read from the global quantization
    state *before* ``prepare_forward``: a call that has to fall back must not enter TE's forward
    bookkeeping twice (delayed-scaling amax registration, activation dtype checks)."""
    if FP8GlobalStateManager.is_fp8_calibration():
        return False
    activation_dtype = _activation_dtype_for(inp)
    if FP8GlobalStateManager.is_fp8_enabled():
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if is_module_grouped_tensor_path_supported is not None:
            return bool(is_module_grouped_tensor_path_supported(recipe, activation_dtype))
        return bool(recipe.mxfp8() and recipe.backward_override is None)
    return activation_dtype in (torch.bfloat16, torch.float16)


def _te_grouped_call(module, is_first_microbatch, policy, is_grad_enabled, skip_fp8_weight_update):
    """Module-level plumbing shared by ``te.pytorch.Linear.forward`` and
    ``LayerNormLinear.forward`` for the one-group grouped path: recipe check, quantizers,
    weight preswizzle, cached quantized weight. Must run between ``prepare_forward`` and
    ``end_forward``. Returns ``(weight, non_tensor_args, cache_name)`` or None when this call
    cannot take the grouped path (calibration, unsupported recipe or dtype, quantized output).
    """
    if module.fp8_calibration:
        return None
    if module.fp8:
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        if is_module_grouped_tensor_path_supported is not None:
            if not is_module_grouped_tensor_path_supported(recipe, module.activation_dtype):
                return None
        elif not recipe.mxfp8() or recipe.backward_override is not None:
            return None
    elif module.activation_dtype not in (torch.bfloat16, torch.float16):
        return None

    weight, _ = module._get_weight_and_bias_tensors()  # bias is excluded by the policy
    if _is_distributed_weight(weight):
        return None  # GTP-sharded: only TE's own forward gathers it
    input_quantizer, weight_quantizer, output_quantizer, _, _, grad_output_quantizer = (
        module._get_quantizers(False, False, is_grad_enabled)
    )
    if output_quantizer is not None:
        return None
    if weight_quantizer is not None and hasattr(module, "_enable_weight_preswizzle"):
        weight_quantizer.optimize_for_gemm = module._enable_weight_preswizzle(
            weight_quantizer, weight
        )
    if isinstance(weight, QuantizedTensorStorage):
        # Primary weights kept in FP8: TE uses the weight's own quantizer.
        weight_quantizer = weight._quantizer

    cache_name = None if (is_first_microbatch is None or module.is_fsdp2) else "weight"
    weight_workspace = module._fp8_workspaces.get(cache_name) if cache_name is not None else None
    non_tensor_args = (
        is_first_microbatch,
        module.fp8,
        input_quantizer,
        weight_quantizer,
        grad_output_quantizer,
        module.fuse_wgrad_accumulation,
        module.activation_dtype,
        is_grad_enabled,
        weight_workspace,
        cache_name is not None,
        skip_fp8_weight_update,
        bool(getattr(module, "save_original_input", False)),
        policy.wgrad_scratch,
    )
    return weight, non_tensor_args, cache_name


def _store_weight_workspace(module, cache_name, new_workspaces) -> None:
    if cache_name is not None and new_workspaces[0] is not None:
        workspace = new_workspaces[0]
        if isinstance(workspace, torch.Tensor):
            workspace = workspace.detach()
        module._fp8_workspaces[cache_name] = workspace


def te_linear_padding_aware_forward(
    module, inp: torch.Tensor, m_rows: torch.Tensor, is_first_microbatch, policy
):
    """Run a ``te.pytorch.Linear`` (no bias, no TE-side communication) over the live rows.

    Mirrors the module-level plumbing of ``te.pytorch.Linear.forward`` (quantizers, cached
    quantized weight, graph-capture flag). Returns None when this call cannot take the grouped
    path (debug mode, calibration, CPU offloading, unsupported recipe, quantized output or
    input); the caller then runs the regular forward.
    """
    if (
        isinstance(inp, QuantizedTensorStorage)
        or module.is_debug_iter()
        or is_cpu_offload_enabled()
        or not _te_grouped_path_possible(inp)
    ):
        return None
    is_grad_enabled = torch.is_grad_enabled()
    is_first_microbatch, skip_fp8_weight_update = _graph_capture_flags(is_first_microbatch)

    inp = module.prepare_forward(inp)
    try:
        call = _te_grouped_call(
            module, is_first_microbatch, policy, is_grad_enabled, skip_fp8_weight_update
        )
        if call is None:
            return None
        weight, non_tensor_args, cache_name = call
        if is_grad_enabled:
            out, new_workspaces = PaddingAwareTELinearFunction.apply(
                inp, m_rows, non_tensor_args, weight
            )
        else:
            out, new_workspaces = PaddingAwareTELinearFunction.forward(
                None, inp, m_rows, non_tensor_args, weight
            )
        _store_weight_workspace(module, cache_name, new_workspaces)
    finally:
        module.end_forward()
    return out


# --------------------------------------------------------------------------------------------
# Transformer Engine fused norm + linear layers (``te.pytorch.LayerNormLinear``)
# --------------------------------------------------------------------------------------------


class _SplicedCtx:
    """Lets TE's grouped forward/backward share one autograd ``ctx`` with the norm.

    ``ctx.save_for_backward`` may be called only once per autograd Function. In forward the
    proxy records the tensors TE saves so that the Function can save them together with the
    norm's; in backward it serves TE the first ``n`` saved tensors. Every other attribute access
    goes to the real ctx.
    """

    def __init__(self, ctx, saved_tensors=()):
        object.__setattr__(self, "_ctx", ctx)
        object.__setattr__(self, "_saved", tuple(saved_tensors))
        object.__setattr__(self, "recorded", ())

    def save_for_backward(self, *tensors):
        """Record instead of saving; the owning Function saves."""
        object.__setattr__(self, "recorded", tensors)

    @property
    def saved_tensors(self):
        """The tensors TE saved in forward (served back in backward)."""
        return object.__getattribute__(self, "_saved")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_ctx"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_ctx"), name, value)


def _begin_scratch_wgrad(ctx):
    """If the policy routes the weight gradient through the fp32 scratch, point TE's main_grad
    at the scratch (TE then runs the GEMM with beta 0); returns the state ``_end_scratch_wgrad``
    needs, or None when TE accumulates in-kernel."""
    if not (ctx.pa_wgrad_scratch and ctx.fuse_wgrad_accumulation and ctx.weights_requires_grad):
        return None
    origin_weight = ctx.origin_weight_refs[0]()
    assert origin_weight is not None, "weight was removed while fuse_wgrad_accumulation"
    main_grad = ctx.main_grad_funcs[0]()
    # TE accumulates (beta 1) unless this is the first microbatch or the weight asks for an
    # overwrite; reproduce that with the scratch: GEMM with beta 0, then add or copy.
    te_accumulates = not getattr(ctx, "origin_weights_overwrite_main_grad", False) and (
        ctx.is_first_microbatch is None or not ctx.is_first_microbatch
    )
    scratch = _wgrad_scratch(main_grad.shape, main_grad.dtype, main_grad.device)
    ctx.main_grad_funcs = [lambda: scratch]
    ctx.origin_weights_overwrite_main_grad = True  # => beta 0 into the scratch
    return origin_weight, main_grad, scratch, te_accumulates


def _end_scratch_wgrad(state) -> None:
    if state is None:
        return
    origin_weight, main_grad, scratch, te_accumulates = state
    origin_weight.main_grad = main_grad  # TE pointed it at the scratch
    if te_accumulates:
        main_grad.add_(scratch)
    else:
        main_grad.copy_(scratch)


def _te_norm_forward(
    inputmat, ln_weight, ln_bias, eps, normalization, sm_margin, zero_centered_gamma, dtype
):
    """TE's LayerNorm / RMSNorm forward kernel over all rows: ``(out, mu, rsigma)``."""
    norm_fwd = tex.layernorm_fwd if normalization == "LayerNorm" else tex.rmsnorm_fwd
    args = (inputmat, ln_weight) if ln_bias is None else (inputmat, ln_weight, ln_bias)
    return norm_fwd(*args, eps, None, None, TE_DType[dtype], sm_margin, zero_centered_gamma)


class PaddingAwareTELayerNormLinearFunction(torch.autograd.Function):
    """``te.pytorch.LayerNormLinear`` over the live rows of a THD buffer.

    The norm runs TE's kernel over the whole buffer (padding rows are zeros in, finite out);
    the GEMMs are the one-group grouped GEMMs of ``PaddingAwareTELinearFunction`` over the live
    rows. Backward runs TE's grouped backward, zero-fills the padding tail of the norm-output
    gradient (so the norm weight gradients only see live rows) and TE's norm backward.
    """

    @staticmethod
    def forward(ctx, inp, ln_weight, ln_bias, m_rows, non_tensor_args, weight):
        """Norm over all rows, GEMM over the live rows; the output's padding tail is zero."""
        (
            is_first_microbatch,
            fp8,
            input_quantizer,
            weight_quantizer,
            grad_output_quantizer,
            fuse_wgrad_accumulation,
            activation_dtype,
            is_grad_enabled,
            weight_workspace,
            cache_weight,
            skip_fp8_weight_update,
            save_original_input,
            wgrad_scratch,
            norm_args,
        ) = non_tensor_args
        eps, normalization, fwd_sm_margin, bwd_sm_margin, zero_centered_gamma = norm_args
        inp_shape = inp.shape
        inputmat = inp.reshape(-1, inp_shape[-1])
        if inputmat.dtype != activation_dtype:
            inputmat = inputmat.to(activation_dtype)
        if ln_weight.dtype != activation_dtype:
            ln_weight = ln_weight.to(activation_dtype)
        if ln_bias is not None and ln_bias.dtype != activation_dtype:
            ln_bias = ln_bias.to(activation_dtype)
        ln_out, mu, rsigma = _te_norm_forward(
            inputmat,
            ln_weight,
            ln_bias,
            eps,
            normalization,
            fwd_sm_margin,
            zero_centered_gamma,
            activation_dtype,
        )
        if is_grad_enabled:
            # TE derives the dgrad plan (and the weight's transposed quantization) from the
            # GEMM input's requires_grad; the norm backward always needs that gradient.
            ln_out.requires_grad_()
        te_ctx = _SplicedCtx(ctx) if is_grad_enabled else None
        out, new_workspaces = _GroupedLinear._forward_grouped_tensor(
            te_ctx,
            inp=ln_out,
            m_splits=m_rows,
            use_bias=False,
            is_first_microbatch=is_first_microbatch,
            fp8=fp8,
            wgrad_store=None,
            input_quantizers=[input_quantizer],
            weight_quantizers=[weight_quantizer],
            grad_input_quantizers=[None],
            grad_weight_quantizers=[None],
            grad_output_quantizers=[grad_output_quantizer],
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            activation_dtype=activation_dtype,
            is_grad_enabled=is_grad_enabled,
            weight_workspaces=[weight_workspace],
            cache_weight=cache_weight,
            skip_fp8_weight_update=skip_fp8_weight_update,
            save_original_input=save_original_input,
            single_grouped_weight=False,
            single_grouped_bias=False,
            weights=(weight,),
            biases=(None,),
            **_te_grouped_forward_extra_kwargs(weight),
        )
        zero_tail_rows(out.view(-1, out.shape[-1]), m_rows)
        if is_grad_enabled:
            te_saved = te_ctx.recorded
            ctx.save_for_backward(*te_saved, inputmat, mu, rsigma, ln_weight)
            ctx.pa_num_te_saved = len(te_saved)
            ctx.pa_m_rows = m_rows
            ctx.pa_wgrad_scratch = wgrad_scratch
            ctx.pa_norm = (
                normalization,
                bwd_sm_margin,
                zero_centered_gamma,
                inp_shape,
                inp.requires_grad,
                ln_bias is not None,
            )
            ctx.requires_dgrad = True
        return out.view(*inp_shape[:-1], out.shape[-1]), new_workspaces

    @staticmethod
    def backward(ctx, grad_output, _grad_workspaces):
        """TE's grouped backward, then TE's norm backward on the zero-tailed gradient."""
        saved = ctx.saved_tensors
        n_te = ctx.pa_num_te_saved
        inputmat, mu, rsigma, ln_weight = saved[n_te:]
        te_ctx = _SplicedCtx(ctx, saved[:n_te])
        (
            normalization,
            bwd_sm_margin,
            zero_centered_gamma,
            inp_shape,
            inp_requires_grad,
            has_bias,
        ) = ctx.pa_norm
        m_rows = ctx.pa_m_rows
        scratch_state = _begin_scratch_wgrad(te_ctx)
        result = _GroupedLinear._backward_grouped_tensor(te_ctx, grad_output)
        _end_scratch_wgrad(scratch_state)
        d_ln_out, grad_weight = result[0], result[5]
        d_ln_out = d_ln_out.reshape(-1, inp_shape[-1])
        zero_tail_rows(d_ln_out, m_rows)
        if normalization == "LayerNorm":
            dgrad, dgamma, dbeta = tex.layernorm_bwd(
                d_ln_out, inputmat, mu, rsigma, ln_weight, bwd_sm_margin, zero_centered_gamma
            )
        else:
            dgrad, dgamma = tex.rmsnorm_bwd(
                d_ln_out, inputmat, rsigma, ln_weight, bwd_sm_margin, zero_centered_gamma
            )
            dbeta = None
        return (
            dgrad.view(inp_shape) if inp_requires_grad else None,
            dgamma,
            dbeta if has_bias else None,
            None,
            None,
            grad_weight,
        )


def te_layernorm_linear_padding_aware_forward(
    module, inp: torch.Tensor, m_rows: torch.Tensor, is_first_microbatch, policy
):
    """Run a ``te.pytorch.LayerNormLinear`` (no bias, no TE-side communication, norm output not
    returned) with the GEMMs over the live rows. Same contract as
    ``te_linear_padding_aware_forward``: None means "take the regular path"."""
    if (
        isinstance(inp, QuantizedTensorStorage)
        or module.is_debug_iter()
        or is_cpu_offload_enabled()
        or module.return_layernorm_output
        or module.return_layernorm_output_gathered
        or not _te_grouped_path_possible(inp)
    ):
        return None
    is_grad_enabled = torch.is_grad_enabled()
    is_first_microbatch, skip_fp8_weight_update = _graph_capture_flags(is_first_microbatch)

    inp = module.prepare_forward(inp, allow_non_contiguous=False)
    try:
        call = _te_grouped_call(
            module, is_first_microbatch, policy, is_grad_enabled, skip_fp8_weight_update
        )
        if call is None:
            return None
        weight, non_tensor_args, cache_name = call
        norm_args = (
            module.eps,
            module.normalization,
            module.fwd_ln_sm_margin if is_grad_enabled else module.inf_ln_sm_margin,
            module.bwd_ln_sm_margin,
            module.zero_centered_gamma,
        )
        args = (
            inp,
            module.layer_norm_weight,
            getattr(module, "layer_norm_bias", None),
            m_rows,
            non_tensor_args + (norm_args,),
            weight,
        )
        if is_grad_enabled:
            out, new_workspaces = PaddingAwareTELayerNormLinearFunction.apply(*args)
        else:
            out, new_workspaces = PaddingAwareTELayerNormLinearFunction.forward(None, *args)
        _store_weight_workspace(module, cache_name, new_workspaces)
    finally:
        module.end_forward()
    return out


# --------------------------------------------------------------------------------------------
# Grouped output projection of DSv4 attention: out[t, g, :] = x[t, g, :] @ W[g]^T
# --------------------------------------------------------------------------------------------
#
# ``torch.einsum("...gd,grd->...gr", x, W)`` is a batched GEMM over G groups whose inputs are
# strided views x[:, g, :] of one [rows, G, d] buffer. With Transformer Engine's grouped tensors
# carrying a uniform ``leading_dim`` the three GEMMs (forward, dgrad, wgrad) run as G-group grouped
# GEMMs over the live rows only, with no permute copies.


def _strided_groups(t: torch.Tensor, m_rows: torch.Tensor, offsets: torch.Tensor):
    """Members ``t[:m, g, :]`` of a contiguous ``[rows, G, cols]`` tensor (``offsets`` = g *
    cols)."""
    rows, groups, cols = t.shape
    return GroupedTensorStorage(
        shape=(rows * groups, cols),
        dtype=t.dtype,
        num_tensors=groups,
        quantizer=None,
        data=t.reshape(-1),
        first_dims=m_rows.expand(groups).contiguous(),
        tensor_offsets=offsets,
        leading_dim=groups * cols,
    )


def _packed_groups(w: torch.Tensor):
    """Members ``w[g]`` of a contiguous ``[G, N, K]`` tensor."""
    groups, n, k = w.shape
    return GroupedTensorStorage(
        shape=(groups * n, k),
        dtype=w.dtype,
        num_tensors=groups,
        shapes=[(n, k)] * groups,
        quantizer=None,
        data=w.reshape(-1),
    )


def _group_offsets(groups: int, cols: int, device) -> torch.Tensor:
    """Element offsets of the members ``t[:, g, :]`` of a ``[rows, G, cols]`` tensor."""
    return torch.arange(groups + 1, device=device, dtype=torch.int64) * cols


class PaddingAwareGroupedProjection(torch.autograd.Function):
    """``einsum("tgd,grd->tgr", x, W)`` over the first ``m_rows`` rows (device scalar); the
    padding tail of the output and of the input gradient is zero. bf16/fp16 only."""

    @staticmethod
    def forward(ctx, x, weight, m_rows):
        """Grouped GEMM over the live rows of every group."""
        rows, groups, in_dim = x.shape
        out_dim = weight.shape[1]
        out = torch.empty(rows, groups, out_dim, dtype=x.dtype, device=x.device)
        x_offsets = _group_offsets(groups, in_dim, x.device)
        out_offsets = _group_offsets(groups, out_dim, x.device)
        general_grouped_gemm_for_grouped_tensor(
            _packed_groups(weight),
            _strided_groups(x, m_rows, x_offsets),
            _strided_groups(out, m_rows, out_offsets),
            layout="TN",
        )
        zero_tail_rows(out.view(rows, groups * out_dim), m_rows)
        ctx.save_for_backward(x, weight, m_rows, x_offsets, out_offsets)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """dgrad and wgrad as grouped GEMMs over the live rows."""
        x, weight, m_rows, x_offsets, out_offsets = ctx.saved_tensors
        rows, groups, in_dim = x.shape
        out_dim = weight.shape[1]
        grad_out = grad_out.contiguous()
        grad_x = grad_w = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty_like(x)
            general_grouped_gemm_for_grouped_tensor(
                _packed_groups(weight),
                _strided_groups(grad_out, m_rows, out_offsets),
                _strided_groups(grad_x, m_rows, x_offsets),
                layout="NN",
            )
            zero_tail_rows(grad_x.view(rows, groups * in_dim), m_rows)
        if ctx.needs_input_grad[1]:
            grad_w = torch.empty_like(weight)
            general_grouped_gemm_for_grouped_tensor(
                _strided_groups(x, m_rows, x_offsets),
                _strided_groups(grad_out, m_rows, out_offsets),
                _packed_groups(grad_w),
                layout="NT",
            )
        return grad_x, grad_w, None


def grouped_projection_is_padding_aware(config, groups: int, out_dim: int, in_dim: int) -> bool:
    """Static decision for the grouped output projection (model shape only)."""
    if not getattr(config, "padding_aware_dense_linear", False):
        return False
    if not (HAVE_TE_GROUPED_LEADING_DIM and HAVE_TRITON):
        return False
    if in_dim % 32 or out_dim % 32:
        return False
    capacity = static_capacity_rows(config)
    if capacity is None or capacity < MIN_CAPACITY_ROWS:
        return False
    if os.environ.get("MCORE_PADDING_AWARE_FORCE", "0") == "1":
        return True
    if (
        not training_uses_cuda_graphs(config)
        and os.environ.get("MCORE_PADDING_AWARE_EAGER", "0") != "1"
    ):
        return False
    assumed_padding = float(getattr(config, "padding_aware_assumed_padding", ASSUMED_PADDING))
    saving, overhead = predicted_padding_aware_gain(
        groups * out_dim, in_dim, capacity, False, False, assumed_padding
    )
    enabled = saving >= POLICY_MARGIN * overhead
    log_policy_decision(
        f"grouped projection ({groups} groups)",
        groups * out_dim,
        in_dim,
        enabled,
        f"predicted saving {saving * 1e6:.0f} us vs {POLICY_MARGIN} x overhead "
        f"{overhead * 1e6:.0f} us at {assumed_padding:.0%} padding",
    )
    return enabled


def padding_aware_grouped_projection(x: torch.Tensor, weight: torch.Tensor, m_rows: torch.Tensor):
    """``einsum("...gd,grd->...gr", x, weight)`` over the live rows of a ``[s, b, G, d]`` input."""
    lead = x.shape[:-2]
    x3 = x.reshape(-1, x.shape[-2], x.shape[-1])
    out = PaddingAwareGroupedProjection.apply(x3, weight, m_rows)
    return out.view(*lead, out.shape[-2], out.shape[-1])
