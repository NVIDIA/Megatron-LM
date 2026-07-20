# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generalized Tensor Parallelism (GTP).

GTP factors the weight-parallel domain into ``TP x GTP_remat`` (two orthogonal
sub-axes). A weight is sharded ``1/(TP * GTP_remat)`` along its partition dim:

* The ``TP`` slice stays sharded through the GEMM — ordinary tensor parallelism;
  the output is TP-sharded and reduced/gathered as usual.
* The ``GTP_remat`` slice is *rematerialized* just before the GEMM: only the
  ``gtp_remat`` sub-group async all-gathers its part, so each rank's GEMM sees
  the full TP slice. This trades extra all-gather traffic for ``1/GTP_remat``
  lower weight (and optimizer/grad) memory — ZeRO-3-on-the-weight on top of TP.

``GTP_remat`` (the rematerialization sub-axis) has degree ``gtp_weight_remat_size``,
derived from ``--tensor-parallel-num-weight-shards``.

Materialization uses a per-weight prefetch chain + ticket-based buffer cache
co-designed for CUDA graph capture/replay. Quantized AG (FP8 / MXFP8 / NVFP4)
composes with the sharding for compounding bandwidth reduction.

See ``docs/api-guide/core/generalized_tensor_parallel.md`` for design and usage.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
from packaging.version import Version

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)

_GTP_TE_MIN_VERSION = Version("2.17")

try:
    import transformer_engine as te  # noqa: F401

    _te_version = Version(te.__version__)
    if _te_version < _GTP_TE_MIN_VERSION and not os.environ.get("MEGATRON_GTP_FORCE_ENABLE"):
        raise ImportError(
            f"megatron.core.tensor_parallel.gtp requires TransformerEngine "
            f">= {_GTP_TE_MIN_VERSION} (found {_te_version}). Set MEGATRON_GTP_FORCE_ENABLE=1 "
            "to bypass this check when using a custom TE build with the GTP hook registry."
        )

    import transformer_engine_torch as tex
    from transformer_engine.pytorch.constants import (
        MXFP8_BLOCK_SCALING_SIZE,
        NVFP4_BLOCK_SCALING_SIZE,
    )
    from transformer_engine.pytorch.distributed import (
        _NVFP4AllGatherAsyncHandle,
        gather_along_first_dim,
        in_fp8_activation_recompute_phase,
        reduce_scatter_along_first_dim,
    )
    from transformer_engine.pytorch.module.base import get_dummy_wgrad
    from transformer_engine.pytorch.quantized_tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor import MXFP8TensorStorage, NVFP4TensorStorage
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
    from transformer_engine.pytorch.utils import (
        nvtx_range_pop,
        nvtx_range_push,
        round_up_to_nearest_multiple,
    )

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # TE unavailable/too old -> stub the TE-backed names so this module still imports,
    # and flag GTP unusable via HAVE_TE (gtp.py surfaces this as HAVE_GTP=False). No
    # GTP path runs without TE. The `annotations` future-import keeps the lone
    # module-level TE reference (a dataclass field annotation) from being evaluated.
    from unittest.mock import MagicMock

    te = tex = MagicMock()
    MXFP8_BLOCK_SCALING_SIZE = NVFP4_BLOCK_SCALING_SIZE = None
    _NVFP4AllGatherAsyncHandle = MagicMock()
    gather_along_first_dim = reduce_scatter_along_first_dim = MagicMock()
    in_fp8_activation_recompute_phase = MagicMock()
    get_dummy_wgrad = MagicMock()
    QuantizedTensor = MagicMock()
    MXFP8TensorStorage = NVFP4TensorStorage = MagicMock()
    MXFP8Quantizer = MagicMock()
    nvtx_range_pop = nvtx_range_push = round_up_to_nearest_multiple = MagicMock()
    HAVE_TE = False


class GTPChain(str, Enum):
    """Prefetch chain identifier for n GTPShardedParam.

    GRAPHED   — fwd/bwd captured by a CUDA graph (MLM _CudaGraphRunner).
    UNGRAPHED — fwd/bwd runs eagerly.

    Chains never cross-link (prev_w/next_w stay within one chain). See
    _classify_param_chain for the GRAPHED/UNGRAPHED rule.
    """

    GRAPHED = "GTP_graphed"
    UNGRAPHED = "GTP_ungraphed"


# Active cuda_graph config, set by the integrator via set_cuda_graph_modules() before
# classify_gtp_chains(); consumed by _classify_param_chain.
_CUDA_GRAPH_MODULES: Optional[set] = None  # scope tags, e.g. {"mamba","attn","moe_router"}
_MOE_SHARED_EXPERT_OVERLAP: bool = False  # overlapped shared_experts can't be captured -> UNGRAPHED
_FULL_ITERATION: bool = False  # whole step in one graph -> every param GRAPHED
# Empty cuda_graph_modules under per-layer CG = "graph every layer" == all tags present.
_ALL_LAYER_SCOPE_TAGS = frozenset({"mamba", "attn", "moe", "moe_router"})


def set_cuda_graph_modules(
    scope, moe_shared_expert_overlap: bool = False, cuda_graph_impl: str = "none"
):
    """Record the active cuda_graph config for GTP chain classification.

    Called by MLM at init, before classify_gtp_chains(). ``cuda_graph_impl``
    disambiguates the empty-``scope`` cases:
      - "none"           -> CG disabled; all params UNGRAPHED.
      - "full_iteration" -> whole step in one graph; all params GRAPHED.
      - "local"/"transformer_engine" + empty scope -> graph every layer.
    """
    global _CUDA_GRAPH_MODULES, _MOE_SHARED_EXPERT_OVERLAP, _FULL_ITERATION
    _MOE_SHARED_EXPERT_OVERLAP = bool(moe_shared_expert_overlap)
    _FULL_ITERATION = cuda_graph_impl == "full_iteration"
    if _FULL_ITERATION:
        _CUDA_GRAPH_MODULES = None  # scope unused
    elif cuda_graph_impl != "none" and not scope:
        _CUDA_GRAPH_MODULES = set(_ALL_LAYER_SCOPE_TAGS)  # graph every layer
    else:
        _CUDA_GRAPH_MODULES = set(scope) if scope else None


def _classify_param_chain(param_name: str) -> "GTPChain":
    """Map a GTPShardedParam name + active cuda_graph config to its chain.

    Full-iteration -> GRAPHED. Otherwise embedding/output_layer are UNGRAPHED, and
    each layer kind (mixer, attention, shared/routed experts) is GRAPHED iff its
    scope tag is in cuda_graph_modules.
    """
    n = param_name

    if _FULL_ITERATION:
        return GTPChain.GRAPHED

    # embedding/output_layer live outside any per-layer CG runner.
    if "embedding" in n or "output_layer" in n:
        return GTPChain.UNGRAPHED

    scope = _CUDA_GRAPH_MODULES
    if not scope:  # CG disabled
        return GTPChain.UNGRAPHED

    if ".mlp.shared_experts." in n:
        if _MOE_SHARED_EXPERT_OVERLAP:
            return GTPChain.UNGRAPHED
        return GTPChain.GRAPHED if ("moe" in scope or "moe_router" in scope) else GTPChain.UNGRAPHED

    if ".mlp.experts." in n:
        return GTPChain.GRAPHED if "moe" in scope else GTPChain.UNGRAPHED

    if ".self_attention." in n or ".cross_attention." in n:
        return GTPChain.GRAPHED if "attn" in scope else GTPChain.UNGRAPHED

    if ".mixer." in n:
        return GTPChain.GRAPHED if "mamba" in scope else GTPChain.UNGRAPHED

    return GTPChain.UNGRAPHED


def classify_gtp_chains(model) -> None:
    """Walk model.named_parameters() and set chain_id on every GTPShardedParam.

    Call once at init, AFTER set_cuda_graph_modules() and BEFORE the first fwd of any
    graphed param. Raises if an already-initialized param would be reclassified into a
    different chain (its prev/next links are already wired into the wrong list).
    """
    conflicts = []
    for name, param in model.named_parameters():
        if not is_gtp_param(param):
            continue
        target = _classify_param_chain(name).value
        if param.prefetch_initialized and param.chain_id != target:
            conflicts.append((name, param.chain_id, target))
            continue
        param.chain_id = target

        # Bwd-prefetch opt-out: embedding weight needs no bwd AG (wgrad is a
        # scatter-add on sharded rows, input has no dgrad) — saves one collective.
        if "embedding" in name:
            param._need_weight_prefetch_bwd = False
    if conflicts:
        raise RuntimeError(
            "classify_gtp_chains: the following params were already chain-initialized "
            "with a different chain_id than the classifier would assign — this means "
            "their chain links are already wired into the wrong list. Move classification "
            "earlier in init. Conflicts: "
            + ", ".join(f"{n}: {old!r}->{new!r}" for n, old, new in conflicts[:3])
            + ("..." if len(conflicts) > 3 else "")
        )


class GTPWeightState(Enum):
    """State of a GTPShardedParam's AG / RS lifecycle (debug / stale-read guard)."""

    NONE = "NONE"  # Sharded, no pending operation
    ASYNC_WAIT = "ASYNC_WAIT"  # Async all-gather in progress
    DATA_READY = "DATA_READY"  # Async all-gather complete, result in cache
    DATA_READY_SYNC = "DATA_READY_SYNC"  # Sync all-gather complete, result in cache


# Global GTP buffer cache (persists across clear(); never set to None after creation).
_GTP_CACHE = None
_GTP_PARAMS = []

# Global set of GTPShardedParam with in-flight async comms (AG or RS).
_inflight_comm_params: set = set()
_AG_STREAMS: Dict[str, torch.cuda.Stream] = {}
_RS_STREAMS: Dict[str, torch.cuda.Stream] = {}

# Wgrad input buffer pool, keyed by (shape, dtype). UNGRAPHED-only: GRAPHED
# wgrad bufs need address stability for CG replay and are not pool-recycled.
_wgrad_buf_pool: Dict[tuple, list] = {}


def _wgrad_pool_get(shape: tuple, dtype: torch.dtype, device) -> torch.Tensor:
    """Get a pool buffer or allocate fresh, tagged so _wgrad_pool_put accepts only
    pool-owned buffers (other callers fall through to the caching allocator on release)."""
    key = (shape, dtype)
    pool = _wgrad_buf_pool.get(key)
    if pool:
        buf = pool.pop()
    else:
        buf = torch.empty(shape, dtype=dtype, device=device, requires_grad=False)
    buf._from_gtp_wgrad_pool = True
    return buf


def _wgrad_pool_put(buf: torch.Tensor):
    """Return a pool-owned buffer for reuse (no-op for untagged buffers; see
    _wgrad_pool_get)."""
    if not getattr(buf, "_from_gtp_wgrad_pool", False):
        return
    key = (tuple(buf.shape), buf.dtype)
    if key not in _wgrad_buf_pool:
        _wgrad_buf_pool[key] = []
    _wgrad_buf_pool[key].append(buf)


def _stream_key(chain_id: str, group) -> tuple:
    """Key for the per-(chain, group) AG/RS stream dicts.

    Partitioned on two axes: chain_id (captured GRAPHED vs eager UNGRAPHED ops must not
    share a stream) and group (independent NCCL, e.g. GTP_remat vs EGTP_remat, no serialization).
    """
    return (chain_id, id(group) if group is not None else 0)


def get_ag_stream(chain_id: str = GTPChain.GRAPHED.value, group=None) -> torch.cuda.Stream:
    """Return the GTP all-gather stream for (chain_id, group). See _stream_key."""
    key = _stream_key(chain_id, group)
    if key not in _AG_STREAMS:
        _AG_STREAMS[key] = torch.cuda.Stream()
    return _AG_STREAMS[key]


def get_rs_stream(chain_id: str = GTPChain.GRAPHED.value, group=None) -> torch.cuda.Stream:
    """Return the GTP reduce-scatter stream for (chain_id, group). See _stream_key."""
    key = _stream_key(chain_id, group)
    if key not in _RS_STREAMS:
        _RS_STREAMS[key] = torch.cuda.Stream()
    return _RS_STREAMS[key]


def wait_for_gtp_grad_reduction_on_current_stream() -> None:
    """Fence the current stream against all GTP backward grad work before the DP gradient sync.

    Drains the eager AG/RS side streams, then waits on each CG runner's replay stream
    (its tail = captured Phase 2 main_grad.add_). No-op when GTP is inactive.
    """
    wait_async_comms()
    cur = torch.cuda.current_stream()
    for s in _AG_STREAMS.values():
        cur.wait_stream(s)
    for s in _RS_STREAMS.values():
        cur.wait_stream(s)
    # Local import: cuda_graphs imports this module, so a module-level import would be circular.
    from megatron.core.transformer.cuda_graphs import get_gtp_runner_streams

    for s in get_gtp_runner_streams():
        cur.wait_stream(s)


@dataclass
class GTPRematConfig:
    """Global configuration for Generalized Tensor Parallelism (weight remat)."""

    pad_for_alignment: int = 16
    check_param_states: bool = False
    weight_prefetch: bool = True
    # True (default): non-chain-head wgrad RS is async_op=True and finalizes
    # (handle.wait + main_grad.add_) in a later bwd's cascade walk, overlapping RS with
    # compute. False: every wgrad RS is synchronous + inline (no overlap).
    async_reduction: bool = True
    # Mirrors config.calculate_per_token_loss. When True, DDP applies NO 1/dp pre-scaling
    # (gradient_scaling_factor=1.0) and finalize_model_grads normalizes every gradient by
    # 1/total_global_tokens instead. In that mode the gtp_remat axis must be SUM-reduced (plain
    # reduce-scatter, like DP), NOT mean-reduced — a 1/gtp mean would double-count the
    # normalization. When False, the gtp_remat reduce-scatter takes the MEAN so it composes with
    # DDP's 1/replicate scaling to yield the full (replicate x gtp) mean.
    calculate_per_token_loss: bool = False


GTP_CONFIG = GTPRematConfig()


def update_gtp_config(**kwargs):
    """Update the global GTP configuration."""
    for key, value in kwargs.items():
        if not hasattr(GTP_CONFIG, key):
            raise ValueError(f"Unknown GTP config option: {key}")
        setattr(GTP_CONFIG, key, value)


def tag_gtp_params_with_names(model):
    """Populate _debug_name on every GTPShardedParam with its full dotted parameter name.

    Call once after model construction so the linking log prints human-readable names
    instead of raw tensor ids.
    """
    for name, param in model.named_parameters():
        if is_gtp_param(param):
            param._debug_name = name


def configure_gtp_remat_from_recipe(
    *, fp4=False, fp8_recipe=None, fp8=False, calculate_per_token_loss=False
):
    """
    Configure GTP weight-remat (padding + loss reduction) from the quantization recipe.
    Must be called once BEFORE model construction.
    """
    # gtp_remat grad reduction SUMs (not means) the gtp_remat axis under per-token-loss.
    # check_param_states=False: GTP buffer reuse (notably under CUDA-graph capture) trips the
    # param-state debug asserts, so keep them off for GTP runs.
    update_gtp_config(calculate_per_token_loss=calculate_per_token_loss, check_param_states=False)
    if fp4:
        update_gtp_config(pad_for_alignment=16)
    elif fp8_recipe == "mxfp8":
        update_gtp_config(pad_for_alignment=32)
    elif fp8:
        update_gtp_config(pad_for_alignment=16)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info("> GTP_remat enabled. %s", GTP_CONFIG)


def classify_gtp_remat_chains(
    model, *, cuda_graph_modules=None, moe_shared_expert_overlap=False, cuda_graph_impl="none"
):
    """
    Tag and classify every GTP param's prefetch chain (GRAPHED vs UNGRAPHED).
    Must be called once AFTER model build + DDP wrap and before the first forward (which
    lazily builds chain links).
    """
    cg_modules = (
        {getattr(s, "name", str(s)) for s in cuda_graph_modules} if cuda_graph_modules else None
    )
    set_cuda_graph_modules(
        cg_modules,
        moe_shared_expert_overlap=moe_shared_expert_overlap,
        cuda_graph_impl=cuda_graph_impl,
    )
    # Clear stale process-global chain state so a rebuilt model starts fresh.
    reset_gtp_state()
    for model_module in model if isinstance(model, list) else [model]:
        tag_gtp_params_with_names(model_module)
        classify_gtp_chains(model_module)


def gtp_remat_shard_dim0(dim0, gtp_remat_group):
    """Return ``(shard_dim0, pad_length)`` for allocating a dim-0 GTP weight-remat shard."""
    gtp_remat_size = gtp_remat_group.size()
    if GTP_CONFIG.pad_for_alignment > 0:
        alignment = GTP_CONFIG.pad_for_alignment * gtp_remat_size
        pad_length = (alignment - dim0 % alignment) % alignment
    else:
        assert dim0 % gtp_remat_size == 0, (
            f"gtp_remat_shard_dim0: dim0={dim0} not divisible by gtp_remat_size={gtp_remat_size}. "
            "Enable padding (GTP_CONFIG.pad_for_alignment > 0) or make dim-0 a multiple of the "
            "GTP group size."
        )
        pad_length = 0
    padded = dim0 + pad_length
    return padded // gtp_remat_size, pad_length


def _gtp_slice_one_param(param, gtp_remat_group, *, name="<unnamed>"):
    """Pad + slice a full-size BF16 weight to this rank's GTP shard.

    Caller attaches GTP attrs (see _gtp_attach_attrs). On the legacy post-init path under
    fp8_model_init, tensor may be a QuantizedTensor — F.pad dequantizes it before slicing.
    """
    gtp_remat_size = gtp_remat_group.size()
    gtp_rank = gtp_remat_group.rank()
    tensor = param.data

    if GTP_CONFIG.pad_for_alignment > 0:
        # Pad before slicing so shards stay alignment-divisible and padding
        # ends up contiguous at the tail of the gathered result.
        alignment = GTP_CONFIG.pad_for_alignment * gtp_remat_size
        dim0 = tensor.shape[0]
        pad_length = (alignment - dim0 % alignment) % alignment
        if pad_length > 0:
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_length))
    else:
        # No-pad mode: dim-0 must divide gtp_remat_size or AG output loses tail rows.
        assert tensor.shape[0] % gtp_remat_size == 0, (
            f"_gtp_slice_one_param: {name}.shape[0]={tensor.shape[0]} is not "
            f"divisible by gtp_remat_size={gtp_remat_size}. Either enable padding by "
            "setting GTP_CONFIG.pad_for_alignment > 0, or ensure the weight's "
            "dim-0 is a multiple of the GTP group size."
        )
        pad_length = 0

    shard_size = tensor.shape[0] // gtp_remat_size
    shard = tensor[gtp_rank * shard_size : (gtp_rank + 1) * shard_size]
    gtp_shard = GTPShardedParam(shard.clone())
    gtp_shard.pad_length = pad_length
    # Preserve the source weight's TP attributes (dropped when wrapping into GTPShardedParam),
    # so param_is_not_tensor_parallel_duplicate still classifies it without GTP-specific code.
    from megatron.core.tensor_parallel import copy_tensor_model_parallel_attributes

    copy_tensor_model_parallel_attributes(gtp_shard, param)
    return gtp_shard


def _gtp_attach_attrs(gtp_shard, gtp_remat_group, *, is_grouped=False, expert_idx=0):
    """Attach group / gtp_remat_size / routed-expert tags and register in _GTP_PARAMS.

    Separate from _gtp_slice_one_param so attrs land on the post-quantize param (when
    quantize fires between slice and attach).
    """
    # DistributedWeight requires implementers stay torch.Tensor subclasses; enforce at construction.
    assert isinstance(gtp_shard, torch.Tensor), (
        "GTP param must remain a torch.Tensor subclass (DistributedWeight requirement); got "
        f"{type(gtp_shard).__name__}."
    )
    if is_grouped:
        gtp_shard.expert_idx = expert_idx
        gtp_shard.is_routed_expert = True
        # Default to UNGRAPHED; classify_gtp_chains() reclassifies based on the
        # cuda_graph_modules at init time.
        gtp_shard.chain_id = GTPChain.UNGRAPHED.value
    gtp_shard.group = gtp_remat_group
    gtp_shard.gtp_remat_size = gtp_remat_group.size()
    global _GTP_PARAMS
    _GTP_PARAMS.append(gtp_shard)


def _gtp_wrap_bf16_shard(module, name, param):
    """Re-register a BF16 pre-sharded weight as a :class:`GTPShardedParam`.

    The weight already IS this rank's shard (built pre-sharded), so — unlike the post-init path
    :func:`_gtp_slice_one_param`, which slices a full weight — this only wraps it, no slicing.
    Returns the new param (also swapped into the module).
    """
    from megatron.core.tensor_parallel import copy_tensor_model_parallel_attributes

    gtp_shard = GTPShardedParam(param.data)
    copy_tensor_model_parallel_attributes(gtp_shard, param)
    delattr(module, name)
    module._parameters[name] = gtp_shard
    return gtp_shard


def _gtp_reclass_native_fp8_shard(param):
    """Reclass a native-FP8 pre-sharded weight into a GTP subclass in place (buffer-resident).

    The dynamic ``GTP_<Fp8TensorClass>`` subclass carries GTPShardedParam's gather/RS methods while
    keeping ``is_float8tensor`` True (so DDP/distopt keep it buffer-resident) and TE's forward can
    call ``weight.all_gather_and_prefetch()``. The param IS this rank's FP8 shard (``quantized`` is
    itself; never re-quantized). Returns the same (mutated) param.
    """
    # Preserve the param's own _quantizer (used by TE ops + the optimizer copy_->quantize_ update);
    # _init_gtp_runtime_attrs clears it, so stash/restore.
    native_quantizer = getattr(param, "_quantizer", None)
    param.__class__ = _gtp_native_fp8_subclass(type(param))
    _init_gtp_runtime_attrs(param)
    param._gtp_native_fp8 = True
    param._quantizer = native_quantizer
    # Gather uses a SEPARATE quantizer copy for its per-direction set_usage; reusing the param's own
    # would leave rowwise=False after a bwd gather, freezing the rowwise data the forward reads. The
    # copy also keeps MXFP8 scales compact for byte-concat scale all-gather.
    gather_q = None
    if native_quantizer is not None:
        gather_q = native_quantizer.copy()
        gather_q.internal = False
        gather_q.optimize_for_gemm = not isinstance(gather_q, MXFP8Quantizer)
    param._gtp_gather_quantizer = gather_q
    param.quantized = param
    return param


def attach_gtp_to_presharded_module(module, gtp_remat_group, pad_length, is_grouped=False):
    """Turn each pre-sharded weight into a GTP param (FP8/BF16) and attach GTP wiring."""
    # GTP shards per-expert weight0..weight{num_gemms-1}; a coalesced single weight has no sibling
    # shards to attach, so reject it here (once, at setup) instead of silently attaching nothing.
    if is_grouped:
        assert not getattr(
            module, "single_grouped_weight", False
        ), f"GTP grouped module {type(module).__name__} requires single_grouped_weight=False."
    # Use the module's weight_names if it declares them; otherwise create them (grouped modules
    # expose per-expert weight0..weight{num_gemms-1}, non-grouped a single "weight").
    weight_names = getattr(module, "weight_names", None)
    if not weight_names:
        weight_names = (
            [f"weight{idx}" for idx in range(module.num_gemms)] if is_grouped else ["weight"]
        )
    new_weights = []
    for idx, name in enumerate(weight_names):
        param = getattr(module, name, None)
        if param is None or is_gtp_param(param):
            continue
        if isinstance(param, QuantizedTensor):
            gtp_param = _gtp_reclass_native_fp8_shard(param)
        else:
            gtp_param = _gtp_wrap_bf16_shard(module, name, param)
        gtp_param.pad_length = pad_length
        _gtp_attach_attrs(gtp_param, gtp_remat_group, is_grouped=is_grouped, expert_idx=idx)
        new_weights.append(gtp_param)
    if is_grouped and new_weights:
        new_weights[0].weight_list = new_weights


# Cache of dynamic ``GTP_<Fp8TensorClass>`` subclasses, keyed by the FP8 base class.
_GTP_NATIVE_FP8_SUBCLASSES: Dict[type, type] = {}

# GTPShardedParam members NOT copied into the dynamic ``GTP_<Fp8Class>`` subclass (and why):
#   - __new__ / __init__: construction hooks — we only *reclass* an existing FP8 instance, never
#     construct one; GTP attrs are set afterwards by _init_gtp_runtime_attrs.
#   - __torch_function__: keep the FP8 tensor's OWN tensor dispatch, not GTPShardedParam's.
#   - __dict__ / __weakref__ / __module__ / __doc__ / __qualname__ / __slots__: per-class machinery
#     for GTPShardedParam itself; copying it would corrupt the new subclass's identity/layout.
_GTP_SUBCLASS_SKIP = frozenset(
    {
        "__new__",
        "__init__",
        "__torch_function__",
        "__dict__",
        "__weakref__",
        "__module__",
        "__doc__",
        "__qualname__",
        "__slots__",
    }
)


def _gtp_native_fp8_subclass(base_cls: type) -> type:
    """Cached ``base_cls`` subclass + GTPShardedParam's methods (isinstance(base_cls) kept True).

    CRITICAL: skip names already in the FP8 MRO — a GTPShardedParam method shadowing one TE needs
    (e.g. get_data_tensors) silently freezes optimizer updates.
    """
    sub = _GTP_NATIVE_FP8_SUBCLASSES.get(base_cls)
    if sub is None:
        base_mro_names = set()
        for klass in base_cls.__mro__:
            base_mro_names.update(vars(klass).keys())
        ns = {
            k: v
            for k, v in vars(GTPShardedParam).items()
            if k not in _GTP_SUBCLASS_SKIP and k not in base_mro_names
        }
        # Share the (class-level) prefetch chain-state dicts with GTPShardedParam.
        ns["_chain_state"] = GTPShardedParam._chain_state
        ns["_recompute_chain_state"] = GTPShardedParam._recompute_chain_state
        sub = type(f"GTP_{base_cls.__name__}", (base_cls,), ns)
        _GTP_NATIVE_FP8_SUBCLASSES[base_cls] = sub
    return sub


def is_gtp_param(param) -> bool:
    """True if ``param`` is a GTP weight-remat shard (BF16 or native-FP8)."""
    return getattr(param, "is_gtp_weight_remat", False)


def dequantize_gtp_native_fp8(param):
    """Dequantize a native-FP8 GTP param to a plain BF16 tensor (used at the checkpoint boundary).

    TE's ``tex.dequantize`` dispatches on the *exact* FP8 class and rejects our dynamic
    ``GTP_<Fp8TensorClass>`` subclass, so restore the base FP8 class for the call and reclass after.
    """
    from megatron.core.fp8_utils import dequantize_fp8_tensor

    sub_cls = type(param)
    base_cls = sub_cls.__mro__[1]  # _gtp_native_fp8_subclass builds type("GTP_X", (base_cls,), ...)
    param.__class__ = base_cls
    try:
        return dequantize_fp8_tensor(param)
    finally:
        param.__class__ = sub_cls


@contextmanager
def gtp_native_fp8_load_context(module):
    """Restore the base FP8 class on native-FP8 GTP params under ``module`` for a load copy.

    ``load_state_dict`` does ``param.copy_(bf16)`` -> TE ``convert_and_update_tensor``, whose
    ``IsMXFP8Tensor`` C++ check rejects our dynamic subclass (the load-side twin of
    :func:`dequantize_gtp_native_fp8`). Presenting the base class lets TE re-quantize into the FP8
    storage; instance attrs live in ``__dict__`` and survive the swap, so the GTP surface persists.
    """
    from megatron.core.fp8_utils import is_float8tensor

    swapped = []
    for param in module.parameters(recurse=True):
        if is_gtp_param(param) and is_float8tensor(param):
            sub_cls = type(param)
            base_cls = sub_cls.__mro__[1]
            if base_cls is not sub_cls:
                param.__class__ = base_cls
                swapped.append((param, sub_cls))
    try:
        yield
    finally:
        for param, sub_cls in swapped:
            param.__class__ = sub_cls


def wrap_module_params_gtp(module, weight_names, gtp_remat_group, is_grouped=None):
    """Shard and re-register module params as GTPShardedParam (post-init slice).

    Called post-init for Megatron-style local modules (ColumnParallelLinear, etc.), which build
    the full weight and slice it here. TE modules do NOT use this path — they are constructed
    already-shard-sized (GTP-agnostic init) and wired via :func:`attach_gtp_to_presharded_module`.
    Params that are already GTP are skipped.
    """
    if gtp_remat_group.size() == 1:
        return

    for idx, name in enumerate(weight_names):
        param = getattr(module, name, None)
        if param is None:
            continue

        # Already a GTP param (TE-side slice, or native-FP8 attach) — skip.
        if is_gtp_param(param):
            continue

        # delete the original parameter, which will be replaced by an GTP sharded one
        delattr(module, name)
        gtp_shard = _gtp_slice_one_param(param, gtp_remat_group, name=name)
        del param
        _gtp_attach_attrs(gtp_shard, gtp_remat_group, is_grouped=bool(is_grouped), expert_idx=idx)
        # register the newly sharded param back to the module
        module._parameters[name] = gtp_shard

    if is_grouped:
        allweights = [getattr(module, name) for name in weight_names]
        allweights[0].weight_list = allweights


class GTPShardHandle:
    """Wrapper around a ``dist`` async-work handle for a GTP AG / RS.

    Tracks the participating shards so the wait-site can transition their GTPWeightState
    and prune the param from _inflight_comm_params when the collective completes.
    """

    def __init__(self, handle, gtp_shards, reduce_scatter=False):
        self.handle = handle
        self.gtp_shards = gtp_shards
        self.reduce_scatter = reduce_scatter
        _inflight_comm_params.add(gtp_shards[0])

    def wait(self):
        """Wait on the underlying NCCL work and update the shards' state."""
        if self.handle is not None:
            self.handle.wait()
            self.handle = None  # Release NCCL Work and its C++ tensor references promptly
        if GTP_CONFIG.check_param_states:
            for w in self.gtp_shards:
                if self.reduce_scatter:
                    w._set_rs_state(GTPWeightState.DATA_READY)
                else:
                    w._set_state(GTPWeightState.DATA_READY)

        _inflight_comm_params.discard(self.gtp_shards[0])


def _init_gtp_runtime_attrs(obj):
    """Initialize the full GTP runtime-state attribute surface on ``obj``.

    Shared by :meth:`GTPShardedParam.__init__` (legacy BF16 slice path) and
    :func:`attach_gtp_to_presharded_module` (native-FP8 reclass path), so both param-class
    representations carry identical state. chain_id/group are set by the caller afterward.
    """
    # Canonical flag — also set on distopt's main_param copy so both kinds
    # of param can be classified via a single attribute check.
    obj.is_gtp_weight_remat = True
    # all gather
    obj.state = GTPWeightState.NONE
    obj._ag_ticket_fwd = None
    obj._ag_ticket_bwd = None
    obj._prefetch_handle = None
    obj._need_weight_prefetch = True
    # Per-direction prefetch opt-outs (default True). The embedding weight needs no bwd AG
    # (wgrad is a token-indexed scatter-add, input non-differentiable). classify_gtp_chains()
    # sets this False for embedding.word_embeddings.weight.
    obj._need_weight_prefetch_bwd = True
    obj.ag_event = torch.cuda.Event(external=True)
    # DDP backward hook (set by register_grad_accum_hook); invoked after
    # the wgrad RS accumulation completes (Graphed.backward / chain cascade).
    obj._grad_accum_hook = None
    # Quantization. For native-FP8 GTP the reclass path overwrites _quantizer with the tensor's
    # own MXFP8 quantizer and points quantized at self; BF16 GTP leaves both unset.
    obj._quantizer = None
    obj.quantized = None
    # Prefetching linked list
    obj.prefetch_initialized = False
    obj.next_w = None
    obj.prev_w = None
    # Recompute-forward prefetch chain: a SEPARATE chain (own slot) for weights re-gathered
    # rowwise during an activation-recompute forward in backward. Distinct from the
    # state/_prefetch_handle/ag_event above so it never clobbers the concurrent columnwise
    # dgrad lifecycle. Self-populates from the first backward's recompute gathers.
    obj._recompute_initialized = False
    obj._recompute_next = None
    obj._recompute_prev = None
    obj._recompute_prefetch_handle = None
    obj._recompute_ag_event = torch.cuda.Event(external=True)
    obj._recompute_already_drained = False
    # Chain identity (GRAPHED/UNGRAPHED). Defaults to UNGRAPHED; classify_gtp_chains(model)
    # walks the model at init (after set_cuda_graph_modules) and reclassifies on param name +
    # active cuda_graph_modules.
    obj.chain_id = GTPChain.UNGRAPHED.value
    # Grouped gemm
    obj.is_routed_expert = False
    obj.expert_idx = None
    obj.group = None
    obj.weight_list = None
    # Reduce-scatter state (set during wgrad_reduce_scatter)
    obj.rs_state = GTPWeightState.NONE
    obj._wgrad_rs_handle = None
    obj.rs_event = torch.cuda.Event(external=True)
    obj._rs_ticket = None
    # Padding
    obj.pad_length = 0
    # Debug
    obj._debug_name = ""
    # Hot-path caches (populated lazily on first use).  chain_id/group are
    # set after init, so we can't resolve streams eagerly here.
    obj._cached_ag_stream = None
    obj._cached_rs_stream = None
    obj._cached_dtypes = None
    obj._cached_gtp_remat_group = None


class GTPShardedParam(torch.nn.Parameter):
    """A weight parameter sharded 1/N across a GTP process group.

    Materialized on-demand via async all-gather and gradient-reduced via reduce-scatter.
    Carries its own prefetch-chain wiring (prev_w/next_w), per-chain state, AG/RS cache
    tickets, and the metadata the integrator needs to overlap with captured compute.
    """

    # TransformerEngine DistributedWeight protocol (see te.pytorch.distributed_weight).
    # `is_distributed_weight` is the capability marker TE dispatches on; no other
    # GTP-specific state leaks to TE.
    is_distributed_weight: bool = True

    # Per-chain linked-list state, keyed by chain_id; chains never cross-link (prev_w/next_w join
    # only same-chain params). Call reset_gtp_state() before rebuilding a GTP model in-process.
    _chain_state: Dict[str, dict] = {}

    # Recompute-forward prefetch cursor, keyed by chain_id; also cleared by reset_gtp_state().
    _recompute_chain_state: Dict[str, dict] = {}

    @classmethod
    def _get_chain_state(cls, chain_id: str) -> dict:
        if chain_id not in cls._chain_state:
            cls._chain_state[chain_id] = {
                "last_weight": None,
                "link_node_count": 0,
                "link_table_buffer": [],
                "link_table_flushed": False,
            }
        return cls._chain_state[chain_id]

    @classmethod
    def _get_recompute_chain_state(cls, chain_id: str) -> dict:
        if chain_id not in cls._recompute_chain_state:
            cls._recompute_chain_state[chain_id] = {"last_weight": None}
        return cls._recompute_chain_state[chain_id]

    @classmethod
    def _buffer_link_table_row(
        cls, prev: "GTPShardedParam", curr: "GTPShardedParam", chain: dict
    ) -> None:
        """Buffer one prefetch-link row (flushed atomically on the second forward pass)."""
        _W = 70
        _D = 20
        _S = 20

        def _layer_id(name: str) -> str:
            m = re.search(r"\d+", name)
            return m.group() if m else "-"

        def _shape(param: "GTPShardedParam") -> str:
            # Full (unsharded) weight shape that will be all-gathered across the gtp_remat
            # group — i.e. the size actually prefetched into the chain, not the local shard.
            try:
                return str(tuple(param._unsharded_shape))
            except Exception:
                return str(tuple(param.shape))

        def _dtype(param: "GTPShardedParam") -> str:
            # Report the dtype of the tensor that is ACTUALLY all-gathered, not the
            # GTPShardedParam wrapper (whose logical dtype is the high-precision model-weight
            # shard, i.e. params_dtype — bf16 in mixed precision). When the param has an FP8
            # representation (``param.quantized`` populated — by --fp8-param-gather's optimizer
            # FP32->FP8 write, or by the per-forward cast otherwise), that quantized tensor is
            # what gets gathered, yet a TE QuantizedTensor still reports a "fake" params_dtype
            # ``.dtype``. So surface its raw storage dtype (e.g. uint8) tagged with the quantized
            # class to make the FP8 all-gather unambiguous.
            q = getattr(param, "quantized", None)
            if getattr(param, "_gtp_native_fp8", False) and q is not None:
                raw = getattr(q, "_rowwise_data", None)
                if raw is None:
                    raw = getattr(q, "_data", None)
                raw_dt = str(raw.dtype).replace("torch.", "") if raw is not None else "?"
                return f"{type(q).__name__}/{raw_dt}"
            return str(getattr(param, "dtype", "-"))

        chain["link_node_count"] += 1
        if chain["link_node_count"] == 1:
            chain_id = getattr(curr, "chain_id", GTPChain.UNGRAPHED.value)
            chain["link_table_buffer"].append(
                f"\n[{chain_id} chain]\n{'node_id':>7} | {'layer_id':>8} |"
                f" {'dtype':<{_D}} | {'shape':<{_S}} | {'curr_weight_name':<{_W}} |"
                f" prev_weight_name\n{'-'*7}-+-{'-'*8}-+-{'-'*_D}-+-{'-'*_S}-+-{'-'*_W}-+-{'-'*_W}"
            )
            # Seed weight (first GTP param) as row 0
            chain["link_table_buffer"].append(
                f"{'0':>7} | {_layer_id(prev._debug_name):>8} | "
                f"{_dtype(prev):<{_D}} | {_shape(prev):<{_S}} | {prev._debug_name:<{_W}} | -"
            )
        chain["link_table_buffer"].append(
            f"{chain['link_node_count']:>7} | {_layer_id(curr._debug_name):>8} | "
            f"{_dtype(curr):<{_D}} | {_shape(curr):<{_S}} | "
            f"{curr._debug_name:<{_W}} | {prev._debug_name}"
        )

    @staticmethod
    def __new__(cls, tensor, *args, **kwargs):  # pylint: disable=unused-argument
        requires_grad = kwargs.get("requires_grad", True)
        # pylint: disable-next=unexpected-keyword-arg
        return super(GTPShardedParam, cls).__new__(cls, tensor, requires_grad=requires_grad)

    def __init__(self, tensor, *args, **kwargs):
        del tensor, args, kwargs
        super().__init__()
        _init_gtp_runtime_attrs(self)

    @property
    def _weights(self):
        """Individual weight shards (self for non-routed, weight_list for routed)."""
        weights = self.weight_list if self.is_routed_expert else [self]
        # Only meaningful when _set_state is actively tracking transitions.
        if GTP_CONFIG.check_param_states:
            assert all(w.state == weights[0].state for w in weights)
        return list(weights)

    @property
    def _unsharded_shape_padded(self):
        """Full unsharded shape *including* the pad rows on the last rank."""
        out_shape = list(self.size())
        out_shape[0] = out_shape[0] * self.group.size()
        return tuple(out_shape)

    @property
    def _unsharded_shape(self):
        """Full unsharded shape with the pad rows stripped (logical shape)."""
        out_shape = list(self._unsharded_shape_padded)
        out_shape[0] -= self.pad_length
        return tuple(out_shape)

    @property
    def _sharded_padded_shape(self):
        """This rank's local shard shape, padding included."""
        return tuple(self.size())

    def get_padded_shard(self):
        """Return the local shard already containing its share of padding (identity)."""
        return self

    def _set_state(self, new_state: GTPWeightState):
        """Advance the AG state (only inspected when ``check_param_states`` is on)."""
        # Only inspected when check_param_states is on; skip writes otherwise.
        if not GTP_CONFIG.check_param_states:
            return
        self.state = new_state

    def _set_rs_state(self, new_state: GTPWeightState):
        """Advance the RS state (only inspected when ``check_param_states`` is on)."""
        if not GTP_CONFIG.check_param_states:
            return
        self.rs_state = new_state

    def _get_cache_key(self, dtype, fwd: bool, reduce_scatter: bool) -> tuple:
        """Build cache key from output shape + dtype.

        Weights with matching gathered shape and dtype share a buffer. For experts gathered
        in parallel, self.expert_idx keeps each distinct; same-indexed experts across layers share.
        """

        if not isinstance(dtype, torch.dtype):
            return (
                self._unsharded_shape_padded,
                dtype,
                fwd,
                not fwd,
                self.expert_idx,
                reduce_scatter,
            )
        return (self._unsharded_shape_padded, dtype, self.expert_idx, reduce_scatter)

    def _strip_padding(self, tensor):
        if self.pad_length == 0:
            return tensor

        if isinstance(tensor, QuantizedTensor):
            assert isinstance(
                tensor, (NVFP4TensorStorage, MXFP8TensorStorage)
            ), f"Unsupported quantized tensor type for GTP padding: {type(tensor)}"

            metadata = tensor.get_metadata()
            if metadata.get("rowwise_data") is not None:
                metadata["rowwise_data"] = metadata["rowwise_data"][: -self.pad_length]
            if metadata.get("columnwise_data") is not None:
                if isinstance(tensor, NVFP4TensorStorage):
                    # NVFP4 transposes columnwise and packs 2 values per byte
                    metadata["columnwise_data"] = metadata["columnwise_data"][
                        ..., : -self.pad_length // 2
                    ].contiguous()
                else:
                    # MXFP8 columnwise is not transposed, strip first dim
                    metadata["columnwise_data"] = metadata["columnwise_data"][: -self.pad_length]
            M = self._unsharded_shape[0]
            if isinstance(tensor, NVFP4TensorStorage):
                # NVFP4 scale_inv shapes (see NVFP4Quantizer.get_scale_shape):
                #   rowwise_scale_inv:    [round_up(M, 128),  round_up(ceil(K/16), 4)]
                #   columnwise_scale_inv: [round_up(K, 128),  round_up(ceil(M/16), 4)]
                # GTP shards M (dim 0 of the weight), so strip to the unpadded sizes.
                if metadata.get("rowwise_scale_inv") is not None:
                    m_rows = round_up_to_nearest_multiple(M, 128)
                    metadata["rowwise_scale_inv"] = metadata["rowwise_scale_inv"][:m_rows]
                if metadata.get("columnwise_scale_inv") is not None:
                    m_tiles = round_up_to_nearest_multiple(
                        math.ceil(M / NVFP4_BLOCK_SCALING_SIZE), 4
                    )
                    metadata["columnwise_scale_inv"] = metadata["columnwise_scale_inv"][
                        :, :m_tiles
                    ].contiguous()
            else:
                # MXFP8 scale_inv shapes (see MXFP8Quantizer.get_scale_shape):
                #   rowwise_scale_inv:    [round_up(M, 128),     round_up(K//32, 4)]
                #   columnwise_scale_inv: [round_up(M//32, 4),   round_up(K, 128)]
                # GTP shards M (dim 0 of the weight), so strip to the unpadded sizes.
                if metadata.get("rowwise_scale_inv") is not None:
                    m_rows = round_up_to_nearest_multiple(M, 128)
                    metadata["rowwise_scale_inv"] = metadata["rowwise_scale_inv"][:m_rows]
                if metadata.get("columnwise_scale_inv") is not None:
                    m_tiles = round_up_to_nearest_multiple(M // MXFP8_BLOCK_SCALING_SIZE, 4)
                    metadata["columnwise_scale_inv"] = metadata["columnwise_scale_inv"][:m_tiles]

            return type(tensor)(**metadata, shape=self._unsharded_shape, dtype=torch.bfloat16)

        return tensor[: -self.pad_length]

    def _all_gather_weight(self, async_op, fwd, nvtx_label=None):
        """Quantize (if needed) and all-gather weight. Returns (weight_total, handle)."""
        if nvtx_label is None:
            nvtx_label = (
                self._debug_name + (".fwd" if fwd else ".bwd") + (".async" if async_op else ".sync")
            )
        nvtx_range_push(f"{nvtx_label}.all_gather_weight")

        weights = self._weights

        # 1. Transition state for async gathers. Skip during recompute-forward: it gathers
        #    rowwise (_ag_ticket_fwd) while a bwd-chain prefetch may hold an in-flight columnwise
        #    AG state (_ag_ticket_bwd) on the same weight — clobbering breaks the dgrad consume.
        if GTP_CONFIG.check_param_states and not in_fp8_activation_recompute_phase():
            new_state = GTPWeightState.ASYNC_WAIT if async_op else GTPWeightState.DATA_READY_SYNC
            for w in weights:
                w._set_state(new_state)

        # 2. Set FP8 usage direction (rowwise for fwd, columnwise for bwd) on the GATHER
        #    quantizer copy — NEVER on the param's own quantizer: the optimizer's
        #    copy_ -> quantize_ update writes whichever usages the param quantizer has enabled,
        #    so leaving it rowwise=False after a bwd gather would freeze the rowwise (fwd) data.
        #    No re-quantize here: with mxfp8 + --fp8-param-gather the shard already IS a native
        #    FP8 tensor. BF16 GTP carries no quantizer and gathers the BF16 shard as-is.
        native_fp8 = getattr(self, "_gtp_native_fp8", False)
        quantizers = [getattr(w, "_gtp_gather_quantizer", None) for w in weights]
        if native_fp8:
            for q in quantizers:
                q.set_usage(rowwise=fwd, columnwise=not fwd)

        # 3. Build gather inputs. The gather collective takes the per-weight quantizers so it can
        #    reconstruct the gathered FP8 tensor's scale/metadata (None for BF16 GTP).
        if native_fp8:
            gather_weights = [w.quantized for w in weights]
        else:
            gather_weights = list(w.get_padded_shard() for w in weights)

        # 4. Cache checkout — use pooled buffers for both async and sync gathers
        #    to avoid allocating fresh memory each iteration. gather-buffer dtypes are stable
        #    post-construction (FP8 quantizer dtype for native-FP8 shards, else the BF16 dtype),
        #    so cache them on the anchor (self == weights[0]) instead of rebuilding each call.
        dtypes = self._cached_dtypes
        if dtypes is None:
            dtypes = [q.dtype if q is not None else w.dtype for q, w in zip(quantizers, weights)]
            self._cached_dtypes = dtypes
        out_buffers = []
        cache = get_global_GTP_cache()
        for p, dt in zip(weights, dtypes):
            if fwd:
                if p._ag_ticket_fwd is None:
                    p._ag_ticket_fwd = cache.reserve(p, dt, fwd=True)
                    cache.get(p._ag_ticket_fwd)
                    cache.release(p._ag_ticket_fwd)
                out_buffers.append(cache.get(p._ag_ticket_fwd))
            else:
                if p._ag_ticket_bwd is None:
                    p._ag_ticket_bwd = cache.reserve(p, dt, fwd=False)
                out_buffers.append(cache.get(p._ag_ticket_bwd))

        # 5. Communicate.
        gtp_remat_group = self._cached_gtp_remat_group
        if gtp_remat_group is None:
            gtp_remat_group = weights[0].group
            self._cached_gtp_remat_group = gtp_remat_group
        if GTP_CONFIG.check_param_states and len(gather_weights) > 1:
            # Debug invariant: batched AG needs distinct output buffers per expert.
            assert len(set(id(b) for b in out_buffers)) == len(
                out_buffers
            ), "Duplicate output buffers in batched all-gather — experts need distinct cache keys"

        # ASYNC AG: issue on ag_stream so its tail reflects the collective's full lifecycle
        # (what external wait_stream(ag_stream) drains depend on). The explicit outer→ag_stream
        # sync event preserves the upstream quantize-writer edge the bare stream context drops;
        # held on self so the event pool can't recycle it between capture and replay.
        # SYNC AG: stay on caller — output ready on return.
        if async_op:
            outer_stream = torch.cuda.current_stream()
            ag_stream = get_ag_stream(self.chain_id, gtp_remat_group)
            if getattr(self, "_ag_outer_sync_event", None) is None:
                self._ag_outer_sync_event = torch.cuda.Event()
            outer_sync_event = self._ag_outer_sync_event
            outer_sync_event.record(outer_stream)
            ag_stream.wait_event(outer_sync_event)
            ag_ctx = torch.cuda.stream(ag_stream)
        else:
            ag_ctx = nullcontext()

        with ag_ctx:
            if len(gather_weights) > 1:
                nvtx_range_push(f"{nvtx_label}.batched_gtp_ag")
                results, handle = grouped_gather_along_first_dim(
                    gather_weights,
                    gtp_remat_group,
                    async_op=async_op,
                    quantizers=quantizers,
                    output_tensors=out_buffers,
                )
                nvtx_range_pop(f"{nvtx_label}.batched_gtp_ag")
            else:
                nvtx_range_push(f"{nvtx_label}.gtp_ag")
                weight_total, handle = gather_along_first_dim(
                    gather_weights[0],
                    gtp_remat_group,
                    quantizer=quantizers[0],
                    async_op=async_op,
                    output_tensor=out_buffers[0] if out_buffers is not None else None,
                )
                nvtx_range_pop(f"{nvtx_label}.gtp_ag")
                results = [weight_total]

        result = results if self.is_routed_expert else results[0]

        # 6. Wrap handle.
        if async_op:
            handle = GTPShardHandle(handle, weights)
        else:
            handle = None

        nvtx_range_pop(f"{nvtx_label}.all_gather_weight")
        return result, handle

    def _wait_param_gather(self):
        # Enter ag_stream context so handle.wait() + ag_event.record() both
        # land on ag_stream. That makes ag_event mark ag_stream's tail, which
        # is what external drains via wait_stream(ag_stream) actually block on.
        ag_stream = self._cached_ag_stream
        if ag_stream is None:
            ag_stream = get_ag_stream(self.chain_id, self.group)
            self._cached_ag_stream = ag_stream
        with torch.cuda.stream(ag_stream):
            if self._prefetch_handle is not None:
                self._prefetch_handle.wait()
                self._prefetch_handle = None
                self.ag_event.record()

    def _all_gather_weight_on_demand(self, fwd):
        result, _ = self._all_gather_weight(async_op=False, fwd=fwd)
        result = result if self.is_routed_expert else [result]
        result = [self._strip_padding(r) for r in result]
        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result, self._weights)]
        return result if self.is_routed_expert else result[0]

    def _get_prefetched_weight(self, fwd):
        # Stale-read guard: state must reflect an AG issued for this cycle;
        # otherwise cache.get() would return the prior iter's AG buffer.
        if GTP_CONFIG.check_param_states:
            for w in self._weights:
                assert w.state in (
                    GTPWeightState.ASYNC_WAIT,
                    GTPWeightState.DATA_READY,
                    GTPWeightState.DATA_READY_SYNC,
                ), (
                    f"[GTP] _get_prefetched_weight({'fwd' if fwd else 'bwd'}) on "
                    f"{self._debug_name} with state={w.state!r} — no AG issued; "
                    "cache.get() would return stale data. Check the chain's "
                    "_need_weight_prefetch flag and issuer's prefetch logic."
                )
        _was_drained = getattr(self, "_already_ag_drained", False)
        if _was_drained:
            # Producer already drained via wait_async_comms; skip the captured cross-graph
            # wait (a CUDA no-op anyway). Correctness comes from the eager main_stream sync.
            self._already_ag_drained = False
        else:
            # Intra-graph or eager consume: drain inline.
            self._wait_param_gather()
            self.ag_event.wait()

        # Retrieve prefetched results from cache
        result = []
        cache = get_global_GTP_cache()
        for w in self._weights:
            ticket = w._ag_ticket_fwd if fwd else w._ag_ticket_bwd
            result.append(cache.get(ticket))

        result = [self._strip_padding(r) for r in result]

        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result, self._weights)]
        return result if self.is_routed_expert else result[0]

    def _wait_recompute_param_gather(self):
        # Recompute-chain analogue of _wait_param_gather, on the _recompute_* slot.
        ag_stream = self._cached_ag_stream
        if ag_stream is None:
            ag_stream = get_ag_stream(self.chain_id, self.group)
            self._cached_ag_stream = ag_stream
        with torch.cuda.stream(ag_stream):
            if self._recompute_prefetch_handle is not None:
                self._recompute_prefetch_handle.wait()
                self._recompute_prefetch_handle = None
                self._recompute_ag_event.record()

    def _recompute_prefetch_next(self, target, nvtx_label=None):
        # Issue target's rowwise (fwd) AG into its recompute slot. _all_gather_weight skips the
        # AG-state transition under recompute, so target's dgrad state is untouched; result lands
        # in target._ag_ticket_fwd.
        _, handle = target._all_gather_weight(async_op=True, fwd=True, nvtx_label=nvtx_label)
        target._recompute_prefetch_handle = handle

    def _get_recompute_prefetched_weight(self):
        # Recompute-chain analogue of _get_prefetched_weight (state-neutral; reads the
        # rowwise _ag_ticket_fwd via the _recompute_* slot).
        if self._recompute_already_drained:
            # Producer already drained via wait_async_comms (CG capture); skip the
            # captured cross-graph wait (CUDA no-op anyway).
            self._recompute_already_drained = False
        else:
            self._wait_recompute_param_gather()
            self._recompute_ag_event.wait()

        result = []
        cache = get_global_GTP_cache()
        for w in self._weights:
            result.append(cache.get(w._ag_ticket_fwd))
        result = [self._strip_padding(r) for r in result]
        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result, self._weights)]
        return result if self.is_routed_expert else result[0]

    def all_gather_and_prefetch_bwd(self, nvtx_label=None):
        """Backward variant: get the current weight (cached if prefetched, else sync gather)
        and async-prefetch prev_w.

        Safe via the coat-check cache: get() returns the current buffer to the pool, and the
        prefetch's checkout allocates a separate buffer if the pool is empty (current buffer
        still live via the caller's reference).

        Returns:
            weight_total
        """

        if GTP_CONFIG.weight_prefetch and self.next_w is not None:
            result = self._get_prefetched_weight(False)
        else:
            result = self._all_gather_weight_on_demand(False)

        if (
            GTP_CONFIG.weight_prefetch
            and self.prev_w is not None
            and self.prev_w._need_weight_prefetch
            and self.prev_w._need_weight_prefetch_bwd
        ):
            # Pre-AG work (quantize, ticket lookup) runs on caller's stream; the NCCL collective
            # is wrapped on ag_stream inside _all_gather_weight (see its async/sync gate).
            _, handle = self.prev_w._all_gather_weight(
                async_op=True, fwd=False, nvtx_label=nvtx_label
            )
            self.prev_w._prefetch_handle = handle

        # The unsharded tensor has been returned, no pending work so reset state to NONE
        if GTP_CONFIG.check_param_states:
            for w in self._weights:
                w._set_state(GTPWeightState.NONE)

        if GTP_CONFIG.weight_prefetch and self.next_w is not None:
            cache = get_global_GTP_cache()
            for w in self._weights:
                cache.release(w._ag_ticket_bwd)

        return result

    def batched_all_gather_and_prefetch_bwd(self, nvtx_label=None):
        """Batched backward all-gather + prefetch. Wrapper around all_gather_and_prefetch_bwd."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.all_gather_and_prefetch_bwd(nvtx_label=nvtx_label)

    def all_gather_and_prefetch(self, fwd: bool = True, nvtx_label: str = None):
        """All-gather the current weight and async-prefetch the next.

        Returns:
            weight_total
        """
        # During an activation-recompute forward (runs in backward), route consume +
        # prefetch through the recompute-forward chain on its own _recompute_* slot
        # (see __init__) instead of the fwd/bwd chains; lazy-built below.
        in_recompute = in_fp8_activation_recompute_phase()
        use_recompute_chain = in_recompute and GTP_CONFIG.weight_prefetch

        # Consume current weight.
        if use_recompute_chain and self._recompute_prev is not None:
            result = self._get_recompute_prefetched_weight()
        elif not in_recompute and GTP_CONFIG.weight_prefetch and self.prev_w is not None:
            result = self._get_prefetched_weight(True)
        else:
            # On-demand: chain head (fwd or recompute global-first) or first-iter build.
            result = self._all_gather_weight_on_demand(True)

        # Prefetch next weight on the matching chain.
        if (
            use_recompute_chain
            and self._recompute_next is not None
            and self._recompute_next._need_weight_prefetch
        ):
            self._recompute_prefetch_next(self._recompute_next, nvtx_label=nvtx_label)
        elif (
            not in_recompute
            and GTP_CONFIG.weight_prefetch
            and self.next_w is not None
            and self.next_w._need_weight_prefetch
        ):
            # Pre-AG work on caller; NCCL wrap lives at the collective site
            # inside _all_gather_weight. See all_gather_and_prefetch_bwd.
            _, handle = self.next_w._all_gather_weight(
                async_op=True, fwd=fwd, nvtx_label=nvtx_label
            )
            self.next_w._prefetch_handle = handle

        # Unsharded tensor returned, no pending work → reset state to NONE. Skip during recompute:
        # a bwd-chain prefetch may hold an in-flight AG state this weight's later dgrad needs.
        if GTP_CONFIG.check_param_states and not in_recompute:
            for w in self._weights:
                w._set_state(GTPWeightState.NONE)

        cls = type(self)

        # Lazy-build the recompute-forward prefetch chain (first backward, in recompute order).
        # Consume/prefetch above used the prior iter's links, so the first backward runs on-demand
        # while these are established.
        if in_recompute and not self._recompute_initialized:
            rchain = cls._get_recompute_chain_state(self.chain_id)
            last_r = rchain["last_weight"]
            if last_r is not None and last_r._recompute_next is None:
                last_r._recompute_next = self
                self._recompute_prev = last_r
            self._recompute_initialized = True
            rchain["last_weight"] = self

        # Lazy population of the fwd/bwd linked list: link previous weight to current.
        # Uses per-chain state so dense and expert chains never cross-link.
        chain = cls._get_chain_state(self.chain_id)
        if not self.prefetch_initialized:
            last_w = chain["last_weight"]
            if last_w is not None and last_w.next_w is None:
                cls._buffer_link_table_row(last_w, self, chain)
                last_w.next_w = self
                self.prev_w = last_w

            cache = get_global_GTP_cache()

            # Set the fwd ag buffer (gather quantizer copy — the param's own quantizer is
            # reserved for the optimizer's update path; see attach_gtp_to_presharded_module).
            quantizers = [getattr(w, "_gtp_gather_quantizer", None) for w in self._weights]
            dtypes = [
                q.dtype if q is not None else w.dtype for q, w in zip(quantizers, self._weights)
            ]
            for w, dt in zip(self._weights, dtypes):
                w._ag_ticket_fwd = cache.reserve(w, dt, fwd=True)
                cache.get(w._ag_ticket_fwd)
                cache.release(w._ag_ticket_fwd)

            self.prefetch_initialized = True
            chain["last_weight"] = self
        elif not chain["link_table_flushed"] and chain["link_table_buffer"]:
            # Second forward pass: flush the complete table atomically to avoid interleaving
            chain["link_table_flushed"] = True
            log_single_rank(logger, logging.INFO, "\n".join(chain["link_table_buffer"]) + "\n")

        return result

    def batched_all_gather_and_prefetch(self, **kwargs):
        """Batched all-gather + prefetch for expert weights (wraps all_gather_and_prefetch)."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.all_gather_and_prefetch(**kwargs)

    def get_wgrad_tensor(self):
        """Pool-allocate a wgrad scratch tensor of unsharded shape for the bwd GEMM."""
        return _wgrad_pool_get(self._unsharded_shape, self.main_grad.dtype, self.device)

    def register_grad_accum_hook(self, grad_accum_node, hook):
        """Register a DDP backward hook to call after the wgrad RS finalize.

        For GTP params autograd may receive None (async RS), so the normal grad-accumulator
        hook never fires; the integrator (Graphed.backward for captured chains, or the eager
        chain-tail cascade) calls this hook explicitly after RS wait + accumulation, so DDP's
        register_grad_ready fires at the right time. grad_accum_node is accepted for API
        compatibility but not retained — only the hook callable.
        """
        del grad_accum_node
        self._grad_accum_hook = hook

    @staticmethod
    def _handle_megatron_grad_accum(param):
        """Handle megatron DDP and gradient-accumulation fusion.

        Do NOT set param.grad before calling the hook — the hook checks param.grad and would
        accumulate it into main_grad if zero_out_wgrad is True, corrupting it with a dummy.

        Returns a cached dummy wgrad; sync callers use it as the graph-safe grad, async drains
        discard it.
        """
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = True
        dummy_grad = get_dummy_wgrad(list(param.main_grad.shape), param.dtype)
        if getattr(param, "_grad_accum_hook", None) is not None:
            param._grad_accum_hook()

        param._set_rs_state(GTPWeightState.NONE)
        return dummy_grad

    def _wait_reduce_scatter(self, finalize_grad=False):
        # Enter rs_stream context so handle.wait() + rs_event.record() land on rs_stream
        # (mirrors _wait_param_gather). With finalize_grad=True, main_grad.add_ also runs on
        # rs_stream right after the NCCL RS — starts during AG drain, not after, avoiding
        # SM-saturation that blocks cross-graph overlap.
        rs_stream = self._cached_rs_stream
        if rs_stream is None:
            rs_stream = get_rs_stream(self.chain_id, self.group)
            self._cached_rs_stream = rs_stream
        with torch.cuda.stream(rs_stream):
            if self._wgrad_rs_handle is not None:
                self._wgrad_rs_handle.wait()
                self._wgrad_rs_handle = None
                self.rs_event.record()
                if finalize_grad:
                    cache = get_global_GTP_cache()
                    for w in self._weights:
                        wgrad_rs = cache.get(w._rs_ticket)
                        w.main_grad.add_(wgrad_rs)
                        cache.release(w._rs_ticket)
                    # Fire grad-ready AFTER all adds (separate loop so a bucket-completing
                    # grad-ready can't dispatch the RS before a sibling's add). With autograd
                    # grad-ready suppressed for GTP params (DDP register_grad_accum_hook), this
                    # is the only grad-ready for a weight finalized here; else the bucket orphans.
                    for w in self._weights:
                        self._handle_megatron_grad_accum(w)
                    self._already_finalized = True
        # Release stashed wgrad inputs: UNGRAPHED buffers go back to the pool;
        # GRAPHED just drops Python refs (addresses must stay stable for CG).
        if getattr(self, "_wgrad_input_bufs", None) is not None:
            if self.chain_id == GTPChain.UNGRAPHED.value:
                for buf in self._wgrad_input_bufs:
                    _wgrad_pool_put(buf)
            self._wgrad_input_bufs = None

    def _prescale_wgrads_for_mean_rs(self, wgrads):
        """Pre-scale wgrad by 1/gtp_remat so the SUM reduce-scatter yields the gtp_remat mean.

        Single choke point for every RS path. Composes with DDP's 1/replicate prescale and
        finalize's AVG to give the full (replicate x gtp_remat) mean. Skipped under
        calculate_per_token_loss, where DDP does no 1/dp scaling and total_global_tokens (which
        counts gtp_remat peers' tokens) normalizes instead — there the gtp_remat axis must SUM
        like the DP axis (a 1/gtp_remat mean would shrink every gtp_remat grad).
        """
        gtp_remat_size = self.group.size()
        if gtp_remat_size > 1 and not GTP_CONFIG.calculate_per_token_loss:
            torch._foreach_mul_(list(wgrads), 1.0 / gtp_remat_size)

    def _reduce_scatter(self, wgrads, async_op, nvtx_label=None):
        """Reduce-scatter one or more wgrads → (outputs, handle). Single tensor: plain RS;
        multiple: coalesced RS."""
        if nvtx_label is None:
            nvtx_label = self._debug_name + ".bwd" + (".async" if async_op else ".sync")

        # MEAN reduce-scatter: pre-scale wgrad so the SUM collective yields the gtp_remat mean.
        self._prescale_wgrads_for_mean_rs(wgrads)

        if GTP_CONFIG.check_param_states:
            new_rs_state = GTPWeightState.ASYNC_WAIT if async_op else GTPWeightState.DATA_READY_SYNC
            for w in self._weights:
                w._set_rs_state(new_rs_state)

        if self.pad_length > 0:
            wgrads = [torch.nn.functional.pad(w, (0, 0, 0, self.pad_length)) for w in wgrads]

        if async_op:
            dtypes = [w.dtype for w in wgrads]
            out_buffers = []
            cache = get_global_GTP_cache()
            for p, dt in zip(self._weights, dtypes):
                if p._rs_ticket is None:
                    p._rs_ticket = cache.reserve(p, dt, fwd=False, reduce_scatter=True)
                out_buffers.append(cache.get(p._rs_ticket))
        else:
            out_buffers = [None] * len(wgrads)

        # ASYNC RS: issue on rs_stream so its tail reflects the collective's full lifecycle
        # (what external wait_stream(rs_stream) drains depend on). The explicit outer→rs_stream
        # sync event preserves the wgrad-GEMM writer edge the bare stream context drops; held on
        # self so the event pool can't recycle it between capture and replay. Mirrors the AG path.
        # SYNC RS: stay on caller — output ready on return.
        if async_op:
            outer_stream = torch.cuda.current_stream()
            rs_stream = get_rs_stream(self.chain_id, self.group)
            if getattr(self, "_rs_outer_sync_event", None) is None:
                self._rs_outer_sync_event = torch.cuda.Event()
            outer_sync_event = self._rs_outer_sync_event
            outer_sync_event.record(outer_stream)
            rs_stream.wait_event(outer_sync_event)
            rs_ctx = torch.cuda.stream(rs_stream)
        else:
            rs_ctx = nullcontext()

        with rs_ctx:
            if len(wgrads) == 1:
                nvtx_range_push(f"{nvtx_label}.gtp_rs")
                out, handle = reduce_scatter_along_first_dim(
                    wgrads[0], self.group, async_op=async_op, output=out_buffers[0]
                )
                nvtx_range_pop(f"{nvtx_label}.gtp_rs")
                return [out], handle

            outputs = []
            nvtx_range_push(f"{nvtx_label}.batched_gtp_rs")
            with torch.distributed._coalescing_manager(
                group=self.group, device=wgrads[0].device, async_ops=async_op
            ) as cm:
                for out_buffer, tensor in zip(out_buffers, wgrads):
                    out, _ = reduce_scatter_along_first_dim(tensor, self.group, output=out_buffer)
                    outputs.append(out)
            nvtx_range_pop(f"{nvtx_label}.batched_gtp_rs")

            return outputs, cm if async_op else None

    def wgrad_reduce_scatter(self, wgrad, nvtx_label=None):
        """Reduce-scatter wgrad(s): sync for the last weight, async+deferred for others.
        Accepts a single tensor (non-routed) or a list (routed experts).

        Returns:
            Single tensor or list for sync (last weight) — backward returns this.
            None or tuple of Nones for async — backward returns this.
        """
        batched = isinstance(wgrad, (list, tuple))
        wgrads = list(wgrad) if batched else [wgrad]
        weights = self._weights

        # UNGRAPHED wgrads recycle via the standalone pool (_wgrad_pool_put); GRAPHED wgrads
        # cannot, since CUDA graphs require stable buffer addresses across replay.
        poolable = self.chain_id == GTPChain.UNGRAPHED.value

        if GTP_CONFIG.async_reduction and self.prev_w is not None:
            # Async RS (not last weight — deferred finish). Pre-RS work on caller; NCCL wrap
            # lives at the collective site inside _reduce_scatter (mirrors the AG prefetch sites).
            _, rs_handle = self._reduce_scatter(wgrads, async_op=True, nvtx_label=nvtx_label)
            self._wgrad_rs_handle = GTPShardHandle(rs_handle, weights, reduce_scatter=True)
            # Stash wgrad input buffers — cannot recycle yet because the async RS
            # kernel is still reading them on rs_stream.
            self._wgrad_input_bufs = wgrads
            ret = tuple([None] * len(wgrads)) if batched else None
        else:
            # Sync reduce-scatter — reached as the natural chain-head case, recycle immediately
            wgrads, _ = self._reduce_scatter(wgrads, async_op=False, nvtx_label=nvtx_label)
            nvtx_range_push(f"{nvtx_label}.gtp_wgrad_accum")
            if len(weights) == 1:
                weights[0].main_grad.add_(wgrads[0])
            else:
                torch._foreach_add_([p.main_grad for p in weights], wgrads)
            nvtx_range_pop(f"{nvtx_label}.gtp_wgrad_accum")
            result = [self._handle_megatron_grad_accum(p) for p in weights]

            if poolable:
                for buf in wgrads:
                    _wgrad_pool_put(buf)
            ret = result if batched else result[0]

        # Wait for last reduce scatter if it was async
        # Currently only support reduce scattering in reverse order
        if GTP_CONFIG.async_reduction and self.next_w is not None:
            self.next_w._wait_reduce_scatter()

            if getattr(self.next_w, "_already_finalized", False):
                self.next_w._already_finalized = False
            else:
                self.next_w.rs_event.wait()
                cache = get_global_GTP_cache()
                next_weights = self.next_w._weights
                wgrads = [cache.get(w._rs_ticket) for w in next_weights]
                nvtx_range_push(f"{self.next_w._debug_name}.gtp_wgrad_accum_deferred")
                # Only batch with _foreach_add_ when finalizing multiple (routed) weights.
                if len(next_weights) == 1:
                    next_weights[0].main_grad.add_(wgrads[0])
                else:
                    torch._foreach_add_([w.main_grad for w in next_weights], wgrads)
                nvtx_range_pop(f"{self.next_w._debug_name}.gtp_wgrad_accum_deferred")
                for w in next_weights:
                    self._handle_megatron_grad_accum(w)
                    cache.release(w._rs_ticket)

        return ret

    def batched_wgrad_reduce_scatter(self, wgrad_list, nvtx_label=None):
        """Batched version of wgrad_reduce_scatter."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.wgrad_reduce_scatter(wgrad_list, nvtx_label=nvtx_label)

    # ------------------------------------------------------------------
    # TransformerEngine DistributedWeight protocol. TE's fwd/bwd dispatch through these generic
    # names (see te.pytorch.distributed_weight.materialize_weights_for_forward et al.). The leader
    # param encapsulates the whole group via self._weights, so a single call covers both the Linear
    # (one weight) and GroupedLinear (routed-expert list) cases; the underlying methods already
    # return a single tensor or a list accordingly. These are thin adapters over the GTP methods.
    # ------------------------------------------------------------------
    def materialize_group_for_forward(self):
        """Protocol: all-gather the group's shard(s) for the forward GEMM."""
        return self.all_gather_and_prefetch(fwd=True)

    def materialize_group_for_backward(self, nvtx_label=None):
        """Protocol: re-materialize the group's weight(s) for the backward GEMMs."""
        return self.all_gather_and_prefetch_bwd(nvtx_label=nvtx_label)

    def finalize_group_grads(self, wgrads, nvtx_label=None):
        """Protocol: reduce-scatter the group's freshly computed weight grad(s)."""
        return self.wgrad_reduce_scatter(wgrads, nvtx_label=nvtx_label)

    def grad_buffer(self):
        """Protocol: the wgrad accumulation scratch buffer for this weight."""
        return self.get_wgrad_tensor()

    def get_data_tensors(self):
        """Expose self as the lone data tensor for TE's offload-marking interface.

        TE's mark_activation_offload treats any non-plain tensor as a storage wrapper and calls
        get_data_tensors() on it; a sharded param has no inner buffers, so it is its own.
        """
        return (self,)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Subclass-preserving dispatch for ``detach`` (other ops fall through)."""
        del types  # required by protocol, unused here
        if kwargs is None:
            kwargs = {}

        if func is torch.Tensor.detach:
            with torch._C.DisableTorchFunctionSubclass():
                # Perform the raw detach
                result = func(*args, **kwargs)
            # Re-wrap it in your subclass so PyTorch is happy
            return result.as_subclass(type(self))

        # 2. For everything else (add, mul, etc.), be transparent/decay.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)


@dataclass
class _TicketSlot:
    """Internal slot backing a persistent ticket in the GTP buffer cache."""

    key: tuple  # cache key (shape, dtype, ...)
    param: "GTPShardedParam"  # for lazy allocation metadata
    dtype: object  # torch.dtype or tex.DType
    reduce_scatter: bool
    fwd: bool
    chain_id: str = GTPChain.GRAPHED.value  # chain this slot belongs to
    buf: Optional[torch.Tensor] = field(default=None)  # None when released or after clear()


# CUDA-graph memory pool: routes GRAPHED-chain allocations (AG/RS buffers, quantized weight
# storage) into the capture pool at creation time, avoiding post-hoc reallocation. Registered
# via set_cuda_graph_mempool before the first graphed forward; stays None when CG is off, where
# _graphed_alloc is a no-op (regular allocator).
_CG_MEMPOOL_DEVICE = None
_CG_MEMPOOL = None


def set_cuda_graph_mempool(device, mempool):
    """Register the CUDA-graph memory pool for GRAPHED-chain GTP allocations."""
    global _CG_MEMPOOL_DEVICE, _CG_MEMPOOL
    _CG_MEMPOOL_DEVICE = device
    _CG_MEMPOOL = mempool


@contextmanager
def _graphed_alloc(chain_id):
    """Route allocations in this block into the registered CG mempool when ``chain_id``
    is GRAPHED and a pool is registered; otherwise a no-op (regular allocator)."""
    if _CG_MEMPOOL is not None and chain_id == GTPChain.GRAPHED.value:
        torch._C._cuda_beginAllocateCurrentThreadToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
    else:
        yield


class GTPWeightCache:
    """Ticket-based buffer pool for GTP all-gather / reduce-scatter buffers.

    - reserve(param, dtype, fwd) → ticket: assign a persistent ticket (no buffer yet).
    - get(ticket) → buffer: return the buffer, lazily (re)allocating from pool or fresh.
    - release(ticket): return the buffer to the pool; ticket stays valid.
    - clear(): drop all buffers/pools; tickets stay valid, next get() allocates fresh.
    """

    # Bytes per element for known dtypes (for logging). Add entries when GTP caches buffers of
    # new quantized dtypes — only DType values the TE pybind bindings expose (verify via
    # hasattr(tex.DType, ...) before adding speculative entries).
    _BYTES_PER_ELEMENT = {
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.float32: 4,
        tex.DType.kFloat4E2M1: 0.5,
        tex.DType.kFloat8E4M3: 1,
        tex.DType.kFloat8E5M2: 1,
    }

    def __init__(self):
        self._pool: Dict[tuple, List[torch.Tensor]] = defaultdict(list)
        self._slots: Dict[int, _TicketSlot] = {}
        self._next_ticket: int = 0
        self._total_bytes: int = 0  # running total of allocated bytes
        self.key_to_allocate_func = {}

    @staticmethod
    def _buf_bytes(shape, dtype) -> int:
        """Estimate buffer size in bytes."""
        numel = 1
        for d in shape:
            numel *= d
        if dtype not in GTPWeightCache._BYTES_PER_ELEMENT:
            raise KeyError(
                f"GTPWeightCache._buf_bytes: unknown dtype {dtype!r}.  "
                "Add it to GTPWeightCache._BYTES_PER_ELEMENT with its bytes-per-element."
            )
        return int(numel * GTPWeightCache._BYTES_PER_ELEMENT[dtype])

    def _allocate_buffer(
        self, param: "GTPShardedParam", dtype, reduce_scatter, fwd
    ) -> torch.Tensor:
        if reduce_scatter:
            out_shape = param._sharded_padded_shape
        else:
            out_shape = param._unsharded_shape_padded

        # Route GRAPHED-chain buffers into the CG mempool at creation (see _graphed_alloc).
        with _graphed_alloc(getattr(param, "chain_id", GTPChain.UNGRAPHED.value)):
            if not isinstance(dtype, torch.dtype):
                # Use the gather quantizer copy: mutating the param's own quantizer usage
                # would corrupt the optimizer's quantize_ update direction (frozen weights).
                quantizer = getattr(param, "_gtp_gather_quantizer", None) or param._quantizer
                assert quantizer is not None
                quantizer.set_usage(rowwise=fwd, columnwise=not fwd)

                buf = quantizer.make_empty(
                    out_shape, dtype=torch.bfloat16, device=torch.cuda.current_device()
                )
            else:
                buf = torch.empty(
                    out_shape,
                    dtype=dtype,
                    device=param.device,
                    memory_format=torch.contiguous_format,
                )

        buf_bytes = self._buf_bytes(out_shape, dtype)
        self._total_bytes += buf_bytes
        dtype_str = (
            str(dtype) if isinstance(dtype, torch.dtype) else getattr(dtype, "name", str(dtype))
        )
        op_str = "RS(grad)" if reduce_scatter else ("AG(fwd)" if fwd else "AG(bwd)")
        log_single_rank(
            logger,
            logging.INFO,
            f"[GTP Cache] +{buf_bytes / 1024**2:.1f} MB  (shape={out_shape}, dtype={dtype_str})  "
            f"total={self._total_bytes / 1024**2:.1f} MB  param: {param._debug_name} "
            f"op: {op_str}",
        )
        return buf

    def reserve(self, param: "GTPShardedParam", dtype, fwd: bool, reduce_scatter=False) -> int:
        """Assign a persistent ticket.  No buffer is allocated until ``get()``."""
        key = param._get_cache_key(dtype, fwd, reduce_scatter)
        ticket = self._next_ticket
        self._next_ticket += 1

        self._slots[ticket] = _TicketSlot(
            key=key,
            param=param,
            dtype=dtype,
            reduce_scatter=reduce_scatter,
            fwd=fwd,
            chain_id=getattr(param, "chain_id", GTPChain.UNGRAPHED.value),
        )
        return ticket

    def get(self, ticket: int) -> torch.Tensor:
        """Return the buffer for *ticket*, lazily allocating if needed."""
        slot = self._slots[ticket]
        if slot.buf is None:
            pool = self._pool[slot.key]
            slot.buf = (
                pool.pop()
                if pool
                else self._allocate_buffer(
                    slot.param, slot.dtype, slot.reduce_scatter, fwd=slot.fwd
                )
            )
            self.key_to_allocate_func[slot.key] = (
                slot.param,
                slot.dtype,
                slot.reduce_scatter,
                slot.fwd,
            )

        return slot.buf

    def release(self, ticket: int):
        """Return the buffer to the pool (ticket stays valid).

        slot.buf is intentionally NOT cleared: get() must stay idempotent so CUDA-graph-captured
        buffers keep their fixed address across replays.
        """
        slot = self._slots[ticket]
        if slot.buf is None:
            return
        # Use identity check — tensor == tensor returns a multi-element bool tensor
        # which crashes in a boolean context ("Boolean value of Tensor is ambiguous").
        if not any(b is slot.buf for b in self._pool.get(slot.key, [])):
            self._pool[slot.key].append(slot.buf)

    def clear(self):
        """Drop all buffers; tickets remain valid and lazily re-allocate on next get()."""
        for slot in self._slots.values():
            slot.buf = None
        self._pool.clear()
        self._total_bytes = 0


def get_global_GTP_cache() -> GTPWeightCache:
    """Get or lazily create the global cache instance."""
    global _GTP_CACHE
    if _GTP_CACHE is None:
        _GTP_CACHE = GTPWeightCache()
    return _GTP_CACHE


def wait_async_comms(
    chain_id: str = None, skip_rs: bool = False, finalize_after_drain: bool = False
):
    """Drain in-flight GTP async AG / RS handles.

    Inside CUDA graph capture the drains are captured into the graph — the producer-side hook
    for cross-graph overlap. A captured cudaStreamWaitEvent on another capture session's event is
    a CUDA no-op, so consumers can't wait cross-graph; instead the producer drains here and flags
    the param, and the consumer skips its captured wait.

    Args:
        chain_id: If specified, only drain params on this chain.
        skip_rs:  Drain AG only; leave RS in flight.
        finalize_after_drain: After RS drain, also accumulate wgrad into
                 main_grad. Runs main_grad.add_ on rs_stream (right after
                 NCCL RS) so it starts during AG drain rather than after,
                 avoiding SM-saturation that blocks cross-graph overlap.
                 Falls back to caller-stream accumulation if no RS handle.

    Per-param side effects:
        * _already_ag_drained = True   (if an AG handle was drained)
        * _already_finalized  = True   (if finalize_after_drain=True)
    """
    for param in list(_inflight_comm_params):
        if (
            chain_id is not None
            and getattr(param, "chain_id", GTPChain.UNGRAPHED.value) != chain_id
        ):
            continue
        had_ag = param._prefetch_handle is not None
        param._wait_param_gather()
        if had_ag:
            param._already_ag_drained = True
        # Recompute-forward chain: drain its separate in-flight rowwise AG so the
        # captured recompute consumer skips its cross-graph wait (full-iteration CG).
        if param._recompute_prefetch_handle is not None:
            param._wait_recompute_param_gather()
            param._recompute_already_drained = True
        if not skip_rs:
            param._wait_reduce_scatter(finalize_grad=finalize_after_drain)
            # Fallback inline-accumulation: only when finalize is requested, _wait_reduce_scatter
            # didn't already finalize, and an RS actually ran (rs_ticket set). Skips pure-AG
            # prefetches in _inflight_comm_params (no wgrad).
            need_fallback_accumulation = (
                finalize_after_drain
                and not getattr(param, "_already_finalized", False)
                and any(w._rs_ticket is not None for w in param._weights)
            )
            if need_fallback_accumulation:
                cache = get_global_GTP_cache()
                param.rs_event.wait()
                for w in param._weights:
                    w._set_rs_state(GTPWeightState.NONE)
                    wgrad_rs = cache.get(w._rs_ticket)
                    w.main_grad.add_(wgrad_rs)
                    cache.release(w._rs_ticket)
                    if hasattr(w, "grad_added_to_main_grad"):
                        w.grad_added_to_main_grad = True
                param._already_finalized = True


@dataclass
class BatchedNVFP4AllGatherAsyncHandle:
    """Handle for batched asynchronous NVFP4 all-gathers."""

    output_handles: List[_NVFP4AllGatherAsyncHandle]
    outer_async_handle: torch.distributed.Work
    _synchronized: bool = False

    def wait(self) -> None:
        """Wait for the async operation to complete and post-process the tensor."""
        if self._synchronized:
            return
        self.outer_async_handle.wait()
        # Fixes interleaved data for transposed tensor/scale inv and pads scale inv if needed.
        for output_handle in self.output_handles:
            if output_handle is not None:
                assert output_handle.async_handle is None
                output_handle.wait()
                # release any tensor references just in case
                output_handle.output = None
                output_handle.columnwise_data_interleaved = None
                output_handle.columnwise_scale_inv_interleaved = None

        self._synchronized = True


def grouped_gather_along_first_dim(
    weights: list,
    process_group,
    async_op: bool = False,
    quantizers: list = None,
    output_tensors: list = None,
):
    """All-gather multiple weights in one coalesced op; handles NVFP4 post-processing for both
    sync and async paths."""
    # Determine device from first weight.
    inp = weights[0]
    if isinstance(inp, NVFP4TensorStorage):
        device = (
            inp._rowwise_data.device
            if inp._rowwise_data is not None
            else inp._columnwise_data.device
        )
    else:
        device = inp.device

    weights_all = []
    weight_handles = []
    with torch.distributed._coalescing_manager(
        group=process_group, device=device, async_ops=async_op
    ) as gather_coalescing_manager:
        for i, weight in enumerate(weights):
            weight_all, weight_handle = gather_along_first_dim(
                weight,
                process_group,
                quantizer=quantizers[i],
                output_tensor=output_tensors[i] if output_tensors is not None else None,
                external_coalescing=True,
            )
            weights_all.append(weight_all)
            weight_handles.append(weight_handle)

    if async_op:
        handle = gather_coalescing_manager
        has_nvfp4_handles = any(isinstance(wh, _NVFP4AllGatherAsyncHandle) for wh in weight_handles)
        if has_nvfp4_handles:
            handle = BatchedNVFP4AllGatherAsyncHandle(weight_handles, handle)
    else:
        for wh in weight_handles:
            if isinstance(wh, _NVFP4AllGatherAsyncHandle):
                wh.wait()
        handle = None

    return weights_all, handle


class GTPEmbeddingWeight(torch.autograd.Function):
    """All-gather the embedding weight across the GTP group in forward, reduce-scatter its
    gradient in backward.

    The weight is stored sharded along the vocab dimension; this materializes the full weight
    for the lookup and distributes the gradient back to the shard.
    """

    @staticmethod
    def forward(ctx, weight):
        """All-gather the full embedding weight across the GTP group for the lookup."""
        ctx.save_for_backward(weight)
        return weight.all_gather_and_prefetch(fwd=True)

    @staticmethod
    def backward(ctx, grad_output):
        """Reduce-scatter the gradient back to this rank's vocab-dim shard."""
        (weight,) = ctx.saved_tensors
        return weight.wgrad_reduce_scatter(grad_output)


def reset_gtp_state():
    """Clear the process-global GTP prefetch-chain state (GTPShardedParam._chain_state /
    ._recompute_chain_state).

    These class-level dicts survive model teardown, so a GTP model rebuilt in-process would
    inherit stale last_weight pointers / flushed link tables. Call once before the per-chunk
    classify_gtp_chains loop (never inside it — chains span chunks). No-op on a fresh process.
    """
    GTPShardedParam._chain_state.clear()
    GTPShardedParam._recompute_chain_state.clear()


# ------------------------------------------------------------------------
# Distributed-checkpointing helpers
# ------------------------------------------------------------------------
# GTP shards axis 0 on top of TP, but the vanilla utils helpers only know TP, so their offsets
# miss the GTP slice. The helper below detects GTPShardedParam per-tensor and composes TP × GTP
# into one axis-0 offset (or two offsets), with replica_id = the DP-with-GTP-with-CP rank.


def make_sharded_tensors_for_checkpoint_with_gtp_remat(
    state_dict,
    prefix,
    tensor_parallel_layers_axis_map=None,
    sharded_offsets=(),
    extra_state_suffix="_extra_state",
    *,
    tp_group,
    dp_cp_group,
    intra_dp_cp_group=None,
):
    """GTP-aware analogue of make_sharded_tensors_for_checkpoint.

    Per-tensor (is_gtp_param): GTP tensors layer the axis-0 GTP split on the vanilla offsets (FP8
    shards dequantized to BF16 for save); non-GTP tensors delegate to the vanilla helper unchanged,
    so this is zero-cost when GTP is inactive.
    """
    from megatron.core.transformer.utils import (  # noqa: E402
        make_sharded_object_for_checkpoint,
        make_sharded_tensors_for_checkpoint,
    )
    from megatron.core.utils import (  # noqa: E402
        get_pg_rank,
        get_pg_size,
        make_sharded_tensor_for_checkpoint,
        make_tp_sharded_tensor_for_checkpoint,
    )

    # Fast path: no GTP-sharded params → defer to vanilla helper, same output.
    if not any(is_gtp_param(t) for t in state_dict.values()):
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            tensor_parallel_layers_axis_map,
            sharded_offsets,
            extra_state_suffix=extra_state_suffix,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )

    if tensor_parallel_layers_axis_map is None:
        tensor_parallel_layers_axis_map = {}

    tp_rank = get_pg_rank(tp_group)
    tp_size = get_pg_size(tp_group)
    # All GTP params in this state_dict share the same gtp_remat_group (set by the
    # wrap hook at module init), so pick it off the first GTP shard.
    gtp_remat_group = next(t.group for t in state_dict.values() if is_gtp_param(t))
    gtp_rank = get_pg_rank(gtp_remat_group)
    gtp_remat_size = get_pg_size(gtp_remat_group)

    # Replicate-group rank — the true replicas of a given GTP chunk live here.
    if intra_dp_cp_group is not None:
        dp_replica_rank = get_pg_rank(intra_dp_cp_group)
    else:
        from megatron.core import parallel_state  # noqa: E402

        dp_replica_rank = parallel_state.get_data_parallel_rank(
            with_context_parallel=True, with_gtp_remat=False
        )

    sharded_state_dict = {}
    for layer_name, tensor in state_dict.items():
        layer_key = f"{prefix}{layer_name}"
        is_gtp_weight_remat = is_gtp_param(tensor)

        if layer_name.endswith(extra_state_suffix):
            # ShardedObject (extra_state metadata): GTP-REPLICATED across the GTP group. Fold
            # gtp_rank into position 1 of the replica_id (PP, TP-replica-coord, DP) tuple so
            # GTP-peer ranks within the same TP slice get unique replica_ids.
            replica_id = (0, tp_rank * gtp_remat_size + gtp_rank, dp_replica_rank)
            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(
                tensor, layer_key, sharded_offsets, replica_id=replica_id
            )
            continue

        if not is_gtp_weight_remat:
            # Non-GTPShardedParam under a GTP-active module (e.g. bias): GTP-replicated, so GTP
            # ranks would collide on the same replica_id. Inject gtp_rank into replica_id
            # position 1 (same as the GTP-sharded branch below).
            if layer_name in tensor_parallel_layers_axis_map:
                replica_id = (0, gtp_rank, dp_replica_rank)
                sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                    tensor,
                    layer_key,
                    tp_axis=tensor_parallel_layers_axis_map[layer_name],
                    replica_id=replica_id,
                    prepend_offsets=sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=dp_cp_group,
                )
            else:
                replica_id = (0, tp_rank * gtp_remat_size + gtp_rank, dp_replica_rank)
                sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                    tensor,
                    layer_key,
                    replica_id=replica_id,
                    prepend_offsets=sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=dp_cp_group,
                )
            continue

        # GTP-sharded tensor: delegate to the GTP-aware single-tensor helper — it layers the
        # axis-0 GTP split onto TP, elects the writer over the gtp_remat-excluded DP group, and sets
        # allow_shape_mismatch for alignment padding. (tp_axis None → 0; tp_size 1 when no TP.)
        tp_axis = tensor_parallel_layers_axis_map.get(layer_name, None)
        sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
            tensor,
            layer_key,
            tp_axis=tp_axis if tp_axis is not None else 0,
            prepend_offsets=sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )

    return sharded_state_dict
