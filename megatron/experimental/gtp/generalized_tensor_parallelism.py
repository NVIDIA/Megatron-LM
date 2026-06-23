# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Generalized Tensor Parallelism (GTP).

Shards weight tensors 1/N across a GTP process group along ``out_features``
and materializes them on-demand via async all-gather, with a per-weight
prefetch chain + ticket-based buffer cache co-designed for CUDA graph
capture/replay.  Quantized AG (FP8 / MXFP8 / NVFP4) composes with the
sharding for compounding bandwidth reduction.
"""

import math
import os
import re
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import torch
from packaging.version import Version

_GTP_TE_MIN_VERSION = Version("2.17")

try:
    import transformer_engine as te  # noqa: F401

    _te_version = Version(te.__version__)
    if _te_version < _GTP_TE_MIN_VERSION and not os.environ.get("MEGATRON_GTP_FORCE_ENABLE"):
        raise ImportError(
            f"megatron.experimental.gtp requires TransformerEngine >= {_GTP_TE_MIN_VERSION} "
            f"(found {_te_version}). Set MEGATRON_GTP_FORCE_ENABLE=1 to bypass this check "
            "when using a custom TE build that includes the GTP hook registry."
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
except (ImportError, ModuleNotFoundError) as _gtp_te_import_err:
    raise ImportError(
        "megatron.experimental.gtp requires TransformerEngine with FP8 / MXFP8 / "
        "NVFP4 tensor primitives. Original error: " + str(_gtp_te_import_err)
    ) from _gtp_te_import_err


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

    Call once at init, AFTER set_cuda_graph_modules() and BEFORE the first fwd
    of any graphed param. Raises if an already chain-initialized param would
    be reclassified into a different chain (its prev/next links are already
    wired into the wrong list).
    """
    conflicts = []
    for name, param in model.named_parameters():
        if not isinstance(param, GTPShardedParam):
            continue
        target = _classify_param_chain(name).value
        if param.prefetch_initialized and param.chain_id != target:
            conflicts.append((name, param.chain_id, target))
            continue
        param.chain_id = target

        # Bwd-prefetch opt-out: embedding.word_embeddings.weight does not need
        # an AG in the bwd pass (its wgrad is a scatter-add on sharded rows
        # and its input has no dgrad). Skipping its bwd AG saves one collective.
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
    """Get a pool buffer or allocate fresh. Tagged so _wgrad_pool_put accepts
    only pool-owned buffers — callers that don't use _wgrad_pool_get (e.g.
    Megatron layers.py wgrad GEMM, aten F.embedding bwd) fall through to the
    caching allocator on release."""
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

    Two partitioning axes:
      - chain_id: captured (GRAPHED) vs eager (UNGRAPHED) ops must not share
        a stream (eager ops would contaminate capture/replay state).
      - group: independent NCCL communicators (e.g. GTP vs EGTP) get their
        own user-level stream to avoid cross-group serialization.
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

    Drains in-flight AG/RS on the side streams (eager expert backward may still be writing
    main_grad) and waits on each CUDA-graph runner's captured dense-GTP bwd Phase 2
    (main_grad.add_) completion event. No-op when GTP is inactive (empty streams / events).
    """
    wait_async_comms()
    cur = torch.cuda.current_stream()
    for s in _AG_STREAMS.values():
        cur.wait_stream(s)
    for s in _RS_STREAMS.values():
        cur.wait_stream(s)
    # Local import: cuda_graphs imports this module, so a module-level import would be circular.
    from megatron.core.transformer.cuda_graphs import get_gtp_phase2_completion_events

    for evt in get_gtp_phase2_completion_events():
        cur.wait_event(evt)


@dataclass
class GTPConfig:
    """Global configuration for Generalized Tensor Parallelism."""

    pad_for_alignment: int = 16
    check_param_states: bool = False
    weight_prefetch: bool = True
    # When True (default), wgrad reduce-scatter for non-chain-head GTP
    # params uses async_op=True; finalize (handle.wait + main_grad.add_)
    # runs in the cascade walk of a later bwd call, allowing RS-compute
    # overlap. When False, every wgrad RS is synchronous and finalizes
    # inline, at the cost of that overlap.
    async_reduction: bool = True
    # GTP companion to Megatron --fp8-param-gather: optimizer casts FP32 master
    # directly into GTPShardedParam.quantized; forward's _quantize_if_needed
    # short-circuits to the cached FP8. Moves BF16->FP8 off the fwd critical path.
    fp8_param_gather: bool = False


GTP_CONFIG = GTPConfig()


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
        if isinstance(param, GTPShardedParam):
            param._debug_name = name


def _gtp_slice_one_param(param, gtp_group, *, name="<unnamed>"):
    """Pad + slice a full-size BF16 weight to this rank's GTP shard.

    Caller attaches GTP attrs (see _gtp_attach_attrs). When called from the
    legacy post-init path under fp8_model_init, tensor may be a
    QuantizedTensor — F.pad dequantizes it before slicing.
    """
    gtp_size = gtp_group.size()
    gtp_rank = gtp_group.rank()
    tensor = param.data

    if GTP_CONFIG.pad_for_alignment > 0:
        # Pad before slicing so shards stay alignment-divisible and padding
        # ends up contiguous at the tail of the gathered result.
        alignment = GTP_CONFIG.pad_for_alignment * gtp_size
        dim0 = tensor.shape[0]
        pad_length = (alignment - dim0 % alignment) % alignment
        if pad_length > 0:
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, pad_length))
    else:
        # No-pad mode: dim-0 must divide gtp_size or AG output loses tail rows.
        assert tensor.shape[0] % gtp_size == 0, (
            f"_gtp_slice_one_param: {name}.shape[0]={tensor.shape[0]} is not "
            f"divisible by gtp_size={gtp_size}. Either enable padding by "
            "setting GTP_CONFIG.pad_for_alignment > 0, or ensure the weight's "
            "dim-0 is a multiple of the GTP group size."
        )
        pad_length = 0

    shard_size = tensor.shape[0] // gtp_size
    shard = tensor[gtp_rank * shard_size : (gtp_rank + 1) * shard_size]
    gtp_shard = GTPShardedParam(shard.clone())
    gtp_shard.pad_length = pad_length
    # Preserve the source weight's tensor-model-parallel attributes (dropped when wrapping
    # into GTPShardedParam). GTP only shards TP-parallel linears, so this keeps the param
    # correctly classified by param_is_not_tensor_parallel_duplicate without GTP-specific code.
    from megatron.core.tensor_parallel import copy_tensor_model_parallel_attributes

    copy_tensor_model_parallel_attributes(gtp_shard, param)
    return gtp_shard


def _gtp_attach_attrs(gtp_shard, gtp_group, *, is_grouped=False, expert_idx=0):
    """Attach group / gtp_size / routed-expert tags and register in _GTP_PARAMS.

    Kept separate from _gtp_slice_one_param so attrs land on the post-quantize
    param (when quantize fires between slice and attach).
    """
    if is_grouped:
        gtp_shard.expert_idx = expert_idx
        gtp_shard.is_routed_expert = True
        # Default to UNGRAPHED; classify_gtp_chains() reclassifies based on the
        # cuda_graph_modules at init time.
        gtp_shard.chain_id = GTPChain.UNGRAPHED.value
    gtp_shard.group = gtp_group
    gtp_shard.gtp_size = gtp_group.size()
    global _GTP_PARAMS
    _GTP_PARAMS.append(gtp_shard)


def wrap_module_params_gtp(module, weight_names, gtp_group, is_grouped=None):
    """Shard and re-register module params as GTPShardedParam.

    Two call paths:
    1. Megatron-style modules (ColumnParallelLinear, etc.): full post-init slice.
    2. TE modules: per-param body no-ops because the reset_parameters hook
       already produced GTPShardedParam instances.

    """
    if gtp_group.size() == 1:
        return

    for idx, name in enumerate(weight_names):
        param = getattr(module, name, None)
        if param is None:
            continue

        # TE-side hook already sliced this one.
        if isinstance(param, GTPShardedParam):
            continue

        # delete the original parameter, which will be replaced by an GTP sharded one
        delattr(module, name)
        gtp_shard = _gtp_slice_one_param(param, gtp_group, name=name)
        del param
        _gtp_attach_attrs(gtp_shard, gtp_group, is_grouped=bool(is_grouped), expert_idx=idx)
        # register the newly sharded param back to the module
        module._parameters[name] = gtp_shard

    if is_grouped:
        allweights = [getattr(module, name) for name in weight_names]
        allweights[0].weight_list = allweights


def gtp_slice_in_reset_parameters(module, name, param, expert_idx=0):
    """Slice + attach attrs for one param. Called between init_fn(param) and
    the optional quantizer(param) in TransformerEngineBaseModule.reset_parameters.

    Only fires for params in module.weight_names (the GEMM weights);
    layer-norm gammas, biases, etc. are left full-size.

    Returns the new GTPShardedParam or None (GTP not active for this param).
    """
    gtp_group = getattr(module, "_gtp_group", None)
    if gtp_group is None or gtp_group.size() == 1:
        return None
    weight_names = getattr(module, "weight_names", None)
    if weight_names is None or name not in weight_names:
        return None
    is_grouped = bool(getattr(module, "_gtp_is_grouped", False))
    gtp_shard = _gtp_slice_one_param(param, gtp_group, name=name)
    _gtp_attach_attrs(gtp_shard, gtp_group, is_grouped=is_grouped, expert_idx=expert_idx)
    return gtp_shard


def gtp_finalize_module_in_reset_parameters(module, weight_names):
    """GroupedLinear-only: attach weight_list to expert 0's shard for batched
    all-gather. No-op when module._gtp_is_grouped is False.
    """
    if not getattr(module, "_gtp_is_grouped", False):
        return
    gtp_group = getattr(module, "_gtp_group", None)
    if gtp_group is None or gtp_group.size() == 1:
        return
    allweights = [getattr(module, n) for n in weight_names]
    if allweights:
        allweights[0].weight_list = allweights


class GTPShardHandle:
    """Wrapper around a ``dist`` async-work handle for a GTP AG / RS.

    Tracks the participating shards so the wait-site can transition their
    ``GTPWeightState`` and so the GTP module can prune the param from
    ``_inflight_comm_params`` when the collective completes.
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


class GTPShardedParam(torch.nn.Parameter):
    """A weight parameter sharded 1/N across a GTP process group.

    Materialized on-demand via async all-gather and gradient-reduced via
    reduce-scatter.  Carries its own prefetch-chain wiring (``prev_w`` /
    ``next_w``), per-chain state, AG/RS cache tickets, and the metadata the
    integrator needs to drive overlap with captured compute.
    """

    # Per-chain linked-list state, keyed by chain_id (GTPChain.GRAPHED/UNGRAPHED); chains
    # never cross-link (prev_w/next_w join only same-chain_id params). Call reset_gtp_state()
    # before rebuilding a GTP model in the same process.
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
        """Buffer one row of the prefetch-link table (flushed atomically on the second forward pass)."""
        _W = 70

        def _layer_id(name: str) -> str:
            m = re.search(r"\d+", name)
            return m.group() if m else "-"

        chain["link_node_count"] += 1
        if chain["link_node_count"] == 1:
            chain_id = getattr(curr, "chain_id", GTPChain.UNGRAPHED.value)
            chain["link_table_buffer"].append(
                f"\n[{chain_id} chain]\n{'node_id':>7} | {'layer_id':>8} |"
                f" {'curr_weight_name':<{_W}} |"
                f" prev_weight_name\n{'-'*7}-+-{'-'*8}-+-{'-'*_W}-+-{'-'*_W}"
            )
            # Seed weight (first GTP param) as row 0
            chain["link_table_buffer"].append(
                f"{'0':>7} | {_layer_id(prev._debug_name):>8} | {prev._debug_name:<{_W}} | -"
            )
        chain["link_table_buffer"].append(
            f"{chain['link_node_count']:>7} | {_layer_id(curr._debug_name):>8} | "
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

        # Canonical flag — also set on distopt's main_param copy so both kinds
        # of param can be classified via a single attribute check.
        self.is_gtp = True

        # all gather
        self.state = GTPWeightState.NONE
        self._ag_ticket_fwd = None
        self._ag_ticket_bwd = None
        self._prefetch_handle = None
        self._need_weight_prefetch = True
        # Per-direction prefetch opt-outs. Default True. The embedding weight
        # never needs an AG during bwd (its wgrad is a scatter-add indexed by
        # token ids, and its input is non-differentiable, so no dgrad either).
        # classify_gtp_chains() sets this to False for embedding.word_embeddings.weight.
        self._need_weight_prefetch_bwd = True
        self.ag_event = torch.cuda.Event(external=True)
        # DDP backward hook (set by register_grad_accum_hook); invoked after
        # the wgrad RS accumulation completes (Graphed.backward / chain cascade).
        self._grad_accum_hook = None
        # Quantization
        self._quantizer = None
        self.did_cast_to_low_precision = False
        self.quantized = None
        # Prefetching linked list
        self.prefetch_initialized = False
        self.next_w = None
        self.prev_w = None
        # Recompute-forward prefetch chain: a SEPARATE chain (own slot) linking
        # the weights re-gathered rowwise during an activation-recompute forward
        # in backward. Kept distinct from state/_prefetch_handle/ag_event above so
        # it never clobbers the concurrent columnwise dgrad lifecycle of the same
        # weight. Self-populates lazily from the first backward's recompute-fwd
        # gathers (see all_gather_and_prefetch).
        self._recompute_initialized = False
        self._recompute_next = None
        self._recompute_prev = None
        self._recompute_prefetch_handle = None
        self._recompute_ag_event = torch.cuda.Event(external=True)
        self._recompute_already_drained = False
        # Chain identity (GTPChain.GRAPHED / GTPChain.UNGRAPHED). Defaults to
        # UNGRAPHED as a safe fallback; classify_gtp_chains(model) walks the
        # model at init time (after set_cuda_graph_modules) and reclassifies
        # based on param name + active cuda_graph_modules.
        self.chain_id = GTPChain.UNGRAPHED.value
        # Grouped gemm
        self.is_routed_expert = False
        self.expert_idx = None
        self.group = None
        self.weight_list = None
        # Reduce-scatter state (set during wgrad_reduce_scatter)
        self.rs_state = GTPWeightState.NONE
        self._wgrad_rs_handle = None
        self.rs_event = torch.cuda.Event(external=True)
        self._rs_ticket = None
        # Padding
        self.pad_length = 0
        # Debug
        self._debug_name = ""
        # Hot-path caches (populated lazily on first use).  chain_id/group are
        # set after __init__, so we can't resolve streams eagerly here.
        self._cached_ag_stream = None
        self._cached_rs_stream = None
        self._cached_quantizers = None
        self._cached_dtypes = None
        self._cached_gtp_group = None

    def setup(self, weight_quantizer=None):
        """Set quantizer and create quantized shard."""

        if self._quantizer is None:

            def _configure_quantizer(q, group):
                q = q.copy()
                if hasattr(q, "with_amax_reduction"):
                    q.with_amax_reduction = True
                    q.amax_reduction_group = group
                q.internal = False
                # MXFP8 scales must stay in compact (unswizzled) layout so that
                # per-shard scale_inv can be all-gathered via byte concatenation.
                # GEMM-swizzled scales from independent shards don't compose into
                # a valid swizzled layout for the full tensor after AG.
                q.optimize_for_gemm = not isinstance(q, MXFP8Quantizer)
                return q

            weights = (
                self.weight_list
                if self.is_routed_expert and self.weight_list is not None
                else [self]
            )
            for quantizer, weight in zip(weight_quantizer, weights):
                if quantizer is None:
                    continue

                weight._quantizer = _configure_quantizer(quantizer, weight.group)
                # This init quantize is the only allocation of the quantized storage
                # (re-quantize writes in place), so route it via _graphed_alloc.
                with _graphed_alloc(getattr(weight, "chain_id", GTPChain.UNGRAPHED.value)):
                    weight.quantized = weight._quantizer.quantize(weight.get_padded_shard())
                weight.quantized.is_routed_expert = getattr(weight, "is_routed_expert", False)
                # fp8_param_gather: the init quantize above already produced a
                # valid FP8 cache from the BF16 shard; flag did_cast so iter-0's
                # forward _quantize_if_needed short-circuits and the redundant
                # BF16->FP8 cast on iter 0 is skipped.
                if GTP_CONFIG.fp8_param_gather:
                    weight.did_cast_to_low_precision = True

    @property
    def _weights(self):
        """Return the list of individual weight shards (self for non-routed, weight_list for routed)."""
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
        """Build cache key using output shape + dtype.

        Weights with matching gathered shape and dtype share a buffer.
        For expert weights gathered in parallel, self.expert_idx distinguishes them so
        each gets a distinct buffer, while same-indexed experts across layers share.
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

    def _quantize_if_needed(self, skip_weight_cast=False, cast_noop_flag=None):
        """Re-quantize sharded weight into existing buffer. Returns quantized weight or self."""
        if self._quantizer is None:
            self.did_cast_to_low_precision = False
            return self

        # fp8_param_gather fast-path: optimizer already filled self.quantized;
        # reuse it and keep BF16->FP8 off the forward critical path.
        if GTP_CONFIG.fp8_param_gather and self.did_cast_to_low_precision:
            return self.quantized

        self._quantizer.set_usage(rowwise=True, columnwise=True)
        if skip_weight_cast is False or cast_noop_flag is not None:
            tex.quantize(
                tensor=self.get_padded_shard(),
                quantizer=self._quantizer,
                output=self.quantized,
                noop=cast_noop_flag,
            )
        self.did_cast_to_low_precision = True

        return self.quantized

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

    def _all_gather_weight(self, async_op, skip_weight_cast, cast_noop_flag, fwd, nvtx_label=None):
        """Quantize (if needed) and all-gather weight. Returns (weight_total, handle)."""
        if nvtx_label is None:
            nvtx_label = (
                self._debug_name + (".fwd" if fwd else ".bwd") + (".async" if async_op else ".sync")
            )
        nvtx_range_push(f"{nvtx_label}.all_gather_weight")

        weights = self._weights

        # 1. Transition state for async gathers. Skip during a recompute-forward:
        #    it gathers this weight rowwise (into _ag_ticket_fwd) while a bwd-chain
        #    prefetch may hold an in-flight columnwise AG state on the same weight
        #    (separate _ag_ticket_bwd) — clobbering it would break the dgrad consume.
        if GTP_CONFIG.check_param_states and not in_fp8_activation_recompute_phase():
            new_state = GTPWeightState.ASYNC_WAIT if async_op else GTPWeightState.DATA_READY_SYNC
            for w in weights:
                w._set_state(new_state)

        # 2. Prepare: quantize, set usage direction.
        fp8_pg_hit = GTP_CONFIG.fp8_param_gather and self.did_cast_to_low_precision

        if not fp8_pg_hit:
            for w in weights:
                w._quantize_if_needed(skip_weight_cast, cast_noop_flag)

        for w in weights:
            if w.did_cast_to_low_precision:
                w._quantizer.set_usage(rowwise=fwd, columnwise=not fwd)

        # 3. Build gather inputs.
        # quantizers / dtypes / gtp_group are stable after model construction —
        # cache on the anchor (self == weights[0]) to avoid rebuilding lists
        # every call.  w.quantized is NOT cached because it can rebind.
        quantizers = self._cached_quantizers
        if quantizers is None:
            quantizers = [w._quantizer for w in weights]
            self._cached_quantizers = quantizers
        if weights[0].did_cast_to_low_precision:
            gather_weights = [w.quantized for w in weights]
        else:
            gather_weights = list(w.get_padded_shard() for w in weights)

        # 4. Cache checkout — use pooled buffers for both async and sync gathers
        #    to avoid allocating fresh memory each iteration.
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
        gtp_group = self._cached_gtp_group
        if gtp_group is None:
            gtp_group = weights[0].group
            self._cached_gtp_group = gtp_group
        if GTP_CONFIG.check_param_states and len(gather_weights) > 1:
            # Debug invariant: batched AG needs distinct output buffers per expert.
            assert len(set(id(b) for b in out_buffers)) == len(
                out_buffers
            ), "Duplicate output buffers in batched all-gather — experts need distinct cache keys"

        # ASYNC AG: wrap issue on ag_stream — ag_stream's tail then reflects
        # the collective's full lifecycle (what external wait_stream(ag_stream)
        # drains depend on). The explicit outer→ag_stream sync event preserves
        # the upstream quantize writer edge that the bare stream context would
        # drop; held on self so PyTorch's event pool can't recycle the handle
        # between capture and replay.
        # SYNC AG: stay on caller — output ready on return.
        if async_op:
            outer_stream = torch.cuda.current_stream()
            ag_stream = get_ag_stream(self.chain_id, gtp_group)
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
                    gtp_group,
                    async_op=async_op,
                    quantizers=quantizers,
                    output_tensors=out_buffers,
                )
                nvtx_range_pop(f"{nvtx_label}.batched_gtp_ag")
            else:
                nvtx_range_push(f"{nvtx_label}.gtp_ag")
                weight_total, handle = gather_along_first_dim(
                    gather_weights[0],
                    gtp_group,
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

    def _all_gather_weight_on_demand(self, fwd, skip_weight_cast=False, cast_noop_flag=None):
        result, _ = self._all_gather_weight(
            async_op=False,
            skip_weight_cast=skip_weight_cast,
            cast_noop_flag=cast_noop_flag,
            fwd=fwd,
        )
        result = result if self.is_routed_expert else [result]
        result = [self._strip_padding(r) for r in result]
        result = [r.detach().requires_grad_(w.requires_grad) for r, w in zip(result, self._weights)]
        return result if self.is_routed_expert else result[0]

    def _get_prefetched_weight(self, fwd, skip_weight_cast=False, cast_noop_flag=None):
        # ``skip_weight_cast`` and ``cast_noop_flag`` are accepted to keep the
        # signature symmetric with ``_all_gather_weight_on_demand``.
        del skip_weight_cast, cast_noop_flag
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
            # Producer already drained via wait_async_comms; skip the captured
            # cross-graph wait (CUDA no-op anyway). Correctness is provided by
            # the eager main_stream sync chain in the surrounding training loop.
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
        # Issue target's rowwise (fwd) AG into its recompute slot. _all_gather_weight
        # skips the AG-state transition under recompute, so the dgrad `state` of
        # target is untouched; result lands in target._ag_ticket_fwd.
        _, handle = target._all_gather_weight(
            async_op=True,
            skip_weight_cast=True,
            cast_noop_flag=None,
            fwd=True,
            nvtx_label=nvtx_label,
        )
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
        """
        Backward variant: get current weight (from cache if prefetched, else
        sync gather) and async-prefetch prev_w.

        Safe thanks to the coat-check cache: get() returns the current buffer
        to the pool, and the prefetch's checkout() will allocate a separate
        buffer if the pool is empty (i.e. the current buffer is still live
        via the caller's tensor reference).

        Returns:
            weight_total
        """

        if GTP_CONFIG.weight_prefetch and self.next_w is not None:
            result = self._get_prefetched_weight(False, skip_weight_cast=True)
        else:
            result = self._all_gather_weight_on_demand(False, skip_weight_cast=True)

        if (
            GTP_CONFIG.weight_prefetch
            and self.prev_w is not None
            and self.prev_w._need_weight_prefetch
            and self.prev_w._need_weight_prefetch_bwd
        ):
            # Pre-AG work (quantize, ticket lookup) runs on caller's stream;
            # the NCCL collective itself is wrapped on ag_stream inside
            # _all_gather_weight (see the async/sync gate there for rationale).
            _, handle = self.prev_w._all_gather_weight(
                async_op=True,
                skip_weight_cast=True,
                cast_noop_flag=None,
                fwd=False,
                nvtx_label=nvtx_label,
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

    def all_gather_and_prefetch(
        self,
        fwd: bool = True,
        skip_weight_cast: bool = False,
        cast_noop_flag: torch.Tensor = None,
        nvtx_label: str = None,
    ):
        """
        All-gather current weight and async-prefetch the next weight.

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
            result = self._get_prefetched_weight(True, skip_weight_cast, cast_noop_flag)
        else:
            # On-demand: chain head (fwd or recompute global-first) or first-iter build.
            result = self._all_gather_weight_on_demand(True, skip_weight_cast, cast_noop_flag)

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
                async_op=True,
                skip_weight_cast=skip_weight_cast,
                cast_noop_flag=cast_noop_flag,
                fwd=fwd,
                nvtx_label=nvtx_label,
            )
            self.next_w._prefetch_handle = handle

        # The unsharded tensor has been returned, no pending work so reset state to NONE.
        # Skip during recompute: a bwd-chain prefetch may hold an in-flight AG state on
        # this weight that its later dgrad consume still needs.
        if GTP_CONFIG.check_param_states and not in_recompute:
            for w in self._weights:
                w._set_state(GTPWeightState.NONE)

        cls = type(self)

        # Lazy-build the recompute-forward prefetch chain (first backward, in
        # recompute order). Consume/prefetch above used the prior iteration's links,
        # so the first backward runs on-demand while these links are established.
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

            # Set the fwd ag buffer
            quantizers = [w._quantizer for w in self._weights]
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
            print_rank_0("\n".join(chain["link_table_buffer"]) + "\n")

        return result

    def batched_all_gather_and_prefetch(self, **kwargs):
        """Batched all-gather + prefetch for expert weights. Wrapper around all_gather_and_prefetch."""
        assert self.is_routed_expert and self.weight_list is not None
        return self.all_gather_and_prefetch(**kwargs)

    def get_wgrad_tensor(self):
        """Pool-allocate a wgrad scratch tensor of unsharded shape for the bwd GEMM."""
        return _wgrad_pool_get(self._unsharded_shape, self.main_grad.dtype, self.device)

    def register_grad_accum_hook(self, grad_accum_node, hook):
        """Register a DDP backward hook to be called after the wgrad RS finalize.

        For GTP params, autograd may receive None (async RS) so the normal grad
        accumulator hook never fires. Instead, the integrator (Graphed.backward
        for captured chains, or the eager chain-tail cascade) calls this hook
        explicitly after RS wait + gradient accumulation, ensuring DDP's
        register_grad_ready fires at exactly the right time.

        ``grad_accum_node`` is accepted for caller-API compatibility but the
        node itself is not retained — only the hook callable.
        """
        del grad_accum_node
        self._grad_accum_hook = hook

    @staticmethod
    def _handle_megatron_grad_accum(param):
        """Handle megatron DDP and gradient accumulation fusion.

        Do NOT set param.grad before calling the hook — the hook checks
        param.grad and would accumulate it into main_grad if zero_out_wgrad
        is True, corrupting the gradient with a non-zero dummy.
        """
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = True
        dummy_grad = get_dummy_wgrad(list(param.main_grad.shape), param.dtype)
        if getattr(param, "_grad_accum_hook", None) is not None:
            param._grad_accum_hook()

        param._set_rs_state(GTPWeightState.NONE)
        return dummy_grad

    def _wait_reduce_scatter(self, finalize_grad=False):
        # Enter rs_stream context so handle.wait() + rs_event.record() land
        # on rs_stream — mirrors _wait_param_gather for the RS path.
        # When finalize_grad=True, main_grad.add_ also runs on rs_stream
        # (right after NCCL RS), so it starts during AG drain rather than
        # after it — avoids SM-saturation blocking cross-graph overlap.
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

    def _reduce_scatter(self, wgrads, async_op, nvtx_label=None):
        """Reduce-scatter one or more wgrads. Returns (outputs, handle).

        Single tensor: plain reduce-scatter (no coalescing).
        Multiple tensors: coalesced reduce-scatter.
        """
        if nvtx_label is None:
            nvtx_label = self._debug_name + ".bwd" + (".async" if async_op else ".sync")

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

        # ASYNC RS: wrap issue on rs_stream — rs_stream's tail then reflects
        # the collective's full lifecycle (what external wait_stream(rs_stream)
        # drains depend on). The explicit outer→rs_stream sync event preserves
        # the wgrad-GEMM writer edge that the bare stream context would drop;
        # held on self so PyTorch's event pool can't recycle the handle
        # between capture and replay. Mirrors AG path.
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
        """Reduce-scatter wgrad(s). Sync for last weight, async+deferred for others.

        Accepts a single tensor (non-routed) or list of tensors (routed experts).

        Returns:
            Single tensor or list for sync (last weight) — backward should return this.
            None or tuple of Nones for async — backward should return this.
        """
        batched = isinstance(wgrad, (list, tuple))
        wgrads = list(wgrad) if batched else [wgrad]
        weights = self._weights

        # UNGRAPHED-chain wgrads are recycled via the standalone pool (_wgrad_pool_put).
        # GRAPHED-chain wgrads cannot pool-recycle because CUDA graphs require
        # stable buffer addresses across replay.
        poolable = self.chain_id == GTPChain.UNGRAPHED.value

        if GTP_CONFIG.async_reduction and self.prev_w is not None:
            # Async reduce-scatter (not last weight — deferred finish). Pre-RS
            # work on caller; NCCL wrap lives at the collective site inside
            # _reduce_scatter (mirrors the AG prefetch sites).
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

    def get_data_tensors(self):
        """Expose self as the lone data tensor for TE's offload-marking interface.

        TE's ``mark_activation_offload`` treats any non-plain tensor as a storage
        wrapper and calls ``get_data_tensors()`` on it; a sharded param has no inner
        buffers, so it is its own data tensor.
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


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            print(message, flush=True)
    elif torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


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


# CUDA-graph memory pool for routing GRAPHED-chain allocations (AG/RS buffers and
# quantized weight storage) into the capture pool *at creation time*, so no post-hoc
# reallocation is needed. Registered by the integrator (set_cuda_graph_mempool) after
# the pool is created and before the first graphed forward; stays None when CG is off,
# in which case _graphed_alloc is a no-op and allocations use regular memory.
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
    """
    Ticket-based buffer pool for GTP all-gather / reduce-scatter buffers.

    - ``reserve(param, dtype, fwd)`` → ``ticket``
      Assigns a persistent ticket (no buffer allocated yet).
    - ``get(ticket)`` → ``buffer``
      Returns the buffer, lazily allocating from pool or fresh if needed.
    - ``release(ticket)``
      Returns the buffer to the pool.  Ticket remains valid; next ``get()``
      will re-allocate from the pool.
    - ``clear()``
      Drops all buffers and pools.  Tickets remain valid; next ``get()``
      lazily allocates fresh buffers.
    """

    # Bytes per element for known dtypes (used for logging).  Add new entries
    # here when GTP starts caching buffers of additional quantized dtypes.
    # Only DType values guaranteed exposed by the TE pybind bindings — verify
    # via ``hasattr(tex.DType, ...)`` before adding speculative entries.
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
                quantizer = param._quantizer
                assert quantizer is not None
                param._quantizer.set_usage(rowwise=fwd, columnwise=not fwd)

                buf = param._quantizer.make_empty(
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
        print_rank_0(
            f"[GTP Cache] +{buf_bytes / 1024**2:.1f} MB  (shape={out_shape}, dtype={dtype})  "
            f"total={self._total_bytes / 1024**2:.1f} MB  param: {param._debug_name} fwd: {fwd}"
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
        """Return the buffer to the pool.  Ticket remains valid.

        slot.buf is intentionally NOT cleared: get() must stay idempotent so that
        CUDA-graph-captured buffers keep their fixed address across replays.
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

    When called inside CUDA graph capture, the drains are captured into that
    graph. This is the producer-side hook for cross-graph AG/RS overlap:
    captured cudaStreamWaitEvent on an event recorded in a different capture
    session is a CUDA no-op, so consumer graphs can't safely wait on
    cross-graph events. Instead, the producer drains here and flags the
    param; the consumer reads the flag and skips its captured wait.

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
            # Fallback inline-accumulation: only when finalize is requested,
            # _wait_reduce_scatter didn't already finalize, and an RS actually
            # ran for this param (rs_ticket set).  Skips pure-AG prefetches in
            # _inflight_comm_params (no wgrad to accumulate).
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
                output_handle.post_process_nvfp4_gather()
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
    """
    All-gather multiple weights in a single coalesced operation.

    Handles NVFP4 post-processing for both sync and async paths.
    """
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
                grouped=True,
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
                wh.post_process_nvfp4_gather()
        handle = None

    return weights_all, handle


class GTPEmbeddingWeight(torch.autograd.Function):
    """All-gather the embedding weight across the GTP group in forward, and
    reduce-scatter its gradient back in backward.

    The embedding weight is stored sharded along the vocab dimension across
    the GTP group; this autograd function materializes the full weight for
    the embedding lookup and distributes the gradient back to the shard.
    """

    @staticmethod
    def forward(ctx, weight):
        ctx.save_for_backward(weight)
        return weight.all_gather_and_prefetch(fwd=True)

    @staticmethod
    def backward(ctx, grad_output):
        (weight,) = ctx.saved_tensors
        return weight.wgrad_reduce_scatter(grad_output)


def reset_gtp_state():
    """Clear the process-global GTP prefetch-chain state (``GTPShardedParam._chain_state`` /
    ``._recompute_chain_state``).

    These class-level dicts survive model teardown, so a GTP model rebuilt in the same process
    would otherwise inherit the prior model's stale ``last_weight`` pointers / flushed link
    tables. Call once before the per-chunk ``classify_gtp_chains`` loop (never inside it — chains
    span chunks). No-op on a fresh process.
    """
    GTPShardedParam._chain_state.clear()
    GTPShardedParam._recompute_chain_state.clear()


def reset_gtp_quantize_cache(model):
    """Invalidate the per-shard low-precision cache after a checkpoint load.

    DCP load copies new data into ``GTPShardedParam.data`` in-place, leaving
    the stale FP8 / MXFP8 / NVFP4 buffer in ``self.quantized`` behind. Call
    this once after ``load_state_dict`` / ``dist_checkpointing.load`` so the
    next forward re-quantizes from the freshly-loaded high-precision weight.
    """
    for param in model.parameters():
        if isinstance(param, GTPShardedParam):
            param.did_cast_to_low_precision = False


# ------------------------------------------------------------------------
# Distributed-checkpointing helpers
# ------------------------------------------------------------------------
#
# GTP shards weights further along axis 0 on top of TP. The vanilla helpers
# in ``megatron.core.transformer.utils`` only know about TP, so the
# ShardedTensor offsets they emit don't reflect the GTP slice the local
# rank actually owns. The helper below detects GTPShardedParam per-tensor
# and either composes TP × GTP into a single axis-0 offset (when TP is
# also on axis 0) or emits two offsets (TP on its axis, GTP on axis 0).
# ``replica_id`` uses the DP-with-GTP-with-CP rank — the set of ranks that
# actually hold identical copies of this chunk.
#


def make_sharded_tensors_for_checkpoint_with_gtp(
    state_dict,
    prefix,
    tensor_parallel_layers_axis_map=None,
    sharded_offsets=(),
    extra_state_suffix="_extra_state",
    *,
    tp_group,
    dp_cp_group,
    intra_dp_cp_no_gtp_group=None,
):
    """GTP-aware analogue of ``make_sharded_tensors_for_checkpoint``.

    Detects GTP sharding per-tensor (via ``isinstance(tensor, GTPShardedParam)``).
    Non-GTP tensors keep the vanilla offsets exactly; GTP tensors layer the
    GTP axis-0 split on top.

    No-op (delegates to the vanilla helper) when no tensor in ``state_dict``
    is a ``GTPShardedParam``, so plumbing this helper through has zero cost
    when GTP is inactive.
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
    if not any(isinstance(t, GTPShardedParam) for t in state_dict.values()):
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
    # All GTP params in this state_dict share the same gtp_group (set by the
    # wrap hook at module init), so pick it off the first GTP shard.
    gtp_group = next(t.group for t in state_dict.values() if isinstance(t, GTPShardedParam))
    gtp_rank = get_pg_rank(gtp_group)
    gtp_size = get_pg_size(gtp_group)

    # DP-with-GTP-with-CP rank — replicas of a given GTP chunk live here.
    if intra_dp_cp_no_gtp_group is not None:
        dp_no_gtp_rank = get_pg_rank(intra_dp_cp_no_gtp_group)
    else:
        from megatron.core import parallel_state  # noqa: E402

        dp_no_gtp_rank = parallel_state.get_data_parallel_rank(
            with_context_parallel=True, no_gtp=True
        )

    sharded_state_dict = {}
    for layer_name, tensor in state_dict.items():
        layer_key = f"{prefix}{layer_name}"
        is_gtp = isinstance(tensor, GTPShardedParam)

        if layer_name.endswith(extra_state_suffix):
            # ShardedObject (extra_state metadata): GTP-REPLICATED across the GTP group. Fold
            # gtp_rank into position 1 of the replica_id (PP, TP-replica-coord, DP) tuple so
            # GTP-peer ranks within the same TP slice get unique replica_ids.
            replica_id = (0, tp_rank * gtp_size + gtp_rank, dp_no_gtp_rank)
            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(
                tensor, layer_key, sharded_offsets, replica_id=replica_id
            )
            continue

        if not is_gtp:
            # Non-GTPShardedParam under a GTP-active module (e.g. bias). The tensor is
            # GTP-replicated, so different GTP ranks would collide on the same replica_id
            # without intervention. Inject gtp_rank into position 1 of the replica_id, the same
            # way the GTP-sharded branch below does.
            if layer_name in tensor_parallel_layers_axis_map:
                replica_id = (0, gtp_rank, dp_no_gtp_rank)
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
                replica_id = (0, tp_rank * gtp_size + gtp_rank, dp_no_gtp_rank)
                sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                    tensor,
                    layer_key,
                    replica_id=replica_id,
                    prepend_offsets=sharded_offsets,
                    tp_group=tp_group,
                    dp_cp_group=dp_cp_group,
                )
            continue

        # GTP-sharded tensor: delegate to the (GTP-aware) single-tensor helper, the one place
        # that knows how to layer the axis-0 GTP split onto TP, elect the writer over the
        # gtp-excluded DP group, and set allow_shape_mismatch for GTP alignment padding.
        # (tp_axis None → 0: GTP always shards axis 0; tp_size is 1 when this param has no TP.)
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


# Wire GTP into TE's hook registry. Done at module import time so any later
# ``te.Linear(gtp_group=...)`` call routes through the hooks below. The
# warning fires if TE is too old to expose ``register_gtp_hooks`` — in that
# case GTP silently no-ops, which is the failure mode we want to surface.
try:
    from transformer_engine.pytorch.module.base import (  # noqa: E402
        register_gtp_hooks as _te_register_gtp_hooks,
    )

    _te_register_gtp_hooks(
        slice_fn=gtp_slice_in_reset_parameters,
        finalize_fn=gtp_finalize_module_in_reset_parameters,
        wrap_fn=wrap_module_params_gtp,
    )
except ImportError:
    warnings.warn(
        "megatron.experimental.gtp: TransformerEngine does not expose register_gtp_hooks; "
        "GTP will be a no-op for te.Linear / te.LayerNormLinear / te.GroupedLinear. "
        "GTP requires TransformerEngine >= 2.17 (planned release). "
        "Upgrade TransformerEngine to a build that includes the GTP hook registry.",
        RuntimeWarning,
        stacklevel=2,
    )
