# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import gc
import inspect
import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import partial
from itertools import chain, zip_longest
from math import ceil
from typing import Any, Dict, List
from weakref import WeakKeyDictionary, WeakSet

import torch
from torch.utils._pytree import tree_flatten as tree_flatten_pyt
from torch.utils._pytree import tree_map as tree_map_pyt

from megatron.core.extensions.te_cuda_graphs import TECudaGraphMixin
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    CudaRNGStatesTracker,
    cudagraph_needs_generator_registration,
    get_all_rng_states,
    get_cuda_rng_tracker,
    is_checkpointing,
)
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.experimental_attention_variant.dsv4_layer_config import (
    CSALayerConfig,
)
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    ensure_params_ready,
    get_attr_wrapped_model,
    get_torch_version,
    is_te_min_version,
    log_on_each_pipeline_stage,
    log_single_rank,
    make_viewless_tensor,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from transformer_engine.pytorch.distributed import is_fp8_activation_recompute_enabled
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.graph import restore_fp8_tensors, save_fp8_tensors
    from transformer_engine.pytorch.graph import set_capture_end as te_set_capture_end
    from transformer_engine.pytorch.graph import set_capture_start as te_set_capture_start
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.utils import make_weak_ref

    HAVE_TE_GRAPHS = True
except:
    HAVE_TE_GRAPHS = False

try:
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP
except ImportError:
    # GTP requires TransformerEngine with the GTP hook registry; treat it as
    # unavailable when that import path cannot be resolved.
    HAVE_GTP = False

if HAVE_GTP:
    from megatron.core.tensor_parallel.gtp_api import (
        GTP_CONFIG,
        get_rs_stream,
        initialize_graph_wgrad_rings,
        is_graphed_chain,
        preserve_gtp_prefetch_state,
        set_cuda_graph_mempool,
        track_gtp_capture_comms,
        wait_async_comms,
    )
else:
    # Placeholders so static analysis does not flag these GTP-only symbols as
    # possibly-used-before-assignment; every use site is guarded by HAVE_GTP /
    # gtp_remat at runtime.
    is_graphed_chain = None
    GTP_CONFIG = None
    get_rs_stream = None
    initialize_graph_wgrad_rings = None
    preserve_gtp_prefetch_state = None
    set_cuda_graph_mempool = None
    track_gtp_capture_comms = None
    wait_async_comms = None

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

_IS_GRAPH_CAPTURING = False
_IS_GRAPH_WARMUP = False
_CUDA_GRAPH_STREAM_POOL_SIZE = 3
_CUDA_GRAPH_STREAM_POOLS = None
_CUDA_GRAPH_STREAM_NEXT_SLOT = 0
logger = logging.getLogger(__name__)


def _get_cuda_graph_stream() -> torch.cuda.Stream:
    """Assign local CUDA-graph runners across a bounded set of distinct streams.

    PyTorch obtains streams from a bounded pool, so repeatedly constructing
    ``torch.cuda.Stream`` objects can eventually return the same underlying CUDA stream.
    Assigning capture and replay streams explicitly makes that sharing intentional.

    ``CUDA_DEVICE_MAX_CONNECTIONS`` controls how CUDA maps logical streams to hardware work
    queues. It does not make PyTorch stream handles unique and cannot prevent stream aliasing.
    """
    global _CUDA_GRAPH_STREAM_POOLS, _CUDA_GRAPH_STREAM_NEXT_SLOT

    pool = _CUDA_GRAPH_STREAM_POOLS
    if pool is None:
        # CUDA recommends at least as many connections as independently active streams. GTP can
        # concurrently use graph, GTP/EGTP AG and RS, EP, DDP, and activation-offload streams, so
        # CUDA's default of eight connections can introduce false dependencies between them.
        max_connections = os.getenv("CUDA_DEVICE_MAX_CONNECTIONS")
        warning_reason = None
        if max_connections is None:
            warning_reason = "is unset (CUDA defaults to 8)"
        else:
            try:
                max_connections_value = int(max_connections)
            except ValueError:
                warning_reason = f"must be an integer from 1 to 32, but is {max_connections!r}"
            else:
                if max_connections_value < 16:
                    warning_reason = f"is {max_connections_value}"

        if warning_reason is not None:
            logger.warning(
                "CUDA_DEVICE_MAX_CONNECTIONS %s. For GTP workloads, consider setting it to "
                "16 or 32 before process startup to avoid implicit stream aliasing.",
                warning_reason,
            )

        pool = tuple(torch.cuda.Stream() for _ in range(_CUDA_GRAPH_STREAM_POOL_SIZE))
        if len({stream.cuda_stream for stream in pool}) != len(pool):
            raise RuntimeError("CUDA graph stream pool contains aliased streams")
        _CUDA_GRAPH_STREAM_POOLS = pool

    slot = _CUDA_GRAPH_STREAM_NEXT_SLOT
    _CUDA_GRAPH_STREAM_NEXT_SLOT = (slot + 1) % len(pool)
    return pool[slot]


def _get_tensor_alias_chain(tensor):
    """Return a tensor followed by each underlying base tensor."""
    aliases = []
    while torch.is_tensor(tensor):
        aliases.append(tensor)
        base = getattr(tensor, "_base", None)
        if base is None or base is tensor:
            break
        tensor = base
    return aliases


def _apply_cudagraph_buffer_metadata(tensor, *, is_output=False):
    """Attach one shared CUDA graph metadata object to a tensor and its base chain."""
    aliases = _get_tensor_alias_chain(tensor)
    metadata = next(
        (alias.cg_buffer_metadata for alias in aliases if hasattr(alias, "cg_buffer_metadata")),
        None,
    )
    if is_output:
        metadata = CudagraphBufferMetadata(
            is_cudagraph_output=True,
            is_saved_for_backward=bool(metadata and metadata.is_saved_for_backward),
        )
    elif metadata is None:
        metadata = CudagraphBufferMetadata()
    for alias in aliases:
        alias.cg_buffer_metadata = metadata
    return metadata


def _tag_cudagraph_buffer_saved_for_backward(tensor):
    """Tag a CUDA graph input or output observed in a Python 'save_for_backward' call."""
    if not torch.is_tensor(tensor):
        return

    # Views of the same graph buffer share one metadata object. If this tensor has not reached a
    # graph boundary yet, initialize its metadata now so record-time input/output classification
    # can preserve the saved-for-backward lifetime.
    metadata = _apply_cudagraph_buffer_metadata(tensor)
    metadata.is_saved_for_backward = True


_GTP_RUNNER_STREAMS: List[torch.cuda.Stream] = []


def get_gtp_runner_streams() -> List[torch.cuda.Stream]:
    """Replay streams of all GTP CG runners; finalize_model_grads waits on these
    (tail = captured Phase 2 main_grad.add_) before reading main_grad.
    """
    return _GTP_RUNNER_STREAMS


def _set_skip_fp8_weight_update_tensor(skip: bool) -> None:
    """Toggle TE's FP8 "skip weight refresh" flag between microbatches.

    TE 2.15 (PR #2759) moved this flag from a classmethod on FP8GlobalStateManager
    to a dataclass field on FP8GlobalStateManager.quantization_state, with no
    shim. This helper handles both layouts.
    """
    if not HAVE_TE_GRAPHS:
        return

    # TE <= 2.14: classmethod setter still exists.
    if hasattr(FP8GlobalStateManager, "set_skip_fp8_weight_update_tensor"):
        FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(skip)
        return

    # TE >= 2.15: write the dataclass field directly. Allocate the persistent
    # 1-element CUDA scalar once so its data pointer stays valid across
    # cudagraph replays, then fill the value in place.
    qstate = FP8GlobalStateManager.quantization_state
    if qstate.skip_fp8_weight_update_tensor is None:
        qstate.skip_fp8_weight_update_tensor = torch.empty(1, dtype=torch.float32, device="cuda")
    qstate.skip_fp8_weight_update_tensor.fill_(skip)


# Freeze GC during capture, which makes capture faster before PyTorch 2.9.
# TODO (@lmcafee): remove the freeze once most users are on PyTorch 2.9+.
FREEZE_GC = os.getenv("CUDA_GRAPH_CAPTURE_FREEZE_GC") != "0"
try:
    from packaging.version import Version as PkgVersion

    FREEZE_GC_MAX_TORCH_VERSION = PkgVersion("2.9.0a0")
    if get_torch_version() >= FREEZE_GC_MAX_TORCH_VERSION:
        FREEZE_GC = False
except ImportError:
    pass


# Whether a capture guard (no_gc_during_capture) has frozen the heap.
_CAPTURE_FROZE_HEAP = False


@contextmanager
def no_gc_during_capture(freeze: bool = False):
    """Keep Python's automatic garbage collection out of a CUDA graph capture.

    A collection that runs while a stream is capturing can free tensors held by unreachable
    objects, such as a released inference engine's pinned host buffers. Releasing them mid-capture
    can leave PyTorch's pinned host allocator failing its next request with "CUDA error: invalid
    argument". Automatic collection is therefore paused for the capture and resumes afterwards,
    when it collects what the capture left unreachable.

    With `freeze` and FREEZE_GC, the heap is also frozen for the capture, as the layer and TE
    captures always did: that makes them faster before PyTorch 2.9, but it also keeps the
    collection PyTorch 2.8 runs before a capture from freeing unreachable tensors left by warmup.
    gc.unfreeze() thaws every frozen object, so a guard inside one that froze the heap leaves the
    freeze to it. A freeze from other code cannot be told apart: since Python 3.12, immortal
    objects share the frozen generation, so gc.get_freeze_count() is never zero. Set
    CUDA_GRAPH_CAPTURE_FREEZE_GC=0 to keep such a freeze through captures before PyTorch 2.9.
    """
    global _CAPTURE_FROZE_HEAP
    gc_was_enabled = gc.isenabled()
    freeze = freeze and FREEZE_GC and not _CAPTURE_FROZE_HEAP
    gc.disable()
    if freeze:
        gc.freeze()
        _CAPTURE_FROZE_HEAP = True
    try:
        yield
    finally:
        if freeze:
            _CAPTURE_FROZE_HEAP = False
            gc.unfreeze()
        if gc_was_enabled:
            gc.enable()


def is_graph_capturing():
    """Query if currently capturing."""
    return _IS_GRAPH_CAPTURING


def _set_capture_start():
    """Set graph capture has started."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def _set_capture_end():
    """Set graph capture has ended."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


def is_graph_warmup():
    """Query if currently warming up for graph capture."""
    return _IS_GRAPH_WARMUP


def _set_warmup_start():
    """Set graph warmup has started."""
    global _IS_GRAPH_WARMUP
    _IS_GRAPH_WARMUP = True


def _set_warmup_end():
    """Set graph warmup has ended."""
    global _IS_GRAPH_WARMUP
    _IS_GRAPH_WARMUP = False


@dataclass
class CudagraphBufferMetadata:
    """
    Metadata saved to tensors during cudagraph capture. This data will be used to determine
    during graph captue when a cudagraph can reuse a buffer or directly write its output into
    a subsequent's graph's input.

    Set during recording:
      is_cudagraph_input / is_cudagraph_output — which graph boundary this buffer sits on.
      is_saved_for_backward — set by the save_for_backward observer; means the forward
          buffer must outlive the forward graph and stay allocator-owned until backward
          capture.

    Reuse accounting (used during graph creation):
      input_use_count — times this buffer appears as a graph input.
      cudagraph_reuse_ref_count / capture_reuse_count — remaining reuses; drives
          can_skip_replay_copy and when args_to_clear_buffers fires.
      fwd_cudagraph_buffer / bwd_cudagraph_buffer — the shared strong-ref buffer other
          graphs alias for this input/grad.

    """

    is_cudagraph_input: bool = False
    is_cudagraph_output: bool = False
    is_saved_for_backward: bool = False
    input_use_count: int = 0
    cudagraph_reuse_ref_count: int = 0
    capture_reuse_count: int = 0
    fwd_cudagraph_buffer: torch.Tensor = None
    bwd_cudagraph_buffer: torch.Tensor = None


class ArgMetadata:
    """Arg meta."""

    def __init__(self, arg, keep_values: bool = False):
        self.type = type(arg)
        self.values = None
        if isinstance(arg, torch.Tensor):
            self.shape = arg.shape
            self.dtype = arg.dtype
            self.device = arg.device
            self.value = arg.data_ptr()
            self.requires_grad = arg.requires_grad
            if keep_values:
                self.values = arg.detach().clone()
            if hasattr(arg, "cg_buffer_metadata"):
                # Its important this is a reference copy
                self.cg_buffer_metadata = arg.cg_buffer_metadata
        else:
            self.value = arg

    def zeros_like(self):
        """Reconstruct a tensor with the properties as the meta arg."""

        assert self.type == torch.Tensor
        return torch.zeros(
            *self.shape, dtype=self.dtype, device=self.device, requires_grad=self.requires_grad
        )


def alloc_tensor_from_graph_mempool(meta: ArgMetadata):
    """Allocates a tensor specified by a ArgMetadata into the graph mempool."""

    torch._C._cuda_beginAllocateCurrentThreadToPool(
        torch.cuda.current_device(), CudaGraphManager.global_mempool
    )
    out = meta.zeros_like()
    if meta.values is not None:
        # Warmup runs the layer on this buffer before capture.
        # Sequence metadata (cu_seqlens) must hold real values until TransfomerEngine#3205.
        out.copy_(meta.values)
    out.is_from_global_mempool = True
    out.requires_grad_(meta.requires_grad)

    torch._C._cuda_endAllocateToPool(torch.cuda.current_device(), CudaGraphManager.global_mempool)
    return out


def tree_map(func, tree):
    """
    Wrapper around pytorch's tree_map, but also recurses into dataclasses.
    """

    def wrapper(arg):
        # If it's a dataclass, map over its fields
        if is_dataclass(arg) and not isinstance(arg, type):
            changes = {
                f.name: tree_map_pyt(func, getattr(arg, f.name)) for f in dataclasses.fields(arg)
            }
            mapped_arg = dataclasses.replace(arg, **changes)

            # 'dataclasses.replace' reruns '__post_init__', which may overwrite a tensor
            # field that was explicitly mapped above. In particular, PackedSeqParams rebuilds
            # 'seq_idx' from 'cu_seqlens'. CUDA graph input buffers are zero-initialized, so
            # that rebuild assigns every token the padded sequence count and can make Mamba
            # kernels access out of bounds during graph capture. Preserve the tensor selected by
            # the mapping operation; replay will populate that buffer with the real input value.
            for name, value in changes.items():
                if torch.is_tensor(value) and getattr(mapped_arg, name) is not value:
                    object.__setattr__(mapped_arg, name, value)

            return mapped_arg

        # Otherwise, apply the user function
        return func(arg)

    return tree_map_pyt(wrapper, tree)


def _check_supported_type(meta):
    """Check if arg meta is a supported type for cudagraph input/outputs."""

    assert isinstance(meta, ArgMetadata)

    # Import inference contexts here to guard against circular import.
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.contexts.static_context import StaticInferenceContext

    _SUPPORTED_TYPES = {
        torch.Tensor,
        torch.distributed.ProcessGroup,
        type(None),
        bool,
        int,
        str,
        float,
        dataclass,
        StaticInferenceContext,
        DynamicInferenceContext,
        ArgMetadata,
    }
    assert meta.type in _SUPPORTED_TYPES or is_dataclass(
        meta.value
    ), f"Cudagraphs received an arg of type {meta.type} which is not supported."


def annotate_first_last_layer(layers):
    """Annotate the first and last modules in an ordered layer collection."""
    for i, layer in enumerate(layers):
        layer.is_first_layer = i == 0
        layer.is_last_layer = i == len(layers) - 1


def _clone_nested_tensors(value: Any) -> Any:
    """Recursively clone tensors inside nested containers."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_nested_tensors(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_nested_tensors(v) for k, v in value.items()}
    if isinstance(value, set):
        raise TypeError(
            "Sets of tensors are unsupported in cudagraph helpers; use list/tuple instead"
        )
    return value


def _ensure_generator_state_is_cudagraph_safe(gen: torch.Generator) -> torch.Generator:
    """Make generator state safe for CUDA graph capture/replay.

    Generator state tensors can become inference tensors if created under `torch.inference_mode()`.
    CUDA graph capture may later attempt in-place updates on that state; this fails for inference
    tensors. Fix the generator *in-place* (preserving identity) by cloning its state outside
    inference mode and setting it back.
    """
    with torch.inference_mode(mode=False):
        if hasattr(gen, "graphsafe_get_state"):
            state = gen.graphsafe_get_state()
        else:
            state = gen.get_state()

        cloned_state = _clone_nested_tensors(state)
        if hasattr(gen, "graphsafe_set_state"):
            gen.graphsafe_set_state(cloned_state)
        else:
            gen.set_state(cloned_state)

    return gen


def make_weakref(ten, inplace=True):
    """Creates a weak reference to a tensor by creating a tensor that replaces storage with
    raw-pointer wrappers that do not hold a storage reference"""

    # Only graph mempool tensors in the graph mempool (e.g. a previous layer's
    # output reused as this graph's input) are safe to weak-ref since their memory is
    # driver-pinned with stable addresses. Everything else, including, stray tensors
    # from dataclass __post_init__ side-effects (e.g. seq_idx created by
    # PackedSeqParams.__post_init__ during dataclasses.replace inside the tree_map) must
    # retain strong refs, or it will cause a use-after-free on replay that manifests as a
    # segfault under memory pressure.
    if not (
        HAVE_TE_GRAPHS and torch.is_tensor(ten) and getattr(ten, "is_from_global_mempool", False)
    ):
        return ten

    try:
        wr = make_weak_ref(ten)
        if inplace:
            ten.data = wr
            wr = ten

    except RuntimeError:
        # Fallback to keeping a strong reference. There is a known bug where some
        # dtypes (e.g. torch.float64) are not mapped to a representation in
        # transformer_engine/pytorch/utils.py.
        if torch.distributed.get_rank() == 0:
            logger.warning(
                f"Could not create weak ref for tensor with dtype {arg.dtype}; "
                f"keeping strong ref with a potential memory overhead."
            )

    return wr


def create_strong_ref(ten: torch.Tensor):
    """Create a strong reference to a tensor that keeps memory allocated"""

    ref = ten.detach()
    if hasattr(ten, "is_from_global_mempool"):
        ref.is_from_global_mempool = ten.is_from_global_mempool
    if hasattr(ten, "cg_buffer_metadata"):
        ref.cg_buffer_metadata = deepcopy(ten.cg_buffer_metadata)
    if hasattr(ten, "can_skip_replay_copy"):
        ref.can_skip_replay_copy = ten.can_skip_replay_copy
    ref.requires_grad_(ten.requires_grad)
    return ref


fwd_buffer_reuse_ref_count = 0
bwd_buffer_reuse_ref_count = 0


def _backup_grads_before_capture(runner, parameters=None):
    """Snapshot main_grad so CUDA-graph capture cannot corrupt finalized gradients."""
    backup = {}
    parameters = tuple(runner.base_module.parameters() if parameters is None else parameters)
    for p in parameters:
        mg = getattr(p, "main_grad", None)
        if mg is not None:
            backup[id(p)] = (p, mg.clone())

    if runner.gtp_remat:
        # GTP only: also protect the cross-graph next_w the cascade accumulates into.
        for p in parameters:
            nw = getattr(p, "next_w", None) if getattr(p, "is_gtp_weight_remat", False) else None
            if nw is None:
                continue
            shards = nw.weight_list if getattr(nw, "is_routed_expert", False) else [nw]
            for w in shards or []:
                mg = getattr(w, "main_grad", None)
                if mg is not None and id(w) not in backup:
                    backup[id(w)] = (w, mg.clone())
    return backup


def _restore_grads_after_capture(backup):
    """Restore the main_grad snapshots taken by '_backup_grads_before_capture'."""
    for p, saved in backup.values():
        p.main_grad.copy_(saved)


@contextmanager
def _preserve_parameter_grads_during_backward_capture(runner):
    """Give each backward runner fresh accumulation state without changing the training step."""

    parameters = tuple(runner.params_to_backprop)
    main_grad_backup = _backup_grads_before_capture(runner, parameters=parameters)
    missing = object()
    parameter_state = [
        (param, param.grad, getattr(param, "grad_added_to_main_grad", missing))
        for param in parameters
    ]

    # A module can back multiple runners, as with a repeated MTP layer. Each runner must capture
    # its own contribution instead of inheriting another runner's accumulation flag or stale grad.
    for param, _, _ in parameter_state:
        param.grad = None
        if hasattr(param, "grad_added_to_main_grad"):
            param.grad_added_to_main_grad = False

    try:
        yield
    finally:
        _restore_grads_after_capture(main_grad_backup)
        for param, grad, grad_added_to_main_grad in parameter_state:
            param.grad = grad
            if grad_added_to_main_grad is missing:
                if hasattr(param, "grad_added_to_main_grad"):
                    delattr(param, "grad_added_to_main_grad")
            else:
                param.grad_added_to_main_grad = grad_added_to_main_grad


class _CudagraphGlobalRecord:
    """A global datastructure that records of the ordering of all _CudaGraphRunner's
    first fwd or bwd passes. 'create_cudagraphs' will use this to create
    cudagraphs in execution order, which is required for cudagraphs sharing a mempool."""

    """A global flag that if true, all cudagraph runners
    fwd and bwd passes will be performed using their cudagraphed versions."""
    cudagraph_created = False

    """A record of fwd and bwd graph creation, populated with 'record_fwd_graph' and
    'record_bwd_graph."""
    cudagraph_record: list[tuple] = []
    cudagraph_inference_record: list[tuple] = []
    _saved_tensors_observer = None
    _te_capture_started = False

    # Owners are local/full-iteration graphs or TE graph callables.
    # Weak keys avoid extending their lifetimes merely to retain their constants.
    capture_constants: WeakKeyDictionary[torch.cuda.CUDAGraph, dict[int, torch.Tensor]] = (
        WeakKeyDictionary()
    )
    active_capture_constants: dict[int, torch.Tensor] | None = None
    # Without a graph owner these references cannot be released safely before process exit.
    unowned_capture_constants: dict[int, torch.Tensor] = {}
    captured_runners: WeakSet["_CudaGraphRunner"] = WeakSet()

    @classmethod
    def _enable_saved_tensors_observer(cls):
        """Observe Python 'save_for_backward' calls while recording and capturing graphs."""
        if cls.cudagraph_created or cls._saved_tensors_observer is not None:
            return

        function_ctx = torch.autograd.function.FunctionCtx
        original_save_for_backward = function_ctx.save_for_backward

        def observing_save_for_backward(ctx, *tensors):
            for tensor in tensors:
                _tag_cudagraph_buffer_saved_for_backward(tensor)
            return original_save_for_backward(ctx, *tensors)

        cls._saved_tensors_observer = (original_save_for_backward, observing_save_for_backward)
        function_ctx.save_for_backward = observing_save_for_backward

    @classmethod
    def _disable_saved_tensors_observer(cls):
        """Restore Python's original 'save_for_backward' implementation."""
        if cls._saved_tensors_observer is None:
            return

        original_save_for_backward, observing_save_for_backward = cls._saved_tensors_observer
        function_ctx = torch.autograd.function.FunctionCtx
        if function_ctx.save_for_backward is observing_save_for_backward:
            function_ctx.save_for_backward = original_save_for_backward
        cls._saved_tensors_observer = None

    @classmethod
    def record_fwd_graph(cls, runner, args, kwargs, out):
        """Record a fwd graph to 'cudagraph_record"""
        cls.cudagraph_record.append((runner, "fwd", args, kwargs, out))

    @classmethod
    def record_bwd_graph(cls, runner):
        """Record a bwd graph to 'cudagraph_record"""
        cls.cudagraph_record.append((runner, "bwd"))

    @classmethod
    def create_cudagraphs(cls):
        """Create recorded CUDA graphs, then remove the saved-tensor observer."""
        # Forward-only schedules can reach this before training records any graphs.
        # Keep observing saved tensors until there is a capture to complete.
        if not cls.cudagraph_created and not cls.cudagraph_record:
            return

        try:
            return cls._create_cudagraphs()
        finally:
            # A failed capture must not leave DDP hooks disabled for subsequent backwards.
            if is_graph_capturing():
                _set_warmup_end()
                _set_capture_end()
                if cls._te_capture_started:
                    te_set_capture_end()
                    cls._te_capture_started = False
            cls._disable_saved_tensors_observer()

    @classmethod
    def _create_cudagraphs(cls):
        """Iterate through 'cudagraph_record' creating graphs in the order in which
        they were recorded."""
        # Cudagraphs have already been created, check that no cudagraphed modules ran in eager mode
        if cls.cudagraph_created:
            assert len(cls.cudagraph_record) == 0, (
                "One or more _CudaGraphRunners requested to create a graph after cudagraphs",
                "were already created!",
            )
            return

        # No cudagraphs have been created or recorded, so do nothing
        if len(cls.cudagraph_record) == 0:
            return

        # Otherwise, create all the recorded cudagraphs.
        has_te_modules = False
        if HAVE_TE_GRAPHS:
            for g in cls.cudagraph_record:
                base_module = g[0].base_module
                has_te_modules = has_te_modules or any(
                    [isinstance(m, TransformerEngineBaseModule) for m in base_module.modules()]
                )

        progress_bar = enumerate(cls.cudagraph_record)
        time_start = time.time()
        mem_stats_start = torch.cuda.memory_stats()

        if torch.distributed.get_rank() == 0:
            if HAVE_TQDM:
                progress_bar = tqdm(
                    progress_bar, "create cuda graphs", total=len(cls.cudagraph_record)
                )

            logger.info(f"Creating {len(cls.cudagraph_record)} CUDA graphs")
            if not HAVE_TE_GRAPHS:
                logger.warning(
                    "Transformer Engine was not detected while capturing training cudagraphs."
                    "As a result cudagraph memory overhead may significantly increase as "
                    "Transformer Engine's weak reference feature is used on cudagraph input and "
                    "output buffers. This allows the memory of input and output buffers to be "
                    " reclaimed across graphs while remaining valid buffers for when the graph "
                    "is replayed. For more information see: "
                    "https://github.com/NVIDIA/TransformerEngine/blob/v2.10/transformer_engine/pytorch/utils.py#L759"  # pylint: disable=line-too-long
                )

        gtp_active = any(r[0].gtp_remat for r in cls.cudagraph_record)
        if gtp_active:
            # GTP buffer reuse during capture trips the param-state debug asserts; disable them.
            GTP_CONFIG.check_param_states = False
            initialize_graph_wgrad_rings()

        # Recording/warmup can keep AccumulateGrad nodes alive on the default stream.
        # Match full-iteration capture: let autograd redirect those stale references to
        # the capturing stream rather than introduce an illegal default-stream dependency.
        # Intentionally leave this process-global setting enabled, as full-iteration capture
        # does. The API has no getter for restoring its previous value; resetting it to False
        # could disable an override enabled by another caller. It only affects capture-time
        # stream handling, not ordinary eager backward.
        if hasattr(torch.autograd.graph, 'set_override_stale_capture_stream'):
            torch.autograd.graph.set_override_stale_capture_stream(True)

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()
            cls._te_capture_started = True

        global bwd_buffer_reuse_ref_count, fwd_buffer_reuse_ref_count

        def format_mem_bytes(mem_bytes):
            sign, n = ("-", -mem_bytes) if mem_bytes < 0 else ("", mem_bytes)
            for p, s in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb")]:
                if n >= 1024**p:
                    return f"{sign}{n / 1024**p:.1f} {s}"
            return f"{sign}{n} bytes"

        for g_idx, g in progress_bar:
            if torch.distributed.get_rank() == 0:
                mem_stats = torch.cuda.memory_stats()
                progress_str = "create cuda graphs | mem: alloc %s, res %s" % (
                    format_mem_bytes(mem_stats["allocated_bytes.all.current"]),
                    format_mem_bytes(mem_stats["reserved_bytes.all.current"]),
                )
                if HAVE_TQDM:
                    progress_bar.set_description(progress_str)
                elif g_idx % 100 == 0 or g_idx == len(cls.cudagraph_record) - 1:
                    logger.info(f"{g_idx}/{len(cls.cudagraph_record)}. {progress_str}")

            runner, graph_type = g[0:2]
            if graph_type == 'fwd':
                args, kwargs, out = g[2:]
                runner.create_fwd_graph(args, kwargs, out, clone_inputs=True)
            else:
                assert fwd_buffer_reuse_ref_count == 0
                runner.create_bwd_graph()

        # Memory usage.
        time_end = time.time()
        mem_stats_end = torch.cuda.memory_stats()
        capture_stats = {
            "time": time_end - time_start,
            "allocated_bytes": (
                mem_stats_end["allocated_bytes.all.current"]
                - mem_stats_start["allocated_bytes.all.current"]
            ),
            "reserved_bytes": (
                mem_stats_end["reserved_bytes.all.current"]
                - mem_stats_start["reserved_bytes.all.current"]
            ),
        }

        log_single_rank(
            logger,
            logging.INFO,
            "> built %d cuda graph(s) in %.2f sec, with total memory usage: "
            "allocated %s, reserved %s."
            % (
                len(cls.cudagraph_record),
                capture_stats["time"],
                format_mem_bytes(capture_stats["allocated_bytes"]),
                format_mem_bytes(capture_stats["reserved_bytes"]),
            ),
        )

        # Mark cuda graphs as created.
        for g in cls.cudagraph_record:
            runner = g[0]
            runner.cudagraph_created = True

        # Reset global record.
        cls.cudagraph_created = True
        cls.cudagraph_record = []

        # Finished capturing.
        _set_capture_end()
        if has_te_modules:
            te_set_capture_end()
            cls._te_capture_started = False

        # Train on the default stream from here on (see CudaGraphManager.__init__).
        default_stream = torch.cuda.default_stream()
        default_stream.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(default_stream)

        # Return capture time and memory usage.
        return capture_stats


def create_cudagraphs():
    """Should be called at the end of each schedule function,
    (e.g. forward_backward_pipelining_with_interleaving) in
    `megatron.core.pipeline_parallel.schedules.py`. During the first step, _CudaGraphRunners
    populate _CudagraphGlobalRecord with the global order in which cudagraphs should be created.
    At the end for the first step, this function calls each runner's `create_fwd_graph` and
    `create_bwd_graph` in the order recorded in _CudagraphGlobalRecord, which allows cudagraphs
    to be created in execution order, which allows multiple cudagraphs to share a single
    memory pool, minimizing cudagraph memory usage."""

    return _CudagraphGlobalRecord.create_cudagraphs()


@contextmanager
def capture_constants(owner_graph: torch.cuda.CUDAGraph) -> Iterator[None]:
    """Retain constants for a graph owner until teardown or owner destruction.

    Wrap the capture context, not replay. For APIs that capture several graphs at
    once (TE), the owner must outlive every graph and release its constants only
    after destroying all of them. Failed captures discard their registrations.
    """
    previous = _CudagraphGlobalRecord.active_capture_constants
    constants = {}
    _CudagraphGlobalRecord.active_capture_constants = constants
    try:
        yield
        _CudagraphGlobalRecord.capture_constants[owner_graph] = constants
    finally:
        _CudagraphGlobalRecord.active_capture_constants = previous


def register_capture_constant(tensor: torch.Tensor) -> None:
    """Retain a capture constant for its graph, or process lifetime when no owner is known."""
    if not torch.cuda.is_current_stream_capturing():
        return

    constants = _CudagraphGlobalRecord.active_capture_constants
    if constants is None:
        constants = _CudagraphGlobalRecord.unowned_capture_constants
        if not constants:
            logger.warning(
                "CUDA graph captured outside capture_constants(); cached positional "
                "embeddings it reads are retained for the process lifetime."
            )
    constants[id(tensor)] = tensor


def release_capture_constants(owner_graph: torch.cuda.CUDAGraph) -> None:
    """Release an owner's constants after all of its graphs have been destroyed."""
    _CudagraphGlobalRecord.capture_constants.pop(owner_graph, None)


def delete_cuda_graphs():
    """Delete all CUDA graphs."""

    _CudagraphGlobalRecord._disable_saved_tensors_observer()

    # Reset runners.
    runners = set(_CudagraphGlobalRecord.captured_runners)
    for record in [
        *_CudagraphGlobalRecord.cudagraph_record,
        *_CudagraphGlobalRecord.cudagraph_inference_record,
    ]:
        runners.add(record[0])
    if runners:
        # Replay can still be using external constants when graph destruction is requested.
        torch.cuda.synchronize()
    for runner in runners:
        assert isinstance(runner, _CudaGraphRunner)

        # Some callers (e.g., test_cuda_graphs.py) delete graph attributes
        # before the global teardown runs.
        for graph in (getattr(runner, "fwd_graph", None), getattr(runner, "bwd_graph", None)):
            if graph is not None:
                graph.reset()
                release_capture_constants(graph)
        runner.cudagraph_created = False
        runner.fwd_graph_recorded = False
        runner.bwd_graph_recorded = False
        runner.fwd_graph = None
        runner.bwd_graph = None
        runner.mempool = None
        runner._gtp_fwd_params_to_ensure_ready = ()

    # Reset global tracking state
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    _CudagraphGlobalRecord.cudagraph_inference_record = []
    _CudagraphGlobalRecord.captured_runners.clear()
    _GTP_RUNNER_STREAMS.clear()

    if runners:
        # Free the memory the reset graphs and their capture inputs held. With no graphs
        # there is nothing to free, and a full collection costs ~0.4 s on a large heap.
        gc.collect()
    torch.cuda.empty_cache()

    CudaGraphManager.global_mempool = None


class _GraphStatus(Enum):
    """An Enum to track if a cudagraph is ready to perform a forward or backward pass."""

    FWD_READY = 0  # Set immediately after a bwd pass
    BWD_READY = 1  # Set immediately after a fwd pass


class _CudagraphRecordNode(torch.autograd.Function):
    """Inserts a noop node into the autograd graph, used to record when a bwd graph needs
    to be created."""

    @staticmethod
    def forward(ctx, runner, inputs):
        """Forward pass, does nothing but registers an autograd node."""

        assert (
            runner.status == _GraphStatus.FWD_READY
        ), "Tried calling the fwd cudagraph when the bwd cudagraph was expected to be called next!"

        ctx.runner = runner
        return inputs

    @staticmethod
    def backward(ctx, grads):
        """If this is the first bwd pass of this runner, record that a
        bwd graph needs to be created."""

        runner = ctx.runner
        assert (
            runner.status == _GraphStatus.BWD_READY
        ), "Tried calling the bwd cudagraph when the fwd cudagraph was expected to be called next!"
        runner.status = _GraphStatus.FWD_READY
        if not runner.bwd_graph_recorded:
            _CudagraphGlobalRecord.record_bwd_graph(runner)
            runner.bwd_graph_recorded = True

        return None, grads


class _CudagraphReplayNode(torch.autograd.Function):
    """Replays the runner's cudagraphs with autograd. Handles copying data into/out of the
    cudagraph io and fp8/fp4 if used."""

    @staticmethod
    def forward(ctx, runner, is_first_microbatch, *inputs):
        """Replay the forward graph of the passed runner."""

        assert (
            runner.fwd_graph is not None
        ), "Tried replaying fwd cudagraph before calling 'create_fwd_cudagraph!"
        assert (
            runner.status == _GraphStatus.FWD_READY
        ), "Tried calling the fwd cudagraph when the bwd cudagraph was expected to be called next!"
        assert len(inputs) == len(
            runner.fwd_graph_input_surface
        ), "Fwd cudagraph received a different number of tensors than what it was graphed with!"

        # Copy new data into fwd graph input buffer
        for user_input, cudagraph_input in zip(inputs, runner.fwd_graph_input_surface):
            can_skip_replay_copy = getattr(
                cudagraph_input, "can_skip_replay_copy", False
            ) and getattr(user_input, "can_skip_replay_copy", True)

            # When the same input (like cu_seqlens) is passed to multiple cudagraphs, the first
            # cudagraph copies it into the corresponding 'cudagraph_input'. Subsequent cudagraphs
            # will then read the same cudagraph_input, leading to a case where the passed tensor
            # doesn't need a copy despite being a different data_ptr as its 'cudagraph_input'.
            if can_skip_replay_copy and cudagraph_input.cg_buffer_metadata.input_use_count == 1:
                assert user_input.data_ptr() == cudagraph_input.data_ptr()
            elif user_input.data_ptr() != cudagraph_input.data_ptr():
                cudagraph_input.copy_(user_input)

        ctx.runner = runner
        if runner.fp8_enabled or runner.fp4_enabled:
            if isinstance(FP8GlobalStateManager.get_fp8_recipe(), te.common.recipe.DelayedScaling):
                for m in runner.base_module.modules():
                    if isinstance(m, TransformerEngineBaseModule):
                        m.fp8_meta["fp8_group"] = FP8GlobalStateManager.get_fp8_group()
                        m.fp8_meta["recipe"] = FP8GlobalStateManager.get_fp8_recipe()

                        if is_te_min_version("1.13.0"):
                            FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(m.fp8_meta)
                        else:
                            FP8GlobalStateManager.add_fp8_tensors_to_global_buffer(
                                m.fp8_meta, fp8_weights=m._get_fp8_params()
                            )

            # Note that FP8GlobalStateManager.is_first_fp8_module() is inacccurate as each
            # layer may be in its own fp8 context, when the fp8 recipe != delayed_scaling
            if runner.is_first_layer and (runner.fp8_param_cache_updated != is_first_microbatch):
                _set_skip_fp8_weight_update_tensor(not is_first_microbatch)
                runner.fp8_param_cache_updated = is_first_microbatch

        if runner.use_stream:
            runner.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(runner.stream):
                runner.fwd_graph.replay()
            torch.cuda.current_stream().wait_event(runner.fwd_completion_event)
        else:
            runner.fwd_graph.replay()
        return runner.fwd_graph_output_surface

    @staticmethod
    def backward(ctx, *grads):
        """Replay the backward graph of the passed runner."""

        runner = ctx.runner
        assert (
            runner.bwd_graph is not None
        ), "Tried replaying bwd cudagraph before calling 'create_bwd_cudagraph'!"
        assert (
            runner.status == _GraphStatus.BWD_READY
        ), "Tried calling the bwd cudagraph when the fwd cudagraph was expected to be called next!"
        assert len(grads) == len(
            runner.static_grad_outputs
        ), "Bwd cudagraph received a different number of tensors than what it was graphed with!"

        # Copy new data into bwd graph input buffer
        for user_output_grad, cudagraph_output_grad in zip(grads, runner.static_grad_outputs):
            if cudagraph_output_grad is None:
                continue
            if user_output_grad.data_ptr() != cudagraph_output_grad.data_ptr():
                cudagraph_output_grad.copy_(user_output_grad)

        if runner.use_stream:
            runner.stream.wait_stream(torch.cuda.current_stream())
            if runner.gtp_remat:
                for slot in runner._gtp_wgrad_ring_slots:
                    runner.stream.wait_event(slot.ready_event)
            with torch.cuda.stream(runner.stream):
                runner.bwd_graph.replay()
                if runner.gtp_remat:
                    for slot in runner._gtp_wgrad_ring_slots:
                        slot.ready_event.record(runner.stream)
            torch.cuda.current_stream().wait_event(runner.bwd_completion_event)
        else:
            runner.bwd_graph.replay()

        runner.bwd_graph_replay_complete_event.record(torch.cuda.current_stream())
        for param in runner.params_to_backprop:
            param._cudagraph_wgrad_ready_event = runner.bwd_graph_replay_complete_event
            if hasattr(param, 'grad_added_to_main_grad'):
                param.grad_added_to_main_grad = True
        runner.status = _GraphStatus.FWD_READY

        # Update FP8 scale factors if needed
        if runner.fp8_enabled and isinstance(
            FP8GlobalStateManager.get_fp8_recipe(), te.common.recipe.DelayedScaling
        ):
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # DDP grad-ready hook is silenced at capture/replay, so fire it here (on each param's
        # rs_stream, after wait_stream(runner.stream) fences Phase 2) to let DDP RS overlap bwd.
        if runner.gtp_remat:
            for gtp_rs_stream, params in runner._gtp_finalize_hook_plan:
                gtp_rs_stream.wait_stream(runner.stream)
                with torch.cuda.stream(gtp_rs_stream):
                    for param in params:
                        param.grad = None
                        hook = getattr(param, '_grad_accum_hook', None)
                        if hook is not None:
                            hook()

        return None, None, *runner.static_grad_inputs, *(None,) * len(runner.params_to_backprop)


class _CudaGraphRunner(torch.nn.Module):
    """Represents the execution of a cudagraphed module for a single microbatch.
    If there are multiple outstanding microbatches per module, such as for pipeline parallelism,
    CudaGraphManager automatically creates multiple _CudaGraphRunners per module."""

    def __init__(
        self,
        base_module: MegatronModule,
        mempool: int,
        fwd_graph_input_args: List[Any],
        fwd_graph_input_kwargs: Dict[str, Any],
        func,
        need_backward,
    ):
        """Creates a _CudaGraphRunner, which holds a single pair of fwd and bwd cudagraphs, which
        are not created until this runner records its graph creation into
        '_CudagraphGlobalRecord', and 'create_cudagraphs()' is called."""

        super().__init__()

        self.base_module = base_module
        self.mempool = mempool

        self.fwd_graph_input_arg_metas = [ArgMetadata(a) for a in fwd_graph_input_args]
        self.fwd_graph_input_kwarg_metas = {
            k: ArgMetadata(a) for k, a in fwd_graph_input_kwargs.items()
        }

        self.fwd_graph = None
        self.bwd_graph = None
        self.bwd_graph_replay_complete_event = torch.cuda.Event()

        self.fwd_graph_recorded = False
        self.bwd_graph_recorded = False
        self.cudagraph_created = False
        self.status = _GraphStatus.FWD_READY

        self.backward_retain_grad = False
        self.fp8_enabled = False
        self.fp4_enabled = False
        self.fp8_runtime_enabled = None
        self.fp4_runtime_enabled = None
        self.deallocate_pipeline_outputs = False
        self.num_warmup_steps = 0
        self.needs_recompute_param_discovery = False
        self.use_stream = False
        self.gtp_remat = False
        # Populated by create_bwd_graph: one entry per captured GTP wgrad-finalization occurrence.
        # Repeated parameters intentionally appear more than once so replay matches eager DDP
        # grad-ready accounting.
        self.finalized_during_bwd_capture = []
        # (rs_stream, params) DDP grad-ready hook plan; built in create_bwd_graph.
        self._gtp_finalize_hook_plan = []
        # Persistent wgrad slots written by this graph. Replay waits for each slot's previous RS
        # reader before launching the graph.
        self._gtp_wgrad_ring_slots = []
        # GTP weights read by this forward graph before their owning module pre-hooks execute.
        self._gtp_fwd_params_to_ensure_ready = ()

        self.grad_enabled = need_backward and torch.is_grad_enabled()
        self.func = super(MegatronModule, self.base_module).__call__ if func is None else func
        self.is_first_layer = getattr(base_module, "is_first_layer", True)
        self.is_last_layer = getattr(base_module, "is_last_layer", True)

        # We use this attribute to record the value of 'is_first_microbatch' each fwd cudagraph
        # replay so that way we only update the value of this flag in FP8GlobalStateManager when
        # it changes which incurs an HtoD sync
        if self.is_first_layer:
            self.fp8_param_cache_updated = None

        if hasattr(self.base_module, "config") and isinstance(
            self.base_module.config, TransformerConfig
        ):
            self.backward_retain_grad = self.base_module.config.cuda_graph_retain_backward_graph
            self.deallocate_pipeline_outputs = self.base_module.config.deallocate_pipeline_outputs
            self.num_warmup_steps = self.base_module.config.cuda_graph_warmup_steps
            self.needs_recompute_param_discovery = (
                self.grad_enabled and self.base_module.config.recompute_granularity is not None
            )
            self.fp8_enabled = self.base_module.config.fp8 is not None
            self.fp4_enabled = self.base_module.config.fp4 is not None
            self.fp8_runtime_enabled = None
            self.fp4_runtime_enabled = None
            self.gtp_remat = self.base_module.config.gtp_weight_remat_size > 1

            if self.gtp_remat:
                # Ensure internal warmup (inside create_fwd_graph) has >= 2 steps
                # for GTP: 1st builds chain + tickets, 2nd exercises prefetch path.
                self.num_warmup_steps = max(self.num_warmup_steps, 2)

                self.use_stream = True
                self.stream = _get_cuda_graph_stream()
                self.fwd_completion_event = torch.cuda.Event(external=True, interprocess=True)
                self.bwd_completion_event = torch.cuda.Event(external=True, interprocess=True)
                # Registered for finalize_model_grads to wait on (Phase 2 fence).
                _GTP_RUNNER_STREAMS.append(self.stream)

            if self.fp8_enabled:
                self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                _set_skip_fp8_weight_update_tensor(False)

            if self.fp4_enabled:
                from megatron.core.fp4_utils import get_fp4_recipe  # to avoid circular import

                self.fp4_recipe = get_fp4_recipe(self.base_module.config)
                _set_skip_fp8_weight_update_tensor(False)

    def _wait_side_streams(self, side_streams):
        """Wait for side streams that carried work owned by one capture tracker.

        Joining only streams discovered by ``track_gtp_capture_comms`` avoids waiting on an empty
        captured stream.
        """
        for s in side_streams:
            torch.cuda.current_stream().wait_stream(s)

    def _set_gtp_finalize_hook_plan(self, finalized_params):
        """Build the replay hook plan from captured GRAPHED finalization occurrences."""
        self.finalized_during_bwd_capture = list(finalized_params) if self.gtp_remat else []
        self._gtp_finalize_hook_plan = []
        if not self.finalized_during_bwd_capture:
            return

        params_by_stream = defaultdict(list)
        for param in self.finalized_during_bwd_capture:
            params_by_stream[get_rs_stream(param.chain_id, param.group)].append(param)
        self._gtp_finalize_hook_plan = list(params_by_stream.items())

    def __str__(self):
        return "%s; hid %s" % (
            self.base_module.__class__.__name__,
            tuple(self.fwd_graph_input_kwarg_metas["hidden_states"].shape),
        )

    def get_quantization_context(self):
        """Return appropriate quantization context (FP8 or FP4) in cudagraph mode."""
        if self.fp8_runtime_enabled:
            from megatron.core.fp8_utils import get_fp8_context  # to avoid circular import

            return get_fp8_context(self.base_module.config, self.base_module.layer_number - 1)
        elif self.fp4_runtime_enabled:
            from megatron.core.fp4_utils import get_fp4_context  # to avoid circular import

            return get_fp4_context(self.base_module.config, self.base_module.layer_number - 1)
        else:
            return nullcontext()

    def get_connected_params(self, outputs):
        """Iterate through the autograd graph of 'outputs' and returns all parameters connected.
        In theory this should return all parameters that return a nonzero wgrad when computing
        the backward pass of 'outputs'."""
        # Flatten outputs and start traversal from roots that require gradients
        args = (outputs,) if torch.is_tensor(outputs) else outputs
        stack = [
            t.grad_fn
            for t in self.get_tensors(args, check_types=False)
            if t.requires_grad and t.grad_fn
        ]
        visited, p_ids = set(), set()

        while stack:
            if (fn := stack.pop()) not in visited:
                visited.add(fn)
                # AccumulateGrad nodes (leafs) hold the 'variable' (Parameter) they accumulate into
                if hasattr(fn, 'variable'):
                    p_ids.add(id(fn.variable))
                stack.extend(f for f, _ in fn.next_functions if f)

        # Return module params that were found in the graph, preserving original order
        return tuple(p for p in self.base_module.parameters() if id(p) in p_ids)

    def _weakref_forward_buffers(self, preserve_forward_to_backward_lifetimes: bool) -> None:
        """Release ownership only when CUDA graph topology proves the buffer reclaimable.

        `make_weakref` preserves a captured address but releases allocator ownership.
        Although CUDA graph memory is pinned to a stable address, the graph-pool allocator may
        reuse an unowned allocation before backward capture and overwrite its contents.
        these conditions only within that interval avoids retaining every boundary tensor.
        """

        def is_saved_for_backward(tensor) -> bool:
            """Return whether a tensor is needed for the backward pass graph.

            Preserving allocator ownership guards against the graph pool reusing and overwriting
            that storage before backward capture records the read.
            """

            metadata = getattr(tensor, "cg_buffer_metadata", None)
            return bool(
                torch.is_tensor(tensor) and metadata is not None and metadata.is_saved_for_backward
            )

        def is_differentiable_cudagraph_output_escape(tensor) -> bool:
            """Return whether a differentiable graph output escapes to eager code.

            Outputs that are also inputs to another CUDA graph are protected by graph-to-graph
            reuse accounting. However, an output that is not another graph's input has no
            such owner. Preserving it's ownership guards against premature graph-pool storage
            reuse across that graph boundary.
            """
            metadata = getattr(tensor, "cg_buffer_metadata", None)
            return bool(
                torch.is_tensor(tensor)
                and tensor.requires_grad
                and metadata is not None
                and metadata.is_cudagraph_output
                and not metadata.is_cudagraph_input
            )

        def weakref_input(tensor):
            if preserve_forward_to_backward_lifetimes:
                if is_saved_for_backward(tensor):
                    return tensor
            return make_weakref(tensor)

        def weakref_output(tensor):
            if preserve_forward_to_backward_lifetimes:
                if is_saved_for_backward(tensor):
                    return tensor
                if is_differentiable_cudagraph_output_escape(tensor):
                    return tensor
            return make_weakref(tensor)

        self.fwd_graph_input_surface = tree_map(weakref_input, self.fwd_graph_input_surface)
        self.fwd_graph_input_args = tree_map(weakref_input, self.fwd_graph_input_args)
        self.fwd_graph_input_kwargs = tree_map(weakref_input, self.fwd_graph_input_kwargs)
        self.fwd_graph_outputs = tree_map(weakref_output, self.fwd_graph_outputs)
        self.fwd_graph_output_surface = tree_map(weakref_output, self.fwd_graph_output_surface)

    def create_fwd_graph(self, args, kwargs, outputs=None, clone_inputs=True):
        """Create a fwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        global fwd_buffer_reuse_ref_count

        self.args = args
        self.kwargs = kwargs
        self.outputs = outputs

        # Save buffers, grads, and other variables that may be affected by graph warmup.
        # For example, megatron/core/transformer/moe/router.py's expert_bias is a persistent
        # buffer updated each forward pass by '_apply_expert_bias()'. So we need to ensure
        # graph capture's forward passes do not corrupt its value. Inference is not affected
        # (no known buffer mutators) and would add new buffers (lazy MoE _fc1_weight/
        # _fc2_weight) that misalign the positional restore.

        if self.training and torch.is_grad_enabled():
            buffer_backup = []
            for buf in self.base_module.buffers():
                buffer_backup.append(buf.clone())

            grad_backup = _backup_grads_before_capture(self)

            saved_fp8_tensors = None
            if self.fp8_enabled:
                if is_te_min_version("1.13.0"):
                    saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp8_recipe)
                else:
                    saved_fp8_tensors = save_fp8_tensors(
                        [self.base_module], self.fp8_recipe.amax_history_len
                    )
            elif self.fp4_enabled:
                if is_te_min_version("2.7.0.dev0"):
                    saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp4_recipe)
                else:
                    raise ValueError("FP4 requires TE >= 2.7.0.dev0 for NVFP4BlockScaling support.")

        # cache the moe aux loss if needed, which is accumulated inside the forward pass
        from megatron.core.transformer.transformer_layer import MoETransformerLayer

        is_moe = isinstance(self.base_module, MoETransformerLayer)
        if is_moe:
            from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker

            moe_metrics_tracker = get_moe_metrics_tracker()
            cached_aux_losses = {}
            for name, entry in moe_metrics_tracker.metrics.items():
                cached_aux_losses[name] = entry.values.clone()

        self.fwd_graph = torch.cuda.CUDAGraph()
        _CudagraphGlobalRecord.captured_runners.add(self)

        # For cases with multiple active RNG states, e.g. TP.
        rng_states = get_all_rng_states()
        with torch.inference_mode(mode=False):
            for gen in rng_states.values():
                gen = _ensure_generator_state_is_cudagraph_safe(gen)
                if cudagraph_needs_generator_registration():
                    self.fwd_graph.register_generator_state(gen)

        args_to_clear_buffers = []

        def _resolve_input_buffer(ten):
            if not isinstance(ten, ArgMetadata):
                return ten
            metadata = getattr(ten, "cg_buffer_metadata", None)

            # the input tensor is resued from another cudagraph's input or output
            if metadata is not None and metadata.fwd_cudagraph_buffer is not None:
                shared_buf = metadata.fwd_cudagraph_buffer
                buf_metadata = shared_buf.cg_buffer_metadata

                assert metadata.is_cudagraph_input and buf_metadata.capture_reuse_count > 0

                can_skip_replay_copy = not (
                    metadata.input_use_count > 1
                    and metadata.input_use_count == buf_metadata.capture_reuse_count
                )

                buf_metadata.capture_reuse_count -= 1
                if buf_metadata.capture_reuse_count == 0:
                    args_to_clear_buffers.append(ten)

                buf = create_strong_ref(shared_buf)
            else:
                # need to provide a fresh buffer from the pool
                buf = alloc_tensor_from_graph_mempool(ten)
                if metadata is not None:
                    buf.cg_buffer_metadata = deepcopy(metadata)
                can_skip_replay_copy = False

            buf.can_skip_replay_copy = can_skip_replay_copy
            return buf

        if clone_inputs:
            # Recorded graph arguments are ArgMetadata, not tensors. Preallocate a shared
            # buffer before resolving each occurrence so later graph inputs can alias it.
            for ten in self.get_arg_metas(args, kwargs):
                metadata = getattr(ten, "cg_buffer_metadata", None)
                if (
                    metadata is not None
                    and metadata.input_use_count > 1
                    and metadata.fwd_cudagraph_buffer is None
                ):
                    buf = alloc_tensor_from_graph_mempool(ten)
                    buf.cg_buffer_metadata = deepcopy(metadata)
                    buf.cg_buffer_metadata.capture_reuse_count = metadata.input_use_count
                    metadata.fwd_cudagraph_buffer = buf
                    fwd_buffer_reuse_ref_count += 1

            self.fwd_graph_input_args = tree_map(_resolve_input_buffer, args)
            self.fwd_graph_input_kwargs = tree_map(_resolve_input_buffer, kwargs)
        else:
            self.fwd_graph_input_args, self.fwd_graph_input_kwargs = args, kwargs

        self.fwd_graph_input_surface = self.get_tensors(
            self.fwd_graph_input_args, self.fwd_graph_input_kwargs
        )

        grad_context = torch.no_grad() if not self.grad_enabled else nullcontext()
        warmup_comm_context = track_gtp_capture_comms() if self.gtp_remat else nullcontext()
        with grad_context, warmup_comm_context as warmup_comms:
            # Warm up again because CUDA graph capture mode may execute a different codepath
            _set_warmup_start()

            # Recompute parameter discovery needs one warmup output: graph warmup bypasses
            # CheckpointWithoutOutput, making parameters hidden by its nested backward reachable
            # from that output's autograd graph. Force this pass even when configured warmup is 0.
            num_warmup_steps = max(self.num_warmup_steps, int(self.needs_recompute_param_discovery))
            for _ in range(num_warmup_steps):
                preserve_prefetch_context = (
                    preserve_gtp_prefetch_state(self.base_module.parameters())
                    if self.gtp_remat
                    else nullcontext()
                )
                with preserve_prefetch_context:
                    with self.get_quantization_context():

                        def clone_ten(ten):
                            if not torch.is_tensor(ten):
                                return ten
                            return torch.clone(ten).detach().requires_grad_(ten.requires_grad)

                        warmup_args = tree_map(clone_ten, self.fwd_graph_input_args)
                        warmup_kwargs = tree_map(clone_ten, self.fwd_graph_input_kwargs)
                        warmup_outputs = self.func(*warmup_args, **warmup_kwargs)

                    if self.grad_enabled:
                        warmup_outputs = self.get_tensors(warmup_outputs)
                        warmup_outputs = tuple(o for o in warmup_outputs if o.requires_grad)
                        input_tensors = self.get_tensors(warmup_args, warmup_kwargs)
                        torch.autograd.grad(
                            outputs=warmup_outputs,
                            inputs=tuple(i for i in input_tensors if i.requires_grad),
                            grad_outputs=tuple(torch.zeros_like(o) for o in warmup_outputs),
                            only_inputs=True,
                            allow_unused=True,
                        )

                    if self.gtp_remat:
                        # Join only GRAPHED-chain communication issued by this runner. Selecting
                        # by both capture ownership and chain keeps UNGRAPHED work on its eager
                        # lifecycle instead of turning it into a cross-graph handoff.
                        warmup_params, warmup_ag_streams, warmup_rs_streams = (
                            warmup_comms.get_comms_for_chains(is_graphed_chain)
                        )
                        wait_async_comms(graphed_only=True, params=warmup_params)
                        self._wait_side_streams(warmup_ag_streams)
                        self._wait_side_streams(warmup_rs_streams)
            _set_warmup_end()

        with grad_context:
            with self.get_quantization_context():
                torch.cuda.synchronize()
                # Register default CUDA generators ourselves (fixed in-place to have normal tensors)
                # before capture begins, to avoid inference-tensor state issues during capture.
                with torch.inference_mode(mode=False):
                    for device_idx in range(torch.cuda.device_count()):
                        default_gen = _ensure_generator_state_is_cudagraph_safe(
                            torch.cuda.default_generators[device_idx]
                        )
                        if cudagraph_needs_generator_registration():
                            self.fwd_graph.register_generator_state(default_gen)

                capture_comm_context = (
                    track_gtp_capture_comms() if self.gtp_remat else nullcontext(None)
                )
                with (
                    no_gc_during_capture(freeze=True),
                    capture_comm_context as capture_comms,
                    capture_constants(self.fwd_graph),
                    torch.cuda.graph(
                        self.fwd_graph,
                        pool=self.mempool,
                        stream=CudaGraphManager.global_capture_stream,
                        capture_error_mode="thread_local",
                    ),
                ):

                    fwd_graph_outputs = self.func(
                        *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                    )

                    if self.use_stream:
                        # Release the caller before joining outgoing GTP prefetches. The next local
                        # graph can launch while this graph's tail AG runs, then wait on that AG's
                        # external event before reading the prefetched weight.
                        self.fwd_completion_event.record()

                    if self.gtp_remat:
                        # Forward only issues AG prefetches. Drain the operations directly.
                        captured_params, captured_ag_streams, _ = (
                            capture_comms.get_comms_for_chains(is_graphed_chain)
                        )
                        wait_async_comms(graphed_only=True, skip_rs=True, params=captured_params)
                        self._wait_side_streams(captured_ag_streams)

                if self.gtp_remat:
                    self._gtp_fwd_params_to_ensure_ready = tuple(
                        capture_comms.params_to_ensure_ready
                    )

                # gc.collect() drops references to unreachable tensors created during capture,
                # returning their storage to the allocator to avoid a slowdown during replay.
                # However, it forces expensive global garbage collection, so must be done
                # only on the last layer per-device to avoid slowing down graph creation.
                if FREEZE_GC and self.is_last_layer:
                    gc.collect()

        # Deallocate buffers forwarded from previous graphs that are no longer in use
        for arg in args_to_clear_buffers:
            arg.cg_buffer_metadata.fwd_cudagraph_buffer = None
            fwd_buffer_reuse_ref_count -= 1

        # save cudagraph output buffer
        self.fwd_graph_outputs = fwd_graph_outputs
        self.fwd_graph_output_surface = self.get_tensors(fwd_graph_outputs)

        for fwd_graph_out, o in zip(
            self.get_tensors(fwd_graph_outputs), self.get_arg_metas(self.outputs)
        ):
            metadata = getattr(o, "cg_buffer_metadata", None)
            assert metadata is not None and metadata.is_cudagraph_output
            fwd_graph_out.is_from_global_mempool = True
            fwd_graph_out.cg_buffer_metadata = deepcopy(metadata)

            if metadata.is_cudagraph_input and metadata.fwd_cudagraph_buffer is None:
                buf = create_strong_ref(fwd_graph_out)
                buf.cg_buffer_metadata.capture_reuse_count = metadata.cudagraph_reuse_ref_count
                metadata.fwd_cudagraph_buffer = buf
                fwd_buffer_reuse_ref_count += 1

        if self.training and torch.is_grad_enabled():
            assert (
                len(self.fwd_graph_output_surface) > 0
            ), """Tried graphing a module that returned no tensors in training mode, 
                however the graphed module must output at least one tensor, 
                so that a corresponding backward node may be registered in the autograd graph."""

            # Preserve only forward buffers whose lifetime crosses into backward capture.
            self._weakref_forward_buffers(preserve_forward_to_backward_lifetimes=True)

            if self.needs_recompute_param_discovery:
                self.params_to_backprop = self.get_connected_params(warmup_outputs)
            else:
                self.params_to_backprop = self.get_connected_params(fwd_graph_outputs)
            self.num_dgrads = len(self.fwd_graph_input_surface)
            self.fwd_graph_input_surface = self.fwd_graph_input_surface + self.params_to_backprop

            if self.fp8_enabled:
                restore_fp8_tensors([self.base_module], saved_fp8_tensors)
            # restore cached grads
            _restore_grads_after_capture(grad_backup)

            # restore cached buffers
            for buf_copy, buf in zip(buffer_backup, self.base_module.buffers()):
                buf.copy_(buf_copy)

        if is_moe:
            for name, cached_values in cached_aux_losses.items():
                assert (
                    name in moe_metrics_tracker.metrics
                ), "cached metrics must be found in the tracker."
                moe_metrics_tracker.metrics[name].values.copy_(cached_values)

    def create_bwd_graph(self):
        """Create a bwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        # unlike 'fwd_buffer_reuse_ref_count', 'bwd_buffer_reuse_ref_count' may not decrement
        # to 0 when activation checkpointing is used. See [interaction with recompute].
        global bwd_buffer_reuse_ref_count

        assert self.grad_enabled
        self.bwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        if cudagraph_needs_generator_registration():
            for _, state in get_all_rng_states().items():
                self.bwd_graph.register_generator_state(state)

        self.static_grad_outputs = []
        args_to_clear_buffers = []
        for o in self.get_arg_metas(self.outputs):
            out_grad = None
            if o.requires_grad:
                metadata = o.cg_buffer_metadata
                # TODO: (jiemingz) [interaction with recompute]
                # for activation recompute, the fwd pass is rerun in the backward pass and
                # the metadata we attach in record_graph_capture is lost. As a result the next
                # cudagraph expects the buffer to be provided 'fwd_cudagraph_buffer' but is missing.
                # So, we cannot always assume this metadata exists. Consequently, there are extra
                # copies between the outputs of the fwd-bwd pass and the bwd pass.
                if metadata.is_cudagraph_input and metadata.bwd_cudagraph_buffer is not None:
                    out_grad = metadata.bwd_cudagraph_buffer
                    args_to_clear_buffers.append(o)
                    out_grad.cg_buffer_metadata.capture_reuse_count -= 1
                else:
                    out_grad = alloc_tensor_from_graph_mempool(o)
            self.static_grad_outputs.append(out_grad)

        capture_comm_context = track_gtp_capture_comms() if self.gtp_remat else nullcontext(None)
        with (
            no_gc_during_capture(freeze=True),
            capture_comm_context as capture_comms,
            _preserve_parameter_grads_during_backward_capture(self),
            capture_constants(self.bwd_graph),
            torch.cuda.graph(
                self.bwd_graph, pool=self.mempool, stream=CudaGraphManager.global_capture_stream
            ),
        ):

            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in self.fwd_graph_output_surface if o.requires_grad),
                inputs=tuple(i for i in self.fwd_graph_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in self.static_grad_outputs if o is not None),
                retain_graph=self.backward_retain_grad,
                only_inputs=True,
                allow_unused=True,
            )
            # Accumulate wgrads directly into main_grad inside the graph
            n_act_grads = sum(
                1 for i in self.fwd_graph_input_surface[: self.num_dgrads] if i.requires_grad
            )
            for param, wgrad in zip(self.params_to_backprop, grad_inputs[n_act_grads:]):
                grad_already_accumulated = getattr(param, 'grad_added_to_main_grad', False)
                if wgrad is not None and not grad_already_accumulated:
                    param.main_grad.add_(wgrad)
                if param.grad is not None and not grad_already_accumulated:
                    # A nested backward can accumulate directly into param.grad instead of
                    # returning that contribution through the outer autograd.grad call.
                    param.main_grad.add_(param.grad)
                param.grad_added_to_main_grad = True

            # GTP cross-graph RS overlap, two phases:
            #   Phase 1 — drain AG, fence runner_stream past ag_stream's tail,
            #             then record bwd_completion_event so main_stream can
            #             release the next runner while RS is still in flight.
            #   Phase 2 — drain RS wait on rs_stream. For cross-graph chain
            #             tails the wait is captured here, the add in the
            #             consumer's cascade; for within-graph tails both
            #             happen here (see wait_async_comms).
            if self.gtp_remat:
                captured_params, captured_ag_streams, captured_rs_streams = (
                    capture_comms.get_comms_for_chains(is_graphed_chain)
                )
                # Phase 1: drain AG
                wait_async_comms(graphed_only=True, skip_rs=True, params=captured_params)
                self._wait_side_streams(captured_ag_streams)

                # Release the next runner after AG drain but before RS drain.
                self.bwd_completion_event.record()

                # Phase 2: in-graph RS drain + finalize.
                wait_async_comms(
                    graphed_only=True, finalize_after_drain=True, params=captured_params
                )
                self._wait_side_streams(captured_rs_streams)

            if self.use_stream and not self.gtp_remat:
                # Non-GTP path: record after the side-stream join.
                self.bwd_completion_event.record()

        # Backward reads saved forward constants without calling the embedding again.
        _CudagraphGlobalRecord.capture_constants[self.bwd_graph].update(
            _CudagraphGlobalRecord.capture_constants.get(self.fwd_graph, {})
        )

        self._gtp_wgrad_ring_slots = list(capture_comms.wgrad_ring_slots) if self.gtp_remat else []
        self._set_gtp_finalize_hook_plan(capture_comms.finalized_params if self.gtp_remat else ())

        for arg in args_to_clear_buffers:
            arg.cg_buffer_metadata.bwd_cudagraph_buffer = None
            bwd_buffer_reuse_ref_count -= 1

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs
        # that don't require grad
        grad_inputs = list(grad_inputs)
        self.static_grad_inputs = []
        for input_tensor in self.get_arg_metas(self.args, self.kwargs):
            if input_tensor.requires_grad:
                metadata = input_tensor.cg_buffer_metadata
                input_grad = grad_inputs.pop(0)
                input_grad.is_from_global_mempool = True
                input_grad.cg_buffer_metadata = deepcopy(metadata)

                if metadata.is_cudagraph_output and metadata.bwd_cudagraph_buffer is None:
                    buf = create_strong_ref(input_grad)
                    metadata.bwd_cudagraph_buffer = buf
                    buf.cg_buffer_metadata.capture_reuse_count += 1
                    bwd_buffer_reuse_ref_count += 1
                self.static_grad_inputs.append(input_grad)
            else:
                self.static_grad_inputs.append(None)

        assert len(self.params_to_backprop) == len(grad_inputs)
        self.static_grad_inputs = tuple(self.static_grad_inputs)
        self.static_grad_outputs = tuple(self.static_grad_outputs)

        # It is safe to weakref static_grad_inputs as any inuse input grads have a strong ref
        # stored in 'bwd_cudagraph_buffer'
        self.static_grad_inputs = tree_map(make_weakref, self.static_grad_inputs)
        self.static_grad_outputs = tree_map(make_weakref, self.static_grad_outputs)
        # Backward capture is the final recorded use of forward buffers retained for autograd.
        self._weakref_forward_buffers(preserve_forward_to_backward_lifetimes=False)

        delattr(self, "args")
        delattr(self, "kwargs")
        delattr(self, "outputs")

    def apply_cudagraph_record_metadata(self, args, kwargs, outputs):
        """Attaches graph capture metadata to all passed in tensors."""

        for t in self.get_tensors(args, kwargs):
            cg_buffer_metadata = _apply_cudagraph_buffer_metadata(t)
            cg_buffer_metadata.is_cudagraph_input = True
            cg_buffer_metadata.input_use_count += 1

            if cg_buffer_metadata.is_cudagraph_output:
                cg_buffer_metadata.cudagraph_reuse_ref_count += 1

        for t in self.get_tensors(outputs):
            _apply_cudagraph_buffer_metadata(t, is_output=True)

    def _make_pipeline_output_viewless(self, output):
        """Make a last-layer record or replay output safe for pipeline deallocation."""

        # A cross-module pipeline stage may have module-local PP size 1 while its output is still
        # sent and pseudo-deallocated, so deallocation—not local PP size—is the required signal.
        if not (self.is_last_layer and self.deallocate_pipeline_outputs):
            return output

        return tree_map(
            lambda value: (
                make_viewless_tensor(inp=value, requires_grad=value.requires_grad, keep_graph=True)
                if torch.is_tensor(value)
                else value
            ),
            output,
        )

    def record_graph_capture(self, args, kwargs):
        """Records the data needed to create this runner's forward cudagraph.
        The first pass records a graph and appends the runner to _CudagraphGlobalRecord.
        The actual cudagraph will be created when 'create_cudagraphs()` is called. Subsequent
        passes should replay the graph."""

        # Run the forward pass as normal in eager mode.
        out = self.func(*args, **kwargs)

        if type(out) != tuple:
            out = (out,)

        # Register a noop autograd node that toggles `self.graph_status` in the bwd pass, which
        # tracks when the runner completes its bwd pass.
        # If it's the first bwd encountered by this runner, record it to _CudagraphGlobalRecord
        # We record the noop autograd node to the first output tensor. This is sufficient for
        # TransformerLayer and MambaLayer as their output is just the hidden_states.
        out = tuple(
            [
                _CudagraphRecordNode.apply(self, o) if torch.is_tensor(o) and i == 0 else o
                for i, o in enumerate(out)
            ]
        )
        # Custom autograd Function outputs are views. Pipeline schedules may pseudo-deallocate
        # this first-pass output before create_cudagraphs() switches the runner to replay mode.
        out = self._make_pipeline_output_viewless(out)

        if not self.fwd_graph_recorded:
            logger.debug(f"Recording forward graph creation...")

            self.apply_cudagraph_record_metadata(args, kwargs, out)

            # Sequence metadata (cu_seqlens) must hold real values until TransfomerEngine#3205.
            warmup_value_ids = {
                id(value)
                for leaf in tree_flatten_pyt((args, kwargs))[0]
                if isinstance(leaf, PackedSeqParams)
                for value in vars(leaf).values()
                if torch.is_tensor(value)
            }

            def _replace_with_meta(arg):
                if not torch.is_tensor(arg):
                    return arg
                return ArgMetadata(arg, keep_values=id(arg) in warmup_value_ids)

            m_args = tree_map(_replace_with_meta, args)
            m_kwargs = tree_map(_replace_with_meta, kwargs)
            m_out = tree_map(_replace_with_meta, out)
            _CudagraphGlobalRecord.record_fwd_graph(self, m_args, m_kwargs, m_out)

            if HAVE_TE_GRAPHS:
                if FP8GlobalStateManager.is_fp8_enabled():
                    # check if the low precision recipe is either fp4 or fp8
                    if is_te_min_version("2.7.0.dev0"):
                        from transformer_engine.common.recipe import NVFP4BlockScaling

                        recipe = FP8GlobalStateManager.get_fp8_recipe()
                        if isinstance(recipe, NVFP4BlockScaling):
                            self.fp4_runtime_enabled = True
                        else:
                            self.fp8_runtime_enabled = True
                    else:
                        self.fp8_runtime_enabled = True

            self.fwd_graph_recorded = True

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def replay_graph_capture(self, is_first_microbatch, args, kwargs):
        """Replay the fwd cuda graph with autograd."""

        # Arguments passed to a cudagraph for replay must match the args in the captured graph.
        #  Tensor arguments need to have the same shape, dtype, and device location.
        #  All other arguments must have the exact same memory addresses for graph safety.
        mismatch_errors = self.get_mismatch_errors(args, kwargs)
        if mismatch_errors:
            error_msg = "CUDA graph argument mismatch:\n" + "\n".join(mismatch_errors)
            raise AssertionError(error_msg)

        if self._gtp_fwd_params_to_ensure_ready:
            ensure_params_ready(self._gtp_fwd_params_to_ensure_ready)

        inp_tensors = self.get_tensors(args, kwargs, check_types=False)
        if self.grad_enabled:
            func_args = inp_tensors + self.params_to_backprop
        else:
            func_args = inp_tensors

        out = _CudagraphReplayNode.apply(self, is_first_microbatch, *func_args)
        # The replay node has the same custom-autograd output-view behavior as the record node.
        out = self._make_pipeline_output_viewless(out)

        out_iter = iter(self.to_list(out))
        fwd_outputs = self.to_list(self.fwd_graph_outputs)
        return tuple(next(out_iter) if torch.is_tensor(o) else o for o in fwd_outputs)

    def get_mismatch_errors(self, args, kwargs):
        """Return list of detailed errors for mismatched cudagraph args."""
        errors = []

        def add_error(msg):
            errors.append(f"  - {msg}")

        def check(val, ref, context):

            assert isinstance(val, ArgMetadata)
            assert isinstance(ref, ArgMetadata)

            _check_supported_type(val)
            _check_supported_type(ref)

            if val.type != ref.type and not (is_dataclass(val.value) and is_dataclass(ref.value)):
                add_error(
                    f"Type mismatch at {context}: Received {val.type} but expected {ref.type}"
                )
                return False

            if ref.type == torch.Tensor or issubclass(ref.type, torch.Tensor):
                mismatches = []
                if val.shape != ref.shape:
                    mismatches.append(f"Received shape {val.shape} but expected {ref.shape}")
                if val.dtype != ref.dtype:
                    mismatches.append(f"Received dtype {val.dtype} but expected {ref.dtype}")
                if val.device != ref.device:
                    mismatches.append(f"Received device {val.device} but expected {ref.device}")
                if mismatches:
                    add_error(f"Tensor mismatch at {context}: {', '.join(mismatches)}")

            elif is_dataclass(ref.value):
                for field in dataclasses.fields(ref.value):
                    check(
                        ArgMetadata(getattr(val.value, field.name)),
                        ArgMetadata(getattr(ref.value, field.name)),
                        f"{context}.{field.name}",
                    )
            elif val.value != ref.value:
                add_error(f"Value mismatch at {context}: {val.value} vs {ref.value}")

        # Check positional arguments
        if len(args) != len(self.fwd_graph_input_arg_metas):
            add_error(
                f"Argument count mismatch: {len(args)} vs {len(self.fwd_graph_input_arg_metas)}"
            )
        else:
            for i, (arg, graph_arg_meta) in enumerate(zip(args, self.fwd_graph_input_arg_metas)):
                check(ArgMetadata(arg), graph_arg_meta, f"args[{i}]")

        # Check keyword arguments
        kwargs_keys = set(kwargs.keys())
        graph_keys = set(self.fwd_graph_input_kwarg_metas.keys())

        if missing_keys := graph_keys - kwargs_keys:
            add_error(f"Missing kwargs: {missing_keys}")
        if extra_keys := kwargs_keys - graph_keys:
            add_error(f"Unexpected kwargs: {extra_keys}")

        for k in kwargs_keys & graph_keys:
            check(ArgMetadata(kwargs[k]), self.fwd_graph_input_kwarg_metas[k], f"kwargs['{k}']")

        return errors

    def get_arg_metas(self, args, kwargs=None):
        """Replaces all passed in tensors with 'ArgMetadata' and returns them as a list."""
        arg_metas = []

        def collect(item):
            if isinstance(item, ArgMetadata):
                arg_metas.append(item)
            return item  # tree_map expects a return value to rebuild the tree

        tree_map(collect, args)
        if kwargs is not None:
            tree_map(collect, kwargs)

        return arg_metas

    def get_tensors(self, args, kwargs=None, check_types=True):
        """
        Filter and flatten all tensors from args and kwargs using list comprehensions
        and itertools.chain for faster flattening.
        """

        def extract_tensors(arg):
            if check_types:
                _check_supported_type(ArgMetadata(arg))
            if torch.is_tensor(arg):
                return [arg]

            if is_dataclass(arg):
                return [
                    attr
                    for field in dataclasses.fields(arg)
                    if torch.is_tensor(attr := getattr(arg, field.name))
                ]

            return []

        if torch.is_tensor(args):
            return (args,)

        args_tens = [tensor for arg in args for tensor in extract_tensors(arg)] if args else []
        kwargs_tens = (
            [tensor for val in kwargs.values() for tensor in extract_tensors(val)] if kwargs else []
        )

        return tuple(chain(args_tens, kwargs_tens))

    def to_list(self, x):
        """Helper function to wrap an input into a list"""
        return [x] if torch.is_tensor(x) else list(x)


class CudaGraphManager(torch.nn.Module):
    """Creates and runs cudagraphs for a megatron module"""

    """A global mempool for when 'cuda_graph_use_single_mempool' is used."""
    global_mempool = None

    """The stream local CUDA graphs warm up and capture on."""
    global_capture_stream = None

    def __init__(
        self,
        config: TransformerConfig,
        base_module=None,
        function_name=None,
        need_backward=True,
        pg_collection=None,
        inline_capture=False,
        num_warmup_steps=None,
    ):
        super().__init__()
        """Creates a CudaGraphManager to manage CUDA graphs for a Megatron module.

        Args:
            config: TransformerConfig object containing CUDA graph settings for memory
                pooling, graph retention, gradient accumulation, FP8/FP4, and warmup steps.
            inline_capture: Normally, whether the inline capture path is taken depends on whether
                `inference_context` is present in the kwargs of the forward call.
                Setting this argument to True always forces the inline capture path to be taken.
            num_warmup_steps: If set, overrides the per-runner warmup step count.
        """
        self._inline_capture = inline_capture
        self._num_warmup_steps = num_warmup_steps
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection
        rng_tracker = get_cuda_rng_tracker()
        self.need_backward = need_backward

        if function_name is not None:
            func = getattr(base_module, function_name)

            def wrapped_func(*args, eager=False, cache_key=None, **kwargs):
                if eager:
                    return func(*args, **kwargs)
                out = self(base_module, args, kwargs, cache_key=cache_key)
                # Unwrap single-element tuple to match the original function's return type.
                if isinstance(out, tuple) and len(out) == 1:
                    return out[0]
                return out

            setattr(base_module, function_name, wrapped_func)
        else:
            func = None
        self.func = func

        # need to delay the import here to avoid a circular import
        global HAVE_TE_GRAPHS
        try:
            from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
        except ImportError:
            TECudaRNGStatesTracker = None

        assert (
            rng_tracker.is_inference_rng_tracker
            or (HAVE_TE_GRAPHS and isinstance(rng_tracker, TECudaRNGStatesTracker))
            or (isinstance(rng_tracker, CudaRNGStatesTracker) and rng_tracker.use_cudagraphable_rng)
        ), "RNG tracker does not support cudagraphs!"

        assert config.cuda_graph_impl == "local", "Option cuda_graph_impl=local not enabled."
        self.cudagraph_runners: list[_CudaGraphRunner] = []
        self.custom_cudagraphs_lookup_table: dict = defaultdict(lambda: None)
        self.is_first_microbatch = False

        # Without pipeline parallelism, microbatches execute one at a time.
        # Therefore modules will always execute in the same order, so cudagraphs
        # can both be reused and share a single mempool.
        self.reuse_cudagraphs = self.pg_collection.pp.size() == 1
        self._initialize_mempool()
        if CudaGraphManager.global_capture_stream is None:
            # Capture on the stream DDP is initialized on, so the AccumulateGrad nodes DDP retains
            # carry the capturing stream.
            from megatron.core.full_cuda_graph import get_shared_capture_stream

            CudaGraphManager.global_capture_stream = get_shared_capture_stream()
            # Run the rest of model, optimizer and DDP construction, the eager iterations and the
            # warmup before capture on this stream. Training returns to the default stream after
            # capture, whose allocator cache then holds no long-lived state to fragment it.
            CudaGraphManager.global_capture_stream.wait_stream(torch.cuda.current_stream())
            torch.cuda.set_stream(CudaGraphManager.global_capture_stream)

        # Enable one hook for the eager recording phase. Repeated manager construction is
        # idempotent, and graph creation removes the hook before capture begins.
        if need_backward:
            _CudagraphGlobalRecord._enable_saved_tensors_observer()

    @staticmethod
    def _initialize_mempool() -> None:
        """Create and register the graph pool without changing the current CUDA stream."""
        if CudaGraphManager.global_mempool is None:
            CudaGraphManager.global_mempool = torch.cuda.graph_pool_handle()
            # Register the pool so GTP allocates GRAPHED-chain buffers + quantized
            # storage directly into it (created before the first graphed forward).
            if HAVE_GTP:
                set_cuda_graph_mempool(torch.cuda.current_device(), CudaGraphManager.global_mempool)

    def call_ddp_preforward_hook(self, module):
        """Call any DDP pre-forward hooks which are used to launch async data parallel
        param gather. Any other pre-forward hooks are not allowed."""

        from megatron.core.distributed import distributed_data_parallel

        if module._forward_pre_hooks:
            for _, hook in module._forward_pre_hooks.items():
                assert (
                    inspect.getmodule(hook) == distributed_data_parallel
                ), "Tried to cudagraph a module with user registered pre-forward hooks, \
                which is not allowed."
                # Only hooks from Mcore DDP, which take no args, should be called at this point.
                hook(module)

    def get_cudagraph_runner(self, megatron_module, args, kwargs, reuse_cudagraphs, cache_key=None):
        '''Returns a valid cudagraph runner for the current forward call.
        The cudagraph corresponding to this call is the first element of 'self.cudagraph_runners'.
        We iterate through the list by 1 for each call, and the number of calls is equal to the
        length of 'self.cudagraph_runners'.
        Otherwise, we assign a mempool per microbatch, which allows cudagraphs to be reused
        over different microbatches by tracking their respective fwd and bwd passes.'''

        self._initialize_mempool()
        if self.need_backward and not _CudagraphGlobalRecord.cudagraph_created:
            _CudagraphGlobalRecord._enable_saved_tensors_observer()

        if reuse_cudagraphs:
            if cache_key is not None:
                runner = self.custom_cudagraphs_lookup_table[cache_key]
            else:
                # Todo: For training, we could also cache runners based on input shape.
                # If autograd is currently disabled, it doesnt matter if a runner was created
                # with or without autograd, so just get the first fwd ready runner.
                require_grad = self.need_backward and torch.is_grad_enabled()

                def is_valid(r):
                    return (
                        r.status == _GraphStatus.FWD_READY
                        and not r.get_mismatch_errors(args, kwargs)
                        and (not require_grad or r.grad_enabled)
                    )

                # We must choose the first available runner, as the order of
                # self.cudagraph_runners corresponds to the capture order.
                runner = next((r for r in self.cudagraph_runners if is_valid(r)), None)

            if runner is None:
                if _CudagraphGlobalRecord.cudagraph_created:
                    assert False, (
                        f"`cudagraph_created` is set to True but no matching cudagraph "
                        f"runners were found. This module has {len(self.cudagraph_runners)} "
                        f"existing runners. Use `get_mismatch_errors` to debug mismatches."
                    )
                else:
                    runner = _CudaGraphRunner(
                        megatron_module,
                        CudaGraphManager.global_mempool,
                        args,
                        kwargs,
                        self.func,
                        self.need_backward,
                    )
                    if self._num_warmup_steps is not None:
                        runner.num_warmup_steps = self._num_warmup_steps
                    self.cudagraph_runners.append(runner)
                    if cache_key is not None:
                        self.custom_cudagraphs_lookup_table[cache_key] = runner
        else:
            # Create cudagraphs for every microbatch
            if _CudagraphGlobalRecord.cudagraph_created:
                runner = self.cudagraph_runners[0]
                assert runner.status == _GraphStatus.FWD_READY
                self.cudagraph_runners = self.cudagraph_runners[1:] + self.cudagraph_runners[:1]
            else:
                runner = _CudaGraphRunner(
                    megatron_module,
                    CudaGraphManager.global_mempool,
                    args,
                    kwargs,
                    self.func,
                    self.need_backward,
                )
                self.cudagraph_runners.append(runner)

        if runner.mempool is None:
            runner.mempool = CudaGraphManager.global_mempool
        return runner

    def __call__(self, megatron_module, args, kwargs, cache_key=None):
        """Calls the forward pass of the cudagraphed module.

        Args:
            megatron_module (torch.nn.module): The megatron module to be graphed and run

            args (tuple):  The positional args to be passed to the module.

            kwargs (dict):  The keyword args to be passed to the module.

            cache_key: Optional hashable key for O(1) runner lookup.
                If `inference_context` is provided, this gets set to the correct value.
        """
        is_inference_mode = 'inference_context' in kwargs.keys() and kwargs['inference_context']
        if cache_key is None and is_inference_mode:
            inference_context = kwargs['inference_context']
            if inference_context.is_static_batching():
                hidden_states = args[0] if args else kwargs['hidden_states']
                batch_size = hidden_states.shape[0]
                cache_key = (batch_size, inference_context.is_decode_only())
            else:
                cache_key = inference_context.padded_batch_dimensions
        is_in_checkpoint_fwd = is_checkpointing()
        if HAVE_TE_GRAPHS:
            is_in_checkpoint_fwd = is_in_checkpoint_fwd or is_fp8_activation_recompute_enabled()

        if _CudagraphGlobalRecord.cudagraph_created:
            if self.training and torch.is_grad_enabled():
                # Trigger Mcore DDP pre-forward hooks
                self.call_ddp_preforward_hook(megatron_module)
                for module in megatron_module.modules():
                    self.call_ddp_preforward_hook(module)

            runner = self.get_cudagraph_runner(
                megatron_module, args, kwargs, self.reuse_cudagraphs, cache_key=cache_key
            )
            out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)
        else:
            if is_inference_mode or self._inline_capture:
                # Inference generation mode creates graphs immediately
                runner = self.get_cudagraph_runner(
                    megatron_module, args, kwargs, True, cache_key=cache_key
                )

                if not runner.fwd_graph_recorded:
                    # Reuse graph input-output buffers for inference
                    local_args, local_kwargs = args, kwargs
                    if not runner.is_first_layer:
                        # Find previous layer's runner in the global record.
                        # Method-wrapped managers (e.g. the MTP wrapper around
                        # `compute_mtp_single_step`) have a base_module without
                        # `layer_number`; `getattr(..., None)` makes those rows
                        # harmlessly skipped by the predicate.
                        try:
                            previous_runner = next(
                                r
                                for r in _CudagraphGlobalRecord.cudagraph_inference_record
                                if (
                                    getattr(r[0].base_module, 'layer_number', None)
                                    == runner.base_module.layer_number - 1
                                    and r[0].fwd_graph is not None
                                    and ArgMetadata(r[2][0] if r[2] else r[3]['hidden_states'])
                                    == ArgMetadata(args[0] if args else kwargs['hidden_states'])
                                )
                            )
                            # Replace the hidden states from previous layer's output buffer
                            previous_output = previous_runner[0].fwd_graph_outputs[0]
                            if args:
                                local_args = (previous_output, *args[1:])
                            else:
                                local_kwargs = dict(kwargs)
                                local_kwargs['hidden_states'] = previous_output
                        except StopIteration:
                            # No match found for previous layer, continue with no buffer reuse
                            pass

                    runner.create_fwd_graph(
                        local_args, local_kwargs, outputs=None, clone_inputs=runner.is_first_layer
                    )
                    runner.fwd_graph_recorded = True
                    runner.cudagraph_created = True
                    runner = runner.eval()

                    # Record this to the global execution record
                    _CudagraphGlobalRecord.cudagraph_inference_record.append(
                        (runner, "fwd", args, kwargs)
                    )

                # Now replay the graph
                out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)
            elif self.training or is_in_checkpoint_fwd:
                runner = self.get_cudagraph_runner(
                    megatron_module, args, kwargs, self.reuse_cudagraphs
                )
                out = runner.record_graph_capture(args, kwargs)
            else:
                # No cudagraphs were found in training mode with grad disabled, so fallback to
                # eager since autograd is needed to correctly trace the backward graph.
                if self.func is not None:
                    return self.func(*args, **kwargs)
                else:
                    return super(MegatronModule, megatron_module).__call__(*args, **kwargs)

        self.is_first_microbatch = False
        # If forward only, next replay should be a forward pass as well
        if is_inference_mode or not torch.is_grad_enabled() or not runner.fwd_graph_recorded:
            runner.status = _GraphStatus.FWD_READY
        else:
            runner.status = _GraphStatus.BWD_READY

        return out


# The following functions are for capturing CUDA Graphs using TE make_graphed_callables().
def _te_cuda_graph_layer_is_graphable(layer, config):
    """Check whether a module registered a TE graph for the requested scope."""
    return (
        config.cuda_graph_impl == "transformer_engine"
        and isinstance(layer, TECudaGraphMixin)
        and layer.config.cuda_graph_impl == "transformer_engine"
        and layer.is_graphable_for_scope(config.cuda_graph_modules)
    )


@contextmanager
def _keep_class_calls(callables) -> Iterator[None]:
    """Give the classes of TE graphed callables back their own __call__ after capture.

    make_graphed_callables wraps the __call__ of each callable's class in an FP8/FP4 autocast that
    looks up its enabled flag by module, and afterwards assigns each class the __call__ it had read
    through the MRO. A subclass wrapped after its base class, such as MoETransformerLayer after
    TransformerLayer, therefore keeps the base class's wrapper. Every later call of its layers then
    runs with FP8/FP4 off unless the layer was itself a callable of that capture, which the
    feed-forward layer of a pair and an eager MTP layer are not.

    Args:
        callables: The modules passed to make_graphed_callables.
    """
    missing = object()
    saved_calls = {}
    for module in callables:
        saved_calls.setdefault(type(module), vars(type(module)).get("__call__", missing))
    try:
        yield
    finally:
        for module_class, call in saved_calls.items():
            if call is not missing:
                module_class.__call__ = call
            elif "__call__" in vars(module_class):
                del module_class.__call__


class _TECudaGraphPairLink:
    """Replay-time link from a Hybrid layer to the TE graphed callable it shares.

    A plain object rather than a module, so assigning it to a layer leaves the layer's module tree,
    parameter names and state dict unchanged.

    Args:
        pair (_TECudaGraphLayerPair): The callable the layer shares with its neighbor.
        role (str): "mixer" for the layer that launches the graph, "partner" for the feed-forward
            layer that takes the graph outputs.
    """

    def __init__(self, pair: "_TECudaGraphLayerPair", role: str) -> None:
        self.pair = pair
        self.role = role


class _TECudaGraphLayerPair(torch.nn.Module):
    """One TE graphed callable for a Hybrid mixer layer and the feed-forward layer after it.

    HybridStack runs each block as two layers: a mixer (attention, GDN, ...) without a feed-forward
    branch, then an MoE or MLP layer (the partner) without an attention branch. Per-layer graphs
    launch two forward and two backward graphs per block. This callable captures the mixer's graph
    scope followed by the partner's graph scope as one graph. At replay, the mixer launches it and
    keeps the partner's graph outputs, and the partner runs only its eager remainder, such as the
    MoE dispatch, experts and combine.

    The pair is never registered in the model, so parameter names, checkpoints and the module tree
    stay as they are. Its ``parameters()`` are the union of both layers' parameters, which TE uses
    as the graph's parameter inputs, so the one backward graph returns the gradients of both.

    The partner takes the mixer's output in the dtype of its own graph's hidden_states input, as
    with per-layer graphs: with fp32_residual_connection, the mixer returns an FP32 residual, and
    the partner's graph input is BF16.

    Args:
        mixer (TransformerLayer): The attention-type layer, which launches the graph.
        partner (TransformerLayer): The feed-forward layer that follows it.
    """

    def __init__(self, mixer: TECudaGraphMixin, partner: TECudaGraphMixin) -> None:
        super().__init__()
        self.mixer = mixer
        self.partner = partner
        # Static-input keys owned by the partner, such as its router's padding_mask.
        self._partner_input_keys: tuple = ()
        # The dtype of the partner's own static hidden_states input. Set with the keys above, by
        # get_layer_static_inputs.
        self._partner_input_dtype: torch.dtype | None = None
        # Whether the mixer's output is appended to the graph outputs. Partial MoE scopes already
        # return it as the partner's residual, the last output, when the partner takes it without
        # a cast. Set at capture.
        self._appends_mixer_output: bool | None = None
        # (mixer output, partner graph outputs) from the mixer's replay, for the partner's call.
        self._stash = None
        mixer._te_cuda_graph_pair_link = _TECudaGraphPairLink(self, "mixer")
        partner._te_cuda_graph_pair_link = _TECudaGraphPairLink(self, "partner")

    @property
    def layer_number(self) -> int:
        """The mixer's layer number, which selects the FP8/FP4 first/last-layer status."""
        return self.mixer.layer_number

    @property
    def te_cuda_graphs(self) -> list:
        """The pair's graphs, one per captured microbatch."""
        return self.mixer.te_cuda_graphs

    @te_cuda_graphs.setter
    def te_cuda_graphs(self, graphs: list) -> None:
        # Both layers hold the same list, so both take the TE replay path.
        self.mixer.te_cuda_graphs = graphs
        self.partner.te_cuda_graphs = graphs

    @property
    def te_cuda_graph_manual_hooks(self) -> list:
        """The pre-forward hooks of both layers, run by the mixer before it launches the graph."""
        return self.mixer.te_cuda_graph_manual_hooks

    @te_cuda_graph_manual_hooks.setter
    def te_cuda_graph_manual_hooks(self, hooks: list) -> None:
        self.mixer.te_cuda_graph_manual_hooks = hooks
        self.partner.te_cuda_graph_manual_hooks = []

    def setup_manual_hooks(self, make_hook_func) -> None:
        """Set up the hooks of both layers' graphed parameters on the mixer."""
        self.mixer.setup_manual_hooks(make_hook_func)
        self.partner.setup_manual_hooks(make_hook_func)
        self.te_cuda_graph_manual_hooks = (
            self.mixer.te_cuda_graph_manual_hooks + self.partner.te_cuda_graph_manual_hooks
        )

    def _after_te_cuda_graphs_deleted(self) -> None:
        """Release what the pair's graphs pinned in either layer, after the graphs were deleted."""
        self.mixer._after_te_cuda_graphs_deleted()
        self.partner._after_te_cuda_graphs_deleted()

    def get_layer_static_inputs(self, seq_length: int, micro_batch_size: int) -> dict:
        """Return the mixer's static inputs plus the partner's other inputs."""
        static_inputs = self.mixer.get_layer_static_inputs(seq_length, micro_batch_size)
        partner_inputs = self.partner.get_layer_static_inputs(seq_length, micro_batch_size)
        # The partner takes the mixer's output instead, cast to this input's dtype; see forward.
        self._partner_input_dtype = partner_inputs.pop("hidden_states").dtype
        collisions = static_inputs.keys() & partner_inputs.keys()
        if collisions:
            raise ValueError(f"Paired TE CUDA graph inputs collide: {sorted(collisions)}.")
        self._partner_input_keys = tuple(partner_inputs)
        static_inputs.update(partner_inputs)
        return static_inputs

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple:
        """Capture body: the mixer's graph scope, then the partner's graph scope."""
        partner_kwargs = {key: kwargs.pop(key) for key in self._partner_input_keys if key in kwargs}
        mixer_outputs = self.mixer._te_cuda_graph_capture(hidden_states, **kwargs)
        if len(mixer_outputs) != 1:
            raise RuntimeError("A paired mixer's CUDA graph must return only its layer output.")
        mixer_output = mixer_outputs[0]
        # The partner's own graph would copy the mixer's output into its static input, rounding an
        # FP32 residual to BF16, and autograd would upcast that input's BF16 gradient for the
        # mixer. The cast does the same inside the pair, and returns mixer_output itself when the
        # dtypes already match.
        partner_input = mixer_output.to(self._partner_input_dtype)
        outputs = list(self.partner._te_cuda_graph_capture(partner_input, **partner_kwargs))
        appends_mixer_output = outputs[-1] is not mixer_output
        if self._appends_mixer_output is None:
            self._appends_mixer_output = appends_mixer_output
        elif self._appends_mixer_output != appends_mixer_output:
            raise RuntimeError("Paired CUDA graph outputs changed between capture passes.")
        if appends_mixer_output:
            outputs.append(mixer_output)
        return tuple(outputs)

    def stash_partner_outputs(self, graph_outputs: list) -> torch.Tensor:
        """At the mixer's replay, keep the partner's graph outputs and return the mixer output."""
        mixer_output = graph_outputs.pop() if self._appends_mixer_output else graph_outputs[-1]
        self._stash = (mixer_output, graph_outputs)
        return mixer_output

    def take_partner_outputs(self, hidden_states: torch.Tensor) -> list:
        """At the partner's replay, return the graph outputs of its mixer's replay."""
        stash, self._stash = self._stash, None
        if stash is None or stash[0] is not hidden_states:
            raise RuntimeError(
                "A paired feed-forward layer must be called right after its mixer, with the "
                "mixer's output, while replaying TE CUDA graphs."
            )
        return list(stash[1])


def _te_cuda_graph_pair_supported(mixer, partner, config) -> bool:
    """Whether two adjacent Hybrid layers can be captured as one TE graphed callable.

    The mixer must be an attention-type layer whose feed-forward branch is an identity, so that its
    graph output is its layer output, and the partner a feed-forward layer whose attention branch
    is an identity, so that its graph starts at its layer input. Both layers must replay TE graphs
    under their own configs: the mixer launches the pair's graph, and the partner takes its outputs.
    """
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if not (isinstance(mixer, TransformerLayer) and isinstance(partner, TransformerLayer)):
        return False
    # _te_cuda_graph_layer_is_graphable also reads each layer's own cuda_graph_impl, which a
    # hybrid_layer_config_list can set per layer. A mixer that runs eagerly never launches the
    # graph, so its partner finds no outputs to take; a partner that runs eagerly repeats the part
    # of its layer that the graph already ran.
    if not (
        _te_cuda_graph_layer_is_graphable(mixer, config)
        and _te_cuda_graph_layer_is_graphable(partner, config)
    ):
        return False
    if mixer.is_mtp_layer != partner.is_mtp_layer:
        return False
    mixer_is_attention_only = (
        not isinstance(mixer.self_attention, IdentityOp)
        and isinstance(mixer.cross_attention, IdentityOp)
        and isinstance(mixer.pre_mlp_layernorm, IdentityOp)
        and isinstance(mixer.mlp, IdentityOp)
        and isinstance(mixer.mlp_bda, IdentityOp)
    )
    partner_is_feed_forward_only = (
        isinstance(partner.input_layernorm, IdentityOp)
        and isinstance(partner.self_attention, IdentityOp)
        and isinstance(partner.cross_attention, IdentityOp)
        and not isinstance(partner.mlp, IdentityOp)
    )
    if not (mixer_is_attention_only and partner_is_feed_forward_only):
        return False
    # Hash routing reads input_ids, which HybridStack passes to the hash-routed layer only.
    if partner.is_moe_layer and getattr(partner.mlp.router, "is_hash_layer", False):
        return False
    # A graph that contains an MoE megakernel needs the layer's eager weight-view refresh
    # (_before_te_cuda_graph_replay) before every replay, which only the layer's own graph runs.
    if partner._cuda_graph_megakernel() is not None:
        return False
    # One graph runs both layers under one FP8/FP4 setting. All layers of an MTP block run under
    # the block's setting, to which the first/last-layer BF16 rule does not apply.
    if (config.fp8 or config.fp4) and not mixer.is_mtp_layer:
        from megatron.core.fp8_utils import is_first_last_bf16_layer

        if is_first_last_bf16_layer(config, mixer.layer_number - 1) != is_first_last_bf16_layer(
            config, partner.layer_number - 1
        ):
            return False
    return True


def _pair_hybrid_layers_enabled(config: TransformerConfig) -> bool:
    """Whether TE graphs pair adjacent Hybrid mixer and feed-forward layers.

    A pair skips whatever would run between its two layers, so features that convert layouts or
    schedule work between layers turn pairing off with a warning instead of failing.
    """
    if not config.cuda_graph_pair_hybrid_layers or config.cuda_graph_impl != "transformer_engine":
        return False
    blocking_features = [
        name
        for name, enabled in (
            ("overlap_moe_expert_parallel_comm", config.overlap_moe_expert_parallel_comm),
            ("delay_wgrad_compute", config.delay_wgrad_compute),
            ("fine_grained_activation_offloading", config.fine_grained_activation_offloading),
            (
                "sequence_packing_scheduler",
                getattr(config, "sequence_packing_scheduler", None) is not None,
            ),
            ("enable_mhc_connections", config.enable_mhc_connections),
            # TransformerConfig rejects wide residual streams with TE CUDA graphs today; this
            # keeps pairing off should that combination ever be allowed.
            ("wide_residual", config.wide_residual is not None),
            ("context_parallel_size > 1", config.context_parallel_size > 1),
        )
        if enabled
    ]
    if blocking_features:
        log_single_rank(
            logger,
            logging.WARNING,
            f'cuda_graph_pair_hybrid_layers is ignored with {", ".join(blocking_features)}; '
            'every layer keeps its own CUDA graph.',
        )
        return False
    return True


def _hybrid_mtp_graphs_supported(config: TransformerConfig) -> bool:
    """Whether cuda_graph_pair_hybrid_layers can also graph the layers of Hybrid MTP blocks.

    A graph keeps the activations of one call per microbatch, so a block that runs once per MTP
    depth (mtp_use_repeated_layer) stays eager. TE graphs capture MTP layers with the model's
    FP8/FP4 setting, so a block kept out of it (keep_mtp_in_bf16) stays eager too, and so does a
    block whose layer configs select another recipe (see _hybrid_mtp_recipe_differences). A
    graph's hidden_states input is BF16, so with fp32_residual_connection graphed layers would
    round the FP32 residual stream that the block's eager layers keep, and the block stays eager
    as well.
    """
    blocking_features = [
        name
        for name, enabled in (
            (
                "mtp_use_repeated_layer",
                config.mtp_use_repeated_layer and (config.mtp_num_layers or 0) > 1,
            ),
            ("keep_mtp_in_bf16", config.keep_mtp_in_bf16 and bool(config.fp8 or config.fp4)),
            ("fp32_residual_connection", config.fp32_residual_connection),
        )
        if enabled
    ]
    if blocking_features:
        log_single_rank(
            logger,
            logging.WARNING,
            f'Hybrid MTP layers are not CUDA-graphed with {", ".join(blocking_features)}; '
            'they run eagerly.',
        )
        return False
    return True


def _hybrid_mtp_moe_blocking_features(config: TransformerConfig) -> list:
    """Return the settings with which pairing keeps an MoE layer of a Hybrid MTP block eager.

    Per-layer graphs leave Hybrid MTP blocks eager, so their MoE layers never had to be capturable.
    A graph that ends inside the MoE layer (moe_router, moe_preprocess) returns its intermediate
    tensors, which the chunk loop of mlp_chunks_for_training cannot join. A graph that runs the
    token dispatch (moe, or empty cuda_graph_modules for the whole layer) needs static dispatch
    shapes. Pairing captures it only with drop-and-pad MoE or virtual-expert load balancing, the
    settings with which TransformerConfig allows the moe scope, and never with the allgather
    dispatcher, which copies token counts to the host in every mode. Such MoE layers stay eager,
    and the mixers before them keep their own graphs.

    Args:
        config (TransformerConfig): The MoE layer's own config, whose graph scope, MLP chunks and
            token dispatcher its capture uses.

    Returns:
        list: The names of the blocking settings. Empty when pairing can graph the layer.
    """
    modules = config.cuda_graph_modules
    partial_moe_graph = (
        CudaGraphModule.moe_router in modules or CudaGraphModule.moe_preprocess in modules
    )
    whole_moe_graph = not modules or CudaGraphModule.moe in modules
    drop_and_pad = (
        config.moe_expert_capacity_factor is not None and config.moe_pad_expert_input_to_capacity
    )
    return [
        name
        for name, enabled in (
            (
                "mlp_chunks_for_training > 1 and the moe_router scope",
                partial_moe_graph and config.mlp_chunks_for_training > 1,
            ),
            (
                "the moe or whole-layer scope with neither drop-and-pad MoE nor virtual-expert "
                "load balancing",
                whole_moe_graph and not (drop_and_pad or config.moe_virtual_expert_load_balance),
            ),
            (
                "the moe or whole-layer scope and the allgather dispatcher",
                whole_moe_graph and config.moe_token_dispatcher_type == "allgather",
            ),
        )
        if enabled
    ]


# The TransformerConfig fields from which get_fp8_recipe, get_fp4_recipe, get_fp8_context and
# get_fp4_context build a layer's FP8/FP4 recipe and amax reduction group.
_QUANTIZATION_RECIPE_FIELDS = (
    "fp8",
    "fp8_recipe",
    "fp8_margin",
    "fp8_interval",
    "fp8_amax_history_len",
    "fp8_amax_compute_algo",
    "fp8_wgrad",
    "fp8_dot_product_attention",
    "fp8_multi_head_attention",
    "fp8_quantizer_factory",
    "fp8_recipe_attrs",
    "fp4",
    "fp4_recipe",
    "fp4_quantizer_factory",
    "fp4_recipe_attrs",
    "tp_only_amax_red",
)


def _hybrid_mtp_recipe_differences(mtp_block, config: TransformerConfig) -> list:
    """Return the FP8/FP4 recipe settings in which Hybrid MTP layers differ from the model.

    make_graphed_callables takes one recipe for all callables, the model's. The eager MTP block
    runs inside the model's FP8/FP4 context and, except with delayed scaling, enters each layer's
    own context inside it, so a layer whose config selects another recipe can run differently
    once graphed. Such a block stays eager, as with per-layer graphs. The first/last-layer BF16
    settings are not compared: inside the model's context they cannot turn FP8/FP4 off for an MTP
    layer, and TE graphs enable FP8/FP4 for every MTP layer.

    Args:
        mtp_block (HybridStack): The mtp_model_layer of a Hybrid MTP block.
        config (TransformerConfig): The model config that TE graphs are captured with.

    Returns:
        list: The names of the differing settings, sorted. Empty without FP8/FP4 training, in
        which neither the eager block nor the graphs quantize.
    """
    if not (config.fp8 or config.fp4):
        return []
    return sorted(
        {
            name
            for layer_config in mtp_block.layer_config_list
            for name in _QUANTIZATION_RECIPE_FIELDS
            if getattr(layer_config, name) != getattr(config, name)
        }
    )


def _te_graph_callables(
    layers, config: TransformerConfig, pair_hybrid_layers: bool, eager_layers=()
) -> tuple:
    """Return the TE graphed callables of consecutive layers and how many of them are pairs.

    With pair_hybrid_layers, each mixer is captured with the layer after it where
    _te_cuda_graph_pair_supported allows it. Every other graphable layer keeps its own callable;
    as with per-layer graphs, a layer whose own config selects another cuda_graph_impl is not
    graphable and stays eager. The eager_layers are neither graphed nor paired.
    """
    callables, num_pairs, index = [], 0, 0
    while index < len(layers):
        layer = layers[index]
        partner = layers[index + 1] if pair_hybrid_layers and index + 1 < len(layers) else None
        if partner in eager_layers:
            partner = None
        if partner is not None and _te_cuda_graph_pair_supported(layer, partner, config):
            callables.append(_TECudaGraphLayerPair(layer, partner))
            num_pairs += 1
            index += 2
            continue
        if _te_cuda_graph_layer_is_graphable(layer, config) and layer not in eager_layers:
            callables.append(layer)
        index += 1
    return callables, num_pairs


def _is_mtp_te_graph_layer(layer, chunk_with_decoder) -> bool:
    """Whether a layer is the model layer of an MTP block, or a layer of a Hybrid MTP block."""
    for mtp_layer in getattr(getattr(chunk_with_decoder, "mtp", None), "layers", ()):
        mtp_model_layer = mtp_layer.mtp_model_layer
        if layer is mtp_model_layer or any(
            layer is inner_layer for inner_layer in getattr(mtp_model_layer, "layers", ())
        ):
            return True
    return False


class TECudaGraphHelper:
    """
    Helper class to capture CUDA Graphs using TE make_graphed_callables().
    It is used in the beginning of the training loop to capture per-layer CUDA Graphs.
    `self.create_cudagraphs()` should be called to capture the CUDA Graphs and
    `self.cuda_graph_set_manual_hooks()` should be called to set manual pre-forward hooks for the
    parameters that are covered by cudagraphs.
    """

    def __init__(
        self, model, config, seq_length, micro_batch_size, optimizers=[], pg_collection=None
    ):
        assert (
            config.cuda_graph_impl == "transformer_engine"
        ), "Option cuda_graph_impl=transformer_engine not enabled."
        assert HAVE_TE_GRAPHS, "CUDA Graphs are not supported without TE."
        assert (
            "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
            or os.getenv("NCCL_GRAPH_REGISTER", "") == "0"
        ), (
            "Setting NCCL_GRAPH_REGISTER=0 to avoid illegal memory access when using "
            "CUDA Graph with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
        )
        if getattr(config, "sequence_packing_scheduler", None) is not None:
            assert micro_batch_size == 1, "Scheduler THD graphs require micro_batch_size=1."
        self.model = model
        self.config = config
        self.seq_length = (
            config.max_seqlen_per_dp_cp_rank * config.context_parallel_size
            if getattr(config, "sequence_packing_scheduler", None) is not None
            and config.pad_packed_seq_alignment is not None
            else seq_length
        )
        self.micro_batch_size = micro_batch_size
        self.optimizers = optimizers
        self.pg_collection = pg_collection
        if self.pg_collection is None:
            self.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.tp_group = self.pg_collection.tp
        self.dp_cp_group = self.pg_collection.dp_cp
        self.pp_group = self.pg_collection.pp
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator

        self.p2p_communicator = P2PCommunicator(pp_group=self.pp_group, config=self.config)
        self.num_model_chunks = len(model)

        # Number of microbatches to capture. The value will be set in _get_cuda_graph_input_data().
        self.num_microbatches = None

        self._discover_layers()

        # Flags to track CUDA Graph state:
        # - _capture_finished: Whether create_cudagraphs() has been called (used by training loop)
        # - _graphs_created: Whether any graphs were actually created (may be False if no
        #   layers found)
        self._capture_finished = False
        self._graphs_created = False

    def _discover_layers(self):
        """Discover captureable layers from the model and populate internal data structures."""
        self.chunks_with_decoder = []
        self.num_layers_per_chunk = []
        self.callables_per_chunk = []
        self.callables_per_chunk_is_mtp = []
        self.flattened_callables = []
        self.flattened_callables_is_mtp = []
        pair_hybrid_layers = _pair_hybrid_layers_enabled(self.config)
        # Per-layer graphs leave Hybrid MTP blocks eager: such a block is a HybridStack, which is
        # not graphable as a whole. With pairing, its layers are graphed like decoder layers.
        graph_hybrid_mtp = (
            pair_hybrid_layers
            and bool(self.config.mtp_num_layers)
            and _hybrid_mtp_graphs_supported(self.config)
        )
        num_pairs, num_hybrid_mtp_callables = 0, 0
        # The settings that keep MoE layers of Hybrid MTP blocks eager, for one warning.
        mtp_moe_blocking_features = set()
        # The FP8/FP4 settings that keep Hybrid MTP blocks eager, for one warning.
        mtp_recipe_differences = set()
        for chunk_number, model_chunk in enumerate(self.model):
            try:
                chunk_with_decoder = get_attr_wrapped_model(
                    model_chunk, 'decoder', allow_none=False, return_model_obj=True
                )
                # unwrap the language model for vlm models, such as qwen3-vl.
                if hasattr(chunk_with_decoder, 'language_model'):
                    chunk_with_decoder = chunk_with_decoder.language_model
            except RuntimeError:
                num_graphable_layers = 0
                log_on_each_pipeline_stage(
                    logger=logger,
                    tp_group=self.tp_group,
                    dp_cp_group=self.dp_cp_group,
                    level=logging.DEBUG,
                    msg=f'Rank {torch.distributed.get_rank()}: '
                    f'No valid layer in model chunk {chunk_number}.',
                )
            else:
                num_graphable_layers = 0
                decoder_layers = chunk_with_decoder.decoder.layers
                mtp_layers = (
                    chunk_with_decoder.mtp.layers if hasattr(chunk_with_decoder, 'mtp') else []
                )
                num_decoder_layers, num_mtp_layers = len(decoder_layers), len(mtp_layers)
                hybrid_mtp_blocks = [
                    mtp_layer.mtp_model_layer
                    for mtp_layer in mtp_layers
                    if getattr(mtp_layer, "is_hybrid_mtp", False)
                ]
                for layers in (decoder_layers, *(block.layers for block in hybrid_mtp_blocks)):
                    for layer in layers:
                        # Drop pair links left by an earlier helper, so that layers graphed
                        # again with a different setting never replay a stale pair.
                        if getattr(layer, "_te_cuda_graph_pair_link", None) is not None:
                            layer._te_cuda_graph_pair_link = None
                callables, chunk_pairs = _te_graph_callables(
                    decoder_layers, self.config, pair_hybrid_layers
                )
                callables_is_mtp = [False] * len(callables)
                for mtp_layer in mtp_layers:
                    if _te_cuda_graph_layer_is_graphable(mtp_layer.mtp_model_layer, self.config):
                        mtp_callables = [mtp_layer.mtp_model_layer]
                    elif graph_hybrid_mtp and getattr(mtp_layer, "is_hybrid_mtp", False):
                        recipe_differences = _hybrid_mtp_recipe_differences(
                            mtp_layer.mtp_model_layer, self.config
                        )
                        if recipe_differences:
                            mtp_recipe_differences.update(recipe_differences)
                            continue
                        mtp_block_layers = mtp_layer.mtp_model_layer.layers
                        # An MoE layer is captured with its own config, which a
                        # hybrid_layer_config_list can set apart from the model config.
                        eager_moe_layers = []
                        for layer in mtp_block_layers:
                            blocking_features = (
                                _hybrid_mtp_moe_blocking_features(layer.config)
                                if getattr(layer, "is_moe_layer", False)
                                else []
                            )
                            if blocking_features:
                                eager_moe_layers.append(layer)
                                mtp_moe_blocking_features.update(blocking_features)
                        mtp_callables, mtp_pairs = _te_graph_callables(
                            mtp_block_layers,
                            self.config,
                            pair_hybrid_layers,
                            eager_layers=eager_moe_layers,
                        )
                        chunk_pairs += mtp_pairs
                        num_hybrid_mtp_callables += len(mtp_callables)
                    else:
                        continue
                    callables += mtp_callables
                    callables_is_mtp += [True] * len(mtp_callables)
                num_pairs += chunk_pairs
                num_graphable_layers = len(callables)
                log_on_each_pipeline_stage(
                    logger=logger,
                    tp_group=self.tp_group,
                    dp_cp_group=self.dp_cp_group,
                    level=logging.DEBUG,
                    msg=f'Rank {torch.distributed.get_rank()}: '
                    f'{num_decoder_layers} decoder layers and {num_mtp_layers} MTP layers in '
                    f'model chunk {chunk_number}. {num_graphable_layers} graphable layers.',
                )
            finally:
                if num_graphable_layers > 0:
                    self.chunks_with_decoder.append(chunk_with_decoder)
                    self.num_layers_per_chunk.append(num_graphable_layers)
                    self.callables_per_chunk.append(callables)
                    self.callables_per_chunk_is_mtp.append(callables_is_mtp)
                    self.flattened_callables.extend(callables)
                    self.flattened_callables_is_mtp.extend(callables_is_mtp)
                else:
                    self.chunks_with_decoder.append(None)
                    self.num_layers_per_chunk.append(0)
                    self.callables_per_chunk.append([])
                    self.callables_per_chunk_is_mtp.append([])

        # Logged per pipeline stage: only the stage that holds the MTP blocks has their layers.
        if mtp_moe_blocking_features:
            log_on_each_pipeline_stage(
                logger=logger,
                tp_group=self.tp_group,
                dp_cp_group=self.dp_cp_group,
                level=logging.WARNING,
                msg='MoE layers of Hybrid MTP blocks are not CUDA-graphed with '
                f'{", ".join(sorted(mtp_moe_blocking_features))}; they run eagerly.',
            )
        if mtp_recipe_differences:
            log_on_each_pipeline_stage(
                logger=logger,
                tp_group=self.tp_group,
                dp_cp_group=self.dp_cp_group,
                level=logging.WARNING,
                msg='Hybrid MTP layers are not CUDA-graphed because their layer configs differ '
                f'from the model in {", ".join(sorted(mtp_recipe_differences))}, and TE graphs '
                "would capture them with the model's FP8/FP4 recipe; they run eagerly.",
            )
        # _te_cuda_graph_pair_supported never pairs such a layer, so it is graphed on its own.
        if pair_hybrid_layers and any(
            getattr(c, "is_moe_layer", False) and c._cuda_graph_megakernel() is not None
            for c in self.flattened_callables
        ):
            log_on_each_pipeline_stage(
                logger=logger,
                tp_group=self.tp_group,
                dp_cp_group=self.dp_cp_group,
                level=logging.WARNING,
                msg='cuda_graph_pair_hybrid_layers does not pair MoE layers whose CUDA graph '
                'contains the MoE megakernel (moe_megakernel_backend); they keep their own CUDA '
                'graphs.',
            )
        log_on_each_pipeline_stage(
            logger=logger,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
            level=logging.INFO,
            msg=f'Rank {torch.distributed.get_rank()}: '
            f'{len(self.flattened_callables)} graphable layers.',
        )
        if pair_hybrid_layers:
            log_on_each_pipeline_stage(
                logger=logger,
                tp_group=self.tp_group,
                dp_cp_group=self.dp_cp_group,
                level=logging.INFO,
                msg=f'Rank {torch.distributed.get_rank()}: {num_pairs} Hybrid mixer and '
                'feed-forward layer pairs are graphed as one callable each, and '
                f'{num_hybrid_mtp_callables} graphed callables belong to Hybrid MTP blocks.',
            )

    def capture_finished(self):
        """
        Returns whether create_cudagraphs() has been called.

        This is used by the training loop to determine if the capture process has run.
        Returns True after create_cudagraphs() completes, regardless of whether any
        graphs were actually created.
        """
        return self._capture_finished

    def graphs_created(self):
        """
        Returns whether any CUDA Graphs were actually created.

        This returns False if create_cudagraphs() was called but no graphable layers
        were found, and True if at least one graph was successfully created.
        """
        return self._graphs_created

    def _get_sample_arguments(self, order, chunk_id_list=None):
        """
        Generate sample arguments and keyword arguments for CUDA Graph capturing with
        memory-optimized buffer reuse.

        This method creates static input tensors for each (layer, microbatch) pair needed
        by TE's make_graphed_callables(). It optimizes memory usage by reusing input buffers
        across non-overlapping forward passes based on the pipeline parallel schedule.
        This optimization is essential for reducing peak memory during CUDA Graph capturing with
        many microbatches, as it allows buffers to be reused instead of allocating new ones for
        later microbatches.

        Memory Optimization Strategy:
            The 1F1B (one-forward-one-backward) interleaved schedule in pipeline parallelism
            means that once a microbatch's backward pass completes, its input buffers are no
            longer needed. This method tracks buffer lifecycle and reuses "consumed" buffers
            (those whose backward has completed) for new forward passes with matching tensor
            signatures (shape, dtype, layout).

            Example schedule: [1, 1, 1, 2, 2, 2, -2, 1, -2, 1, -2, 2, -1, 2, -1, -1, -2, -2, -1, -1]
            - Positive values indicate forward passes (chunk_id = value)
            - Negative values indicate backward passes (chunk_id = -value)
            - When processing -2 (backward of chunk 2), its buffers become available for reuse
            - The next forward with matching signature can reuse those buffers

        Args:
            order (List[int]): The forward/backward execution order from
                convert_te_cuda_graph_schedule_table_to_order(). Positive integers represent
                forward passes (1-indexed chunk ID), negative integers represent backward passes.
            chunk_id_list (List[Tuple[int, int]]): The list of chunk IDs and layer IDs in the
                order. This is useful only when overlap_moe_expert_parallel_comm is enabled,
                the order maps each layers' idx to their original chunk id.

        Returns:
            Tuple[List[Tuple], List[Dict]]: A tuple containing:
                - sample_args: List of positional argument tuples for each (layer, microbatch).
                    Length = num_layers * num_microbatches. Elements with the same tensor
                    signature may share references to reduce memory allocation.
                - sample_kwargs: List of keyword argument dicts for each (layer, microbatch).
                    Length = num_layers * num_microbatches. Elements with the same tensor
                    signature may share references to reduce memory allocation.

        Data Structures:
            - fwd_sample_queues: Dict[chunk_id, List[Tuple[sample_keys, fwd_idx]]]
                Queue of forward samples per chunk awaiting their backward pass.
            - consumed_sample_queue: Dict[sample_keys, List[fwd_idx]]
                Pool of buffer indices whose backward is complete, keyed by tensor signature.
            - sample_keys: Tuple of (shape, dtype, layout) for args + (key, shape, dtype, layout)
                for kwargs, used to match compatible buffers for reuse.
        """
        assert self.num_model_chunks == max(
            order
        ), "num_model_chunks must match the max chunk id in order."
        if chunk_id_list is None:
            # check only if 1f1b overlap is disabled.
            assert (
                self.num_microbatches == len(order) // self.num_model_chunks // 2
            ), "num_microbatches must match the number of microbatches in order."

        # Generate sample arguments and keyword arguments for capturing.
        sample_args = [None] * (len(self.flattened_callables) * self.num_microbatches)
        sample_kwargs = [None] * (len(self.flattened_callables) * self.num_microbatches)

        rotary_pos_emb_cache = {}

        def _get_layer_static_inputs(layer, chunk_of_the_layer):
            """
            Get the static inputs for a layer.
            """
            graphed_callable = layer
            if isinstance(layer, _TECudaGraphLayerPair):
                # A pair's inputs are its mixer's plus the partner's extras, so check the chunk
                # and add the rotary input for the mixer.
                layer = layer.mixer
            assert layer in chunk_of_the_layer.decoder.layers or _is_mtp_te_graph_layer(
                layer, chunk_of_the_layer
            ), "Layer is not in the chunk"

            def get_rotary_pos_emb(transformer_module, transformer_input):
                uses_mla = self.config.multi_latent_attention
                if getattr(transformer_module, "hybrid_layer_config_list", None) is not None:
                    # Config types select the layer implementation; copied flags may differ.
                    config_type = type(layer.config)
                    if config_type is AttentionLayerConfig:
                        uses_mla = False
                    elif config_type in (MLALayerConfig, DSALayerConfig, CSALayerConfig):
                        uses_mla = True
                if transformer_module.position_embedding_type == 'rope' and not uses_mla:
                    rotary_seq_len = transformer_module.rotary_pos_emb.get_rotary_seq_len(
                        None, transformer_module.decoder, transformer_input, self.config, None
                    )
                    if rotary_seq_len not in rotary_pos_emb_cache:
                        rotary_pos_emb_cache[rotary_seq_len] = transformer_module.rotary_pos_emb(
                            rotary_seq_len,
                            packed_seq=getattr(self.config, "sequence_packing_scheduler", None)
                            is not None,
                        )
                    return rotary_pos_emb_cache[rotary_seq_len]
                else:
                    return None

            static_inputs = graphed_callable.get_layer_static_inputs(
                self.seq_length, self.micro_batch_size
            )

            from megatron.core.transformer.identity_op import IdentityOp
            from megatron.core.transformer.transformer_layer import TransformerLayer

            contains_self_attn = (
                isinstance(layer, TransformerLayer)
                and not isinstance(layer.self_attention, IdentityOp)
                and (
                    not self.config.cuda_graph_modules
                    or CudaGraphModule.attn in self.config.cuda_graph_modules
                )
            )

            _sample_kwargs = {}
            if is_te_min_version("1.10.0"):
                # te.make_graphed_callables() accepts keyword arguments since 1.10.0.
                hidden_states = static_inputs.pop("hidden_states")
                _sample_args = (hidden_states,)
                # A layer may provide its own rotary input (e.g. batch-dependent mRoPE in VLMs that
                # report "rope" for the language model); keep it instead of a static embedding.
                if contains_self_attn and "rotary_pos_emb" not in static_inputs:
                    rotary_pos_emb = get_rotary_pos_emb(chunk_of_the_layer, hidden_states)
                    if rotary_pos_emb is not None:
                        static_inputs["rotary_pos_emb"] = rotary_pos_emb
                _sample_kwargs = static_inputs
            elif contains_self_attn:
                _sample_args = (
                    static_inputs.pop("hidden_states"),
                    static_inputs.pop("attention_mask"),
                )
            else:
                _sample_args = (static_inputs.pop("hidden_states"),)
            return _sample_args, _sample_kwargs

        # Calculate the starting index of each chunk in callables for future use.
        prefix_num_layers = [0]
        for model_chunk_idx in range(self.num_model_chunks):
            num_layers = self.num_layers_per_chunk[model_chunk_idx]
            prefix_num_layers.append(prefix_num_layers[-1] + num_layers)

        # Reorganize args and kwargs for input tensor reuse.
        # fwd_sample_queues is keyed by model chunk index. The value is a queue of tuples.
        # Each tuple contains the sample key signature and its fwd_idx. When we finish a backward
        # chunk, we pop the corresponding fwd_idx and push to the consumed_sample_queue.
        # consumed_sample_queue is keyed by the sample key signature. The value is a queue of the
        # fwd_idx whose backward has been called so that we can reuse the same static buffers.
        # In this way, we can reuse the same static input buffers for the non-overlapping samples
        # with the same input signature.
        fwd_sample_queues = {}
        consumed_sample_queue = {}
        layer_sample_keys_cache = {}
        fwd_idx = [0] * self.num_model_chunks
        for idx, chunk_id in enumerate(order):
            model_chunk_idx = abs(ceil(chunk_id)) - 1

            if chunk_id > 0:
                if model_chunk_idx not in fwd_sample_queues:
                    fwd_sample_queues[model_chunk_idx] = []

                sample_start_idx = (prefix_num_layers[model_chunk_idx] * self.num_microbatches) + (
                    fwd_idx[model_chunk_idx] * self.num_layers_per_chunk[model_chunk_idx]
                )
                if chunk_id_list:
                    model_chunk_idx = chunk_id_list[idx][0]
                    callables_curr_chunk = [
                        self.callables_per_chunk[model_chunk_idx][chunk_id_list[idx][1]]
                    ]
                else:
                    callables_curr_chunk = self.callables_per_chunk[model_chunk_idx]
                for layer_idx, layer in enumerate(callables_curr_chunk):
                    per_callable_fwd_idx = sample_start_idx + layer_idx

                    # Get sample_args and sample_kwargs for index per_callable_fwd_idx.
                    assert (
                        sample_args[per_callable_fwd_idx] is None
                        and sample_kwargs[per_callable_fwd_idx] is None
                    ), (
                        f"sample_args and sample_kwargs must be None before assigning static data, "
                        f"but got sample_args[{per_callable_fwd_idx}] = "
                        f"{sample_args[per_callable_fwd_idx]} and "
                        f"sample_kwargs[{per_callable_fwd_idx}] = "
                        f"{sample_kwargs[per_callable_fwd_idx]}."
                    )
                    if id(layer) not in layer_sample_keys_cache:
                        # Have not generated the static inputs for this layer yet. So we don't
                        # know the input signature of this layer. Generate the static inputs, and
                        # cache the signature.
                        sample_args[per_callable_fwd_idx], sample_kwargs[per_callable_fwd_idx] = (
                            _get_layer_static_inputs(
                                layer, self.chunks_with_decoder[model_chunk_idx]
                            )
                        )
                        sample_args_keys = tuple(
                            (t.shape, t.dtype, t.layout) for t in sample_args[per_callable_fwd_idx]
                        )
                        sample_kwargs_keys = tuple(
                            (k, v.shape, v.dtype, v.layout)
                            for k, v in sorted(sample_kwargs[per_callable_fwd_idx].items())
                        )
                        sample_keys = sample_args_keys + sample_kwargs_keys
                        layer_sample_keys_cache[id(layer)] = sample_keys
                    else:
                        # Get signature from cache. This signature will be used to see if we can
                        # reuse the static inputs of a previous forward pass for this forward pass.
                        # If not, we still need to generate the new static inputs.
                        sample_keys = layer_sample_keys_cache[id(layer)]
                    model_chunk_idx = abs(chunk_id) - 1
                    fwd_sample_queues[model_chunk_idx].append((sample_keys, per_callable_fwd_idx))
                    if consumed_sample_queue.get(sample_keys, []):
                        # We can reuse the static inputs of a previous forward pass for this
                        # forward pass, because they are of the same input signature and the
                        # backward pass of the previous forward pass has completed.
                        reuse_fwd_idx = consumed_sample_queue[sample_keys].pop(0)
                        assert (
                            sample_args[reuse_fwd_idx] is not None
                            and sample_kwargs[reuse_fwd_idx] is not None
                        ), (
                            f"sample_args and sample_kwargs must not be None when reusing, but got "
                            f"sample_args[{reuse_fwd_idx}] = {sample_args[reuse_fwd_idx]} and "
                            f"sample_kwargs[{reuse_fwd_idx}] = {sample_kwargs[reuse_fwd_idx]}.",
                        )
                        sample_args[per_callable_fwd_idx] = sample_args[reuse_fwd_idx]
                        sample_kwargs[per_callable_fwd_idx] = sample_kwargs[reuse_fwd_idx]

                    if sample_args[per_callable_fwd_idx] is None:
                        # Unfortunately, no previous static inputs are available for reuse,
                        # sample_args is still None. Last attempt: generate the new static inputs
                        # for this forward pass.
                        if chunk_id_list:
                            model_chunk_idx = chunk_id_list[idx][0]
                        sample_args[per_callable_fwd_idx], sample_kwargs[per_callable_fwd_idx] = (
                            _get_layer_static_inputs(
                                layer, self.chunks_with_decoder[model_chunk_idx]
                            )
                        )
                        model_chunk_idx = abs(chunk_id) - 1
                fwd_idx[model_chunk_idx] += 1
            elif ceil(chunk_id) == chunk_id:
                num_consumed_samples = min(
                    len(fwd_sample_queues[model_chunk_idx]),
                    self.num_layers_per_chunk[model_chunk_idx],
                )
                for sample_keys, per_callable_fwd_idx in fwd_sample_queues[model_chunk_idx][
                    :num_consumed_samples
                ]:
                    if sample_keys not in consumed_sample_queue:
                        consumed_sample_queue[sample_keys] = []
                    consumed_sample_queue[sample_keys].append(per_callable_fwd_idx)
                fwd_sample_queues[model_chunk_idx] = fwd_sample_queues[model_chunk_idx][
                    num_consumed_samples:
                ]
            else:
                # skip register static inputs for wgrad backward graphs
                continue

        return sample_args, sample_kwargs

    def _get_amax_reduction_group(self, with_context_parallel=False, tp_only_amax_red=False):
        """Get the FP8 amax reduction group the caller rank belongs to."""
        if with_context_parallel:
            if not tp_only_amax_red:
                assert self.pg_collection.tp_dp_cp is not None
                return self.pg_collection.tp_dp_cp
            else:
                assert self.pg_collection.tp_cp is not None
                return self.pg_collection.tp_cp
        else:
            if not tp_only_amax_red:
                assert self.pg_collection.tp_dp is not None
                return self.pg_collection.tp_dp
            else:
                assert self.pg_collection.tp is not None
                return self.pg_collection.tp

    def _get_cuda_graph_input_data(self):
        """
        Create the CUDA Graph capturing input data.
        The data is organized per-chunk per-microbatch per-layer.
        """

        # Get the PP and VPP scheduling order.
        from megatron.core.pipeline_parallel.schedules import (
            get_pp_rank_microbatches,
            get_schedule_table,
        )

        # If PP is not enabled, we only need to capture one microbatch.
        if self.pp_group.size() == 1 and not self.config.overlap_moe_expert_parallel_comm:
            assert (
                self.num_model_chunks == 1
            ), "If PP is not enabled, there should be only one model chunk."
            self.num_microbatches = 1
        else:
            # Keep the source-batch bound when dp_balanced produces fewer packs.
            # With a fixed source batch and MBS=1, each aligned packed schedule is
            # a subsequence of this capture order, preserving the existing input
            # buffer lifetimes without wrapping around the captured graph list.
            self.num_microbatches = get_num_microbatches()

        _, _, num_warmup_microbatches, _ = get_pp_rank_microbatches(
            self.num_microbatches,
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
            forward_only=False,
            p2p_communicator=self.p2p_communicator,
        )
        schedule_table = get_schedule_table(
            self.num_microbatches,
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
        )
        order = convert_te_cuda_graph_schedule_table_to_order(
            num_warmup_microbatches, self.num_model_chunks, schedule_table
        )
        log_on_each_pipeline_stage(
            logger=logger,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
            level=logging.DEBUG,
            msg=f'Rank {torch.distributed.get_rank()}: ORDER {order}',
        )
        chunk_id_list = None
        if self.config.overlap_moe_expert_parallel_comm:
            wgrad_in_graph_scope = CudaGraphModule.attn in self.config.cuda_graph_modules or (
                CudaGraphModule.moe_router in self.config.cuda_graph_modules
                and self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            )
            capture_wgrad_graph = self.config.delay_wgrad_compute and wgrad_in_graph_scope
            order, chunk_id_list = get_te_cuda_graph_overlap_moe_expert_parallel_comm_order(
                order, self.num_layers_per_chunk, capture_wgrad_graph
            )
            self.num_layers_per_chunk = [1] * sum(self.num_layers_per_chunk)
            self.num_model_chunks = max(order)
            _order_without_wgrad = []
            for c_id in order:
                if ceil(c_id) != c_id:
                    continue
                _order_without_wgrad.append(c_id)
            self.num_microbatches = len(_order_without_wgrad) // self.num_model_chunks // 2
            log_on_each_pipeline_stage(
                logger=logger,
                tp_group=self.tp_group,
                dp_cp_group=self.dp_cp_group,
                level=logging.DEBUG,
                msg=f'Rank {torch.distributed.get_rank()}: '
                f'ORDER after overlap_moe_expert_parallel_comm {order}',
            )

        # Generate sample arguments and keyword arguments for capturing.
        sample_args, sample_kwargs = self._get_sample_arguments(order, chunk_id_list)

        def get_make_graphed_callables_kwargs():
            kwargs = {
                'allow_unused_input': True,
                '_order': order,
                'retain_graph_in_backward': self.config.cuda_graph_retain_backward_graph,
            }

            # Calculate the number of warmup iterations per layer per microbatch inside TE
            # make_graphed_callables(). There are two rules:
            # 1. There should be at least 1 warmup iteration per layer per microbatch inside TE
            # make_graphed_callables().
            # 2. There should be at least 10 warmup iterations per layer, counting the MCore warmup
            # steps before going into this capture routine.
            kwargs['num_warmup_iters'] = max(
                1,
                math.ceil(
                    (10 - self.config.cuda_graph_warmup_steps * get_num_microbatches())
                    / self.num_microbatches
                ),
            )

            if is_te_min_version("2.6.0"):
                # Starting from TE 2.6.0, make_graphed_callables() accepts different number
                # of layers per chunk.
                kwargs['_num_layers_per_chunk'] = self.num_layers_per_chunk
            if is_te_min_version("2.7.0"):
                # Starting from TE 2.7.0, make_graphed_callables() optimizes the graph memory usage
                # by reusing input/output data buffers between graphs.
                kwargs['_reuse_graph_input_output_buffers'] = True
            if is_te_min_version("2.19.0"):
                # MCore consumes parameter gradients through its accumulation hooks during each
                # replay, before TE can overwrite the static buffers on a later replay.
                kwargs['clone_param_grads_on_return'] = False

            if sample_kwargs:
                kwargs['sample_kwargs'] = sample_kwargs

            from megatron.core.fp4_utils import get_fp4_recipe
            from megatron.core.fp8_utils import get_fp8_recipe

            if self.config.fp8 or self.config.fp4:
                # FP4 and FP8 are mutually exclusive, so use fp8_* kwargs for FP4 too
                # since TE currently uses fp8_autocast for both FP8 and FP4 quantization

                def _get_fp8_enabled():
                    if is_te_min_version("2.8.0"):
                        from megatron.core.fp8_utils import is_first_last_bf16_layer

                        fp8_enabled = []
                        for callable, is_mtp in zip(
                            self.flattened_callables, self.flattened_callables_is_mtp
                        ):
                            fp8_enabled.append(
                                not is_first_last_bf16_layer(
                                    self.config, callable.layer_number - 1 if not is_mtp else -1
                                )
                            )
                        return tuple(fp8_enabled)
                    else:
                        return True

                kwargs['fp8_enabled'] = _get_fp8_enabled()
                kwargs['fp8_recipe'] = (
                    get_fp8_recipe(self.config) if self.config.fp8 else get_fp4_recipe(self.config)
                )
                kwargs['fp8_weight_caching'] = True
                if (
                    is_te_min_version("1.14.0")
                    and self.pg_collection is not None
                    and self.pg_collection.tp is not None
                ):
                    kwargs['fp8_group'] = self._get_amax_reduction_group(
                        with_context_parallel=True, tp_only_amax_red=self.config.tp_only_amax_red
                    )
            else:
                kwargs['fp8_enabled'] = False

            from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
                FineGrainedActivationOffloadingInterface as off_interface,
            )

            # TE CUDA graph warmup should establish graph state without launching
            # activation D2H copies; the post-warmup hook restores offloading for
            # the measured/replay iterations.
            if self.config.fine_grained_activation_offloading:
                kwargs['pre_warmup_hook'] = off_interface.disable_offload
                kwargs['post_warmup_hook'] = off_interface.enable_offload
            return kwargs

        kwargs = get_make_graphed_callables_kwargs()
        return sample_args, kwargs

    def _start_capturing(self):
        """
        Start capturing CUDA Graphs.
        """
        assert not self._capture_finished, "CUDA Graph capture has already been finished."

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        _set_capture_start()
        log_single_rank(logger, logging.INFO, f'Start CUDA Graphs capture...')
        return time.time()

    def _reset_after_capture(self):
        """
        Reset the model and optimizer state after capturing CUDA Graphs.
        """
        from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
        from megatron.core.transformer.moe.moe_logging import get_moe_metrics_tracker

        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        get_moe_metrics_tracker().clear()
        reset_model_temporary_tensors(self.config, self.model)

    def _finish_capturing(self, start_time):
        """
        Finish capturing CUDA Graphs and clean up the related state.
        """
        log_single_rank(
            logger,
            logging.INFO,
            f'Time spent in CUDA Graphs capture on rank {torch.distributed.get_rank()}: '
            f'{time.time() - start_time}s',
        )
        _set_capture_end()

        from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
            FineGrainedActivationOffloadingInterface as off_interface,
        )

        if self.config.fine_grained_activation_offloading:
            off_interface.reset()
        torch.cuda.synchronize()
        self._reset_after_capture()
        gc.collect()
        torch.cuda.empty_cache()

        self._capture_finished = True

    def create_cudagraphs(self):
        """
        Capture CUDA Graphs per TransformerLayer per microbatch.
        """
        from transformer_engine.pytorch.graph import make_graphed_callables

        start_time = self._start_capturing()

        if not self.flattened_callables:
            # Check if there are any graphable layers. If not, log a warning and skip capture,
            # but still call _finish_capturing to ensure all ranks complete the capture phase.
            logger.warning(
                'TECudaGraphHelper: No graphable layers found. Skipping CUDA graph capture.'
            )
        else:
            # Prepare CUDA Graph capturing input data and call `make_graphed_callables`.
            sample_args, kwargs = self._get_cuda_graph_input_data()
            if self.config.sequence_parallel:
                rng_context = get_cuda_rng_tracker().fork()
            else:
                rng_context = nullcontext()
            with (
                rng_context,
                capture_constants(self),
                _keep_class_calls(self.flattened_callables),
                no_gc_during_capture(freeze=True),
            ):
                graphs = make_graphed_callables(
                    tuple(self.flattened_callables), sample_args, **kwargs
                )

            # The returned callables can outlive this helper. Share the capture's
            # constants with those actual graph owners instead of retaining them here.
            constants = _CudagraphGlobalRecord.capture_constants.pop(self)
            for graph in graphs:
                _CudagraphGlobalRecord.capture_constants[graph] = constants

            # Push the captured graphs to the corresponding TransformerBlock.
            num_layers_accumulated = 0
            for layers in self.callables_per_chunk:
                for layer_number, layer in enumerate(layers):
                    layer.te_cuda_graphs = []
                    for batch_number in range(self.num_microbatches):
                        if self.config.overlap_moe_expert_parallel_comm:
                            graph_idx = (
                                num_layers_accumulated + layer_number
                            ) * self.num_microbatches + batch_number
                        else:
                            graph_idx = (
                                num_layers_accumulated * self.num_microbatches
                                + batch_number * len(layers)
                                + layer_number
                            )
                        layer.te_cuda_graphs.append(graphs[graph_idx])
                num_layers_accumulated += len(layers)

            self._graphs_created = True

        self._finish_capturing(start_time)

    def cuda_graph_set_manual_hooks(self):
        """
        Set CUDA Graph manual hooks for the modules that contain direct parameters and
        are covered by cudagraphs.
        """
        for chunk_number, layers in enumerate(self.callables_per_chunk):
            model_chunk = self.model[chunk_number]
            for layer in layers:
                layer.setup_manual_hooks(model_chunk._make_forward_pre_hook)

    def delete_cuda_graphs(self):
        """
        Delete all CUDA graphs.
        """
        assert self._graphs_created, "No CUDA Graphs were created to delete."

        torch.cuda.synchronize()

        graph_resettable = is_te_min_version("2.10.0")
        graphs_reset, graphs_not_reset = 0, 0
        for layers in self.callables_per_chunk:
            for layer in layers:
                for graph in layer.te_cuda_graphs:
                    if graph_resettable:
                        graph.reset()
                        release_capture_constants(graph)
                        graphs_reset += 1
                    else:
                        graphs_not_reset += 1
                layer.te_cuda_graphs = []
                layer.te_cuda_graph_manual_hooks = []
                layer._after_te_cuda_graphs_deleted()

        log_on_each_pipeline_stage(
            logger=logger,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
            level=logging.INFO,
            msg=f'Rank {torch.distributed.get_rank()}: '
            f'{graphs_reset} graphs deleted with explicit reset, '
            f'{graphs_not_reset} graphs deleted without explicit reset.',
        )
        self._graphs_created = False


def convert_te_cuda_graph_schedule_table_to_order(
    num_warmup_microbatches, num_model_chunks, schedule_table
):
    """Convert a tunable schedule lookup table to the te.make_graphed_callables() accepted
    order format. For example, the tunable schedule table for PP2 N3M5 with VP2 is as below:
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1

    Then the forward backward separated order is:
    forward               | 1 1 1 2 2 2 1 1 2 2
    backward              | -2 -2 -2 -1 -1 -1 -2 -2 -1 -1

    If num_warmup_microbatches is 5, the output order is:
    1 1 1 2 2 2 -2 1 -2 1 -2 2 -1 2 -1 -1 -2 -2 -1 -1
    """
    _, model_chunk_id_table = zip(*schedule_table)
    forward_order = [chunk_id + 1 for chunk_id in model_chunk_id_table]
    backward_order = [chunk_id - num_model_chunks for chunk_id in model_chunk_id_table]
    order = forward_order[:num_warmup_microbatches]
    for i in range(num_warmup_microbatches, len(forward_order)):
        order.append(forward_order[i])
        order.append(backward_order[i - num_warmup_microbatches])
    if num_warmup_microbatches > 0:
        order.extend(backward_order[-num_warmup_microbatches:])
    return order


def get_te_cuda_graph_overlap_moe_expert_parallel_comm_order(
    order, num_layers_per_chunk, capture_wgrad_graph
):
    """
    This functions gets the order for overlap_moe_expert_parallel_comm schedule for the original
    chunk-wise order list. Each chunk is transformered to chunks with only 1 layer so that
    layers between 2 chunks can now overlap with each other while following the graph order.
    If capture_wgrad_graph is True, the wgrad backward graph is also added to the order by
    decreasing the layer id by 0.5.

    Args:
        order (List[int]): The original chunk-wise order list. Positive values represent forward
            passes for chunks, negative values represent backward passes. The absolute value
            indicates the chunk ID (1-indexed).
        num_layers_per_chunk (List[int]): Number of graphable layers in each chunk. The length
            of this list equals the number of chunks.
        capture_wgrad_graph (bool): If True, weight gradient computation graphs are added to the
            order by appending entries with layer_id - 0.5.

    Returns:
        Tuple[List[float], List[Optional[List[int]]]]: A tuple containing:
            - new_order: The layer-wise order list where each chunk is expanded to individual
              layers. Positive values are forward passes, negative values are backward passes.
              Values with .5 suffix indicate weight gradient computations.
            - chunk_id_list: A list parallel to new_order. For forward passes, contains
              [chunk_id, layer_index_within_chunk]. For backward passes, contains None.

    Example:
        original_order: [1, 2, -2, 1, -1, -1]
        num_layers_per_chunk: [1, 2]
        capture_wgrad_graph=True:
            new_order: [1, 2, 3, 1, -3, -3.5, -2, -2.5, -1, -1.5, -1, -1.5]
            chunk_id_list: [[0, 0], [1, 0], [1, 1], [0, 0], None,
                            None, None, None, None, None, None, None]
        capture_wgrad_graph=False:
            new_order: [1, 2, 3, 1, -3, -2, -1, -1]
            chunk_id_list: [[0, 0], [1, 0], [1, 1], [0, 0], None, None, None, None]
    """

    def _add_order(new_order, chunk_id_list, c_id, layer_id, is_wgrad=False, index=None):
        if is_wgrad:
            new_order.append(layer_id - 0.5)
        else:
            new_order.append(layer_id)
        if c_id > 0:
            chunk_id_list.append([abs(c_id) - 1, index])
        else:
            chunk_id_list.append(None)

    new_order = []
    chunk_id_list = []
    add_order = partial(_add_order, new_order, chunk_id_list)
    first_backward_idx, last_forward_idx = None, None
    for idx, c_id in enumerate(order):
        if first_backward_idx is None and c_id < 0:
            first_backward_idx = idx
        if c_id > 0:
            last_forward_idx = idx

    def get_layer_range(c_id):
        num_layers = num_layers_per_chunk[abs(c_id) - 1]
        num_layers_previous_chunks = sum(num_layers_per_chunk[: abs(c_id) - 1])
        if c_id > 0:
            return list(
                range(num_layers_previous_chunks + 1, num_layers_previous_chunks + num_layers + 1)
            )
        return list(range(-num_layers_previous_chunks - num_layers, -num_layers_previous_chunks))

    # warmup stage
    for c_id in order[:first_backward_idx]:
        layer_range = get_layer_range(c_id)
        new_order += layer_range
        chunk_id_list.extend([abs(c_id) - 1, i] for i in range(len(layer_range)))

    # 1f1b overlap stage
    if first_backward_idx < last_forward_idx:
        for c_id_b, c_id_f in zip(
            order[first_backward_idx : last_forward_idx + 1 : 2],
            order[first_backward_idx + 1 : last_forward_idx + 1 : 2],
        ):
            layer_range_f = get_layer_range(c_id_f)
            layer_range_b = get_layer_range(c_id_b)
            index = 0
            for l_b, l_f in zip_longest(layer_range_b, layer_range_f, fillvalue=0):
                # always forward graph before backward graph
                if l_f != 0:
                    add_order(c_id_f, l_f, index=index)
                if l_b != 0:
                    add_order(c_id_b, l_b)
                    if capture_wgrad_graph and index < len(layer_range_b) - 1:
                        add_order(c_id_b, l_b, is_wgrad=True)
                index += 1
            # last wgrad backward
            if capture_wgrad_graph and layer_range_b:
                add_order(c_id_b, layer_range_b[-1], is_wgrad=True)

    # cool down stage, backward graphs only
    for c_id in order[last_forward_idx + 1 :]:
        for l_b in get_layer_range(c_id):
            add_order(c_id, l_b)
            if capture_wgrad_graph:
                add_order(c_id, l_b, is_wgrad=True)

    return new_order, chunk_id_list


# ---------------------------------------------------------------------------
# Vision encoder CUDA graph helpers
# ---------------------------------------------------------------------------


def _wrap_te_cuda_graph_for_vision(graph_fn):
    """Wrap a graphed callable to filter out None outputs.

    During make_graphed_callables warmup, vision encoder layers go through their
    normal forward() path which returns (output, context=None). _te_cuda_graph_replay
    asserts len(output) == 1 but gets 2 elements. This wrapper filters out None
    values so replay sees (output,) instead of (output, None).
    """

    def wrapped(*args, **kwargs):
        result = graph_fn(*args, **kwargs)
        if isinstance(result, tuple):
            filtered = tuple(r for r in result if r is not None)
            return filtered if filtered else result
        return result

    for attr in ('backward_dw', 'reset'):
        if hasattr(graph_fn, attr):
            setattr(wrapped, attr, getattr(graph_fn, attr))
    return wrapped


def get_vision_te_cuda_graph_seq_length(vision_config, default_seq_length: int = 4096) -> int:
    """Calculate the sequence length for vision encoder CUDA graphs.

    For vision encoders, the sequence length depends on:
    - max_vision_cuda_graph_seq_length: explicit maximum (if set)
    - num_position_embeddings: maximum number of patches
    - spatial_merge_size: pooling factor that reduces sequence length

    Args:
        vision_config: The TransformerConfig for vision encoder
        default_seq_length: Default sequence length if cannot be calculated

    Returns:
        The sequence length to use for CUDA graph capture
    """
    if (
        hasattr(vision_config, 'max_vision_cuda_graph_seq_length')
        and vision_config.max_vision_cuda_graph_seq_length
    ):
        return vision_config.max_vision_cuda_graph_seq_length

    if hasattr(vision_config, 'num_position_embeddings'):
        seq_length = vision_config.num_position_embeddings
        if hasattr(vision_config, 'spatial_merge_size'):
            merge_factor = vision_config.spatial_merge_size**2
            seq_length = seq_length // merge_factor
        return seq_length

    return default_seq_length


class VisionTECudaGraphHelper(TECudaGraphHelper):
    """Helper to capture CUDA Graphs for vision encoder layers using TE.

    Inherits from TECudaGraphHelper and overrides only the
    vision-specific behaviour:

    * Layer discovery finds vision_model.decoder.layers instead of the
      language decoder layers.
    * num_model_chunks is always 1 (vision has no virtual pipeline stages).
    * Batch dimension is always 1 (images are concatenated along the sequence
      dimension).
    * Sample argument generation uses a simple loop (no rotary embeddings or
      buffer-reuse optimization).
    * _finish_capturing wraps captured graphs to filter None values that arise
      from vision encoder layers returning (output, None), and skips cleanup
      that is handled by the LM decoder helper.

    Note:
        With pipeline parallelism > 1, only the first pipeline stage typically
        has vision layers. Ranks without vision layers can safely skip calling
        create_cudagraphs() or will gracefully return with no graphs created.

    Args:
        model: The full model (list of model chunks) containing vision_model.
        vision_config: TransformerConfig for the vision encoder.
        vision_seq_length: Sequence length for vision (max vision tokens).
        micro_batch_size: Micro-batch size (unused for sample-arg generation
            since the vision encoder always uses batch-dim = 1).
        num_microbatches: Number of microbatches per step.
    """

    def __init__(
        self,
        model,
        vision_config,
        vision_seq_length: int,
        micro_batch_size: int,
        num_microbatches: int = 1,
        pg_collection=None,
    ):
        super().__init__(
            model, vision_config, vision_seq_length, micro_batch_size, pg_collection=pg_collection
        )
        # Vision encoder concatenates all images along the sequence dimension
        # with a fixed batch dimension of 1, regardless of the training MBS.
        self.micro_batch_size = 1
        self.num_model_chunks = 1
        self.num_microbatches = num_microbatches

    def _discover_layers(self):
        """Discover captureable layers from the vision encoder."""
        self.vision_model = None
        vision_layers = []

        for model_chunk in self.model:
            try:
                unwrapped = get_attr_wrapped_model(
                    model_chunk, 'vision_model', allow_none=True, return_model_obj=True
                )
                if unwrapped is not None and hasattr(unwrapped, 'vision_model'):
                    self.vision_model = unwrapped.vision_model
                    break
            except (RuntimeError, AttributeError):
                continue

        if self.vision_model is not None:
            if hasattr(self.vision_model, 'decoder') and hasattr(
                self.vision_model.decoder, 'layers'
            ):
                for layer in self.vision_model.decoder.layers:
                    if _te_cuda_graph_layer_is_graphable(layer, self.config):
                        vision_layers.append(layer)

        if vision_layers:
            self.chunks_with_decoder = [self.vision_model]
            self.num_layers_per_chunk = [len(vision_layers)]
            self.callables_per_chunk = [vision_layers]
            self.callables_per_chunk_is_mtp = [[False] * len(vision_layers)]
            self.flattened_callables = list(vision_layers)
            self.flattened_callables_is_mtp = [False] * len(vision_layers)
        else:
            if self.vision_model is None:
                logger.warning(
                    'VisionTECudaGraphHelper: No vision_model found in model. '
                    'CUDA graphs will not be captured for vision encoder.'
                )
            self.chunks_with_decoder = [None]
            self.num_layers_per_chunk = [0]
            self.callables_per_chunk = [[]]
            self.callables_per_chunk_is_mtp = [[]]
            self.flattened_callables = []
            self.flattened_callables_is_mtp = []

        # backward-compat aliases used by callers / tests
        self.callables = vision_layers
        self.num_layers = len(vision_layers)

        if vision_layers:
            logger.info(
                f'VisionTECudaGraphHelper: Found {self.num_layers} graphable vision encoder '
                f'layers. seq_length={self.seq_length} (all images concatenated, batch_dim=1)'
            )

    def _reset_after_capture(self):
        """
        No-op: vision encoder layers do not require any reset:
        - model_chunk.zero_grad_buffer() / optimizer.zero_grad() (handled
          by the LM decoder helper's _finish_capturing which runs on all ranks).
        - clear_aux_losses_tracker / reset_model_temporary_tensors
          (LM-specific cleanup already handled by the LM helper).
        """

    def _finish_capturing(self, start_time):
        """
        Before calling super()._finish_capturing, wrap the captured graphs with
        _wrap_te_cuda_graph_for_vision to filter None from (output, None) tuples so that
        _te_cuda_graph_replay's len == 1 assertion passes.
        """
        # Wrap the captured graphs before finishing
        for layer in self.flattened_callables:
            if hasattr(layer, 'te_cuda_graphs'):
                layer.te_cuda_graphs = [
                    _wrap_te_cuda_graph_for_vision(g) for g in layer.te_cuda_graphs
                ]

        super()._finish_capturing(start_time)

    def _get_sample_arguments(self, order, chunk_id_list=None):
        """Generate sample arguments for vision encoder CUDA Graph capturing.

        Vision uses a simple per-layer-per-microbatch loop with batch_dim=1
        and no rotary embeddings (unlike the parent's buffer-reuse
        optimization). The order and chunk_id_list arguments are
        unused because vision has num_model_chunks=1 and does not need
        the pipeline-schedule-aware buffer lifecycle tracking.

        Returns:
            Tuple of (sample_args, sample_kwargs) lists for each
            (layer, microbatch) pair.
        """
        if not self.flattened_callables:
            return [], []

        sample_args = []
        sample_kwargs_list = []
        hidden_size = self.config.hidden_size

        for _microbatch_idx in range(self.num_microbatches):
            for layer in self.flattened_callables:
                hidden_states = torch.zeros(
                    self.seq_length,
                    1,
                    hidden_size,
                    dtype=torch.bfloat16,
                    device='cuda',
                    requires_grad=True,
                )

                if hasattr(layer, 'get_layer_static_inputs'):
                    static_inputs = layer.get_layer_static_inputs(self.seq_length, 1)
                    hidden_states = static_inputs.pop('hidden_states', hidden_states)
                    sample_args.append((hidden_states,))
                    sample_kwargs_list.append(static_inputs)
                else:
                    sample_args.append((hidden_states,))
                    sample_kwargs_list.append({})

        return sample_args, sample_kwargs_list

    def cuda_graph_set_manual_hooks(self):
        """No-op: vision encoder layers do not use DDP parameter-gather hooks.

        The parent derives hooks from model_chunk._make_forward_pre_hook which
        requires overlap_param_gather=True.  Vision encoder parameters are not
        distributed with the same overlap strategy, so we skip hook setup.
        """
