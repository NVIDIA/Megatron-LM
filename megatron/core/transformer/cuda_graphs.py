# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import gc
import inspect
import logging
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import partial
from itertools import chain, zip_longest
from math import ceil
from typing import Any, Dict, List

import torch
from torch.utils._pytree import tree_map as tree_map_pyt

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.tensor_parallel.random import (
    CudaRNGStatesTracker,
    get_all_rng_states,
    get_cuda_rng_tracker,
    is_checkpointing,
)
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.module import GraphableMegatronModule, MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_torch_version,
    is_te_min_version,
    log_on_each_pipeline_stage,
    log_single_rank,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from transformer_engine.pytorch.distributed import is_fp8_activation_recompute_enabled
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.graph import (
        make_graphed_callables,
        restore_fp8_tensors,
        save_fp8_tensors,
    )
    from transformer_engine.pytorch.graph import set_capture_end as te_set_capture_end
    from transformer_engine.pytorch.graph import set_capture_start as te_set_capture_start
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.utils import make_weak_ref

    HAVE_TE_GRAPHS = True
except:
    HAVE_TE_GRAPHS = False

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

_IS_GRAPH_CAPTURING = False
_IS_GRAPH_WARMUP = False
logger = logging.getLogger(__name__)

# Freeze GC during capture.
# TODO (@lmcafee): remove all freeze-GC code once most users are on PyTorch 2.9+.
FREEZE_GC = os.getenv("CUDA_GRAPH_CAPTURE_FREEZE_GC") != "0"
try:
    from packaging.version import Version as PkgVersion

    FREEZE_GC_MAX_TORCH_VERSION = PkgVersion("2.9.0a0")
    if get_torch_version() >= FREEZE_GC_MAX_TORCH_VERSION:
        FREEZE_GC = False
except ImportError:
    pass


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


@dataclass
class CudagraphBufferMetadata:
    """
    Metadata saved to tensors during cudagraph capture. This data will be used to determine
    during graph captue when a cudagraph can reuse a buffer or directly write its output into
    a subsequent's graph's input.
    """

    is_cudagraph_input: bool = False
    is_cudagraph_output: bool = False
    input_use_count: int = 0
    cudagraph_reuse_ref_count: int = 0
    capture_reuse_count: int = 0
    fwd_cudagraph_buffer: torch.Tensor = None
    bwd_cudagraph_buffer: torch.Tensor = None


class ArgMetadata:
    """Arg meta."""

    def __init__(self, arg):
        self.type = type(arg)
        if isinstance(arg, torch.Tensor):
            self.shape = arg.shape
            self.dtype = arg.dtype
            self.device = arg.device
            self.value = arg.data_ptr()
            self.requires_grad = arg.requires_grad
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


class TensorReusePool:
    """
    A pool-like list of tensors that can be reused as input and output buffers during graph capture.
    Also maintains strong references to all tensors created by this pool, so that they will never be
    freed by the memory allocator.
    """

    """Record strong references to buffers created by the pool so they cannot be deallocated between
    graph captures."""
    tensor_strong_refs: list = []

    """Record the data_ptrs of buffers created by the pool to check when a tensor came was 
    allocated from this pool. """
    tensor_strong_refs_dataptrs: set = set()

    """Buffers that have been returned to the pool and are available for reuse. """
    pool: list[torch.Tensor] = []

    def insert(self, tensor: torch.Tensor):
        """Return a tensor to the pool reuse."""
        assert self.owns(tensor)
        self.pool.append(tensor)

    def owns(self, tensor: torch.Tensor):
        """Check if a tensor was created from this pool."""
        return tensor.data_ptr() in self.tensor_strong_refs_dataptrs

    def get(self, meta: ArgMetadata):
        """Try to get a buffer from the pool. If a matching tensor is already in the pool, its
        assumed to be available and returned. Otherwise, allocate a new buffer."""

        assert isinstance(meta, ArgMetadata)
        # Find first matching buffer in pool
        for i, buf in enumerate(self.pool):
            if buf.shape == meta.shape and buf.dtype == meta.dtype and buf.device == meta.device:
                return self.pool.pop(i)

        out = meta.zeros_like()
        self.tensor_strong_refs.append(out)
        self.tensor_strong_refs_dataptrs.add(out.data_ptr())
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
            return dataclasses.replace(arg, **changes)

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


def _determine_if_first_last_layer_of_this_vp_chunk(base_module):
    """Determine if the given module is the first/last layer of the PP+VPP chunk it belongs to.
    Returns a tuple of two booleans indicating if the module is the first/last layer of the chunk.
    """

    # import modules here to avoid a circular import
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    if not hasattr(base_module, "layer_number"):
        return True, True

    # find all first/last layers of this PP stage
    first_layer_numbers = []
    last_layer_numbers = []
    vp_size = base_module.config.virtual_pipeline_model_parallel_size or 1
    for i in range(vp_size):
        # layer numbers are 1-indexed
        layer_offset = get_transformer_layer_offset(base_module.config, vp_stage=i)
        num_layers_to_build = get_num_layers_to_build(base_module.config, vp_stage=i)
        if num_layers_to_build > 0:
            first_layer_numbers.append(layer_offset + 1)
            last_layer_numbers.append(layer_offset + num_layers_to_build)
    return (
        base_module.layer_number in first_layer_numbers,
        base_module.layer_number in last_layer_numbers,
    )


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


fwd_buffer_reuse_ref_count = 0
bwd_buffer_reuse_ref_count = 0


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

    """A pool-like data structure to reuse input and output buffers across cudagraph."""
    tensor_reuse_pool = TensorReusePool()

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

        gc.collect()
        torch.cuda.empty_cache()

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()

        global bwd_buffer_reuse_ref_count, fwd_buffer_reuse_ref_count

        def format_mem_bytes(mem_bytes):
            for power, suffix in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb"), (0, "bytes")]:
                suffix_bytes = 1024**power
                if mem_bytes >= suffix_bytes:
                    return "%.1f %s" % (mem_bytes / suffix_bytes, suffix)
            return "%d bytes" % mem_bytes

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

        torch.cuda.set_stream(torch.cuda.default_stream())

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


def delete_cuda_graphs():
    """Delete all CUDA graphs."""

    # Reset runners.
    for record in [
        *_CudagraphGlobalRecord.cudagraph_record,
        *_CudagraphGlobalRecord.cudagraph_inference_record,
    ]:
        runner = record[0]
        assert isinstance(runner, _CudaGraphRunner)

        runner.cudagraph_created = False
        runner.fwd_graph_recorded = False
        runner.bwd_graph_recorded = False
        runner.fwd_graph = None
        runner.bwd_graph = None
        runner.mempool = None

    # Reset global tracking state
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []
    _CudagraphGlobalRecord.cudagraph_inference_record = []

    # TODO: Optional?: Force garbage collection to clean up memory
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
        need_copy_inputs = []
        for user_input, cudagraph_input in zip(inputs, runner.fwd_graph_input_surface):
            if (
                hasattr(cudagraph_input, "can_skip_replay_copy")
                and cudagraph_input.can_skip_replay_copy
            ):
                need_copy_inputs.append(user_input)
                assert user_input.data_ptr() == cudagraph_input.data_ptr()
            else:
                cudagraph_input.copy_(user_input)

        ctx.runner = runner
        ctx.save_for_backward(*need_copy_inputs)

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
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(not is_first_microbatch)
                runner.fp8_param_cache_updated = is_first_microbatch

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

        need_copy_inputs = list(ctx.saved_tensors)
        for cudagraph_input in runner.fwd_graph_input_surface:
            if (
                hasattr(cudagraph_input, "can_skip_replay_copy")
                and cudagraph_input.can_skip_replay_copy
            ):
                cudagraph_input.copy_(need_copy_inputs.pop(0))

        # Copy new data into bwd graph input buffer
        for user_output_grad, cudagraph_output_grad in zip(grads, runner.static_grad_outputs):
            if cudagraph_output_grad is None:
                continue
            if user_output_grad.data_ptr() != cudagraph_output_grad.data_ptr():
                cudagraph_output_grad.copy_(user_output_grad)

        runner.bwd_graph.replay()
        runner.status = _GraphStatus.FWD_READY

        # Update FP8 scale factors if needed
        if runner.fp8_enabled and isinstance(
            FP8GlobalStateManager.get_fp8_recipe(), te.common.recipe.DelayedScaling
        ):
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # If using gradient_accumulation_fusion, whenever `main_grad` is calculated
        # the `grad_added_to_main_grad` attribute is expected to set. However when using
        # cudagraphs this doesn't occur so we emulate this behavior here.
        for param, grad_added in runner.groundtruth_grad_added_to_main_grad.items():
            param.grad_added_to_main_grad = grad_added

        # Replaying the next bwd graph destroys the data held in static_grad_inputs, so clone
        # wgrads as autograd may launch the next graph before wgrads are accumulated
        dgrads = runner.static_grad_inputs[: runner.num_dgrads]
        wgrads = (g.clone() for g in runner.static_grad_inputs[runner.num_dgrads :])

        return None, None, *dgrads, *wgrads


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

        self.fwd_graph_recorded = False
        self.bwd_graph_recorded = False
        self.cudagraph_created = False
        self.status = _GraphStatus.FWD_READY

        self.fuse_wgrad_accumulation = False
        self.backward_retain_grad = False
        self.fp8_enabled = False
        self.fp4_enabled = False
        self.deallocate_pipeline_outputs = False

        self.grad_enabled = need_backward and torch.is_grad_enabled()
        self.func = super(MegatronModule, self.base_module).__call__ if func is None else func
        self.is_first_layer, self.is_last_layer = _determine_if_first_last_layer_of_this_vp_chunk(
            base_module
        )

        # We use this attribute to record the value of 'is_first_microbatch' each fwd cudagraph
        # replay so that way we only update the value of this flag in FP8GlobalStateManager when
        # it changes which incurs an HtoD sync
        if self.is_first_layer:
            self.fp8_param_cache_updated = None

        if hasattr(self.base_module, "config") and isinstance(
            self.base_module.config, TransformerConfig
        ):
            self.fuse_wgrad_accumulation = self.base_module.config.gradient_accumulation_fusion
            self.backward_retain_grad = self.base_module.config.cuda_graph_retain_backward_graph
            self.deallocate_pipeline_outputs = self.base_module.config.deallocate_pipeline_outputs
            self.num_warmup_steps = self.base_module.config.cuda_graph_warmup_steps
            self.fp8_enabled = self.base_module.config.fp8 is not None
            self.fp4_enabled = self.base_module.config.fp4 is not None
            self.fp8_runtime_enabled = None
            self.fp4_runtime_enabled = None

            if self.fp8_enabled:
                self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

            if self.fp4_enabled:
                from megatron.core.fp4_utils import get_fp4_recipe  # to avoid circular import

                self.fp4_recipe = get_fp4_recipe(self.base_module.config)
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

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

    def create_fwd_graph(self, args, kwargs, outputs=None, clone_inputs=True):
        """Create a fwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        global fwd_buffer_reuse_ref_count

        self.args = args
        self.kwargs = kwargs
        self.outputs = outputs

        # save grads and other variables that may be affected by graph warmup
        if self.training and torch.is_grad_enabled():
            grad_backup = []
            for param in self.base_module.parameters():
                grad_backup.append(param.main_grad.clone() if hasattr(param, "main_grad") else None)

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
            from megatron.core.transformer.moe.moe_utils import get_moe_layer_wise_logging_tracker

            tracker = get_moe_layer_wise_logging_tracker()
            cached_aux_losses = {}
            for name in tracker:
                if "values" in tracker[name]:
                    cached_aux_losses[name] = torch.clone(tracker[name]["values"])

        self.fwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        rng_states = get_all_rng_states()
        with torch.inference_mode(mode=False):
            for gen in rng_states.values():
                self.fwd_graph.register_generator_state(
                    _ensure_generator_state_is_cudagraph_safe(gen)
                )

        def _resolve_input_buffer(ten):
            if not isinstance(ten, ArgMetadata):
                return ten

            # the input tensor is resued from another cudagraph's input or output
            if (
                hasattr(ten, "cg_buffer_metadata")
                and ten.cg_buffer_metadata.fwd_cudagraph_buffer is not None
            ):
                global fwd_buffer_reuse_ref_count
                buf = ten.cg_buffer_metadata.fwd_cudagraph_buffer

                assert (
                    ten.cg_buffer_metadata.is_cudagraph_input
                    and buf.cg_buffer_metadata.capture_reuse_count > 0
                )

                if (
                    ten.cg_buffer_metadata.input_use_count > 1
                    and ten.cg_buffer_metadata.input_use_count
                    == buf.cg_buffer_metadata.capture_reuse_count
                ):
                    can_skip_replay_copy = False
                else:
                    can_skip_replay_copy = True

                buf.cg_buffer_metadata.capture_reuse_count -= 1
                if buf.cg_buffer_metadata.capture_reuse_count == 0:
                    ten.cg_buffer_metadata.fwd_cudagraph_buffer = None
                    fwd_buffer_reuse_ref_count -= 1
            else:
                # need to provide a fresh buffer from the reuse pool
                buf = _CudagraphGlobalRecord.tensor_reuse_pool.get(ten)
                can_skip_replay_copy = False

            buf = buf.detach().requires_grad_(ten.requires_grad)
            buf.can_skip_replay_copy = can_skip_replay_copy
            return buf

        if clone_inputs:
            # if a buffer is used for multiple inputs, create it now
            for ten in self.get_tensors(args, kwargs):
                if (
                    hasattr(ten, 'cg_buffer_metadata')
                    and ten.cg_buffer_metadata.input_use_count > 1
                    and ten.cg_buffer_metadata.fwd_cudagraph_buffer is None
                ):
                    buf = _CudagraphGlobalRecord.tensor_reuse_pool.get(ten)
                    buf.cg_buffer_metadata = deepcopy(ten.cg_buffer_metadata)
                    buf.cg_buffer_metadata.capture_reuse_count = (
                        ten.cg_buffer_metadata.input_use_count
                    )
                    ten.cg_buffer_metadata.fwd_cudagraph_buffer = buf
                    fwd_buffer_reuse_ref_count += 1

            self.fwd_graph_input_args = tree_map(_resolve_input_buffer, args)
            self.fwd_graph_input_kwargs = tree_map(_resolve_input_buffer, kwargs)
        else:
            self.fwd_graph_input_args, self.fwd_graph_input_kwargs = args, kwargs

        self.fwd_graph_input_surface = self.get_tensors(
            self.fwd_graph_input_args, self.fwd_graph_input_kwargs
        )

        ctx = torch.no_grad() if not self.grad_enabled else nullcontext()
        with ctx:
            # warmup again as case graph capture mode may execute a different codepath
            _set_warmup_start()
            for _ in range(self.num_warmup_steps):
                with self.get_quantization_context():

                    def clone_ten(ten):
                        if not torch.is_tensor(ten):
                            return ten
                        return torch.zeros_like(ten).requires_grad_(ten.requires_grad)

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
            _set_warmup_end()

            with self.get_quantization_context():
                torch.cuda.synchronize()
                # Register default CUDA generators ourselves (fixed in-place to have normal tensors)
                # before capture begins, to avoid inference-tensor state issues during capture.
                with torch.inference_mode(mode=False):
                    for device_idx in range(torch.cuda.device_count()):
                        default_gen = torch.cuda.default_generators[device_idx]
                        self.fwd_graph.register_generator_state(
                            _ensure_generator_state_is_cudagraph_safe(default_gen)
                        )

                # Freeze GC, to speed up capture time ~15-20x.
                if FREEZE_GC:
                    gc.freeze()

                with torch.cuda.graph(
                    self.fwd_graph, pool=self.mempool, capture_error_mode="thread_local"
                ):
                    fwd_graph_outputs = self.func(
                        *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                    )

                # Unfreeze GC.
                if FREEZE_GC:
                    gc.unfreeze()

                    # gc.collect() drops references to unreachable tensors created during capture,
                    # returning their storage to the allocator to avoid a slowdown during replay.
                    # However, it forces expensive global garbage collection, so must be done
                    # only on the last layer per-device to avoid slowing down graph creation.
                    if self.is_last_layer:
                        gc.collect()

        # save cudagraph output buffer
        self.fwd_graph_outputs = fwd_graph_outputs
        self.fwd_graph_output_surface = self.get_tensors(fwd_graph_outputs)

        for fwd_graph_out, o in zip(
            self.fwd_graph_output_surface, self.get_arg_metas(self.outputs)
        ):
            assert hasattr(o, "cg_buffer_metadata") and o.cg_buffer_metadata.is_cudagraph_output

            if (
                o.cg_buffer_metadata.is_cudagraph_input
                and o.cg_buffer_metadata.fwd_cudagraph_buffer is None
            ):
                fwd_graph_out.cg_buffer_metadata = deepcopy(o.cg_buffer_metadata)
                fwd_graph_out.cg_buffer_metadata.capture_reuse_count = (
                    o.cg_buffer_metadata.cudagraph_reuse_ref_count
                )
                o.cg_buffer_metadata.fwd_cudagraph_buffer = fwd_graph_out
                fwd_buffer_reuse_ref_count += 1

        # if an input buffer requires a copy, and does not have metadata attached to it at this
        # point, it will not be reused after this forward pass, so return it to the pool
        for buf in self.fwd_graph_input_surface:
            if (
                hasattr(buf, "can_skip_replay_copy")
                and not buf.can_skip_replay_copy
                and not hasattr(buf, "cg_buffer_metadata")
            ):
                assert _CudagraphGlobalRecord.tensor_reuse_pool.owns(buf)
                _CudagraphGlobalRecord.tensor_reuse_pool.insert(buf)

        if self.training and torch.is_grad_enabled():
            assert (
                len(self.fwd_graph_output_surface) > 0
            ), """Tried graphing a module that returned no tensors in training mode, 
                however the graphed module must output at least one tensor, 
                so that a corresponding backward node may be registered in the autograd graph."""

            self.params_to_backprop = self.get_connected_params(fwd_graph_outputs)
            self.num_wgrads = len(self.params_to_backprop)
            self.num_dgrads = len(self.fwd_graph_input_surface)
            self.fwd_graph_input_surface = self.fwd_graph_input_surface + self.params_to_backprop

            if self.fp8_enabled:
                restore_fp8_tensors([self.base_module], saved_fp8_tensors)
            # restore cached grads
            for main_grad_copy, param in zip(grad_backup, self.base_module.parameters()):
                if main_grad_copy is not None:
                    param.main_grad.copy_(main_grad_copy)

        if is_moe:
            for name in tracker:
                tracker[name]["values"].copy_(cached_aux_losses[name])

    def create_bwd_graph(self):
        """Create a bwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        # unlike 'fwd_buffer_reuse_ref_count', 'bwd_buffer_reuse_ref_count' may not decrement
        # to 0 when activation checkpointing is used. See [interaction with recompute].
        global bwd_buffer_reuse_ref_count

        assert self.grad_enabled
        self.bwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.bwd_graph.register_generator_state(state)

        self.static_grad_outputs = []
        for o in self.get_arg_metas(self.outputs):
            out_grad = None
            if o.requires_grad:
                # TODO: (jiemingz) [interaction with recompute]
                # for activation recompute, the fwd pass is rerun in the backward pass and
                # the metadata we attach in record_graph_capture is lost. As a result the next
                # cudagraph expects the buffer to be provided 'fwd_cudagraph_buffer' but is missing.
                # So, we cannot always assume this metadata exists. Consequently, there are extra
                # copies between the outputs of the fwd-bwd pass and the bwd pass.
                if (
                    o.cg_buffer_metadata.is_cudagraph_input
                    and o.cg_buffer_metadata.bwd_cudagraph_buffer is not None
                ):
                    o.cg_buffer_metadata.bwd_cudagraph_buffer.shape == o.shape

                    out_grad = o.cg_buffer_metadata.bwd_cudagraph_buffer
                    o.cg_buffer_metadata.bwd_cudagraph_buffer = None
                    out_grad.cg_buffer_metadata.capture_reuse_count -= 1
                    bwd_buffer_reuse_ref_count -= 1
                else:
                    out_grad = _CudagraphGlobalRecord.tensor_reuse_pool.get(o)
                out_grad.requires_grad = True
            self.static_grad_outputs.append(out_grad)

        # Freeze GC, to speed up capture time ~15-20x.
        if FREEZE_GC:
            gc.freeze()

        with torch.cuda.graph(self.bwd_graph, pool=self.mempool):
            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in self.fwd_graph_output_surface if o.requires_grad),
                inputs=tuple(i for i in self.fwd_graph_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in self.static_grad_outputs if o is not None),
                retain_graph=self.backward_retain_grad,
                only_inputs=True,
                allow_unused=True,
            )

        # Unfreeze GC.
        if FREEZE_GC:
            gc.unfreeze()

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs
        # that don't require grad
        grad_inputs = list(grad_inputs)
        self.static_grad_inputs = []
        for input_tensor in self.get_arg_metas(self.args, self.kwargs):
            if input_tensor.requires_grad:
                input_grad = grad_inputs.pop(0)
                input_grad.cg_buffer_metadata = deepcopy(input_tensor.cg_buffer_metadata)
                if input_tensor.cg_buffer_metadata.is_cudagraph_output:
                    if input_tensor.cg_buffer_metadata.bwd_cudagraph_buffer is None:
                        input_tensor.cg_buffer_metadata.bwd_cudagraph_buffer = input_grad
                        input_grad.cg_buffer_metadata.capture_reuse_count += 1
                        bwd_buffer_reuse_ref_count += 1
                self.static_grad_inputs.append(input_grad)
            else:
                self.static_grad_inputs.append(None)

        # at this point static_grad_inputs hold the input dgrads, add the wgrads next
        assert self.num_wgrads == len(grad_inputs)
        self.static_grad_inputs.extend(grad_inputs)
        self.static_grad_inputs = tuple(self.static_grad_inputs)
        self.static_grad_outputs = tuple(self.static_grad_outputs)

        self.groundtruth_grad_added_to_main_grad = {}
        if self.fuse_wgrad_accumulation:
            for param in self.params_to_backprop:
                if hasattr(param, "grad_added_to_main_grad"):
                    self.groundtruth_grad_added_to_main_grad[param] = param.grad_added_to_main_grad

        # After backward pass grad_output buffers are no longer used and returned to the pool
        for ten in self.static_grad_outputs:
            if torch.is_tensor(ten):
                # Check that the tensor is not in use. This scenario may occur when a cudagraph
                # passes its input directly as an output, and places this output as the
                # input of a subsequent cudgraph, leading to a grad output buffer to be still in use
                # even after the backward pass.
                reuse_count = (
                    ten.cg_buffer_metadata.capture_reuse_count
                    if hasattr(ten, "cg_buffer_metadata")
                    else 0
                )

                if _CudagraphGlobalRecord.tensor_reuse_pool.owns(ten) and reuse_count == 0:
                    _CudagraphGlobalRecord.tensor_reuse_pool.insert(ten)

        # now weakref everything
        if HAVE_TE_GRAPHS:

            def replace_with_weak_ref(arg):
                if not torch.is_tensor(arg):
                    return arg

                ref = make_weak_ref(arg)
                ref.requires_grad = arg.requires_grad
                if hasattr(arg, "can_skip_replay_copy"):
                    ref.can_skip_replay_copy = arg.can_skip_replay_copy
                return ref

            self.fwd_graph_input_surface = tree_map(
                replace_with_weak_ref, self.fwd_graph_input_surface
            )
            self.fwd_graph_input_args = tree_map(replace_with_weak_ref, self.fwd_graph_input_args)
            self.fwd_graph_input_kwargs = tree_map(
                replace_with_weak_ref, self.fwd_graph_input_kwargs
            )
            self.fwd_graph_output_surface = tree_map(
                replace_with_weak_ref, self.fwd_graph_output_surface
            )
            # It is safe to weakref static_grad_inputs as any inuse input grads have a strong ref
            # stored in 'bwd_cudagraph_buffer'
            self.static_grad_inputs = tree_map(replace_with_weak_ref, self.static_grad_inputs)
            self.static_grad_outputs = tree_map(replace_with_weak_ref, self.static_grad_outputs)

        delattr(self, "args")
        delattr(self, "kwargs")
        delattr(self, "outputs")

    def apply_cudagraph_record_metadata(self, args, kwargs, outputs):
        """Attaches graph capture metadata to all passed in tensors."""

        for t in self.get_tensors(args, kwargs):
            if not hasattr(t, "cg_buffer_metadata"):
                t.cg_buffer_metadata = CudagraphBufferMetadata()

            t.cg_buffer_metadata.is_cudagraph_input = True
            t.cg_buffer_metadata.input_use_count += 1

            if t.cg_buffer_metadata.is_cudagraph_output:
                t.cg_buffer_metadata.cudagraph_reuse_ref_count += 1

        # mark all outputs, so that the fwd graph we may reuse cudagraph output buffers as inputs
        for o in self.get_tensors(outputs):
            o.cg_buffer_metadata = CudagraphBufferMetadata()
            o.cg_buffer_metadata.is_cudagraph_output = True

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

        if not self.fwd_graph_recorded:
            logger.debug(f"Recording forward graph creation...")

            self.apply_cudagraph_record_metadata(args, kwargs, out)

            def _replace_with_meta(arg):
                return ArgMetadata(arg) if torch.is_tensor(arg) else arg

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

        inp_tensors = self.get_tensors(args, kwargs, check_types=False)
        if self.grad_enabled:
            func_args = inp_tensors + self.params_to_backprop
        else:
            func_args = inp_tensors

        out = _CudagraphReplayNode.apply(self, is_first_microbatch, *func_args)

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

    def __init__(
        self, config: TransformerConfig, base_module=None, function_name=None, need_backward=True
    ):
        super().__init__()
        """Creates a CudaGraphManager to manage CUDA graphs for a Megatron module.

        Args:
            config: TransformerConfig object containing CUDA graph settings for memory
                pooling, graph retention, gradient accumulation, FP8/FP4, and warmup steps.
        """
        rng_tracker = get_cuda_rng_tracker()
        self.need_backward = need_backward

        if function_name is not None:
            func = getattr(base_module, function_name)

            def wrapped_func(*args, **kwargs):
                out = self(base_module, args, kwargs)
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
        if torch.cuda.get_device_capability()[0] < 10:
            assert (
                "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
                or os.getenv("NCCL_GRAPH_REGISTER", "") == "0"
            ), (
                "Setting NCCL_GRAPH_REGISTER=0 to avoid illegal memory access when using "
                "CUDA Graph with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
            )

        self.cudagraph_runners: list[_CudaGraphRunner] = []
        self.inference_cudagraphs_lookup_table: dict = defaultdict(lambda: None)
        self.is_first_microbatch = False

        # Without pipeline parallelism, microbatches execute one at a time.
        # Therefore modules will always execute in the same order, so cudagraphs
        # can both be reused and share a single mempool.
        self.reuse_cudagraphs = parallel_state.get_pipeline_model_parallel_world_size() == 1
        if CudaGraphManager.global_mempool is None:
            CudaGraphManager.global_mempool = torch.cuda.graph_pool_handle()
            # Cudagraph stream capture requires no operations on the default stream prior to the
            # capture, so change to a side stream.
            torch.cuda.set_stream(torch.cuda.Stream())

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

    def get_cudagraph_runner(self, megatron_module, args, kwargs, reuse_cudagraphs):
        '''Returns a valid cudagraph runner for the current forward call.
        The cudagraph corresponding to this call is the first element of 'self.cudagraph_runners'.
        We iterate through the list by 1 for each call, and the number of calls is equal to the
        length of 'self.cudagraph_runners'.
        Otherwise, we assign a mempool per microbatch, which allows cudagraphs to be reused
        over different microbatches by tracking their respective fwd and bwd passes.'''
        if reuse_cudagraphs:
            is_inference_mode = 'inference_context' in kwargs.keys() and kwargs['inference_context']
            if is_inference_mode:
                is_static_batching = kwargs['inference_context'].is_static_batching()
                if is_static_batching:
                    batch_size = kwargs['hidden_states'].shape[0]
                    is_decode_only = kwargs["inference_context"].is_decode_only()
                    runner = self.inference_cudagraphs_lookup_table[(batch_size, is_decode_only)]
                else:
                    padded_batch_dimensions = kwargs['inference_context'].padded_batch_dimensions
                    runner = self.inference_cudagraphs_lookup_table[padded_batch_dimensions]
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
                    self.cudagraph_runners.append(runner)
                    if is_inference_mode:
                        # Cache the newly created runner in the inference lookup table.
                        if is_static_batching:
                            self.inference_cudagraphs_lookup_table[(batch_size, is_decode_only)] = (
                                runner
                            )
                        else:
                            self.inference_cudagraphs_lookup_table[padded_batch_dimensions] = runner
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

        return runner

    def __call__(self, megatron_module, args, kwargs):
        """Calls the forward pass of the cudagraphed module.

        Args:
            megatron_module (torch.nn.module): The megatron module to be graphed and run

            args (tuple):  The positional args to be passed to the module.

            kwargs (dict):  The keyword args to be passed to the module.
        """
        is_inference_mode = 'inference_context' in kwargs.keys() and kwargs['inference_context']
        is_in_checkpoint_fwd = is_checkpointing()
        if HAVE_TE_GRAPHS:
            is_in_checkpoint_fwd = is_in_checkpoint_fwd or is_fp8_activation_recompute_enabled()

        if _CudagraphGlobalRecord.cudagraph_created:
            if self.training and torch.is_grad_enabled():
                # Trigger Mcore DDP pre-forward hooks
                self.call_ddp_preforward_hook(megatron_module)
                for module in megatron_module.modules():
                    self.call_ddp_preforward_hook(module)

            runner = self.get_cudagraph_runner(megatron_module, args, kwargs, self.reuse_cudagraphs)
            out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)
        else:
            if is_inference_mode:
                # Inference generation mode creates graphs immediately
                runner = self.get_cudagraph_runner(megatron_module, args, kwargs, True)
                runner.eval()

                if not runner.fwd_graph_recorded:
                    # Reuse graph input-output buffers for inference
                    local_args, local_kwargs = args, kwargs
                    if not runner.is_first_layer:
                        # Find previous layer's runner in the global record
                        try:
                            previous_runner = next(
                                r
                                for r in _CudagraphGlobalRecord.cudagraph_inference_record
                                if (
                                    r[0].base_module.layer_number
                                    == runner.base_module.layer_number - 1
                                    and r[0].fwd_graph is not None
                                    and ArgMetadata(r[3]['hidden_states'])
                                    == ArgMetadata(kwargs['hidden_states'])
                                )
                            )
                            # Replace the hidden states from previous layer's output buffer
                            local_kwargs = dict(kwargs)
                            local_kwargs['hidden_states'] = previous_runner[0].fwd_graph_outputs[0]
                        except StopIteration:
                            # No match found for previous layer, continue with no buffer reuse
                            pass

                    runner.create_fwd_graph(
                        local_args, local_kwargs, outputs=None, clone_inputs=runner.is_first_layer
                    )
                    runner.fwd_graph_recorded = True
                    runner.cudagraph_created = True

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
                # check if a layer is frozen during training.
                if not torch.is_grad_enabled():
                    # If the layer is frozen, we need to set the runner to eval mode.
                    runner.eval()
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
        if is_inference_mode or not torch.is_grad_enabled():
            runner.status = _GraphStatus.FWD_READY
        else:
            runner.status = _GraphStatus.BWD_READY

        return out


# The following functions are for capturing CUDA Graphs using TE make_graphed_callables().
def _layer_is_graphable(layer, config):
    """
    Check if a layer is graphable.
    """

    # Only GraphableMegatronModule can be graphed.
    if not isinstance(layer, GraphableMegatronModule):
        return False

    # If cuda_graph_scope is not set, every layer is graphed.
    if not config.cuda_graph_scope:
        return True

    # import modules here to avoid a circular import
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if isinstance(layer, MambaLayer) and CudaGraphScope.mamba in config.cuda_graph_scope:
        # mamba layer.
        return True
    if isinstance(layer, TransformerLayer):
        if CudaGraphScope.attn in config.cuda_graph_scope and not (
            isinstance(layer.self_attention, IdentityOp)
            and isinstance(layer.cross_attention, IdentityOp)
        ):
            # attn layer.
            return True
        if (
            CudaGraphScope.moe in config.cuda_graph_scope
            or CudaGraphScope.moe_router in config.cuda_graph_scope
            or CudaGraphScope.moe_preprocess in config.cuda_graph_scope
        ) and isinstance(layer.mlp, MoELayer):
            # moe layer.
            return True
        if CudaGraphScope.mlp in config.cuda_graph_scope and isinstance(layer.mlp, MLP):
            # mlp layer.
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

    def __init__(self, model, config, seq_length, micro_batch_size, optimizers=[]):
        assert HAVE_TE_GRAPHS, "CUDA Graphs are not supported without TE."
        assert (
            config.cuda_graph_impl == "transformer_engine"
        ), "Option cuda_graph_impl=transformer_engine not enabled."
        assert (
            "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
            or os.getenv("NCCL_GRAPH_REGISTER", "") == "0"
        ), (
            "Setting NCCL_GRAPH_REGISTER=0 to avoid illegal memory access when using "
            "CUDA Graph with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
        )
        assert CudaGraphScope.full_iteration not in config.cuda_graph_scope, (
            "full_iteration cuda graph is not supported for cuda_graph_impl=transformer_engine. "
            "Please use cuda_graph_impl=local instead."
        )

        self.model = model
        self.config = config
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.optimizers = optimizers
        self.num_model_chunks = len(model)

        # Number of microbatches to capture. The value will be set in _get_cuda_graph_input_data().
        self.num_microbatches = None

        # Get callables with captureable layers.
        self.chunks_with_decoder = []
        self.num_layers_per_chunk = []
        self.callables_per_chunk = []
        self.callables_per_chunk_is_mtp = []
        self.flattened_callables = []
        self.flattened_callables_is_mtp = []
        for chunk_number, model_chunk in enumerate(model):
            try:
                chunk_with_decoder = get_attr_wrapped_model(
                    model_chunk, 'decoder', allow_none=False, return_model_obj=True
                )
            except RuntimeError:
                num_graphable_layers = 0
                log_on_each_pipeline_stage(
                    logger=logger,
                    tp_group=None,
                    dp_cp_group=None,
                    level=logging.DEBUG,
                    msg=f'Rank {torch.distributed.get_rank()}: '
                    f'No valid layer in model chunk {chunk_number}.',
                )
            else:
                num_decoder_layers = len(chunk_with_decoder.decoder.layers)
                if hasattr(chunk_with_decoder, 'mtp'):
                    num_mtp_layers = len(chunk_with_decoder.mtp.layers)
                else:
                    num_mtp_layers = 0
                num_graphable_layers = 0
                callables, callables_is_mtp = [], []
                for layer_number in range(num_decoder_layers):
                    layer = chunk_with_decoder.decoder.layers[layer_number]
                    if _layer_is_graphable(layer, config):
                        num_graphable_layers += 1
                        callables.append(layer)
                        callables_is_mtp.append(False)
                for layer_number in range(num_mtp_layers):
                    layer = chunk_with_decoder.mtp.layers[layer_number].mtp_model_layer
                    if _layer_is_graphable(layer, config):
                        num_graphable_layers += 1
                        callables.append(layer)
                        callables_is_mtp.append(True)
                log_on_each_pipeline_stage(
                    logger=logger,
                    tp_group=None,
                    dp_cp_group=None,
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

        log_on_each_pipeline_stage(
            logger=logger,
            tp_group=None,
            dp_cp_group=None,
            level=logging.INFO,
            msg=f'Rank {torch.distributed.get_rank()}: '
            f'{len(self.flattened_callables)} graphable layers.',
        )

        # One helper object can only capture CUDA Graphs once. Use this flag to check if the graphs
        # have been created.
        self._graphs_created = False

    def graphs_created(self):
        """
        Returns whether the CUDA Graphs have been created.
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
                convert_schedule_table_to_order(). Positive integers represent forward passes
                (1-indexed chunk ID), negative integers represent backward passes.
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
            assert layer in chunk_of_the_layer.decoder.layers or any(
                layer is mtp_layer.mtp_model_layer for mtp_layer in chunk_of_the_layer.mtp.layers
            ), "Layer is not in the chunk"

            def get_rotary_pos_emb(transformer_module, transformer_input):
                if (
                    transformer_module.position_embedding_type == 'rope'
                    and not self.config.multi_latent_attention
                ):
                    rotary_seq_len = transformer_module.rotary_pos_emb.get_rotary_seq_len(
                        None, transformer_module.decoder, transformer_input, self.config, None
                    )
                    if rotary_seq_len not in rotary_pos_emb_cache:
                        rotary_pos_emb_cache[rotary_seq_len] = transformer_module.rotary_pos_emb(
                            rotary_seq_len
                        )
                    return rotary_pos_emb_cache[rotary_seq_len]
                else:
                    return None

            static_inputs = layer.get_layer_static_inputs(self.seq_length, self.micro_batch_size)

            from megatron.core.transformer.identity_op import IdentityOp
            from megatron.core.transformer.transformer_layer import TransformerLayer

            contains_self_attn = (
                isinstance(layer, TransformerLayer)
                and not isinstance(layer.self_attention, IdentityOp)
                and (
                    not self.config.cuda_graph_scope
                    or CudaGraphScope.attn in self.config.cuda_graph_scope
                )
            )

            _sample_kwargs = {}
            if is_te_min_version("1.10.0"):
                # te.make_graphed_callables() accepts keyword arguments since 1.10.0.
                hidden_states = static_inputs.pop("hidden_states")
                _sample_args = (hidden_states,)
                if contains_self_attn:
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
        if (
            parallel_state.get_pipeline_model_parallel_world_size() == 1
            and not self.config.overlap_moe_expert_parallel_comm
        ):
            assert (
                self.num_model_chunks == 1
            ), "If PP is not enabled, there should be only one model chunk."
            self.num_microbatches = 1
        else:
            self.num_microbatches = get_num_microbatches()

        _, _, num_warmup_microbatches, _ = get_pp_rank_microbatches(
            self.num_microbatches,
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
            False,
        )
        schedule_table = get_schedule_table(
            self.num_microbatches,
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
        )
        order = convert_schedule_table_to_order(
            num_warmup_microbatches, self.num_model_chunks, schedule_table
        )
        log_on_each_pipeline_stage(
            logger=logger,
            tp_group=None,
            dp_cp_group=None,
            level=logging.DEBUG,
            msg=f'Rank {torch.distributed.get_rank()}: ORDER {order}',
        )
        chunk_id_list = None
        if self.config.overlap_moe_expert_parallel_comm:
            wgrad_in_graph_scope = CudaGraphScope.attn in self.config.cuda_graph_scope or (
                CudaGraphScope.moe_router in self.config.cuda_graph_scope
                and self.config.moe_shared_expert_intermediate_size is not None
                and not self.config.moe_shared_expert_overlap
            )
            capture_wgrad_graph = self.config.delay_wgrad_compute and wgrad_in_graph_scope
            order, chunk_id_list = get_overlap_moe_expert_parallel_comm_order(
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
                tp_group=None,
                dp_cp_group=None,
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
                if is_te_min_version("1.14.0") and parallel_state.model_parallel_is_initialized():
                    kwargs['fp8_group'] = parallel_state.get_amax_reduction_group(
                        with_context_parallel=True, tp_only_amax_red=self.config.tp_only_amax_red
                    )
            else:
                kwargs['fp8_enabled'] = False
            return kwargs

        kwargs = get_make_graphed_callables_kwargs()
        return sample_args, kwargs

    def _start_capturing(self):
        """
        Start capturing CUDA Graphs.
        """
        assert not self._graphs_created, "CUDA Graphs have already been created."

        torch.distributed.barrier()
        gc.collect()
        torch.cuda.empty_cache()
        if FREEZE_GC:
            gc.freeze()

        _set_capture_start()
        log_single_rank(logger, logging.INFO, f'Start CUDA Graphs capture...')
        return time.time()

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

        from megatron.core.distributed.finalize_model_grads import reset_model_temporary_tensors
        from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

        torch.distributed.barrier()
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        clear_aux_losses_tracker()
        reset_model_temporary_tensors(self.config, self.model)

        if FREEZE_GC:
            gc.unfreeze()
        gc.collect()
        torch.cuda.empty_cache()

        self._graphs_created = True

    def create_cudagraphs(self):
        """
        Capture CUDA Graphs per TransformerLayer per microbatch.
        """
        start_time = self._start_capturing()

        # Prepare CUDA Graph capturing input data and call `make_graphed_callables`.
        sample_args, kwargs = self._get_cuda_graph_input_data()
        if self.config.sequence_parallel:
            rng_context = get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            graphs = make_graphed_callables(tuple(self.flattened_callables), sample_args, **kwargs)

        # Push the captured graphs to the corresponding TransformerBlock.
        num_layers_accumulated = 0
        for layers in self.callables_per_chunk:
            for layer_number, layer in enumerate(layers):
                layer.cuda_graphs = []
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
                    layer.cuda_graphs.append(graphs[graph_idx])
            num_layers_accumulated += len(layers)

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
        assert self._graphs_created, "CUDA Graphs have not been created."

        graph_resettable = is_te_min_version("2.10.0")
        graphs_reset, graphs_not_reset = 0, 0
        for layers in self.callables_per_chunk:
            for layer in layers:
                for graph in layer.cuda_graphs:
                    if graph_resettable:
                        graph.reset()
                        graphs_reset += 1
                    else:
                        graphs_not_reset += 1
                layer.cuda_graphs = []
                layer.cuda_graph_manual_hooks = []

        log_on_each_pipeline_stage(
            logger=logger,
            tp_group=None,
            dp_cp_group=None,
            level=logging.INFO,
            msg=f'Rank {torch.distributed.get_rank()}: '
            f'{graphs_reset} graphs deleted with explicit reset, '
            f'{graphs_not_reset} graphs deleted without explicit reset.',
        )
        self._graphs_created = False


def convert_schedule_table_to_order(num_warmup_microbatches, num_model_chunks, schedule_table):
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


def get_overlap_moe_expert_parallel_comm_order(order, num_layers_per_chunk, capture_wgrad_graph):
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
