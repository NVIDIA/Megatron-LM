# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import gc
import inspect
import logging
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from torch.utils._pytree import tree_flatten

from megatron.core import parallel_state
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.tensor_parallel.random import (
    CudaRNGStatesTracker,
    get_all_rng_states,
    get_cuda_rng_tracker,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_attr_wrapped_model,
    is_te_min_version,
    log_on_each_pipeline_stage,
    log_single_rank,
)

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.graph import (
        make_graphed_callables,
        restore_fp8_tensors,
        save_fp8_tensors,
    )
    from transformer_engine.pytorch.graph import set_capture_end as te_set_capture_end
    from transformer_engine.pytorch.graph import set_capture_start as te_set_capture_start
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    HAVE_TE_GRAPHS = True
except:
    HAVE_TE_GRAPHS = False

try:
    from tqdm import tqdm

    HAVE_TQDM = True
except:
    HAVE_TQDM = False

_IS_GRAPH_CAPTURING = False

logger = logging.getLogger(__name__)


def is_graph_capturing():
    """Query if currently capturing."""
    global _IS_GRAPH_CAPTURING
    return _IS_GRAPH_CAPTURING


def _set_capture_start():
    """Set graph capture has started."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = True


def _set_capture_end():
    """Set graph capture has ended."""
    global _IS_GRAPH_CAPTURING
    _IS_GRAPH_CAPTURING = False


class ArgMetadata:
    """Arg meta."""

    def __init__(self, arg):
        self.type = type(arg)
        if isinstance(arg, torch.Tensor):
            self.shape = arg.shape
            self.dtype = arg.dtype
            self.device = arg.device
        else:
            self.value = arg


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
        StaticInferenceContext,
        DynamicInferenceContext,
    }
    assert meta.type in _SUPPORTED_TYPES or is_dataclass(
        meta.value
    ), f"Cudagraphs recieved an arg of type {meta.type} which is not supported."


def _determine_if_transformer_decoder_layer(base_module):
    """Determine if the given module is a transformer decoder layer."""
    # import modules here to avoid a circular import
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.transformer_layer import BaseTransformerLayer, TransformerLayer

    is_potential_decoder_layer = isinstance(
        base_module, (TransformerLayer, BaseTransformerLayer, MambaLayer)
    )
    if not is_potential_decoder_layer:
        return False
    if isinstance(base_module, TransformerLayer) and not isinstance(
        base_module.cross_attention, IdentityOp
    ):
        # If the layer has a cross attention, it is not a decoder layer
        return False
    else:
        # Otherwise it is a decoder layer
        return True


def _determine_if_first_last_layer_of_this_vp_chunk(base_module):
    """Determine if the given module is the first/last layer of the PP+VPP chunk it belongs to.
    Returns a tuple of two booleans indicating if the module is the first/last layer of the chunk.
    """

    # import modules here to avoid a circular import
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

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


class _CudagraphGlobalRecord:
    """A global datastructure that records of the ordering of all _CudaGraphRunner's
    first fwd or bwd passes. 'create_cudagraphs' will use this to create
    cudagraphs in execution order, which is required for cudagraphs sharing a mempool."""

    """A global flag that if true, all cudagraph runners
    fwd and bwd passes will be performed using their cudagraphed versions."""
    cudagraph_created = False

    """A record of fwd and bwd graph creation, populated with 'record_fwd_graph' and
    'record_bwd_graph."""
    cudagraph_record = []
    cudagraph_inference_record = []

    @classmethod
    def record_fwd_graph(cls, runner, args, kwargs):
        """Record a fwd graph to 'cudagraph_record"""
        cls.cudagraph_record.append((runner, "fwd", args, kwargs))

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
        logging.getLogger(__name__).info(f"Creating {len(cls.cudagraph_record)} CUDA graphs")

        has_te_modules = False
        if HAVE_TE_GRAPHS:
            for g in cls.cudagraph_record:
                base_module = g[0].base_module
                has_te_modules = has_te_modules or any(
                    [isinstance(m, TransformerEngineBaseModule) for m in base_module.modules()]
                )

        # If graphing only transformer layers with self attention, then apply the following
        # transformer layer specific optimizations that reduce memory usage and tensor copies:
        # These eventually will become unneccessary with:
        # https://github.com/pytorch/pytorch/pull/137318
        # 1. Some inputs to TransformerLayer (e.g. rotary_emb) are the same over all layers
        #    and only need to be set once.
        # 2. Because the next layer consumes the previous layer's hidden states, all fwd
        #    cudagraphs can alternate reusing the same hidden_state input, output buffer.
        #    Similarly, bwd graphs can alternate the same output, input grad buffers.
        optimize_transformer_layer_graph_buffers = all(
            [g[0].reuse_input_output_buffer for g in cls.cudagraph_record]
        )
        if optimize_transformer_layer_graph_buffers:
            prev_fwd_hidden_state_output = None
            prev_bwd_hidden_state_inputgrad = None

        gc.collect()
        torch.cuda.empty_cache()

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()

        def format_mem_bytes(mem_bytes):
            for power, suffix in [(4, "tb"), (3, "gb"), (2, "mb"), (1, "kb"), (0, "bytes")]:
                suffix_bytes = 1024**power
                if mem_bytes >= suffix_bytes:
                    return "%.1f %s" % (mem_bytes / suffix_bytes, suffix)
            return "%d bytes" % mem_bytes

        time_start = time.time()
        mem_stats_start = torch.cuda.memory_stats()
        progress_bar = enumerate(cls.cudagraph_record)
        if HAVE_TQDM:
            progress_bar = tqdm(progress_bar, "create cuda graphs", total=len(cls.cudagraph_record))
        for g_idx, g in progress_bar:

            runner, graph_type = g[0:2]

            mem_stats = torch.cuda.memory_stats()
            progress_str = "create cuda graphs | mem: alloc %s, res %s" % (
                format_mem_bytes(mem_stats["allocated_bytes.all.current"]),
                format_mem_bytes(mem_stats["reserved_bytes.all.current"]),
            )
            if HAVE_TQDM:
                progress_bar.set_description(progress_str)
            elif g_idx % 100 == 0 or g_idx == len(cls.cudagraph_record) - 1:
                logger.info(f"{g_idx}/{len(cls.cudagraph_record)}. {progress_str}")

            if optimize_transformer_layer_graph_buffers:
                if graph_type == 'fwd':
                    args, kwargs = g[2:]

                    if not runner.is_first_layer:
                        kwargs['hidden_states'] = prev_fwd_hidden_state_output
                    runner.create_fwd_graph(args, kwargs, clone_inputs=False)

                    # The output of TransformerLayer is: (hidden_states, None)
                    # The output of MambaLayer is: (hidden_states,)
                    # make sure to get the hidden states tensor from the tuple
                    prev_fwd_hidden_state_output = runner.fwd_graph_outputs[0]

                else:
                    # In vision models, encoder and decoder transformers have different
                    # hidden_states shapes. Each has its own first and last layers that
                    # are noncontiguous. Reset prev_bwd_hidden_state_inputgrad to None at
                    # each last layer to avoid shape mismatch when transitioning between
                    # encoder and decoder.
                    if runner.is_last_layer:
                        prev_bwd_hidden_state_inputgrad = None

                    runner.create_bwd_graph(prev_bwd_hidden_state_inputgrad)

                    # The first input grad TransformerLayer is for 'hidden_states'
                    prev_bwd_hidden_state_inputgrad = runner.static_grad_inputs[0]
            else:
                runner, graph_type = g[0:2]
                if graph_type == 'fwd':
                    args, kwargs = g[2:]
                    runner.create_fwd_graph(args, kwargs)
                else:
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
        logger.info(
            "> built %d cuda graph(s) in %.2f sec, with total memory usage: "
            "allocated %s, reserved %s."
            % (
                len(cls.cudagraph_record),
                capture_stats["time"],
                format_mem_bytes(capture_stats["allocated_bytes"]),
                format_mem_bytes(capture_stats["reserved_bytes"]),
            )
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

    # Reset global tracking state
    _CudagraphGlobalRecord.cudagraph_created = False
    _CudagraphGlobalRecord.cudagraph_record = []

    # TODO: Optional?: Force garbage collection to clean up memory
    gc.collect()
    torch.cuda.empty_cache()

    CudaGraphManager.global_mempool = None
    CudaGraphManager.fwd_mempools = None
    CudaGraphManager.bwd_mempool = None


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
            if user_input.data_ptr() != cudagraph_input.data_ptr():
                cudagraph_input.copy_(user_input)

        ctx.runner = runner
        if runner.fp8_enabled or runner.fp4_enabled:
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

            is_first_fp8_module = FP8GlobalStateManager.is_first_fp8_module()
            if is_first_fp8_module:
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(not is_first_microbatch)
            ctx.is_first_fp8_module = is_first_fp8_module

        runner.fwd_graph.replay()

        # if last transformer layer, return a clone of the cudagraph output buffer, as releasing
        # the cudagraph output buffer into the rest of the system may allow it to be corrupted
        if runner.is_last_layer:
            out = tuple(o.clone().detach() for o in runner.fwd_graph_output_surface)
        else:
            out = tuple(o.detach() for o in runner.fwd_graph_output_surface)
        return out

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
            if user_output_grad.data_ptr() != cudagraph_output_grad.data_ptr():
                cudagraph_output_grad.copy_(user_output_grad)

        runner.bwd_graph.replay()
        runner.status = _GraphStatus.FWD_READY

        # Update FP8/FP4 scale factors if needed
        if (runner.fp8_enabled or runner.fp4_enabled) and ctx.is_first_fp8_module:
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # If using gradient_accumulation_fusion, whenever `main_grad` is calculated
        # the `grad_added_to_main_grad` attribute is expected to set. However when using
        # cudagraphs this doesn't occur so we emulate this behavior here.
        for param, grad_added in runner.groundtruth_grad_added_to_main_grad.items():
            param.grad_added_to_main_grad = grad_added

        grads, is_dummy_grad = runner.get_input_grads_with_dummy_flags()
        if runner.is_first_layer:
            output_grads = tuple(
                b.clone().detach() if not (b is None or dummy) else b
                for dummy, b in zip(is_dummy_grad, grads)
            )
        else:
            output_grads = tuple(
                b.detach() if not (b is None or dummy) else b
                for dummy, b in zip(is_dummy_grad, grads)
            )
        return None, None, *output_grads


class _CudaGraphRunner(torch.nn.Module):
    """Represents the execution of a cudagraphed module for a single microbatch.
    If there are multiple outstanding microbatches per module, such as for pipeline parallelism,
    CudaGraphManager automatically creates multiple _CudaGraphRunners per module."""

    def __init__(
        self,
        base_module: MegatronModule,
        fwd_mempool: int,
        bwd_mempool: int,
        fwd_graph_input_args: List[Any],
        fwd_graph_input_kwargs: Dict[str, Any],
        share_cudagraph_io_buffers=None,
    ):
        """Creates a _CudaGraphRunner, which holds a single pair of fwd and bwd cudagraphs, which
        are not created until this runner records its graph creation into
        '_CudagraphGlobalRecord', and 'create_cudagraphs()' is called. share_cudagraph_io_buffers
        is a boolean flag to indicate whether to reuse the cudagraph input and output buffers for
        transformer layer specific optimizations that reduce memory usage and tensor copies."""

        super().__init__()

        self.base_module = base_module
        self.fwd_mempool = fwd_mempool
        self.bwd_mempool = bwd_mempool

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
        self.num_warmup_steps = 2
        if isinstance(self.base_module.config, TransformerConfig):
            self.fuse_wgrad_accumulation = self.base_module.config.gradient_accumulation_fusion
            self.backward_retain_grad = self.base_module.config.cuda_graph_retain_backward_graph
            self.fp8_enabled = self.base_module.config.fp8 is not None
            self.fp4_enabled = self.base_module.config.fp4 is not None
            self.deallocate_pipeline_outputs = self.base_module.config.deallocate_pipeline_outputs
            self.num_warmup_steps = self.base_module.config.cuda_graph_warmup_steps

            if self.fp8_enabled:
                self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

            if self.fp4_enabled:
                from megatron.core.fp4_utils import get_fp4_recipe  # to avoid circular import

                self.fp4_recipe = get_fp4_recipe(self.base_module.config)
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

        # Decide whether to reuse the input and output buffer, and if so,
        # whether this layer is the first layer (which needs an input buffer)
        # or the last layer (which needs an output buffer)

        self.is_transformer_decoder_layer = _determine_if_transformer_decoder_layer(base_module)
        self.reuse_input_output_buffer = (
            share_cudagraph_io_buffers and self.is_transformer_decoder_layer
        )
        if self.reuse_input_output_buffer:
            self.is_first_layer, self.is_last_layer = (
                _determine_if_first_last_layer_of_this_vp_chunk(base_module)
            )
        else:
            self.is_first_layer, self.is_last_layer = True, True

    def __str__(self):
        return "%s; hid %s" % (
            self.base_module.__class__.__name__,
            tuple(self.fwd_graph_input_kwarg_metas["hidden_states"].shape),
        )

    def get_fp8_context(self):
        """Return a new fp8 context in cudagraph mode."""
        from megatron.core.fp8_utils import get_fp8_context  # to avoid circular import

        return get_fp8_context(self.base_module.config, self.base_module.layer_number - 1)

    def get_fp4_context(self):
        """Return a new fp4 context in cudagraph mode."""
        from megatron.core.fp4_utils import get_fp4_context  # to avoid circular import

        return get_fp4_context(self.base_module.config, self.base_module.layer_number - 1)

    def get_quantization_context(self):
        """Return appropriate quantization context (FP8 or FP4) in cudagraph mode."""
        if self.fp8_enabled:
            return self.get_fp8_context()
        elif self.fp4_enabled:
            return self.get_fp4_context()
        else:
            return nullcontext()

    def create_fwd_graph(self, args, kwargs, clone_inputs=True):
        """Create a fwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        # Freeze GC, to speed up capture time ~15-20x.
        freeze_gc = os.getenv("CUDA_GRAPH_CAPTURE_FREEZE_GC") != "0"
        if freeze_gc:
            gc.freeze()

        # save grads and other variables that may be affected by graph warmup
        if self.training and torch.is_grad_enabled():
            save_main_grads = [
                param.main_grad.clone()
                for param in self.base_module.parameters()
                if hasattr(param, 'main_grad')
            ]

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

        if clone_inputs:
            args, kwargs = self.zero_out_tensors(args, kwargs)

        input_tensors = self.get_tensors(args, kwargs)
        self.fwd_graph_input_surface = input_tensors + tuple(self.base_module.parameters())

        self.fwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.fwd_graph.register_generator_state(state)

        # warmup again as case graph capture mode may execute a different codepath
        for _ in range(self.num_warmup_steps):
            with self.get_quantization_context():
                outputs = self.base_module.forward(*args, **kwargs)
            if self.training and torch.is_grad_enabled():
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                outputs = self.get_tensors(outputs)
                grad_inputs = torch.autograd.grad(
                    outputs=tuple(o for o in outputs if o.requires_grad),
                    inputs=tuple(i for i in self.fwd_graph_input_surface if i.requires_grad),
                    grad_outputs=tuple(
                        torch.zeros_like(o) if o.requires_grad else None for o in outputs
                    ),
                    only_inputs=True,
                    allow_unused=True,
                )

        with self.get_quantization_context():
            torch.cuda.synchronize()
            with torch.cuda.graph(
                self.fwd_graph, pool=self.fwd_mempool, capture_error_mode="thread_local"
            ):
                outputs = self.base_module.forward(*args, **kwargs)

        # save cudagraph output buffer
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        self.fwd_graph_outputs = outputs
        self.fwd_graph_output_surface = self.get_tensors(outputs)

        if self.training and torch.is_grad_enabled():
            assert (
                len(self.fwd_graph_output_surface) > 0
            ), """Tried graphing a moudule that returned no tensors in training mode,
                however the graphed module must output at least one tensor,
                so that a corresponding backward node may be registered in the autograd graph."""

            # restore cached grads
            for param in self.base_module.parameters():
                if hasattr(param, 'main_grad'):
                    saved_grad = save_main_grads.pop(0)
                    assert (
                        param.main_grad.shape == saved_grad.shape
                    ), "Error restoring grads while cudagraphing!"
                    param.main_grad.copy_(saved_grad)

        if self.fp8_enabled or self.fp4_enabled:
            restore_fp8_tensors([self.base_module], saved_fp8_tensors)

        # Unfreeze GC.
        if freeze_gc:
            gc.unfreeze()

            # gc.collect() drops references to unreachable tensors created during capture,
            # returning their storage to the allocator to avoid a slowdown during replay. However,
            # it forces expensive global garbage collection, so must be done only on the last layer
            # per-device to avoid slowing down graph creation.
            if self.is_last_layer:
                gc.collect()

    def create_bwd_graph(self, static_grad_outputs=None):
        """Create a bwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        # Freeze GC, to speed up capture time ~15-20x.
        freeze_gc = os.getenv("CUDA_GRAPH_CAPTURE_FREEZE_GC") != "0"
        if freeze_gc:
            gc.freeze()

        self.bwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.bwd_graph.register_generator_state(state)

        if static_grad_outputs is None:
            static_grad_outputs = tuple(
                torch.zeros_like(o) if o.requires_grad else None
                for o in self.fwd_graph_output_surface
            )
        else:
            # canoncalize as tuple
            if torch.is_tensor(static_grad_outputs):
                static_grad_outputs = (static_grad_outputs,)

        torch.cuda.synchronize()
        with torch.cuda.graph(
            self.bwd_graph, pool=self.bwd_mempool, capture_error_mode="thread_local"
        ):
            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in self.fwd_graph_output_surface if o.requires_grad),
                inputs=tuple(i for i in self.fwd_graph_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                retain_graph=self.backward_retain_grad,
                only_inputs=True,
                allow_unused=True,
            )

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs
        # that don't require grad. I couldn't think of a one-liner for this pattern.
        static_grad_inputs = []
        grad_idx = 0
        for arg in self.fwd_graph_input_surface:
            has_wgrad_fusion = self.fuse_wgrad_accumulation and getattr(
                arg, "grad_added_to_main_grad", False
            )
            if arg.requires_grad:
                if has_wgrad_fusion:
                    static_grad_inputs.append(None)
                else:
                    static_grad_inputs.append(grad_inputs[grad_idx])
                grad_idx += 1
            else:
                static_grad_inputs.append(None)

        self.groundtruth_grad_added_to_main_grad = {}
        if self.fuse_wgrad_accumulation:
            for param in self.base_module.parameters():
                if hasattr(param, "grad_added_to_main_grad"):
                    self.groundtruth_grad_added_to_main_grad[param] = param.grad_added_to_main_grad

        self.static_grad_outputs = static_grad_outputs
        self.static_grad_inputs = static_grad_inputs

        # Unfreeze GC.
        if freeze_gc:
            gc.unfreeze()

            if self.is_first_layer:
                gc.collect()

    def get_input_grads_with_dummy_flags(self):
        """Get the inputs grads that are returned by the bwd cudagraph call. If using grad accum
        fusion, wgrads have already been accumulated, so return dummy wgrads."""

        is_dummy_grad = [False] * len(self.static_grad_inputs)
        if not self.fuse_wgrad_accumulation:
            return self.static_grad_inputs, is_dummy_grad
        else:
            num_dgrads = len(self.static_grad_inputs) - len(list(self.base_module.parameters()))
            dgrads = self.static_grad_inputs[:num_dgrads]
            wgrads = self.static_grad_inputs[num_dgrads:]

            wgrads_with_placeholders = []
            is_dummy_grad = [False] * len(dgrads)
            for idx, param in enumerate(self.base_module.parameters()):
                wgrad_is_dummy = getattr(param, "grad_added_to_main_grad", False)
                if wgrad_is_dummy:
                    if getattr(param, "zero_out_wgrad", False):
                        wgrad = torch.zeros(
                            param.main_grad.shape,
                            dtype=param.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                    else:
                        wgrad = torch.empty(
                            param.main_grad.shape,
                            dtype=param.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                else:
                    wgrad = wgrads[idx]
                wgrads_with_placeholders.append(wgrad)
                is_dummy_grad.append(wgrad_is_dummy)
            return tuple(dgrads + wgrads_with_placeholders), is_dummy_grad

    def record_graph_capture(self, args, kwargs):
        """Records the data needed to create this runner's forward cudagraph.
        The first pass records a graph and appends the runner to _CudagraphGlobalRecord.
        The actual cudagraph will be created when 'create_cudagraphs()` is called. Subsequent
        passes should replay the graph."""

        if not self.fwd_graph_recorded:
            logger.debug(f"Recording forward graph creation...")
            if self.is_transformer_decoder_layer and not self.is_first_layer:
                # transformer layers hidden_states are already saved as the output of the previous
                # layer's cudagraph so avoid saving again
                kwargs_copy = dict(kwargs)
                kwargs_copy['hidden_states'] = None
                _CudagraphGlobalRecord.record_fwd_graph(self, args, kwargs_copy)
            else:
                _CudagraphGlobalRecord.record_fwd_graph(self, args, kwargs)

            self.fwd_graph_recorded = True

        # Run the forward pass as normal in eager mode.
        out = super(MegatronModule, self.base_module).__call__(*args, **kwargs)

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

        # autograd nodes return inputs as views, so clone the tensor as returning views may cause
        # issues, for instance with pipeline parallelism
        return tuple(o.clone() if torch.is_tensor(o) else o for o in out)

    def replay_graph_capture(self, is_first_microbatch, args, kwargs):
        """Replay the fwd cuda graph with autograd."""

        # Arguments passed to a cudagraph for replay must match the args in the captured graph.
        #  Tensor arguments need to have the same shape, dtype, and device location.
        #  All other arguments must have the exact same memory addresses for graph safety.
        mismatch_errors = self.get_mismatch_errors(args, kwargs)
        if mismatch_errors:
            error_msg = "CUDA graph argument mismatch:\n" + "\n".join(mismatch_errors)
            raise AssertionError(error_msg)

        inp_tensors = self.get_tensors(args, kwargs)
        func_args = inp_tensors + tuple(self.parameters())
        out = _CudagraphReplayNode.apply(self, is_first_microbatch, *func_args)
        out = list(out)

        if torch.is_tensor(self.fwd_graph_outputs):
            self.fwd_graph_outputs = [self.fwd_graph_outputs]

        return tuple(out.pop(0) if torch.is_tensor(o) else o for o in self.fwd_graph_outputs)

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
                add_error(f"Type mismatch at {context}: {val.type} vs {ref.type}")
                return False

            if ref.type == torch.Tensor or issubclass(ref.type, torch.Tensor):
                mismatches = []
                if val.shape != ref.shape:
                    mismatches.append(f"expected shape {val.shape} vs. {ref.shape}")
                if val.dtype != ref.dtype:
                    mismatches.append(f"expected dtype {val.dtype} vs. {ref.dtype}")
                if val.device != ref.device:
                    mismatches.append(f"expected device {val.device} vs. {ref.device}")
                if mismatches:
                    add_error(f"Tensor mismatch at {context}: {', '.join(mismatches)}")

            elif is_dataclass(ref.value):
                for field in fields(ref.value):
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

    def zero_out_tensors(self, args, kwargs=None):
        """Replace all tensors inside arg, kwargs with zeroed copies."""

        def clone_tensor(ten):
            cloned = torch.zeros_like(ten)
            cloned.requires_grad = ten.requires_grad
            return cloned

        def process_arg(arg):
            _check_supported_type(ArgMetadata(arg))
            if torch.is_tensor(arg):
                return clone_tensor(arg)
            elif is_dataclass(arg):
                for field in fields(arg):
                    attr = getattr(arg, field.name)
                    if torch.is_tensor(attr):
                        setattr(arg, field.name, clone_tensor(attr))
            return arg

        args_replaced = []
        for arg in args:
            args_replaced.append(process_arg(arg))
        if kwargs is None:
            return args_replaced

        kwargs_replaced = {}
        for k, v in kwargs.items():
            kwargs_replaced[k] = process_arg(v)

        return args_replaced, kwargs_replaced

    @classmethod
    def get_tensors(cls, args, kwargs=None):
        """Filter and flatten all tensors from args and kwargs."""

        def extract_tensors(arg):
            _check_supported_type(ArgMetadata(arg))
            if torch.is_tensor(arg):
                return [arg]
            elif is_dataclass(arg):
                tens = []
                for field in fields(arg):
                    attr = getattr(arg, field.name)
                    if torch.is_tensor(attr):
                        tens.append(attr)
                return tens
            else:
                return []

        tens = []
        args, _ = tree_flatten(args)
        for a in args:
            tens.extend(extract_tensors(a))

        if kwargs is not None:
            kwargs, _ = tree_flatten(kwargs)
            for k in kwargs:
                tens.extend(extract_tensors(k))

        return tuple(tens)


class CudaGraphManager(torch.nn.Module):
    """Creates and runs cudagraphs for a megatron module"""

    """A global mempool for when 'cuda_graph_use_single_mempool' is used."""
    global_mempool = None

    """Forward pass mempools, used with cudagraph reuse mode."""
    fwd_mempools = None

    """Backward pass mempool, used with cudagraph reuse mode."""
    bwd_mempool = None

    def __init__(
        self,
        config: TransformerConfig,
        share_cudagraph_io_buffers: bool = True,
        vp_stage: Optional[int] = None,
    ):
        super().__init__()
        """Creates a CudaGraphManager to manage CUDA graphs for a Megatron module.

        Args:
            config: TransformerConfig object containing CUDA graph settings for memory
                pooling, graph retention, gradient accumulation, FP8/FP4, and warmup steps.
            share_cudagraph_io_buffers (bool, optional): (DEPRECATED, will be replaced by
                config.cuda_graph_share_io_buffers) If None (default) or True, enables
                buffer reuse optimizations for transformer and mamba layers. If False,
                disables buffer reuse.
        """
        rng_tracker = get_cuda_rng_tracker()
        self.share_cudagraph_io_buffers = share_cudagraph_io_buffers
        self.vp_stage = vp_stage

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
        assert "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""), (
            "expandable_segments:True may not be safe when using CUDA Graphs, and may result in"
            "a crash due to illegal memory access or other undefined behaviour."
        )

        self.cudagraph_runners = []
        self.inference_cudagraphs_lookup_table = defaultdict(lambda: None)
        self.is_first_microbatch = False

        # Without pipeline parallelism, microbatches execute one at a time.
        # Therefore modules will always execute in the same order, so cudagraphs
        # can both be reused and share a single mempool.
        if parallel_state.get_pipeline_model_parallel_world_size() == 1:
            self.reuse_cudagraphs = True
            self.use_single_mempool = True
        else:
            if config.cuda_graph_use_single_mempool:
                self.reuse_cudagraphs = False
                self.use_single_mempool = True
            else:
                self.reuse_cudagraphs = True
                self.use_single_mempool = False

        # Mempools are static so that multiple cudagraph managers may share the same mempool
        if self.use_single_mempool:
            if CudaGraphManager.global_mempool is None:
                CudaGraphManager.global_mempool = torch.cuda.graph_pool_handle()
        else:
            # All cudagraphs in the same microbatch use the same mempool. For pipeline parallelism,
            # additonally all bwd passes share the same mempool
            if CudaGraphManager.fwd_mempools is None:
                CudaGraphManager.fwd_mempools = defaultdict(
                    lambda: defaultdict(torch.cuda.graph_pool_handle)
                )
                CudaGraphManager.bwd_mempool = torch.cuda.graph_pool_handle()

        # Cudagraph stream capture requires no operations on the default stream prior to the
        # capture, so change to a side stream.
        self.stream = torch.cuda.current_stream()
        torch.cuda.set_stream(torch.cuda.Stream())

    def set_is_first_microbatch(self, is_first_microbatch: bool):
        """Update the is_first_microbatch flag for weight caching.

        Args:
            is_first_microbatch (bool): Whether this is the first microbatch in the step.
        """
        self.is_first_microbatch = is_first_microbatch

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

    def get_cudagraph_runner(self, megatron_module, args, kwargs):
        '''Returns a valid cudagraph runner for the current forward call.
        For single mempool mode, we create a cudagraph for each call, if the module is called
        multiple times per step, for instance in the case of pipeline parallelism.
        The cudagraph corresponding to this call is the first element of 'self.cudagraph_runners'.
        We iterate through the list by 1 for each call, and the number of calls is equal to the
        length of 'self.cudagraph_runners'.
        Otherwise, we assign a mempool per microbatch, which allows cudagraphs to be reused
        over different microbatches by tracking their respective fwd and bwd passes.'''

        if self.use_single_mempool:
            fwd_mempool = CudaGraphManager.global_mempool
            bwd_mempool = CudaGraphManager.global_mempool
        else:
            if megatron_module.config.virtual_pipeline_model_parallel_size is not None:
                assert (
                    self.vp_stage is not None
                ), "vp_stage must be passed if virtual pipeline is enabled"
                vpp_rank = self.vp_stage
            else:
                vpp_rank = 0
            fwd_mempool = CudaGraphManager.fwd_mempools[vpp_rank][len(self.cudagraph_runners)]
            bwd_mempool = CudaGraphManager.bwd_mempool

        if self.reuse_cudagraphs:
            is_inference_mode = 'inference_context' in kwargs.keys() and kwargs['inference_context']
            if is_inference_mode:
                batch_size = kwargs['hidden_states'].shape[0]
                is_decode_only = kwargs["inference_context"].is_decode_only()
                # Attempt to retrieve the corresponding runner from the lookup table.
                # The table is keyed on (batch_size, is_decode_only).
                # It returns None if no match is found, in which case a new runner is created
                # and cached in the lookup table.
                runner = self.inference_cudagraphs_lookup_table[(batch_size, is_decode_only)]
            else:
                # Todo: For training, we could also cache runners based on input shape.
                runner = next(
                    (
                        r
                        for r in self.cudagraph_runners
                        if r.status == _GraphStatus.FWD_READY
                        and not r.get_mismatch_errors(args, kwargs)
                    ),
                    None,
                )

            if runner is None:
                if _CudagraphGlobalRecord.cudagraph_created:
                    assert False
                else:
                    runner = _CudaGraphRunner(
                        megatron_module,
                        fwd_mempool,
                        bwd_mempool,
                        args,
                        kwargs,
                        self.share_cudagraph_io_buffers,
                    )
                    self.cudagraph_runners.append(runner)
                    if is_inference_mode:
                        # Cache the newly created runner in the inference lookup table.
                        self.inference_cudagraphs_lookup_table[(batch_size, is_decode_only)] = (
                            runner
                        )
        else:
            # Create cudagraphs for every microbatch
            if _CudagraphGlobalRecord.cudagraph_created:
                runner = self.cudagraph_runners[0]
                assert runner.status == _GraphStatus.FWD_READY
                self.cudagraph_runners = self.cudagraph_runners[1:] + self.cudagraph_runners[:1]
            else:
                runner = _CudaGraphRunner(
                    megatron_module,
                    fwd_mempool,
                    bwd_mempool,
                    args,
                    kwargs,
                    self.share_cudagraph_io_buffers,
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
        # Set the is_first_microbatch flag on the megatron module if it's the first microbatch
        if self.is_first_microbatch and hasattr(megatron_module, 'set_is_first_microbatch'):
            megatron_module.set_is_first_microbatch()

        if _CudagraphGlobalRecord.cudagraph_created:
            if self.training and torch.is_grad_enabled():
                # param.data_ptr() below is used to trigger any hooks that have attached to the
                # parameter. Specifically, this is trying to trigger the param sync hook for the
                # APEX optimizer, which triggers param syncs by hooking into any param references.
                # However cudagraphs disables this, so we workaround by manually referencing
                # params here. For more information see:
                # https://github.com/NVIDIA/apex/blob/7001836/apex/contrib/optimizers/distributed_fused_adam.py#L885C9
                for param in megatron_module.parameters():
                    param.data_ptr()

                # Trigger Mcore DDP pre-forward hooks
                self.call_ddp_preforward_hook(megatron_module)
                for module in megatron_module.modules():
                    self.call_ddp_preforward_hook(module)

            runner = self.get_cudagraph_runner(megatron_module, args, kwargs)
            out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)
        else:
            if 'inference_context' in kwargs.keys() and kwargs['inference_context']:
                # Inference generation mode creates graphs immediately
                runner = self.get_cudagraph_runner(megatron_module, args, kwargs)
                runner.eval()

                if not runner.fwd_graph_recorded:
                    # Reuse graph input-output buffers for inference
                    local_args, local_kwargs = args, kwargs
                    if runner.reuse_input_output_buffer and not runner.is_first_layer:
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

                    clone_inputs = not (
                        runner.reuse_input_output_buffer and not runner.is_first_layer
                    )
                    runner.create_fwd_graph(local_args, local_kwargs, clone_inputs=clone_inputs)
                    runner.fwd_graph_recorded = True
                    runner.cudagraph_created = True

                    # Record this to the global execution record
                    _CudagraphGlobalRecord.cudagraph_inference_record.append(
                        (runner, "fwd", args, kwargs)
                    )

                # Now replay the graph
                out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)

            elif self.training:
                # Training mode
                runner = self.get_cudagraph_runner(megatron_module, args, kwargs)
                # check if a layer is frozen during training.
                if not torch.is_grad_enabled():
                    # If the layer is frozen, we need to set the runner to eval mode.
                    runner.eval()
                out = runner.record_graph_capture(args, kwargs)
            else:
                # No cudagraphs were found in training mode with grad disabled, so fallback to
                # eager since autograd is needed to correctly trace the backward graph.
                return super(MegatronModule, megatron_module).__call__(*args, **kwargs)

        # If forward only, next replay should be a forward pass as well
        if self.training and torch.is_grad_enabled():
            runner.status = _GraphStatus.BWD_READY
        else:
            runner.status = _GraphStatus.FWD_READY

        return out


# The following functions are for capturing CUDA Graphs using TE make_graphed_callables().
def _layer_is_graphable(layer, config):
    """
    Check if a layer is graphable.
    """

    # import modules here to avoid a circular import
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.mlp import MLP
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if isinstance(layer, MambaLayer) and 'mamba' in config.cuda_graph_scope:
        # mamba layer.
        return True
    if isinstance(layer, TransformerLayer):
        if 'attn' in config.cuda_graph_scope and not (
            isinstance(layer.self_attention, IdentityOp)
            and isinstance(layer.cross_attention, IdentityOp)
        ):
            # attn layer.
            return True
        if (
            'moe' in config.cuda_graph_scope
            or 'moe_router' in config.cuda_graph_scope
            or 'moe_preprocess' in config.cuda_graph_scope
        ) and isinstance(layer.mlp, MoELayer):
            # moe layer.
            return True
        if 'mlp' in config.cuda_graph_scope and isinstance(layer.mlp, MLP):
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
        assert "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""), (
            "expandable_segments:True may not be safe when using CUDA Graphs, and may result in"
            "a crash due to illegal memory access or other undefined behaviour."
        )
        assert "full_iteration" not in config.cuda_graph_scope, (
            "full_iteration cuda graph is not supported for cuda_graph_impl=transformer_engine. "
            "Please use cuda_graph_impl=local instead."
        )

        self.model = model
        self.config = config
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.optimizers = optimizers
        self.num_model_chunks = len(model)

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
                    layer = chunk_with_decoder.mtp.layers[layer_number].transformer_layer
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

    def _get_cuda_graph_input_data(self):
        """
        Create the CUDA Graph capturing input data.
        The data is organized per-chunk per-microbatch per-layer.
        """

        rotary_pos_emb_cache = {}

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

        # Generate sample arguments and keyword arguments for capturing.
        sample_args = []
        sample_kwargs = []
        for chunk_number, chunk_with_decoder in enumerate(self.chunks_with_decoder):
            if chunk_with_decoder is None:
                continue
            layers = self.callables_per_chunk[chunk_number]
            for _ in range(get_num_microbatches()):
                for layer in layers:
                    static_inputs = layer.get_layer_static_inputs(
                        self.seq_length, self.micro_batch_size
                    )

                    from megatron.core.transformer.identity_op import IdentityOp
                    from megatron.core.transformer.transformer_layer import TransformerLayer

                    contains_self_attn = (
                        isinstance(layer, TransformerLayer)
                        and not isinstance(layer.self_attention, IdentityOp)
                        and 'attn' in self.config.cuda_graph_scope
                    )
                    if is_te_min_version("1.10.0"):
                        # te.make_graphed_callables() accepts keyword arguments since 1.10.0.
                        hidden_states = static_inputs.pop("hidden_states")
                        sample_args.append((hidden_states,))
                        if contains_self_attn:
                            rotary_pos_emb = get_rotary_pos_emb(chunk_with_decoder, hidden_states)
                            if rotary_pos_emb is not None:
                                static_inputs["rotary_pos_emb"] = rotary_pos_emb
                        sample_kwargs.append(static_inputs)
                    elif contains_self_attn:
                        sample_args.append(
                            (
                                static_inputs.pop("hidden_states"),
                                static_inputs.pop("attention_mask"),
                            )
                        )
                    else:
                        sample_args.append((static_inputs.pop("hidden_states"),))

        # Get the PP and VPP scheduling order.
        from megatron.core.pipeline_parallel.schedules import (
            convert_schedule_table_to_order,
            get_pp_rank_microbatches,
            get_schedule_table,
        )

        _, _, num_warmup_microbatches, _ = get_pp_rank_microbatches(
            get_num_microbatches(),
            self.num_model_chunks,
            self.config.microbatch_group_size_per_vp_stage,
            False,
        )
        schedule_table = get_schedule_table(
            get_num_microbatches(),
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

        def get_make_graphed_callables_kwargs():
            kwargs = {'num_warmup_iters': 11, 'allow_unused_input': True, '_order': order}

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
        torch.distributed.barrier()
        gc.collect()
        torch.cuda.empty_cache()

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
        gc.collect()
        torch.cuda.empty_cache()

    def create_cudagraphs(self):
        """
        Capture CUDA Graphs per TransformerLayer per microbatch.
        """
        start_time = self._start_capturing()

        # Prepare CUDA Graph capturing input data and call `make_graphed_callables`.
        sample_args, kwargs = self._get_cuda_graph_input_data()
        graphs = make_graphed_callables(tuple(self.flattened_callables), sample_args, **kwargs)

        # Push the captured graphs to the corresponding TransformerBlock.
        num_layers_accumulated = 0
        for layers in self.callables_per_chunk:
            for layer_number, layer in enumerate(layers):
                layer.cuda_graphs = []
                for batch_number in range(get_num_microbatches()):
                    layer.cuda_graphs.append(
                        graphs[
                            num_layers_accumulated * get_num_microbatches()
                            + batch_number * len(layers)
                            + layer_number
                        ]
                    )
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
