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
from typing import Any, Dict, List

import torch
from torch.utils._pytree import tree_flatten

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import (
    CudaRNGStatesTracker,
    get_all_rng_states,
    get_cuda_rng_tracker,
)
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version

try:
    import transformer_engine as te  # pylint: disable=unused-import
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
    from transformer_engine.pytorch.graph import restore_fp8_tensors, save_fp8_tensors
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
    vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
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
                print(f"{g_idx}/{len(cls.cudagraph_record)}. {progress_str}")

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
        print(
            "> built %d cuda graph(s) in %.2f sec, with total memory usage: "
            "allocated %s, reserved %s."
            % (
                len(cls.cudagraph_record),
                time_end - time_start,
                format_mem_bytes(
                    mem_stats_end["allocated_bytes.all.current"]
                    - mem_stats_start["allocated_bytes.all.current"]
                ),
                format_mem_bytes(
                    mem_stats_end["reserved_bytes.all.current"]
                    - mem_stats_start["reserved_bytes.all.current"]
                ),
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


def create_cudagraphs():
    """Should be called at the end of each schedule function,
    (e.g. forward_backward_pipelining_with_interleaving) in
    `megatron.core.pipeline_parallel.schedules.py`. During the first step, _CudaGraphRunners
    populate _CudagraphGlobalRecord with the global order in which cudagraphs should be created.
    At the end for the first step, this function calls each runner's `create_fwd_graph` and
    `create_bwd_graph` in the order recorded in _CudagraphGlobalRecord, which allows cudagraphs
    to be created in execution order, which allows multiple cudagraphs to share a single
    memory pool, minimizing cudagraph memory usage."""

    _CudagraphGlobalRecord.create_cudagraphs()


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
    cudagraph io and fp8 if used."""

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
        if runner.fp8_enabled:
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

        # Update FP8 scale factors if needed
        if runner.fp8_enabled and ctx.is_first_fp8_module:
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
        self.deallocate_pipeline_outputs = False
        self.num_warmup_steps = 2
        if isinstance(self.base_module.config, TransformerConfig):
            self.fuse_wgrad_accumulation = self.base_module.config.gradient_accumulation_fusion
            self.backward_retain_grad = self.base_module.config.cuda_graph_retain_backward_graph
            self.fp8_enabled = self.base_module.config.fp8 is not None
            self.deallocate_pipeline_outputs = self.base_module.config.deallocate_pipeline_outputs
            self.num_warmup_steps = self.base_module.config.cuda_graph_warmup_steps

            if self.fp8_enabled:
                self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
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

        if self.fp8_enabled:
            return fp8_autocast(
                enabled=True, calibrating=False, fp8_recipe=self.fp8_recipe, _graph=True
            )
        return nullcontext()

    def run_module_forward(self, args, kwargs, *, graph=None, pool=None):
        """Run module forward, using given graph and memory pool."""

        inference_context = kwargs.get("inference_context", None)

        # Initialize inference context.
        if inference_context and inference_context.is_dynamic_batching():
            num_warmup_requests = kwargs["hidden_states"].size(0)
            inference_context.initialize_attention_state(num_warmup_requests=num_warmup_requests)

        context = (
            torch.cuda.graph(cuda_graph=graph, pool=pool) if graph is not None else nullcontext()
        )

        # Module forward.
        with context:
            outputs = self.base_module.forward(*args, **kwargs)

        # Reset inference context.
        if inference_context and inference_context.is_dynamic_batching():
            inference_context.reset()

        return outputs

    def create_fwd_graph(self, args, kwargs, clone_inputs=True):
        """Create a fwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        # save grads and other variables that may be affected by graph warmup
        if self.training and torch.is_grad_enabled():
            save_main_grads = [
                param.main_grad.clone()
                for param in self.base_module.parameters()
                if hasattr(param, 'main_grad')
            ]

        if self.fp8_enabled:
            if is_te_min_version("1.13.0"):
                saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp8_recipe)
            else:
                saved_fp8_tensors = save_fp8_tensors(
                    [self.base_module], self.fp8_recipe.amax_history_len
                )

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
            with self.get_fp8_context():
                outputs = self.run_module_forward(args, kwargs)
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

        with self.get_fp8_context():
            torch.cuda.synchronize()
            outputs = self.run_module_forward(
                args, kwargs, graph=self.fwd_graph, pool=self.fwd_mempool
            )

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

        if self.fp8_enabled:
            restore_fp8_tensors([self.base_module], saved_fp8_tensors)

    def create_bwd_graph(self, static_grad_outputs=None):
        """Create a bwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

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
        with torch.cuda.graph(self.bwd_graph, pool=self.bwd_mempool):
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
            if not self.is_first_layer:
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

    def __init__(self, config: TransformerConfig, share_cudagraph_io_buffers: bool = True):
        super().__init__()
        """Creates a CudaGraphManager to manage CUDA graphs for a Megatron module.

        Args:
            config: TransformerConfig object containing CUDA graph settings for memory
                pooling, graph retention, gradient accumulation, FP8, and warmup steps.
            share_cudagraph_io_buffers (bool, optional): (DEPRECATED, will be replaced by
                config.cuda_graph_share_io_buffers) If None (default) or True, enables 
                buffer reuse optimizations for transformer and mamba layers. If False,
                disables buffer reuse.
        """
        rng_tracker = get_cuda_rng_tracker()
        self.share_cudagraph_io_buffers = share_cudagraph_io_buffers

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

        if config.enable_cuda_graph or config.external_cuda_graph:
            assert "expandable_segments:True" not in os.getenv("PYTORCH_CUDA_ALLOC_CONF", ""), (
                "expandable_segments:True may not be safe when using CUDA Graphs, and may result in"
                "a crash due to illegal memory access or other undefined behaviour."
            )

        self.cudagraph_runners = []
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
            vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
            vpp_rank = 0 if vpp_rank is None else vpp_rank
            fwd_mempool = CudaGraphManager.fwd_mempools[vpp_rank][len(self.cudagraph_runners)]
            bwd_mempool = CudaGraphManager.bwd_mempool

        if self.reuse_cudagraphs:
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
                # Inference generation mode
                runner = self.get_cudagraph_runner(megatron_module, args, kwargs)
                runner.eval()
                out = runner.record_graph_capture(args, kwargs)
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
