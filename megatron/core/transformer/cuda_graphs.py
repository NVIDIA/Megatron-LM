# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import copy
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
        self.is_tensor = isinstance(arg, torch.Tensor)
        if self.is_tensor:
            self.shape = arg.shape
            self.dtype = arg.dtype
            self.device = arg.device
        else:
            self.value = arg

    def __eq__(self, other):
        if not isinstance(other, ArgMetadata):
            return NotImplemented
        if self.is_tensor != other.is_tensor:
            return False
        if self.is_tensor:
            # Pointer equality is not checked since input tensors are copied into static buffers.
            return (
                self.shape == other.shape
                and self.dtype == other.dtype
                and self.device == other.device
            )
        else:
            return self.value == other.value


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

    cudagraph_record = []  # global record of all graphs, in execution record


def create_cudagraphs():
    """TODO(helenn): Necessary for backward compatibility with MCore 0.14. To be removed."""
    return


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

        prev_bwd_hidden_state_inputgrad = None
        if len(_CudagraphGlobalRecord.cudagraph_record) > 0 and runner.reuse_input_output_buffer:
            if not runner.is_last_layer:
                previous_runner = next(
                    r
                    for r in _CudagraphGlobalRecord.cudagraph_record[::-1]
                    if r[0].base_module.layer_number == runner.base_module.layer_number + 1
                    and r[0].base_module.current_microbatch == runner.base_module.current_microbatch
                    and hasattr(r[0], "static_grad_inputs")
                    and grads.shape == r[0].static_grad_inputs[0].shape
                )
                prev_bwd_hidden_state_inputgrad = previous_runner[0].static_grad_inputs[0]

        if not runner.bwd_graph_recorded:
            runner.create_bwd_graph(prev_bwd_hidden_state_inputgrad)
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
        """Creates a _CudaGraphRunner, which holds a single pair of fwd and bwd cudagraphs.
        share_cudagraph_io_buffers is a boolean flag to indicate whether to reuse the cudagraph
        input and output buffers for optimizations that reduce memory usage and tensor copies when
        the input and output buffer shapes match and can be reallocated between layers."""

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
        from megatron.core.fp8_utils import get_fp8_context  # to avoid circular import

        return get_fp8_context(self.base_module.config, self.base_module.layer_number - 1)

    def run_module_forward(self, args, kwargs, *, graph=None, pool=None):
        """Run module forward, using given graph and memory pool."""

        inference_context = kwargs.get("inference_context", None)

        # Initialize inference context.
        if inference_context and inference_context.is_dynamic_batching():
            num_warmup_requests = kwargs["hidden_states"].size(0)
            inference_context.initialize_attention_state(num_warmup_requests=num_warmup_requests)

        context = (
            torch.cuda.graph(cuda_graph=graph, pool=pool, capture_error_mode="thread_local")
            if graph is not None
            else nullcontext()
        )

        # Module forward.
        with context:
            outputs = self.base_module.forward(*args, **kwargs)

        # Reset inference context.
        if inference_context and inference_context.is_dynamic_batching():
            inference_context.reset()

        return outputs

    def create_fwd_graph(self, args, kwargs, clone_inputs=True):
        """Create a fwd cudagraph for this runner."""

        has_te_modules = False
        if HAVE_TE_GRAPHS:
            base_module = self.base_module
            has_te_modules = has_te_modules or any(
                [isinstance(m, TransformerEngineBaseModule) for m in base_module.modules()]
            )

        if clone_inputs:
            args, kwargs = self.zero_out_tensors(args, kwargs)

        # Check whether optimize_transformer_layer_graph_buffers is enabled, so the output buffer of
        #  the previous forward pass can be used as the input buffer for the current layer
        if self.reuse_input_output_buffer and not self.is_first_layer:
            previous_runner = next(
                r
                for r in _CudagraphGlobalRecord.cudagraph_record
                if r[0].base_module.layer_number == self.base_module.layer_number - 1
                and ArgMetadata(r[3]['hidden_states']) == ArgMetadata(kwargs['hidden_states'])
            )
            kwargs['hidden_states'] = previous_runner[0].fwd_graph_outputs[0]

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

        gc.collect()
        torch.cuda.empty_cache()

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()

        with self.get_fp8_context():
            torch.cuda.synchronize()
            with torch.cuda.graph(
                self.fwd_graph, pool=self.fwd_mempool, capture_error_mode="thread_local"
            ):
                outputs = self.base_module.forward(*args, **kwargs)

        # TODO(helenn): This is needed for backward compatibility in MCore <= 0.13.0. Remove.
        _CudagraphGlobalRecord.cudagraph_created = True

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

        _set_capture_end()
        if has_te_modules:
            te_set_capture_end()

        # Unfreeze GC.
        if freeze_gc:
            gc.unfreeze()

    def create_bwd_graph(self, static_grad_outputs=None):
        """Create a bwd cudagraph for this runner."""

        has_te_modules = False
        if HAVE_TE_GRAPHS:
            base_module = self.base_module
            has_te_modules = has_te_modules or any(
                [isinstance(m, TransformerEngineBaseModule) for m in base_module.modules()]
            )

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()

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
            # canonicalize as tuple
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

        _set_capture_end()
        if has_te_modules:
            te_set_capture_end()

        # Unfreeze GC.
        if freeze_gc:
            gc.unfreeze()

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
                arg = copy.deepcopy(arg)
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
                pooling, graph retention, gradient accumulation, FP8, and warmup steps.
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
                out, runner = self.create_runner(
                    args, bwd_mempool, fwd_mempool, kwargs, megatron_module
                )

                # autograd nodes return inputs as views, so clone the tensor as returning views may
                # cause issues, for instance with pipeline parallelism
                return runner, tuple(o.clone() if torch.is_tensor(o) else o for o in out)
            return runner, None
        else:
            # Create cudagraphs for every microbatch.
            #  In this case, a single memory pool is shared by all graphs.
            #  For each training step (one full global batch), every CUDA graph that was captured
            #  during that step must be replayed only a single time within a step. If you try to
            #  replay the same graph multiple times (e.g., for several microbatches in a step),
            #  you could run into memory errors or data corruption because all the graphs share the
            #  single memory pool, and memory allocations/regions are only guaranteed to be valid
            #  for the original sequence/order of graph replays. Extra replays or parallel replay
            #  can invalidate the memory layout.

            # Check if a runner already exists for this microbatch.
            if megatron_module.current_microbatch >= len(self.cudagraph_runners):
                out, runner = self.create_runner(
                    args, bwd_mempool, fwd_mempool, kwargs, megatron_module
                )
            else:
                # Replay with existing runner for this microbatch.
                runner = self.cudagraph_runners[megatron_module.current_microbatch]
                out = (None,)

            return runner, tuple(o.clone() if torch.is_tensor(o) else o for o in out)

    def create_runner(self, args, bwd_mempool, fwd_mempool, kwargs, megatron_module):
        """Creates a new cudagraph runner, registers a hook for the backward pass, calls
        `create_fwd_graph`, then adds runner to the _CudagraphGlobalRecord.cudagraph_record."""

        runner = _CudaGraphRunner(
            megatron_module, fwd_mempool, bwd_mempool, args, kwargs, self.share_cudagraph_io_buffers
        )

        # Run the forward pass as normal in eager mode.
        out = super(MegatronModule, runner.base_module).__call__(*args, **kwargs)
        if type(out) != tuple:
            out = (out,)

        # Register a noop autograd node that toggles `self.graph_status` in the bwd pass,
        # which tracks when the runner completes its bwd pass. If it's the first bwd
        # encountered by this runner, record it to _CudagraphGlobalRecord.
        # We record the noop autograd node to the first output tensor. This is sufficient
        # for TransformerLayer and MambaLayer as their output is just the hidden_states.
        out = tuple(
            [
                (_CudagraphRecordNode.apply(runner, o) if torch.is_tensor(o) and i == 0 else o)
                for i, o in enumerate(out)
            ]
        )

        # Track microbatch for pipeline parallelism
        if hasattr(megatron_module, "current_microbatch"):
            runner.microbatch_id = megatron_module.current_microbatch

        # Warmup and create the cudagraphs
        runner.create_fwd_graph(args, kwargs)
        self.cudagraph_runners.append(runner)
        runner.fwd_graph_recorded = True
        _CudagraphGlobalRecord.cudagraph_record.append((runner, "fwd", args, kwargs))

        return out, runner

    def __call__(self, megatron_module, args, kwargs):
        """Calls the forward pass of the cudagraphed module.

        Args:
            megatron_module (torch.nn.module): The megatron module to be graphed and run

            args (tuple):  The positional args to be passed to the module.

            kwargs (dict):  The keyword args to be passed to the module.

        """

        if len(self.cudagraph_runners) > 0:
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

            if not self.training:
                megatron_module.eval()

            runner, out = self.get_cudagraph_runner(megatron_module, args, kwargs)
            if self.training and runner.fwd_graph is not None and runner.bwd_graph is not None:
                out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)
            elif not self.training and runner.fwd_graph is not None:
                out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)

        else:
            if (
                'inference_context' in kwargs.keys()
                and kwargs['inference_context']
                and kwargs['inference_context'].is_decode_only()
            ):
                # Inference generation mode
                megatron_module.eval()
                runner, out = self.get_cudagraph_runner(megatron_module, args, kwargs)
                runner.eval()
            elif self.training:
                # Training mode
                runner, out = self.get_cudagraph_runner(megatron_module, args, kwargs)
                # check if a layer is frozen during training.
                if not torch.is_grad_enabled():
                    # If the layer is frozen, we need to set the runner to eval mode.
                    runner.eval()
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
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.transformer_layer import TransformerLayer

    if isinstance(layer, TransformerLayer):
        if config.cuda_graph_scope == 'attn':
            if not (
                isinstance(layer.self_attention, IdentityOp)
                and isinstance(layer.cross_attention, IdentityOp)
            ):
                return True
        else:
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
        assert config.external_cuda_graph, "Option --external-cuda-graph not enabled."
        assert config.cuda_graph_scope != "full_iteration", (
            "full_iteration cuda graph is not supported for --external-cuda-graph. "
            "Please use --enable-cuda-graph instead."
        )
        assert config.cuda_graph_scope in [
            'full',
            'attn',
        ], f"--cuda-graph-scope should be full or attn, got {config.cuda_graph_scope}."

        self.model = model
        self.config = config
        self.seq_length = seq_length
        self.micro_batch_size = micro_batch_size
        self.optimizers = optimizers

        # Get the number of models chunks and microbatches.
        self.num_model_chunks = len(model)
        self.num_microbatches = get_num_microbatches()

        # Get callables with captureable layers.
        self.chunks_with_decoder = []
        self.num_layers_per_chunk = []
        self.callables_per_chunk = []
        self.flattened_callables = []
        for chunk_number, model_chunk in enumerate(model):
            try:
                chunk_with_decoder = get_attr_wrapped_model(
                    model_chunk, 'decoder', allow_none=False, return_model_obj=True
                )
            except RuntimeError:
                num_graphable_layers = 0
                log_on_each_pipeline_stage(
                    logger,
                    logging.DEBUG,
                    f'Rank {torch.distributed.get_rank()}: '
                    f'No valid layer in model chunk {chunk_number}.',
                )
            else:
                num_decoder_layers = len(chunk_with_decoder.decoder.layers)
                if hasattr(chunk_with_decoder, 'mtp'):
                    num_mtp_layers = len(chunk_with_decoder.mtp.layers)
                else:
                    num_mtp_layers = 0
                num_graphable_layers = 0
                callables = []
                for layer_number in range(num_decoder_layers):
                    layer = chunk_with_decoder.decoder.layers[layer_number]
                    if _layer_is_graphable(layer, config):
                        num_graphable_layers += 1
                        callables.append(layer)
                for layer_number in range(num_mtp_layers):
                    layer = chunk_with_decoder.mtp.layers[layer_number].transformer_layer
                    if _layer_is_graphable(layer, config):
                        num_graphable_layers += 1
                        callables.append(layer)
                log_on_each_pipeline_stage(
                    logger,
                    logging.DEBUG,
                    f'Rank {torch.distributed.get_rank()}: '
                    f'{num_decoder_layers} decoder layers and {num_mtp_layers} MTP layers in '
                    f'model chunk {chunk_number}. {num_graphable_layers} graphable layers.',
                )
            finally:
                if num_graphable_layers > 0:
                    self.chunks_with_decoder.append(chunk_with_decoder)
                    self.num_layers_per_chunk.append(num_graphable_layers)
                    self.callables_per_chunk.append(callables)
                    self.flattened_callables.extend(callables)
                else:
                    self.chunks_with_decoder.append(None)
                    self.num_layers_per_chunk.append(0)
                    self.callables_per_chunk.append([])

        log_on_each_pipeline_stage(
            logger,
            logging.INFO,
            f'Rank {torch.distributed.get_rank()}: '
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
            for _ in range(self.num_microbatches):
                for layer in layers:
                    static_inputs = layer.get_layer_static_inputs(
                        self.seq_length, self.micro_batch_size
                    )
                    if is_te_min_version("1.10.0"):
                        # te.make_graphed_callables() accepts keyword arguments since 1.10.0.
                        hidden_states = static_inputs.pop("hidden_states")
                        sample_args.append((hidden_states,))
                        rotary_pos_emb = get_rotary_pos_emb(chunk_with_decoder, hidden_states)
                        if rotary_pos_emb is not None:
                            static_inputs["rotary_pos_emb"] = rotary_pos_emb
                        sample_kwargs.append(static_inputs)
                    else:
                        sample_args.append(
                            (
                                static_inputs.pop("hidden_states"),
                                static_inputs.pop("attention_mask"),
                            )
                        )

        # Get the PP and VPP scheduling order.
        from megatron.core.pipeline_parallel.schedules import (
            convert_schedule_table_to_order,
            get_pp_rank_microbatches,
            get_schedule_table,
        )

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
            logger, logging.DEBUG, f'Rank {torch.distributed.get_rank()}: ORDER {order}'
        )

        def get_make_graphed_callables_kwargs():
            kwargs = {'num_warmup_iters': 11, 'allow_unused_input': True, '_order': order}

            if is_te_min_version("2.6.0"):
                # Starting from TE 2.6.0, make_graphed_callables() accepts different number
                # of layers per chunk and optimizes the graph memory usage.
                kwargs['_num_layers_per_chunk'] = self.num_layers_per_chunk
                kwargs['_reuse_graph_input_output_buffers'] = True

            if sample_kwargs:
                kwargs['sample_kwargs'] = sample_kwargs

            from megatron.core.fp8_utils import get_fp8_recipe

            if self.config.fp8:
                kwargs['fp8_enabled'] = True
                kwargs['fp8_recipe'] = get_fp8_recipe(self.config)
                # fp8 weight caching will be ignored by TE if cudagraph doesn't capture attn,
                # even if we set it to True in the arguments. So we just pass a False in this case.
                kwargs['fp8_weight_caching'] = 'attn' in self.config.cuda_graph_scope
                if is_te_min_version("1.14.0") and parallel_state.model_parallel_is_initialized():
                    kwargs['fp8_group'] = parallel_state.get_amax_reduction_group(
                        with_context_parallel=True, tp_only_amax_red=self.config.tp_only_amax_red
                    )
            else:
                kwargs['fp8_enabled'] = False
            return kwargs

        kwargs = get_make_graphed_callables_kwargs()
        return sample_args, kwargs

    def _finish_capturing(self):
        """
        Finish capturing CUDA Graphs and clean up the related state.
        """
        from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker

        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        clear_aux_losses_tracker()
        gc.collect()
        torch.cuda.empty_cache()

    def create_cudagraphs(self):
        """
        Capture CUDA Graphs per TransformerLayer per microbatch.
        """
        _set_capture_start()

        # Set back to the default stream. Graph will still be captured on a side stream in
        # make_graphed_callables().
        torch.cuda.set_stream(torch.cuda.default_stream())
        torch.distributed.barrier()
        start = time.time()
        log_single_rank(logger, logging.INFO, f'Start CUDA Graphs capture...')

        # Prepare CUDA Graph capturing input data and call `make_graphed_callables`.
        sample_args, kwargs = self._get_cuda_graph_input_data()
        graphs = make_graphed_callables(tuple(self.flattened_callables), sample_args, **kwargs)

        # Push the captured graphs to the corresponding TransformerBlock.
        num_layers_accumulated = 0
        for layers in self.callables_per_chunk:
            for layer_number, layer in enumerate(layers):
                layer.cuda_graphs = []
                for batch_number in range(self.num_microbatches):
                    layer.cuda_graphs.append(
                        graphs[
                            num_layers_accumulated * self.num_microbatches
                            + batch_number * len(layers)
                            + layer_number
                        ]
                    )
            num_layers_accumulated += len(layers)

        # Finish CUDA Graph capturing.
        torch.distributed.barrier()
        log_single_rank(
            logger,
            logging.INFO,
            f'Time spent in CUDA Graphs capture on rank {torch.distributed.get_rank()}: '
            f'{time.time() - start}s',
        )
        self._finish_capturing()
        _set_capture_end()

    def cuda_graph_set_manual_hooks(self):
        """
        Set CUDA Graph manual hooks for the modules that contain direct parameters and
        are covered by cudagraphs.
        """
        for chunk_number, layers in enumerate(self.callables_per_chunk):
            model_chunk = self.model[chunk_number]
            for layer in layers:
                layer.setup_manual_hooks(model_chunk._make_forward_pre_hook)
