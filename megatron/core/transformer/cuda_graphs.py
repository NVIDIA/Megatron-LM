# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import gc
import inspect
import logging
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from enum import Enum

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
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
    from transformer_engine.pytorch.graph import restore_fp8_tensors, save_fp8_tensors
    from transformer_engine.pytorch.graph import set_capture_end as te_set_capture_end
    from transformer_engine.pytorch.graph import set_capture_start as te_set_capture_start
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

    from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker

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


def _check_supported_type(arg):
    """Check if arg is a supported type for cudagraph input/outputs."""

    from megatron.core import InferenceParams  # guard against circular import

    _SUPPORTED_TYPES = {torch.Tensor, type(None), bool, int, str, float, InferenceParams}
    assert type(arg) in _SUPPORTED_TYPES or is_dataclass(
        arg
    ), f"Cudagraphs recieved an arg of type {type(arg)} which is not supported."


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
            [g[0].is_transformer_decoder_layer for g in cls.cudagraph_record]
        )
        if optimize_transformer_layer_graph_buffers:
            prev_fwd_hidden_state_output = None
            prev_bwd_hidden_state_inputgrad = None

        gc.collect()
        torch.cuda.empty_cache()

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()

        for idx, g in enumerate(cls.cudagraph_record):
            runner, graph_type = g[0:2]

            if optimize_transformer_layer_graph_buffers:
                if graph_type == 'fwd':
                    args, kwargs = g[2:]

                    if not runner.is_first_layer:
                        kwargs['hidden_states'] = prev_fwd_hidden_state_output
                    runner.create_fwd_graph(args, kwargs, clone_inputs=False)

                    # The output of TransformerLayer is: (hidden_states, None)
                    prev_fwd_hidden_state_output, _ = runner.fwd_graph_outputs
                else:
                    runner.create_bwd_graph(prev_bwd_hidden_state_inputgrad)

                    # The first input grad TransformerLayer is for 'hidden_states'
                    if not runner.is_last_layer:
                        prev_bwd_hidden_state_inputgrad = runner.static_grad_inputs[0]
            else:
                runner, graph_type = g[0:2]
                if graph_type == 'fwd':
                    args, kwargs = g[2:]
                    runner.create_fwd_graph(args, kwargs)
                else:
                    runner.create_bwd_graph()

        for g in cls.cudagraph_record:
            runner = g[0]
            runner.cudagraph_created = True

        cls.cudagraph_created = True
        cls.cudagraph_record = []

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

        if runner.is_first_layer:
            output_grads = tuple(
                b.clone().detach() if b is not None else b for b in runner.get_input_grads()
            )
        else:
            output_grads = tuple(
                b.detach() if b is not None else b for b in runner.get_input_grads()
            )
        return None, None, *output_grads


class _CudaGraphRunner(torch.nn.Module):
    """Represents the execution of a cudagraphed module for a single microbatch.
    If there are multiple outstanding microbatches per module, such as for pipeline parallelism,
    CudaGraphManager automatically creates multiple _CudaGraphRunners per module."""

    def __init__(self, base_module, fwd_mempool, bwd_mempool):
        """Creates a _CudaGraphRunner, which holds a single pair of fwd and bwd cudagraphs, which
        are not created until this runner records its graph creation into
        '_CudagraphGlobalRecord', and 'create_cudagraphs()' is called."""

        super().__init__()

        self.base_module = base_module
        self.fwd_mempool = fwd_mempool
        self.bwd_mempool = bwd_mempool

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

        from megatron.core.transformer.transformer_layer import TransformerLayer

        self.is_first_layer = None
        self.is_last_layer = None
        self.is_transformer_decoder_layer = False
        if isinstance(base_module, TransformerLayer) and isinstance(
            base_module.cross_attention, IdentityOp
        ):
            self.is_transformer_decoder_layer = True

            total_num_layers = base_module.config.num_layers
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            vpp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is None:
                vpp_size = 1

            layers_per_chunk = total_num_layers // vpp_size // pp_size
            self.is_first_layer = ((base_module.layer_number - 1) % layers_per_chunk) == 0
            self.is_last_layer = (base_module.layer_number % layers_per_chunk) == 0

    def get_fp8_context(self):
        """Return a new fp8 context in cudagraph mode."""

        if self.fp8_enabled:
            return fp8_autocast(
                enabled=True, calibrating=False, fp8_recipe=self.fp8_recipe, _graph=True
            )
        return nullcontext()

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

        self.fwd_graph_input_args = args
        self.fwd_graph_input_kwargs = kwargs

        input_tensors = self.get_tensors(args, kwargs)
        self.fwd_graph_input_surface = input_tensors + tuple(self.base_module.parameters())

        self.fwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.fwd_graph.register_generator_state(state)

        # warmup again as case graph capture mode may execute a different codepath
        for _ in range(self.num_warmup_steps):
            with self.get_fp8_context():
                outputs = self.base_module.forward(
                    *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                )
            if self.training and torch.is_grad_enabled():
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
            with torch.cuda.graph(self.fwd_graph, pool=self.fwd_mempool):
                outputs = self.base_module.forward(
                    *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                )

        # save cudagraph output buffer
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

    def get_input_grads(self):
        """Get the inputs grads that are returned by the bwd cudagraph call. If using grad accum
        fusion, wgrads have already been accumulated, so return dummy wgrads."""

        if not self.fuse_wgrad_accumulation:
            return self.static_grad_inputs
        else:
            num_dgrads = len(self.static_grad_inputs) - len(list(self.base_module.parameters()))
            dgrads = self.static_grad_inputs[:num_dgrads]
            wgrads = self.static_grad_inputs[num_dgrads:]

            wgrads_with_placeholders = []
            for idx, param in enumerate(self.base_module.parameters()):
                if getattr(param, "grad_added_to_main_grad", False):
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
            return tuple(dgrads + wgrads_with_placeholders)

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

        # Register a noop autograd node that toggles `self.graph_status` in the bwd pass, which
        # tracks when the runner completes its bwd pass.
        # If it's the first bwd encountered by this runner, record it to _CudagraphGlobalRecord
        out = tuple(_CudagraphRecordNode.apply(self, o) if torch.is_tensor(o) else o for o in out)

        # autograd nodes return inputs as views, so clone the tensor as returning views may cause
        # issues, for instance with pipeline parallelism
        return tuple(o.clone() if torch.is_tensor(o) else o for o in out)

    def replay_graph_capture(self, is_first_microbatch, args, kwargs):
        """Replay the fwd cuda graph with autograd."""

        assert self.matches_graph_inputs(
            args, kwargs
        ), "Tried replaying a cudagraph with different arguments than what if was created with!"

        inp_tensors = self.get_tensors(args, kwargs)
        func_args = inp_tensors + tuple(self.parameters())

        out = _CudagraphReplayNode.apply(self, is_first_microbatch, *func_args)
        out = list(out)
        return tuple(out.pop(0) if torch.is_tensor(o) else o for o in self.fwd_graph_outputs)

    def matches_graph_inputs(self, args, kwargs):
        """Check that the passed args, kwargs match with the arg, kwargs
        the graph was created with."""

        def check(val, ref):

            _check_supported_type(val)
            _check_supported_type(ref)

            # check that the args are the same type
            if not ((type(val) == type(ref)) or (is_dataclass(val) and is_dataclass(ref))):
                return False

            # if tensors, check they have the same shape, device and type
            # differing memory layout is allowed as 'copy_' is able interop different layouts
            if isinstance(ref, torch.Tensor):
                return (
                    val.shape == ref.shape and val.dtype == ref.dtype and val.device == ref.device
                )

            # if dataclass, check args in fields are the same
            elif is_dataclass(ref):
                for field in fields(ref):
                    if not check(getattr(val, field.name), getattr(ref, field.name)):
                        return False
                return True
            else:
                return ref == val

        if len(args) != len(self.fwd_graph_input_args):
            return False
        for arg, graph_arg in zip(args, self.fwd_graph_input_args):
            if not check(args, graph_arg):
                return False

        if kwargs.keys() != self.fwd_graph_input_kwargs.keys():
            return False
        for k, v in self.fwd_graph_input_kwargs.items():
            if not check(kwargs[k], v):
                return False
        return True

    def zero_out_tensors(self, args, kwargs=None):
        """Replace all tensors inside arg, kwargs with zeroed copies."""

        def clone_tensor(ten):
            cloned = torch.zeros_like(ten)
            cloned.requires_grad = ten.requires_grad
            return cloned

        def process_arg(arg):
            _check_supported_type(arg)
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

    def get_tensors(self, args, kwargs=None):
        """Filter and flatten all tensors from args and kwargs."""

        def extract_tensors(arg):
            _check_supported_type(arg)
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

    def __init__(self, config: TransformerConfig):
        super().__init__()

        rng_tracker = get_cuda_rng_tracker()
        assert (
            (HAVE_TE_GRAPHS and isinstance(rng_tracker, TECudaRNGStatesTracker))
            or (
                isinstance(rng_tracker, CudaRNGStatesTracker) and rng_tracker.use_cudagraphable_rng
            ),
        ), "RNG tracker does not support cudagraphs!"

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

    def get_cudagraph_runner(self, megatron_module):
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
                (r for r in self.cudagraph_runners if r.status == _GraphStatus.FWD_READY), None
            )
            if runner is None:
                if _CudagraphGlobalRecord.cudagraph_created:
                    assert False
                else:
                    runner = _CudaGraphRunner(megatron_module, fwd_mempool, bwd_mempool)
                    self.cudagraph_runners.append(runner)
        else:
            # Create cudagraphs for every microbatch
            if _CudagraphGlobalRecord.cudagraph_created:
                runner = self.cudagraph_runners[0]
                assert runner.status == _GraphStatus.FWD_READY
                self.cudagraph_runners = self.cudagraph_runners[1:] + self.cudagraph_runners[:1]
            else:
                runner = _CudaGraphRunner(megatron_module, fwd_mempool, bwd_mempool)
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

            runner = self.get_cudagraph_runner(megatron_module)
            out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)
        else:
            if 'inference_params' in kwargs.keys() and kwargs['inference_params']:
                # Inference generation mode
                runner = self.get_cudagraph_runner(megatron_module)
                runner.eval()
                out = runner.record_graph_capture(args, kwargs)
            elif self.training and torch.is_grad_enabled():
                # Training mode
                runner = self.get_cudagraph_runner(megatron_module)
                out = runner.record_graph_capture(args, kwargs)
            else:
                # No cudagraphs were found in training mode with grad disabled, so fallback to
                # eager since autograd is needed to correctly trace the backward graph.
                return super(MegatronModule, megatron_module).__call__(*args, **kwargs)

            runner = self.get_cudagraph_runner(megatron_module)
            out = runner.record_graph_capture(args, kwargs)

        # If forward only, next replay should be a forward pass as well
        if self.training and torch.is_grad_enabled():
            runner.status = _GraphStatus.BWD_READY
        else:
            runner.status = _GraphStatus.FWD_READY

        return out
