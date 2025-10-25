# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import gc
import inspect
import logging
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from enum import Enum
from itertools import chain

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
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.utils import get_submodule_with_decoder, is_te_min_version

try:
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    from transformer_engine.pytorch.graph import make_graphed_callables
    from transformer_engine.pytorch.graph import restore_fp8_tensors, save_fp8_tensors
    from transformer_engine.pytorch.graph import set_capture_end as te_set_capture_end
    from transformer_engine.pytorch.graph import set_capture_start as te_set_capture_start
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    import transformer_engine as te

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
    assert type(arg) in _SUPPORTED_TYPES or is_dataclass(
        arg
    ), f"Cudagraphs recieved an arg of type {type(arg)} which is not supported."

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

def is_moe_layer_with_early_return(module):
    from megatron.core.transformer.transformer_layer import TransformerLayer

    return isinstance(module, TransformerLayer) \
        and module.is_moe_layer \
        and 'moe_router' in module.config.cuda_graph_scope

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
        logging.getLogger(__name__).info(f"Creating {len(cls.cudagraph_record)} CUDA graphs")

        gc.collect()
        torch.cuda.empty_cache()

        for idx, g in enumerate(cls.cudagraph_record):
            runner, graph_type = g[0:2]
            if graph_type == 'fwd':
                args, kwargs, out = g[2:]
                runner.create_fwd_graph(args, kwargs, out)
            else:
                runner.create_bwd_graph()
 
        for g in cls.cudagraph_record:
            runner = g[0]
            runner.cudagraph_created = True

        cls.cudagraph_created = True
        cls.cudagraph_record = []

        torch.cuda.set_stream(torch.cuda.default_stream())


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
            if hasattr(cudagraph_input, 'can_skip_input_copy'):
                continue
            if user_input.data_ptr() != cudagraph_input.data_ptr():
                cudagraph_input.copy_(user_input)

        ctx.runner = runner
        if runner.fp8_enabled:
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

            # Note that FP8GlobalStateManager.is_first_fp8_module() does not work as each layer may be in its 
            # own fp8 context, when the fp8 recipe != delayed_scaling
            if runner.is_first_layer and (runner.fp8_param_cache_updated != is_first_microbatch):
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(not is_first_microbatch)
                runner.fp8_param_cache_updated = is_first_microbatch

        runner.fwd_graph.replay()

        out = tuple(o.detach() if not hasattr(o, "is_cudagraph_input") else o for o in runner.fwd_graph_output_surface)
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
            if cudagraph_output_grad is None:
                continue
            if user_output_grad.data_ptr() != cudagraph_output_grad.data_ptr():
                cudagraph_output_grad.copy_(user_output_grad)

        runner.bwd_graph.replay()
        runner.status = _GraphStatus.FWD_READY

        # Update FP8 scale factors if needed
        if runner.fp8_enabled \
            and isinstance(FP8GlobalStateManager.get_fp8_recipe(), te.common.recipe.DelayedScaling):
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # If using gradient_accumulation_fusion, whenever `main_grad` is calculated
        # the `grad_added_to_main_grad` attribute is expected to set. However when using
        # cudagraphs this doesn't occur so we emulate this behavior here.
        for param, grad_added in runner.groundtruth_grad_added_to_main_grad.items():
            param.grad_added_to_main_grad = grad_added

        output_grads = tuple(runner.static_grad_inputs)
        return None, None, *output_grads


class _CudaGraphRunner(torch.nn.Module):
    """Represents the execution of a cudagraphed module for a single microbatch.
    If there are multiple outstanding microbatches per module, such as for pipeline parallelism,
    CudaGraphManager automatically creates multiple _CudaGraphRunners per module."""

    def __init__(
        self, 
        base_module, 
        fwd_mempool, 
        bwd_mempool, 
        func, 
        params_to_calculate_wgrads,
        need_backward,
    ):
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
        self.fp4_enabled = False
        self.deallocate_pipeline_outputs = False
        self.num_warmup_steps = 2

        # Decide whether to reuse the input and output buffer, and if so,
        # whether this layer is the first layer (which needs an input buffer)
        # or the last layer (which needs an output buffer)

        self.is_transformer_decoder_layer = _determine_if_transformer_decoder_layer(base_module)
        self.grad_enabled = need_backward and torch.is_grad_enabled()

        if not self.grad_enabled or not isinstance(base_module, torch.nn.Module):
            self.params_to_calculate_wgrads = []
        else:
            if func is None:
                self.params_to_calculate_wgrads = list(self.base_module.parameters())
            else:
                assert params_to_calculate_wgrads is not None, \
                    "Parameters must be specified when cudagraphing a function that requires grad"
                self.params_to_calculate_wgrads = params_to_calculate_wgrads

        self.func = super(MegatronModule, self.base_module).__call__ if func is None else func

        from megatron.core.ssm.mamba_layer import MambaLayer
        from megatron.core.transformer.transformer_layer import (
            BaseTransformerLayer,
            TransformerLayer,
        )

        self.is_first_layer = False
        self.is_transformer_decoder_layer = False

        # decides if this is an LLM layer
        is_potential_decoder_layer = isinstance(
            base_module, (TransformerLayer, BaseTransformerLayer, MambaLayer)
        )
        if is_potential_decoder_layer:
            if isinstance(base_module, TransformerLayer) and not isinstance(
                base_module.cross_attention, IdentityOp
            ):
                self.is_transformer_decoder_layer = False
            else:
                self.is_transformer_decoder_layer = True
        else:
            self.is_transformer_decoder_layer = False

        if self.is_transformer_decoder_layer:
            total_num_layers = base_module.config.num_layers
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            vpp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is None:
                vpp_size = 1

            layers_per_chunk = total_num_layers // vpp_size // pp_size
            self.is_first_layer = ((base_module.layer_number - 1) % layers_per_chunk) == 0
            # We use this attribute to record the value of 'is_first_microbatch' each fwd cudagraph replay
            # so that way we only update the value of this flag in FP8GlobalStateManager when it changes, 
            # which incurs an expensive HtoD sync
            if self.is_first_layer:
                self.fp8_param_cache_updated = None

        if hasattr(self.base_module, "config") and isinstance(self.base_module.config, TransformerConfig):
            self.fuse_wgrad_accumulation = self.base_module.config.gradient_accumulation_fusion
            self.backward_retain_grad = self.base_module.config.cuda_graph_retain_backward_graph
            self.fp8_enabled = self.base_module.config.fp8 is not None
            self.fp4_enabled = self.base_module.config.fp4 is not None
            self.deallocate_pipeline_outputs = self.base_module.config.deallocate_pipeline_outputs
            self.num_warmup_steps = self.base_module.config.cuda_graph_warmup_steps

            if self.fp8_enabled:
                self.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe()
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(True)

            if self.fp4_enabled:
                from megatron.core.fp4_utils import get_fp4_recipe  # to avoid circular import

                self.fp4_recipe = get_fp4_recipe(self.base_module.config)
                FP8GlobalStateManager.set_skip_fp8_weight_update_tensor(False)

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

        # TODO jiemingz: this is a temporary hack to disable grabbing the fp8 context outside of the 
        # transformer layer
        if not self.is_transformer_decoder_layer:
            return nullcontext()

        if self.fp8_enabled:
            return self.get_fp8_context()
        elif self.fp4_enabled:
            return self.get_fp4_context()
        else:
            return nullcontext()

    def create_fwd_graph(self, args, kwargs, outputs):
        """Create a fwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        # save grads and other variables that may be affected by graph warmup
        if self.grad_enabled:
            save_main_grads = [
                param.main_grad.clone()
                for param in self.params_to_calculate_wgrads
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

        self.fwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.fwd_graph.register_generator_state(state)

        has_te_modules = False
        if HAVE_TE_GRAPHS and isinstance(self.base_module, torch.nn.Module):
            base_module = self.base_module
            has_te_modules = has_te_modules or any(
                [isinstance(m, TransformerEngineBaseModule) for m in base_module.modules()]
            )

        # Finalize the cudagraph input buffers
        args, kwargs = self.create_fwd_graph_input_buffer(args, kwargs)
        self.fwd_graph_input_args = args
        self.fwd_graph_input_kwargs = kwargs

        if self.is_transformer_decoder_layer:
            if 'rotary_pos_emb' in self.fwd_graph_input_kwargs and \
                torch.is_tensor( kwargs['rotary_pos_emb']):
                kwargs['rotary_pos_emb'].can_skip_input_copy = True       

        input_tensors = self.get_tensors(
            self.fwd_graph_input_args, 
            self.fwd_graph_input_kwargs
        )
        self.fwd_graph_input_surface = list(input_tensors)

        for ten in self.fwd_graph_input_surface:
            ten.is_cudagraph_input = True
        # input buffer now finalized

        if not self.grad_enabled:
            ctx = torch.no_grad()
        else:
            ctx = nullcontext()

        _set_capture_start()
        if has_te_modules:
            te_set_capture_start()

        with ctx:
            # warmup again as case graph capture mode may execute a different codepath
            for _ in range(self.num_warmup_steps):
                with self.get_quantization_context():
                    warmup_outputs = self.func(
                        *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                    )
                if self.grad_enabled:
                    if isinstance(warmup_outputs, torch.Tensor):
                        warmup_outputs = (warmup_outputs,)
                    warmup_outputs = self.get_tensors(warmup_outputs)
                    warmup_outputs = tuple(o for o in warmup_outputs if o.requires_grad)
                    input_tensors = self.get_tensors(args, kwargs)

                    torch.autograd.grad(
                        outputs=warmup_outputs,
                        inputs=tuple(i for i in input_tensors if i.requires_grad),
                        grad_outputs=tuple(torch.zeros_like(o) for o in warmup_outputs),
                        only_inputs=True,
                        allow_unused=True,
                    )

            with self.get_quantization_context():
                torch.cuda.synchronize()
                with torch.cuda.graph(self.fwd_graph, pool=self.fwd_mempool):
                    fwd_graph_outputs = self.func(
                        *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                    )

        _set_capture_end()
        if has_te_modules:
            te_set_capture_end()

        # save cudagraph output buffer
        if isinstance(fwd_graph_outputs, torch.Tensor):
            fwd_graph_outputs = (fwd_graph_outputs,)
        self.fwd_graph_outputs = fwd_graph_outputs

        self.fwd_graph_output_surface = self.get_tensors(self.fwd_graph_outputs)

        for idx, o in enumerate(self.get_tensors(outputs)):
            assert o.is_cudagraph_output
            if hasattr(o, "is_cudagraph_input"):
                self.fwd_graph_output_surface[idx].is_cudagraph_input = True
                o.fwd_cudagraph_buffer = self.fwd_graph_output_surface[idx]
            self.fwd_graph_output_surface[idx].is_cudagraph_output = True

        if self.grad_enabled:
            assert (
                len(self.fwd_graph_output_surface) > 0
            ), """Tried graphing a module that returned no tensors in training mode, 
                however the graphed module must output at least one tensor, 
                so that a corresponding backward node may be registered in the autograd graph."""

            if self.fp8_enabled:
                restore_fp8_tensors([self.base_module], saved_fp8_tensors)
            # restore cached grads
            for param in self.params_to_calculate_wgrads:
                if hasattr(param, 'main_grad'):
                    saved_grad = save_main_grads.pop(0)
                    assert (
                        param.main_grad.shape == saved_grad.shape
                    ), "Error restoring grads while cudagraphing!"
                    param.main_grad.copy_(saved_grad)

        if self.fp8_enabled:
            restore_fp8_tensors([self.base_module], saved_fp8_tensors)

    def create_bwd_graph(self):
        """Create a bwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        assert self.grad_enabled

        self.bwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.bwd_graph.register_generator_state(state)

        static_grad_outputs = []

        for idx, o in enumerate(self.fwd_graph_output_surface):
            if o.requires_grad:
                if hasattr(o, "is_cudagraph_input") and hasattr(o, "bwd_cudagraph_buffer"):
                    out_grad = o.bwd_cudagraph_buffer
                    assert out_grad.shape == o.shape
                    out_grad.is_cudagraph_input = True
                else:
                    out_grad = torch.zeros_like(o)
                out_grad.requires_grad = True
            else:
                out_grad = None
            static_grad_outputs.append(out_grad)

        input_tensors = self.get_tensors(
            self.fwd_graph_input_args, 
            self.fwd_graph_input_kwargs
        )
        fwd_input_surface = input_tensors + tuple(self.params_to_calculate_wgrads)

        torch.cuda.synchronize()

        with torch.cuda.graph(self.bwd_graph, pool=self.bwd_mempool):
            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in self.fwd_graph_output_surface if o.requires_grad),
                inputs=tuple(i for i in fwd_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                retain_graph=self.backward_retain_grad,
                only_inputs=True,
                allow_unused=True,
            )
        grad_inputs = list(grad_inputs)

        self.static_grad_outputs = static_grad_outputs
        self.static_grad_inputs = []
        self.params_to_backprop = []

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs
        # that don't require grad
        self.static_grad_inputs = []
        for idx, input_tensor in enumerate(input_tensors):
            if input_tensor.requires_grad:
                input_grad = grad_inputs.pop(0)
                input_grad.is_cudagraph_output = True
                input_tensor.bwd_cudagraph_buffer = input_grad
                self.static_grad_inputs.append(input_grad)
            else:
                self.static_grad_inputs.append(None)

        # at this point static_grad_inputs hold the input dgrads, add the wgrads next
        assert len(grad_inputs) == len(tuple(self.params_to_calculate_wgrads))
        assert len(self.static_grad_inputs) == len(input_tensors)

        # filter out params that did not return a wgrad
        for idx, param in enumerate(self.params_to_calculate_wgrads):
            wgrad_buffer = grad_inputs.pop(0)
            if (wgrad_buffer is not None) and param.requires_grad:
                self.fwd_graph_input_surface.append(param)
                self.params_to_backprop.append(param)
                self.static_grad_inputs.append(wgrad_buffer)

        self.static_grad_inputs = tuple(self.static_grad_inputs)
        self.params_to_backprop = tuple(self.params_to_backprop)
        self.fwd_graph_input_surface = tuple(self.fwd_graph_input_surface)

        self.groundtruth_grad_added_to_main_grad = {}
        if self.fuse_wgrad_accumulation:
            for param in self.params_to_calculate_wgrads:
                if hasattr(param, "grad_added_to_main_grad"):
                    self.groundtruth_grad_added_to_main_grad[param] = param.grad_added_to_main_grad


    def record_graph_capture(self, args, kwargs):
        """Records the data needed to create this runner's forward cudagraph.
        The first pass records a graph and appends the runner to _CudagraphGlobalRecord.
        The actual cudagraph will be created when 'create_cudagraphs()` is called. Subsequent
        passes should replay the graph."""

        for t in self.get_tensors(args, kwargs):
            t.is_cudagraph_input = True

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

        # autograd nodes return inputs as views, so clone the tensor as returning views may cause
        # issues, for instance with pipeline parallelism
        out = tuple(o.clone() if torch.is_tensor(o) else o for o in out)

        # mark all output tensors as cudagraph outputs, so there when we created the fwd graph we may
        # reused cudagraph output buffers as inputs        
        for o in self.get_tensors(out):
            o.is_cudagraph_output = True

        if not self.fwd_graph_recorded:
            logger.debug(f"Recording forward graph creation...")
            _CudagraphGlobalRecord.record_fwd_graph(self, args, kwargs, out)
            self.fwd_graph_recorded = True

        if len(out) == 1:
            return out[0]    
        return out

    def replay_graph_capture(self, is_first_microbatch, args, kwargs):
        """Replay the fwd cuda graph with autograd."""

        assert self.matches_graph_inputs(
            args, kwargs, check_types=False
        ), "Tried replaying a cudagraph with different arguments than what if was created with!"

        inp_tensors = self.get_tensors(args, kwargs, check_types=False)

        if self.grad_enabled:
            func_args = inp_tensors + self.params_to_backprop
        else:
            func_args = inp_tensors

        out = _CudagraphReplayNode.apply(self, is_first_microbatch, *func_args)

        # reconstruct the output, replacing tensors with the cudagraph output buffers
        if len(out) == 0:
            return None

        out = list(out)
        out = tuple(out.pop(0) if torch.is_tensor(o) else o for o in self.fwd_graph_outputs)
        if len(out) == 1:
            out = out[0]

        return out


    def _check_graph_input_match(self, val, ref, check_types):
        if torch.is_tensor(ref):
            if check_types:
                _check_supported_type(val)
                _check_supported_type(ref)

            return (
                torch.is_tensor(val)
                and val.shape == ref.shape
                and val.dtype == ref.dtype
                and val.device == ref.device
            )

        if is_dataclass(ref):
            if check_types:
                _check_supported_type(val)
                _check_supported_type(ref)

            if not is_dataclass(val):
                return False
                
            # Recursive check on fields
            for field in fields(ref):
                field_name = field.name
                
                if not hasattr(val, field_name):
                    return False
                    
                if not self._check_graph_input_match(
                    getattr(val, field_name), 
                    getattr(ref, field_name), 
                    check_types
                ):
                    return False
            return True

        if check_types:
            _check_supported_type(val)
            _check_supported_type(ref)
            if type(val) != type(ref):
                return False
        
        return ref == val

    def matches_graph_inputs(self, args, kwargs, check_types=True):
        if not args:
            if self.fwd_graph_input_args:
                return False        
        else:
            if len(args) != len(self.fwd_graph_input_args):
                return False
            for arg, graph_arg in zip(args, self.fwd_graph_input_args):
                if not self._check_graph_input_match(arg, graph_arg, check_types):
                    return False
   
        if len(kwargs) != len(self.fwd_graph_input_kwargs):
            return False
        for k, v_ref in self.fwd_graph_input_kwargs.items():
            if k not in kwargs:
                return False
            if not self._check_graph_input_match(kwargs[k], v_ref, check_types):
                return False
        return True

    def create_fwd_graph_input_buffer(self, args, kwargs=None):
        """Replace all tensors inside arg, kwargs with zeroed copies."""

        def clone_tensor(ten):
            assert ten.is_cudagraph_input
            if hasattr(ten, "is_cudagraph_output") and hasattr(ten, "fwd_cudagraph_buffer"):
                assert ten.fwd_cudagraph_buffer.shape == ten.shape
                out = ten.fwd_cudagraph_buffer.detach()
            else:
                out = torch.clone(ten.detach())
            out.requires_grad = ten.requires_grad
            return out

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

    def get_tensors(self, args, kwargs=None, check_types=True):
        """
        Filter and flatten all tensors from args and kwargs using list comprehensions
        and itertools.chain for faster flattening.
        """

        def extract_tensors(arg):
            if check_types:
                _check_supported_type(arg)
            
            if torch.is_tensor(arg):
                return [arg]
            
            if is_dataclass(arg):
                return [
                    attr for field in fields(arg)
                    if torch.is_tensor(attr := getattr(arg, field.name))
                ]
                
            return []

        args_tens = [tensor for arg in args for tensor in extract_tensors(arg)] if args else []        
        kwargs_tens = [
            tensor for val in kwargs.values() 
            for tensor in extract_tensors(val)
        ] if kwargs else []

        return tuple(chain(args_tens, kwargs_tens))

class CudaGraphManager(torch.nn.Module):
    """Creates and runs cudagraphs for a megatron module"""

    """A global mempool for when 'cuda_graph_use_single_mempool' is used."""
    global_mempool = None

    """Forward pass mempools, used with cudagraph reuse mode."""
    fwd_mempools = None

    """Backward pass mempool, used with cudagraph reuse mode."""
    bwd_mempool = None

    def __init__(self, 
        config: TransformerConfig, 
        base_module = None,
        function_name = None,
        params_to_calculate_wgrads = None,
        need_backward = True,
    ):
        super().__init__()
        """Creates a CudaGraphManager to manage CUDA graphs for a Megatron module.

        Args:
            config: TransformerConfig object containing CUDA graph settings for memory
                pooling, graph retention, gradient accumulation, FP8, and warmup steps.
        """


        if function_name is not None:
            func = getattr(base_module, function_name)
            def wrapped_func(*args, **kwargs):
                out = self(base_module, args, kwargs)
                return out
            setattr(base_module, function_name, wrapped_func)
        else:
            func = None

        self.func = func
        self.params_to_calculate_wgrads = params_to_calculate_wgrads
        self.need_backward = need_backward

        # need to delay the import here to avoid a circular import
        global HAVE_TE_GRAPHS
        try:
            from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
        except ImportError:
            TECudaRNGStatesTracker = None

        rng_tracker = get_cuda_rng_tracker()
        assert (
            rng_tracker.is_inference_rng_tracker
            or (HAVE_TE_GRAPHS and isinstance(rng_tracker, TECudaRNGStatesTracker))
            or (isinstance(rng_tracker, CudaRNGStatesTracker) and rng_tracker.use_cudagraphable_rng)
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


        if not _CudagraphGlobalRecord.cudagraph_created:
            if self.use_single_mempool:
                fwd_mempool = CudaGraphManager.global_mempool
                bwd_mempool = CudaGraphManager.global_mempool
            else:
                vpp_rank = parallel_state.get_virtual_pipeline_model_parallel_rank()
                vpp_rank = 0 if vpp_rank is None else vpp_rank
                fwd_mempool = CudaGraphManager.fwd_mempools[vpp_rank][len(self.cudagraph_runners)]
                bwd_mempool = CudaGraphManager.bwd_mempool

        if self.reuse_cudagraphs:
            for r in self.cudagraph_runners:
                if self.need_backward and torch.is_grad_enabled():
                    if r.status == _GraphStatus.FWD_READY and r.grad_enabled == True:
                        return r
                else:
                    # if not training, we dont expect any backward passes to run, so doesnt matter if runner was created
                    # with autograd or not
                    if r.status == _GraphStatus.FWD_READY:
                        return r

            if _CudagraphGlobalRecord.cudagraph_created:
                assert False

            runner = _CudaGraphRunner(
                megatron_module, 
                fwd_mempool, 
                bwd_mempool, 
                self.func, 
                self.params_to_calculate_wgrads,
                self.need_backward,
            )
            self.cudagraph_runners.append(runner)
            return runner

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
                    self.func, 
                    self.params_to_calculate_wgrads,
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

        if _CudagraphGlobalRecord.cudagraph_created:
            runner = self.get_cudagraph_runner(megatron_module)

            if self.training and torch.is_grad_enabled() and isinstance(megatron_module, torch.nn.Module):
                # Trigger Mcore DDP pre-forward hooks
                self.call_ddp_preforward_hook(megatron_module)
                for module in megatron_module.modules():
                    self.call_ddp_preforward_hook(module)

            out = runner.replay_graph_capture(self.is_first_microbatch, args, kwargs)

            self.is_first_microbatch = False

            # If forward only, next replay should be a forward pass as well
            if self.training and torch.is_grad_enabled() and self.need_backward:
                runner.status = _GraphStatus.BWD_READY
            else:
                runner.status = _GraphStatus.FWD_READY
        else:
            if 'inference_context' in kwargs.keys() and kwargs['inference_context']:
                # Inference generation mode
                runner = self.get_cudagraph_runner(megatron_module)
                runner.eval()
                out = runner.record_graph_capture(args, kwargs)
                runner.status = _GraphStatus.FWD_READY
            elif self.training:
                # Training mode
                runner = self.get_cudagraph_runner(megatron_module)
                out = runner.record_graph_capture(args, kwargs)

                if runner.grad_enabled:
                    runner.status = _GraphStatus.BWD_READY
                else:
                    runner.status = _GraphStatus.FWD_READY

        return out

    
# The following functions are for capturing CUDA Graphs using TE make_graphed_callables().
def layer_is_graphable(layer, config):
    """
    Check if a layer is graphable.
    """
    from megatron.core.transformer.transformer_layer import TransformerLayer
    from megatron.core.ssm.mamba_layer import MambaLayer
    from megatron.core.transformer.identity_op import IdentityOp
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.mlp import MLP

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


def get_cuda_graph_input_data(
    model,
    callables,
    config,
    num_microbatches,
    num_model_chunks,
    num_layers_per_chunk,
    seq_length,
    micro_batch_size,
):
    """
    Create the CUDA Graph capturing input data. The data is organized per-chunk per-microbatch per-layer.
    """

    rotary_pos_emb_cache = {}

    def get_rotary_pos_emb(transformer_module, transformer_input):
        if (
            transformer_module.position_embedding_type == 'rope'
            and not config.multi_latent_attention
        ):
            rotary_seq_len = transformer_module.rotary_pos_emb.get_rotary_seq_len(
                None, transformer_module.decoder, transformer_input, config, None
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
    num_layers_per_chunk_prefix_sum = [0]
    for n in num_layers_per_chunk:
        num_layers_per_chunk_prefix_sum.append(num_layers_per_chunk_prefix_sum[-1] + n)
    for chunk_number in range(num_model_chunks):
        model_chunk = model[chunk_number]
        chunk_with_decoder = get_submodule_with_decoder(model_chunk)
        if chunk_with_decoder is None:
            continue
        layers = callables[
            num_layers_per_chunk_prefix_sum[chunk_number] : num_layers_per_chunk_prefix_sum[
                chunk_number + 1
            ]
        ]
        for _ in range(num_microbatches):
            for layer in layers:
                static_inputs = layer.get_layer_static_inputs(seq_length, micro_batch_size)
                if is_te_min_version("1.10.0"):
                    # te.make_graphed_callables() accepts keyword arguments since 1.10.0.
                    hidden_states = static_inputs.pop("hidden_states")
                    sample_args.append((hidden_states,))

                    from megatron.core.transformer.transformer_layer import TransformerLayer
                    from megatron.core.transformer.identity_op import IdentityOp

                    if (
                        isinstance(layer, TransformerLayer)
                        and not isinstance(layer.self_attention, IdentityOp)
                        and 'attn' in config.cuda_graph_scope
                    ):
                        rotary_pos_emb = get_rotary_pos_emb(chunk_with_decoder, hidden_states)
                        if rotary_pos_emb is not None:
                            static_inputs["rotary_pos_emb"] = rotary_pos_emb
                    sample_kwargs.append(static_inputs)
                elif 'attn' in config.cuda_graph_scope:
                    sample_args.append(
                        (static_inputs.pop("hidden_states"), static_inputs.pop("attention_mask"))
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
        num_microbatches, num_model_chunks, config.microbatch_group_size_per_vp_stage, False
    )
    schedule_table = get_schedule_table(
        num_microbatches, num_model_chunks, config.microbatch_group_size_per_vp_stage
    )
    order = convert_schedule_table_to_order(
        num_warmup_microbatches, num_model_chunks, schedule_table
    )
    if torch.distributed.get_rank() == 0:
        print(f'Rank 0: ORDER {order}')

    def get_make_graphed_callables_kwargs():
        kwargs = {'num_warmup_iters': 11, 'allow_unused_input': True, '_order': order}

        if is_te_min_version("2.6.0"):
            # Starting from TE 2.6.0, make_graphed_callables() accepts different number
            # of layers per chunk and optimizes the graph memory usage.
            kwargs['_num_layers_per_chunk'] = num_layers_per_chunk
            kwargs['_reuse_graph_input_output_buffers'] = True

        if sample_kwargs:
            kwargs['sample_kwargs'] = sample_kwargs

        from megatron.core.fp8_utils import get_fp8_recipe

        if config.fp8:
            kwargs['fp8_enabled'] = True
            kwargs['fp8_recipe'] = get_fp8_recipe(config)
            # fp8 weight caching will be ignored by TE if cudagraph doesn't capture the attn part,
            # even if we set it to True in the arguments. So we just pass a False in this case.
            kwargs['fp8_weight_caching'] = 'attn' in config.cuda_graph_scope
            if is_te_min_version("1.14.0") and parallel_state.model_parallel_is_initialized():
                kwargs['fp8_group'] = parallel_state.get_amax_reduction_group(
                    with_context_parallel=True
                )
        else:
            kwargs['fp8_enabled'] = False
        return kwargs

    kwargs = get_make_graphed_callables_kwargs()
    return sample_args, kwargs


def cuda_graph_capture(model, config, seq_length, micro_batch_size):
    """
    Capture CUDA Graphs per TransformerLayer per microbatch.
    """
    _set_capture_start()
    assert HAVE_TE_GRAPHS, "CUDA Graphs are not supported with TE."
    assert config.external_cuda_graph, "Option --external-cuda-graph not enabled."

    # Set back to the default stream. Graph will still be captured on a side stream in
    # make_graphed_callables().
    torch.cuda.set_stream(torch.cuda.default_stream())
    torch.distributed.barrier()
    start = time.time()
    if torch.distributed.get_rank() == 0:
        print(f'Start cuda_graph_capture on rank{torch.distributed.get_rank()}')

    # Get the number of models chunks and microbatches.
    num_model_chunks = len(model)
    num_microbatches = get_num_microbatches()
    if torch.distributed.get_rank() == 0:
        print(f'num_model_chunks {num_model_chunks}, num_microbatches {num_microbatches}')

    # Get callables.
    callables = []
    num_layers_per_chunk = []
    for chunk_number in range(num_model_chunks):
        model_chunk = model[chunk_number]
        chunk_with_decoder = get_submodule_with_decoder(model_chunk)
        if chunk_with_decoder is None:
            continue
        num_layers = len(chunk_with_decoder.decoder.layers)
        if hasattr(chunk_with_decoder, 'mtp'):
            num_mtp_layers = len(chunk_with_decoder.mtp.layers)
        else:
            num_mtp_layers = 0
        num_graphable_layers = 0
        for layer_number in range(num_layers):
            layer = chunk_with_decoder.decoder.layers[layer_number]
            if layer_is_graphable(layer, config):
                num_graphable_layers += 1
                callables.append(layer)
        for layer_number in range(num_mtp_layers):
            layer = chunk_with_decoder.mtp.layers[layer_number].transformer_layer
            if layer_is_graphable(layer, config):
                num_graphable_layers += 1
                callables.append(layer)
        num_layers_per_chunk.append(num_graphable_layers)
        if torch.distributed.get_rank() == 0:
            print(
                f'{num_layers} layers and {num_mtp_layers} mtp layers in model chunk '
                f'{chunk_number}. {num_graphable_layers} graphable layers.'
            )
    if torch.distributed.get_rank() == 0:
        print(f'Total #layers {len(callables)}')

    # Prepare CUDA Graph capturing input data and call `make_graphed_callables`.
    sample_args, kwargs = get_cuda_graph_input_data(
        model,
        callables,
        config,
        num_microbatches,
        num_model_chunks,
        num_layers_per_chunk,
        seq_length,
        micro_batch_size,
    )

    graphs = make_graphed_callables(tuple(callables), sample_args, **kwargs)

    # Push the captured graphs to the corresponding TransformerBlock.
    num_layers_per_chunk_prefix_sum = [0]
    for n in num_layers_per_chunk:
        num_layers_per_chunk_prefix_sum.append(num_layers_per_chunk_prefix_sum[-1] + n)
    for chunk_number in range(num_model_chunks):
        model_chunk = model[chunk_number]
        chunk_with_decoder = get_submodule_with_decoder(model_chunk)
        if chunk_with_decoder is None:
            continue
        layers = callables[
            num_layers_per_chunk_prefix_sum[chunk_number] : num_layers_per_chunk_prefix_sum[
                chunk_number + 1
            ]
        ]
        for layer_number, layer in enumerate(layers):
            layer.cuda_graphs = []
            for batch_number in range(num_microbatches):
                layer.cuda_graphs.append(
                    graphs[
                        num_layers_per_chunk_prefix_sum[chunk_number] * num_microbatches
                        + batch_number * num_layers_per_chunk[chunk_number]
                        + layer_number
                    ]
                )

    # Finish CUDA Graph capturing.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            f'Time spent in cuda_graph_capture on rank {torch.distributed.get_rank()}:'
            f' {time.time() - start}s'
        )
    _set_capture_end()


def cuda_graph_set_manual_hooks(model, config):
    """
    Set CUDA Graph manual hooks for the modules that contain direct parameters and
    are covered by cudagraphs.
    """
    for model_chunk in model:
        chunk_with_decoder = get_submodule_with_decoder(model_chunk)
        if chunk_with_decoder is None:
            continue
        for layer in chunk_with_decoder.decoder.layers:
            if layer_is_graphable(layer, config):
                layer.setup_manual_hooks(model_chunk._make_forward_pre_hook)
        if hasattr(chunk_with_decoder, 'mtp'):
            for layer in chunk_with_decoder.mtp.layers:
                if layer_is_graphable(layer, config):
                    layer.transformer_layer.setup_manual_hooks(model_chunk._make_forward_pre_hook)
