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
import sys

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
    from transformer_engine.pytorch.utils import make_weak_ref
    import transformer_engine as te

    HAVE_TE_GRAPHS = True
except:
    HAVE_TE_GRAPHS = False

_IS_GRAPH_CAPTURING = False

Cudagraph_VPP_Stage = None

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


def tensors_match(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.device == b.device


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
        tic = time.time()
        gc.collect()
        torch.cuda.empty_cache()

        global_tensor_pool = []
        for idx, g in enumerate(cls.cudagraph_record):
            runner, graph_type = g[0:2]

            if graph_type == 'fwd':
                args, kwargs, out = g[2:]
                runner.create_fwd_graph(global_tensor_pool, args, kwargs, out)
            else:
                runner.create_bwd_graph(global_tensor_pool)

        # all references inside the cudagraph_runners are turned into weakrefs, so keep a reference
        # to global_tensor_pool, which is all strong refs
        cls.global_tensor_pool = global_tensor_pool
        for g in cls.cudagraph_record:
            runner = g[0]
            runner.cudagraph_created = True

        cls.cudagraph_created = True
        cls.cudagraph_record = []

        global bwd_buffer_reuse_ref_count, fwd_buffer_reuse_ref_count
        assert bwd_buffer_reuse_ref_count == 0
        assert fwd_buffer_reuse_ref_count == 0

        # since we've been freeze/unfreezing the gc
        gc.collect()
        torch.cuda.empty_cache()
        toc = time.time()
        logging.getLogger(__name__).info(f"Creating CUDA graphs tooks {toc-tic:.2f} s")

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

        out = tuple(
            o.clone() if not hasattr(o, "is_cudagraph_input") else o
            for o in runner.fwd_graph_output_surface
        )

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
        if runner.fp8_enabled and isinstance(
            FP8GlobalStateManager.get_fp8_recipe(), te.common.recipe.DelayedScaling
        ):
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # If using gradient_accumulation_fusion, whenever `main_grad` is calculated
        # the `grad_added_to_main_grad` attribute is expected to set. However when using
        # cudagraphs this doesn't occur so we emulate this behavior here.
        for param, grad_added in runner.groundtruth_grad_added_to_main_grad.items():
            param.grad_added_to_main_grad = grad_added

        input_grads = []
        for inp_grad in runner.static_grad_inputs:
            if not hasattr(inp_grad, "is_cudagraph_input") and torch.is_tensor(inp_grad):
                inp_grad = torch.clone(inp_grad)
            input_grads.append(inp_grad)

        input_grads = tuple(input_grads)
        return None, None, *input_grads


strong_reference_cache_dataptrs = set()
strong_reference_cache = []


class _CudaGraphRunner(torch.nn.Module):
    """Represents the execution of a cudagraphed module for a single microbatch.
    If there are multiple outstanding microbatches per module, such as for pipeline parallelism,
    CudaGraphManager automatically creates multiple _CudaGraphRunners per module."""

    def __init__(
        self,
        base_module,
        mempool,
        func,
        params_to_calculate_wgrads,
        need_backward,
        num_warmup_steps,
    ):
        """Creates a _CudaGraphRunner, which holds a single pair of fwd and bwd cudagraphs, which
        are not created until this runner records its graph creation into
        '_CudagraphGlobalRecord', and 'create_cudagraphs()' is called."""

        super().__init__()

        self.base_module = base_module
        self.mempool = mempool

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
        self.num_warmup_steps = num_warmup_steps

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
                assert (
                    params_to_calculate_wgrads is not None
                ), "Parameters must be specified when cudagraphing a function that requires grad"
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

        if hasattr(self.base_module, "config") and isinstance(
            self.base_module.config, TransformerConfig
        ):
            self.fuse_wgrad_accumulation = self.base_module.config.gradient_accumulation_fusion
            self.backward_retain_grad = self.base_module.config.cuda_graph_retain_backward_graph
            self.fp8_enabled = self.base_module.config.fp8 is not None
            self.fp4_enabled = self.base_module.config.fp4 is not None

            # TODO(kwyss): This solution is only OK in the context of pretrain_mamba.py; there is
            # no check whether the parent module is a TransformerBlock or MambaBlock, and the flag
            # is named with mamba_stack. Ideally a single source of truth could control both
            # cuda graph and mamba_block.py logic on this flag.
            if (
                isinstance(base_module, TransformerLayer)
                and self.base_module.config.keep_mamba_stack_attention_linear_in_bf16
                and not self.base_module.is_moe_layer
            ):
                self.fp8_enabled = False
                self.fp4_enabled = False

            self.deallocate_pipeline_outputs = self.base_module.config.deallocate_pipeline_outputs

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

    def create_fwd_graph(self, global_tensor_pool, args, kwargs, outputs):
        """Create a fwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""

        def get_fwd_input_buffer(ten):
            assert ten.is_cudagraph_input
            if hasattr(ten, "is_cudagraph_output"):
                out = ten.fwd_cudagraph_buffer
                out.buffer_reuse_count -= 1
                if out.buffer_reuse_count == 0:
                    delattr(ten, 'fwd_cudagraph_buffer')
                    global fwd_buffer_reuse_ref_count
                    fwd_buffer_reuse_ref_count -= 1

                out = out.detach()
                out.requires_grad = ten.requires_grad
                return out
            else:
                for idx, buf in enumerate(global_tensor_pool):
                    if tensors_match(buf, ten):
                        out = global_tensor_pool.pop(idx).detach()
                        out.requires_grad = ten.requires_grad
                        return out

            out = torch.zeros_like(ten).detach()
            global strong_reference_cache, strong_reference_cache_dataptrs
            strong_reference_cache.append(out)
            strong_reference_cache_dataptrs.add(out.data_ptr())

            out.requires_grad = ten.requires_grad
            return out

        self.args = args
        self.kwargs = kwargs
        self.outputs = outputs

        # save grads and other variables that may be affected by graph warmup
        saved_fp8_tensors = None
        if self.grad_enabled:
            save_main_grads = [
                param.main_grad.clone()
                for param in self.params_to_calculate_wgrads
                if hasattr(param, 'main_grad')
            ]

            if self.fp8_enabled:
                if is_te_min_version("1.13.0"):
                    saved_fp8_tensors = save_fp8_tensors([self.base_module], self.fp8_recipe)
                else:
                    saved_fp8_tensors = save_fp8_tensors(
                        [self.base_module], self.fp8_recipe.amax_history_len
                    )

        # cache the moe aux loss if needed, this is needed because the moe aux loss is accumulated inside
        # the transformer layer forward pass:
        is_moe = (
            self.is_transformer_decoder_layer
            and hasattr(self.base_module, "is_moe_layer")
            and self.base_module.is_moe_layer
        )
        if is_moe:
            from megatron.core.transformer.moe.moe_utils import get_moe_layer_wise_logging_tracker

            tracker = get_moe_layer_wise_logging_tracker()
            cached_aux_losses = {}
            for name in tracker:
                cached_aux_losses[name] = torch.clone(tracker[name]["values"])

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
        self.fwd_graph_input_args, self.fwd_graph_input_kwargs = self._apply_to_all_tensors(
            get_fwd_input_buffer, args, kwargs
        )

        if self.is_transformer_decoder_layer:
            if 'rotary_pos_emb' in self.fwd_graph_input_kwargs and torch.is_tensor(
                kwargs['rotary_pos_emb']
            ):
                kwargs['rotary_pos_emb'].can_skip_input_copy = True

        input_tensors = self.get_tensors(self.fwd_graph_input_args, self.fwd_graph_input_kwargs)
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
            # jiemingz: cudagraph warmup seems to be not needed for nemotron6 training
            # warmup again as case graph capture mode may execute a different codepath
            # for _ in range(self.num_warmup_steps):
            #     with self.get_quantization_context():
            #         warmup_args, warmup_kwargs = self.clone_tensors(args, kwargs)
            #         warmup_outputs = self.func(
            #             *warmup_args, **warmup_kwargs
            #         )
            #     if self.grad_enabled:
            #         if isinstance(warmup_outputs, torch.Tensor):
            #             warmup_outputs = (warmup_outputs,)
            #         warmup_outputs = self.get_tensors(warmup_outputs)
            #         warmup_outputs = tuple(o for o in warmup_outputs if o.requires_grad)

            #         torch.autograd.grad(
            #             outputs=warmup_outputs,
            #             inputs=tuple(i for i in self.get_tensors(warmup_args, warmup_kwargs) if i.requires_grad),
            #             grad_outputs=tuple(torch.zeros_like(o) for o in warmup_outputs),
            #             only_inputs=True,
            #             allow_unused=True,
            #         )

            gc.freeze()

            with self.get_quantization_context():
                torch.cuda.synchronize()
                with torch.cuda.graph(self.fwd_graph, pool=self.mempool):
                    fwd_graph_outputs = self.func(
                        *self.fwd_graph_input_args, **self.fwd_graph_input_kwargs
                    )

            gc.unfreeze()

        _set_capture_end()
        if has_te_modules:
            te_set_capture_end()

        # save cudagraph output buffer
        if isinstance(fwd_graph_outputs, torch.Tensor):
            fwd_graph_outputs = (fwd_graph_outputs,)
        self.fwd_graph_outputs = fwd_graph_outputs

        self.fwd_graph_output_surface = list(self.get_tensors(fwd_graph_outputs))

        for idx, o in enumerate(self.get_tensors(self.outputs)):
            assert o.is_cudagraph_output
            if hasattr(o, "is_cudagraph_input"):
                self.fwd_graph_output_surface[idx].is_cudagraph_input = True
                if not hasattr(o, "fwd_cudagraph_buffer"):
                    self.fwd_graph_output_surface[idx].buffer_reuse_count = (
                        o.cudagraph_reuse_ref_count
                    )
                    o.fwd_cudagraph_buffer = self.fwd_graph_output_surface[idx]
                    global fwd_buffer_reuse_ref_count
                    fwd_buffer_reuse_ref_count += 1

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

        if is_moe:
            for name in tracker:
                tracker[name]["values"].copy_(cached_aux_losses[name])

    def create_bwd_graph(self, global_tensor_pool):
        """Create a bwd cudagraph for this runner. Should be called inside
        'create_cudagraphs()'."""
        global bwd_buffer_reuse_ref_count, strong_reference_cache, strong_reference_cache_dataptrs

        assert self.grad_enabled

        self.bwd_graph = torch.cuda.CUDAGraph()

        # For cases with multiple active RNG states, e.g. TP.
        for _, state in get_all_rng_states().items():
            self.bwd_graph.register_generator_state(state)

        static_grad_outputs = []

        for idx, o in enumerate(self.get_tensors(self.outputs)):
            out_grad = None
            if o.requires_grad:
                if hasattr(o, "is_cudagraph_input"):
                    assert (
                        hasattr(o, "bwd_cudagraph_buffer")
                        and o.bwd_cudagraph_buffer.shape == o.shape
                    )
                    out_grad = o.bwd_cudagraph_buffer
                    delattr(o, 'bwd_cudagraph_buffer')

                    out_grad.buffer_reuse_count -= 1
                    bwd_buffer_reuse_ref_count -= 1
                else:
                    for idx, buf in enumerate(global_tensor_pool):
                        if tensors_match(buf, o):
                            out_grad = global_tensor_pool.pop(idx).detach()
                            break  # Found a buffer, exit loop
                    else:
                        out_grad = torch.zeros_like(o)
                        strong_reference_cache.append(out_grad)
                        strong_reference_cache_dataptrs.add(out_grad.data_ptr())
                out_grad.requires_grad = True
            static_grad_outputs.append(out_grad)

        input_tensors = self.get_tensors(self.fwd_graph_input_args, self.fwd_graph_input_kwargs)
        fwd_input_surface = input_tensors + tuple(self.params_to_calculate_wgrads)

        torch.cuda.synchronize()

        gc.freeze()

        with torch.cuda.graph(self.bwd_graph, pool=self.mempool):
            grad_inputs = torch.autograd.grad(
                outputs=tuple(o for o in self.fwd_graph_output_surface if o.requires_grad),
                inputs=tuple(i for i in fwd_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                retain_graph=False,
                only_inputs=True,
                allow_unused=True,
            )

        gc.unfreeze()

        grad_inputs = list(grad_inputs)
        self.static_grad_outputs = static_grad_outputs
        self.static_grad_inputs = []
        self.params_to_backprop = []

        # Constructs a tuple suitable for returning from Graphed.backward:
        # Pads out the actually-needed grads with Nones in gradient slots for inputs
        # that don't require grad
        self.static_grad_inputs = []
        for idx, input_tensor in enumerate(self.get_tensors(self.args, self.kwargs)):
            if input_tensor.requires_grad:
                input_grad = grad_inputs.pop(0)
                if hasattr(input_tensor, "is_cudagraph_output"):
                    input_grad.is_cudagraph_output = True
                    if not hasattr(input_tensor, "bwd_cudagraph_buffer"):
                        input_tensor.bwd_cudagraph_buffer = input_grad
                        count = getattr(input_grad, "buffer_reuse_count", 0)
                        input_grad.buffer_reuse_count = count + 1
                        bwd_buffer_reuse_ref_count += 1
                self.static_grad_inputs.append(input_grad)
            else:
                self.static_grad_inputs.append(None)

        # at this point static_grad_inputs hold the input dgrads, add the wgrads next
        self.num_dgrads = len(self.static_grad_inputs)
        self.num_wgrads = len(tuple(self.params_to_calculate_wgrads))
        assert self.num_wgrads == len(grad_inputs)
        assert self.num_dgrads == len(input_tensors)

        # filter out params that did not return a wgrad
        for idx, param in enumerate(self.params_to_calculate_wgrads):
            wgrad_buffer = grad_inputs.pop(0)
            if (wgrad_buffer is not None) and param.requires_grad:
                self.fwd_graph_input_surface.append(param)
                self.params_to_backprop.append(param)
                self.static_grad_inputs.append(wgrad_buffer)

        self.static_grad_inputs = tuple(self.static_grad_inputs)
        self.params_to_backprop = tuple(self.params_to_backprop)

        self.groundtruth_grad_added_to_main_grad = {}
        if self.fuse_wgrad_accumulation:
            for param in self.params_to_calculate_wgrads:
                if hasattr(param, "grad_added_to_main_grad"):
                    self.groundtruth_grad_added_to_main_grad[param] = param.grad_added_to_main_grad

        # tensors outside the cudagraph mempool are added to the global_tensor_pool for reuse
        # global_tensor_pool will also keep a strong reference so that these tensors aren't deallocated
        all_tensors = (
            list(self.fwd_graph_input_surface)
            + list(self.fwd_graph_output_surface)
            + list(self.static_grad_outputs)
            + list(self.static_grad_inputs)
        )

        for ten in all_tensors:
            if not torch.is_tensor(ten):
                continue
            if ten.data_ptr() not in strong_reference_cache_dataptrs:
                continue
            if hasattr(ten, "buffer_reuse_count") and ten.buffer_reuse_count != 0:
                continue
            global_tensor_pool.append(ten)

        # now weakref everything
        self.fwd_graph_input_surface = self.replace_tensors_with_weak_refs(
            self.fwd_graph_input_surface
        )
        self.fwd_graph_input_args, self.fwd_graph_input_kwargs = (
            self.replace_tensors_with_weak_refs(
                self.fwd_graph_input_args, self.fwd_graph_input_kwargs
            )
        )
        self.fwd_graph_output_surface = self.replace_tensors_with_weak_refs(
            self.fwd_graph_output_surface
        )
        self.fwd_graph_outputs = self.replace_tensors_with_weak_refs(self.fwd_graph_outputs)
        # It is safe to weakref static_grad_inputs since any input grad that's reused as an output grad has a
        # strong ref in the attribute 'bwd_cudagraph_buffer'
        self.static_grad_inputs = self.replace_tensors_with_weak_refs(self.static_grad_inputs)
        self.static_grad_outputs = self.replace_tensors_with_weak_refs(self.static_grad_outputs)

        delattr(self, "args")
        delattr(self, "kwargs")
        delattr(self, "outputs")

    def record_graph_capture(self, args, kwargs):
        """Records the data needed to create this runner's forward cudagraph.
        The first pass records a graph and appends the runner to _CudagraphGlobalRecord.
        The actual cudagraph will be created when 'create_cudagraphs()` is called. Subsequent
        passes should replay the graph."""

        for t in self.get_tensors(args, kwargs):
            t.is_cudagraph_input = True
            if hasattr(t, "is_cudagraph_output"):
                # increment input/output reuse count
                count = getattr(t, "cudagraph_reuse_ref_count", 0)
                t.cudagraph_reuse_ref_count = count + 1

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

        # mark all output tensors as cudagraph outputs, so there when we created the fwd graph we may
        # reused cudagraph output buffers as inputs
        for o in self.get_tensors(out):
            o.is_cudagraph_output = True

        if not self.fwd_graph_recorded:
            logger.debug(f"Recording forward graph creation...")
            m_args, m_kwargs = self.replace_tensors_with_weak_refs(args, kwargs, cache_refs=True)
            m_out = self.replace_tensors_with_weak_refs(out, cache_refs=True)
            _CudagraphGlobalRecord.record_fwd_graph(self, m_args, m_kwargs, m_out)
            self.fwd_graph_recorded = True

        # # autograd nodes return inputs as views, so clone the tensor as returning views may cause
        # # issues, for instance with pipeline parallelism
        # out = tuple(o.clone() if torch.is_tensor(o) else o for o in out)

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
                    getattr(val, field_name), getattr(ref, field_name), check_types
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

    def _apply_to_all_tensors(self, func, args, kwargs):
        """Replace all tensors inside arg, kwargs with zeroed copies."""

        def process_arg(arg):
            if torch.is_tensor(arg):
                return func(arg)
            elif is_dataclass(arg):
                for field in fields(arg):
                    attr = getattr(arg, field.name)
                    if torch.is_tensor(attr):
                        setattr(arg, field.name, func(arg))
            return arg

        if torch.is_tensor(args):
            return func(args)

        args_replaced = []
        if args is not None and len(args) > 0:
            for arg in args:
                args_replaced.append(process_arg(arg))
        if kwargs is None:
            return args_replaced

        kwargs_replaced = {}
        for k, v in kwargs.items():
            kwargs_replaced[k] = process_arg(v)

        return args_replaced, kwargs_replaced

    def replace_tensors_with_weak_refs(self, args, kwargs=None, cache_refs=False):
        """Replace all tensors inside arg, kwargs with zeroed copies."""

        def replace_with_weak_ref(arg):
            if cache_refs and hasattr(arg, "weak_ref"):
                ref = arg.weak_ref
            else:
                ref = make_weak_ref(arg)
                if cache_refs:
                    arg.weak_ref = ref

            ref.requires_grad = arg.requires_grad
            if hasattr(arg, "is_cudagraph_input"):
                ref.is_cudagraph_input = True
            if hasattr(arg, "is_cudagraph_output"):
                ref.is_cudagraph_output = True
            if hasattr(arg, "cudagraph_reuse_ref_count"):
                ref.cudagraph_reuse_ref_count = arg.cudagraph_reuse_ref_count
            if hasattr(arg, "fwd_cudagraph_buffer"):
                ref.fwd_cudagraph_buffer = arg.fwd_cudagraph_buffer
            if hasattr(arg, "bwd_cudagraph_buffer"):
                ref.bwd_cudagraph_buffer = arg.bwd_cudagraph_buffer
            return ref

        return self._apply_to_all_tensors(replace_with_weak_ref, args, kwargs)

    def clone_tensors(self, args, kwargs=None):
        """Replace all tensors inside arg, kwargs with zeroed copies."""

        def clone_arg(arg):
            ten = torch.zeros_like(arg)
            ten.requires_grad = arg.requires_grad
            return ten

        return self._apply_to_all_tensors(clone_arg, args, kwargs)

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
                    attr
                    for field in fields(arg)
                    if torch.is_tensor(attr := getattr(arg, field.name))
                ]

            return []

        args_tens = [tensor for arg in args for tensor in extract_tensors(arg)] if args else []
        kwargs_tens = (
            [tensor for val in kwargs.values() for tensor in extract_tensors(val)] if kwargs else []
        )

        return tuple(chain(args_tens, kwargs_tens))


class CudaGraphManager(torch.nn.Module):
    """Creates and runs cudagraphs for a megatron module"""

    """A global mempool shared across all cuda graph captures."""
    global_mempool = None

    def __init__(
        self,
        config: TransformerConfig,
        base_module=None,
        function_name=None,
        params_to_calculate_wgrads=None,
        need_backward=True,
        num_warmup_steps=1,
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
        self.num_warmup_steps = num_warmup_steps

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

        # Without pipeline parallelism, the exeuction order is always fwd->bwd
        # Therefore modules will always execute in the same order, so cudagraphs
        # can both be reused and share a single mempool.
        self.reuse_cudagraphs = parallel_state.get_pipeline_model_parallel_world_size() == 1
        # Mempools are static so that multiple cudagraph managers may share the same mempool
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

    def get_cudagraph_runner(self, megatron_module):
        '''Returns a valid cudagraph runner for the current forward call.
        For 'reuse_cudagraphs' mode, we create a cudagraph for each call, if the module is called
        multiple times per step, for instance in the case of pipeline parallelism.
        The cudagraph corresponding to this call is the first element of 'self.cudagraph_runners'.
        We iterate through the list by 1 for each call, and the number of calls is equal to the
        length of 'self.cudagraph_runners'.
        Otherwise, we assign a mempool per microbatch, which allows cudagraphs to be reused
        over different microbatches by tracking their respective fwd and bwd passes.'''

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
                CudaGraphManager.global_mempool,
                self.func,
                self.params_to_calculate_wgrads,
                self.need_backward,
                self.num_warmup_steps,
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
                    CudaGraphManager.global_mempool,
                    self.func,
                    self.params_to_calculate_wgrads,
                    self.need_backward,
                    self.num_warmup_steps,
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

            if (
                self.training
                and torch.is_grad_enabled()
                and isinstance(megatron_module, torch.nn.Module)
            ):
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
