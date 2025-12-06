# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import weakref
from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.utils import ScheduleNode, make_viewless
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    get_mtp_layer_offset,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor


def weak_method(method):
    """Creates a weak reference to a method to prevent circular references.

    This function creates a weak reference to a method and returns a wrapper function
    that calls the method when invoked. This helps prevent memory leaks from circular
    references.
    """
    method_ref = weakref.WeakMethod(method)
    del method

    def wrapped_func(*args, **kwarg):
        # nonlocal object_ref
        return method_ref()(*args, **kwarg)

    return wrapped_func


def should_free_input(name, is_moe, config):
    """Determine if the node should free its input memory.

    Args:
        name: Node name
        is_moe: Whether it's a MoE model
        config: TransformerConfig object

    Returns:
        bool: Whether to free input memory
    """
    # For dense layers [attn, fake, mlp, fake], the input is needed during backward pass
    if not is_moe:
        return False
    # Define which nodes should free input memory
    # Since we split the computing graph into multiple nodes, we can manually control
    # when and how to free the input memory.
    # The input and output of A2A are not needed anymore after the forward pass,
    # so we can free the input memory after the forward pass.
    free_input_nodes = {
        "mlp": True,
        "moe_combine": True,
        # For non-deepep mode, the input is the un-dispatched tokens and probs before dispatch A2A
        # and it's not needed anymore after the forward pass. If moe_preprocess is in cuda graph
        # scope, tokens and probs are fixed size tensors, so they cannot be freed.
        # For deepep mode, they are both needed in backward pass, so they cannot be freed.
        "moe_dispatch": config.moe_token_dispatcher_type == "alltoall"
        and (CudaGraphScope.moe_preprocess not in config.cuda_graph_scope),
    }

    return free_input_nodes.get(name, False)


class TransformerLayerState:
    """State shared within a transformer layer.

    This class holds state that is shared between different nodes
    within a transformer layer.
    """

    pass


class PreProcessNode(ScheduleNode):
    """Node responsible for preprocessing operations in the model.

    This node handles embedding and rotary positional embedding computations
    before the main transformer layers.
    """

    def __init__(self, gpt_model, chunk_state, event, stream):
        """Initializes a preprocessing node.

        Args:
            gpt_model: The GPT model instance.
            chunk_state (TransformerChunkState): State shared within a chunk
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="pre_process")
        self.gpt_model = gpt_model
        self.chunk_state = chunk_state

    def forward_impl(self):
        """forward pass for pre-processing.

        This method handles:
        1. Decoder embedding computation
        2. Rotary positional embedding computation
        3. Sequence length offset computation for flash decoding

        Returns:
            The processed decoder input tensor.
        """
        # Get decoder input
        if not self.gpt_model.pre_process:
            self.chunk_state.decoder_input = self.gpt_model.decoder.input_tensor
        # Run GPTModel._preprocess
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self.gpt_model._preprocess(
                input_ids=self.chunk_state.input_ids,
                position_ids=self.chunk_state.position_ids,
                decoder_input=self.chunk_state.decoder_input,
                packed_seq_params=self.chunk_state.packed_seq_params,
            )
        )

        # Saved for later use
        self.chunk_state.decoder_input = decoder_input
        self.chunk_state.rotary_pos_emb = rotary_pos_emb
        self.chunk_state.rotary_pos_cos = rotary_pos_cos
        self.chunk_state.rotary_pos_sin = rotary_pos_sin
        self.chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):
    """Node responsible for postprocessing operations in the model.

    This node handles final layer normalization and output layer computation
    after the main transformer layers.
    """

    def __init__(self, gpt_model, chunk_state, event, stream):
        """Initializes a postprocessing node.

        Args:
            gpt_model: The GPT model instance.
            chunk_state (TransformerChunkState): State shared within a chunk
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="post_process")
        self.gpt_model = gpt_model
        self.chunk_state = chunk_state

    def forward_impl(self, hidden_states):
        """Implements the forward pass for postprocessing.

        This method handles:
        1. Final layer normalization
        2. Output layer computation
        3. Loss computation if labels are provided

        Args:
            hidden_states: The hidden states from the transformer layers.

        Returns:
            The logits or loss depending on whether labels are provided.
        """
        # Final layer norm from Decoder
        if self.gpt_model.decoder.final_layernorm and not self.gpt_model.mtp_process:
            hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # Run GPTModel._postprocess
        loss = self.gpt_model._postprocess(
            hidden_states=hidden_states,
            input_ids=self.chunk_state.input_ids,
            position_ids=self.chunk_state.position_ids,
            labels=self.chunk_state.labels,
            decoder_input=self.chunk_state.decoder_input,
            rotary_pos_emb=self.chunk_state.rotary_pos_emb,
            rotary_pos_cos=self.chunk_state.rotary_pos_cos,
            rotary_pos_sin=self.chunk_state.rotary_pos_sin,
            mtp_in_postprocess=False,
            loss_mask=self.chunk_state.loss_mask,
            attention_mask=self.chunk_state.attention_mask,
            packed_seq_params=self.chunk_state.packed_seq_params,
            sequence_len_offset=self.chunk_state.sequence_len_offset,
            runtime_gather_output=self.chunk_state.runtime_gather_output,
            extra_block_kwargs=self.chunk_state.extra_block_kwargs,
        )

        # For now, 1f1b only supports fp16 module
        return float16_to_fp32(loss)


class TransformerLayerNode(ScheduleNode):
    """Base class for transformer layer computation nodes.

    This class provides common functionality for different types of
    transformer layer nodes (attention, MLP, etc.)
    """

    def __init__(
        self,
        stream,
        event,
        layer_state,
        chunk_state,
        submodule,
        name="default",
        bwd_dw_callables=None,
        extra_args={},
    ):
        """Initialize a transformer layer node.

        Args:
            stream (torch.cuda.Stream): CUDA stream for execution
            event (torch.cuda.Event): Synchronization event
            layer_state (TransformerLayerState): State shared within a layer
            chunk_state (TransformerChunkState): State shared within a chunk
            submodule (function): The submodule contain forward and dw function
            it's the per_batch_state_context, o.w. nullcontext
            name (str): Node name, also used to determine memory strategy
            bwd_dw_callables (list): List of weight gradient functions for the layer.
            extra_args (dict): Extra arguments for the node: is_moe, enable_deepep.
        """
        # determine whether to free input memory
        config = extra_args.get("config", None)
        assert config is not None, "model config must be passed to TransformerLayerNode."
        is_moe = extra_args.get("is_moe", False)
        free_input = should_free_input(name, is_moe, config)
        self.delay_wgrad_compute = extra_args.get("delay_wgrad_compute", False)

        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_input=free_input,
            name=name,
        )
        self.layer_state = layer_state
        self.chunk_state = chunk_state
        self.submodule = submodule
        self.detached = tuple()
        self.before_detached = tuple()

        # Create flags to indicate first and last layer
        self.is_first_layer = extra_args.get("is_first_layer", False)
        self.is_last_layer = extra_args.get("is_last_layer", False)

        # Initialize list to store registered dw callables
        self.bwd_dw_callables = []
        if bwd_dw_callables is not None:
            self.bwd_dw_callables = (
                bwd_dw_callables if isinstance(bwd_dw_callables, list) else [bwd_dw_callables]
            )

    def detach(self, t):
        """Detaches a tensor and stores it for backward computation."""
        detached = make_viewless(t).detach()
        detached.requires_grad = t.requires_grad
        self.before_detached = self.before_detached + (t,)
        self.detached = self.detached + (detached,)
        return detached

    def forward_impl(self, *args):
        """Calls the submodule as the forward pass."""
        return self.submodule(self, *args)

    def backward_impl(self, outputs, output_grad):
        """Implements the backward pass for the transformer layer node."""
        detached_grad = tuple([e.grad for e in self.detached])
        grads = output_grad + detached_grad
        self.default_backward_func(outputs + self.before_detached, grads)
        self._release_state()
        # return grads for record stream
        return grads

    def backward_dw(self):
        """Computes the weight gradients for the transformer layer node."""
        if not self.delay_wgrad_compute:
            return
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            for module in self.bwd_dw_callables:
                module.backward_dw()
        self.bwd_dw_callables = None

    def _release_state(self):
        # Release reference as early as possible, this helps avoid memory leak.
        self.before_detached = None
        self.detached = None
        self.layer_state = None
        self.chunk_state = None
        self.submodule = None


def build_transformer_layer_callables(layer: TransformerLayer):
    """Create callables for transformer layer nodes.
    Divides the transformer layer's operations into a sequence of smaller, independent
    functions. This decomposition separates computation-heavy tasks (e.g., self-attention,
    MLP) from communication-heavy tasks (e.g., MoE's All-to-All).

    The five callables are:
    1. Attention (computation)
    2. Post-Attention (computation)
    3. MoE Dispatch (communication)
    4. MLP / MoE Experts (computation)
    5. MoE Combine (communication)

    By assigning these functions to different CUDA streams (e.g., a compute stream
    and a communication stream), the scheduler can overlap their execution, preventing
    tasks from competing for resources and hiding communication latency by running them
    in parallel with functions from other micro-batches.

    Args:
        layer: The transformer layer to build callables for.

    Returns:
        A tuple containing:
        - forward_funcs: List of callable functions for the layer
        - backward_dw: Dict of weight gradient functions for the layer
    """

    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "deepep"
    )

    class _BackwardDWWrapper:
        def __init__(self):
            self.graphed_backward_dw_callable = None
            self.attn_dw_callable = layer.self_attention.backward_dw
            if isinstance(layer.mlp, MoELayer):
                self.shared_expert_dw_callable = partial(
                    layer.mlp.backward_dw, routed_experts=False, shared_experts=True
                )
            else:
                self.shared_expert_dw_callable = None
            self.cuda_graph_scope = layer.config.cuda_graph_scope

        def set_graphed_backward_dw_callable(self, graphed_backward_dw_callable):
            """Store the CUDA graphed backward weight gradient callable."""
            self.graphed_backward_dw_callable = graphed_backward_dw_callable

        def backward_dw(self):
            """Execute weight gradients, skipping CUDA graphed components during replay."""
            is_replay = hasattr(layer, 'cuda_graphs') and layer.cuda_graphs
            if self.shared_expert_dw_callable is not None and (
                not is_replay or CudaGraphScope.moe_router not in self.cuda_graph_scope
            ):
                self.shared_expert_dw_callable()
            if not is_replay or CudaGraphScope.attn not in self.cuda_graph_scope:
                self.attn_dw_callable()
            if is_replay and self.graphed_backward_dw_callable is not None:
                self.graphed_backward_dw_callable()

    attn_backward_dw_wrapper = _BackwardDWWrapper()

    def submodule_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Performs same attnention forward logic as GPT Model and forward pass for
        computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """

        if hasattr(layer, 'cuda_graphs') and layer.cuda_graphs:
            assert (
                CudaGraphScope.mlp not in layer.config.cuda_graph_scope
                and CudaGraphScope.moe not in layer.config.cuda_graph_scope
            ), (
                "Supported CUDA graph scope with EP overlap: "
                "attn, moe_router, moe_preprocess, mlp, got {}".format(
                    layer.config.cuda_graph_scope
                )
            )
            forward_func = layer._te_cuda_graph_replay
            attn_backward_dw_wrapper.set_graphed_backward_dw_callable(
                partial(layer.backward_dw_cudagraph, layer.current_microbatch)
            )
        else:
            # wrapper function that keeps consistent api with cuda graph replay
            def forward_func(
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                rotary_pos_emb: Optional[Tensor] = None,
                rotary_pos_cos: Optional[Tensor] = None,
                rotary_pos_sin: Optional[Tensor] = None,
                packed_seq_params: Optional[PackedSeqParams] = None,
                sequence_len_offset: Optional[Tensor] = None,
            ):
                hidden_states, _ = layer._forward_attention(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
                if not isinstance(layer.mlp, MoELayer):
                    return hidden_states, None, None, None
                if layer.recompute_pre_mlp_layernorm:
                    layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                    pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                        layer.pre_mlp_layernorm, hidden_states
                    )
                else:
                    pre_mlp_layernorm_output = layer.pre_mlp_layernorm(hidden_states)

                shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
                probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
                local_tokens, probs, _ = layer.mlp.preprocess(
                    pre_mlp_layernorm_output, probs, routing_map
                )
                return hidden_states, local_tokens, probs, shared_expert_output

        hidden_states, local_tokens, probs, shared_expert_output = forward_func(
            hidden_states=hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )
        if not isinstance(layer.mlp, MoELayer):
            return hidden_states

        # Detach here for mlp_bda residual connection
        node.layer_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            # Detach here for shared expert connection in moe_combine
            node.layer_state.shared_expert_output = node.detach(shared_expert_output)

        return local_tokens, probs

    def submodule_dispatch_forward(
        node: ScheduleNode, local_tokens: torch.Tensor, probs: torch.Tensor
    ):
        """
        Dispatches tokens to the experts based on the router output.
        """
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to attn submodule
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: torch.Tensor):
        """
        Run forward pass for computations between dispatch and combine:
            post dispatch->experts->combine preprocess
        """
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        expert_output, _ = layer.mlp.routed_experts_compute(
            dispatched_tokens, dispatched_probs, None
        )

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)
        # release tensor reference after use
        node.layer_state.dispatched_probs = None
        node.layer_state.pre_mlp_layernorm_output = None

        return expert_output

    def submodule_combine_forward(node: ScheduleNode, output: torch.Tensor):
        """
        # Triggers token combine and the remaining computation in the transformer layer.
        # The `mlp_bda` computation is placed after `mlp.combine` due to data dependency.
        # This ordering is also critical for pipeline performance. Starting the `mlp.combine`
        # communication at first allows it to be overlapped with computation from another
        # microbatch. If `mlp_bda` were to run first, it would compete for SM resources
        # with another microbatch's computation and expose the communication.
        """
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
        output = layer.mlp.combine(output, shared_expert_output)
        mlp_output_with_bias = (output, None)
        if hasattr(layer, 'cuda_graphs') and layer.cuda_graphs:
            layer.mlp.cudagraph_tensor_store.clear()
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record residual to comm stream, since it's created on comp stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.layer_state.residual = None
        return output

    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for Dense forward."""
        return layer._forward_mlp(*args, **kwargs)

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    attn_func = submodule_attn_forward
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_moe_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented

    forward_funcs = [attn_func, dispatch_func, mlp_func, combine_func, None]
    backward_dw = {"attn": attn_backward_dw_wrapper, "mlp": layer.mlp}
    return forward_funcs, backward_dw


def build_mtp_layer_callables(layer):
    """Callables for multi-token prediction layer nodes.

    This class contains the callable functions for different types of
    multi-token prediction layer nodes (attention, MLP, etc.)
    """

    forward_funcs, backward_dw = build_transformer_layer_callables(layer.transformer_layer)
    attn_forward, dispatch_forward, mlp_forward, combine_forward, _ = forward_funcs
    is_moe = isinstance(layer.transformer_layer.mlp, MoELayer)
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_attn_forward(node, hidden_states):
        # MTP Block Preprocess
        if node.is_first_layer:
            # Final layer norm from Decoder
            final_layernorm = node.chunk_state.model.decoder.final_layernorm
            if final_layernorm:
                hidden_states = final_layernorm(hidden_states)
                hidden_states = make_viewless_tensor(
                    inp=hidden_states, requires_grad=True, keep_graph=True
                )
                hidden_states = node.detach(hidden_states)
            offset = get_mtp_layer_offset(layer.config)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]

        input_ids, position_ids, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=node.chunk_state.input_ids,
            position_ids=node.chunk_state.position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
        )
        node.chunk_state.input_ids = input_ids
        node.chunk_state.position_ids = position_ids

        # MTP Layer Preprocess
        # norm, linear projection and transformer
        assert (
            node.chunk_state.context is None
        ), f"multi token prediction + cross attention is not yet supported."
        assert (
            node.chunk_state.packed_seq_params is None
        ), f"multi token prediction + sequence packing is not yet supported."

        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # fp8 context is added in 1f1b schedule, so we don't need to add it here
        with rng_context:
            hidden_states = layer._concat_embeddings(hidden_states, decoder_input)
            return attn_forward(node, hidden_states)

    def submodule_mtp_postprocess_forward(node, hidden_states):
        hidden_states = layer._postprocess(hidden_states)
        node.chunk_state.mtp_hidden_states.append(hidden_states)
        if node.is_last_layer:
            hidden_states = torch.cat(node.chunk_state.mtp_hidden_states, dim=0)
            node.chunk_state.mtp_hidden_states = None
        return hidden_states

    def rng_context_wrapper(func, *args, **kwargs):
        """
        Wrapper to add rng context to submodule callables
        """
        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()
        with rng_context:
            return func(*args, **kwargs)

    # Build forward and backward callable functions
    # attn_forward already has rng context, no need to wrap
    attn_func = submodule_mtp_attn_forward
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [attn_func, dispatch_func, mlp_func, combine_func, mtp_post_process_func]
    if isinstance(backward_dw["attn"], list):
        backward_dw["attn"].append(layer.eh_proj)
    else:
        backward_dw["attn"] = [backward_dw["attn"], layer.eh_proj]

    return forward_funcs, backward_dw


def build_layer_callables(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer and MultiTokenPredictionLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables(layer)
    elif isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
