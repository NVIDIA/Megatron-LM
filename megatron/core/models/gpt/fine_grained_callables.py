import weakref

import torch

from megatron.core.pipeline_parallel.utils import ScheduleNode, make_viewless
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
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


def should_free_input(name, is_moe, is_deepep):
    """Determine if the node should free its input memory.

    Args:
        name: Node name
        is_moe: Whether it's a MoE model
        is_deepep: Whether it's a DeepEP model

    Returns:
        bool: Whether to free input memory
    """
    # For dense layers [attn, fake, mlp, fake], mlp input is needed during backward pass
    if not is_moe:
        return False
    # Define which nodes should free input memory
    free_input_nodes = {
        "mlp": True,  # Free input after MLP node usage
        "combine": True,  # Free input after Combine node usage
        "dispatch": not is_deepep,  # Free input after dispatch node usage in non-deepep mode
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

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        """Initializes a preprocessing node.

        Args:
            gpt_model: The GPT model instance.
            model_chunk_state: State shared across the model chunk.
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="pre_process")
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

    def forward_impl(self):
        """Implements the forward pass for preprocessing.

        This method handles:
        1. Decoder embedding computation
        2. Rotary positional embedding computation
        3. Sequence length offset computation for flash decoding

        Returns:
            The processed decoder input tensor.
        """
        # Get decoder input
        if not self.gpt_model.pre_process:
            self.model_chunk_state.decoder_input = self.gpt_model.decoder.input_tensor
        # Run GPTModle._preprocess
        decoder_input, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset = (
            self.gpt_model._preprocess(
                input_ids=self.model_chunk_state.input_ids,
                position_ids=self.model_chunk_state.position_ids,
                decoder_input=self.model_chunk_state.decoder_input,
                packed_seq_params=self.model_chunk_state.packed_seq_params,
            )
        )

        # Saved for later use
        self.model_chunk_state.decoder_input = decoder_input
        self.model_chunk_state.rotary_pos_emb = rotary_pos_emb
        self.model_chunk_state.rotary_pos_cos = rotary_pos_cos
        self.model_chunk_state.rotary_pos_sin = rotary_pos_sin
        self.model_chunk_state.sequence_len_offset = sequence_len_offset
        return decoder_input


class PostProcessNode(ScheduleNode):
    """Node responsible for postprocessing operations in the model.

    This node handles final layer normalization and output layer computation
    after the main transformer layers.
    """

    def __init__(self, gpt_model, model_chunk_state, event, stream):
        """Initializes a postprocessing node.

        Args:
            gpt_model: The GPT model instance.
            model_chunk_state: State shared across the model chunk.
            event: CUDA event for synchronization.
            stream: CUDA stream for execution.
        """
        super().__init__(weak_method(self.forward_impl), stream, event, name="post_process")
        self.gpt_model = gpt_model
        self.model_chunk_state = model_chunk_state

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
        labels = self.model_chunk_state.labels
        # Final layer norm from Decoder
        if self.gpt_model.decoder.final_layernorm is not None:
            hidden_states = self.gpt_model.decoder.final_layernorm(hidden_states)
            # TENorm produces a "viewed" tensor. This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        # Run GPTModle._postprocess
        loss = self.gpt_model._postprocess(
            hidden_states=hidden_states,
            input_ids=self.model_chunk_state.input_ids,
            position_ids=self.model_chunk_state.position_ids,
            labels=labels,
            decoder_input=self.model_chunk_state.decoder_input,
            rotary_pos_emb=self.model_chunk_state.rotary_pos_emb,
            rotary_pos_cos=self.model_chunk_state.rotary_pos_cos,
            rotary_pos_sin=self.model_chunk_state.rotary_pos_sin,
            use_mtp=False,  # MTP changes assumptions of final layer norm, not supported for now
            loss_mask=self.model_chunk_state.loss_mask,
            attention_mask=self.model_chunk_state.attention_mask,
            packed_seq_params=self.model_chunk_state.packed_seq_params,
            sequence_len_offset=self.model_chunk_state.sequence_len_offset,
            runtime_gather_output=self.model_chunk_state.runtime_gather_output,
            extra_block_kwargs=self.model_chunk_state.extra_block_kwargs,
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
        common_state,
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
            common_state (TransformerLayerState): State shared within a transformer layer
            submodule (function): The submodule contain forward and dw function
            it's the per_batch_state_context, o.w. nullcontext
            name (str): Node name, also used to determine memory strategy
        """
        # determine whether to free input memory
        is_moe = extra_args.get("is_moe", False)
        enable_deepep = extra_args.get("enable_deepep", False)
        free_input = should_free_input(name, is_moe, enable_deepep)

        super().__init__(
            weak_method(self.forward_impl),
            stream,
            event,
            weak_method(self.backward_impl),
            free_input=free_input,
            name=name,
        )
        self.common_state = common_state
        self.chunk_state = chunk_state
        self.submodule = submodule
        self.detached = tuple()
        self.before_detached = tuple()

        # Create flags to indicate first and last layer
        self.is_first = extra_args.get("is_first", False)
        self.is_last = extra_args.get("is_last", False)

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
        self.before_detached = None
        self.detached = None
        # return grads for record stream
        return grads

    def backward_dw(self):
        """Computes the weight gradients for the transformer layer node."""
        with torch.cuda.nvtx.range(f"{self.name} wgrad"):
            for module in self.bwd_dw_callables:
                module.backward_dw()


def build_transformer_layer_callables(layer):
    """
    Create callables for transformer layer nodes.
    """

    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = layer.config.moe_enable_deepep

    def submodule_attn_forward(node, hidden_states):
        """
        Performs same attnention forward logic as GPT Model.
        """
        hidden_states, _ = layer._forward_attention(
            hidden_states=hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            attention_bias=node.chunk_state.attention_bias,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )
        return hidden_states

    def submodule_post_attn_forward(node, hidden_states):
        """
        Run forward pass for computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """
        pre_mlp_layernorm_output = layer._pre_mlp_layernorm_maybe_recompute(hidden_states)
        probs, routing_map = layer.mlp.router(pre_mlp_layernorm_output)
        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of unpermuted probs
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(probs)

        local_tokens, probs = layer.mlp.token_dispatcher.dispatch_preprocess(
            pre_mlp_layernorm_output, routing_map, probs
        )

        # Detach here for mlp_bda residual connection
        node.common_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert:
            # Detach here for shared expert connection
            node.common_state.pre_mlp_layernorm_output = node.detach(pre_mlp_layernorm_output)

        return local_tokens, probs

    def submodule_dispatch_forward(node, local_tokens, probs):
        """
        Dispatches tokens to the experts based on the router output.
        """
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to attn submodule
            token_dispatcher._comm_manager.token_probs = probs

        return token_dispatcher.dispatch_all_to_all(local_tokens, probs)

    def submodule_moe_forward(node, dispatched_tokens, probs):
        """
        Run forward pass for computations between dispatch and combine:
            post dispatch->experts->combine preprocess
        """
        shared_expert_output = None
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = probs
        dispatched_tokens, tokens_per_expert, permuted_probs = (
            token_dispatcher.dispatch_postprocess(dispatched_tokens, probs)
        )
        expert_output, mlp_bias = layer.mlp.experts(
            dispatched_tokens, tokens_per_expert, permuted_probs
        )
        assert mlp_bias is None, f"Bias is not supported in {token_dispatcher.__class__.__name__}"
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            shared_expert_output = layer.mlp.shared_experts(
                node.common_state.pre_mlp_layernorm_output
            )
        expert_output = layer.mlp.token_dispatcher.combine_preprocess(expert_output)

        # release tensor reference after use
        node.common_state.pre_mlp_layernorm_output = None
        if shared_expert_output is None:
            # Return only expert_output, since shared_expert_output causes backward on None
            return expert_output
        return expert_output, shared_expert_output

    def submodule_combine_forward(node, output, shared_expert_output=None):
        """
        Combines tokens and performs post processing.
        """
        residual = node.common_state.residual
        token_dispatcher = layer.mlp.token_dispatcher
        output = token_dispatcher.combine_all_to_all(output)
        output = token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        mlp_output_with_bias = (output, None)
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record residual to comm stream, since it's created on comp stream
        node.common_state.residual.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.common_state.residual = None
        return output

    def mlp_wrapper(node, *args, **kwargs):
        """Wrapper for Dense forward."""
        return layer._forward_mlp(*args, **kwargs)

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    attn_func = submodule_attn_forward
    post_attn_func = submodule_post_attn_forward if is_moe else raise_not_implemented
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_moe_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented

    forward_funcs = [attn_func, post_attn_func, dispatch_func, mlp_func, combine_func, None]
    backward_dw = {"attn": layer.self_attention, "mlp": layer.mlp}
    return forward_funcs, backward_dw


def build_layer_callables(layer):
    """
    Builds the callable functions(forward and dw) for the given layer.
    For now, 1f1b overlap only support TransformerLayer.

    Args:
        layer: The layer to build callables for.

    Returns:
        forward_funcs: list of callable functions for the layer.
        backward_dw: dict of weight gradient functions for the layer.
    """
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
