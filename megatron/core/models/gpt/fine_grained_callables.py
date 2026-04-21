# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.pipeline_parallel.utils import ScheduleNode
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor


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
    enable_deepep = layer.config.moe_enable_deepep

    def submodule_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor):
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

    def submodule_post_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Run forward pass for computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """
        if layer.recompute_pre_mlp_layernorm:
            layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                layer.pre_mlp_layernorm, hidden_states
            )
        else:
            pre_mlp_layernorm_output = layer.pre_mlp_layernorm(hidden_states)

        local_tokens, probs, _ = layer.mlp.router_and_preprocess(pre_mlp_layernorm_output)

        # Detach here for mlp_bda residual connection
        node.common_state.residual = node.detach(hidden_states)
        if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
            # Detach here for shared expert connection
            node.common_state.pre_mlp_layernorm_output = node.detach(pre_mlp_layernorm_output)

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

        return layer.mlp.dispatch(local_tokens, probs)

    def submodule_moe_forward(
        node: ScheduleNode, dispatched_tokens: torch.Tensor, probs: torch.Tensor
    ):
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

        pre_mlp_layernorm_output = getattr(node.common_state, 'pre_mlp_layernorm_output', None)
        expert_output, shared_expert_output, mlp_bias = layer.mlp.experts_compute(
            dispatched_tokens, probs, pre_mlp_layernorm_output
        )

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        # release tensor reference after use
        node.common_state.pre_mlp_layernorm_output = None
        if shared_expert_output is None:
            # Return only expert_output, since shared_expert_output causes backward on None
            return expert_output
        return expert_output, shared_expert_output

    def submodule_combine_forward(
        node: ScheduleNode,
        output: torch.Tensor,
        shared_expert_output: Optional[torch.Tensor] = None,
    ):
        """
        # Triggers token combine and the remaining computation in the transformer layer.
        # The `mlp_bda` computation is placed after `mlp.combine` due to data dependency.
        # This ordering is also critical for pipeline performance. Starting the `mlp.combine`
        # communication at first allows it to be overlapped with computation from another
        # microbatch. If `mlp_bda` were to run first, it would compete for SM resources
        # with another microbatch's computation and expose the communication.
        """
        residual = node.common_state.residual

        output = layer.mlp.combine(output, shared_expert_output)
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

    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
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
