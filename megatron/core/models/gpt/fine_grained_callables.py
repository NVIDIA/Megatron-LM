# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor
from megatron.core.typed_torch import apply_module, copy_signature


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
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )

    def submodule_attn_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Performs same attnention forward logic as GPT Model and forward pass for
        computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """

        if (
            isinstance(layer, GraphableMegatronModule)
            and hasattr(layer, 'cuda_graphs')
            and layer.cuda_graphs
        ):
            layer.set_te_cuda_graph_backward_dw_wrapper()
            forward_func = layer._te_cuda_graph_replay
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
                    with off_interface(
                        layer.offload_mlp_norm, hidden_states, "mlp_norm"
                    ) as hidden_states:
                        pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                            apply_module(layer.pre_mlp_layernorm), hidden_states
                        )
                else:
                    with off_interface(
                        layer.offload_mlp_norm, hidden_states, "mlp_norm"
                    ) as hidden_states:
                        pre_mlp_layernorm_output = apply_module(layer.pre_mlp_layernorm)(
                            hidden_states
                        )

                # When using fused residual norm (e.g. TEFusedResidualRMSNorm),
                # the layernorm returns (normalized_output, residual). Unpack
                # and use the fused residual for the downstream BDA connection.
                if isinstance(pre_mlp_layernorm_output, tuple):
                    if len(pre_mlp_layernorm_output) != 2:
                        raise ValueError(
                            f"When the output of pre_mlp_layernorm is a tuple, it is "
                            f"expected to have 2 elements (output, residual), but "
                            f"got {len(pre_mlp_layernorm_output)}"
                        )
                    pre_mlp_layernorm_output, hidden_states = pre_mlp_layernorm_output

                shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
                probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output)
                local_tokens, probs = layer.mlp.preprocess(
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
        if enable_deepep or enable_hybridep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to attn submodule
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)

        # `dispatched_probs` is needed by backward pass of swiglu, therefore it's
        # passed to moe_forward within `layer_state` to avoid the free_input process
        # of the input tensors.
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def submodule_moe_forward(node: ScheduleNode, dispatched_tokens: torch.Tensor):
        """
        Run forward pass for computations between dispatch and combine:
            post dispatch->experts->combine preprocess
        """
        dispatched_probs = node.layer_state.dispatched_probs
        token_dispatcher = layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

        # For HybridEP, tokens_per_expert is generated on comm stream, as the input to
        # `routed_experts_compute`, a ref is needed to prevent it from being freed.
        if enable_hybridep:
            tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

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
        output = layer.mlp.combine(output)
        output = layer.mlp.postprocess(output, shared_expert_output)

        mlp_output_with_bias = (output, None)
        if hasattr(layer, 'cuda_graphs') and layer.cuda_graphs:
            layer.mlp.cudagraph_tensor_store.clear()
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )
        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        if layer.offload_mlp_norm:
            hidden_states = off_interface.group_commit(
                hidden_states, name="mlp_norm", forced_released_tensors=[residual]
            )
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        # Need to record tensors created on comp stream to comm stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.layer_state.residual = None
        node.layer_state.shared_expert_output = None

        # final layer norm from decoder
        final_layernorm = node.chunk_state.model.decoder.final_layernorm
        if not node.is_mtp and final_layernorm and node.is_last_layer:
            output = final_layernorm(output)
            output = make_viewless_tensor(inp=output, requires_grad=True, keep_graph=True)
        return output

    @copy_signature(layer._forward_mlp, handle_first_dst_param='preserve')
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

    layer.init_backward_dw_wrapper()

    forward_funcs = [attn_func, dispatch_func, mlp_func, combine_func, None]
    backward_dw = {"attn": layer.backward_dw_wrapper, "mlp": layer.mlp}
    return forward_funcs, backward_dw

