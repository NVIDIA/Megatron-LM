# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.common.utils import (  # noqa: F401  # pylint: disable=unused-import
    PostProcessNode,
    PreProcessNode,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode, StageDispatchBwdGrad
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    make_viewless_tensor,
)
from megatron.core.typed_torch import apply_module, copy_signature
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


def finalize_decoder_layer_output(node, hidden_states):
    """Apply the decoder block boundary at whichever node is terminal for the layer.

    The decoder block boundary (mHC output contraction + final layer norm, see
    ``TransformerBlock.postprocess_for_layer_schedule``) must run on the last decoder
    layer regardless of whether that layer's terminal schedule node is the MoE combine,
    the standalone mHC-post, or the dense MLP. Embedding the boundary only in the MoE
    closures skips it for mixed patterns whose final layer is dense (for example
    ``moe_layer_freq=[1, 0]``), letting an uncontracted ``[s, b, n*h]`` tensor reach GPT
    postprocessing without ``learned_output_contract`` or the final layer norm. Factoring
    it here keeps the math independent of layer type and mHC-post placement.

    When MTP is enabled the boundary also produces the pre-contraction mHC multi-stream
    consumed by the MTP depths. That side output is detached at its producer so MTP reads
    a leaf; ``TransformerLayerNode.backward_impl`` reconnects the accumulated gradient when
    the scheduler runs this node's backward, exactly as ``residual`` / ``mlp_h_res`` /
    ``mlp_hc_h_post`` are bridged. Storing it undetached would let MTP backward traverse the
    decoder mHC graph out of schedule order, producing a second-backward error after saved
    tensors are freed or bypassing the point where the contracted and MTP branches merge.

    Args:
        node: The terminal ``TransformerLayerNode`` for the layer.
        hidden_states: The layer output prior to the decoder boundary.

    Returns:
        The node output: contracted + normalized ``[s, b, h]`` on the final decoder layer,
        otherwise a viewless view of ``hidden_states``.
    """
    # Layer nodes exist only for concrete layers; empty decoder chunks use PostProcessNode.
    # MTP layers finalize via submodule_mtp_postprocess_forward, not the decoder boundary.
    if node.is_mtp or not node.is_last_layer:
        return make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

    output, mhc_multistream = node.chunk_state.model.decoder.postprocess_for_layer_schedule(
        hidden_states, return_mhc_multistream=True
    )
    # postprocess_for_layer_schedule already makes final-layernorm outputs viewless; keep
    # this wrapper for the no-layernorm and mHC contraction-only exits.
    output = make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
    # Detach the pre-contraction multi-stream at its producer so MTP reads a leaf and this
    # node's backward_impl reconnects the accumulated gradient under scheduler control.
    node.chunk_state.mhc_multistream = (
        node.detach(mhc_multistream) if mhc_multistream is not None else None
    )
    return output


# The schedule-node classes, ``weak_method`` / ``should_free_input`` and the MTP /
# dispatch builders were relocated to ``megatron/core/models/common/{utils,
# fine_grained_callables}.py`` by main's combined-1F1B refactor (ffbe018c8) so
# HybridModel can share them; the mHC support that dev added to the copies here
# was grafted into the relocated versions. ``PreProcessNode`` / ``PostProcessNode``
# stay re-exported above because tests/unit_tests/pipeline_parallel/
# test_pp_mhc_compatibility.py imports them from this module.


def build_transformer_layer_callables(layer: TransformerLayer):
    """Create callables for transformer layer nodes.

    Divides the transformer layer's operations into a sequence of smaller, independent
    functions. This decomposition separates computation-heavy tasks (e.g., self-attention,
    MLP) from communication-heavy tasks (e.g., MoE's All-to-All).

    The five callable slots are:
    1. Attention and routing preprocess (computation)
    2. MoE Dispatch (communication)
    3. MLP / MoE Experts (computation)
    4. MoE Combine and MLP-side mHC post-processing (communication)
    5. MTP post-processing (computation, MTP layers only)

    By assigning these functions to different CUDA streams (e.g., a compute stream
    and a communication stream), the scheduler can overlap their execution, preventing
    tasks from competing for resources and hiding communication latency by running them
    in parallel with functions from other micro-batches.

    Args:
        layer: The transformer layer to build callables for.

    Returns:
        A tuple containing:
        - forward_funcs: List of 5 callables, one per slot in the schedule plan
          (pre_dispatch_computation, moe_dispatch, mlp, moe_combine,
          mtp_post_process=None).
        - backward_dw: Dict mapping slot name to the delayed-wgrad callable
          (keys: "pre_dispatch_computation", "mlp").
    """
    is_moe = isinstance(layer.mlp, MoELayer)
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend in ("deepep", "deepepv2")
    )
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )
    enable_ncclep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "ncclep"
    )
    is_hyper_connection_layer = isinstance(layer, HyperConnectionTransformerLayer)
    is_mhc_layer = is_moe and is_hyper_connection_layer

    def submodule_pre_dispatch_forward(node: ScheduleNode, hidden_states: torch.Tensor):
        """
        Performs the same attention forward logic as GPTModel and the forward pass for
        computations between attention and dispatch:
            pre mlp layernorm->router->dispatch preprocess
        """

        mhc_recompute_manager = getattr(node, "mhc_recompute_manager", None)
        is_last_in_mhc_recompute_group = getattr(
            node, "is_last_layer_in_mhc_recompute_group", False
        )
        if mhc_recompute_manager is not None:
            mhc_recompute_manager.is_last_layer_in_recompute_block = is_last_in_mhc_recompute_group

        using_cuda_graph_replay = (
            isinstance(layer, GraphableMegatronModule)
            and hasattr(layer, 'cuda_graphs')
            and layer.cuda_graphs
        )
        if using_cuda_graph_replay:
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
                mhc_recompute_manager=None,
            ):
                attention_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )
                if is_hyper_connection_layer:
                    attention_kwargs["mhc_recompute_manager"] = mhc_recompute_manager
                hidden_states, _ = layer._forward_attention(**attention_kwargs)
                if not isinstance(layer.mlp, MoELayer):
                    return hidden_states, None, None, None
                if is_mhc_layer:
                    nvtx_range_push(suffix="mlp_hyper_connection")
                    hidden_states, mlp_h_res, mlp_hc_h_post, residual = layer.mlp_hyper_connection(
                        hidden_states, mhc_recompute_manager=mhc_recompute_manager
                    )
                    nvtx_range_pop(suffix="mlp_hyper_connection")
                else:
                    mlp_h_res, mlp_hc_h_post = None, None
                    residual = hidden_states
                mlp_norm_manager = off_interface(layer.offload_mlp_norm, hidden_states, "mlp_norm")
                node.layer_state.mlp_norm_manager = mlp_norm_manager
                checkpoint_pre_mlp_layernorm = layer.recompute_pre_mlp_layernorm or (
                    mhc_recompute_manager is not None and layer.mhc_checkpoint_pre_mlp_layernorm
                )
                if checkpoint_pre_mlp_layernorm:
                    layer.pre_mlp_norm_checkpoint = tensor_parallel.CheckpointWithoutOutput(
                        ckpt_manager=mhc_recompute_manager
                    )
                    with mlp_norm_manager as hidden_states:
                        pre_mlp_layernorm_output = layer.pre_mlp_norm_checkpoint.checkpoint(
                            apply_module(layer.pre_mlp_layernorm), hidden_states
                        )
                else:
                    with mlp_norm_manager as hidden_states:
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
                    if not is_mhc_layer:
                        residual = hidden_states

                shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
                probs, routing_map = layer.mlp.route(
                    pre_mlp_layernorm_output, padding_mask=node.chunk_state.padding_mask
                )
                local_tokens, probs = layer.mlp.preprocess(
                    pre_mlp_layernorm_output, probs, routing_map
                )
                if is_mhc_layer:
                    return (
                        residual,
                        local_tokens,
                        probs,
                        shared_expert_output,
                        mlp_h_res,
                        mlp_hc_h_post,
                    )
                return hidden_states, local_tokens, probs, shared_expert_output

        forward_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=node.chunk_state.attention_mask,
            rotary_pos_emb=node.chunk_state.rotary_pos_emb,
            rotary_pos_cos=node.chunk_state.rotary_pos_cos,
            rotary_pos_sin=node.chunk_state.rotary_pos_sin,
            packed_seq_params=node.chunk_state.packed_seq_params,
            sequence_len_offset=node.chunk_state.sequence_len_offset,
        )
        if is_hyper_connection_layer and (
            not using_cuda_graph_replay
            or CudaGraphModule.attn not in layer.config.cuda_graph_modules
        ):
            forward_kwargs["mhc_recompute_manager"] = mhc_recompute_manager
        forward_outputs = forward_func(**forward_kwargs)
        if is_mhc_layer:
            hidden_states, local_tokens, probs, shared_expert_output, mlp_h_res, mlp_hc_h_post = (
                forward_outputs
            )
        else:
            hidden_states, local_tokens, probs, shared_expert_output = forward_outputs
            mlp_h_res, mlp_hc_h_post = None, None
        if not isinstance(layer.mlp, MoELayer):
            return hidden_states

        # Detach here for mlp_bda residual connection
        node.layer_state.residual = node.detach(hidden_states)
        if is_mhc_layer:
            node.layer_state.mlp_h_res = node.detach(mlp_h_res)
            node.layer_state.mlp_hc_h_post = node.detach(mlp_hc_h_post)
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
        if enable_deepep or enable_hybridep or enable_ncclep:
            # update token_probs to be the detached version, prevents
            # backward graph from connecting to pre_dispatch_computation submodule
            token_dispatcher._comm_manager.token_probs = probs

        dispatched_tokens, dispatched_probs = layer.mlp.dispatch(local_tokens, probs)

        if enable_ncclep and layer.config.moe_ncclep_zero_copy:
            # Insert an identity node as the sole consumer of the dispatch output, so the
            # dispatch-backward gets the symm buffer instead of a non-symm AccumulateGrad clone.
            # Must stay inside this node's graph segment (before the next node detaches it).
            dispatched_tokens = StageDispatchBwdGrad.apply(dispatched_tokens, token_dispatcher)

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
        if enable_deepep or enable_hybridep or enable_ncclep:
            # update dispatched_probs to be detached version, prevents
            # backward graph from connecting to dispatch submodule
            token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

        expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

        # For HybridEP and NCCL EP, tokens_per_expert is generated on comm stream, as the
        # input to `routed_experts_compute`, a ref is needed to prevent it from being freed.
        if enable_hybridep or enable_ncclep:
            tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            node.layer_state.tokens_per_expert = tokens_per_expert

        if layer.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of expert_output
            layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

        return expert_output

    def submodule_combine_forward(node: ScheduleNode, output: torch.Tensor):
        """
        Trigger token combine and the remaining layer computation.

        MHC post-processing stays in this communication-stream node so it preserves the
        existing EP overlap stream topology.
        """
        residual = node.layer_state.residual
        shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
        output = layer.mlp.combine(output)
        output = layer.mlp.postprocess(output, shared_expert_output)

        if hasattr(layer, 'cuda_graphs') and layer.cuda_graphs:
            layer.mlp.cudagraph_tensor_store.clear()

        if shared_expert_output is not None:
            shared_expert_output.record_stream(torch.cuda.current_stream())
        node.layer_state.shared_expert_output = None

        if is_mhc_layer:
            return submodule_mhc_post_forward(node, output)

        mlp_output_with_bias = (output, None)
        with layer.bias_dropout_add_exec_handler():
            hidden_states = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, layer.hidden_dropout
            )

        # Delay the offload of the mlp norm until after the mlp_bda has been computed
        # because the residual is needed in the mlp_bda.
        mlp_norm_manager = getattr(node.layer_state, 'mlp_norm_manager', None)
        if mlp_norm_manager is not None:
            hidden_states = mlp_norm_manager.group_offload(
                hidden_states, forced_released_tensors=[residual]
            )
            node.layer_state.mlp_norm_manager = None
        output = finalize_decoder_layer_output(node, hidden_states)

        # Need to record tensors created on comp stream to comm stream
        node.layer_state.residual.record_stream(torch.cuda.current_stream())

        # release tensor reference after use
        node.layer_state.residual = None
        return output

    def submodule_mhc_post_forward(node: ScheduleNode, output: torch.Tensor):
        """Run MLP-side mHC post-processing after combine communication completes."""
        residual = node.layer_state.residual
        manager = getattr(node, "mhc_recompute_manager", None)
        is_group_end = getattr(node, "is_last_layer_in_mhc_recompute_group", False)
        bda_manager = None if is_group_end else manager
        hidden_states = layer._forward_mhc_mlp_post(
            output,
            node.layer_state.mlp_h_res,
            residual,
            node.layer_state.mlp_hc_h_post,
            bda_manager,
        )

        mlp_norm_manager = getattr(node.layer_state, 'mlp_norm_manager', None)
        if mlp_norm_manager is not None:
            hidden_states = mlp_norm_manager.group_offload(
                hidden_states, forced_released_tensors=[residual]
            )
            node.layer_state.mlp_norm_manager = None

        output = finalize_decoder_layer_output(node, hidden_states)

        node.layer_state.residual.record_stream(torch.cuda.current_stream())
        node.layer_state.mlp_h_res.record_stream(torch.cuda.current_stream())
        node.layer_state.mlp_hc_h_post.record_stream(torch.cuda.current_stream())
        node.layer_state.residual = None
        node.layer_state.mlp_h_res = None
        node.layer_state.mlp_hc_h_post = None

        if manager is not None and is_group_end:
            manager.discard_all_outputs()

        return output

    @copy_signature(layer._forward_mlp, handle_first_dst_param='preserve')
    def mlp_wrapper(node: ScheduleNode, *args, **kwargs):
        """Wrapper for dense forward with explicit mHC recompute management."""
        manager = (
            getattr(node, "mhc_recompute_manager", None) if is_hyper_connection_layer else None
        )
        if manager is not None:
            manager.is_last_layer_in_recompute_block = getattr(
                node, "is_last_layer_in_mhc_recompute_group", False
            )
            kwargs["mhc_recompute_manager"] = manager
        output = layer._forward_mlp(*args, **kwargs)
        # Dense layers are terminal for their own layer, so the decoder boundary (mHC
        # contraction + final layer norm) must be applied here for a dense final layer.
        output = finalize_decoder_layer_output(node, output)
        if manager is not None and getattr(node, "is_last_layer_in_mhc_recompute_group", False):
            manager.discard_all_outputs()
        return output

    def raise_not_implemented(*args):
        """Raise NotImplementedError for Dense layer."""
        raise NotImplementedError("This callable is not implemented for Dense layer.")

    # Build forward and backward callable functions
    pre_dispatch_func = submodule_pre_dispatch_forward
    dispatch_func = submodule_dispatch_forward if is_moe else raise_not_implemented
    mlp_func = submodule_moe_forward if is_moe else mlp_wrapper
    combine_func = submodule_combine_forward if is_moe else raise_not_implemented

    layer.init_backward_dw_wrapper()

    forward_funcs = [pre_dispatch_func, dispatch_func, mlp_func, combine_func, None]
    backward_dw = {"pre_dispatch_computation": layer.backward_dw_wrapper, "mlp": layer.mlp}
    return forward_funcs, backward_dw
