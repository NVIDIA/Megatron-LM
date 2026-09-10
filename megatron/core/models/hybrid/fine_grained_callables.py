# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import is_first_last_bf16_layer
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.fine_grained_callables import (
    PostProcessNode,
    PreProcessNode,
    build_transformer_layer_callables,
    finalize_decoder_layer_output,
)
from megatron.core.transformer.module import GraphableMegatronModule, float16_to_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_layer import TransformerLayer


class HybridPostProcessNode(PostProcessNode):
    """Use HybridModel's ordinary output/loss path at the schedule boundary."""

    def forward_impl(self, hidden_states):
        """Finish the Hybrid decoder and compute logits or per-token loss."""
        model = self.gpt_model
        if len(model.decoder.layers) == 0:
            hidden_states = model.decoder.postprocess_for_layer_schedule(hidden_states)
        state = self.chunk_state
        return float16_to_fp32(
            model._postprocess(
                hidden_states,
                input_ids=state.input_ids,
                position_ids=state.position_ids,
                labels=state.labels,
                attention_mask=state.attention_mask,
                rotary_pos_emb=state.rotary_pos_emb,
                packed_seq_params=state.packed_seq_params,
                loss_mask=state.loss_mask,
                padding_mask=state.padding_mask,
                runtime_gather_output=state.runtime_gather_output,
            )
        )


class HybridModelChunkSchedulePlan(TransformerModelChunkSchedulePlan):
    """Schedule the supported DSv4 Hybrid decoder using the common 1F1B engine."""

    def __init__(self, model, *args, **kwargs):
        config = model.config
        if any(symbol not in "CEHW-" for symbol in model.decoder.layer_type_list):
            raise ValueError("Hybrid EP overlap currently supports DSv4 C/H/W, E and - layers")
        if config.mtp_num_layers:
            raise ValueError("Hybrid EP overlap does not support MTP")
        if config.moe_n_hash_layers:
            raise ValueError("Hybrid EP overlap does not support hash MoE layers")
        if config.delay_wgrad_compute:
            raise ValueError("Hybrid EP overlap does not support delayed weight gradients")
        if (
            config.recompute_granularity is not None
            or config.fine_grained_activation_offloading
            or config.cpu_offloading
        ):
            raise ValueError(
                "Hybrid EP overlap does not yet support recompute or activation offload"
            )
        if config.fp8 and config.fp8_recipe == Fp8Recipe.delayed:
            raise ValueError("Hybrid EP overlap requires a current-scaling FP8 recipe")
        if config.fp4:
            raise ValueError("Hybrid EP overlap does not support FP4")
        super().__init__(model, *args, **kwargs)
        if config.fp8 and config.first_last_layers_bf16:
            for index in range(self.num_layers()):
                layer_plan = self.get_layer(index)
                if is_first_last_bf16_layer(config, layer_plan.layer.layer_number - 1):
                    # The shared schedule assumes FP8 experts save quantized inputs.
                    # BF16 boundary layers can instead save the dispatch buffer
                    # directly, so keep its storage until their backward completes.
                    layer_plan.mlp.free_input = False

    @staticmethod
    def _get_pre_post_process_nodes():
        return PreProcessNode, HybridPostProcessNode


def build_hybrid_layer_callables(layer):
    """Split an mHC Hybrid layer at its actual compute/communication boundaries.

    Attention/dense wrappers execute in one compute node. A MoE wrapper retains
    its mHC state across routing, dispatch, experts, combine and the final compute
    node. The latter owns the n-stream residual; the inner layer's local BDA must
    not run a second time. Unlike TransformerBlock, ordinary HybridStack does not
    fork the TP RNG around its layers. Preserve that default RNG stream here,
    including sequence-parallel dropout and router jitter.
    """
    inner = layer.inner_layer
    if not isinstance(inner, TransformerLayer):
        raise ValueError("Hybrid EP overlap currently supports TransformerLayer-backed DSv4 layers")

    def forward_kwargs(node):
        state = node.chunk_state
        return dict(
            attention_mask=state.attention_mask,
            rotary_pos_emb=state.rotary_pos_emb,
            packed_seq_params=state.packed_seq_params,
            padding_mask=state.padding_mask,
        )

    if not isinstance(inner.mlp, MoELayer):

        def compute(node, hidden_states):
            hidden_states, context = layer(hidden_states, **forward_kwargs(node))
            assert context is None, "Hybrid EP overlap does not support cross-attention"
            return hidden_states

        def finish(node, hidden_states):
            output = finalize_decoder_layer_output(node, hidden_states)
            if (
                node.is_last_layer
                and layer.config.deallocate_pipeline_outputs
                and output is hidden_states
            ):
                # A non-final pipeline chunk has no output contraction or norm.
                # Its boundary would return this node's detached input leaf.
                # Pseudo-deallocating that leaf changes its gradient shape to
                # [1]; retain an autograd edge to the original-shaped input.
                output = output.clone()
            return output

        return [compute, None, finish, None, None, None], {}

    moe = inner.mlp
    inner_callables, _ = build_transformer_layer_callables(inner)
    dispatch, experts = inner_callables[1:3]

    def prefix(node, hidden_states):
        if layer.training and getattr(layer, "cuda_graphs", None):
            kwargs = forward_kwargs(node)
            layer._decompose_packed_seq_params_to_kwargs(kwargs)
            outputs = list(
                GraphableMegatronModule._te_cuda_graph_replay(layer, hidden_states, **kwargs)
            )
            residual, h_res, h_post = outputs.pop(), outputs.pop(), outputs.pop()
            hidden_states, probs, routing_map, shared_output = (
                inner.restore_moe_prefix_after_partial_cudagraph(outputs)
            )
            probs, routing_map = moe.route(hidden_states)
            local_tokens, probs = moe.preprocess(hidden_states, probs, routing_map)
        else:
            aggregated, h_res, h_post, residual = layer.hyper_connection(hidden_states)
            normalized = inner.pre_mlp_layernorm(aggregated)
            if isinstance(normalized, tuple):
                normalized, _ = normalized
            shared_output = moe.shared_experts_compute(normalized)
            padding_mask = moe._normalize_padding_mask(normalized, node.chunk_state.padding_mask)
            probs, routing_map = moe.route(normalized, padding_mask=padding_mask)
            local_tokens, probs = moe.preprocess(normalized, probs, routing_map)

        node.layer_state.residual = node.detach(residual)
        node.layer_state.mlp_h_res = node.detach(h_res)
        node.layer_state.mlp_hc_h_post = node.detach(h_post)
        if shared_output is not None:
            node.layer_state.shared_expert_output = node.detach(shared_output)
        return local_tokens, probs

    def combine(node, expert_output):
        output = moe.combine(expert_output)
        shared_output = getattr(node.layer_state, "shared_expert_output", None)
        output = moe.postprocess(output, shared_output)
        moe.cudagraph_tensor_store.clear()
        if shared_output is not None:
            shared_output.record_stream(torch.cuda.current_stream())
        node.layer_state.shared_expert_output = None
        return output

    def post(node, output):
        state = node.layer_state
        hidden_states = layer.hyper_connection.fused_h_res_h_post_bda(
            state.mlp_h_res,
            state.residual,
            state.mlp_hc_h_post,
            (output, None),
            dropout_prob=inner.hidden_dropout,
            training=layer.training,
            fused=inner.config.bias_dropout_fusion,
            manager=None,
        )
        if layer.config.fp32_residual_connection and layer.config.params_dtype is not None:
            hidden_states = hidden_states.to(layer.config.params_dtype)
        hidden_states = finalize_decoder_layer_output(node, hidden_states)
        for name in ("residual", "mlp_h_res", "mlp_hc_h_post"):
            getattr(state, name).record_stream(torch.cuda.current_stream())
            setattr(state, name, None)
        return hidden_states

    return [prefix, dispatch, experts, combine, None, post], {}
