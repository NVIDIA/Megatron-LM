# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
from torch import Tensor

from megatron.core.enums import Fp8Recipe
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import (
    LayerPatternItem,
    Symbols as LayerSymbols,
    is_layer_group,
)
from megatron.core.pipeline_parallel.utils import ScheduleNode
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor


def _get_inner_quant_context(layer):
    config = layer.config
    if config.fp8 and config.fp8_recipe != Fp8Recipe.delayed:
        return get_fp8_context(config, layer.layer_number - 1)
    if config.fp4:
        return get_fp4_context(config, layer.layer_number - 1)
    return nullcontext()


def _as_hybrid_layers(layer, layer_type: Optional[LayerPatternItem]):
    """Return ``(layer_type, layer)`` pairs for a hybrid logical layer."""
    if isinstance(layer, HybridStack):
        return list(zip(layer.layer_type_list, layer.layers))
    assert layer_type is not None, "Hybrid layer scheduling requires the layer type symbol."
    return [(layer_type, layer)]


def _apply_attention_layer(
    layer: TransformerLayer,
    node: ScheduleNode,
    hidden_states: Tensor,
):
    hidden_states, _ = layer._forward_attention(
        hidden_states=hidden_states,
        attention_mask=node.chunk_state.attention_mask,
        rotary_pos_emb=node.chunk_state.rotary_pos_emb,
        rotary_pos_cos=node.chunk_state.rotary_pos_cos,
        rotary_pos_sin=node.chunk_state.rotary_pos_sin,
        packed_seq_params=node.chunk_state.packed_seq_params,
        sequence_len_offset=node.chunk_state.sequence_len_offset,
    )
    return hidden_states


def _apply_mamba_layer(layer, node: ScheduleNode, hidden_states: Tensor):
    return layer(
        hidden_states=hidden_states,
        attention_mask=node.chunk_state.attention_mask,
        inference_context=getattr(node.chunk_state, "inference_context", None),
        packed_seq_params=node.chunk_state.packed_seq_params,
    )


def _maybe_apply_final_norm(node: ScheduleNode, hidden_states: Tensor):
    final_norm = getattr(node.chunk_state.model.decoder, "final_norm", None)
    final_norm = final_norm or getattr(node.chunk_state.model.decoder, "final_layernorm", None)
    if not node.is_mtp and final_norm is not None and node.is_last_layer:
        hidden_states = final_norm(hidden_states)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )
    return hidden_states


def _get_moe_padding_mask(node: ScheduleNode):
    padding_mask = node.chunk_state.padding_mask
    if padding_mask is not None:
        # MoELayer.forward receives [batch, seq] and transposes before routing.
        padding_mask = padding_mask.transpose(0, 1).bool()
    return padding_mask


class _SharedExpertBackwardDWWrapper:
    """Backward weight-gradient wrapper for MoE-only hybrid terminal layers."""

    def __init__(self, layer):
        self.layer = layer
        self.shared_expert_dw_callable = None
        if layer.mlp.use_shared_expert:
            self.shared_expert_dw_callable = partial(
                layer.mlp.backward_dw, routed_experts=False, shared_experts=True
            )

    def backward_dw(self):
        if self.shared_expert_dw_callable is not None:
            self.shared_expert_dw_callable()
        self.layer = None
        self.shared_expert_dw_callable = None


def _run_moe_preprocess(layer, node: ScheduleNode, hidden_states: Tensor):
    pre_mlp_layernorm_output = layer._forward_pre_mlp_layernorm(hidden_states)
    if isinstance(pre_mlp_layernorm_output, tuple):
        if len(pre_mlp_layernorm_output) != 2:
            raise ValueError(
                f"When the output of pre_mlp_layernorm is a tuple, it is expected to have "
                f"2 elements (output, residual), but got {len(pre_mlp_layernorm_output)}"
            )
        pre_mlp_layernorm_output, residual = pre_mlp_layernorm_output
    else:
        residual = hidden_states

    if layer.config.fp32_residual_connection:
        residual = residual.float()

    shared_expert_output = layer.mlp.shared_experts_compute(pre_mlp_layernorm_output)
    probs, routing_map = layer.mlp.route(pre_mlp_layernorm_output, _get_moe_padding_mask(node))
    local_tokens, probs = layer.mlp.preprocess(pre_mlp_layernorm_output, probs, routing_map)

    node.layer_state.residual = node.detach(residual)
    if layer.mlp.use_shared_expert and not layer.mlp.shared_expert_overlap:
        node.layer_state.shared_expert_output = node.detach(shared_expert_output)

    return local_tokens, probs


def _run_moe_experts(layer, node: ScheduleNode, dispatched_tokens: Tensor):
    dispatched_probs = node.layer_state.dispatched_probs
    enable_hybridep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "hybridep"
    )
    enable_deepep = (
        layer.config.moe_token_dispatcher_type == "flex"
        and layer.config.moe_flex_dispatcher_backend == "deepep"
    )
    token_dispatcher = layer.mlp.token_dispatcher
    if enable_deepep or enable_hybridep:
        token_dispatcher._comm_manager.dispatched_probs = dispatched_probs

    expert_output, _ = layer.mlp.routed_experts_compute(dispatched_tokens, dispatched_probs)

    if enable_hybridep:
        tokens_per_expert = token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
        node.layer_state.tokens_per_expert = tokens_per_expert

    if layer.recompute_pre_mlp_layernorm:
        layer.pre_mlp_norm_checkpoint.discard_output_and_register_recompute(expert_output)

    return expert_output


def _run_moe_combine(layer, node: ScheduleNode, output: Tensor):
    residual = node.layer_state.residual
    shared_expert_output = getattr(node.layer_state, 'shared_expert_output', None)
    output = layer.mlp.combine(output)
    output = layer.mlp.postprocess(output, shared_expert_output)
    output = layer._forward_post_mlp((output, None), residual)

    node.layer_state.residual.record_stream(torch.cuda.current_stream())
    if shared_expert_output is not None:
        shared_expert_output.record_stream(torch.cuda.current_stream())

    node.layer_state.residual = None
    node.layer_state.shared_expert_output = None

    return _maybe_apply_final_norm(node, output)


def build_hybrid_stack_callables(layer, layer_type: Optional[LayerPatternItem] = None):
    """Create fine-grained callables for one logical HybridStack layer.

    A logical layer may be a bracketed nested ``HybridStack`` (for example ``[M*E]``)
    or a single legacy hybrid layer symbol. The split is:
    pre-dispatch compute -> dispatch -> MLP/experts -> combine.
    """
    layer_items = _as_hybrid_layers(layer, layer_type)
    if any(is_layer_group(item_type) for item_type, _ in layer_items):
        raise ValueError("Nested HybridStack groups are not supported in overlap scheduling.")

    terminal_idx = None
    for idx, (item_type, _) in enumerate(layer_items):
        if item_type in (LayerSymbols.MLP, LayerSymbols.MOE):
            terminal_idx = idx
            break

    if terminal_idx is not None and terminal_idx != len(layer_items) - 1:
        raise ValueError("HybridStack overlap requires MLP/MoE to be the last layer in a group.")

    terminal_type = layer_items[terminal_idx][0] if terminal_idx is not None else None
    terminal_layer = layer_items[terminal_idx][1] if terminal_idx is not None else None
    pre_layers = layer_items[:terminal_idx] if terminal_idx is not None else layer_items
    is_moe = terminal_type == LayerSymbols.MOE
    num_local_experts = terminal_layer.mlp.num_local_experts if is_moe else None

    def pre_dispatch_computation(node: ScheduleNode, hidden_states: Tensor):
        for item_type, item_layer in pre_layers:
            with _get_inner_quant_context(item_layer):
                if item_type == LayerSymbols.MAMBA:
                    hidden_states = _apply_mamba_layer(item_layer, node, hidden_states)
                elif item_type in (
                    LayerSymbols.ATTENTION,
                    LayerSymbols.DS_ATTENTION,
                    LayerSymbols.GDN,
                ):
                    hidden_states = _apply_attention_layer(item_layer, node, hidden_states)
                else:
                    raise ValueError(
                        f"HybridStack overlap does not support layer type '{item_type}' before "
                        "the terminal MLP/MoE layer."
                    )

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

        if terminal_type == LayerSymbols.MOE:
            with _get_inner_quant_context(terminal_layer):
                return _run_moe_preprocess(terminal_layer, node, hidden_states)

        if terminal_type is None:
            return _maybe_apply_final_norm(node, hidden_states)

        return hidden_states

    def dispatch(node: ScheduleNode, local_tokens: Tensor, probs: Tensor):
        enable_hybridep = (
            terminal_layer.config.moe_token_dispatcher_type == "flex"
            and terminal_layer.config.moe_flex_dispatcher_backend == "hybridep"
        )
        enable_deepep = (
            terminal_layer.config.moe_token_dispatcher_type == "flex"
            and terminal_layer.config.moe_flex_dispatcher_backend == "deepep"
        )
        token_dispatcher = terminal_layer.mlp.token_dispatcher
        if enable_deepep or enable_hybridep:
            token_dispatcher._comm_manager.token_probs = probs
        with _get_inner_quant_context(terminal_layer):
            dispatched_tokens, dispatched_probs = terminal_layer.mlp.dispatch(local_tokens, probs)
        node.layer_state.dispatched_probs = node.detach(dispatched_probs)
        return dispatched_tokens

    def mlp(node: ScheduleNode, hidden_states: Tensor):
        if terminal_type == LayerSymbols.MLP:
            with _get_inner_quant_context(terminal_layer):
                hidden_states = terminal_layer._forward_mlp(
                    hidden_states,
                    padding_mask=node.chunk_state.padding_mask,
                )
            return _maybe_apply_final_norm(node, hidden_states)
        if terminal_type == LayerSymbols.MOE:
            with _get_inner_quant_context(terminal_layer):
                return _run_moe_experts(terminal_layer, node, hidden_states)
        return hidden_states

    def combine(node: ScheduleNode, output: Tensor):
        with _get_inner_quant_context(terminal_layer):
            return _run_moe_combine(terminal_layer, node, output)

    def raise_not_implemented(*args):
        raise NotImplementedError("This callable is not implemented for non-MoE hybrid layers.")

    backward_dw = {}
    pre_bwd_dw = []
    for item_type, item_layer in pre_layers:
        if item_type in (LayerSymbols.ATTENTION, LayerSymbols.DS_ATTENTION, LayerSymbols.GDN):
            item_layer.init_backward_dw_wrapper()
            pre_bwd_dw.append(item_layer.backward_dw_wrapper)
    if is_moe:
        shared_expert_dw = _SharedExpertBackwardDWWrapper(terminal_layer)
        if shared_expert_dw.shared_expert_dw_callable is not None:
            pre_bwd_dw.append(shared_expert_dw)
        backward_dw["mlp"] = terminal_layer.mlp
    elif terminal_type == LayerSymbols.MLP:
        backward_dw["mlp"] = terminal_layer.mlp

    if pre_bwd_dw:
        backward_dw["attn"] = pre_bwd_dw

    forward_funcs = [
        pre_dispatch_computation,
        dispatch if is_moe else raise_not_implemented,
        mlp,
        combine if is_moe else raise_not_implemented,
        None,
    ]
    return forward_funcs, backward_dw, is_moe, num_local_experts
