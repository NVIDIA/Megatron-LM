# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-callable builders for the combined-1F1B fine-grained schedule plan.

These build_* functions assemble the per-layer ``(forward_funcs, backward_dw)``
tuple that the schedule plan plugs into ``TransformerLayerNode``.

The TransformerLayer-specific builder lives in ``gpt/fine_grained_callables.py``
because it depends on GPT's MoE wiring; the MTP builder and the dispatcher
``build_layer_callables`` are model-agnostic — both GPTModel and HybridModel
schedule MTP layers identically — so they live here.
"""

from contextlib import nullcontext
from functools import partial

import torch

from megatron.core import tensor_parallel
from megatron.core.models.gpt.fine_grained_callables import build_transformer_layer_callables
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    get_mtp_layer_offset,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


def build_mtp_layer_callables(layer):
    """Callables for multi-token prediction layer nodes.

    Wraps the inner ``layer.mtp_model_layer``'s callables with MTP-specific
    pre-process (chunk and concat embeddings) and post-process (gather across
    depths) steps. The inner layer is built by ``build_layer_callables`` so
    that ``mtp_model_layer`` can be a TransformerLayer (today's case) or a
    HybridStack (when an MTP depth uses the hybrid layout).
    """

    forward_funcs, backward_dw, is_moe, num_local_experts = build_layer_callables(
        layer.mtp_model_layer
    )
    (pre_dispatch_forward, dispatch_forward, mlp_forward, combine_forward, _) = forward_funcs
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def submodule_mtp_pre_dispatch_forward(node, hidden_states):
        # MTP Block Preprocess
        if node.is_first_layer:
            offset = get_mtp_layer_offset(layer.config, node.chunk_state.model.vp_stage)
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
            return pre_dispatch_forward(node, hidden_states)

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

    # Build forward and backward callable functions.
    # pre_dispatch_func already has rng context (rolled into
    # submodule_mtp_pre_dispatch_forward), so it does not need to be wrapped.
    pre_dispatch_func = submodule_mtp_pre_dispatch_forward
    dispatch_func = partial(rng_context_wrapper, dispatch_forward)
    mlp_func = partial(rng_context_wrapper, mlp_forward)
    combine_func = partial(rng_context_wrapper, combine_forward)
    mtp_post_process_func = submodule_mtp_postprocess_forward

    forward_funcs = [
        pre_dispatch_func,
        dispatch_func,
        mlp_func,
        combine_func,
        mtp_post_process_func,
    ]
    pre_dispatch_bwd = backward_dw["pre_dispatch_computation"]
    if isinstance(pre_dispatch_bwd, list):
        pre_dispatch_bwd.append(layer.eh_proj)
    else:
        backward_dw["pre_dispatch_computation"] = [pre_dispatch_bwd, layer.eh_proj]

    return forward_funcs, backward_dw, is_moe, num_local_experts


def build_layer_callables(layer):
    """Dispatch to the appropriate layer-callable builder.

    Returns ``(forward_funcs, backward_dw, is_moe, num_local_experts)`` so the
    schedule plan does not need to re-derive ``is_moe`` /
    ``num_local_experts`` after the call — the build function already knows
    the layer type. ``num_local_experts`` is ``None`` for dense layers.
    """
    from megatron.core.models.hybrid.fine_grained_callables import build_hybrid_stack_callables
    from megatron.core.models.hybrid.hybrid_block import HybridStack

    if isinstance(layer, HybridStack):
        return build_hybrid_stack_callables(layer)
    if isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables(layer)
    if isinstance(layer, TransformerLayer):
        forward_funcs, backward_dw = build_transformer_layer_callables(layer)
        is_moe = isinstance(layer.mlp, MoELayer)
        num_local_experts = layer.mlp.num_local_experts if is_moe else None
        return forward_funcs, backward_dw, is_moe, num_local_experts

    raise ValueError(f"Unsupported layer type: {type(layer)}")
