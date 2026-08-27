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
from megatron.core.packed_seq_params import resolve_cp_group
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.mtp_sequence_roll import (
    MTPSequenceRollField,
    prepare_mtp_sequence_roll_context,
)
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    _scatter_mtp_padding_mask,
    get_mtp_layer_offset,
)
from megatron.core.transformer.transformer_layer import TransformerLayer, make_viewless_tensor


def build_mtp_layer_callables(layer):
    """Callables for multi-token prediction layer nodes.

    Wraps the inner ``layer.mtp_model_layer``'s callables with MTP-specific
    pre-process (chunk and concat embeddings) and post-process (gather across
    depths) steps.
    """

    forward_funcs, backward_dw = build_layer_callables(layer.mtp_model_layer)
    is_moe, _ = get_layer_moe_metadata(layer.mtp_model_layer)
    pre_dispatch_forward, dispatch_forward, mlp_forward, combine_forward, _ = forward_funcs
    assert is_moe, "MTP layer in a2a overlap only supports MoE layer for now."

    def prepare_roll_rows(node):
        """Prepare one absolute-row group shared by scheduled MTP depths and loss."""
        chunk_state = node.chunk_state
        materialized = getattr(chunk_state, "mtp_materialized_roll_rows", None)
        if materialized is not None:
            return materialized

        input_ids = chunk_state.input_ids
        labels = chunk_state.labels
        packed_seq_params = chunk_state.packed_seq_params
        cp_group = resolve_cp_group(layer.cp_group, packed_seq_params)
        sequence_roll_context = prepare_mtp_sequence_roll_context(
            input_ids if input_ids is not None else labels, cp_group, packed_seq_params
        )
        fields = []
        if input_ids is not None:
            fields.append(MTPSequenceRollField("input_ids", input_ids, -1, 0, 0))
        if chunk_state.position_ids is not None and getattr(
            chunk_state.model.embedding, "add_position_embedding", True
        ):
            fields.append(MTPSequenceRollField("position_ids", chunk_state.position_ids, -1, 0, 0))
        if chunk_state.model.post_process and labels is not None:
            fields.append(MTPSequenceRollField("labels", labels, -1, 0, 0))
        if chunk_state.model.post_process and chunk_state.loss_mask is not None:
            fields.append(MTPSequenceRollField("loss_mask", chunk_state.loss_mask, -1, 0, 0))
        raw_padding_mask = getattr(chunk_state, "mtp_padding_mask", chunk_state.padding_mask)
        if raw_padding_mask is not None:
            fields.append(MTPSequenceRollField("padding_mask", raw_padding_mask, -1, 0, True))

        max_offset = layer.config.mtp_num_layers + int(
            chunk_state.model.post_process and labels is None
        )
        if sequence_roll_context is not None and fields and max_offset > 0:
            if not (
                sequence_roll_context.max_offset >= max_offset
                and sequence_roll_context.is_prepared_for_fields(fields)
            ):
                sequence_roll_context = sequence_roll_context.prepare_fields(
                    fields, max_offset=max_offset
                )
            if (
                sequence_roll_context.max_offset >= max_offset
                and sequence_roll_context.is_prepared_for_fields(fields)
            ):
                materialized = {
                    key: sequence_roll_context.materialize_all(key)
                    for key in ("input_ids", "position_ids", "padding_mask")
                    if key in sequence_roll_context.keys
                }
            else:
                materialized = {}
        else:
            materialized = {}
        chunk_state.mtp_sequence_roll_context = sequence_roll_context
        chunk_state.mtp_materialized_roll_rows = materialized
        return materialized

    def submodule_mtp_pre_dispatch_forward(node, hidden_states):
        # MTP Block Preprocess
        if node.is_first_layer:
            # Apply the main decoder's final_norm if this VPP chunk owns it but
            # holds no main HybridStack layers — without this, ``_maybe_apply_final_norm``
            # never fires for the main path and the unnormalized hidden_states feed
            # straight into the LM head (lm_loss explodes by ~10x; grads diverge).
            # Restricted to HybridModel because GPT models go through a different
            # MTP wiring path. Must run before ``torch.chunk`` so every chunk —
            # including the main-decoder slice consumed by the LM head — sees
            # the norm; the MTP slices then go through MTP's own ``hnorm`` as usual.
            from megatron.core.models.hybrid.hybrid_model import HybridModel

            model = node.chunk_state.model
            if isinstance(model, HybridModel) and len(model.decoder.layers) == 0:
                final_norm = getattr(model.decoder, "final_norm", None) or getattr(
                    model.decoder, "final_layernorm", None
                )
                if final_norm is not None:
                    hidden_states = final_norm(hidden_states)
                    hidden_states = make_viewless_tensor(
                        inp=hidden_states, requires_grad=True, keep_graph=True
                    )

            offset = get_mtp_layer_offset(layer.config, node.chunk_state.model.vp_stage)
            node.chunk_state.mtp_hidden_states = list(torch.chunk(hidden_states, 1 + offset, dim=0))
            hidden_states = node.chunk_state.mtp_hidden_states[offset]

        materialized_rows = prepare_roll_rows(node)
        absolute_depth = getattr(node, "mtp_absolute_depth", None) or layer.layer_number
        use_prepared_rows = "input_ids" in materialized_rows
        input_ids = (
            materialized_rows["input_ids"][absolute_depth - 1]
            if use_prepared_rows
            else node.chunk_state.input_ids
        )
        position_ids = (
            materialized_rows["position_ids"][absolute_depth - 1]
            if use_prepared_rows and "position_ids" in materialized_rows
            else node.chunk_state.position_ids
        )
        padding_mask = (
            materialized_rows["padding_mask"][absolute_depth - 1]
            if use_prepared_rows and "padding_mask" in materialized_rows
            else (None if use_prepared_rows else node.chunk_state.padding_mask)
        )

        input_ids, position_ids, padding_mask, decoder_input, hidden_states = layer._get_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            embedding=node.chunk_state.model.embedding,
            hidden_states=hidden_states,
            packed_seq_params=node.chunk_state.packed_seq_params,
            padding_mask=padding_mask,
            sequence_roll_context=getattr(node.chunk_state, "mtp_sequence_roll_context", None),
            roll_depth=absolute_depth - 1,
            _inputs_pre_aligned=use_prepared_rows,
        )
        if use_prepared_rows:
            padding_mask = _scatter_mtp_padding_mask(
                padding_mask,
                sequence_parallel=layer.config.sequence_parallel,
                tp_group=layer.tp_group,
            )
        else:
            node.chunk_state.input_ids = input_ids
            node.chunk_state.position_ids = position_ids
            node.chunk_state.padding_mask = padding_mask

        # MTP Layer Preprocess
        # norm, linear projection and transformer
        assert (
            node.chunk_state.context is None
        ), f"multi token prediction + cross attention is not yet supported."
        if layer.config.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # fp8 context is added in 1f1b schedule, so we don't need to add it here
        with rng_context:
            hidden_states = layer._concat_embeddings(hidden_states, decoder_input)
            return pre_dispatch_forward(node, hidden_states, padding_mask=padding_mask)

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

    return forward_funcs, backward_dw


def get_layer_moe_metadata(layer):
    """Return ``(is_moe, num_local_experts)`` for schedule-node construction."""

    if isinstance(layer, MultiTokenPredictionLayer):
        return get_layer_moe_metadata(layer.mtp_model_layer)
    if isinstance(layer, TransformerLayer):
        is_moe = isinstance(layer.mlp, MoELayer)
        num_local_experts = layer.mlp.num_local_experts if is_moe else None
        return is_moe, num_local_experts

    raise ValueError(f"Unsupported layer type: {type(layer)}")


def build_layer_callables(layer):
    """Dispatch to the appropriate layer-callable builder.

    Returns ``(forward_funcs, backward_dw)``.
    """

    if isinstance(layer, MultiTokenPredictionLayer):
        return build_mtp_layer_callables(layer)
    if isinstance(layer, TransformerLayer):
        return build_transformer_layer_callables(layer)

    raise ValueError(f"Unsupported layer type: {type(layer)}")
