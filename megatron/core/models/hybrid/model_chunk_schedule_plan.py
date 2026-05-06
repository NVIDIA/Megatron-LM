# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule-plan classes for HybridStack-based decoders.

These extend the GPT-side ``TransformerLayerSchedulePlan`` /
``TransformerModelChunkSchedulePlan`` with the per-layer ``layer_type`` symbol
that HybridStack assigns to each entry of its ``layer_type_list`` (including
bracketed groups like ``[*-]``). The base classes remain GPT-only; this module
adds the hybrid-specific dispatch into ``build_hybrid_stack_callables`` and uses
``HybridStackNode`` so the schedule node's free-input policy can diverge from
the GPT default.
"""

from contextlib import nullcontext

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.gpt.fine_grained_callables import (
    PostProcessNode,
    PreProcessNode,
    weak_method,
)
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.transformer.transformer_layer import make_viewless_tensor


class HybridPreProcessNode(PreProcessNode):
    """``PreProcessNode`` that calls ``HybridModel._preprocess``.

    Mirrors the GPT counterpart but takes a HybridModel rather than a GPTModel
    so the EP-overlap schedule plan does not cross-import a GPT-named class
    when scheduling a hybrid model. Behavior matches: ``_preprocess`` returns
    the same 6-tuple shape ``(decoder_input, rotary_pos_emb, rotary_pos_cos,
    rotary_pos_sin, sequence_len_offset, padding_mask)`` and the chunk_state
    fields populated here line up with the slots downstream layer nodes read.
    """

    def __init__(self, hybrid_model, chunk_state, event, stream):
        # Bypass ``PreProcessNode.__init__`` to avoid binding to a
        # ``gpt_model``-named attribute; reuse the underlying ScheduleNode.
        super(PreProcessNode, self).__init__(
            weak_method(self.forward_impl), stream, event, name="pre_process"
        )
        self.hybrid_model = hybrid_model
        self.chunk_state = chunk_state

    def forward_impl(self):
        if not self.hybrid_model.pre_process:
            self.chunk_state.decoder_input = self.hybrid_model.decoder.input_tensor
        (
            decoder_input,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
            padding_mask,
        ) = self.hybrid_model._preprocess(
            input_ids=self.chunk_state.input_ids,
            position_ids=self.chunk_state.position_ids,
            decoder_input=self.chunk_state.decoder_input,
            packed_seq_params=self.chunk_state.packed_seq_params,
            padding_mask=self.chunk_state.padding_mask,
        )

        self.chunk_state.decoder_input = decoder_input
        self.chunk_state.rotary_pos_emb = rotary_pos_emb
        self.chunk_state.rotary_pos_cos = rotary_pos_cos
        self.chunk_state.rotary_pos_sin = rotary_pos_sin
        self.chunk_state.sequence_len_offset = sequence_len_offset
        self.chunk_state.padding_mask = padding_mask
        return decoder_input


class HybridPostProcessNode(PostProcessNode):
    """``PostProcessNode`` that calls ``HybridModel._postprocess``.

    Mirrors the GPT counterpart. Skips MTP inside ``_postprocess`` (sets
    ``mtp_in_postprocess=False``) because the EP-overlap schedule plan handles
    MTP as separate layer nodes in the same chunk plan.
    """

    def __init__(self, hybrid_model, chunk_state, event, stream):
        super(PostProcessNode, self).__init__(
            weak_method(self.forward_impl), stream, event, name="post_process"
        )
        self.hybrid_model = hybrid_model
        self.chunk_state = chunk_state

    def forward_impl(self, hidden_states):
        empty_decoder = len(self.hybrid_model.decoder.layers) == 0
        layer_norm = getattr(self.hybrid_model.decoder, "final_layernorm", None) or getattr(
            self.hybrid_model.decoder, "final_norm", None
        )
        if not self.hybrid_model.config.mtp_num_layers and empty_decoder and layer_norm:
            hidden_states = layer_norm(hidden_states)
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )

        loss = self.hybrid_model._postprocess(
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
        return float16_to_fp32(loss)


class HybridStackSchedulePlan(TransformerLayerSchedulePlan):
    """Per-layer schedule plan for HybridStack decoders.

    Adds the ``layer_type`` extra-arg propagation; routes through
    ``build_hybrid_stack_callables`` when ``layer_type`` is set (i.e. the layer
    is a HybridStack entry, possibly a bracketed group); falls back to the GPT
    path for plain TransformerLayer / MTP layers when ``layer_type`` is None.
    """

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args=None):
        if extra_args is None:
            extra_args = {}
        self.layer_type = extra_args.get("layer_type", None)
        super().__init__(layer, event, chunk_state, comp_stream, comm_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        if self.layer_type is None:
            return super()._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

        # Hybrid grouped path. Imports are local because hybrid pulls in TE / SSM
        # extensions that we don't want to load when only the GPT path is used.
        from megatron.core.models.hybrid.fine_grained_callables import (
            HybridStackNode,
            build_hybrid_stack_callables,
        )
        from megatron.core.pipeline_parallel.utils import NoopScheduleNode

        fwd_callables, bwd_dw_callable_map, is_moe, num_local_experts = (
            build_hybrid_stack_callables(self.layer, layer_type=self.layer_type)
        )

        extra_args["config"] = self.layer.config
        extra_args["is_moe"] = is_moe
        extra_args["num_local_experts"] = num_local_experts
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = False

        def create_node(stream, module, name):
            bwd_dw_callables = bwd_dw_callable_map.get(name, None)
            node_extra_args = dict(extra_args)
            if bwd_dw_callables is None:
                node_extra_args["delay_wgrad_compute"] = False
            return HybridStackNode(
                stream,
                event,
                self.layer_state,
                self.chunk_state,
                module,
                name=name,
                bwd_dw_callables=bwd_dw_callables,
                extra_args=node_extra_args,
            )

        (
            attn_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables

        self.attn = create_node(comp_stream, attn_module, "attn")
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        # HybridStack groups never carry an MTP terminal, so mtp_post_process is
        # always a no-op here.
        self.mtp_post_process = NoopScheduleNode()

    def get_fp8_context(self):
        # Grouped hybrid layers (and inferred-layer-type entries that point at
        # a HybridStack rather than a plain TransformerLayer) don't have a
        # ``layer_number`` we can hand to ``get_fp8_context``; the inner layers
        # manage their own per-layer fp8 context inside the hybrid callables.
        if self.layer_type is not None or not hasattr(self.layer, "layer_number"):
            return nullcontext()
        return super().get_fp8_context()


class HybridStackModelChunkSchedulePlan(TransformerModelChunkSchedulePlan):
    """Model-chunk schedule plan that builds ``HybridStackSchedulePlan`` layer plans.

    Threads HybridStack's ``layer_type_list[layer_idx]`` symbol into each
    layer plan's ``extra_args`` so the per-layer plan can dispatch grouped
    layers correctly. Ordinary GPT/MTP layers (no ``layer_type_list``)
    default to ``layer_type=None`` and follow the GPT path.
    """

    LAYER_SCHEDULE_PLAN_CLASS = HybridStackSchedulePlan
    PRE_PROCESS_NODE_CLASS = HybridPreProcessNode
    POST_PROCESS_NODE_CLASS = HybridPostProcessNode

    def _extra_args_for_layer(self, module, layer_idx, num_layers):
        extra_args = super()._extra_args_for_layer(module, layer_idx, num_layers)
        extra_args["layer_type"] = (
            module.layer_type_list[layer_idx] if hasattr(module, "layer_type_list") else None
        )
        return extra_args
