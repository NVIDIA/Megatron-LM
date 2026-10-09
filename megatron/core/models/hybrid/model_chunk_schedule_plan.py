# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule-plan classes for HybridStack-based decoders.

These extend GPTModel's schedule plans, ``TransformerLayerSchedulePlan`` and
``TransformerModelChunkSchedulePlan``, with the per-layer ``layer_type`` symbol
that HybridStack assigns to each entry of its ``layer_type_list`` (including
bracketed groups like ``[*-]``). The base classes build callables from the layer
module alone; this module adds the hybrid-specific dispatch into
``build_hybrid_stack_callables``, which also needs the layer's symbol, and uses
``HybridStackNode`` so the schedule node's free-input policy can diverge from the
GPTModel default. The pre/post-process nodes from
``core.models.common.utils`` are reused as-is — they already call
``model._preprocess`` / ``model._postprocess`` which work on a HybridModel.
"""

from contextlib import nullcontext

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.hybrid.fine_grained_callables import (
    HybridStackNode,
    build_hybrid_stack_callables,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.pipeline_parallel.utils import NoopScheduleNode
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionLayer,
)
from megatron.core.transformer.transformer_layer import TransformerLayer


class HybridStackSchedulePlan(TransformerLayerSchedulePlan):
    """Per-layer schedule plan for HybridStack decoders.

    Adds the ``layer_type`` extra-arg propagation; routes through
    ``build_hybrid_stack_callables`` when ``layer_type`` is set (i.e. the layer
    is a HybridStack entry, possibly a bracketed group); falls back to the GPTModel
    path for plain TransformerLayer / MTP layers when ``layer_type`` is None.
    """

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args=None):
        if extra_args is None:
            extra_args = {}
        self.layer_type = extra_args.get("layer_type", None)
        self.mtp_layer = extra_args.get("mtp_layer")
        super().__init__(layer, event, chunk_state, comp_stream, comm_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        if self.layer_type is None:
            return super()._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

        fwd_callables, bwd_dw_callable_map, is_moe, num_local_experts = (
            build_hybrid_stack_callables(self.layer, layer_type=self.layer_type)
        )
        if self.mtp_layer is not None:
            from megatron.core.models.common.fine_grained_callables import wrap_mtp_layer_callables

            fwd_callables, bwd_dw_callable_map = wrap_mtp_layer_callables(
                self.mtp_layer,
                fwd_callables,
                bwd_dw_callable_map,
                pre_process=extra_args["mtp_pre_process"],
                post_process=extra_args["mtp_post_process"],
            )

        extra_args["config"] = self.layer.config
        extra_args["is_moe"] = is_moe
        extra_args["num_local_experts"] = num_local_experts
        extra_args["delay_wgrad_compute"] = self.layer.config.delay_wgrad_compute
        extra_args["is_mtp"] = self.mtp_layer is not None

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
            pre_dispatch_module,
            moe_dispatch_module,
            mlp_module,
            moe_combine_module,
            mtp_post_process_module,
        ) = fwd_callables

        self.pre_dispatch_computation = create_node(
            comp_stream, pre_dispatch_module, "pre_dispatch_computation"
        )
        self.mlp = create_node(comp_stream, mlp_module, "mlp")
        if is_moe:
            self.moe_dispatch = create_node(comm_stream, moe_dispatch_module, "moe_dispatch")
            self.moe_combine = create_node(comm_stream, moe_combine_module, "moe_combine")
        else:
            self.moe_dispatch = NoopScheduleNode()
            self.moe_combine = NoopScheduleNode()

        self.mtp_post_process = (
            create_node(comp_stream, mtp_post_process_module, "mtp_post_process")
            if mtp_post_process_module is not None
            else NoopScheduleNode()
        )

    def get_low_precision_context(self):
        """Return the layer-level quantization context for GPTModel-path layers.

        Hybrid callables enter the quantization context of each physical layer
        themselves, so hybrid layer plans use a null context here.
        """
        if self.mtp_layer is not None:
            return self.mtp_layer.get_inner_quantization_context()
        if self.layer_type is None and isinstance(
            self.layer, (TransformerLayer, MultiTokenPredictionLayer)
        ):
            return super().get_low_precision_context()
        return nullcontext()


class HybridStackModelChunkSchedulePlan(TransformerModelChunkSchedulePlan):
    """Model-chunk schedule plan that builds ``HybridStackSchedulePlan`` layer plans.

    Threads HybridStack's ``layer_type_list[layer_idx]`` symbol into each
    layer plan's ``extra_args`` so the per-layer plan can dispatch grouped
    layers correctly. Each MTP depth's inner stack is expanded into the same
    logical-layer sequence, with its projection and final norm attached at
    the depth boundaries. The pre/post-process nodes inherit from the common
    base class and dispatch on ``model._preprocess`` / ``model._postprocess``.
    """

    LAYER_SCHEDULE_PLAN_CLASS = HybridStackSchedulePlan

    def _build_layer_schedule_plan(self, module, comp_stream, comm_stream):
        if not isinstance(module, MultiTokenPredictionBlock):
            return super()._build_layer_schedule_plan(module, comp_stream, comm_stream)

        for depth_idx, mtp_layer in enumerate(module.layers):
            stack = mtp_layer.mtp_model_layer
            for layer_idx, layer in enumerate(stack.layers):
                first_in_depth = layer_idx == 0
                last_in_depth = layer_idx == len(stack.layers) - 1
                extra_args = {
                    "layer_type": stack.layer_type_list[layer_idx],
                    "mtp_layer": mtp_layer,
                    "mtp_pre_process": first_in_depth,
                    "mtp_post_process": last_in_depth,
                    "is_first_layer": depth_idx == 0 and first_in_depth,
                    "is_last_layer": depth_idx == len(module.layers) - 1 and last_in_depth,
                }
                self._transformer_layers.append(
                    self.LAYER_SCHEDULE_PLAN_CLASS(
                        layer, self.event, self.state, comp_stream, comm_stream, extra_args
                    )
                )

    def _extra_args_for_layer(self, module, layer_idx, num_layers):
        extra_args = super()._extra_args_for_layer(module, layer_idx, num_layers)
        extra_args["layer_type"] = (
            module.layer_type_list[layer_idx] if isinstance(module, HybridStack) else None
        )
        return extra_args
