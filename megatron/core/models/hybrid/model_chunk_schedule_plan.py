# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule HybridStack logical layers on the shared combined-1F1B executor."""

from contextlib import nullcontext

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.common.utils import TransformerLayerNode
from megatron.core.models.hybrid.fine_grained_callables import build_hybrid_stack_callables
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.pipeline_parallel.utils import NoopScheduleNode
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer
from megatron.core.transformer.transformer_layer import TransformerLayer


class HybridStackSchedulePlan(TransformerLayerSchedulePlan):
    """Build schedule nodes for one layer symbol or bracketed group, such as ``[*E]``."""

    def __init__(self, layer, event, chunk_state, comp_stream, comm_stream, extra_args=None):
        if extra_args is None:
            extra_args = {}
        self.layer_type = extra_args.get("layer_type", None)
        super().__init__(layer, event, chunk_state, comp_stream, comm_stream, extra_args)

    def _build_callable_nodes(self, event, comp_stream, comm_stream, extra_args):
        if self.layer_type is None:
            return super()._build_callable_nodes(event, comp_stream, comm_stream, extra_args)

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
            return TransformerLayerNode(
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

        # HybridStack groups never carry an MTP terminal, so mtp_post_process is
        # always a no-op here.
        self.mtp_post_process = NoopScheduleNode()

    def get_low_precision_context(self):
        """Hybrid callables enter each physical layer's quantization context themselves."""
        if self.layer_type is None and isinstance(
            self.layer, (TransformerLayer, MultiTokenPredictionLayer)
        ):
            return super().get_low_precision_context()
        return nullcontext()


class HybridStackModelChunkSchedulePlan(TransformerModelChunkSchedulePlan):
    """Pass each HybridStack entry's layer type to its per-layer schedule plan."""

    LAYER_SCHEDULE_PLAN_CLASS = HybridStackSchedulePlan

    def _extra_args_for_layer(self, module, layer_idx, num_layers):
        extra_args = super()._extra_args_for_layer(module, layer_idx, num_layers)
        extra_args["layer_type"] = (
            module.layer_type_list[layer_idx] if isinstance(module, HybridStack) else None
        )
        return extra_args
