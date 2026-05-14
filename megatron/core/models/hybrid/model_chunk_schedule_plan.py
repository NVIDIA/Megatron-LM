# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Schedule-plan classes for HybridStack-based decoders.

These extend the GPT-side ``TransformerLayerSchedulePlan`` /
``TransformerModelChunkSchedulePlan`` with the per-layer ``layer_type`` symbol
that HybridStack assigns to each entry of its ``layer_type_list`` (including
bracketed groups like ``[*-]``). The base classes remain GPT-only; this module
adds the hybrid-specific dispatch into ``build_hybrid_stack_callables`` and
uses ``HybridStackNode`` so the schedule node's free-input policy can diverge
from the GPT default. The pre/post-process nodes from
``core.models.common.utils`` are reused as-is — they already call
``model._preprocess`` / ``model._postprocess`` which work on a HybridModel.
"""

from contextlib import nullcontext

from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)


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
    default to ``layer_type=None`` and follow the GPT path. The pre/post
    process nodes inherit from the GPT base class — they already dispatch
    on ``model._preprocess`` / ``model._postprocess`` which a HybridModel
    implements.
    """

    LAYER_SCHEDULE_PLAN_CLASS = HybridStackSchedulePlan

    def init(self, model, *args, **kwargs):
        assert model.config.cuda_graph_impl == "none", (
            "EP A2A overlap with grouped HybridStack patterns (e.g. '[*E]') does not "
            "support cuda graphs yet. Set cuda_graph_impl='none' or use an ungrouped pattern."
        )
        super().init(model, *args, **kwargs)

    def _extra_args_for_layer(self, module, layer_idx, num_layers):
        extra_args = super()._extra_args_for_layer(module, layer_idx, num_layers)
        extra_args["layer_type"] = (
            module.layer_type_list[layer_idx] if hasattr(module, "layer_type_list") else None
        )
        return extra_args
