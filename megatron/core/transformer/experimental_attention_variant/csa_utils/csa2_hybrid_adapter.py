# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 state lifecycle for the standard, model-independent HybridStack."""

from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State


class CSA2HybridAdapter:
    """Keep shared KV and sparse indices local to one forward invocation.

    Schedules use stack layer indices, as in TransformerBlock. Non-CSA2 layers
    may appear anywhere in the pattern and do not receive CSA2-specific arguments.
    """

    def __init__(
        self,
        config,
        *,
        layer_type_list,
        pp_layer_offset,
        pre_process,
        post_process,
        is_mtp_layer,
        pg_collection,
    ) -> None:
        if not pre_process or not post_process or pp_layer_offset or is_mtp_layer:
            raise NotImplementedError("CSA2 currently requires a complete backbone on one stage")
        self.attention_layer_ids = frozenset(
            i for i, symbol in enumerate(layer_type_list) if symbol == "V"
        )
        if not self.attention_layer_ids:
            raise ValueError("CSA2HybridAdapter requires at least one CSA2 layer")
        if len(config.csa_compress_ratios) != len(layer_type_list):
            raise ValueError("CSA2 ratios must contain one entry per HybridStack layer")
        if not set(config.csa2_index_source_layers).issubset(self.attention_layer_ids):
            raise ValueError("CSA2 sources must identify CSA2 attention layers in the pattern")

    def prepare_forward(self, hidden_states, packed_seq_params, inference_context, context):
        """Create graph-connected sharing state for this microbatch only."""
        if packed_seq_params is not None or inference_context is not None:
            raise NotImplementedError("CSA2 currently accepts full, unpacked training sequences")
        if context.cross_layer_state is None:
            context.cross_layer_state = CSA2State(attention_layer_ids=self.attention_layer_ids)
        elif not isinstance(context.cross_layer_state, CSA2State):
            raise TypeError("CSA2 layers require CSA2State in the forward context")
        return context

    def before_layer(self, layer, hidden_states, context):
        """CSA2 itself requires no residual preprocessing."""
        return hidden_states

    def finalize_forward(self, output, context: HybridStackForwardContext):
        """Keep the ordinary HybridStack output contract."""
        return output
