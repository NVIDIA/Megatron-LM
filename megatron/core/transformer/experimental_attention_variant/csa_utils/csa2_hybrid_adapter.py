# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CSA2 state lifecycle for the standard, model-independent HybridStack."""

from dataclasses import replace

from torch.utils.checkpoint import checkpoint

from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStackForwardContext
from megatron.core.transformer.experimental_attention_variant.csa2 import CSA2State
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_candidates import (
    CSA2CandidateBlocks,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCState
from megatron.core.transformer.stateful_module import _checkpoint_contexts


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
        if inference_context is not None:
            raise NotImplementedError("CSA2 does not support inference contexts")
        if packed_seq_params is not None and packed_seq_params.qkv_format != "thd":
            raise NotImplementedError("CSA2 packed training requires THD metadata")
        if context.cross_layer_state is None:
            context.cross_layer_state = CSA2State(attention_layer_ids=self.attention_layer_ids)
        elif not isinstance(context.cross_layer_state, CSA2State):
            raise TypeError("CSA2 layers require CSA2State in the forward context")
        return context

    def before_layer(self, layer, hidden_states, context):
        """CSA2 itself requires no residual preprocessing."""
        return hidden_states

    def checkpoint_layer(self, layer, layer_kwargs, context):
        """Recompute one layer from its original cross-layer and mHC inputs."""
        state = context.cross_layer_state
        if not isinstance(state, CSA2State):
            raise TypeError("CSA2 full recompute requires a forward-local CSA2State")
        if (layer.config.dsa_indexer_loss_coeff or 0) > 0:
            raise NotImplementedError("CSA2 full recompute does not yet support indexer loss")
        mhc_state = context.mhc_state
        has_mhc_state = mhc_state is not None
        static_kwargs = {
            key: value
            for key, value in layer_kwargs.items()
            if key not in ("hidden_states", "cross_layer_state", "mhc_state")
        }
        candidate_block_size = (
            state.candidates.block_size
            if state.candidates is not None
            else layer.config.csa2_candidate_block_size
        )
        # A replay must start at this layer's original metadata, not the state
        # subsequently mutated by later layers of the same forward.
        initial = replace(
            state, global_kv=None, indexer_k=None, global_indices=None, candidates=None
        )
        metadata = [None]

        def run(hidden, global_kv, indexer_k, global_indices, candidate_indices, pre_mix):
            local_state = replace(
                initial,
                global_kv=global_kv,
                indexer_k=indexer_k,
                global_indices=global_indices,
                candidates=(
                    None
                    if candidate_indices is None
                    else CSA2CandidateBlocks(candidate_indices, candidate_block_size)
                ),
            )
            local_mhc = SinglePassMHCState(pre_mix) if has_mhc_state else None
            kwargs = dict(static_kwargs, hidden_states=hidden, cross_layer_state=local_state)
            if local_mhc is not None:
                kwargs["mhc_state"] = local_mhc
            output, _ = layer(**kwargs)
            if metadata[0] is None:
                metadata[0] = (
                    local_state.kv_source_layer,
                    local_state.index_source_layer,
                    local_state.candidate_source_layer,
                    local_state.last_layer,
                    local_state.sequence_length,
                    local_state.batch_size,
                    local_state.device,
                    local_state.dtype,
                    local_state.compressed_layout,
                )
            return (
                output,
                local_state.global_kv,
                local_state.indexer_k,
                local_state.global_indices,
                None if local_state.candidates is None else local_state.candidates.indices,
                None if local_mhc is None else local_mhc.pre_mix,
            )

        result = checkpoint(
            run,
            layer_kwargs["hidden_states"],
            state.global_kv,
            state.indexer_k,
            state.global_indices,
            None if state.candidates is None else state.candidates.indices,
            None if mhc_state is None else mhc_state.pre_mix,
            use_reentrant=False,
            context_fn=_checkpoint_contexts,
        )
        state.global_kv, state.indexer_k, state.global_indices, candidate_indices, pre_mix = result[
            1:
        ]
        state.candidates = (
            None
            if candidate_indices is None
            else CSA2CandidateBlocks(candidate_indices, candidate_block_size)
        )
        (
            state.kv_source_layer,
            state.index_source_layer,
            state.candidate_source_layer,
            state.last_layer,
            state.sequence_length,
            state.batch_size,
            state.device,
            state.dtype,
            state.compressed_layout,
        ) = metadata[0]
        if mhc_state is not None:
            mhc_state.pre_mix = pre_mix
        return result[0], None

    def finalize_forward(self, output, context: HybridStackForwardContext):
        """Keep the ordinary HybridStack output contract."""
        return output
