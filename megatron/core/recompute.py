# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

from torch import Tensor

from megatron.core.context_parallel import ContextParallelLayoutState, CPLayout
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_layer_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import precision_aware_checkpoint, use_te_checkpoint


def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    rotary_pos_emb: Tensor,
    attention_bias: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    padding_mask: Optional[Tensor] = None,
    padding_mask_by_layout: Optional[dict[CPLayout, Optional[Tensor]]] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
    input_ids: Optional[Tensor] = None,
    input_ids_by_layout: Optional[dict[str, Tensor]] = None,
    cp_layout_state: Optional[ContextParallelLayoutState] = None,
    packed_sequence_cp_metadata: object | None = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Forward method with activation checkpointing.

    Args:
        extract_layer_indices (Set[int], optional): Global layer
            indices (across all pipeline stages) from which to
            extract features.
        layer_offset (int): The global layer offset for the current
            pipeline stage. Used to convert local layer indices to
            global indices when checking extract_layer_indices.
        cp_layout_state (ContextParallelLayoutState, optional): CP layout state for this forward.
        packed_sequence_cp_metadata (optional): Packed-sequence CP metadata for Mamba layers.
        input_ids (Tensor, optional): Token IDs forwarded to hash-routed MoE layers.

    Returns:
        If extract_layer_indices is empty: hidden_states tensor
        If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    # Wrap non-dual RoPE to tuple to unify custom_forward interface.
    is_dual_rope = isinstance(rotary_pos_emb, (tuple, list))
    assert not is_dual_rope or len(rotary_pos_emb) == 2, "Dual RoPE input length is not equal to 2"
    rotary_pos_emb = rotary_pos_emb if is_dual_rope else (None, rotary_pos_emb)

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states,
            attention_mask,
            context,
            context_mask,
            rotary_pos_emb_local,
            rotary_pos_emb_global,
            padding_mask=None,
            input_ids=None,
        ):
            rotary_pos_emb = (
                (rotary_pos_emb_local, rotary_pos_emb_global)
                if is_dual_rope
                else rotary_pos_emb_global
            )

            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]
                layer_packed_seq_params = packed_seq_params
                if cp_layout_state is not None:
                    hidden_states, layer_packed_seq_params = cp_layout_state.prepare_layer(
                        index, hidden_states
                    )
                # Keep both residuals in the layer's layout, inside the CP conversions.
                residual_accumulator = hidden_states
                layer_padding_mask = padding_mask
                if cp_layout_state is not None:
                    layer_padding_mask = cp_layout_state.get_layer_padding_mask(
                        index, padding_mask, padding_mask_by_layout
                    )

                # Get appropriate inner quantization context
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_layer_fp8_context(
                            self.config,
                            layer.layer_number - 1,
                            is_mtp_layer=getattr(self, "is_mtp_layer", False),
                        )
                    # TODO: check if fp4 is supported in this case
                    elif self.config.fp4:
                        inner_quantization_context = get_fp4_context(
                            self.config, layer.layer_number - 1
                        )
                    else:
                        inner_quantization_context = nullcontext()
                else:
                    inner_quantization_context = nullcontext()

                # Build the full TransformerLayer kwarg set. Hybrid mHC wrappers expose
                # an explicit capability flag so this module does not need to import
                # hybrid_block (which would create a circular import).
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=layer_packed_seq_params,
                    padding_mask=layer_padding_mask,
                )
                inner_layer = getattr(layer, "inner_layer", layer)
                router = getattr(getattr(inner_layer, "mlp", None), "router", None)
                if input_ids is not None and getattr(router, "is_hash_layer", False):
                    layer_kwargs["input_ids"] = (
                        input_ids_by_layout[cp_layout_state.manager.layer_layouts[index]]
                        if input_ids_by_layout is not None
                        else input_ids
                    )
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    elif getattr(layer, "supports_hybrid_recompute_kwargs", False):
                        # HyperConnectionHybridLayer accepts the routing metadata
                        # consumed by wrapped MoE layers, but not cross-attention kwargs
                        # from the TransformerLayer interface. This also covers a wrapper
                        # around a MambaLayer; the wrapper narrows kwargs for its inner layer.
                        for k in ("context", "context_mask", "attention_bias"):
                            layer_kwargs.pop(k, None)
                        if (
                            packed_sequence_cp_metadata is not None
                            and type(layer.config) is MambaLayerConfig
                            and layer.config.linear_cp_mode == "chunkwise"
                        ):
                            layer_kwargs["packed_sequence_cp_metadata"] = (
                                packed_sequence_cp_metadata
                            )
                        hidden_states, context = layer(**layer_kwargs)
                    else:  # An unwrapped layer with a narrower interface, e.g. MambaLayer.
                        for k in (
                            "context",
                            "context_mask",
                            "attention_bias",
                            "padding_mask",
                            "input_ids",
                        ):
                            layer_kwargs.pop(k, None)
                        if (
                            packed_sequence_cp_metadata is not None
                            and type(layer.config) is MambaLayerConfig
                            and layer.config.linear_cp_mode == "chunkwise"
                        ):
                            layer_kwargs["packed_sequence_cp_metadata"] = (
                                packed_sequence_cp_metadata
                            )
                        hidden_states = layer(**layer_kwargs)
                        context = None

                # Some layer paths may still return a tuple (defensive).
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
                observe_layer_residuals(layer, residual_accumulator, hidden_states)
                if cp_layout_state is not None:
                    hidden_states = cp_layout_state.finalize_layer(index, hidden_states)
            return hidden_states, context

        return custom_forward

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        cf = custom(start, end)
        # Unpack the RoPE tuple as torch cannot save tuples for backward pass.
        args = (
            hidden_states,
            attention_mask,
            context,
            context_mask,
            *rotary_pos_emb,
            padding_mask,
            input_ids,
        )
        if use_checkpoint:
            hidden_states, context = precision_aware_checkpoint(
                cf,
                self.config,
                self.pg_collection.tp,
                *args,
                distribute_saved_activations=self.config.distribute_saved_activations,
            )
        else:
            # Note: original block-branch no-checkpoint path omitted padding_mask
            # (relied on its default=None); restored here for consistency.
            hidden_states, context = cf(*args)

        if self.config.recompute_method == "uniform":
            if (end - 1 + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)
        else:
            if (start + layer_offset) in extract_layer_indices:
                intermediate_hidden_states.append(hidden_states)

    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of layers and checkpoint
        # the input activation of each divided chunk.
        layer_idx = 0
        while layer_idx < self.num_layers_per_pipeline_rank:
            chunk_end = min(
                layer_idx + self.config.recompute_num_layers, self.num_layers_per_pipeline_rank
            )
            chunk_runner(layer_idx, chunk_end, True)
            layer_idx += self.config.recompute_num_layers
    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # layers and skip the rest. TE's re-entrant checkpoint needs at least one
        # input tensor with gradient computation, so wherever it is the backend we
        # skip checkpointing while hidden_states.requires_grad is False (these
        # slots get pushed past the recompute window).
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if use_te_checkpoint(self.config) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            use_checkpoint = (
                layer_idx >= recompute_skip_num_layers
                and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            )
            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
    else:
        raise ValueError("Invalid activation recompute method.")

    # Return intermediate hidden states if feature extraction was requested
    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states
