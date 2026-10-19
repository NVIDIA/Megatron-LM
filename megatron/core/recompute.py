# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from dataclasses import replace
from typing import List, Optional, Set, Tuple, Union

from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.context_parallel import ContextParallelLayoutState
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.layer_interface import LayerAuxInputs
from megatron.core.transformer.module import MegatronModule

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint
else:
    te_checkpoint = None


def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    aux_inputs: LayerAuxInputs,
    *,
    use_inner_quantization_context: bool,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
    cp_layout_state: Optional[ContextParallelLayoutState] = None,
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
        aux_inputs (LayerAuxInputs): Auxiliary layer inputs, including packed-sequence CP metadata
            and token IDs. Checkpointed tensors are restored into this container on replay.

    Returns:
        If extract_layer_indices is empty: hidden_states tensor
        If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    context = aux_inputs.context
    rotary_pos_emb = aux_inputs.rotary_pos_emb
    # Keep checkpointed activations out of the metadata captured by the closure.
    static_aux_inputs = replace(
        aux_inputs,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        padding_mask=None,
        input_ids=None,
    )

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

            aux_inputs = replace(
                static_aux_inputs,
                attention_mask=attention_mask,
                context=context,
                context_mask=context_mask,
                rotary_pos_emb=rotary_pos_emb,
                padding_mask=padding_mask,
                input_ids=input_ids,
            )
            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]
                if cp_layout_state is not None:
                    hidden_states, layer_packed_seq_params = cp_layout_state.prepare_layer(
                        index, hidden_states
                    )
                    aux_inputs = replace(aux_inputs, packed_seq_params=layer_packed_seq_params)
                # Keep both residuals in the layer's layout, inside the CP conversions.
                residual_accumulator = hidden_states

                # Get appropriate inner quantization context
                if use_inner_quantization_context:
                    if self.config.fp8:
                        inner_quantization_context = get_fp8_context(
                            self.config, layer.layer_number - 1
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

                with inner_quantization_context:
                    hidden_states, aux_outputs = layer(hidden_states, aux_inputs)
                    assert aux_outputs.bias is None, "Recompute requires complete residual layers"
                    context = aux_outputs.context
                    aux_inputs = replace(aux_inputs, context=context)

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
            aux_inputs.attention_mask,
            context,
            aux_inputs.context_mask,
            *rotary_pos_emb,
            aux_inputs.padding_mask,
            aux_inputs.input_ids,
        )
        if use_checkpoint:
            # Precision-aware activation checkpoint: TE under FP8/FP4,
            # tensor_parallel under BF16/FP16/FP32.
            if self.config.fp8 or self.config.fp4:
                hidden_states, context = te_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    *args,
                )
            else:
                hidden_states, context = tensor_parallel.checkpoint(
                    cf, self.config.distribute_saved_activations, *args
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
        # layers and skip the rest. Need at least one input tensor with
        # gradient computation for the re-entrant autograd engine, so under
        # FP8/FP4 we skip checkpointing while hidden_states.requires_grad
        # is False (these slots get pushed past the recompute window).
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
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
