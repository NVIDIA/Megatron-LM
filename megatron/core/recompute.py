# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer

te_checkpoint = None

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import te_checkpoint


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
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Forward method with activation checkpointing.

    Args:
        extract_layer_indices (Set[int], optional): Global layer
            indices (across all pipeline stages) from which to
            extract features.
        layer_offset (int): The global layer offset for the current
            pipeline stage. Used to convert local layer indices to
            global indices when checking extract_layer_indices.

    Returns:
        If extract_layer_indices is empty: hidden_states tensor
        If extract_layer_indices is non-empty: (hidden_states, intermediate_hidden_states) tuple
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask=None
        ):
            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]

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

                # Build the full TransformerLayer kwarg set; for non-TL
                # layers (currently MambaLayer in HybridStack) pop the kwargs
                # they don't accept and treat the return as a single tensor.
                layer_kwargs = dict(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=attention_bias,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                    padding_mask=padding_mask,
                )
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    else:  # MambaLayer (HybridStack `M` slot)
                        for k in ("context", "context_mask", "attention_bias", "padding_mask"):
                            layer_kwargs.pop(k, None)
                        hidden_states = layer(**layer_kwargs)
                        context = None

                # Some layer paths may still return a tuple (defensive).
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
            return hidden_states, context

        return custom_forward

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states, context
        cf = custom(start, end)
        args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask)
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
