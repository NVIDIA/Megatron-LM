# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from typing import List, Optional, Set

from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

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
) -> tuple[Tensor, list[Tensor]]:
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
        Hidden states and extracted intermediate tensors.
    """
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []
    if isinstance(aux_inputs.rotary_pos_emb, (tuple, list)):
        assert len(aux_inputs.rotary_pos_emb) == 2, "Dual RoPE input length is not equal to 2"

    def custom(start: int, end: int, chunk_inputs: LayerAuxInputs):
        # Checkpoint functions accept only tensors. Keep non-tensor metadata in
        # the closure, and reconstruct immutable input snapshots for each replay.
        leaves, input_spec = tree_flatten(vars(chunk_inputs))
        tensor_indices = [i for i, value in enumerate(leaves) if isinstance(value, Tensor)]
        tensor_inputs = tuple(leaves[i] for i in tensor_indices)
        for i in tensor_indices:
            leaves[i] = None

        def custom_forward(hidden_states, *input_tensors):
            restored = list(leaves)
            for i, tensor in zip(tensor_indices, input_tensors):
                restored[i] = tensor
            aux_inputs = LayerAuxInputs(**tree_unflatten(restored, input_spec))
            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]
                if cp_layout_state is not None:
                    hidden_states, layer_packed_seq_params = cp_layout_state.prepare_layer(
                        index, hidden_states
                    )
                    aux_inputs = aux_inputs.update(packed_seq_params=layer_packed_seq_params)
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

                observe_layer_residuals(layer, residual_accumulator, hidden_states)
                if cp_layout_state is not None:
                    hidden_states = cp_layout_state.finalize_layer(index, hidden_states)
            return hidden_states

        return custom_forward, tensor_inputs

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states
        cf, tensor_inputs = custom(start, end, aux_inputs)
        args = (hidden_states, *tensor_inputs)
        if use_checkpoint:
            # Precision-aware activation checkpoint: TE under FP8/FP4,
            # tensor_parallel under BF16/FP16/FP32.
            if self.config.fp8 or self.config.fp4:
                hidden_states = te_checkpoint(
                    cf,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    self.pg_collection.tp,
                    *args,
                )
            else:
                hidden_states = tensor_parallel.checkpoint(
                    cf, self.config.distribute_saved_activations, *args
                )
        else:
            hidden_states = cf(*args)

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

    return hidden_states, intermediate_hidden_states
