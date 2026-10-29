# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from typing import Any, List, Optional, Set

from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_unflatten

from megatron.core.context_parallel import ContextParallelLayoutState, CPLayout
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.layer_interface import AuxInputs, bind_aux_inputs
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import precision_aware_checkpoint
from megatron.core.utils import make_viewless_tensor


@bind_aux_inputs
def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    use_inner_quantization_context: bool,
    padding_mask: Optional[Tensor] = None,
    padding_mask_by_layout: Optional[dict[CPLayout, Optional[Tensor]]] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
    cp_layout_state: Optional[ContextParallelLayoutState] = None,
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | list[Tensor] | None = None,
    *,
    aux_inputs: AuxInputs | None = None,
    **layer_kwargs: Any,
) -> Tensor | tuple[Tensor, list[Tensor]]:
    """Run layers in checkpointed chunks using packed or explicit layer inputs.

    Return hidden states alone unless ``extract_layer_indices`` requests intermediate
    states. Those indices are global; ``layer_offset`` locates this pipeline stage."""
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []
    shared_inputs = aux_inputs is not None
    if shared_inputs and layer_kwargs:
        raise TypeError("Pass either auxiliary inputs or explicit layer arguments")
    input_fields = dict(
        vars(aux_inputs) if shared_inputs else layer_kwargs,
        rotary_pos_emb=rotary_pos_emb,
        padding_mask=padding_mask,
        padding_mask_by_layout=padding_mask_by_layout,
    )
    if isinstance(rotary_pos_emb, (tuple, list)):
        assert len(rotary_pos_emb) == 2, "Dual RoPE input length is not equal to 2"

    def custom(start: int, end: int):
        # Checkpoint functions accept only tensors. Keep non-tensor metadata in
        # the closure, and reconstruct immutable input snapshots for each replay.
        leaves, input_spec = tree_flatten(input_fields)
        tensor_indices = [i for i, value in enumerate(leaves) if isinstance(value, Tensor)]
        tensor_inputs = tuple(leaves[i] for i in tensor_indices)
        for i in tensor_indices:
            leaves[i] = None

        def custom_forward(hidden_states, *input_tensors):
            restored = list(leaves)
            for i, tensor in zip(tensor_indices, input_tensors):
                restored[i] = tensor
            layer_kwargs = tree_unflatten(restored, input_spec)
            padding_mask = layer_kwargs.pop("padding_mask")
            padding_mask_by_layout = layer_kwargs.pop("padding_mask_by_layout")
            # Restore checkpointed metadata once, then select its consumers per layer.
            packed_sequence_cp_metadata = (
                layer_kwargs.pop("packed_sequence_cp_metadata", None) if shared_inputs else None
            )
            for index in range(start, end):
                # Use self.layers[index] (not self._get_layer) so this
                # function works for both TransformerBlock and HybridStack.
                layer = self.layers[index]
                if cp_layout_state is not None:
                    hidden_states, layer_packed_seq_params = cp_layout_state.prepare_layer(
                        index, hidden_states
                    )
                    layer_kwargs["packed_seq_params"] = layer_packed_seq_params
                # Keep both residuals in the layer's layout, inside the CP conversions.
                residual_accumulator = hidden_states
                layer_padding_mask = padding_mask
                if cp_layout_state is not None:
                    layer_padding_mask = cp_layout_state.get_layer_padding_mask(
                        index, padding_mask, padding_mask_by_layout
                    )

                layer_kwargs["padding_mask"] = layer_padding_mask
                layer_inputs = AuxInputs(**layer_kwargs) if shared_inputs else None
                if (
                    shared_inputs
                    and packed_sequence_cp_metadata is not None
                    and type(layer.config) is MambaLayerConfig
                    and layer.config.linear_cp_mode == "chunkwise"
                ):
                    layer_inputs = layer_inputs.update(
                        packed_sequence_cp_metadata=packed_sequence_cp_metadata
                    )

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
                    if layer_inputs is not None:
                        hidden_states, aux_outputs = layer(hidden_states, layer_inputs)
                        assert (
                            aux_outputs.bias is None
                        ), "Recompute requires complete residual layers"
                    else:
                        hidden_states, layer_kwargs["context"] = layer(
                            hidden_states=hidden_states, **layer_kwargs
                        )

                observe_layer_residuals(layer, residual_accumulator, hidden_states)
                if cp_layout_state is not None:
                    hidden_states = cp_layout_state.finalize_layer(index, hidden_states)
            return hidden_states if shared_inputs else (hidden_states, layer_kwargs.get("context"))

        return custom_forward, tensor_inputs

    def chunk_runner(start: int, end: int, use_checkpoint: bool):
        nonlocal hidden_states
        cf, tensor_inputs = custom(start, end)
        if use_checkpoint and self.config.enable_mhc_connections:
            # mHC expands the block input and its zero-dropout BDA path can return views.
            # Distributed activation checkpointing replaces the first input's .data, which
            # requires a viewless tensor at every checkpoint boundary.
            hidden_states = make_viewless_tensor(
                inp=hidden_states, requires_grad=True, keep_graph=True
            )
        args = (hidden_states, *tensor_inputs)
        if use_checkpoint:
            result = precision_aware_checkpoint(
                cf,
                self.config,
                self.pg_collection.tp,
                *args,
                distribute_saved_activations=self.config.distribute_saved_activations,
            )
        else:
            result = cf(*args)
        if shared_inputs:
            hidden_states = result
        else:
            hidden_states, input_fields["context"] = result

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

    if extract_layer_indices:
        return hidden_states, intermediate_hidden_states
    return hidden_states
