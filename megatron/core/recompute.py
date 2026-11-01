# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from contextlib import nullcontext
from typing import List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from megatron.core.context_parallel import ContextParallelLayoutState
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.utils import (
    first_local_layer_index,
    precision_aware_checkpoint,
    resolve_gradient_boundary,
)


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
    cp_layout_state: Optional[ContextParallelLayoutState] = None,
    packed_sequence_cp_metadata: object | None = None,
    input_ids: Optional[Tensor] = None,
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
    resolved = getattr(self, "_gradient_boundary_resolved", None)
    _, gradient_boundary_layer = (
        resolved if resolved is not None else resolve_gradient_boundary(self, hidden_states)
    )
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
                    packed_seq_params=layer_packed_seq_params,
                    padding_mask=padding_mask,
                )
                inner_layer = getattr(layer, "inner_layer", layer)
                router = getattr(getattr(inner_layer, "mlp", None), "router", None)
                if input_ids is not None and getattr(router, "is_hash_layer", False):
                    layer_kwargs["input_ids"] = input_ids
                with inner_quantization_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, context = layer(**layer_kwargs)
                    elif isinstance(getattr(layer, "inner_layer", None), TransformerLayer):
                        # Hybrid mHC wrappers accept the TransformerLayer execution inputs,
                        # including hash-routing token IDs, but not cross-attention-only kwargs.
                        for k in ("context", "context_mask", "attention_bias"):
                            layer_kwargs.pop(k, None)
                        hidden_states, context = layer(**layer_kwargs)
                    else:  # MambaLayer (HybridStack `M` slot)
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

    # Count from this stage's first layer rather than ``layer_offset``: HybridStack does not
    # pass an offset, and its layers carry global layer numbers on every pipeline stage.
    first_local_layer = first_local_layer_index(self.layers)
    boundary_local = (
        None
        if gradient_boundary_layer is None or first_local_layer is None
        else gradient_boundary_layer - first_local_layer
    )
    # A boundary before this stage means trainable layers upstream: keep the input's graph.
    boundary_applied = boundary_local is None or boundary_local < 0
    # The block recompute window counts trainable layers, starting at the boundary.
    first_trainable_local = 0 if boundary_local is None else max(boundary_local, 0)

    def apply_gradient_boundary():
        nonlocal hidden_states, context, boundary_applied
        if boundary_applied:
            return
        hidden_states = hidden_states.detach()
        if torch.is_grad_enabled():
            # Reentrant checkpoints (tensor_parallel and TE) only produce outputs that
            # require grad when an input does. Nothing before the boundary is trainable,
            # so make the boundary activation a grad leaf; otherwise the trainable layers
            # after it would receive no gradient.
            hidden_states.requires_grad_(True)
        if isinstance(context, Tensor):
            context = context.detach()
        boundary_applied = True

    def chunk_runner(start: int, end: int, use_checkpoint: bool, use_grad: bool = True):
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
        if use_grad:
            if use_checkpoint:
                hidden_states, context = precision_aware_checkpoint(
                    cf,
                    self.config,
                    self.pg_collection.tp,
                    *args,
                    distribute_saved_activations=self.config.distribute_saved_activations,
                )
            else:
                hidden_states, context = cf(*args)
        else:
            with torch.no_grad():
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
            if boundary_local is not None and layer_idx < boundary_local:
                prefix_end = min(chunk_end, boundary_local)
                chunk_runner(layer_idx, prefix_end, False, use_grad=False)
                layer_idx = prefix_end
            else:
                apply_gradient_boundary()
                chunk_runner(layer_idx, chunk_end, True)
                layer_idx = chunk_end
    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # layers and skip the rest. Need at least one input tensor with
        # gradient computation for the re-entrant autograd engine, so under
        # FP8/FP4 we skip checkpointing while hidden_states.requires_grad
        # is False (these slots get pushed past the recompute window).
        recompute_skip_num_layers = 0
        for layer_idx in range(self.num_layers_per_pipeline_rank):
            if boundary_local is not None and layer_idx < boundary_local:
                chunk_runner(layer_idx, layer_idx + 1, False, use_grad=False)
                continue
            apply_gradient_boundary()
            if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            window_idx = layer_idx - first_trainable_local
            use_checkpoint = (
                window_idx >= recompute_skip_num_layers
                and window_idx < self.config.recompute_num_layers + recompute_skip_num_layers
            )
            chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
    else:
        raise ValueError("Invalid activation recompute method.")

    if not boundary_applied:
        # Every layer here lies before the boundary; see TransformerBlock.forward.
        hidden_states = hidden_states.detach()

    # Return intermediate hidden states if feature extraction was requested
    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states
