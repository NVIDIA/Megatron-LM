# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.context_parallel import ContextParallelLayoutState
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp4_utils import get_fp4_context
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.tensor_observation import observe_layer_residuals
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import (
        te_checkpoint,
        te_mark_not_offload,
        te_start_offload,
    )
else:
    te_checkpoint = None
    te_mark_not_offload = None
    te_start_offload = None


class CheckpointOffloadScheduler:
    """Drive Transformer Engine's manual CPU-offload controller from the block-recompute loop.

    Used when ``cpu_offloading`` is combined with ``recompute_granularity='full'`` and
    ``recompute_method='block'``. Layer ``i`` is offloaded when ``i < num_offload_layers``.
    For a checkpointed layer the only tensor autograd saves is the chunk input, so that is
    what moves to the CPU; a non-checkpointed layer inside the offload window moves every
    tensor it saves, exactly as on the plain (no-recompute) offload path.

    Forward: once layer ``i`` has been enqueued, its saved tensors start copying to pinned
    CPU memory on TE's offload stream, and the GPU copies of layer ``i - 1`` are released.
    The release makes the compute stream wait for that layer's copy, which by then has had
    a full layer of forward compute to finish.

    Backward: a ``grad_fn`` prehook on the output of layer ``min(i + prefetch, last)``
    starts the asynchronous reload of layer ``i``. The prehook fires right before that
    layer's backward (its recompute, when checkpointed), so the host-to-device copy of
    layer ``i`` overlaps the backward of the ``prefetch`` layers above it. TE's unpack hook
    waits on the per-tensor reload event before the checkpoint backward reads the tensor.
    """

    def __init__(
        self, controller: Any, num_layers: int, num_offload_layers: int, prefetch_num_layers: int
    ):
        assert prefetch_num_layers >= 1, "The reload prefetch distance must be at least one layer."
        self.controller = controller
        self.num_layers = num_layers
        self.num_offload_layers = max(0, min(num_offload_layers, num_layers))
        self.prefetch_num_layers = prefetch_num_layers
        # Offloaded layers whose GPU copies have not been released yet, in layer order.
        self._pending_release: List[int] = []
        # Layer whose backward triggers a reload -> offloaded layers reloaded at that point.
        self._reload_triggers: Dict[int, List[int]] = defaultdict(list)
        for layer_idx in range(self.num_offload_layers):
            trigger = min(layer_idx + prefetch_num_layers, num_layers - 1)
            self._reload_triggers[trigger].append(layer_idx)

    def is_offloaded(self, layer_idx: int) -> bool:
        """Whether the saved tensors of ``layer_idx`` are moved to the CPU."""
        return layer_idx < self.num_offload_layers

    @staticmethod
    def mark_ready(*tensors: Optional[Tensor]) -> None:
        """Record on the current stream that ``tensors`` are complete.

        TE keys the start of each device-to-host copy on this event. The checkpoint input
        is final before its layer runs, so recording it here lets the copy start underneath
        the layer's forward instead of after it.
        """
        ready = tuple(t for t in tensors if isinstance(t, Tensor))
        if te_start_offload is not None and ready:
            te_start_offload(*ready)

    def after_layer_forward(self, layer_idx: int, output: Tensor) -> None:
        """Start the offload of ``layer_idx`` and arm the reload it is responsible for."""
        self._release_below(layer_idx)
        if self.is_offloaded(layer_idx):
            self.controller.start_offload_layer(layer_idx)
            self._pending_release.append(layer_idx)

        to_reload = self._reload_triggers.get(layer_idx)
        if to_reload and output.grad_fn is not None:
            controller = self.controller
            layers = tuple(to_reload)

            def _start_reloads(_grad_outputs):
                for reload_idx in layers:
                    controller.start_reload_layer(reload_idx)

            output.grad_fn.register_prehook(_start_reloads)

    def finish_forward(self) -> None:
        """Release whatever is still pending once every layer has run."""
        self._release_below(self.num_layers)

    def _release_below(self, layer_idx: int) -> None:
        while self._pending_release and self._pending_release[0] < layer_idx:
            self.controller.release_activation_forward_gpu_memory(self._pending_release.pop(0))


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
    intermediate_hidden_states: List[Tensor] = []

    # Wrap non-dual RoPE to tuple to unify custom_forward interface.
    is_dual_rope = isinstance(rotary_pos_emb, (tuple, list))
    assert not is_dual_rope or len(rotary_pos_emb) == 2, "Dual RoPE input length is not equal to 2"
    rotary_pos_emb = rotary_pos_emb if is_dual_rope else (None, rotary_pos_emb)

    # CPU offloading of saved activations along the recompute path. The block's TE offload
    # context installs saved-tensor hooks; for a checkpointed layer they see the
    # ``save_for_backward`` inside CheckpointFunction, i.e. the chunk input.
    offload_commit = getattr(self, "group_prefetch_offload_commit_async", None)
    offload_enabled = (
        torch.is_grad_enabled() and self.config.cpu_offloading and offload_commit is not None
    )
    offload_context = self.offload_context if offload_enabled else nullcontext()
    offload_scheduler: Optional[CheckpointOffloadScheduler] = None
    if offload_enabled:
        assert self.config.recompute_method == "block", (
            "CPU offloading on the recompute path needs recompute_method='block': the TE "
            "offload context identifies layers by entry count, one layer per entry."
        )
        controller = getattr(self, "offload_manual_controller", None)
        if controller is not None:
            offload_scheduler = CheckpointOffloadScheduler(
                controller,
                self.num_layers_per_pipeline_rank,
                self.config.cpu_offloading_num_layers,
                self.config.cpu_offloading_prefetch_num_layers,
            )
        # Masks, rotary embeddings and the cross-attention context are shared by every
        # layer, but the checkpoint saves them per layer. Keep them resident rather than
        # copying them to the CPU once per layer.
        if te_mark_not_offload is not None:
            shared = [
                t
                for t in (attention_mask, context, context_mask, *rotary_pos_emb, padding_mask)
                if isinstance(t, Tensor)
            ]
            if shared:
                te_mark_not_offload(*shared)

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
        if (
            offload_scheduler is not None
            and use_checkpoint
            and offload_scheduler.is_offloaded(start)
        ):
            offload_scheduler.mark_ready(hidden_states)
        with offload_context:
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
                    hidden_states, context = cf(*args)
            else:
                # Note: original block-branch no-checkpoint path omitted padding_mask
                # (relied on its default=None); restored here for consistency.
                hidden_states, context = cf(*args)
        if offload_enabled:
            # Registers TE's backward hook on this layer's output; on a checkpointed layer
            # that is the checkpoint node, so it fires right before the recompute.
            hidden_states = offload_commit(hidden_states)
            if offload_scheduler is not None:
                offload_scheduler.after_layer_forward(start, hidden_states)

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
        if offload_scheduler is not None:
            offload_scheduler.finish_forward()
    else:
        raise ValueError("Invalid activation recompute method.")

    # Return intermediate hidden states if feature extraction was requested
    if len(extract_layer_indices) > 0:
        return hidden_states, intermediate_hidden_states

    return hidden_states
