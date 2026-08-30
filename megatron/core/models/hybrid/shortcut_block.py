# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from typing import Sequence

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core.transformer.module import MegatronModule, TwoStageAttentionLayer
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module


def _get_offloading_interface():
    """Get the fine-grained activation offloading interface lazily."""
    from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
        FineGrainedActivationOffloadingInterface,
    )

    return FineGrainedActivationOffloadingInterface


def group_layers_into_shortcut_blocks(
    layers: torch.nn.ModuleList, layer_type_list: Sequence[str], config: TransformerConfig
) -> torch.nn.ModuleList:
    """Group physical layers into their registered shortcut-block hierarchy.

    Grouping updates the layer names through the returned ``ModuleList`` hierarchy. Layers not
    followed by an MoE remain direct children of the returned ``ModuleList``.

    Args:
        layers: Physical layers in execution order.
        layer_type_list: Physical layer symbols in execution order.
        config: Transformer configuration controlling shortcut scheduling.

    Returns:
        The registered logical layers.

    Raises:
        ValueError: If an MoE has an unsupported predecessor.
    """
    grouped_layers = torch.nn.ModuleList()
    physical_index = 0
    while physical_index < len(layers):
        next_is_moe = (
            physical_index + 1 < len(layers)
            and layer_type_list[physical_index + 1] == LayerSymbols.MOE
        )
        if not next_is_moe:
            grouped_layers.append(layers[physical_index])
            physical_index += 1
            continue

        compute_layer = layers[physical_index]
        paired_type = layer_type_list[physical_index]
        supported_predecessors = {
            LayerSymbols.MAMBA,
            LayerSymbols.GDN,
            *LayerSymbols.ATTENTION_LAYERS,
        }
        if paired_type not in supported_predecessors:
            raise ValueError(
                "Shortcut MoE must be preceded by a Mamba, GDN, or supported attention layer"
            )

        supports_two_stage = (
            isinstance(compute_layer, TwoStageAttentionLayer)
            and compute_layer.supports_two_stage_attention()
        )
        if not supports_two_stage:
            raise ValueError(
                f"Shortcut compute layer {paired_type!r} does not support two-stage attention"
            )
        moe_layer = layers[physical_index + 1]
        grouped_layers.append(
            ShortcutMoEBlock(compute_layer, moe_layer, overlap_a2a=config.moe_shortcut_parallel)
        )
        physical_index += 2

    return grouped_layers


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one compute-layer/shortcut-MoE pair."""

    _parallel_stream: torch.cuda.Stream | None = None

    @classmethod
    def _get_a2a_overlap_stream(cls) -> torch.cuda.Stream:
        """Return the process-wide high-priority shortcut stream."""
        if cls._parallel_stream is None:
            cls._parallel_stream = torch.cuda.Stream(priority=-1)
        return cls._parallel_stream

    def __init__(self, compute_layer, moe_layer, overlap_a2a: bool):
        super().__init__(compute_layer.config)

        self.overlap_mode = overlap_a2a
        self.layer_number = compute_layer.layer_number
        self.attn_layer_num = compute_layer.layer_number - 1
        self.moe_layer_num = moe_layer.layer_number - 1
        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)
        self.tp_group = moe_layer.mlp.tp_group
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer
        self.recompute_shortcut_pre_mlp_layernorm = (
            self.config.recompute_granularity == "selective"
            and "shortcut_pre_mlp_layernorm" in (self.config.recompute_modules or [])
        )
        self.offload_shortcut_post_norm = (
            self.config.fine_grained_activation_offloading
            and "shortcut_post_norm" in (self.config.offload_modules or [])
        )
        self.off_interface = _get_offloading_interface()
        self.shortcut_pre_mlp_layernorm_checkpoint = None

        # The shortcut path uses the same normalization implementation and configuration as
        # the MoE path, but owns an independent parameter.
        self.shortcut_pre_mlp_layernorm = build_module(
            moe_layer.submodules_config.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.shortcut_post_norm = build_module(
            moe_layer.submodules_config.pre_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.route_ready_event = torch.cuda.Event() if self.overlap_mode else None

    def _moe_router_preprocess(self, shortcut_hidden, padding_mask=None):
        """Run shortcut normalization, routing, and dispatch preprocessing."""
        if self.recompute_shortcut_pre_mlp_layernorm:
            self.shortcut_pre_mlp_layernorm_checkpoint = CheckpointWithoutOutput()
            shortcut_input = self.shortcut_pre_mlp_layernorm_checkpoint.checkpoint(
                apply_module(self.shortcut_pre_mlp_layernorm), shortcut_hidden
            )
        else:
            shortcut_input = apply_module(self.shortcut_pre_mlp_layernorm)(shortcut_hidden)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = self.moe_layer.mlp.route(shortcut_input, padding_mask)
        return self.moe_layer.mlp.preprocess(shortcut_input, probs, routing_map)

    def _moe_shared_experts(self, hidden_states):
        """Run the paired MoE layer's pre-MLP norm and shared experts."""
        pre_mlp_output = self.moe_layer._forward_pre_mlp_layernorm(hidden_states)
        return self.moe_layer.mlp.shared_experts_compute(pre_mlp_output)

    def _postprocess(self, hidden_states, combined_output, shared_expert_output):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA."""
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        post_norm_input = output
        post_norm_manager = self.off_interface(
            self.offload_shortcut_post_norm, post_norm_input, "shortcut_post_norm"
        )
        with post_norm_manager as post_norm_input:
            output = self.shortcut_post_norm(post_norm_input)
        output = post_norm_manager.group_offload(output, forced_released_tensors=[post_norm_input])
        output = self.moe_layer._apply_mlp_bda_step((output, None), residual)
        return output[0] if isinstance(output, tuple) else output

    def _launch_dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, async_op: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Launch dispatch on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

        assert self.route_ready_event is not None
        dispatch_stream = self._get_a2a_overlap_stream()
        dispatch_stream.wait_event(self.route_ready_event)
        hidden_states.record_stream(dispatch_stream)
        probs.record_stream(dispatch_stream)

        with torch.cuda.stream(dispatch_stream):
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

    def _wait_dispatch(
        self, dispatched_input: torch.Tensor, dispatched_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Wait for dispatch and return its outputs on the main stream."""
        assert self.overlap_mode

        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        dispatched_input.record_stream(torch.cuda.current_stream())
        dispatched_probs.record_stream(torch.cuda.current_stream())
        return dispatched_input, dispatched_probs

    def _launch_combine(self, output: torch.Tensor, async_op: bool = False) -> torch.Tensor:
        """Launch combine on the current stream or the shortcut side stream."""
        if not async_op:
            return self.moe_layer.mlp.combine(output)

        combine_stream = self._get_a2a_overlap_stream()
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)
        with torch.cuda.stream(combine_stream):
            return self.moe_layer.mlp.combine(output)

    def _wait_combine(self, combined_output: torch.Tensor) -> torch.Tensor:
        """Wait for the asynchronous combine and return its output on the main stream."""
        torch.cuda.current_stream().wait_stream(self._get_a2a_overlap_stream())
        combined_output.record_stream(torch.cuda.current_stream())
        return combined_output

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context,
        rotary_pos_emb,
        sequence_len_offset,
        packed_seq_params,
        padding_mask,
        quant_context_factory,
        quant_config,
    ):
        """Run the eager schedule with each physical layer's quantization context."""

        # Launch the moe_router
        with quant_context_factory(quant_config, self.moe_layer_num):
            route_input, route_probs = self._moe_router_preprocess(
                shortcut_hidden=hidden_states, padding_mask=padding_mask
            )
            if self.overlap_mode:
                self.route_ready_event.record(torch.cuda.current_stream())

        # Launch the input and attn of the attention layer
        with quant_context_factory(quant_config, self.attn_layer_num):
            paired_state = self.compute_layer.forward_pre_attn_and_core_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
            )

        # Launch the dispatch, experts, and combine
        with quant_context_factory(quant_config, self.moe_layer_num):
            dispatched_input, dispatched_probs = self._launch_dispatch(
                route_input, route_probs, async_op=self.overlap_mode
            )
            if self.overlap_mode:
                dispatch_output = self._wait_dispatch(dispatched_input, dispatched_probs)

            output, _ = self.moe_layer.mlp.routed_experts_compute(
                dispatched_input, dispatched_probs
            )
            combined_output = self._launch_combine(output, async_op=self.overlap_mode)
            if self.shortcut_pre_mlp_layernorm_checkpoint is not None:
                self.shortcut_pre_mlp_layernorm_checkpoint.discard_output_and_register_recompute(
                    combined_output
                )
                self.shortcut_pre_mlp_layernorm_checkpoint = None

        # launch the output layer of the attention layer
        with quant_context_factory(quant_config, self.attn_layer_num):
            attn_layer_output = self.compute_layer.forward_post_core_attn(*paired_state)
            if isinstance(attn_layer_output, tuple):
                attn_layer_output = attn_layer_output[0]

        # launch the moe shared experts and combine attn and moe layer outputs
        with quant_context_factory(quant_config, self.moe_layer_num):
            shared_expert_output = self._moe_shared_experts(attn_layer_output)
            if self.overlap_mode:
                combined_output = self._wait_combine(combined_output)

            # Ensure the combine autograd node is scheduled first before shared_experts
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
            return self._postprocess(attn_layer_output, combined_output, shared_expert_output)
