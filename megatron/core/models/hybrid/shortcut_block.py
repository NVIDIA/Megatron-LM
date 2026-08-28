# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

from enum import Enum, auto
from typing import Sequence

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.transformer.module import MegatronModule, SplitOutputProjection
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module

_PairedState = tuple[torch.Tensor, ...]
_DispatchOutput = tuple[torch.Tensor, torch.Tensor]


class ShortcutExecutionMode(Enum):
    """Supported eager execution schedules for a shortcut compute/MoE pair."""

    EAGER_SERIAL = auto()
    EAGER_OVERLAP = auto()

    @classmethod
    def resolve(cls, *, overlap_a2a: bool):
        """Resolve the eager schedule from the shortcut communication-overlap setting."""
        return cls.EAGER_OVERLAP if overlap_a2a else cls.EAGER_SERIAL


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

        paired_type = layer_type_list[physical_index]
        if paired_type not in (LayerSymbols.MAMBA, LayerSymbols.GDN, LayerSymbols.ATTENTION):
            raise ValueError("Shortcut MoE must be preceded by a Mamba, GDN, or attention layer")

        compute_layer = layers[physical_index]
        supports_split = (
            isinstance(compute_layer, SplitOutputProjection)
            and compute_layer.supports_split_output_projection()
        )
        if not supports_split:
            raise ValueError(
                f"Shortcut compute layer {paired_type!r} does not support split output projection"
            )
        moe_layer = layers[physical_index + 1]
        grouped_layers.append(
            ShortcutMoEBlock(compute_layer, moe_layer, overlap_a2a=config.moe_shortcut_parallel)
        )
        physical_index += 2

    return grouped_layers


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one compute-layer/shortcut-MoE pair."""

    _parallel_streams: dict[int, torch.cuda.Stream] = {}

    def __init__(self, compute_layer, moe_layer, overlap_a2a: bool):
        super().__init__(compute_layer.config)

        self.execution_mode = ShortcutExecutionMode.resolve(overlap_a2a=overlap_a2a)
        self.layer_number = compute_layer.layer_number
        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)
        self.tp_group = moe_layer.mlp.tp_group
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer

        # The shortcut path uses the same normalization implementation and configuration as
        # the MoE path, but owns an independent parameter. Keeping ownership on the registered
        # shortcut block avoids adding a shortcut-only field to every transformer-layer spec.
        self.shortcut_pre_mlp_layernorm = moe_layer.submodules_config.pre_mlp_layernorm(
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.shortcut_post_norm = torch.nn.RMSNorm(
            self.config.hidden_size, eps=self.config.layernorm_epsilon
        )
        for parameter in self.shortcut_post_norm.parameters():
            setattr(parameter, 'sequence_parallel', self.config.sequence_parallel)

        self.route_ready_event = (
            torch.cuda.Event()
            if self.execution_mode == ShortcutExecutionMode.EAGER_OVERLAP
            else None
        )

    def route_input_compute(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=None,
        padding_mask=None,
    ) -> tuple[torch.Tensor, torch.Tensor, _PairedState]:
        """Run shortcut routing together with paired input-side compute."""
        route_input, route_probs = self._shortcut_route_preprocess(
            shortcut_hidden=hidden_states, padding_mask=padding_mask
        )
        if self.route_ready_event is not None:
            self.route_ready_event.record(torch.cuda.current_stream())

        paired_state = self.compute_layer.forward_pre_output_proj(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        if not paired_state:
            raise RuntimeError("Shortcut input projection returned an empty paired state")
        return route_input, route_probs, paired_state

    def output_shared(
        self, *compute_state, inference_context=None, padding_mask=None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run output projection and shared experts for the paired compute layer."""
        if not compute_state:
            raise RuntimeError("Shortcut output requires a non-empty paired state")

        compute_result = self.compute_layer.forward_output_proj(
            *compute_state, inference_context=inference_context, padding_mask=padding_mask
        )
        hidden_states = compute_result[0] if isinstance(compute_result, tuple) else compute_result
        shared_expert_output = self._shortcut_shared_experts(hidden_states)
        return hidden_states, shared_expert_output

    def _shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
        """Run shortcut normalization, routing, and dispatch preprocessing."""
        shortcut_input = apply_module(self.shortcut_pre_mlp_layernorm)(shortcut_hidden)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = self.moe_layer.mlp.route(shortcut_input, padding_mask)
        return self.moe_layer.mlp.preprocess(shortcut_input, probs, routing_map)

    def _shortcut_shared_experts(self, hidden_states):
        """Run the paired MoE layer's pre-MLP norm and shared experts."""
        pre_mlp_output = self.moe_layer._forward_pre_mlp_layernorm(hidden_states)
        return self.moe_layer.mlp.shared_experts_compute(pre_mlp_output)

    def _postprocess(self, hidden_states, combined_output, shared_expert_output):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA."""
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        output = self.shortcut_post_norm(output)
        output = self.moe_layer._apply_mlp_bda_step((output, None), residual)
        return output[0] if isinstance(output, tuple) else output

    @classmethod
    def _get_parallel_stream(cls) -> torch.cuda.Stream:
        """Return the shared high-priority shortcut stream for the current CUDA device."""
        device_index = torch.cuda.current_device()
        stream = cls._parallel_streams.get(device_index)
        if stream is None:
            stream = torch.cuda.Stream(priority=-1)
            cls._parallel_streams[device_index] = stream
        return stream

    def _launch_dispatch_async(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, ready_event: torch.cuda.Event
    ) -> _DispatchOutput:
        """Launch the A2A dispatch on the shortcut side stream."""
        dispatch_stream = self._get_parallel_stream()
        dispatch_stream.wait_event(ready_event)
        hidden_states.record_stream(dispatch_stream)
        probs.record_stream(dispatch_stream)

        with torch.cuda.stream(dispatch_stream):
            return self.moe_layer.mlp.dispatch(hidden_states, probs)

    def _wait_dispatch(self, dispatch_output: _DispatchOutput) -> _DispatchOutput:
        """Wait for dispatch and return its outputs on the main stream."""
        dispatch_stream = self._get_parallel_stream()
        torch.cuda.current_stream().wait_stream(dispatch_stream)

        dispatched_input, dispatched_probs = dispatch_output
        main_stream = torch.cuda.current_stream()
        dispatched_input.record_stream(main_stream)
        dispatched_probs.record_stream(main_stream)
        return dispatched_input, dispatched_probs

    def _launch_combine_async(self, output: torch.Tensor) -> torch.Tensor:
        """Launch the A2A combine on the shortcut side stream."""
        combine_stream = self._get_parallel_stream()
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)

        with torch.cuda.stream(combine_stream):
            return self.moe_layer.mlp.combine(output)

    def _wait_combine(self, combined_output: torch.Tensor) -> torch.Tensor:
        """Wait for combine and return its output on the main stream."""
        combine_stream = self._get_parallel_stream()
        torch.cuda.current_stream().wait_stream(combine_stream)

        combined_output.record_stream(torch.cuda.current_stream())
        set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
        return combined_output

    def _wait_dispatch_and_launch_combine(self, dispatch_output: _DispatchOutput) -> torch.Tensor:
        """Run routed experts after dispatch and launch combine asynchronously."""
        dispatched_input, probs = self._wait_dispatch(dispatch_output)
        output, _ = self.moe_layer.mlp.routed_experts_compute(dispatched_input, probs)
        return self._launch_combine_async(output)

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
        """Run the eager schedule selected when this shortcut pair was constructed."""
        layer_number = self.moe_layer.layer_number - 1

        with quant_context_factory(quant_config, layer_number):
            route_input, route_probs, paired_state = self.route_input_compute(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

        if self.execution_mode == ShortcutExecutionMode.EAGER_SERIAL:
            with quant_context_factory(quant_config, layer_number):
                dispatched_input, dispatched_probs = self.moe_layer.mlp.dispatch(
                    route_input, route_probs
                )
                routed_output, _ = self.moe_layer.mlp.routed_experts_compute(
                    dispatched_input, dispatched_probs
                )
                combined_output = self.moe_layer.mlp.combine(routed_output)
                set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)

            with quant_context_factory(quant_config, layer_number):
                projected_hidden, shared_expert_output = self.output_shared(
                    *paired_state, inference_context=inference_context, padding_mask=padding_mask
                )
                return self._postprocess(projected_hidden, combined_output, shared_expert_output)

        assert self.route_ready_event is not None
        dispatch_output = self._launch_dispatch_async(
            route_input, route_probs, self.route_ready_event
        )

        with quant_context_factory(quant_config, layer_number):
            combined_output = self._wait_dispatch_and_launch_combine(dispatch_output)
            projected_hidden, shared_expert_output = self.output_shared(
                *paired_state, inference_context=inference_context, padding_mask=padding_mask
            )

        with quant_context_factory(quant_config, layer_number):
            combined_output = self._wait_combine(combined_output)
            return self._postprocess(projected_hidden, combined_output, shared_expert_output)
