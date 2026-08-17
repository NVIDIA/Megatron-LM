# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from enum import Enum, auto
from functools import partial
from typing import Sequence

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.moe.shortcut_cudagraph import (
    PersistentBuffer,
    RecordCombineGradReady as _RecordCombineGradReady,
    RouteGradFromPersistentBuffers as _RouteGradFromPersistentBuffers,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class ShortcutExecutionMode(Enum):
    """Supported execution schedules for a shortcut compute/MoE pair."""

    EAGER_SERIAL = auto()
    EAGER_OVERLAP = auto()
    CUDA_GRAPH_OVERLAP = auto()

    @classmethod
    def resolve(cls, *, enable_cudagraph: bool, overlap_a2a: bool):
        """Resolve the schedule and reject the unsupported graph/serial combination."""
        if enable_cudagraph:
            if not overlap_a2a:
                raise ValueError("Shortcut MoE CUDA graphs require moe_shortcut_parallel")
            return cls.CUDA_GRAPH_OVERLAP
        if overlap_a2a:
            return cls.EAGER_OVERLAP
        return cls.EAGER_SERIAL


def group_layers_into_shortcut_blocks(
    layers: torch.nn.ModuleList,
    layer_type_list: Sequence[str],
    config: TransformerConfig,
    *,
    is_mtp_layer: bool = False,
) -> torch.nn.ModuleList:
    """Group physical layers into their registered shortcut-block hierarchy.

    Grouping updates the layer names through the returned ``ModuleList`` hierarchy. Layers not
    followed by an MoE remain direct children of the returned ``ModuleList``.

    Args:
        layers: Physical layers in execution order.
        layer_type_list: Physical layer symbols in execution order.
        config: Transformer configuration controlling shortcut scheduling.
        is_mtp_layer: Whether the layers belong to a repeated MTP block.

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
        if paired_type not in (
            LayerSymbols.MAMBA,
            LayerSymbols.GDN,
            LayerSymbols.ATTENTION,
        ):
            raise ValueError("Shortcut MoE must be preceded by a Mamba, GDN, or attention layer")

        compute_layer = layers[physical_index]
        if not getattr(compute_layer, '_supports_split_input_output', False):
            raise ValueError(
                f"Shortcut compute layer {paired_type!r} does not support split input/output"
            )
        moe_layer = layers[physical_index + 1]
        enable_cudagraph = (
            getattr(compute_layer, '_shortcut_graph_output_proj', False)
            and getattr(moe_layer, '_shortcut_graph_shared_experts', False)
        )
        grouped_layers.append(
            ShortcutMoEBlock(
                compute_layer,
                moe_layer,
                is_mamba=paired_type == LayerSymbols.MAMBA,
                is_mtp_layer=is_mtp_layer,
                enable_cudagraph=enable_cudagraph,
                overlap_a2a=config.moe_shortcut_parallel,
            )
        )
        physical_index += 2

    return grouped_layers


class _RoutePersistentSlot:
    """Persistent route/gradient storage and events for one outstanding invocation."""

    def __init__(self, index: int):
        suffix = f" slot {index}"
        self.route_input_buffer = PersistentBuffer(
            f"route input{suffix}", requires_grad=True
        )
        self.route_probs_buffer = PersistentBuffer(
            f"route probabilities{suffix}", requires_grad=True
        )
        self.route_input_grad_buffer = PersistentBuffer(f"route input gradient{suffix}")
        self.route_probs_grad_buffer = PersistentBuffer(
            f"route probability gradient{suffix}"
        )
        self.route_ready_event = torch.cuda.Event(external=True)
        self.route_grad_ready_event = torch.cuda.Event(external=True)


class _OutputPersistentSlot:
    """Persistent combine storage and events for one outstanding invocation."""

    def __init__(self, index: int):
        self.combined_output_buffer = PersistentBuffer(
            f"combined output slot {index}",
            prebound_graph_input=True,
            detach_on_reuse=True,
        )
        self.combine_ready_event = torch.cuda.Event(external=True)
        self.combine_grad_ready_event = torch.cuda.Event(external=True)


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one compute-layer/shortcut-MoE pair."""

    def __init__(
        self,
        compute_layer,
        moe_layer,
        is_mamba: bool,
        enable_cudagraph: bool,
        overlap_a2a: bool,
        is_mtp_layer: bool = False,
    ):
        execution_mode = ShortcutExecutionMode.resolve(
            enable_cudagraph=enable_cudagraph,
            overlap_a2a=overlap_a2a,
        )
        if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            assert compute_layer.config.pipeline_model_parallel_size == 1, (
                "Fused shortcut CUDA graphs currently require pipeline parallel size 1"
            )
        persistent_slot_count = 1
        if (
            execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
            and is_mtp_layer
            and compute_layer.config.mtp_use_repeated_layer
        ):
            if not compute_layer.config.mtp_num_layers:
                raise ValueError("Repeated MTP shortcut CUDA graphs require mtp_num_layers > 0")
            persistent_slot_count = compute_layer.config.mtp_num_layers

        super().__init__(compute_layer.config)

        self.execution_mode = execution_mode
        self._persistent_slot_count = persistent_slot_count
        self._is_mamba = is_mamba
        self._route_layer_boundaries = (
            getattr(compute_layer, "is_first_layer", False),
            getattr(compute_layer, "is_last_layer", False),
        )
        self._output_layer_boundaries = (
            getattr(moe_layer, "is_first_layer", False),
            getattr(moe_layer, "is_last_layer", False),
        )
        self.layer_number = compute_layer.layer_number
        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)
        self.tp_group = moe_layer.mlp.tp_group
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer

        # Move canonical ownership of shortcut post-norm to the registered pair wrapper.
        assert moe_layer.mlp._shortcut_post_norm is not None
        self.shortcut_post_norm = moe_layer.mlp._shortcut_post_norm
        moe_layer.mlp._shortcut_post_norm = None

        self._route_persistent_slots = []
        self._output_persistent_slots = []
        self.route_ready_event = None
        if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            self._route_persistent_slots = [
                _RoutePersistentSlot(index) for index in range(persistent_slot_count)
            ]
            self._output_persistent_slots = [
                _OutputPersistentSlot(index) for index in range(persistent_slot_count)
            ]
        elif execution_mode == ShortcutExecutionMode.EAGER_OVERLAP:
            self.route_ready_event = torch.cuda.Event()

        self._next_persistent_slot = 0
        if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            self.create_mcore_cudagraph_manager(self.config)

    def create_mcore_cudagraph_manager(self, config):
        """Create the two method-level CUDA graphs owned by this registered pair."""
        assert config.cuda_graph_impl == "local"
        if self.execution_mode != ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            return

        from megatron.core.transformer.cuda_graphs import CudaGraphManager

        route_participants = (
            self.compute_layer,
            self.moe_layer.pre_mlp_layernorm,
            self.moe_layer.mlp,
        )
        output_participants = [
            self.compute_layer,
            self.moe_layer.pre_mlp_layernorm,
        ]
        shared_experts = getattr(self.moe_layer.mlp, 'shared_experts', None)
        if shared_experts is not None:
            output_participants.append(shared_experts)
        if self.config.moe_latent_size:
            output_participants.append(self.moe_layer.mlp.fc2_latent_proj)
        output_participants.append(self.shortcut_post_norm)

        self.cudagraph_manager_route_input_compute = CudaGraphManager(
            config,
            self,
            function_name="route_input_compute",
            is_first_layer=self._route_layer_boundaries[0],
            is_last_layer=self._route_layer_boundaries[1],
            participant_modules=route_participants,
        )
        self.cudagraph_manager_output_shared_postprocess = CudaGraphManager(
            config,
            self,
            function_name="output_shared",
            is_first_layer=self._output_layer_boundaries[0],
            is_last_layer=self._output_layer_boundaries[1],
            participant_modules=output_participants,
        )

    def get_route_persistent_slot(self, index: int) -> _RoutePersistentSlot:
        """Return the stable routing state assigned to one graph invocation."""
        if self.execution_mode != ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            raise RuntimeError("Persistent route slots require shortcut CUDA-graph mode")
        if index < 0 or index >= len(self._route_persistent_slots):
            raise IndexError(
                f"Persistent route slot {index} is outside "
                f"[0, {len(self._route_persistent_slots)})"
            )
        return self._route_persistent_slots[index]

    def get_output_persistent_slot(self, index: int) -> _OutputPersistentSlot:
        """Return the stable output state assigned to one graph invocation."""
        if self.execution_mode != ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            raise RuntimeError("Persistent output slots require shortcut CUDA-graph mode")
        if index < 0 or index >= len(self._output_persistent_slots):
            raise IndexError(
                f"Persistent output slot {index} is outside "
                f"[0, {len(self._output_persistent_slots)})"
            )
        return self._output_persistent_slots[index]

    def get_persistent_combined_output_buffer(
        self, persistent_slot: int, like: torch.Tensor
    ) -> torch.Tensor:
        """Return the persistent combine buffer for one graph invocation."""
        slot = self.get_output_persistent_slot(persistent_slot)
        return slot.combined_output_buffer.acquire_like(like)

    def route_input_compute(
        self,
        hidden_states,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=None,
        padding_mask=None,
        persistent_slot: int = 0,
    ):
        """Run shortcut routing together with paired input-side compute."""
        slot = None
        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            slot = self.get_route_persistent_slot(persistent_slot)

        route_outputs = self.moe_layer.shortcut_route_preprocess(
            shortcut_hidden=hidden_states, padding_mask=padding_mask
        )
        route_input, route_probs, *token_dispatcher_attr_outputs = route_outputs

        route_grad_dependency = None
        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            slot.route_input_buffer.copy_from(route_input)
            slot.route_probs_buffer.copy_from(route_probs)
            slot.route_input_grad_buffer.acquire_like(route_input)
            slot.route_probs_grad_buffer.acquire_like(route_probs)
            slot.route_ready_event.record(torch.cuda.current_stream())
            route_grad_dependency = _RouteGradFromPersistentBuffers.apply(
                route_input, route_probs, slot
            )
            set_tensor_grad_fn_sequence_sr(route_grad_dependency, 0)
        elif self.execution_mode == ShortcutExecutionMode.EAGER_OVERLAP:
            self.route_ready_event.record(torch.cuda.current_stream())

        compute_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )
        if self._is_mamba:
            paired_state = self.compute_layer.input_proj_ssm(**compute_kwargs)
        else:
            paired_state = self.compute_layer.input_proj_attn(**compute_kwargs)

        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            paired_state = (paired_state[0] + route_grad_dependency, *paired_state[1:])
            return (*paired_state, *token_dispatcher_attr_outputs)
        return route_input, route_probs, paired_state

    def output_shared(
        self,
        *compute_state,
        combined_output=None,
        inference_context=None,
        padding_mask=None,
        persistent_slot: int = 0,
    ):
        """Run output projection, shared experts, and optional shortcut postprocess."""
        slot = None
        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            slot = self.get_output_persistent_slot(persistent_slot)
            compute_state = (compute_state[0].clone(), *compute_state[1:])

        if self._is_mamba:
            compute_result = self.compute_layer.output_proj(*compute_state)
        else:
            compute_result = self.compute_layer.output_proj(
                *compute_state,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )
        hidden_states = compute_result[0] if isinstance(compute_result, tuple) else compute_result

        shared_expert_output = self.moe_layer._shortcut_shared_experts(hidden_states)
        if combined_output is None:
            return hidden_states, shared_expert_output

        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            torch.cuda.current_stream().wait_event(slot.combine_ready_event)
            combined_output = _RecordCombineGradReady.apply(
                combined_output, slot.combine_grad_ready_event
            )
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)

        return self.postprocess(hidden_states, combined_output, shared_expert_output)

    def postprocess(self, hidden_states, combined_output, shared_expert_output):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA."""
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        output = self.shortcut_post_norm(output)
        output = self.moe_layer._forward_post_mlp((output, None), residual)
        return output[0] if isinstance(output, tuple) else output

    def _acquire_persistent_slot(self) -> int:
        """Assign repeated MTP invocations stable slots in forward execution order."""
        slot = self._next_persistent_slot
        self._next_persistent_slot = (slot + 1) % self._persistent_slot_count
        return slot

    @property
    def cudagraph_manager(self):
        """Return the fused phase's graph manager when capture is enabled."""
        return getattr(self, 'cudagraph_manager_output_shared_postprocess', None)

    @property
    def route_input_cudagraph_manager(self):
        """Return the forward-overlap graph manager when enabled."""
        return getattr(self, 'cudagraph_manager_route_input_compute', None)

    def output_and_shared(
        self,
        *compute_state,
        combined_output=None,
        inference_context=None,
        padding_mask=None,
        persistent_slot: int = 0,
    ):
        """Run output projection/shared experts and optionally fused postprocess."""
        return self.output_shared(
            *compute_state,
            combined_output=combined_output,
            inference_context=inference_context,
            padding_mask=padding_mask,
            persistent_slot=persistent_slot,
        )

    def launch_dispatch(
        self,
        persistent_slot: int,
        backward_dependency: torch.Tensor,
        token_dispatcher_attr_outputs,
    ):
        """Launch dispatch from persistent inputs after the route/input graph is queued."""
        if self.execution_mode != ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            raise RuntimeError("Persistent dispatch inputs require shortcut CUDA-graph mode")

        slot = self.get_route_persistent_slot(persistent_slot)
        route_input = slot.route_input_buffer.tensor
        route_probs = slot.route_probs_buffer.tensor
        self.moe_layer._restore_token_dispatcher_attrs(token_dispatcher_attr_outputs)
        self.moe_layer.shortcut_launch_dispatch(
            route_input,
            route_probs,
            slot.route_ready_event,
            backward_dependency=backward_dependency,
            route_grad_buffers=(
                slot.route_input_grad_buffer.tensor,
                slot.route_probs_grad_buffer.tensor,
            ),
            route_grad_ready_event=slot.route_grad_ready_event,
        )

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
        """Run the schedule selected when this shortcut pair was constructed."""
        forward = {
            ShortcutExecutionMode.EAGER_SERIAL: self._forward_eager_serial,
            ShortcutExecutionMode.EAGER_OVERLAP: self._forward_eager_overlap,
            ShortcutExecutionMode.CUDA_GRAPH_OVERLAP: self._forward_cudagraph_overlap,
        }[self.execution_mode]
        return forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            sequence_len_offset=sequence_len_offset,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
            quant_context_factory=quant_context_factory,
            quant_config=quant_config,
        )

    def _forward_eager_serial(
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
        """Run the reference shortcut schedule with ordinary eager autograd."""
        moe_layer = self.moe_layer
        layer_number = moe_layer.layer_number - 1

        with quant_context_factory(quant_config, layer_number):
            permuted_input, dispatch_probs, paired_state = self.route_input_compute(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

        with quant_context_factory(quant_config, layer_number):
            dispatched_input, dispatch_probs = moe_layer.mlp.dispatch(
                permuted_input,
                dispatch_probs,
            )
            routed_output, _ = moe_layer.mlp.routed_experts_compute(
                dispatched_input,
                dispatch_probs,
            )
            combined_output = moe_layer.mlp.combine(routed_output)
            # Shared experts execute later in the shortcut forward than the routed branch.
            # Without an explicit priority, autograd therefore walks the shared branch first.
            # That can launch overlapped dense gradient reduction while HybridEP/expert
            # backward is still pending. Match the eager-overlap schedule by visiting the
            # routed combine backward first.
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)

        with quant_context_factory(quant_config, layer_number):
            return self.output_and_shared(
                *paired_state,
                combined_output=combined_output,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )

    def _forward_eager_overlap(
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
        """Overlap eager A2A with paired compute using ordinary eager autograd."""
        moe_layer = self.moe_layer
        layer_number = moe_layer.layer_number - 1

        with quant_context_factory(quant_config, layer_number):
            permuted_input, dispatch_probs, paired_state = self.route_input_compute(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

        moe_layer.shortcut_launch_dispatch(
            permuted_input,
            dispatch_probs,
            self.route_ready_event,
        )

        with quant_context_factory(quant_config, layer_number):
            # Eager autograd already joins the routed and paired branches at their shared
            # inputs. Do not add the CUDA-graph-only backward scheduling dependency here:
            # repeated MTP invocations can otherwise make multiple same-priority HybridEP
            # collectives ready at once, with no cross-rank ordering guarantee.
            moe_layer.shortcut_wait_dispatch_and_launch_combine()
            projected_hidden, shared_expert_output = self.output_and_shared(
                *paired_state,
                inference_context=inference_context,
                padding_mask=padding_mask,
            )

        with quant_context_factory(quant_config, layer_number):
            combined_output = moe_layer.mlp.wait_combine()
            return self.postprocess(
                projected_hidden,
                combined_output,
                shared_expert_output,
            )

    def _forward_cudagraph_overlap(
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
        """Overlap eager A2A with the two shortcut CUDA-graph compute regions."""
        moe_layer = self.moe_layer
        layer_number = moe_layer.layer_number - 1
        persistent_slot = self._acquire_persistent_slot()

        with quant_context_factory(quant_config, layer_number):
            route_outputs = self.route_input_compute(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                persistent_slot=persistent_slot,
            )

        attr_names = self.moe_layer._local_cudagraph_attr_names or ()
        attr_count = len(attr_names)
        if attr_count:
            paired_state = tuple(route_outputs[:-attr_count])
            token_dispatcher_attr_outputs = tuple(route_outputs[-attr_count:])
        else:
            paired_state = tuple(route_outputs)
            token_dispatcher_attr_outputs = ()

        # The dispatch stream waits on the external event recorded immediately after routing
        # and preprocessing, before the paired Mamba/attention input projection. Its D2H stream
        # therefore waits only for router metadata while the CUDA graph continues independently.
        self.launch_dispatch(
            persistent_slot,
            paired_state[0],
            token_dispatcher_attr_outputs,
        )

        with quant_context_factory(quant_config, layer_number):
            output_slot = self.get_output_persistent_slot(persistent_slot)
            combined_output = moe_layer.shortcut_wait_dispatch_and_launch_combine(
                persistent_output_factory=partial(
                    self.get_persistent_combined_output_buffer,
                    persistent_slot,
                ),
                ready_event=output_slot.combine_ready_event,
                grad_ready_event=output_slot.combine_grad_ready_event,
            )
            return self.output_and_shared(
                *paired_state,
                combined_output=combined_output,
                inference_context=inference_context,
                padding_mask=padding_mask,
                persistent_slot=persistent_slot,
            )
