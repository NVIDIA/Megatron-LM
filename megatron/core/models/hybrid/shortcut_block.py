# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from enum import Enum, auto
from functools import partial

import torch

from megatron.core.transformer.moe.shortcut_cudagraph import (
    PersistentBuffer,
    RecordCombineGradReady as _RecordCombineGradReady,
    RouteGradFromPersistentBuffers as _RouteGradFromPersistentBuffers,
)
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr


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


class _RouteInputCompute(GraphableMegatronModule):
    """Run shortcut routing together with the paired SSM/attention input compute."""

    def __init__(
        self,
        compute_layer,
        moe_layer,
        is_mamba: bool,
        execution_mode: ShortcutExecutionMode,
        persistent_slot_count: int,
        token_dispatcher_attr_slots: tuple[dict, ...],
    ):
        self._execution_mode = execution_mode
        self._enable_cudagraph = execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
        if self._enable_cudagraph:
            assert compute_layer.config.pipeline_model_parallel_size == 1, (
                "Fused shortcut CUDA graphs currently require pipeline parallel size 1"
            )

        # Keep ownership static without registering a duplicate MoE module path.
        object.__setattr__(self, 'moe_layer', moe_layer)
        self._persistent_slots = []
        self._token_dispatcher_attr_slots = token_dispatcher_attr_slots

        if self._enable_cudagraph:
            self._persistent_slots = [
                _RoutePersistentSlot(index) for index in range(persistent_slot_count)
            ]
            self.route_ready_event = None
            self.route_grad_ready_event = None
        elif execution_mode == ShortcutExecutionMode.EAGER_OVERLAP:
            self.route_ready_event = torch.cuda.Event()
            self.route_grad_ready_event = None
        else:
            self.route_ready_event = None
            self.route_grad_ready_event = None

        super().__init__(compute_layer.config)
        self.layer_number = compute_layer.layer_number
        self.is_first_layer = getattr(compute_layer, "is_first_layer", False)
        self.is_last_layer = getattr(compute_layer, "is_last_layer", False)
        self.compute_layer = compute_layer
        self.shortcut_pre_mlp_layernorm = moe_layer.shortcut_pre_mlp_layernorm
        self.moe_mlp = moe_layer.mlp
        self._is_mamba = is_mamba

    def get_persistent_slot(self, index: int) -> _RoutePersistentSlot:
        """Return the stable route state assigned to one graph invocation."""
        if not self._enable_cudagraph:
            raise RuntimeError("Persistent route slots require shortcut CUDA-graph mode")
        if index < 0 or index >= len(self._persistent_slots):
            raise IndexError(
                f"Persistent route slot {index} is outside [0, {len(self._persistent_slots)})"
            )
        return self._persistent_slots[index]

    def create_mcore_cudagraph_manager(self, config):
        """Create the forward-overlap composite graph."""
        assert config.cuda_graph_impl == "local"
        if self._enable_cudagraph:
            from megatron.core.transformer.cuda_graphs import CudaGraphManager

            self.cudagraph_manager_route_input_compute = CudaGraphManager(
                config, self, function_name="forward"
            )

    def forward(
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
        slot = None
        token_dispatcher_attrs = None
        if self._enable_cudagraph:
            slot = self.get_persistent_slot(persistent_slot)
            token_dispatcher_attrs = self._token_dispatcher_attr_slots[persistent_slot]
        route_kwargs = dict(shortcut_hidden=hidden_states, padding_mask=padding_mask)
        if token_dispatcher_attrs is not None:
            route_kwargs["token_dispatcher_attrs"] = token_dispatcher_attrs
        route_input, route_probs = self.moe_layer.shortcut_route_preprocess(**route_kwargs)

        route_grad_dependency = None
        if self._enable_cudagraph:
            slot.route_input_buffer.copy_from(route_input)
            slot.route_probs_buffer.copy_from(route_probs)
            slot.route_input_grad_buffer.acquire_like(route_input)
            slot.route_probs_grad_buffer.acquire_like(route_probs)
            slot.route_ready_event.record(torch.cuda.current_stream())

            # Create this node before paired compute. Giving it the lowest priority makes the
            # captured backward run newer attention/SSM nodes before this event wait.
            route_grad_dependency = _RouteGradFromPersistentBuffers.apply(
                route_input, route_probs, slot
            )
            set_tensor_grad_fn_sequence_sr(route_grad_dependency, 0)
        elif self._execution_mode == ShortcutExecutionMode.EAGER_OVERLAP:
            assert self.route_ready_event is not None
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

        if self._enable_cudagraph:
            paired_state = (paired_state[0] + route_grad_dependency, *paired_state[1:])
            return paired_state
        return route_input, route_probs, paired_state


class _OutputProjSharedExperts(GraphableMegatronModule):
    """Run output projection, shared experts, and optional shortcut postprocess."""

    def __init__(
        self,
        compute_layer,
        moe_layer,
        is_mamba: bool,
        execution_mode: ShortcutExecutionMode,
        persistent_slot_count: int,
        token_dispatcher_attr_slots: tuple[dict, ...],
    ):
        self._enable_cudagraph = execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
        if self._enable_cudagraph:
            assert compute_layer.config.pipeline_model_parallel_size == 1, (
                "Fused shortcut CUDA graphs currently require pipeline parallel size 1"
            )
        super().__init__(compute_layer.config)
        self.layer_number = compute_layer.layer_number
        self.is_first_layer = getattr(moe_layer, "is_first_layer", False)
        self.is_last_layer = getattr(moe_layer, "is_last_layer", False)

        self._persistent_slots = []
        self._token_dispatcher_attr_slots = token_dispatcher_attr_slots

        # Each outstanding fused output/shared/postprocess graph consumes its own allocation after
        # the side-stream combine populates it. Events are external so their wait/record operations
        # can become CUDA graph nodes while the matching operation remains outside the graph.
        if self._enable_cudagraph:
            self._persistent_slots = [
                _OutputPersistentSlot(index) for index in range(persistent_slot_count)
            ]

        # This module stays outside the model's registered module tree. Registering the
        # participating modules here exposes their parameters to CUDA-graph backward capture
        # without introducing duplicate checkpoint paths in HybridStack.
        self.compute_layer = compute_layer
        self.shared_pre_mlp_layernorm = moe_layer.pre_mlp_layernorm
        self.shared_experts = moe_layer.mlp.shared_experts
        postprocess_modules, postprocess_parameters = moe_layer.mlp.shortcut_graph_participants()
        self.postprocess_modules = torch.nn.ModuleList(postprocess_modules)
        self.postprocess_parameters = torch.nn.ParameterList(postprocess_parameters)
        self._is_mamba = is_mamba
        # Keep ownership static without registering a duplicate MoE module path.
        object.__setattr__(self, 'moe_layer', moe_layer)

    def get_persistent_slot(self, index: int) -> _OutputPersistentSlot:
        """Return the stable combine state assigned to one graph invocation."""
        if not self._enable_cudagraph:
            raise RuntimeError("Persistent output slots require shortcut CUDA-graph mode")
        if index < 0 or index >= len(self._persistent_slots):
            raise IndexError(
                f"Persistent output slot {index} is outside [0, {len(self._persistent_slots)})"
            )
        return self._persistent_slots[index]

    def get_persistent_combined_output_buffer(
        self, persistent_slot: int, like: torch.Tensor
    ) -> torch.Tensor:
        """Return the PP=1 persistent buffer used at the combine/graph boundary."""
        slot = self.get_persistent_slot(persistent_slot)
        return slot.combined_output_buffer.acquire_like(like)

    def create_mcore_cudagraph_manager(self, config):
        """Capture output projection, shared experts, and postprocess as one local graph."""
        assert config.cuda_graph_impl == "local"
        if self._enable_cudagraph:
            from megatron.core.transformer.cuda_graphs import CudaGraphManager

            # Keep this off the reserved ``cudagraph_manager`` attribute. This is a method
            # manager: its wrapper preserves the original return type by unwrapping a singleton
            # graph output. Registering it as the module-level manager would bypass that wrapper
            # in GraphableMegatronModule.__call__ and return ``(hidden_states,)`` after replay.
            self.cudagraph_manager_output_shared_postprocess = CudaGraphManager(
                config, self, function_name="forward"
            )

    def forward(
        self,
        *compute_state,
        combined_output=None,
        inference_context=None,
        padding_mask=None,
        persistent_slot: int = 0,
    ):
        if self._enable_cudagraph:
            slot = self.get_persistent_slot(persistent_slot)
            # End the lifetime of the cross-graph route output with an ordinary captured copy.
            # The output projection and its backward then use graph-local storage, allowing the
            # shared graph mempool to reuse the route-output allocation after this graph starts.
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

        if self._enable_cudagraph:
            torch.cuda.current_stream().wait_event(slot.combine_ready_event)
            # During backward capture this becomes an external event-record node immediately
            # after postprocess has produced d(combined_output). Giving the marker maximum
            # sequence priority ensures it is visited before the independent shared/output path.
            combined_output = _RecordCombineGradReady.apply(
                combined_output, slot.combine_grad_ready_event
            )
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)

        return self.postprocess(
            hidden_states,
            combined_output,
            shared_expert_output,
            token_dispatcher_attrs=(
                self._token_dispatcher_attr_slots[persistent_slot]
                if self._enable_cudagraph
                else None
            ),
        )

    def postprocess(
        self,
        hidden_states,
        combined_output,
        shared_expert_output,
        token_dispatcher_attrs=None,
    ):
        """Apply shortcut postprocess through the module shared by every schedule."""
        if token_dispatcher_attrs is None:
            return self.moe_layer.shortcut_postprocess_with_combined_output(
                hidden_states,
                combined_output,
                shared_expert_output,
            )
        return self.moe_layer.shortcut_postprocess_with_combined_output(
            hidden_states,
            combined_output,
            shared_expert_output,
            token_dispatcher_attrs=token_dispatcher_attrs,
        )


class ShortcutMoEBlock:
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
        self.compute_layer = compute_layer
        self.moe_layer = moe_layer
        self.execution_mode = ShortcutExecutionMode.resolve(
            enable_cudagraph=enable_cudagraph,
            overlap_a2a=overlap_a2a,
        )
        self._persistent_slot_count = 1
        if (
            self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP
            and is_mtp_layer
            and compute_layer.config.mtp_use_repeated_layer
        ):
            if not compute_layer.config.mtp_num_layers:
                raise ValueError("Repeated MTP shortcut CUDA graphs require mtp_num_layers > 0")
            self._persistent_slot_count = compute_layer.config.mtp_num_layers
        self._next_persistent_slot = 0
        self._token_dispatcher_attr_slots = tuple(
            {} for _ in range(self._persistent_slot_count)
        )
        self.route_input_compute = _RouteInputCompute(
            compute_layer,
            moe_layer,
            is_mamba=is_mamba,
            execution_mode=self.execution_mode,
            persistent_slot_count=self._persistent_slot_count,
            token_dispatcher_attr_slots=self._token_dispatcher_attr_slots,
        )
        self.output_shared = _OutputProjSharedExperts(
            compute_layer,
            moe_layer,
            is_mamba=is_mamba,
            execution_mode=self.execution_mode,
            persistent_slot_count=self._persistent_slot_count,
            token_dispatcher_attr_slots=self._token_dispatcher_attr_slots,
        )

    def _acquire_persistent_slot(self) -> int:
        """Assign repeated MTP invocations stable slots in forward execution order."""
        slot = self._next_persistent_slot
        self._next_persistent_slot = (slot + 1) % self._persistent_slot_count
        return slot

    @property
    def cudagraph_manager(self):
        """Return the fused phase's graph manager when capture is enabled."""
        return getattr(
            self.output_shared, 'cudagraph_manager_output_shared_postprocess', None
        )

    @property
    def route_input_cudagraph_manager(self):
        """Return the forward-overlap graph manager when enabled."""
        return getattr(self.route_input_compute, 'cudagraph_manager_route_input_compute', None)

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

    def launch_dispatch(self, persistent_slot: int, backward_dependency: torch.Tensor):
        """Launch dispatch from persistent inputs after the route/input graph is queued."""
        target = self.route_input_compute
        if not target._enable_cudagraph:
            raise RuntimeError("Persistent dispatch inputs require shortcut CUDA-graph mode")
        slot = target.get_persistent_slot(persistent_slot)
        route_input = slot.route_input_buffer.tensor
        route_probs = slot.route_probs_buffer.tensor
        self.moe_layer._restore_token_dispatcher_attrs_for_dispatch(
            route_probs,
            self._token_dispatcher_attr_slots[persistent_slot],
        )
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
            self.route_input_compute.route_ready_event,
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
            return self.output_shared.postprocess(
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
            paired_state = self.route_input_compute(
                hidden_states,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                sequence_len_offset=sequence_len_offset,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
                persistent_slot=persistent_slot,
            )

        self.launch_dispatch(persistent_slot, paired_state[0])

        with quant_context_factory(quant_config, layer_number):
            output_slot = self.output_shared.get_persistent_slot(persistent_slot)
            combined_output = moe_layer.shortcut_wait_dispatch_and_launch_combine(
                persistent_output_factory=partial(
                    self.output_shared.get_persistent_combined_output_buffer,
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
