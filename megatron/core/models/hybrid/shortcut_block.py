# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import weakref
from enum import Enum, auto
from functools import partial
from typing import Sequence

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.module import MegatronModule, SplitOutputProjection
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module


class PersistentBuffer:
    """Own stable CUDA storage used across eager/CUDA-graph boundaries."""

    def __init__(
        self,
        name: str,
        *,
        requires_grad: bool = False,
        prebound_graph_input: bool = False,
        detach_on_reuse: bool = False,
    ):
        self.name = name
        self.requires_grad = requires_grad
        self.prebound_graph_input = prebound_graph_input
        self.detach_on_reuse = detach_on_reuse
        self._tensor = None

    @property
    def tensor(self) -> torch.Tensor:
        if self._tensor is None:
            raise RuntimeError(f"Persistent {self.name} buffer has not been allocated")
        return self._tensor

    @staticmethod
    def _metadata(tensor: torch.Tensor):
        return tuple(tensor.shape), tensor.stride(), tensor.dtype, tensor.device

    def acquire_like(self, like: torch.Tensor) -> torch.Tensor:
        """Allocate once, then validate and return the same storage on every reuse."""
        if not like.is_cuda:
            raise ValueError(f"Persistent {self.name} buffer requires a CUDA tensor template")

        if self._tensor is None:
            from megatron.core.transformer.cuda_graphs import (
                ArgMetadata,
                alloc_tensor_from_graph_mempool,
            )

            metadata = ArgMetadata(like)
            metadata.requires_grad = self.requires_grad
            tensor = alloc_tensor_from_graph_mempool(metadata)
        else:
            expected = self._metadata(self._tensor)
            received = self._metadata(like)
            if received != expected:
                raise AssertionError(
                    f"Persistent {self.name} buffer metadata changed: "
                    f"expected {expected}, received {received}"
                )
            if self.detach_on_reuse:
                is_from_global_mempool = getattr(self._tensor, "is_from_global_mempool", False)
                tensor = self._tensor.detach().requires_grad_(self.requires_grad)
                if is_from_global_mempool:
                    tensor.is_from_global_mempool = True
            else:
                tensor = self._tensor

        if self.prebound_graph_input:
            from megatron.core.transformer.cuda_graphs import mark_cuda_graph_prebound_input

            mark_cuda_graph_prebound_input(tensor)

        self._tensor = tensor
        return tensor

    def copy_from(self, source: torch.Tensor) -> torch.Tensor:
        """Copy a tensor into the stable allocation without adding an autograd edge."""
        destination = self.acquire_like(source)
        with torch.no_grad():
            destination.copy_(source)
        return destination


class AsyncDispatchToPersistentGradBuffers(torch.autograd.Function):
    """Bridge side-stream dispatch with gradients consumed by a CUDA graph."""

    @staticmethod
    def forward(
        ctx,
        route_input,
        route_probs,
        backward_dependency,
        moe_layer,
        dispatch_stream,
        route_input_grad_buffer,
        route_probs_grad_buffer,
        route_grad_ready_event,
    ):
        ctx.dispatch_stream = dispatch_stream
        ctx.route_input_grad_buffer = route_input_grad_buffer
        ctx.route_probs_grad_buffer = route_probs_grad_buffer
        ctx.route_grad_ready_event = route_grad_ready_event

        with torch.enable_grad(), torch.cuda.stream(dispatch_stream):
            dispatch_input = route_input.detach().requires_grad_(route_input.requires_grad)
            dispatch_probs = route_probs.detach().requires_grad_(route_probs.requires_grad)
            token_dispatcher = getattr(moe_layer, 'token_dispatcher', None)
            comm_manager = getattr(token_dispatcher, '_comm_manager', None)
            if comm_manager is not None:
                # Flex retains probabilities as dispatcher state. Point it at the private
                # side-stream autograd alias rather than the CUDA-graph-owned tensor.
                comm_manager.token_probs = dispatch_probs
            dispatched_input, dispatched_probs = moe_layer.dispatch(dispatch_input, dispatch_probs)

        ctx.save_for_backward(dispatch_input, dispatch_probs, dispatched_input, dispatched_probs)
        return dispatched_input.detach(), dispatched_probs.detach()

    @staticmethod
    def backward(ctx, grad_dispatched_input, grad_dispatched_probs):
        dispatch_input, dispatch_probs, dispatched_input, dispatched_probs = ctx.saved_tensors
        dispatch_stream = ctx.dispatch_stream

        dispatch_stream.wait_stream(torch.cuda.current_stream())
        if grad_dispatched_input is not None:
            grad_dispatched_input.record_stream(dispatch_stream)
        if grad_dispatched_probs is not None:
            grad_dispatched_probs.record_stream(dispatch_stream)

        with torch.cuda.stream(dispatch_stream), torch.enable_grad():
            if grad_dispatched_input is None:
                grad_dispatched_input = torch.zeros_like(dispatched_input)
            if grad_dispatched_probs is None:
                grad_dispatched_probs = torch.zeros_like(dispatched_probs)
            grad_route_input, grad_route_probs = torch.autograd.grad(
                outputs=(dispatched_input, dispatched_probs),
                inputs=(dispatch_input, dispatch_probs),
                grad_outputs=(grad_dispatched_input, grad_dispatched_probs),
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            if grad_route_input is None:
                raise RuntimeError(
                    "Private dispatch backward is disconnected from its hidden-state input"
                )
            if grad_route_probs is None:
                grad_route_probs = torch.zeros_like(dispatch_probs)
            with torch.no_grad():
                ctx.route_input_grad_buffer.copy_(grad_route_input)
                ctx.route_probs_grad_buffer.copy_(grad_route_probs)
            ctx.route_grad_ready_event.record(dispatch_stream)

        ctx.dispatch_stream = None
        ctx.route_input_grad_buffer = None
        ctx.route_probs_grad_buffer = None
        ctx.route_grad_ready_event = None
        return None, None, None, None, None, None, None, None


class AsyncCombineToPersistentBuffer(torch.autograd.Function):
    """Bridge eager side-stream combine with a fused main-stream CUDA graph."""

    @staticmethod
    def forward(
        ctx,
        expert_output,
        moe_layer,
        combine_stream,
        persistent_output_factory,
        ready_event,
        grad_ready_event,
    ):
        ctx.main_stream = torch.cuda.current_stream()
        ctx.combine_stream = combine_stream
        ctx.grad_ready_event = grad_ready_event

        with torch.enable_grad(), torch.cuda.stream(combine_stream):
            combine_input = expert_output.detach().requires_grad_(expert_output.requires_grad)
            combined = moe_layer.combine(combine_input)
            persistent_output = persistent_output_factory(combined)
            with torch.no_grad():
                persistent_output.copy_(combined)
            ready_event.record(combine_stream)

        ctx.save_for_backward(combine_input, combined)
        return persistent_output

    @staticmethod
    def backward(ctx, grad_output):
        combine_input, combined = ctx.saved_tensors
        combine_stream = ctx.combine_stream

        combine_stream.wait_event(ctx.grad_ready_event)
        grad_output.record_stream(combine_stream)
        with torch.cuda.stream(combine_stream), torch.enable_grad():
            (grad_input,) = torch.autograd.grad(
                outputs=combined,
                inputs=combine_input,
                grad_outputs=grad_output,
                retain_graph=False,
                create_graph=False,
            )

        main_stream = ctx.main_stream
        main_stream.wait_stream(combine_stream)
        grad_input.record_stream(main_stream)

        ctx.main_stream = None
        ctx.combine_stream = None
        ctx.grad_ready_event = None
        return grad_input, None, None, None, None, None


class RouteGradFromPersistentBuffers(torch.autograd.Function):
    """Inject dispatch gradients into captured router backward at an external event."""

    @staticmethod
    def forward(ctx, route_input, route_probs, slot):
        ctx.slot_ref = weakref.ref(slot)
        return route_input.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        slot = ctx.slot_ref()
        assert slot is not None
        torch.cuda.current_stream().wait_event(slot.route_grad_ready_event)
        ctx.slot_ref = None
        return (slot.route_input_grad_buffer.tensor, slot.route_probs_grad_buffer.tensor, None)


class RecordCombineGradReady(torch.autograd.Function):
    """Record an external event when fused postprocess produces the combine gradient."""

    @staticmethod
    def forward(ctx, combined_output, grad_ready_event):
        ctx.grad_ready_event = grad_ready_event
        return combined_output

    @staticmethod
    def backward(ctx, grad_output):
        ctx.grad_ready_event.record(torch.cuda.current_stream())
        ctx.grad_ready_event = None
        return grad_output, None


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
            ShortcutMoEBlock(
                compute_layer,
                moe_layer,
                is_mtp_layer=is_mtp_layer,
                overlap_a2a=config.moe_shortcut_parallel,
            )
        )
        physical_index += 2

    return grouped_layers


class _RoutePersistentSlot:
    """Persistent route/gradient storage and events for one outstanding invocation."""

    def __init__(self, index: int):
        suffix = f" slot {index}"
        self.route_input_buffer = PersistentBuffer(f"route input{suffix}", requires_grad=True)
        self.route_probs_buffer = PersistentBuffer(
            f"route probabilities{suffix}", requires_grad=True
        )
        self.route_input_grad_buffer = PersistentBuffer(f"route input gradient{suffix}")
        self.route_probs_grad_buffer = PersistentBuffer(f"route probability gradient{suffix}")
        self.route_ready_event = torch.cuda.Event(external=True)
        self.route_grad_ready_event = torch.cuda.Event(external=True)


class _OutputPersistentSlot:
    """Persistent combine storage and events for one outstanding invocation."""

    def __init__(self, index: int):
        self.combined_output_buffer = PersistentBuffer(
            f"combined output slot {index}", prebound_graph_input=True, detach_on_reuse=True
        )
        self.combine_ready_event = torch.cuda.Event(external=True)
        self.combine_grad_ready_event = torch.cuda.Event(external=True)


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one compute-layer/shortcut-MoE pair."""

    _parallel_stream = None

    def __init__(self, compute_layer, moe_layer, overlap_a2a: bool, is_mtp_layer: bool = False):
        enable_cudagraph = CudaGraphModule.shortcut_block in compute_layer.config.cuda_graph_modules
        execution_mode = ShortcutExecutionMode.resolve(
            enable_cudagraph=enable_cudagraph, overlap_a2a=overlap_a2a
        )
        if execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            assert (
                compute_layer.config.pipeline_model_parallel_size == 1
            ), "Fused shortcut CUDA graphs currently require pipeline parallel size 1"
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

        self.shortcut_post_norm = torch.nn.RMSNorm(
            self.config.hidden_size, eps=self.config.layernorm_epsilon
        )
        for parameter in self.shortcut_post_norm.parameters():
            setattr(parameter, 'sequence_parallel', self.config.sequence_parallel)

        self._route_persistent_slots = []
        self._output_persistent_slots = []
        self._cached_dispatch_output = None
        self._cached_combine_output = None
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
        output_participants = [self.compute_layer, self.moe_layer.pre_mlp_layernorm]
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

        route_outputs = self._shortcut_route_preprocess(
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
            route_grad_dependency = RouteGradFromPersistentBuffers.apply(
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
        paired_state = self.compute_layer.forward_pre_output_proj(**compute_kwargs)

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

        compute_result = self.compute_layer.forward_output_proj(
            *compute_state, inference_context=inference_context, padding_mask=padding_mask
        )
        hidden_states = compute_result[0] if isinstance(compute_result, tuple) else compute_result

        shared_expert_output = self._shortcut_shared_experts(hidden_states)
        if combined_output is None:
            return hidden_states, shared_expert_output

        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            torch.cuda.current_stream().wait_event(slot.combine_ready_event)
            combined_output = RecordCombineGradReady.apply(
                combined_output, slot.combine_grad_ready_event
            )
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)

        return self.postprocess(hidden_states, combined_output, shared_expert_output)

    def _get_local_cudagraph_attr_outputs(self):
        """Expose dispatcher state tensors at the shortcut graph boundary."""
        attr_names, attr_outputs = self.moe_layer._get_token_dispatcher_attrs()
        if self.moe_layer._local_cudagraph_attr_names is None:
            self.moe_layer._local_cudagraph_attr_names = attr_names
        else:
            assert attr_names == self.moe_layer._local_cudagraph_attr_names
        return attr_outputs

    def _shortcut_route_preprocess(self, shortcut_hidden, padding_mask=None):
        """Run shortcut normalization, routing, and dispatch preprocessing."""
        shortcut_input = apply_module(self.moe_layer.pre_mlp_layernorm)(shortcut_hidden)
        if padding_mask is not None:
            padding_mask = padding_mask.transpose(0, 1).bool()
        probs, routing_map = self.moe_layer.mlp.route(shortcut_input, padding_mask)
        permuted_input, probs = self.moe_layer.mlp.preprocess(shortcut_input, probs, routing_map)
        token_dispatcher_attr_outputs = []
        if self.execution_mode == ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            token_dispatcher_attr_outputs = self._get_local_cudagraph_attr_outputs()
        return permuted_input, probs, *token_dispatcher_attr_outputs

    def _shortcut_shared_experts(self, hidden_states):
        """Run the paired MoE layer's pre-MLP norm and shared experts."""
        pre_mlp_output = self.moe_layer._forward_pre_mlp_layernorm(hidden_states)
        return self.moe_layer.mlp.shared_experts_compute(pre_mlp_output)

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

    def launch_dispatch_async(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        ready_event: torch.cuda.Event = None,
        route_grad_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
        route_grad_ready_event: torch.cuda.Event | None = None,
        backward_dependency: torch.Tensor | None = None,
    ):
        """Launch the A2A dispatch on the shortcut side stream."""
        if ShortcutMoEBlock._parallel_stream is None:
            ShortcutMoEBlock._parallel_stream = torch.cuda.Stream(priority=-1)

        dispatch_stream = ShortcutMoEBlock._parallel_stream
        if ready_event is not None:
            dispatch_stream.wait_event(ready_event)
        else:
            dispatch_stream.wait_stream(torch.cuda.current_stream())

        hidden_states.record_stream(dispatch_stream)
        probs.record_stream(dispatch_stream)
        moe_layer = self.moe_layer.mlp

        if route_grad_buffers is not None:
            if route_grad_ready_event is None or backward_dependency is None:
                raise ValueError(
                    "Persistent async dispatch requires a backward dependency and "
                    "route-gradient-ready event"
                )
            dispatched_input, dispatched_probs = AsyncDispatchToPersistentGradBuffers.apply(
                hidden_states,
                probs,
                backward_dependency,
                moe_layer,
                dispatch_stream,
                route_grad_buffers[0],
                route_grad_buffers[1],
                route_grad_ready_event,
            )
            token_dispatcher = getattr(moe_layer, 'token_dispatcher', None)
            comm_manager = getattr(token_dispatcher, '_comm_manager', None)
            if comm_manager is not None:
                # Expert preprocessing must consume the custom Function output so its backward
                # re-enters the private dispatch graph and publishes the router gradients.
                comm_manager.dispatched_probs = dispatched_probs
        else:
            with torch.cuda.stream(dispatch_stream):
                dispatched_input, dispatched_probs = moe_layer.dispatch(hidden_states, probs)
        self._cached_dispatch_output = (dispatched_input, dispatched_probs)

    def wait_dispatch(self):
        """Wait for dispatch and return its outputs on the main stream."""
        dispatch_stream = ShortcutMoEBlock._parallel_stream
        torch.cuda.current_stream().wait_stream(dispatch_stream)

        dispatched_input, dispatched_probs = self._cached_dispatch_output
        self._cached_dispatch_output = None
        main_stream = torch.cuda.current_stream()
        dispatched_input.record_stream(main_stream)
        dispatched_probs.record_stream(main_stream)
        return dispatched_input, dispatched_probs

    def launch_combine_async(
        self,
        output: torch.Tensor,
        persistent_output_factory=None,
        ready_event: torch.cuda.Event | None = None,
        grad_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """Launch the A2A combine on the shortcut side stream."""
        combine_stream = ShortcutMoEBlock._parallel_stream
        combine_stream.wait_stream(torch.cuda.current_stream())
        output.record_stream(combine_stream)
        moe_layer = self.moe_layer.mlp

        if persistent_output_factory is not None:
            if ready_event is None or grad_ready_event is None:
                raise ValueError(
                    "Persistent async combine requires forward-ready and gradient-ready events"
                )
            combined = AsyncCombineToPersistentBuffer.apply(
                output,
                moe_layer,
                combine_stream,
                persistent_output_factory,
                ready_event,
                grad_ready_event,
            )
            from megatron.core.transformer.cuda_graphs import mark_cuda_graph_prebound_input

            mark_cuda_graph_prebound_input(combined)
            set_tensor_grad_fn_sequence_sr(combined, torch.iinfo(torch.int).max)
        else:
            with torch.cuda.stream(combine_stream):
                combined = moe_layer.combine(output)
                if ready_event is not None:
                    ready_event.record(combine_stream)

        self._cached_combine_output = combined if persistent_output_factory is None else None
        return combined

    def wait_combine(self):
        """Wait for combine and return its output on the main stream."""
        combine_stream = ShortcutMoEBlock._parallel_stream
        torch.cuda.current_stream().wait_stream(combine_stream)

        combined = self._cached_combine_output
        self._cached_combine_output = None
        combined.record_stream(torch.cuda.current_stream())
        set_tensor_grad_fn_sequence_sr(combined, torch.iinfo(torch.int).max)
        return combined

    def wait_dispatch_and_launch_combine(
        self,
        persistent_output_factory=None,
        ready_event: torch.cuda.Event | None = None,
        grad_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """Run routed experts after dispatch and launch combine asynchronously."""
        dispatched_input, probs = self.wait_dispatch()
        output, _ = self.moe_layer.mlp.routed_experts_compute(dispatched_input, probs)
        return self.launch_combine_async(
            output,
            persistent_output_factory=persistent_output_factory,
            ready_event=ready_event,
            grad_ready_event=grad_ready_event,
        )

    def launch_dispatch(
        self, persistent_slot: int, backward_dependency: torch.Tensor, token_dispatcher_attr_outputs
    ):
        """Launch dispatch from persistent inputs after the route/input graph is queued."""
        if self.execution_mode != ShortcutExecutionMode.CUDA_GRAPH_OVERLAP:
            raise RuntimeError("Persistent dispatch inputs require shortcut CUDA-graph mode")

        slot = self.get_route_persistent_slot(persistent_slot)
        route_input = slot.route_input_buffer.tensor
        route_probs = slot.route_probs_buffer.tensor
        self.moe_layer._restore_token_dispatcher_attrs(token_dispatcher_attr_outputs)
        self.launch_dispatch_async(
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
                permuted_input, dispatch_probs
            )
            routed_output, _ = moe_layer.mlp.routed_experts_compute(
                dispatched_input, dispatch_probs
            )
            combined_output = moe_layer.mlp.combine(routed_output)
            # Shared experts execute later in the shortcut forward than the routed branch.
            # Without an explicit priority, autograd therefore walks the shared branch first.
            # That can launch overlapped dense gradient reduction while HybridEP/expert
            # backward is still pending. Match the eager-overlap schedule by visiting the
            # routed combine backward first.
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)

        with quant_context_factory(quant_config, layer_number):
            return self.output_shared(
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

        self.launch_dispatch_async(permuted_input, dispatch_probs, self.route_ready_event)

        with quant_context_factory(quant_config, layer_number):
            # Eager autograd already joins the routed and paired branches at their shared
            # inputs. Do not add the CUDA-graph-only backward scheduling dependency here:
            # repeated MTP invocations can otherwise make multiple same-priority HybridEP
            # collectives ready at once, with no cross-rank ordering guarantee.
            self.wait_dispatch_and_launch_combine()
            projected_hidden, shared_expert_output = self.output_shared(
                *paired_state, inference_context=inference_context, padding_mask=padding_mask
            )

        with quant_context_factory(quant_config, layer_number):
            combined_output = self.wait_combine()
            return self.postprocess(projected_hidden, combined_output, shared_expert_output)

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
        self.launch_dispatch(persistent_slot, paired_state[0], token_dispatcher_attr_outputs)

        with quant_context_factory(quant_config, layer_number):
            output_slot = self.get_output_persistent_slot(persistent_slot)
            combined_output = self.wait_dispatch_and_launch_combine(
                persistent_output_factory=partial(
                    self.get_persistent_combined_output_buffer, persistent_slot
                ),
                ready_event=output_slot.combine_ready_event,
                grad_ready_event=output_slot.combine_grad_ready_event,
            )
            return self.output_shared(
                *paired_state,
                combined_output=combined_output,
                inference_context=inference_context,
                padding_mask=padding_mask,
                persistent_slot=persistent_slot,
            )
