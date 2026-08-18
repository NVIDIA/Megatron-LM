# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import weakref
from functools import partial
from typing import TYPE_CHECKING, Sequence

import torch

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols as LayerSymbols
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.shared_experts import set_tensor_grad_fn_sequence_sr
from megatron.core.transformer.transformer_config import TransformerConfig

if TYPE_CHECKING:
    from megatron.core.transformer.cuda_graphs import CudaGraphManager


_PairedState = tuple[torch.Tensor, ...]
_DispatchOutput = tuple[torch.Tensor, torch.Tensor]


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

        tensor = self._tensor
        if tensor is None:
            from megatron.core.transformer.cuda_graphs import (
                ArgMetadata,
                alloc_tensor_from_graph_mempool,
            )

            metadata = ArgMetadata(like)
            metadata.requires_grad = self.requires_grad
            tensor = alloc_tensor_from_graph_mempool(metadata)
        else:
            expected, received = self._metadata(tensor), self._metadata(like)
            if received != expected:
                raise AssertionError(
                    f"Persistent {self.name} buffer metadata changed: "
                    f"expected {expected}, received {received}"
                )
            if self.detach_on_reuse:
                is_from_global_mempool = getattr(tensor, "is_from_global_mempool", False)
                tensor = tensor.detach().requires_grad_(self.requires_grad)
                if is_from_global_mempool:
                    tensor.is_from_global_mempool = True

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
            dispatched_input, dispatched_probs = moe_layer.dispatch(
                dispatch_input, dispatch_probs
            )

        ctx.save_for_backward(
            dispatch_input,
            dispatch_probs,
            dispatched_input,
            dispatched_probs,
        )
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
        return (
            slot.route_input_grad_buffer.tensor,
            slot.route_probs_grad_buffer.tensor,
            None,
        )


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
            )
        )
        physical_index += 2

    return grouped_layers


class _PersistentSlot:
    """Persistent route/combine state for one outstanding graph invocation."""

    def __init__(self, index: int):
        suffix = f" slot {index}"
        self.route_input_buffer = PersistentBuffer(f"route input{suffix}", requires_grad=True)
        self.route_probs_buffer = PersistentBuffer(f"route probabilities{suffix}", requires_grad=True)
        self.route_input_grad_buffer = PersistentBuffer(f"route input gradient{suffix}")
        self.route_probs_grad_buffer = PersistentBuffer(f"route probability gradient{suffix}")
        self.route_ready_event = torch.cuda.Event(external=True)
        self.route_grad_ready_event = torch.cuda.Event(external=True)
        self.combined_output_buffer = PersistentBuffer(
            f"combined output slot {index}",
            prebound_graph_input=True,
            detach_on_reuse=True,
        )
        self.combine_ready_event = torch.cuda.Event(external=True)
        self.combine_grad_ready_event = torch.cuda.Event(external=True)

    def acquire_combined_output(self, like: torch.Tensor) -> torch.Tensor:
        """Return the persistent combined-output allocation for this invocation slot."""
        return self.combined_output_buffer.acquire_like(like)


class _GraphState(torch.nn.Module):
    """Own CUDA-graph managers and persistent state for one shortcut block."""

    def __init__(self, slot_count: int):
        super().__init__()
        if slot_count <= 0:
            raise ValueError("Shortcut CUDA-graph state requires at least one slot")
        self.slots = [_PersistentSlot(index) for index in range(slot_count)]
        self.next_slot = 0
        self.route_manager: CudaGraphManager | None = None
        self.output_manager: CudaGraphManager | None = None

    def get_slot(self, index: int) -> _PersistentSlot:
        """Return a persistent slot after validating its graph-visible index."""
        if index < 0 or index >= len(self.slots):
            raise IndexError(f"Persistent slot {index} is outside [0, {len(self.slots)})")
        return self.slots[index]

    def acquire_slot(self) -> tuple[int, _PersistentSlot]:
        """Rotate through persistent slots in repeated-MTP forward order."""
        if not self.slots:
            raise RuntimeError("Shortcut CUDA-graph state has no persistent slots")
        index = self.next_slot
        slot = self.get_slot(index)
        self.next_slot = (index + 1) % len(self.slots)
        return index, slot


class ShortcutMoEBlock(MegatronModule):
    """Own and execute one compute-layer/shortcut-MoE pair."""

    _parallel_streams: dict[int, torch.cuda.Stream] = {}

    def __init__(
        self,
        compute_layer,
        moe_layer,
        is_mamba: bool,
        enable_cudagraph: bool,
        is_mtp_layer: bool = False,
    ):
        if not compute_layer.config.moe_shortcut_parallel:
            raise ValueError("Shortcut MoE requires moe_shortcut_parallel")
        if enable_cudagraph:
            assert compute_layer.config.pipeline_model_parallel_size == 1, (
                "Fused shortcut CUDA graphs currently require pipeline parallel size 1"
            )
        persistent_slot_count = 1
        if enable_cudagraph and is_mtp_layer and compute_layer.config.mtp_use_repeated_layer:
            if not compute_layer.config.mtp_num_layers:
                raise ValueError("Repeated MTP shortcut CUDA graphs require mtp_num_layers > 0")
            persistent_slot_count = compute_layer.config.mtp_num_layers

        super().__init__(compute_layer.config)

        self.enable_cudagraph = enable_cudagraph
        self._is_mamba = is_mamba
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

        self._graph_state: _GraphState | None = None
        if enable_cudagraph:
            self._initialize_cudagraph_state(persistent_slot_count)
        else:
            self.route_ready_event = torch.cuda.Event()

    def _initialize_cudagraph_state(self, slot_count: int) -> None:
        """Construct graph-only persistent slots and method-level graph managers."""
        if self._graph_state is not None:
            raise RuntimeError("Shortcut CUDA-graph state has already been initialized")
        self._graph_state = _GraphState(slot_count)
        self.create_mcore_cudagraph_manager(self.config)

    def _require_graph_state(self) -> _GraphState:
        """Return graph-only state or fail at an invalid graph scheduling boundary."""
        if self._graph_state is None:
            raise RuntimeError("Shortcut CUDA-graph state is not initialized")
        return self._graph_state

    def create_mcore_cudagraph_manager(self, config):
        """Create the two method-level CUDA graphs owned by this registered pair."""
        assert config.cuda_graph_impl == "local"
        if not self.enable_cudagraph:
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

        graph_state = self._require_graph_state()
        manager_factory = partial(CudaGraphManager, config, self)
        graph_state.route_manager = manager_factory(
            function_name="route_input_compute",
            is_first_layer=getattr(self.compute_layer, "is_first_layer", False),
            is_last_layer=getattr(self.compute_layer, "is_last_layer", False),
            participant_modules=route_participants,
        )
        graph_state.output_manager = manager_factory(
            function_name="output_shared",
            is_first_layer=getattr(self.moe_layer, "is_first_layer", False),
            is_last_layer=getattr(self.moe_layer, "is_last_layer", False),
            participant_modules=output_participants,
        )

    def _input_projection(self, **compute_kwargs) -> _PairedState:
        """Run the paired layer's input-side projection."""
        projection = self.compute_layer.input_proj_ssm if self._is_mamba else self.compute_layer.input_proj_attn
        paired_state = projection(**compute_kwargs)
        if not paired_state:
            raise RuntimeError("Shortcut input projection returned an empty paired state")
        return paired_state

    def _output_projection(
        self,
        compute_state: _PairedState,
        *,
        inference_context=None,
        padding_mask=None,
    ) -> torch.Tensor:
        """Run the paired layer's output projection and normalize its signature."""
        if not compute_state:
            raise RuntimeError("Shortcut output projection requires a non-empty paired state")
        projection_kwargs = (
            {} if self._is_mamba else {"inference_context": inference_context, "padding_mask": padding_mask}
        )
        compute_result = self.compute_layer.output_proj(*compute_state, **projection_kwargs)
        return compute_result[0] if isinstance(compute_result, tuple) else compute_result

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
        if self.enable_cudagraph:
            slot = self._require_graph_state().get_slot(persistent_slot)

        route_outputs = self.moe_layer.shortcut_route_preprocess(
            shortcut_hidden=hidden_states, padding_mask=padding_mask
        )
        route_input, route_probs, *token_dispatcher_attr_outputs = route_outputs

        if self.enable_cudagraph:
            slot.route_input_buffer.copy_from(route_input)
            slot.route_probs_buffer.copy_from(route_probs)
            slot.route_input_grad_buffer.acquire_like(route_input)
            slot.route_probs_grad_buffer.acquire_like(route_probs)
            slot.route_ready_event.record(torch.cuda.current_stream())
            route_grad_dependency = RouteGradFromPersistentBuffers.apply(
                route_input, route_probs, slot
            )
            set_tensor_grad_fn_sequence_sr(route_grad_dependency, 0)
        else:
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
        paired_state = self._input_projection(**compute_kwargs)

        if self.enable_cudagraph:
            paired_state = (paired_state[0] + route_grad_dependency, *paired_state[1:])
            return (*paired_state, *token_dispatcher_attr_outputs)

        return route_input, route_probs, paired_state

    def output_shared(
        self,
        *compute_state,
        combined_output: torch.Tensor,
        inference_context=None,
        padding_mask=None,
        persistent_slot: int = 0,
    ):
        """Run output projection and shared experts, then join the routed output."""
        if not compute_state:
            raise RuntimeError("Shortcut output requires a non-empty paired state")
        if self.enable_cudagraph:
            slot = self._require_graph_state().get_slot(persistent_slot)
            compute_state = (compute_state[0].clone(), *compute_state[1:])

        hidden_states = self._output_projection(
            compute_state,
            inference_context=inference_context,
            padding_mask=padding_mask,
        )

        shared_expert_output = self.moe_layer._shortcut_shared_experts(hidden_states)
        if self.enable_cudagraph:
            torch.cuda.current_stream().wait_event(slot.combine_ready_event)
            combined_output = RecordCombineGradReady.apply(
                combined_output, slot.combine_grad_ready_event
            )
            set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
        else:
            # Output projection and shared experts run while the eager combine is in flight.
            combined_output = self._wait_combine(combined_output)

        return self._postprocess(hidden_states, combined_output, shared_expert_output)

    def _postprocess(self, hidden_states, combined_output, shared_expert_output):
        """Join routed/shared output, apply shortcut post-norm, and finish residual/BDA."""
        residual = hidden_states.float() if self.config.fp32_residual_connection else hidden_states
        output = self.moe_layer.mlp.postprocess(combined_output, shared_expert_output)
        output = self.shortcut_post_norm(output)
        output = self.moe_layer._forward_post_mlp((output, None), residual)
        return output[0] if isinstance(output, tuple) else output

    @property
    def cudagraph_manager(self):
        """Return the fused phase's graph manager when capture is enabled."""
        graph_state = getattr(self, '_graph_state', None)
        return None if graph_state is None else graph_state.output_manager

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
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        ready_event: torch.cuda.Event | None = None,
        route_grad_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
        route_grad_ready_event: torch.cuda.Event | None = None,
        backward_dependency: torch.Tensor | None = None,
    ) -> _DispatchOutput:
        """Launch the A2A dispatch on the shortcut side stream."""
        dispatch_stream = self._get_parallel_stream()
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
            return AsyncDispatchToPersistentGradBuffers.apply(
                hidden_states,
                probs,
                backward_dependency,
                moe_layer,
                dispatch_stream,
                route_grad_buffers[0],
                route_grad_buffers[1],
                route_grad_ready_event,
            )
        with torch.cuda.stream(dispatch_stream):
            return moe_layer.dispatch(hidden_states, probs)

    def _wait_dispatch(self, dispatch_output: _DispatchOutput) -> _DispatchOutput:
        """Wait for dispatch and return its outputs on the main stream."""
        dispatch_stream = self._get_parallel_stream()
        torch.cuda.current_stream().wait_stream(dispatch_stream)

        dispatched_input, dispatched_probs = dispatch_output
        main_stream = torch.cuda.current_stream()
        dispatched_input.record_stream(main_stream)
        dispatched_probs.record_stream(main_stream)
        return dispatched_input, dispatched_probs

    def _launch_combine_async(
        self,
        output: torch.Tensor,
        persistent_output_factory=None,
        ready_event: torch.cuda.Event | None = None,
        grad_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """Launch the A2A combine on the shortcut side stream."""
        combine_stream = self._get_parallel_stream()
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
            return combined

        with torch.cuda.stream(combine_stream):
            combined = moe_layer.combine(output)
            if ready_event is not None:
                ready_event.record(combine_stream)

        return combined

    def _wait_combine(self, combined_output: torch.Tensor) -> torch.Tensor:
        """Wait for combine and return its output on the main stream."""
        combine_stream = self._get_parallel_stream()
        torch.cuda.current_stream().wait_stream(combine_stream)

        combined_output.record_stream(torch.cuda.current_stream())
        set_tensor_grad_fn_sequence_sr(combined_output, torch.iinfo(torch.int).max)
        return combined_output

    def _wait_dispatch_and_launch_combine(
        self,
        dispatch_output: _DispatchOutput,
        persistent_output_factory=None,
        ready_event: torch.cuda.Event | None = None,
        grad_ready_event: torch.cuda.Event | None = None,
    ) -> torch.Tensor:
        """Run routed experts after dispatch and launch combine asynchronously."""
        dispatched_input, probs = self._wait_dispatch(dispatch_output)
        output, _ = self.moe_layer.mlp.routed_experts_compute(dispatched_input, probs)
        return self._launch_combine_async(
            output,
            persistent_output_factory=persistent_output_factory,
            ready_event=ready_event,
            grad_ready_event=grad_ready_event,
        )

    def _launch_dispatch(
        self,
        route_outputs,
        persistent_slot: int,
    ) -> tuple[_PairedState, _DispatchOutput]:
        """Normalize route outputs and return paired state plus in-flight dispatch output."""
        if self.enable_cudagraph:
            attr_count = len(self.moe_layer._local_cudagraph_attr_names or ())
            if len(route_outputs) <= attr_count:
                raise RuntimeError(
                    "Shortcut route graph did not return its paired state and registered "
                    "dispatcher attributes"
                )
            paired_state = tuple(route_outputs[:-attr_count] if attr_count else route_outputs)
            token_dispatcher_attr_outputs = (
                tuple(route_outputs[-attr_count:]) if attr_count else ()
            )
            if not paired_state:
                raise RuntimeError("Shortcut dispatch requires a non-empty paired state")

            slot = self._require_graph_state().get_slot(persistent_slot)
            self.moe_layer._restore_token_dispatcher_attrs(token_dispatcher_attr_outputs)
            dispatch_output = self._launch_dispatch_async(
                slot.route_input_buffer.tensor,
                slot.route_probs_buffer.tensor,
                slot.route_ready_event,
                backward_dependency=paired_state[0],
                route_grad_buffers=(
                    slot.route_input_grad_buffer.tensor,
                    slot.route_probs_grad_buffer.tensor,
                ),
                route_grad_ready_event=slot.route_grad_ready_event,
            )
        else:
            route_input, route_probs, paired_state = route_outputs
            if not paired_state:
                raise RuntimeError("Shortcut dispatch requires a non-empty paired state")
            dispatch_output = self._launch_dispatch_async(
                route_input, route_probs, self.route_ready_event
            )

        return paired_state, dispatch_output

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
        """Overlap A2A with paired compute, optionally across two CUDA graph regions."""
        layer_number = self.moe_layer.layer_number - 1
        persistent_slot = 0
        output_slot = None
        if self.enable_cudagraph:
            persistent_slot, output_slot = self._require_graph_state().acquire_slot()

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

        # In graph mode the dispatch stream waits only for router metadata while the paired
        # input graph continues independently.
        paired_state, dispatch_output = self._launch_dispatch(route_outputs, persistent_slot)

        with quant_context_factory(quant_config, layer_number):
            combine_kwargs = {}
            if output_slot is not None:
                combine_kwargs = dict(
                    persistent_output_factory=partial(output_slot.acquire_combined_output),
                    ready_event=output_slot.combine_ready_event,
                    grad_ready_event=output_slot.combine_grad_ready_event,
                )
            combined_output = self._wait_dispatch_and_launch_combine(
                dispatch_output,
                **combine_kwargs,
            )

            return self.output_shared(
                *paired_state,
                combined_output=combined_output,
                inference_context=inference_context,
                padding_mask=padding_mask,
                persistent_slot=persistent_slot,
            )
