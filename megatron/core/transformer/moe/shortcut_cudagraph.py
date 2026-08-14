# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import weakref

import torch


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
        self._has_weak_storage = False

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
            # This allocation is intentionally delayed until the first graph-recording forward.
            # Give each persistent allocation its own profiler range so its lifetime and size are
            # attributable in CUDA memory profiles instead of appearing as anonymous graph setup.
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
                is_from_global_mempool = getattr(
                    self._tensor, "is_from_global_mempool", False
                )
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

    def release_strong_reference(self) -> None:
        """Replace graph-pinned storage ownership with a raw-pointer tensor wrapper."""
        if self._tensor is None or self._has_weak_storage:
            return
        if not getattr(self._tensor, "is_from_global_mempool", False):
            return

        from megatron.core.transformer.cuda_graphs import make_weakref

        weak_tensor = make_weakref(self._tensor, inplace=False)
        if weak_tensor is not self._tensor:
            self._tensor = weak_tensor
            self._has_weak_storage = True

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
        # Dispatch reads detached persistent buffers, so it otherwise has no autograd edge
        # to the route/input graph. This dummy dependency holds that graph's backward until
        # dispatch backward has published the route gradients and recorded the ready event.
        ctx.dispatch_stream = dispatch_stream
        ctx.route_input_grad_buffer = route_input_grad_buffer
        ctx.route_probs_grad_buffer = route_probs_grad_buffer
        ctx.route_grad_ready_event = route_grad_ready_event

        # Retain the real dispatch graph privately so backward can run on the communication stream.
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
        # Distinct aliases leave the saved tensors' private dispatch autograd history intact.
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
        # The dependency receives no gradient, but returning it only after the private
        # dispatch backward and event record makes the ordering an autograd invariant.
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

        # Build a private combine graph so backward can run on the communication stream.
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
        # This scalar is only an autograd dependency. Its value must not alter paired compute.
        return route_input.new_zeros(())

    @staticmethod
    def backward(ctx, grad_output):
        slot = ctx.slot_ref()
        assert slot is not None
        # During backward capture this becomes an external wait node. Attention/SSM backward is
        # ordered before this node; router/preprocess backward consumes the buffers after the wait.
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
