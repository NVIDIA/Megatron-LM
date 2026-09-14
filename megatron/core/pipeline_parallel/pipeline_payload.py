# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed activation boundaries for models with differentiable pipeline side inputs."""

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class PipelineTensorSpec:
    """Concrete tensor descriptor, snapshotted for one boundary and microbatch."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool


@dataclass(frozen=True)
class PipelinePayloadSpec:
    """Host-only description prepared before a microbatch enters the pipeline."""

    tensor_specs: tuple[PipelineTensorSpec, ...]
    metadata: tuple[int, ...]


@dataclass(frozen=True)
class PipelinePayloadPlan:
    """One chunk's incoming/outgoing descriptions in forward microbatch order.

    Every rank prepares the same description for a shared boundary. None denotes
    a terminal boundary, not an unknown shape. Plans belong to one schedule
    invocation and contain only host descriptors, not activation tensors.
    """

    incoming: tuple[PipelinePayloadSpec | None, ...]
    outgoing: tuple[PipelinePayloadSpec | None, ...]

    def __post_init__(self) -> None:
        if len(self.incoming) != len(self.outgoing):
            raise ValueError("Pipeline input/output plans must cover the same microbatches")


class PipelineDataIterator:
    """Replay prepared input batches once, releasing each queue entry at consumption.

    A callable loader may fetch batches lazily when shapes are known without
    staging data (e.g. fixed SBHD). Otherwise pop staged batches in forward order.
    The original iterator's checkpoint/rerun machinery remains authoritative.
    Construct a fresh wrapper and plan for each invocation, including reruns.
    """

    def __init__(
        self, batches: Iterable[Any] | Callable[[], Any], plan: PipelinePayloadPlan
    ) -> None:
        self._batches = deque() if callable(batches) else deque(batches)
        if not callable(batches) and len(self._batches) != len(plan.incoming):
            raise ValueError("Prepared pipeline batches must match the payload plan length")
        self._load_batch = batches if callable(batches) else self._batches.popleft
        self._remaining = len(plan.incoming)
        self.pipeline_payload_plan = plan

    def __iter__(self) -> "PipelineDataIterator":
        return self

    def __next__(self) -> Any:
        if self._remaining == 0:
            raise StopIteration
        batch = self._load_batch()
        self._remaining -= 1
        return batch


@dataclass
class _PipelineBackwardState:
    """Autograd roots and wire shapes without owning the outgoing tensor wrappers."""

    specs: tuple[PipelineTensorSpec, ...]
    device: torch.device
    edges: tuple[torch.autograd.graph.GradientEdge | None, ...]


class PipelinePayload:
    """A model-owned snapshot whose tensors may share storage with the local graph.

    After posting a send, the transport may release the outgoing tensor references
    and retain only their gradient edges. Autograd still owns tensors saved by local
    consumers, including received leaves relayed downstream. Integer metadata and
    floating fields without a gradient slot are excluded from distributed backward.
    """

    tensors: tuple[torch.Tensor, ...]
    _backward_state: _PipelineBackwardState | None = None

    @property
    def tensor_specs(self) -> tuple[PipelineTensorSpec, ...]:
        """Describe the actual outgoing tensors in deterministic wire order."""
        raise NotImplementedError

    @property
    def metadata(self) -> tuple[int, ...]:
        """Return model-owned host integers for reconstructing the input payload."""
        return (0, 0)

    @property
    def device(self) -> torch.device:
        """Return the activation device, including for auxiliary loss scaling."""
        if self._backward_state is not None:
            return self._backward_state.device
        return self.tensors[0].device

    def release_output(self) -> None:
        """Release an exported payload's tensor references after its send is posted.

        GradientEdge preserves hooks and joint autograd without keeping otherwise
        unused outputs alive. No tensor data or shared storage is modified. The P2P
        Work owns detached send buffers until completion, and autograd retains any
        values needed by local backward. A released payload is only valid for backward;
        input payloads remain intact until their gradients have been collected.
        """
        if self._backward_state is not None:
            return
        specs = self.tensor_specs
        with torch.enable_grad():
            # A C++ view node owns the upstream graph even when its producer is a
            # Python autograd.Function. A direct GradientEdge to that Python node
            # can outlive its C++ owner after tensor release (PyTorch 2.6). The view
            # adds no allocation and saves no activation storage.
            edges = tuple(
                (
                    torch.autograd.graph.get_gradient_edge(tensor.view_as(tensor))
                    if spec.requires_grad
                    else None
                )
                for tensor, spec in zip(self.tensors, specs)
            )
        state = _PipelineBackwardState(specs, self.device, edges)
        # Model payloads can be frozen dataclasses: their forward schema remains
        # immutable, but ownership is explicitly transferred to backward here.
        object.__setattr__(self, "_backward_state", state)
        object.__setattr__(self, "tensors", ())


# Gradients keep the forward field order; non-gradient slots are None.
PipelineGradients = tuple[torch.Tensor | None, ...]
# Factories and tensor_specs may inspect only shapes, dtypes and host metadata:
# with overlap, receive buffers are populated after payload construction. The
# schedule waits the receive handle before model execution or state restoration.
PipelinePayloadFactory = Callable[[tuple[torch.Tensor, ...], tuple[int, ...]], PipelinePayload]


def backward_pipeline_payload(
    input_payload: PipelinePayload | None,
    output: PipelinePayload | torch.Tensor,
    output_grad: PipelineGradients | None,
    grad_scale_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> PipelineGradients | None:
    """Backpropagate all differentiable outputs in one traversal of the local graph.

    A shared input may also be an output (a relay). Autograd then adds its local
    consumer contribution and its downstream gradient exactly once. The terminal
    scalar loss uses the normal loss scaler; received activation gradients are
    already scaled.
    """
    input_specs = () if input_payload is None else input_payload.tensor_specs
    if input_payload is not None:
        for tensor, spec in zip(input_payload.tensors, input_specs):
            if spec.requires_grad:
                tensor.retain_grad()

    if isinstance(output, PipelinePayload):
        state = output._backward_state
        specs = output.tensor_specs if state is None else state.specs
        roots = output.tensors if state is None else state.edges
        if output_grad is None or len(output_grad) != len(specs):
            raise ValueError("Pipeline gradient slots must match the forward payload")
        if len(roots) != len(specs):
            raise RuntimeError("Pipeline payload backward has already been consumed")
        outputs, grads = [], []
        for root, spec, grad in zip(roots, specs, output_grad):
            if spec.requires_grad:
                outputs.append(root)
                grads.append(
                    torch.zeros(spec.shape, dtype=spec.dtype, device=output.device)
                    if grad is None
                    else grad
                )
            elif grad is not None:
                raise ValueError(f"Unexpected gradient for pipeline field {spec.name}")
        if outputs:
            torch.autograd.backward(outputs, grad_tensors=grads)
        if state is not None:
            state.edges = ()
    else:
        if output_grad is not None or output.numel() != 1:
            raise ValueError("The last typed pipeline stage requires a scalar loss")
        if grad_scale_func is not None:
            output = grad_scale_func(output)
        if output.requires_grad:
            torch.autograd.backward(output)

    if input_payload is None:
        return None
    return tuple(
        tensor.grad if spec.requires_grad else None
        for tensor, spec in zip(input_payload.tensors, input_specs)
    )
