# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed activation boundaries for models with differentiable pipeline side inputs."""

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from typing import Any, Callable, Iterable

import torch

from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.utils import make_viewless_tensor

# Gradients keep the forward's packed field order; non-differentiable slots are None.
PipelineGradients = tuple[torch.Tensor | None, ...]


@dataclass(frozen=True)
class PipelinePayloadSpec:
    """Host-only description of one boundary message, prepared before the microbatch runs.

    ``fields`` start with the hidden activation and include absent slots. Gradient
    eligibility is static per microbatch: a differentiable field must receive a
    gradient from its consumer, so a branch that is unused for a microbatch is
    declared non-differentiable in that microbatch's plan.
    """

    fields: tuple[TensorField, ...]
    metadata: tuple[int, ...] = ()
    boundary_id: str = "pipeline"

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id:
            raise ValueError("Pipeline payload requires a stable boundary identity")
        if not isinstance(self.metadata, tuple) or any(type(v) is not int for v in self.metadata):
            raise ValueError("Pipeline payload metadata must be a tuple of host integers")
        if not self.fields or not all(isinstance(f, TensorField) for f in self.fields):
            raise ValueError("Pipeline payload fields must be TensorFields, hidden first")
        self.schema

    @cached_property
    def schema(self) -> TensorSchema:
        """Describe complete fields, including absent slots."""
        return TensorSchema(self.fields)

    @cached_property
    def fingerprint(self) -> int:
        """Return a digest that is stable across processes, for peer plan validation."""
        fields = tuple(
            (f.key, f.shape, str(f.dtype), f.layout, f.differentiable, f.present)
            for f in self.fields
        )
        value = repr((self.boundary_id, self.metadata, fields)).encode("utf-8")
        return int.from_bytes(sha256(value).digest()[:8], "big") >> 1


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


@dataclass(eq=False)
class TensorStatePayload:
    """A hidden tensor plus named cross-layer state crossing one pipeline boundary.

    Set a stage's ``pipeline_payload_factory = TensorStatePayload`` to select typed
    transport in the existing 1F1B schedulers. ``tensors`` hold the present fields
    in ``spec`` order and may share storage with the local autograd graph.
    """

    tensors: tuple[torch.Tensor, ...]
    spec: PipelinePayloadSpec
    # from_state() wraps each differentiable root in a new viewless tensor object,
    # which the transport may shrink after the send without touching caller aliases.
    _owns_roots: bool = field(default=False, repr=False)
    _roots: tuple[torch.Tensor | None, ...] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_state(cls, hidden, state, schema: TensorSchema, *, boundary_id="state", metadata=()):
        """Publish explicit producer edges; never detach or copy shared tensors."""
        hidden_field = TensorField("__hidden__", tuple(hidden.shape), hidden.dtype, "strided", True)
        spec = PipelinePayloadSpec((hidden_field, *schema.fields), tuple(metadata), boundary_id)
        tensors = tuple(
            (
                make_viewless_tensor(tensor.view_as(tensor), requires_grad=True, keep_graph=True)
                if f.differentiable and tensor.requires_grad
                else tensor
            )
            for f, tensor in zip(spec.schema.packed_fields, (hidden, *schema.pack(state)))
        )
        return cls(tensors, spec, True)

    @property
    def device(self) -> torch.device:
        """Return the activation device, including for auxiliary loss scaling."""
        roots = self.tensors or self._roots
        return next(t.device for t in roots if t is not None)

    def restore(self):
        """Return the hidden tensor and a fresh state mapping after the receive completes."""
        self.spec.schema.validate(self.tensors)
        if self.spec.fields[0].key != "__hidden__":
            raise ValueError("Tensor state payload must start with its declared hidden field")
        if any(t.device != self.tensors[0].device for t in self.tensors):
            raise ValueError("Tensor state payload fields must share a device")
        return self.tensors[0], TensorSchema(self.spec.fields[1:]).unpack(self.tensors[1:])

    def backward_roots(self) -> tuple[torch.Tensor | None, ...]:
        """Present tensors before the send, or the retained differentiable roots after it."""
        return self.tensors if self._roots is None else self._roots

    def release_output(self) -> None:
        """Keep only differentiable roots for backward, shrinking the ones this payload owns.

        Called after the send is posted: detached wire buffers keep the storage alive
        until the transfer completes, and saved activations keep their own references.
        """
        if self._roots is not None:
            return
        self._roots = tuple(
            tensor if f.differentiable and tensor.requires_grad else None
            for f, tensor in zip(self.spec.schema.packed_fields, self.tensors)
        )
        if self._owns_roots:
            for tensor in self._roots:
                if tensor is not None:
                    tensor.data = tensor.new_empty((1,))
        self.tensors = ()


def backward_pipeline_payload(
    input_payload: TensorStatePayload | None,
    output: TensorStatePayload | torch.Tensor,
    output_grad: PipelineGradients | None,
    grad_scale_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> PipelineGradients | None:
    """Backpropagate all differentiable outputs in one traversal of the local graph.

    A received field may also be sent onward (a relay). Autograd then adds its local
    consumer contribution and its downstream gradient exactly once. The terminal
    scalar loss uses the normal loss scaler; received activation gradients are
    already scaled.
    """
    # An identity stage may return its received payload directly; take its leaves
    # before the output backward below consumes the shared record.
    inputs = () if input_payload is None else input_payload.backward_roots()
    if isinstance(output, TensorStatePayload):
        fields = output.spec.schema.packed_fields
        roots = output.backward_roots()
        if output_grad is None or len(output_grad) != len(fields) or len(roots) != len(fields):
            raise ValueError("Pipeline gradient slots must match an unconsumed forward payload")
        tensors, grads = [], []
        for root, f, grad in zip(roots, fields, output_grad):
            if grad is None:
                continue
            if not f.differentiable or tuple(grad.shape) != f.shape or grad.dtype != f.dtype:
                raise ValueError(f"Invalid gradient for pipeline field {f.key}")
            # Receivers allocate leaves from static eligibility; a frozen sender has no edge.
            if root is not None and root.requires_grad:
                tensors.append(root)
                grads.append(grad)
        if tensors:
            # As in schedules.custom_backward, bypass only the root-shape check:
            # owned roots have had their .data released after the send.
            torch.autograd.Variable._execution_engine.run_backward(
                tensors=tuple(tensors),
                grad_tensors=tuple(grads),
                keep_graph=False,
                create_graph=False,
                inputs=(),
                allow_unreachable=True,
                accumulate_grad=True,
            )
        output._roots = ()
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
        tensor.grad if tensor is not None and f.differentiable else None
        for tensor, f in zip(inputs, input_payload.spec.schema.packed_fields)
    )
