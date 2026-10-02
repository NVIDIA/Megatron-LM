# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Typed activation boundaries for models with differentiable pipeline side inputs."""

from collections import deque
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Iterable

import torch

from megatron.core.transformer.state_boundary import TensorField, TensorSchema
from megatron.core.utils import make_viewless_tensor


@dataclass(frozen=True)
class PipelineTensorSpec:
    """A pipeline display name bound to the common tensor contract.

    Shape, dtype, layout, presence and gradient eligibility have one source of
    truth: ``field``. ``requires_grad`` is an eligibility alias for transport;
    the producer tensor's actual autograd state is evaluated during backward.
    """

    name: str
    field: TensorField

    @property
    def key(self) -> str:
        """Stable field identity shared by all boundaries."""
        return self.field.key

    @property
    def shape(self) -> tuple[int, ...]:
        """Declared local tensor shape."""
        return self.field.shape

    @property
    def dtype(self) -> torch.dtype:
        """Declared tensor dtype."""
        return self.field.dtype

    @property
    def requires_grad(self) -> bool:
        """Static eligibility to receive a backward contribution."""
        return self.field.differentiable

    @property
    def layout(self) -> str:
        """Owner-declared logical layout."""
        return self.field.layout

    @property
    def present(self) -> bool:
        """Whether this field occupies a wire tensor slot."""
        return self.field.present

    @classmethod
    def from_field(cls, field: TensorField) -> "PipelineTensorSpec":
        """Use the canonical field key as the default display name."""
        return cls(field.key, field)


@dataclass(frozen=True)
class PipelinePayloadSpec:
    """Host-only description prepared before a microbatch enters the pipeline."""

    tensor_specs: tuple[PipelineTensorSpec, ...]
    metadata: tuple[int, ...]
    boundary_id: str = "pipeline"

    def __post_init__(self) -> None:
        if not isinstance(self.boundary_id, str) or not self.boundary_id:
            raise ValueError("Pipeline payload requires a stable boundary identity")
        if not isinstance(self.metadata, tuple) or any(type(v) is not int for v in self.metadata):
            raise ValueError("Pipeline payload metadata must be a tuple of host integers")
        self.schema

    @property
    def schema(self) -> TensorSchema:
        """Describe complete fields, including absent slots."""
        return TensorSchema(tuple(spec.field for spec in self.tensor_specs))

    @property
    def fingerprint(self) -> int:
        """Validate the boundary identity, metadata and complete field schema together."""
        value = (self.boundary_id, self.metadata, self.schema.fingerprint)
        return int.from_bytes(sha256(repr(value).encode("utf-8")).digest()[:8], "big") >> 1


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
    """Keep original autograd outputs alive until their backward has been consumed."""

    specs: tuple[PipelineTensorSpec, ...]
    device: torch.device
    outputs: tuple[torch.Tensor | None, ...]


class PipelinePayload:
    """A model-owned snapshot whose tensors may share storage with the local graph.

    After posting a send, the transport transfers the original outputs to a strong
    backward record. Only explicitly owned viewless roots may be pseudo-deallocated.
    Integer fields and floating fields without gradient eligibility are excluded from backward.
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
        return ()

    @property
    def boundary_id(self) -> str:
        """Stable boundary name supplied by the model's existing placement."""
        return "pipeline"

    @property
    def descriptor(self) -> PipelinePayloadSpec:
        """Snapshot the concrete schema before its outputs move into the backward record."""
        return PipelinePayloadSpec(self.tensor_specs, self.metadata, self.boundary_id)

    @property
    def device(self) -> torch.device:
        """Return the activation device, including for auxiliary loss scaling."""
        if self._backward_state is not None:
            return self._backward_state.device
        return self.tensors[0].device

    @property
    def releasable_tensor_indices(self) -> tuple[int, ...]:
        """Return exclusively owned tensor objects whose storage can be released after send."""
        return ()

    def release_output(self) -> None:
        """Retain autograd roots and release only explicitly owned tensor objects.

        Work handles keep detached send buffers alive until communication completes.
        Shared side fields are never shrunk; saved tensors retain storage as needed.
        """
        if self._backward_state is not None:
            return
        specs = self.tensor_specs
        packed_specs = tuple(spec for spec in specs if spec.present)
        roots = tuple(
            tensor if spec.requires_grad and tensor.requires_grad else None
            for spec, tensor in zip(packed_specs, self.tensors)
        )
        state = _PipelineBackwardState(specs, self.device, roots)
        for index in self.releasable_tensor_indices:
            tensor = state.outputs[index]
            if tensor is None:
                continue
            if tensor._base is not None:
                raise ValueError("Releasable pipeline outputs must be viewless owned tensors")
            tensor.data = tensor.new_empty((1,))
        # Model payloads can be frozen dataclasses: their forward schema remains
        # immutable, but ownership is explicitly transferred to backward here.
        object.__setattr__(self, "_backward_state", state)
        object.__setattr__(self, "tensors", ())


@dataclass(frozen=True)
class TensorStatePayload(PipelinePayload):
    """Default payload for a hidden tensor plus named cross-layer tensor state.

    Set a stage's ``pipeline_payload_factory = TensorStatePayload`` to select
    typed transport in the existing 1F1B schedulers. Descriptors, including
    gradient eligibility, come from the model's state contract rather than a
    particular forward's requires_grad flags.
    """

    tensors: tuple[torch.Tensor, ...]
    spec: PipelinePayloadSpec
    _owns_hidden: bool = field(default=False, repr=False, compare=False)

    @property
    def tensor_specs(self):
        return self.spec.tensor_specs

    @property
    def metadata(self):
        return self.spec.metadata

    @property
    def boundary_id(self):
        return self.spec.boundary_id

    @property
    def releasable_tensor_indices(self):
        """Only the dedicated hidden root is owned; shared state retains its storage."""
        return (0,) if self._owns_hidden else ()

    @classmethod
    def from_state(cls, hidden, state, schema: TensorSchema, *, boundary_id="state"):
        """Publish explicit producer edges; do not detach or clone shared tensors."""
        hidden_field = TensorField("__hidden__", tuple(hidden.shape), hidden.dtype, "strided", True)
        fields = (hidden_field, *schema.fields)
        specs = tuple(PipelineTensorSpec.from_field(field) for field in fields)
        # Force a new viewless tensor object without copying storage. Shrinking
        # this root cannot mutate a caller's tensor, saved activation or relay.
        owned_hidden = make_viewless_tensor(
            hidden.view_as(hidden), requires_grad=hidden.requires_grad, keep_graph=True
        )
        return cls(
            (owned_hidden, *schema.pack(state)), PipelinePayloadSpec(specs, (), boundary_id), True
        )

    def restore(self):
        """Restore a fresh per-microbatch state mapping after the receive is complete."""
        self.spec.schema.validate(self.tensors)
        if not self.tensor_specs or self.tensor_specs[0].key != "__hidden__":
            raise ValueError("Tensor state payload must start with its declared hidden field")
        if any(t.device != self.tensors[0].device for t in self.tensors):
            raise ValueError("Tensor state payload fields must share a device")
        schema = TensorSchema(tuple(spec.field for spec in self.tensor_specs[1:]))
        return self.tensors[0], schema.unpack(self.tensors[1:])


# Gradients keep the forward field order; non-gradient slots are None.
PipelineGradients = tuple[torch.Tensor | None, ...]


@dataclass
class PipelineGradientMessage:
    """Fixed gradient buffers plus a received activity bitmap and execution identity.

    The schedule must wait its receive Work before resolving this message. A
    numeric zero is active; a missing gradient stays None and is never a root.
    """

    tensors: PipelineGradients
    control: torch.Tensor
    identity: tuple[int, ...]
    grad_tensor_indices: tuple[int, ...]
    wait_control: Callable[[], None] | None = None

    def resolve(self) -> PipelineGradients:
        """Validate the control header and restore None without inspecting gradient values."""
        if self.wait_control is not None:
            self.wait_control()
            self.wait_control = None
        values = self.control.tolist()
        count = len(self.identity)
        if tuple(values[:count]) != self.identity:
            raise ValueError("Pipeline backward microbatch/chunk/boundary mismatch")
        active = values[count:]
        if len(active) != len(self.grad_tensor_indices) or any(
            value not in (0, 1) for value in active
        ):
            raise ValueError("Invalid pipeline gradient activity bitmap")
        tensors = list(self.tensors)
        for index, present in zip(self.grad_tensor_indices, active):
            if not present:
                tensors[index] = None
        return tuple(tensors)


# Factories and tensor_specs may inspect only shapes, dtypes and host metadata:
# with overlap, receive buffers are populated after payload construction. The
# schedule waits the receive handle before model execution or state restoration.
PipelinePayloadFactory = Callable[[tuple[torch.Tensor, ...], PipelinePayloadSpec], PipelinePayload]


def backward_pipeline_payload(
    input_payload: PipelinePayload | None,
    output: PipelinePayload | torch.Tensor,
    output_grad: PipelineGradients | PipelineGradientMessage | None,
    grad_scale_func: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> PipelineGradients | None:
    """Backpropagate all differentiable outputs in one traversal of the local graph.

    A shared input may also be an output (a relay). Autograd then adds its local
    consumer contribution and its downstream gradient exactly once. The terminal
    scalar loss uses the normal loss scaler; received activation gradients are
    already scaled.
    """
    input_roots, input_specs = (), ()
    if input_payload is not None:
        # An identity stage may return its received payload directly. Its send
        # already moved the leaves into the backward record, so use that snapshot.
        incoming = input_payload._backward_state
        specs = input_payload.tensor_specs if incoming is None else incoming.specs
        input_specs = tuple(spec for spec in specs if spec.present)
        input_roots = input_payload.tensors if incoming is None else incoming.outputs
        for tensor, spec in zip(input_roots, input_specs):
            if tensor is not None and spec.requires_grad and tensor.requires_grad:
                tensor.retain_grad()

    if isinstance(output_grad, PipelineGradientMessage):
        output_grad = output_grad.resolve()

    if isinstance(output, PipelinePayload):
        state = output._backward_state
        specs = tuple(
            s for s in (output.tensor_specs if state is None else state.specs) if s.present
        )
        roots = output.tensors if state is None else state.outputs
        if output_grad is None or len(output_grad) != len(specs):
            raise ValueError("Pipeline gradient slots must match the forward payload")
        if len(roots) != len(specs):
            raise RuntimeError("Pipeline payload backward has already been consumed")
        outputs, grads, positions = [], [], {}
        for root, spec, grad in zip(roots, specs, output_grad):
            if spec.requires_grad and grad is not None:
                if tuple(grad.shape) != spec.shape or grad.dtype != spec.dtype:
                    raise ValueError(f"Invalid gradient for pipeline field {spec.name}")
                # Receivers allocate leaves using static eligibility. A frozen
                # sender has no local edge even when the receiver used the value.
                if root is None or not root.requires_grad:
                    continue
                # Two declared outputs may be the very same Tensor. Sum their
                # contributions by identity, never by storage/data_ptr aliases.
                key = id(root)
                if key in positions:
                    grads[positions[key]] = grads[positions[key]] + grad
                else:
                    positions[key] = len(outputs)
                    outputs.append(root)
                    grads.append(grad)
            elif grad is not None:
                raise ValueError(f"Unexpected gradient for pipeline field {spec.name}")
        if outputs:
            # As in schedules.custom_backward, bypass only the root-shape check:
            # an owned root may have had its .data released after the send.
            torch.autograd.Variable._execution_engine.run_backward(
                tensors=tuple(outputs),
                grad_tensors=tuple(grads),
                keep_graph=False,
                create_graph=False,
                inputs=(),
                allow_unreachable=True,
                accumulate_grad=True,
            )
        if state is not None:
            state.outputs = ()
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
        tensor.grad if tensor is not None and spec.requires_grad else None
        for tensor, spec in zip(input_roots, input_specs)
    )
