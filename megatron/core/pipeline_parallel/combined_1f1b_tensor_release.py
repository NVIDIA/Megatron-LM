# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scheduled tensor release for the combined 1F1B pipeline schedule.

Only tensors produced and consumed by a ``ScheduleNode`` participate.  Each
model-chunk plan records the stream that produced a concrete tensor, consumes
that binding at the next real node, and falls back to allocator ``record_stream``
when an input came from outside the plan.

A cross-stream consumer holds the tensor until a later node from the same plan
acquires the owner stream.  That acquire already waits on the plan event, so the
release state can drop the reference (or empty forward storage) without adding a new
dependency to the overlap schedule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Optional

import torch


class ReleaseAction(Enum):
    """Action performed after a tensor is safe to release."""

    EMPTY_STORAGE = auto()
    DROP_REFERENCE = auto()


@dataclass(slots=True)
class TensorOwner:
    """Strong, plan-local binding from one concrete tensor to its producer stream."""

    tensor: torch.Tensor
    stream: torch.cuda.Stream
    producer_node: str


@dataclass(slots=True)
class DeferredRelease:
    """Strong reference held until the owner stream has acquired the consumer."""

    tensor: torch.Tensor
    action: ReleaseAction


def _iter_unique_cuda_tensors(value: Any) -> Iterable[torch.Tensor]:
    """Yield each CUDA tensor object once from one flat schedule edge.

    Schedule edges may be a tensor or a flat tuple containing tensors and
    ``None``.  Repeated references to the same tensor represent one ownership
    transition, so object identity is used to suppress duplicate work.
    """

    values = (value,) if isinstance(value, torch.Tensor) else value
    if not values:
        return
    seen = set()
    for tensor in values:
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_id = id(tensor)
        if not tensor.is_cuda or tensor_id in seen:
            continue
        seen.add(tensor_id)
        yield tensor


class Combined1F1BTensorRelease:
    """Release selected tensors within one combined 1F1B model-chunk plan.

    Ownership follows concrete tensor objects rather than schedule-node input
    slots.  Bindings are strong and single-consumer: a real node consumes each
    input binding once, while ``NoopScheduleNode`` naturally carries the same
    object and binding to the next real node.  Storage aliases among tensors
    participating in one real-node transition are unsupported: combined EFVPA
    nodes must return fresh storage so tensor-object ownership remains unambiguous.
    """

    def __init__(self):
        self._owners: dict[int, TensorOwner] = {}
        # A pending release is keyed by its producer stream.  When that stream
        # next acquires the plan event, all of its entries become safe to release.
        self._pending: defaultdict[torch.cuda.Stream, list[DeferredRelease]] = defaultdict(list)

    def consume_inputs_and_publish_outputs(
        self,
        consumed: Any,
        produced: Any,
        *,
        stream: torch.cuda.Stream,
        node: str,
        release_consumed: bool,
    ) -> None:
        """Consume input bindings and publish outputs from one forward node."""

        action = ReleaseAction.EMPTY_STORAGE if release_consumed else None
        self._consume_and_publish(consumed, produced, action, stream, node)

    def consume_forward_outputs(self, forward_outputs: Any) -> None:
        """End bindings for recomputed forward outputs consumed by autograd.

        These tensors remain live through the autograd graph, so consuming the
        binding must not resize their storage or register another stream.
        """

        self._consume(forward_outputs, None, None)

    def consume_output_grads_and_publish_input_grads(
        self, output_grads: Any, input_grads: Any, *, stream: torch.cuda.Stream, node: str
    ) -> None:
        """Release output grads and publish input grads produced by one backward node."""

        self._consume_and_publish(
            output_grads, input_grads, ReleaseAction.DROP_REFERENCE, stream, node
        )

    def export(self, value: Any) -> None:
        """Move plan outputs outside scheduled release without changing storage."""

        for tensor in _iter_unique_cuda_tensors(value):
            self._export_tensor(tensor)

    def drain(self, stream: torch.cuda.Stream) -> None:
        """Release entries after ``stream`` has acquired the plan event.

        The event wait orders the producer stream after every consumer that
        deferred a release to it.  Removing the strong references or making
        storage allocator-visible is therefore safe without another event.
        """

        for entry in self._pending.pop(stream, ()):
            self._apply_action(entry.tensor, entry.action)

    def finalize_phase(self, event: torch.cuda.Event, phase: str, outputs: Any = ()) -> None:
        """Hand pending tensors back, export phase outputs, and audit bindings.

        A phase may end before some producer stream is naturally acquired again.
        Enqueueing the final plan-event wait on every such stream completes the
        same hand-back that a later node acquire would have performed.
        """

        for stream in tuple(self._pending):
            event.wait(stream)
            with torch.cuda.stream(stream):
                self.drain(stream)
        if self._pending:
            pending_count = sum(len(entries) for entries in self._pending.values())
            raise RuntimeError(f"Phase {phase!r} finalized with {pending_count} deferred tensors")

        self.export(outputs)
        if self._owners:
            producers = sorted({owner.producer_node for owner in self._owners.values()})
            raise RuntimeError(
                f"Phase {phase!r} finalized with {len(self._owners)} unconsumed tensor-owner "
                f"bindings from producers {producers}"
            )

    def _consume_and_publish(
        self,
        consumed: Any,
        produced: Any,
        action: Optional[ReleaseAction],
        stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        """Consume one edge and publish the next, rejecting storage aliases."""

        consumed_tensors = tuple(_iter_unique_cuda_tensors(consumed))
        produced_tensors = tuple(_iter_unique_cuda_tensors(produced))

        # Object-id ownership cannot safely describe two tensors sharing an
        # allocation, especially when EMPTY_STORAGE releases the whole storage.
        # Validate the complete edge before mutating any binding or storage so a
        # rejected transition leaves the release state and its tensors unchanged.
        edge_tensors = consumed_tensors + produced_tensors
        for tensor_index, tensor in enumerate(edge_tensors):
            for prior_index in range(tensor_index):
                if torch._C._is_alias_of(edge_tensors[prior_index], tensor):
                    raise RuntimeError(
                        f"Node {node!r} passed multiple tensor objects sharing one storage; "
                        "combined 1F1B tensor release does not support storage aliases"
                    )

        for tensor in consumed_tensors:
            self._consume_tensor(tensor, action, stream)

        for tensor in produced_tensors:
            self._publish_tensor(tensor, stream, node)

    def _consume(
        self,
        value: Any,
        action: Optional[ReleaseAction],
        consumer_stream: Optional[torch.cuda.Stream],
    ) -> None:
        """Consume every CUDA tensor binding in one flat schedule edge."""

        for tensor in _iter_unique_cuda_tensors(value):
            self._consume_tensor(tensor, action, consumer_stream)

    def _consume_tensor(
        self,
        tensor: torch.Tensor,
        action: Optional[ReleaseAction],
        consumer_stream: Optional[torch.cuda.Stream],
    ) -> None:
        """Consume one binding and release it according to its producer stream.

        A missing owner means the tensor entered from outside the managed node
        chain.  In that case the allocator's conservative ``record_stream`` path
        remains responsible for preventing premature reuse.
        """

        owner = self._take_owner(tensor)
        if owner is None:
            if action is not None:
                assert consumer_stream is not None
                tensor.record_stream(consumer_stream)
                self._apply_action(tensor, action)
            return

        # action=None removes the tensor from scheduled release without freeing
        # storage; autograd or the caller still controls when it becomes dead.
        if action is None:
            return
        assert consumer_stream is not None
        self._release_tensor(tensor, action, owner, consumer_stream)

    def _take_owner(self, tensor: torch.Tensor) -> Optional[TensorOwner]:
        """Remove and return the binding only when it owns this exact object."""

        tensor_id = id(tensor)
        owner = self._owners.get(tensor_id)
        if owner is None or owner.tensor is not tensor:
            return None
        self._owners.pop(tensor_id)
        return owner

    def _publish_tensor(self, tensor: torch.Tensor, stream: torch.cuda.Stream, node: str) -> None:
        """Bind a newly produced tensor object to its CUDA producer stream."""

        tensor_id = id(tensor)
        existing = self._owners.get(tensor_id)
        if existing is not None and existing.tensor is tensor:
            raise RuntimeError(
                f"Tensor already has an unconsumed owner binding from "
                f"{existing.producer_node!r}; producer={node!r}, "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        self._owners[tensor_id] = TensorOwner(tensor=tensor, stream=stream, producer_node=node)

    def _export_tensor(self, tensor: torch.Tensor) -> None:
        """Remove scheduled-release ownership without changing tensor storage."""

        self._take_owner(tensor)

    def _release_tensor(
        self,
        tensor: torch.Tensor,
        action: ReleaseAction,
        owner: TensorOwner,
        consumer_stream: torch.cuda.Stream,
    ) -> None:
        """Release now on the owner stream or defer until it is acquired again."""

        if owner.stream == consumer_stream:
            self._apply_action(tensor, action)
            return

        # The consumer records the shared plan event after its work.  Keep a
        # strong reference until the producer stream later waits on that event.
        self._pending[owner.stream].append(DeferredRelease(tensor=tensor, action=action))

    @staticmethod
    def _apply_action(tensor: torch.Tensor, action: ReleaseAction) -> None:
        """Make forward storage reusable or drop the held gradient reference."""

        if action is ReleaseAction.EMPTY_STORAGE:
            tensor.untyped_storage().resize_(0)
        elif action is not ReleaseAction.DROP_REFERENCE:
            raise AssertionError(f"Unknown scheduled tensor release action: {action}")
