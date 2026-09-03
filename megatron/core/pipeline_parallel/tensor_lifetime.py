# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-local lifetime management for tensors crossing CUDA streams.

Only tensors produced and consumed by a ``ScheduleNode`` participate.  Each
model-chunk plan records the stream that produced a concrete tensor, consumes
that binding at the next real node, and falls back to allocator ``record_stream``
retirement when an input came from outside the plan.

A cross-stream consumer holds the tensor until a later node from the same plan
acquires the owner stream.  That acquire already waits on the plan event, so the
manager can drop the reference (or empty forward storage) without adding a new
dependency to the overlap schedule.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Optional

import torch


class ReleaseAction(Enum):
    """Action performed after a tensor is safe to retire."""

    EMPTY_STORAGE = auto()
    DROP_REFERENCE = auto()


StreamKey = tuple[torch.device, int]


@dataclass(slots=True)
class TensorOwner:
    """Strong, plan-local binding from one concrete tensor to its producer stream."""

    tensor: torch.Tensor
    stream: torch.cuda.Stream
    stream_key: StreamKey
    producer_node: str


@dataclass(slots=True)
class DeferredRelease:
    """Strong reference held until the owner stream has acquired the consumer."""

    tensor: torch.Tensor
    action: ReleaseAction
    owner_stream: torch.cuda.Stream


def _stream_key(stream: torch.cuda.Stream) -> StreamKey:
    return stream.device, int(stream.cuda_stream)


def _iter_unique_cuda_tensors(values: Iterable[Any]) -> Iterable[torch.Tensor]:
    """Yield unique CUDA tensors from a flat, structured schedule edge."""

    seen = set()
    for tensor in values:
        if not isinstance(tensor, torch.Tensor):
            continue
        tensor_id = id(tensor)
        if not tensor.is_cuda or tensor_id in seen:
            continue
        seen.add(tensor_id)
        yield tensor


class ScheduleTensorLifetimeManager:
    """Retire selected tensors within one model-chunk schedule plan.

    Ownership follows concrete tensor objects rather than schedule-node input
    slots.  Bindings are strong and single-consumer: a real node consumes each
    input binding once, while ``NoopScheduleNode`` naturally carries the same
    object and binding to the next real node.  The schedule's existing
    ``free_input`` contract guarantees that retired inputs do not alias outputs.
    """

    def __init__(self):
        self._owners: dict[int, TensorOwner] = {}
        self._pending: dict[StreamKey, list[DeferredRelease]] = {}
        self._pending_count = 0
        self.stats = {
            "published": 0,
            "consumed": 0,
            "exported": 0,
            "same_stream": 0,
            "deferred": 0,
            "record_stream_fallback": 0,
            "released_natural": 0,
            "released_terminal": 0,
            "max_pending": 0,
            "pending_before_finalize": 0,
            "pending_at_finalize": 0,
            "owners_before_finalize": 0,
            "owners_at_finalize": 0,
        }
        self.last_release_node: Optional[str] = None

    @property
    def owners(self) -> list[TensorOwner]:
        """Return live tensor-owner bindings, primarily for diagnostics and tests."""

        return list(self._owners.values())

    @property
    def pending(self) -> list[DeferredRelease]:
        """Return all deferred entries, primarily for diagnostics and tests."""

        return [entry for entries in self._pending.values() for entry in entries]

    def consume_inputs_and_publish_outputs(
        self,
        consumed: Any,
        produced: Any,
        *,
        stream: torch.cuda.Stream,
        node: str,
        retire_consumed: bool,
    ) -> None:
        """Consume input bindings and publish outputs from one forward node."""

        action = ReleaseAction.EMPTY_STORAGE if retire_consumed else None
        stream_key = _stream_key(stream)
        if isinstance(consumed, torch.Tensor):
            if consumed.is_cuda:
                self._consume_tensor(consumed, action, stream, stream_key)
        else:
            self._consume_structured(consumed, action, stream, stream_key)
        if isinstance(produced, torch.Tensor):
            if produced.is_cuda:
                self._publish_tensor(produced, stream, stream_key, node)
        else:
            self._publish_structured(produced, stream, stream_key, node)

    def consume_forward_outputs(self, forward_outputs: Any) -> None:
        """End owner bindings for forward outputs retained and consumed by autograd."""

        if isinstance(forward_outputs, torch.Tensor):
            if forward_outputs.is_cuda:
                self._consume_tensor(forward_outputs, None, None, None)
        else:
            self._consume_structured(forward_outputs, None, None, None)

    def consume_output_grads_and_publish_input_grads(
        self,
        output_grads: Any,
        input_grads: Any,
        *,
        stream: torch.cuda.Stream,
        node: str,
        additional_consumed_grads: tuple[Any, ...] = (),
    ) -> None:
        """Retire output grads and publish input grads produced by one backward node."""

        stream_key = _stream_key(stream)
        if isinstance(output_grads, torch.Tensor):
            if output_grads.is_cuda:
                self._consume_tensor(output_grads, ReleaseAction.DROP_REFERENCE, stream, stream_key)
        else:
            self._consume_structured(output_grads, ReleaseAction.DROP_REFERENCE, stream, stream_key)
        if additional_consumed_grads:
            if len(additional_consumed_grads) == 1 and isinstance(
                additional_consumed_grads[0], torch.Tensor
            ):
                additional_grad = additional_consumed_grads[0]
                if additional_grad.is_cuda:
                    self._record_tensor_stream(additional_grad, stream)
            else:
                self._record_stream(additional_consumed_grads, stream)
        if isinstance(input_grads, torch.Tensor):
            if input_grads.is_cuda:
                self._publish_tensor(input_grads, stream, stream_key, node)
        else:
            self._publish_structured(input_grads, stream, stream_key, node)

    def publish(self, value: Any, stream: torch.cuda.Stream, node: str) -> None:
        """Bind tensors produced by a real schedule node to its execution stream."""

        stream_key = _stream_key(stream)
        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                self._publish_tensor(value, stream, stream_key, node)
            return
        self._publish_structured(value, stream, stream_key, node)

    def export(self, value: Any) -> None:
        """Move plan outputs outside manager ownership without retiring their storage."""

        if isinstance(value, torch.Tensor):
            if value.is_cuda:
                self._export_tensor(value)
            return
        if not value:
            return
        for tensor in _iter_unique_cuda_tensors(value):
            self._export_tensor(tensor)

    def drain(self, stream: torch.cuda.Stream, release_node: str, terminal: bool = False) -> None:
        """Release entries after the caller has enqueued its normal plan-event wait."""

        entries = self._pending.pop(_stream_key(stream), ())
        if not entries:
            return
        self._pending_count -= len(entries)
        for entry in entries:
            self._apply_action(entry.tensor, entry.action)
            stat = "released_terminal" if terminal else "released_natural"
            self.stats[stat] += 1
        self.last_release_node = release_node

    def finalize_phase(self, event: torch.cuda.Event, phase: str, outputs: Any = ()) -> None:
        """Hand pending tensors back, export outputs, and audit owner bindings."""

        self.stats["pending_before_finalize"] = self._pending_count
        for entries in list(self._pending.values()):
            if not entries:
                continue
            stream = entries[0].owner_stream
            event.wait(stream)
            with torch.cuda.stream(stream):
                self.drain(stream, f"{phase}:phase_finalize", terminal=True)
        self.stats["pending_at_finalize"] = self._pending_count
        if self._pending_count:
            raise RuntimeError(
                f"Phase {phase!r} finalized with {self._pending_count} deferred tensors"
            )

        self.stats["owners_before_finalize"] = len(self._owners)
        self.export(outputs)
        self.stats["owners_at_finalize"] = len(self._owners)
        if self._owners:
            producers = sorted({owner.producer_node for owner in self._owners.values()})
            raise RuntimeError(
                f"Phase {phase!r} finalized with {len(self._owners)} unconsumed tensor-owner "
                f"bindings from producers {producers}"
            )

    def _consume_structured(
        self,
        value: Any,
        action: Optional[ReleaseAction],
        consumer_stream: Optional[torch.cuda.Stream],
        consumer_stream_key: Optional[StreamKey],
    ) -> None:
        if not value:
            return
        for tensor in _iter_unique_cuda_tensors(value):
            self._consume_tensor(tensor, action, consumer_stream, consumer_stream_key)

    def _consume_tensor(
        self,
        tensor: torch.Tensor,
        action: Optional[ReleaseAction],
        consumer_stream: Optional[torch.cuda.Stream],
        consumer_stream_key: Optional[StreamKey],
    ) -> None:
        owner = self._take_owner(tensor)
        if owner is None:
            if action is not None:
                assert consumer_stream is not None
                tensor.record_stream(consumer_stream)
                self.stats["record_stream_fallback"] += 1
                self._apply_action(tensor, action)
            return

        self.stats["consumed"] += 1
        if action is None:
            return
        assert consumer_stream_key is not None
        self._retire_tensor(tensor, action, owner, consumer_stream_key)

    def _take_owner(self, tensor: torch.Tensor) -> Optional[TensorOwner]:
        tensor_id = id(tensor)
        owner = self._owners.get(tensor_id)
        if owner is None or owner.tensor is not tensor:
            return None
        self._owners.pop(tensor_id)
        return owner

    def _record_stream(self, tensors: Any, stream: torch.cuda.Stream) -> None:
        if isinstance(tensors, torch.Tensor):
            if tensors.is_cuda:
                self._record_tensor_stream(tensors, stream)
            return
        if not tensors:
            return
        for tensor in _iter_unique_cuda_tensors(tensors):
            self._record_tensor_stream(tensor, stream)

    def _record_tensor_stream(self, tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
        tensor.record_stream(stream)
        self.stats["record_stream_fallback"] += 1

    def _publish_structured(
        self, value: Any, stream: torch.cuda.Stream, stream_key: StreamKey, node: str
    ) -> None:
        if not value:
            return
        for tensor in _iter_unique_cuda_tensors(value):
            self._publish_tensor(tensor, stream, stream_key, node)

    def _publish_tensor(
        self, tensor: torch.Tensor, stream: torch.cuda.Stream, stream_key: StreamKey, node: str
    ) -> None:
        tensor_id = id(tensor)
        existing = self._owners.get(tensor_id)
        if existing is not None and existing.tensor is tensor:
            raise RuntimeError(
                f"Tensor already has an unconsumed owner binding from "
                f"{existing.producer_node!r}; producer={node!r}, "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
            )
        self._owners[tensor_id] = TensorOwner(
            tensor=tensor, stream=stream, stream_key=stream_key, producer_node=node
        )
        self.stats["published"] += 1

    def _export_tensor(self, tensor: torch.Tensor) -> None:
        owner = self._take_owner(tensor)
        if owner is not None:
            self.stats["exported"] += 1

    def _retire_tensor(
        self,
        tensor: torch.Tensor,
        action: ReleaseAction,
        owner: TensorOwner,
        consumer_stream_key: StreamKey,
    ) -> None:
        if owner.stream_key == consumer_stream_key:
            self._apply_action(tensor, action)
            self.stats["same_stream"] += 1
            return

        self._pending.setdefault(owner.stream_key, []).append(
            DeferredRelease(tensor=tensor, action=action, owner_stream=owner.stream)
        )
        self._pending_count += 1
        self.stats["deferred"] += 1
        self.stats["max_pending"] = max(self.stats["max_pending"], self._pending_count)

    @staticmethod
    def _apply_action(tensor: torch.Tensor, action: ReleaseAction) -> None:
        if action is ReleaseAction.EMPTY_STORAGE:
            tensor.untyped_storage().resize_(0)
        elif action is not ReleaseAction.DROP_REFERENCE:
            raise AssertionError(f"Unknown scheduled tensor release action: {action}")
