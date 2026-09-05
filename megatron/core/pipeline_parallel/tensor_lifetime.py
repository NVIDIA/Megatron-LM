# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-local lifetime management for tensors crossing CUDA streams.

Only tensors that the existing ``ScheduleNode`` path actively retires participate:
forward ``free_input`` tensors and backward incoming gradients.  Their owner
streams are known from the static layer-node topology.  Unknown or externally
produced gradients retain the existing ``record_stream`` behavior.

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


@dataclass
class DeferredRelease:
    """Strong reference held until the owner stream has acquired the consumer."""

    payload: Any
    action: ReleaseAction
    owner_stream: torch.cuda.Stream
    consumer_node: str


def _stream_handle(stream: torch.cuda.Stream) -> int:
    return int(stream.cuda_stream)


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _unique_cuda_tensors(value: Any) -> list[torch.Tensor]:
    tensors = []
    seen = set()
    for tensor in _iter_tensors(value):
        tensor_id = id(tensor)
        if not tensor.is_cuda or tensor_id in seen:
            continue
        seen.add(tensor_id)
        tensors.append(tensor)
    return tensors


class ScheduleTensorLifetimeManager:
    """Retire selected tensors within one model-chunk schedule plan.

    Ownership is passed explicitly by the layer topology, so ordinary forward
    outputs and backward gradients are not registered or scanned.  The only
    dynamic tags identify gradients entering the schedule from an external
    boundary, where the owner stream is deliberately unknown.
    """

    def __init__(self):
        self._external_gradients: dict[int, torch.Tensor] = {}
        self._pending: dict[int, list[DeferredRelease]] = {}
        self._pending_count = 0
        self.stats = {
            "external_gradient_tags": 0,
            "same_stream": 0,
            "deferred": 0,
            "record_stream_fallback": 0,
            "released_natural": 0,
            "released_terminal": 0,
            "max_pending": 0,
            "pending_before_finalize": 0,
            "pending_at_finalize": 0,
        }
        self.last_release_node: Optional[str] = None

    @property
    def pending(self) -> list[DeferredRelease]:
        """Return all deferred entries, primarily for diagnostics and tests."""

        return [entry for entries in self._pending.values() for entry in entries]

    def retire_forward_inputs(
        self,
        consumed: Any,
        *,
        owner_stream: torch.cuda.Stream,
        consumer_stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        """Retire ``free_input`` tensors using the statically known producer stream."""

        self._retire(consumed, ReleaseAction.EMPTY_STORAGE, owner_stream, consumer_stream, node)

    def retire_backward_inputs(
        self,
        consumed: Any,
        *,
        owner_stream: Optional[torch.cuda.Stream],
        consumer_stream: torch.cuda.Stream,
        node: str,
        fallback_consumed: Any = (),
    ) -> None:
        """Protect incoming gradients using the statically known previous node stream."""

        if owner_stream is None or (
            self._external_gradients and self._consume_external_gradients(consumed)
        ):
            self._record_stream(consumed, consumer_stream)
        else:
            # Store the tuple as one strong-reference payload.  There is no need to
            # inspect each ordinary gradient on this hot path.
            self._retire(
                consumed, ReleaseAction.DROP_REFERENCE, owner_stream, consumer_stream, node
            )

        self._record_stream(fallback_consumed, consumer_stream)

    def mark_external_gradients(self, gradients: Any) -> None:
        """Mark gradients entering this plan without a statically known owner stream."""

        for tensor in _unique_cuda_tensors(gradients):
            self._mark_external_gradient(tensor)

    def drain(self, stream: torch.cuda.Stream, release_node: str, terminal: bool = False) -> None:
        """Release entries after the caller has enqueued its normal plan-event wait."""

        entries = self._pending.pop(_stream_handle(stream), ())
        if not entries:
            return
        self._pending_count -= len(entries)
        for entry in entries:
            self._apply_action(entry.payload, entry.action)
            stat = "released_terminal" if terminal else "released_natural"
            self.stats[stat] += 1
        self.last_release_node = release_node

    def finalize_phase(self, event: torch.cuda.Event, phase: str) -> None:
        """Hand back entries that have no later owner-stream node in this phase."""

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
        self._external_gradients.clear()

    def _mark_external_gradient(self, tensor: torch.Tensor) -> None:
        self._external_gradients[id(tensor)] = tensor
        self.stats["external_gradient_tags"] += 1

    def _consume_external_gradients(self, gradients: Any) -> bool:
        found = False
        for tensor in _unique_cuda_tensors(gradients):
            external = self._external_gradients.get(id(tensor))
            if external is tensor:
                self._external_gradients.pop(id(tensor), None)
                found = True
        return found

    def _record_stream(self, tensors: Any, stream: torch.cuda.Stream) -> None:
        for tensor in _unique_cuda_tensors(tensors):
            tensor.record_stream(stream)
            self.stats["record_stream_fallback"] += 1

    def _retire(
        self,
        payload: Any,
        action: ReleaseAction,
        owner_stream: torch.cuda.Stream,
        consumer_stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        if _stream_handle(owner_stream) == _stream_handle(consumer_stream):
            self._apply_action(payload, action)
            self.stats["same_stream"] += 1
            return

        self._pending.setdefault(_stream_handle(owner_stream), []).append(
            DeferredRelease(
                payload=payload, action=action, owner_stream=owner_stream, consumer_node=node
            )
        )
        self._pending_count += 1
        self.stats["deferred"] += 1
        self.stats["max_pending"] = max(self.stats["max_pending"], self._pending_count)

    @staticmethod
    def _apply_action(payload: Any, action: ReleaseAction) -> None:
        if action is ReleaseAction.EMPTY_STORAGE:
            for tensor in _unique_cuda_tensors(payload):
                tensor.untyped_storage().resize_(0)
        elif action is not ReleaseAction.DROP_REFERENCE:
            raise AssertionError(f"Unknown scheduled tensor release action: {action}")
