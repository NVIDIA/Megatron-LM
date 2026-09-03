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


@dataclass
class TensorOwner:
    """Strong, plan-local binding from one concrete tensor to its producer stream."""

    tensor: torch.Tensor
    stream: torch.cuda.Stream
    producer_node: str


@dataclass
class DeferredRelease:
    """Strong reference held until the owner stream has acquired the consumer."""

    tensor: torch.Tensor
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
    # Schedule edges are overwhelmingly a single tensor (or a one-tensor tuple).
    # Keep those hot paths allocation-light; recurse only for structured outputs.
    if isinstance(value, torch.Tensor):
        return [value] if value.is_cuda else []
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], torch.Tensor):
        return [value[0]] if value[0].is_cuda else []

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

    Ownership follows concrete tensor objects rather than schedule-node input
    slots.  Bindings are strong and single-consumer: a real node consumes each
    input binding once, while ``NoopScheduleNode`` naturally carries the same
    object and binding to the next real node.  The schedule's existing
    ``free_input`` contract guarantees that retired inputs do not alias outputs.
    """

    def __init__(self):
        self._owners: dict[int, TensorOwner] = {}
        self._pending: dict[int, list[DeferredRelease]] = {}
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

    def finish_forward(
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
        self._consume(consumed, action=action, consumer_stream=stream, node=node)
        self.publish(produced, stream, node)

    def finish_backward(
        self,
        consumed: Any,
        produced: Any,
        *,
        forward_outputs: Any,
        stream: torch.cuda.Stream,
        node: str,
        fallback_consumed: Any = (),
    ) -> None:
        """Retire incoming gradients and publish gradients produced by one backward node."""

        # A recompute segment's final forward output is consumed by autograd rather
        # than another forward node, so its owner metadata ends at this backward.
        self._consume(forward_outputs, action=None, consumer_stream=stream, node=node)
        self._consume(
            consumed, action=ReleaseAction.DROP_REFERENCE, consumer_stream=stream, node=node
        )
        self._record_stream(fallback_consumed, stream)
        self.publish(produced, stream, node)

    def publish(self, value: Any, stream: torch.cuda.Stream, node: str) -> None:
        """Bind tensors produced by a real schedule node to its execution stream."""

        for tensor in _unique_cuda_tensors(value):
            tensor_id = id(tensor)
            existing = self._owners.get(tensor_id)
            if existing is not None and existing.tensor is tensor:
                raise RuntimeError(
                    f"Tensor already has an unconsumed owner binding from "
                    f"{existing.producer_node!r}; producer={node!r}, "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                )
            self._owners[tensor_id] = TensorOwner(tensor=tensor, stream=stream, producer_node=node)
            self.stats["published"] += 1

    def export(self, value: Any) -> None:
        """Move plan outputs outside manager ownership without retiring their storage."""

        for tensor in _unique_cuda_tensors(value):
            owner = self._take_owner(tensor)
            if owner is not None:
                self.stats["exported"] += 1

    def drain(self, stream: torch.cuda.Stream, release_node: str, terminal: bool = False) -> None:
        """Release entries after the caller has enqueued its normal plan-event wait."""

        entries = self._pending.pop(_stream_handle(stream), ())
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

    def _consume(
        self,
        value: Any,
        *,
        action: Optional[ReleaseAction],
        consumer_stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        for tensor in _unique_cuda_tensors(value):
            owner = self._take_owner(tensor)
            if owner is None:
                if action is not None:
                    self._record_stream(tensor, consumer_stream)
                    self._apply_action(tensor, action)
                continue

            self.stats["consumed"] += 1
            if action is None:
                continue
            self._retire_tensor(tensor, action, owner.stream, consumer_stream, node)

    def _take_owner(self, tensor: torch.Tensor) -> Optional[TensorOwner]:
        owner = self._owners.get(id(tensor))
        if owner is None or owner.tensor is not tensor:
            return None
        self._owners.pop(id(tensor), None)
        return owner

    def _record_stream(self, tensors: Any, stream: torch.cuda.Stream) -> None:
        for tensor in _unique_cuda_tensors(tensors):
            tensor.record_stream(stream)
            self.stats["record_stream_fallback"] += 1

    def _retire_tensor(
        self,
        tensor: torch.Tensor,
        action: ReleaseAction,
        owner_stream: torch.cuda.Stream,
        consumer_stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        if _stream_handle(owner_stream) == _stream_handle(consumer_stream):
            self._apply_action(tensor, action)
            self.stats["same_stream"] += 1
            return

        self._pending.setdefault(_stream_handle(owner_stream), []).append(
            DeferredRelease(
                tensor=tensor, action=action, owner_stream=owner_stream, consumer_node=node
            )
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
