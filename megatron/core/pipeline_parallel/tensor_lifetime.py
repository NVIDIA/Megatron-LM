# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-local lifetime management for tensors crossing CUDA streams.

Each model-chunk schedule plan already owns a CUDA event that orders the nodes
of one microbatch.  This module uses that existing dependency chain to return a
consumed tensor to its allocation stream before releasing its storage.  Tensors
whose owner stream is not known keep using ``record_stream`` as a safe fallback.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Iterable, Optional

import torch


class ReleaseAction(Enum):
    """Action performed after a tensor is safe to retire."""

    EMPTY_STORAGE = auto()
    DROP_REFERENCE = auto()


@dataclass(frozen=True)
class StreamKey:
    """Stable identity for a CUDA stream independent of its Python wrapper."""

    device_index: int
    cuda_stream: int


@dataclass
class _TensorOwner:
    """Weak, schedule-local tag identifying a tensor's allocation stream."""

    tensor_ref: weakref.ReferenceType[torch.Tensor]
    stream: torch.cuda.Stream
    stream_key: StreamKey


@dataclass
class DeferredRelease:
    """Strong reference held until the owner stream has waited for the consumer."""

    tensor: torch.Tensor
    action: ReleaseAction
    owner_stream: torch.cuda.Stream
    owner_stream_key: StreamKey
    consumer_node: str


def _stream_key(stream: torch.cuda.Stream) -> StreamKey:
    device = stream.device
    device_index = device.index if isinstance(device, torch.device) else int(device)
    if device_index is None:
        device_index = torch.cuda.current_device()
    return StreamKey(device_index=device_index, cuda_stream=int(stream.cuda_stream))


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
        if not tensor.is_cuda or id(tensor) in seen:
            continue
        seen.add(id(tensor))
        tensors.append(tensor)
    return tensors


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    storage_identity = getattr(storage, "_cdata", None)
    if storage_identity is None:
        storage_identity = storage.data_ptr()
    return tensor.device.index, int(storage_identity)


class ScheduleTensorLifetimeManager:
    """Manage hand-back only within one model-chunk schedule plan.

    A producer tags its outputs with the stream that owns their storage.  After
    a cross-stream consumer records the plan event, this manager retains the
    tensor in the owner's queue.  A later node from the *same plan* first waits
    on that event and then drains its stream's queue.  No dependency is added to
    nodes from another microbatch, so the scheduler's overlap pattern is kept.
    """

    def __init__(self):
        self._owners: dict[int, _TensorOwner] = {}
        self._pending: dict[StreamKey, list[DeferredRelease]] = {}
        self.stats = {
            "tagged": 0,
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

    def publish(self, consumed: Any, produced: Any, producer_stream: torch.cuda.Stream) -> None:
        """Tag outputs, inheriting ownership when an output aliases an input."""

        consumed_owners = self._owners_by_storage(consumed)
        for tensor in _unique_cuda_tensors(produced):
            owner = consumed_owners.get(_storage_key(tensor))
            if owner is None and _storage_key(tensor) in consumed_owners:
                # An untagged external input was propagated as an alias.  Leaving
                # the output untagged preserves the record_stream fallback.
                self._forget(tensor)
                continue
            owner_stream = owner.stream if owner is not None else producer_stream
            self._tag(tensor, owner_stream)

    def retire_and_publish(
        self,
        consumed: Any,
        produced: Any,
        *,
        action: ReleaseAction,
        producer_stream: torch.cuda.Stream,
        consumer_stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        """Retire inputs and tag outputs without losing storage ownership."""

        consumed_tensors = _unique_cuda_tensors(consumed)
        produced_tensors = _unique_cuda_tensors(produced)
        produced_storage = {_storage_key(tensor) for tensor in produced_tensors}
        consumed_owners = self._owners_by_storage(consumed_tensors)

        for tensor in consumed_tensors:
            storage_key = _storage_key(tensor)
            if storage_key in produced_storage:
                if action is ReleaseAction.EMPTY_STORAGE:
                    raise RuntimeError(
                        f"free_input tensor aliases output storage in node {node!r}; "
                        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                    )
                # The storage is still live through an output.  Its owner tag is
                # transferred below instead of scheduling a release.
                self._forget(tensor)
                continue
            self._retire(tensor, action, consumer_stream, node)

        for tensor in produced_tensors:
            owner = consumed_owners.get(_storage_key(tensor))
            if owner is None and _storage_key(tensor) in consumed_owners:
                self._forget(tensor)
                continue
            owner_stream = owner.stream if owner is not None else producer_stream
            self._tag(tensor, owner_stream)

    def drain(self, stream: torch.cuda.Stream, release_node: str, terminal: bool = False) -> None:
        """Release entries after the caller has enqueued a wait on ``stream``."""

        entries = self._pending.pop(_stream_key(stream), ())
        if not entries:
            return
        for entry in entries:
            self._apply_action(entry.tensor, entry.action)
            stat = "released_terminal" if terminal else "released_natural"
            self.stats[stat] += 1
        self.last_release_node = release_node

    def finalize_phase(self, event: torch.cuda.Event, phase: str) -> None:
        """Hand back entries that have no later owner-stream node in this phase."""

        self.stats["pending_before_finalize"] = len(self.pending)
        for entries in list(self._pending.values()):
            if not entries:
                continue
            stream = entries[0].owner_stream
            event.wait(stream)
            with torch.cuda.stream(stream):
                self.drain(stream, f"{phase}:phase_finalize", terminal=True)
        self.stats["pending_at_finalize"] = len(self.pending)
        if self.pending:
            raise RuntimeError(
                f"Phase {phase!r} finalized with {len(self.pending)} deferred tensors"
            )
        # Owner tags are weak, but they are phase-local by design.  Outputs that
        # leave the schedule simply stop participating in hand-back.
        self._owners.clear()

    def _owners_by_storage(self, value: Any) -> dict[tuple[int, int], Optional[_TensorOwner]]:
        owners: dict[tuple[int, int], Optional[_TensorOwner]] = {}
        for tensor in _unique_cuda_tensors(value):
            storage_key = _storage_key(tensor)
            owner = self._owner(tensor)
            if storage_key not in owners:
                owners[storage_key] = owner
            elif owner is not None and owners[storage_key] is not None:
                existing_owner = owners[storage_key]
                assert existing_owner is not None
                if owner.stream_key != existing_owner.stream_key:
                    raise RuntimeError("Aliases of one storage have different owner streams")
            elif owner is not None:
                owners[storage_key] = owner
        return owners

    def _tag(self, tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
        tensor_id = id(tensor)
        manager_ref = weakref.ref(self)

        def remove_stale(ref, *, key=tensor_id):
            manager = manager_ref()
            if manager is None:
                return
            entry = manager._owners.get(key)
            if entry is not None and entry.tensor_ref is ref:
                manager._owners.pop(key, None)

        self._owners[tensor_id] = _TensorOwner(
            tensor_ref=weakref.ref(tensor, remove_stale),
            stream=stream,
            stream_key=_stream_key(stream),
        )
        self.stats["tagged"] += 1

    def _owner(self, tensor: torch.Tensor) -> Optional[_TensorOwner]:
        owner = self._owners.get(id(tensor))
        if owner is None:
            return None
        if owner.tensor_ref() is not tensor:
            self._owners.pop(id(tensor), None)
            return None
        return owner

    def _forget(self, tensor: torch.Tensor) -> None:
        owner = self._owners.get(id(tensor))
        if owner is not None and owner.tensor_ref() is tensor:
            self._owners.pop(id(tensor), None)

    def _retire(
        self,
        tensor: torch.Tensor,
        action: ReleaseAction,
        consumer_stream: torch.cuda.Stream,
        node: str,
    ) -> None:
        owner = self._owner(tensor)
        self._forget(tensor)
        if owner is None:
            tensor.record_stream(consumer_stream)
            self._apply_action(tensor, action)
            self.stats["record_stream_fallback"] += 1
            return

        if owner.stream_key == _stream_key(consumer_stream):
            self._apply_action(tensor, action)
            self.stats["same_stream"] += 1
            return

        self._pending.setdefault(owner.stream_key, []).append(
            DeferredRelease(
                tensor=tensor,
                action=action,
                owner_stream=owner.stream,
                owner_stream_key=owner.stream_key,
                consumer_node=node,
            )
        )
        self.stats["deferred"] += 1
        self.stats["max_pending"] = max(self.stats["max_pending"], len(self.pending))

    @staticmethod
    def _apply_action(tensor: torch.Tensor, action: ReleaseAction) -> None:
        if action is ReleaseAction.EMPTY_STORAGE:
            tensor.untyped_storage().resize_(0)
        elif action is not ReleaseAction.DROP_REFERENCE:
            raise AssertionError(f"Unknown scheduled tensor release action: {action}")
