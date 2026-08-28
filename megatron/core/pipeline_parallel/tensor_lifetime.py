# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Schedule-aware CUDA tensor lifetime management.

The CUDA caching allocator cannot infer the dependency graph maintained by the
fine-grained 1F1B scheduler.  This module lets the scheduler retain tensors until
their allocation stream has waited for the last consumer, avoiding allocator-wide
``record_stream`` retirement for tensors whose complete lifetime is known.
"""

from __future__ import annotations

import itertools
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
class TensorProvenance:
    """Creation-stream ownership for one live tensor lease."""

    tensor_ref: weakref.ReferenceType[torch.Tensor]
    creation_stream: torch.cuda.Stream
    creation_stream_key: StreamKey
    producer_node: str
    lease_id: int
    manager_id: Optional[int]


@dataclass
class DeferredRelease:
    """Strong reference retained until the allocation stream can safely reuse storage."""

    tensor: torch.Tensor
    action: ReleaseAction
    creation_stream: torch.cuda.Stream
    creation_stream_key: StreamKey
    consumer_stream_key: StreamKey
    consumer_event_generation: int
    consumer_node: str
    lease_id: int


@dataclass
class StreamAcquireToken:
    """Event generation observed and produced by one schedule-node invocation."""

    waited_generation: int
    completion_generation: Optional[int] = None


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


def _unique_tensors(value: Any) -> list[torch.Tensor]:
    tensors = []
    seen = set()
    for tensor in _iter_tensors(value):
        tensor_id = id(tensor)
        if tensor_id not in seen:
            seen.add(tensor_id)
            tensors.append(tensor)
    return tensors


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    storage_identity = getattr(storage, "_cdata", None)
    if storage_identity is None:
        storage_identity = storage.data_ptr()
    return tensor.device.index, int(storage_identity)


class _TensorProvenanceRegistry:
    """Process-local weak registry used to carry provenance across schedule boundaries."""

    def __init__(self):
        self._entries: dict[int, TensorProvenance] = {}
        self._next_lease_id = itertools.count(1)

    def register(
        self,
        tensor: torch.Tensor,
        creation_stream: torch.cuda.Stream,
        producer_node: str,
        manager_id: Optional[int] = None,
    ) -> TensorProvenance:
        """Create a new lease and reject duplicate registration of a live tensor."""

        if not tensor.is_cuda:
            raise RuntimeError("Scheduled tensor lifetime only supports CUDA tensors")

        tensor_id = id(tensor)
        existing = self._entries.get(tensor_id)
        if existing is not None:
            existing_tensor = existing.tensor_ref()
            if existing_tensor is tensor:
                raise RuntimeError(
                    "Tensor already has an unconsumed scheduled-lifetime lease: "
                    f"lease={existing.lease_id}, producer={existing.producer_node}, "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                )
            self._entries.pop(tensor_id, None)

        lease_id = next(self._next_lease_id)

        def remove_stale(ref, *, key=tensor_id, expected_lease=lease_id):
            entry = self._entries.get(key)
            if entry is not None and entry.lease_id == expected_lease and entry.tensor_ref is ref:
                self._entries.pop(key, None)

        provenance = TensorProvenance(
            tensor_ref=weakref.ref(tensor, remove_stale),
            creation_stream=creation_stream,
            creation_stream_key=_stream_key(creation_stream),
            producer_node=producer_node,
            lease_id=lease_id,
            manager_id=manager_id,
        )
        self._entries[tensor_id] = provenance
        return provenance

    def peek(self, tensor: torch.Tensor) -> Optional[TensorProvenance]:
        """Return a lease only when both its id and weak identity still match."""

        entry = self._entries.get(id(tensor))
        if entry is None:
            return None
        if entry.tensor_ref() is not tensor:
            self._entries.pop(id(tensor), None)
            return None
        return entry

    def take(self, tensor: torch.Tensor) -> Optional[TensorProvenance]:
        """Remove and return the matching live lease, if one exists."""

        entry = self.peek(tensor)
        if entry is not None:
            self._entries.pop(id(tensor), None)
        return entry


_PROVENANCE_REGISTRY = _TensorProvenanceRegistry()


def register_external_tensor(
    value: Any, creation_stream: torch.cuda.Stream, producer_node: str = "external"
) -> None:
    """Register tensors allocated outside a ``ScheduleNode`` with an explicit owner stream."""

    for tensor in _unique_tensors(value):
        _PROVENANCE_REGISTRY.register(tensor, creation_stream, producer_node)


class ScheduleTensorLifetimeManager:
    """Own event generations and deferred releases for one model-chunk schedule plan."""

    _manager_ids = itertools.count(1)

    def __init__(self, event: torch.cuda.Event):
        self.event = event
        self.manager_id = next(self._manager_ids)
        self.generation = 0
        self.pending: list[DeferredRelease] = []
        self._live_lease_ids: set[int] = set()
        self.active_phase: Optional[str] = None
        self._dirty_detached_leaves: list[torch.Tensor] = []
        self._dirty_detached_ids: set[int] = set()
        self.stats = {
            "registered": 0,
            "same_stream": 0,
            "deferred": 0,
            "released_natural": 0,
            "released_terminal": 0,
            "unknown_provenance": 0,
            "max_pending": 0,
            "pending_before_finalize": 0,
            "pending_at_finalize": 0,
        }
        self.last_release_node: Optional[str] = None

    def begin_phase(self, phase: str) -> None:
        """Start a schedule phase only after the previous phase is fully drained."""

        if self.active_phase is not None:
            raise RuntimeError(
                f"Cannot begin scheduled-lifetime phase {phase!r}; "
                f"phase {self.active_phase!r} is still active"
            )
        if self.pending:
            raise RuntimeError(
                f"Cannot begin phase {phase!r} with {len(self.pending)} deferred tensors"
            )
        if self._live_lease_ids:
            raise RuntimeError(
                f"Cannot begin phase {phase!r} with {len(self._live_lease_ids)} live leases"
            )
        self.active_phase = phase

    def acquire(self, stream: torch.cuda.Stream, node: str) -> StreamAcquireToken:
        """Wait for the current generation and drain releases owned by ``stream``."""

        if self.active_phase is None:
            raise RuntimeError(f"Schedule node {node!r} acquired outside an active phase")
        waited_generation = self.generation
        self.event.wait(stream)
        self._drain(stream, waited_generation, release_node=node, terminal=False)
        return StreamAcquireToken(waited_generation=waited_generation)

    def record(self, stream: torch.cuda.Stream, token: StreamAcquireToken) -> int:
        """Publish a node-completion generation from ``stream``."""

        self.event.record(stream)
        self.generation += 1
        token.completion_generation = self.generation
        return self.generation

    def record_root(self, stream: torch.cuda.Stream, node: str) -> int:
        """Seed a phase dependency before its first schedule node is acquired."""

        if self.active_phase is None:
            raise RuntimeError(f"Root event {node!r} recorded outside an active phase")
        if self.pending:
            raise RuntimeError(
                f"Root event {node!r} would overwrite dependencies for "
                f"{len(self.pending)} deferred tensors"
            )
        self.event.record(stream)
        self.generation += 1
        return self.generation

    def wait(self, stream: torch.cuda.Stream, node: str) -> None:
        """Wait for the latest generation without recording a new generation."""

        if self.active_phase is None:
            raise RuntimeError(f"Schedule wait {node!r} executed outside an active phase")
        waited_generation = self.generation
        self.event.wait(stream)
        self._drain(stream, waited_generation, release_node=node, terminal=False)

    def publish(self, value: Any, creation_stream: torch.cuda.Stream, producer_node: str) -> None:
        """Register newly produced tensors as leases owned by this manager."""

        if self.active_phase is None:
            raise RuntimeError(f"Cannot publish tensors for {producer_node!r} outside a phase")
        for tensor in _unique_tensors(value):
            provenance = _PROVENANCE_REGISTRY.register(
                tensor, creation_stream, producer_node, manager_id=self.manager_id
            )
            self._live_lease_ids.add(provenance.lease_id)
            self.stats["registered"] += 1

    def consume_and_publish(
        self, consumed: Any, produced: Any, *, producer_stream: torch.cuda.Stream, node: str
    ) -> None:
        """Transfer live leases through a node without retiring their storage.

        Nodes whose inputs are retained for autograd do not need a deferred release: the
        node itself keeps the storage alive.  The input lease is nevertheless consumed,
        and an aliasing output inherits its creation-stream owner.  A newly allocated
        output is owned by ``producer_stream``.
        """

        consumed_tensors = _unique_tensors(consumed)
        produced_tensors = _unique_tensors(produced)
        produced_by_storage: dict[tuple[int, int], list[torch.Tensor]] = {}
        for tensor in produced_tensors:
            produced_by_storage.setdefault(_storage_key(tensor), []).append(tensor)

        transferred_ids = set()
        for tensor in consumed_tensors:
            provenance = self._require_provenance(tensor, node, producer_stream)
            _PROVENANCE_REGISTRY.take(tensor)
            self._live_lease_ids.discard(provenance.lease_id)
            for alias in produced_by_storage.get(_storage_key(tensor), ()):
                if id(alias) in transferred_ids:
                    continue
                alias_provenance = _PROVENANCE_REGISTRY.register(
                    alias,
                    provenance.creation_stream,
                    f"{node}:alias-transfer",
                    manager_id=self.manager_id,
                )
                self._live_lease_ids.add(alias_provenance.lease_id)
                transferred_ids.add(id(alias))
                self.stats["registered"] += 1

        for tensor in produced_tensors:
            if id(tensor) in transferred_ids:
                continue
            provenance = _PROVENANCE_REGISTRY.register(
                tensor, producer_stream, node, manager_id=self.manager_id
            )
            self._live_lease_ids.add(provenance.lease_id)
            self.stats["registered"] += 1

    def export(self, value: Any) -> None:
        """Move phase outputs outside schedule ownership without releasing storage."""

        for tensor in _unique_tensors(value):
            provenance = _PROVENANCE_REGISTRY.peek(tensor)
            if provenance is None:
                raise RuntimeError(
                    "Cannot export a tensor without scheduled-lifetime provenance: "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
                )
            if provenance.manager_id not in (None, self.manager_id):
                raise RuntimeError(
                    f"Cannot export tensor lease {provenance.lease_id} owned by lifetime "
                    f"manager {provenance.manager_id} from manager {self.manager_id}"
                )
            _PROVENANCE_REGISTRY.take(tensor)
            self._live_lease_ids.discard(provenance.lease_id)

    def retire_and_publish(
        self,
        consumed: Any,
        produced: Any,
        *,
        action: ReleaseAction,
        producer_stream: torch.cuda.Stream,
        consumer_stream: torch.cuda.Stream,
        consumer_generation: int,
        node: str,
    ) -> None:
        """Retire consumed tensors and publish produced tensors as one alias-safe transaction."""

        consumed_tensors = _unique_tensors(consumed)
        produced_tensors = _unique_tensors(produced)
        produced_by_storage: dict[tuple[int, int], list[torch.Tensor]] = {}
        for tensor in produced_tensors:
            produced_by_storage.setdefault(_storage_key(tensor), []).append(tensor)

        transferred_ids = set()
        for tensor in consumed_tensors:
            provenance = self._require_provenance(tensor, node, consumer_stream)
            aliases = produced_by_storage.get(_storage_key(tensor), [])
            if aliases:
                if action is ReleaseAction.EMPTY_STORAGE:
                    raise RuntimeError(
                        f"free_input tensor aliases output storage in node {node!r}; "
                        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
                    )
                _PROVENANCE_REGISTRY.take(tensor)
                self._live_lease_ids.discard(provenance.lease_id)
                for alias in aliases:
                    if id(alias) in transferred_ids:
                        continue
                    alias_provenance = _PROVENANCE_REGISTRY.register(
                        alias,
                        provenance.creation_stream,
                        f"{node}:alias-transfer",
                        manager_id=self.manager_id,
                    )
                    self._live_lease_ids.add(alias_provenance.lease_id)
                    transferred_ids.add(id(alias))
                    self.stats["registered"] += 1
                continue

            _PROVENANCE_REGISTRY.take(tensor)
            self._live_lease_ids.discard(provenance.lease_id)
            self._retire_tensor(
                tensor,
                provenance,
                action=action,
                consumer_stream=consumer_stream,
                consumer_generation=consumer_generation,
                node=node,
            )

        for tensor in produced_tensors:
            if id(tensor) in transferred_ids:
                continue
            provenance = _PROVENANCE_REGISTRY.register(
                tensor, producer_stream, node, manager_id=self.manager_id
            )
            self._live_lease_ids.add(provenance.lease_id)
            self.stats["registered"] += 1

    def retire(
        self,
        consumed: Any,
        *,
        action: ReleaseAction,
        consumer_stream: torch.cuda.Stream,
        consumer_generation: int,
        node: str,
    ) -> None:
        """Retire consumed tensors without publishing replacement tensors."""

        self.retire_and_publish(
            consumed,
            (),
            action=action,
            producer_stream=consumer_stream,
            consumer_stream=consumer_stream,
            consumer_generation=consumer_generation,
            node=node,
        )

    def track_detached_leaf(self, tensor: torch.Tensor):
        """Return a hook handle that reports when a detached leaf's ``.grad`` materializes."""

        manager_ref = weakref.ref(self)

        def mark_dirty(leaf: torch.Tensor):
            manager = manager_ref()
            if manager is None:
                return
            leaf_id = id(leaf)
            if leaf_id not in manager._dirty_detached_ids:
                manager._dirty_detached_ids.add(leaf_id)
                manager._dirty_detached_leaves.append(leaf)

        return tensor.register_post_accumulate_grad_hook(mark_dirty)

    def publish_dirty_detached_grads(
        self, creation_stream: torch.cuda.Stream, producer_node: str
    ) -> None:
        """Publish materialized detached-leaf gradients queued by autograd hooks."""

        leaves = self._dirty_detached_leaves
        self._dirty_detached_leaves = []
        self._dirty_detached_ids.clear()
        for leaf in leaves:
            grad = leaf.grad
            if grad is None:
                raise RuntimeError(
                    f"Detached leaf hook fired without a materialized grad in {producer_node!r}"
                )
            provenance = _PROVENANCE_REGISTRY.register(
                grad, creation_stream, f"{producer_node}:detached-grad", manager_id=self.manager_id
            )
            self._live_lease_ids.add(provenance.lease_id)
            self.stats["registered"] += 1

    def finalize_phase(self, phase: str, outputs: Any = ()) -> None:
        """Hand pending tensors back to owner streams, export outputs, and audit leaks."""

        if self.active_phase != phase:
            raise RuntimeError(
                f"Cannot finalize phase {phase!r}; active phase is {self.active_phase!r}"
            )

        self.stats["pending_before_finalize"] = len(self.pending)
        waited_generation = self.generation
        creation_streams = {}
        for entry in self.pending:
            creation_streams[entry.creation_stream_key] = entry.creation_stream
        for stream in creation_streams.values():
            self.event.wait(stream)
            self._drain(
                stream, waited_generation, release_node=f"{phase}:phase_finalize", terminal=True
            )

        if self.pending:
            raise RuntimeError(
                f"Phase {phase!r} finalized with {len(self.pending)} deferred tensors"
            )
        self.stats["pending_at_finalize"] = len(self.pending)
        if self._dirty_detached_leaves:
            raise RuntimeError(
                f"Phase {phase!r} finalized with "
                f"{len(self._dirty_detached_leaves)} unregistered detached gradients"
            )
        self.export(outputs)
        if self._live_lease_ids:
            raise RuntimeError(
                f"Phase {phase!r} finalized with {len(self._live_lease_ids)} live tensor leases"
            )
        self.active_phase = None

    def _require_provenance(
        self, tensor: torch.Tensor, consumer_node: str, consumer_stream: torch.cuda.Stream
    ) -> TensorProvenance:
        provenance = _PROVENANCE_REGISTRY.peek(tensor)
        if provenance is None:
            self.stats["unknown_provenance"] += 1
            stream_key = _stream_key(consumer_stream)
            raise RuntimeError(
                "Missing creation-stream provenance for scheduled tensor: "
                f"consumer={consumer_node}, consumer_stream={stream_key}, "
                f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
            )
        if provenance.manager_id is None:
            provenance.manager_id = self.manager_id
            self._live_lease_ids.add(provenance.lease_id)
        elif provenance.manager_id != self.manager_id:
            raise RuntimeError(
                f"Tensor lease {provenance.lease_id} belongs to lifetime manager "
                f"{provenance.manager_id}, not manager {self.manager_id}; "
                f"consumer={consumer_node}"
            )
        return provenance

    def _retire_tensor(
        self,
        tensor: torch.Tensor,
        provenance: TensorProvenance,
        *,
        action: ReleaseAction,
        consumer_stream: torch.cuda.Stream,
        consumer_generation: int,
        node: str,
    ) -> None:
        consumer_stream_key = _stream_key(consumer_stream)
        if provenance.creation_stream_key == consumer_stream_key:
            self._apply_action(tensor, action)
            self.stats["same_stream"] += 1
            return

        self.pending.append(
            DeferredRelease(
                tensor=tensor,
                action=action,
                creation_stream=provenance.creation_stream,
                creation_stream_key=provenance.creation_stream_key,
                consumer_stream_key=consumer_stream_key,
                consumer_event_generation=consumer_generation,
                consumer_node=node,
                lease_id=provenance.lease_id,
            )
        )
        self.stats["deferred"] += 1
        self.stats["max_pending"] = max(self.stats["max_pending"], len(self.pending))

    def _drain(
        self,
        stream: torch.cuda.Stream,
        waited_generation: int,
        *,
        release_node: str,
        terminal: bool,
    ) -> None:
        stream_key = _stream_key(stream)
        retained = []
        for entry in self.pending:
            if (
                entry.creation_stream_key == stream_key
                and entry.consumer_event_generation <= waited_generation
            ):
                self._apply_action(entry.tensor, entry.action)
                stat = "released_terminal" if terminal else "released_natural"
                self.stats[stat] += 1
                self.last_release_node = release_node
            else:
                retained.append(entry)
        self.pending = retained

    @staticmethod
    def _apply_action(tensor: torch.Tensor, action: ReleaseAction) -> None:
        if action is ReleaseAction.EMPTY_STORAGE:
            tensor.untyped_storage().resize_(0)
        elif action is not ReleaseAction.DROP_REFERENCE:
            raise AssertionError(f"Unknown scheduled tensor release action: {action}")
