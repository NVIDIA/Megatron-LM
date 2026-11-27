# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepEP MoE EP chunk-overlap primitive."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Literal

import torch
import torch.nn as nn

from megatron.lite.primitive.modules.chunked_ep_dispatcher import (
    ChunkedDispatcher as TokenDispatcher,
)
from megatron.lite.primitive.modules.chunked_ep_experts import ChunkedExperts as Experts
from megatron.lite.primitive.modules.chunked_ep_experts import (
    _caller_owned_dummy_wgrad,
    _tensor_byte_ranges_overlap,
)
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import runtime_ep_chunk_ranges
from megatron.lite.primitive.utils.moe import _te_general_gemm, unpermute

# Physical workspace capacity, deliberately independent of logical chunk count.
EP_CHUNK_COUNT = 2
EPChunkOpName = Literal["forward", "backward", "fused_forward_backward"]


def _event_pending(event) -> bool:
    return event is not None and not (hasattr(event, "query") and bool(event.query()))


def _wait_for_consumer_event(event, stream) -> bool:
    """Wait before storage reuse; report whether a dependency was queued."""
    if not _event_pending(event):
        return False
    if stream is not None and hasattr(stream, "wait_event"):
        stream.wait_event(event)
    elif hasattr(event, "current_stream_wait"):
        event.current_stream_wait()
    else:
        raise RuntimeError("Pending EP chunk consumer event is not stream-waitable")
    return True


@dataclass(frozen=True)
class EPChunkShapeProfile:
    """Fixed token and DeepEP receive capacities for one EP rank."""

    max_input_rows: int
    hidden_size: int
    topk: int
    ep_size: int
    chunk_count: int = 2
    expert_intermediate_size: int | None = None
    max_recv_rows: int = field(init=False)
    max_expert_rows: int = field(init=False)

    def __post_init__(self) -> None:
        if any(
            int(value) <= 0
            for value in (
                self.max_input_rows,
                self.hidden_size,
                self.topk,
                self.ep_size,
                self.chunk_count,
            )
        ):
            raise ValueError("EP chunk shape-profile capacities must be positive")
        if self.ep_size <= 1:
            raise ValueError("Two-slot EP chunk profile requires EP > 1")
        if self.chunk_count < 2:
            raise ValueError("EP chunk profile requires at least two chunks")
        if self.max_input_rows < self.chunk_count:
            raise ValueError("EP chunk profile requires at least one row per chunk")
        max_chunk_rows = (self.max_input_rows + self.chunk_count - 1) // self.chunk_count
        # All source chunks may land on one rank: one hidden row per transported
        # token, with local top-k destinations represented in recv_probs.
        max_recv_rows = max_chunk_rows * self.ep_size
        object.__setattr__(self, "max_recv_rows", max_recv_rows)
        # Permutation expands up to top-k rows/token. This is a validation ceiling;
        # workspace allocation remains lazy and uses observed runtime shapes.
        object.__setattr__(self, "max_expert_rows", max_recv_rows * self.topk)

    def validate_rows(self, rows: int, kind: Literal["input", "recv", "expert"]) -> None:
        capacity = getattr(self, f"max_{kind}_rows")
        if rows > capacity:
            raise RuntimeError(f"{kind} rows {rows} exceed capacity {capacity}")

    def validate_input(self, value: torch.Tensor) -> None:
        if value.size(-1) != self.hidden_size:
            raise RuntimeError(f"Hidden size {value.size(-1)} != profile {self.hidden_size}")
        self.validate_rows(value.numel() // self.hidden_size, "input")


def _validate_finished_deepep_dispatch(
    profile: EPChunkShapeProfile, state: dict[str, Any], dispatched: torch.Tensor
) -> None:
    """Validate runtime DeepEP outputs before expert use or arena allocation."""
    recv_hidden = state.get("recv_hidden")
    recv_probs = state.get("recv_probs")
    if not torch.is_tensor(recv_hidden) or recv_hidden.dim() != 2:
        raise RuntimeError("EP chunk DeepEP recv_hidden must be a rank-2 tensor")
    if recv_hidden.size(1) != profile.hidden_size:
        raise RuntimeError(
            f"EP chunk recv hidden size {recv_hidden.size(1)} does not match "
            f"fixed profile {profile.hidden_size}"
        )
    profile.validate_rows(recv_hidden.size(0), "recv")
    if not torch.is_tensor(recv_probs) or recv_probs.dim() != 2:
        raise RuntimeError("EP chunk DeepEP recv_probs must be a rank-2 tensor")
    if recv_probs.size(0) != recv_hidden.size(0):
        raise RuntimeError("EP chunk recv_hidden and recv_probs rows must match")
    if recv_probs.size(1) != profile.topk:
        raise RuntimeError(f"Recv top-k {recv_probs.size(1)} != profile {profile.topk}")
    profile.validate_rows(recv_probs.size(0), "recv")
    if dispatched.dim() != 2 or dispatched.size(1) != profile.hidden_size:
        raise RuntimeError("Expert input must be rank-2 with the fixed hidden size")
    profile.validate_rows(dispatched.size(0), "expert")


def _expert_activation_output_allocation(
    lease: _EPChunkExpertActivationLease, input_tensor: torch.Tensor
) -> Callable[[str, tuple[int, int]], torch.Tensor]:
    """Build the caller-owned expert-output allocation callback for one input."""

    def allocate(name: str, shape: tuple[int, int]) -> torch.Tensor:
        return lease.tensor(name, shape, dtype=input_tensor.dtype, device=input_tensor.device)

    return allocate


@dataclass(frozen=True)
class EPChunkWorkspaceKey:
    """Cross-layer workspace identity; layer and chunk are deliberately absent."""

    op: EPChunkOpName
    device_type: str
    device_index: int | None
    ep_group_id: int
    dtype: torch.dtype
    shape_profile: EPChunkShapeProfile

    def __post_init__(self) -> None:
        if self.op not in {"forward", "backward", "fused_forward_backward"}:
            raise ValueError(f"Unsupported EP chunk op: {self.op!r}")
        if not isinstance(self.shape_profile, EPChunkShapeProfile):
            raise TypeError("EP chunk shape_profile must be EPChunkShapeProfile")


@dataclass
class _WorkspaceSlot:
    dispatcher: TokenDispatcher | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    in_use: bool = False
    consumer_event: Any | None = None


_EXPERT_ACTIVATION_SIZE_CLASS_BYTES = 8 * 1024 * 1024
_EXPERT_ACTIVATION_LOGICAL_NAMES = frozenset(
    {"fc1_input", "fc1_output", "fc2_output", "fc1_dgrad", "fc2_dgrad"}
)

# Grad-enabled OPs retain FC2 output for delayed wgrad; only the dgrads can alias,
# since SwiGLU consumes FC2 dgrad before FC1 dgrad is written.
_NORMAL_EXPERT_ACTIVATION_STORAGE_SLOTS = {"fc1_dgrad": "fc2_dgrad"}

# No-grad FC2 reads only h after FC1/SwiGLU, so its output can overwrite FC1 input.
# Requires-grad paths must not use this mapping.
_FORWARD_EXPERT_ACTIVATION_STORAGE_SLOTS = {
    **_NORMAL_EXPERT_ACTIVATION_STORAGE_SLOTS,
    "fc2_output": "fc1_input",
}


def _expert_activation_capacity_bytes(requested_bytes: int) -> int:
    """Round an observed activation request to its 8 MiB reuse class."""
    if requested_bytes <= 0:
        raise ValueError("EP chunk expert activation must have positive byte size")
    return (
        (requested_bytes + _EXPERT_ACTIVATION_SIZE_CLASS_BYTES - 1)
        // _EXPERT_ACTIVATION_SIZE_CLASS_BYTES
    ) * _EXPERT_ACTIVATION_SIZE_CLASS_BYTES


@dataclass(frozen=True)
class _EPChunkExpertActivationArenaKey:
    """Physical storage compatibility deliberately excludes the logical op."""

    device_type: str
    device_index: int | None
    ep_group_id: int
    dtype: torch.dtype
    shape_profile: EPChunkShapeProfile


@dataclass
class _EPChunkExpertActivationArenaCoordinator:
    key: _EPChunkExpertActivationArenaKey
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    claimed_op: EPChunkOpName | None = None
    consumer_event: Any | None = None
    allocations: int = 0
    grows: int = 0
    issued_storage_slots: set[str] = field(default_factory=set)
    capacity_bytes: dict[str, int] = field(default_factory=dict)
    logical_trailing_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pointer_signatures_by_op: dict[EPChunkOpName, dict[str, int]] = field(default_factory=dict)
    backing_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    # A graph owns captured allocations through replay. Keep Python backing
    # references only for that lifetime; eager park returns them to the pool.
    graph_backing_owned: bool = False
    frozen: bool = False

    def reserve(self, *, max_expert_rows: int, expert_intermediate_size: int) -> None:
        """Freeze caller-declared capacities; physical backing remains lazy."""
        profile = self.key.shape_profile
        if not 0 < max_expert_rows <= profile.max_expert_rows:
            raise ValueError("EP chunk max_expert_rows must be positive and within profile ceiling")
        if expert_intermediate_size <= 0:
            raise ValueError("EP chunk expert_intermediate_size must be positive")
        capacities = {
            "fc1_input": max_expert_rows * profile.hidden_size,
            "fc1_output": max_expert_rows * 2 * expert_intermediate_size,
            "fc2_output": max_expert_rows * profile.hidden_size,
            # Normal-mode dgrads alias: reserve max(SwiGLU intermediate, FC1 hidden).
            "fc2_dgrad": max_expert_rows * max(profile.hidden_size, expert_intermediate_size),
        }
        itemsize = torch.empty((), dtype=self.key.dtype).element_size()
        capacities = {
            name: _expert_activation_capacity_bytes(elements * itemsize)
            for name, elements in capacities.items()
        }
        if self.frozen:
            if capacities != self.capacity_bytes:
                raise RuntimeError("EP chunk expert activation reservation is already frozen")
            return
        self.capacity_bytes = capacities
        self.frozen = True

    def _wait_for_consumer(self, stream):
        _wait_for_consumer_event(self.consumer_event, stream)
        self.consumer_event = None

    def acquire(self, *, op: EPChunkOpName, stream: Any | None) -> None:
        if self.claimed_op is not None:
            raise RuntimeError(f"Activation arena is already claimed by {self.claimed_op}")
        self._wait_for_consumer(stream)
        self.claimed_op = op
        self.issued_storage_slots.clear()

    def release(self, *, op: EPChunkOpName, event: Any) -> None:
        if self.claimed_op != op:
            raise RuntimeError("EP chunk expert activation coordinator release lost owner")
        if self.frozen and self.graph_backing_owned:
            self._record_pointer_signature(
                op, {name: tensor.data_ptr() for name, tensor in self.tensors.items()}
            )
        self.consumer_event = event
        self.claimed_op = None

    def _record_pointer_signature(self, op: EPChunkOpName, current: dict[str, int]) -> None:
        """Reject moved frozen backing, while allowing later lazy slot creation."""
        previous = self.pointer_signatures_by_op.setdefault(op, {})
        changed = {
            name: (previous[name], pointer)
            for name, pointer in current.items()
            if name in previous and previous[name] != pointer
        }
        if changed:
            details = ", ".join(
                f"{name}: previous={old} current={new}"
                for name, (old, new) in sorted(changed.items())
            )
            raise RuntimeError(f"Frozen activation pointers changed after rehydrate: {details}")
        previous.update(current)

    def park(self, *, stream: Any | None) -> None:
        """Event-wait and park eager backing; retain capacity and graph backing until close."""
        if self.claimed_op is not None:
            raise RuntimeError("Cannot park a leased EP chunk expert activation arena")
        self._wait_for_consumer(stream)
        if stream is not None and not self.graph_backing_owned:
            # Record the reset stream after its consumer wait, before dropping eager
            # references, to order default-allocator reuse.
            for tensor in self.tensors.values():
                if tensor.is_cuda:
                    tensor.record_stream(stream)
        self.tensors.clear()
        if not self.graph_backing_owned:
            self.backing_tensors.clear()
            # Eager re-acquisition may receive a new allocator address; stability is
            # a CUDA graph capture/replay contract only.
            self.pointer_signatures_by_op.clear()

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        storage_name: str | None = None,
    ) -> torch.Tensor:
        requested = tuple(int(dim) for dim in shape)
        if name not in _EXPERT_ACTIVATION_LOGICAL_NAMES:
            raise RuntimeError(f"Unknown EP chunk expert activation {name!r}")
        profile = self.key.shape_profile
        ceiling = (
            (profile.max_expert_rows, profile.hidden_size)
            if name == "fc1_input"
            else (profile.max_expert_rows, *requested[1:])
        )
        if (
            not requested
            or any(dim < 0 for dim in requested)
            or len(requested) != 2
            or requested[0] > profile.max_expert_rows
            or dtype != self.key.dtype
            or (name == "fc1_input" and requested[1:] != ceiling[1:])
        ):
            raise RuntimeError(
                f"EP chunk expert activation {name!r} shape {requested} dtype {dtype} "
                f"exceeds profile ceiling {ceiling} dtype {self.key.dtype}"
            )
        storage_name = name if storage_name is None else storage_name
        trailing_shape = requested[1:]
        previous_trailing_shape = self.logical_trailing_shapes.get(name)
        if previous_trailing_shape is not None and previous_trailing_shape != trailing_shape:
            raise RuntimeError(
                f"Activation {name!r} trailing shape {trailing_shape} != {previous_trailing_shape}"
            )
        existing = self.tensors.get(storage_name)
        requested_numel = math.prod(requested)
        element_size = int(torch.empty((), dtype=dtype).element_size())
        requested_bytes = requested_numel * element_size
        incompatible = existing is not None and (
            existing.dtype != dtype or existing.device != torch.device(device)
        )
        if incompatible:
            raise RuntimeError(f"Activation {name!r} incompatible with storage {storage_name!r}")
        if requested_numel == 0:
            # A zero-token expert is a logical view, not a request for the
            # allocator's positive-size reuse class.  In particular, a parked
            # frozen reservation must not rehydrate merely to represent it.
            if previous_trailing_shape is None:
                self.logical_trailing_shapes[name] = trailing_shape
            if existing is not None:
                return existing.narrow(0, 0, 0).view(requested).detach()
            return torch.empty(requested, dtype=dtype, device=device).detach()
        reserved_capacity_bytes = self.capacity_bytes.get(storage_name, 0)
        if (
            self.frozen
            and self.key.device_type == "cuda"
            and torch.cuda.is_current_stream_capturing()
        ):
            # The outer graph manager owns graph_pool_handle and replay lifetime.
            self.graph_backing_owned = True
        if existing is None and self.frozen and self.graph_backing_owned:
            existing = self.backing_tensors.get(storage_name)
            if existing is not None:
                self.tensors[storage_name] = existing
        growing = requested_bytes > reserved_capacity_bytes > 0
        if existing is not None:
            growing = growing or requested_bytes > existing.numel() * existing.element_size()
        if self.frozen and requested_bytes > reserved_capacity_bytes:
            raise RuntimeError(
                f"Frozen EP chunk expert activation {storage_name!r} requested "
                f"{requested_bytes} bytes over reserved {reserved_capacity_bytes}"
            )
        if growing and storage_name in self.issued_storage_slots:
            raise RuntimeError(f"Cannot grow {storage_name!r} during an active activation lease")
        if existing is None or growing:
            requested_capacity_bytes = _expert_activation_capacity_bytes(requested_bytes)
            capacity_bytes = max(reserved_capacity_bytes, requested_capacity_bytes)
            ceiling_numel = math.prod(ceiling)
            ceiling_capacity_bytes = _expert_activation_capacity_bytes(ceiling_numel * element_size)
            capacity_bytes = min(capacity_bytes, ceiling_capacity_bytes)
            capacity_numel = (capacity_bytes + element_size - 1) // element_size
            # MCore supplies graph_pool_handle to the outer torch.cuda.graph;
            # eager allocation likewise uses the default caching allocator.
            existing = torch.empty((capacity_numel,), dtype=dtype, device=device)
            self.tensors[storage_name] = existing
            if self.frozen and self.graph_backing_owned:
                self.backing_tensors[storage_name] = existing
            self.allocations += int(reserved_capacity_bytes == 0)
            self.grows += int(growing)
            self.capacity_bytes[storage_name] = capacity_bytes
        if previous_trailing_shape is None:
            self.logical_trailing_shapes[name] = trailing_shape
        self.issued_storage_slots.add(storage_name)
        return existing.narrow(0, 0, requested_numel).view(requested).detach()


class _EPChunkExpertActivationLease:
    def __init__(self, workspace: "EPChunkWorkspace"):
        self.workspace = workspace
        self._active = True

    def check_active(self):
        if not self._active:
            raise RuntimeError("EP chunk expert activation lease has already been released")

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Lease stable storage through delayed TE wgrad; release with its completion event."""
        self.check_active()
        # Acquire already waited for the previous consumer; grow only to observed demand.
        op = self.workspace.key.op
        if op == "forward":
            storage_slots = _FORWARD_EXPERT_ACTIVATION_STORAGE_SLOTS
        else:
            storage_slots = _NORMAL_EXPERT_ACTIVATION_STORAGE_SLOTS
        return self.workspace._activation_arena.tensor(
            name, shape, dtype=dtype, device=device, storage_name=storage_slots.get(name, name)
        )

    def release(self, consumer_event: Any) -> None:
        self.check_active()
        if consumer_event is None:
            raise RuntimeError("EP chunk expert activation release requires a consumer event")
        self.workspace._activation_arena.release(op=self.workspace.key.op, event=consumer_event)
        self._active = False


class EPChunkWorkspaceLease:
    def __init__(self, workspace: "EPChunkWorkspace", slot: int, *, require_dispatcher: bool):
        self.workspace = workspace
        self.slot = slot
        self._require_dispatcher = require_dispatcher
        self._active = True

    @property
    def dispatcher(self) -> TokenDispatcher:
        if not self._require_dispatcher:
            raise RuntimeError("EP chunk scratch-only lease has no dispatcher")
        return self.workspace.dispatcher(self.slot)

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        self.check_active()
        requested = tuple(int(dim) for dim in shape)
        self.workspace._validate_runtime_tensor(name, requested, dtype)
        tensor = self.workspace._reserve_tensor(
            self.slot, name, requested, dtype=dtype, device=device
        )
        slices = tuple(slice(0, dim) for dim in requested)
        return tensor[slices].view(requested).detach()

    def check_active(self):
        if not self._active:
            raise RuntimeError("EP chunk workspace lease has already been released")

    def release(self, consumer_event: Any) -> None:
        self.check_active()
        if consumer_event is None:
            raise RuntimeError("EP chunk workspace release requires a consumer event")
        slot = self.workspace._slots[self.slot]
        slot.consumer_event = consumer_event
        slot.in_use = False
        self._active = False


class EPChunkWorkspace:
    """Exactly two stable dispatcher/tensor slots owned by one explicit EP op."""

    def __init__(
        self, key: EPChunkWorkspaceKey, dispatcher_factory: Callable[[int], TokenDispatcher]
    ):
        self.key = key
        self._dispatcher_factory = dispatcher_factory
        self._registry: EPChunkWorkspaceRegistry | None = None
        self._slots = [_WorkspaceSlot() for _ in range(EP_CHUNK_COUNT)]
        self._activation_arena = _EPChunkExpertActivationArenaCoordinator(
            _EPChunkExpertActivationArenaKey(
                key.device_type, key.device_index, key.ep_group_id, key.dtype, key.shape_profile
            )
        )
        self._runtime_allocations = 0
        self._grows = 0
        self._waits = 0
        self._materialized = False
        self._bound_device: torch.device | None = None

    def dispatcher(self, slot: int) -> TokenDispatcher:
        self._validate_slot(slot)
        dispatcher = self._slots[slot].dispatcher
        if dispatcher is None:
            raise RuntimeError("EP chunk workspace is not materialized")
        return dispatcher

    def materialize(self, *, device: torch.device | str | None = None) -> None:
        """Create this op's dispatchers on its bound runtime device."""
        profile_device = self.prepare_scratch(device=device)
        if all(slot.dispatcher is not None for slot in self._slots):
            return
        device_context = (
            torch.cuda.device(profile_device) if self.key.device_type == "cuda" else nullcontext()
        )
        with device_context:
            dispatchers = [self._dispatcher_factory(slot) for slot in range(EP_CHUNK_COUNT)]
        if len({id(dispatcher) for dispatcher in dispatchers}) != EP_CHUNK_COUNT:
            raise RuntimeError("EP chunk workspace requires two distinct dispatchers")
        for chunk_idx, dispatcher in enumerate(dispatchers):
            if hasattr(dispatcher, "use_deepep") and not dispatcher.use_deepep:
                raise RuntimeError(f"EP chunk dispatcher {chunk_idx} has DeepEP disabled")
        for chunk_idx, dispatcher in enumerate(dispatchers):
            self._slots[chunk_idx].dispatcher = dispatcher
        self._materialized = True

    def prepare_scratch(self, *, device: torch.device | str | None = None) -> torch.device:
        """Bind scratch ownership without constructing a dispatcher."""
        if self._materialized:
            self._validate_bound_device(device)
            if self._bound_device is None:
                raise RuntimeError("Materialized EP chunk workspace has no bound device")
            return self._bound_device
        if self._registry is not None:
            self._registry._claim(self)
        profile_device = self._resolve_materialize_device(device)
        self._bound_device = profile_device
        self._materialized = True
        return profile_device

    def reserve_expert_activations(
        self,
        *,
        max_expert_rows: int,
        expert_intermediate_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Freeze capacity: eager reset keeps sizes, graphs keep backing, close drops both."""
        self.prepare_scratch(device=device)
        intermediate = expert_intermediate_size
        if intermediate is None:
            intermediate = self.key.shape_profile.expert_intermediate_size
        if intermediate is None:
            raise ValueError("EP chunk reservation requires expert_intermediate_size")
        self._activation_arena.reserve(
            max_expert_rows=max_expert_rows, expert_intermediate_size=intermediate
        )

    def acquire(
        self, slot: int, *, stream: Any | None = None, require_dispatcher: bool = True
    ) -> EPChunkWorkspaceLease:
        self._validate_slot(slot)
        runtime_device = getattr(stream, "device", None)
        if require_dispatcher:
            self.materialize(device=runtime_device)
        else:
            self.prepare_scratch(device=runtime_device)
        state = self._slots[slot]
        if state.in_use:
            raise RuntimeError(f"EP chunk workspace slot {slot} is already leased")
        self._waits += int(_wait_for_consumer_event(state.consumer_event, stream))
        state.consumer_event = None
        state.in_use = True
        return EPChunkWorkspaceLease(self, slot, require_dispatcher=require_dispatcher)

    def acquire_expert_activation(
        self, *, stream: Any | None = None
    ) -> _EPChunkExpertActivationLease:
        self.prepare_scratch(device=getattr(stream, "device", None))
        self._activation_arena.acquire(op=self.key.op, stream=stream)
        return _EPChunkExpertActivationLease(self)

    def close(self, *, stream: Any | None = None) -> None:
        """Release resident state without a device-wide synchronization."""
        if not self._materialized:
            return
        self._prepare_slots_for_reset(stream=stream, operation="close")
        for slot in self._slots:
            slot.dispatcher = None
        self._runtime_allocations = 0
        self._grows = 0
        self._waits = 0
        self._bound_device = None
        self._materialized = False

    def reset_tensors(self, *, stream: Any | None = None) -> None:
        """Park activations and drop slot scratch without releasing DeepEP state."""
        if not self._materialized:
            return
        self._prepare_slots_for_reset(stream=stream, operation="reset tensors")

    def park_expert_activations(self, *, stream: Any | None = None) -> None:
        """Park after fused outputs are ready; preserve slots, DeepEP dispatchers and scratch."""
        if not self._materialized:
            return
        self._activation_arena.park(stream=stream)

    def _prepare_slots_for_reset(self, *, stream: Any | None, operation: str) -> None:
        coordinator = self._activation_arena
        if coordinator.claimed_op is not None:
            raise RuntimeError(f"Cannot {operation}: activation arena is leased")
        for slot_idx, slot in enumerate(self._slots):
            if slot.in_use:
                raise RuntimeError(f"Cannot {operation}: slot {slot_idx} is leased")
        activation_event = coordinator.consumer_event
        if _event_pending(activation_event) and not (
            (stream is not None and hasattr(stream, "wait_event"))
            or hasattr(activation_event, "current_stream_wait")
        ):
            raise RuntimeError(f"Cannot {operation}: pending activation event")
        pending_slots = [slot for slot in self._slots if _event_pending(slot.consumer_event)]
        if pending_slots and (stream is None or not hasattr(stream, "wait_event")):
            raise RuntimeError(f"Cannot {operation}: pending consumer event")

        coordinator.park(stream=stream)
        for slot in pending_slots:
            stream.wait_event(slot.consumer_event)
            for tensor in slot.tensors.values():
                if tensor.is_cuda:
                    tensor.record_stream(stream)
        for slot in self._slots:
            slot.consumer_event = None
            slot.tensors.clear()

    def _validate_runtime_tensor(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        profile = self.key.shape_profile
        if name != "grad_expert_out":
            return
        capacity, expected_dtype = (profile.max_expert_rows, profile.hidden_size), self.key.dtype
        if (
            len(shape) != len(capacity)
            or any(want > have for want, have in zip(shape, capacity, strict=True))
            or shape[1:] != capacity[1:]
            or dtype != expected_dtype
        ):
            raise RuntimeError(f"{name!r}: {shape}/{dtype} exceeds {capacity}/{expected_dtype}")

    def _reserve_tensor(
        self,
        slot: int,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        self._validate_slot(slot)
        requested = tuple(int(dim) for dim in shape)
        if not requested or any(dim < 0 for dim in requested):
            raise ValueError("EP chunk workspace tensor shape must be non-negative")
        existing = self._slots[slot].tensors.get(name)
        if existing is not None:
            capacity = tuple(existing.shape)
            if (
                existing.dtype != dtype
                or existing.device != torch.device(device)
                or len(capacity) != len(requested)
            ):
                raise RuntimeError(f"{name!r}: shape {requested} exceeds capacity {capacity}")
            if all(want <= have for want, have in zip(requested, capacity, strict=True)):
                return existing
        growing = existing is not None
        existing = torch.empty(requested, dtype=dtype, device=device)
        self._slots[slot].tensors[name] = existing
        self._runtime_allocations += 1
        self._grows += int(growing)
        return existing

    def _resolve_materialize_device(self, device: torch.device | str | None) -> torch.device:
        key_device = (
            torch.device(self.key.device_type)
            if self.key.device_index is None
            else torch.device(self.key.device_type, self.key.device_index)
        )
        requested = key_device if device is None else torch.device(device)
        if requested.type != self.key.device_type:
            raise RuntimeError(f"Device {requested} != workspace type {self.key.device_type}")
        if self.key.device_index is not None and requested.index != self.key.device_index:
            raise RuntimeError(f"Device {requested} != workspace key {key_device}")
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        return requested

    def _validate_bound_device(self, device: torch.device | str | None) -> None:
        if device is None:
            return
        requested = torch.device(device)
        if requested.type == "cuda" and requested.index is None:
            requested = torch.device("cuda", torch.cuda.current_device())
        if requested != self._bound_device:
            raise RuntimeError(f"Workspace bound to {self._bound_device}, not {requested}")

    @staticmethod
    def _validate_slot(slot: int) -> None:
        if not isinstance(slot, int) or not 0 <= slot < EP_CHUNK_COUNT:
            raise IndexError(f"EP chunk slot must be 0 or 1, got {slot!r}")


class EPChunkWorkspaceRegistry:
    def __init__(self):
        self._workspaces: dict[EPChunkWorkspaceKey, EPChunkWorkspace] = {}
        self._expert_activation_arenas: dict[
            _EPChunkExpertActivationArenaKey, _EPChunkExpertActivationArenaCoordinator
        ] = {}

    def get_or_create(
        self, key: EPChunkWorkspaceKey, dispatcher_factory: Callable[[int], TokenDispatcher]
    ) -> EPChunkWorkspace:
        workspace = self._workspaces.get(key)
        if workspace is None:
            workspace = EPChunkWorkspace(key, dispatcher_factory)
            workspace._registry = self
            arena = workspace._activation_arena
            workspace._activation_arena = self._expert_activation_arenas.setdefault(
                arena.key, arena
            )
            self._workspaces[key] = workspace
        return workspace

    def _claim(self, workspace: EPChunkWorkspace) -> None:
        current = self._workspaces.get(workspace.key)
        if current is not None and current is not workspace:
            raise RuntimeError("Cannot rematerialize: workspace key was reused")
        arena_key = workspace._activation_arena.key
        current_arena = self._expert_activation_arenas.get(arena_key)
        if current_arena is None:
            self._expert_activation_arenas[arena_key] = workspace._activation_arena
        elif current_arena is not workspace._activation_arena:
            raise RuntimeError("Cannot rematerialize: replaced activation arena")
        self._workspaces[workspace.key] = workspace

    def release(self, key: EPChunkWorkspaceKey, *, stream: Any | None = None) -> None:
        """Close and unregister one workspace; missing keys are idempotent."""
        workspace = self._workspaces.get(key)
        if workspace is None:
            return
        if workspace._activation_arena.claimed_op == key.op:
            raise RuntimeError("Cannot release leased EP chunk expert activation owner")
        workspace.close(stream=stream)
        if self._workspaces.get(key) is workspace:
            del self._workspaces[key]
        coordinator = workspace._activation_arena
        if not any(
            candidate._activation_arena is coordinator for candidate in self._workspaces.values()
        ):
            event = coordinator.consumer_event
            if _event_pending(event):
                if stream is None or not hasattr(stream, "wait_event"):
                    raise RuntimeError("Cannot release activation arena with pending event")
                stream.wait_event(event)
            if coordinator.claimed_op is not None:
                raise RuntimeError("Cannot release leased EP chunk expert activation arena")
            self._expert_activation_arenas.pop(coordinator.key, None)
            coordinator.consumer_event = None
            coordinator.tensors.clear()
            coordinator.backing_tensors.clear()
            coordinator.capacity_bytes.clear()
            coordinator.logical_trailing_shapes.clear()
            coordinator.pointer_signatures_by_op.clear()
            coordinator.graph_backing_owned = False
            coordinator.frozen = False


_EP_CHUNK_WORKSPACES = EPChunkWorkspaceRegistry()


def get_ep_chunk_workspace(
    key: EPChunkWorkspaceKey, dispatcher_factory: Callable[[int], TokenDispatcher]
) -> EPChunkWorkspace:
    """Return the process-local workspace shared by every matching model layer."""
    return _EP_CHUNK_WORKSPACES.get_or_create(key, dispatcher_factory)


def release_ep_chunk_workspace(key: EPChunkWorkspaceKey, *, stream: Any | None = None) -> None:
    """Close and unregister a process-local EP chunk workspace."""
    _EP_CHUNK_WORKSPACES.release(key, stream=stream)


def _make_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    if not torch.cuda.is_available():
        raise RuntimeError("EP chunk overlap requires CUDA streams.")
    return torch.cuda.Stream(device=device)


_EP_CHUNK_STREAMS: dict[tuple[int, str], torch.cuda.Stream] = {}


def _cuda_device_index(device: torch.device | int | str) -> int:
    if isinstance(device, int):
        return device
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise RuntimeError("EP chunk overlap requires CUDA tensors.")
    return torch.cuda.current_device() if cuda_device.index is None else cuda_device.index


def _shared_stream(device: torch.device | int | str, kind: Literal["comm", "wgrad"]):
    key = (_cuda_device_index(device), kind)
    if key not in _EP_CHUNK_STREAMS:
        _EP_CHUNK_STREAMS[key] = _make_stream(key[0])
    return _EP_CHUNK_STREAMS[key]


def _queue_backward_stream_wait(event: torch.cuda.Event, device: torch.device) -> None:
    """Make work queued after backward wait for deferred expert wgrad."""

    def wait_for_wgrad() -> None:
        with torch.cuda.device(device):
            torch.cuda.current_stream(device).wait_event(event)

    torch.autograd.Variable._execution_engine.queue_callback(wait_for_wgrad)


def _event_current_stream_wait(event: Any) -> None:
    if event is None:
        return
    if hasattr(event, "current_stream_wait"):
        event.current_stream_wait()
    else:
        torch.cuda.current_stream().wait_event(event)


def _record_state_tensors_current_stream(state: dict[str, Any]) -> None:
    for value in state.values():
        if torch.is_tensor(value) and value.is_cuda:
            value.record_stream(torch.cuda.current_stream(value.device))
        elif isinstance(value, dict):
            _record_state_tensors_current_stream(value)


@dataclass
class _ChunkContext:
    idx: int
    start: int
    end: int
    x: torch.Tensor
    scores: torch.Tensor | None
    handle: Any
    row_id_map: torch.Tensor
    prob_flat_indices: torch.Tensor
    recv_hidden_shape: torch.Size
    recv_hidden_dtype: torch.dtype
    recv_probs_shape: torch.Size
    recv_probs_dtype: torch.dtype
    recv_probs_base: torch.Tensor | None
    dispatched: torch.Tensor | None
    probs: torch.Tensor | None
    expert_out: torch.Tensor | None
    dispatcher: TokenDispatcher
    scores_edge: Any | None = None
    scores_shape: torch.Size | None = None
    scores_dtype: torch.dtype | None = None
    expert_out_edge: Any | None = None
    expert_out_shape: torch.Size | None = None
    expert_out_dtype: torch.dtype | None = None

    @classmethod
    def from_dispatch(cls, state, metadata, scores, expert_out, *, retain_output=False, **kwargs):
        scores_edge = torch.autograd.graph.get_gradient_edge(scores)
        expert_out_edge = torch.autograd.graph.get_gradient_edge(expert_out)
        return cls(
            **kwargs,
            scores=None,
            handle=state["handle"],
            row_id_map=metadata["manual_row_id_map"].detach(),
            prob_flat_indices=metadata["manual_prob_flat_indices"].detach(),
            recv_hidden_shape=state["recv_hidden"].shape,
            recv_hidden_dtype=state["recv_hidden"].dtype,
            recv_probs_shape=state["recv_probs"].shape,
            recv_probs_dtype=state["recv_probs"].dtype,
            recv_probs_base=state["recv_probs"],
            expert_out=expert_out if retain_output else None,
            scores_edge=scores_edge,
            scores_shape=scores.shape,
            scores_dtype=scores.dtype,
            expert_out_edge=expert_out_edge,
            expert_out_shape=expert_out.shape,
            expert_out_dtype=expert_out.dtype,
        )


@dataclass(kw_only=True)
class _BackwardChunk(_ChunkContext):
    workspace_lease: EPChunkWorkspaceLease


@dataclass(kw_only=True)
class _ForwardChunkContext(_ChunkContext):
    recv_consumed_event: Any


@dataclass
class _SavedForwardContext:
    chunks: list[_ForwardChunkContext]
    input_shape: torch.Size


class _EPChunkOperationBase:
    """Shared schedule mechanics; each public operation owns its own workspace."""

    def __init__(self, *, router: nn.Module, experts: Experts, workspace: EPChunkWorkspace):
        self.router = router
        self.experts = experts
        self.workspace = workspace

    def _streams(self, device: torch.device) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        return torch.cuda.current_stream(device), _shared_stream(device, "comm")

    @property
    def _logical_chunk_count(self) -> int:
        return self.workspace.key.shape_profile.chunk_count

    def _forward_streams(self, x_2d, ranges):
        self.workspace.key.shape_profile.validate_input(x_2d)
        if len(ranges) != self._logical_chunk_count:
            raise RuntimeError("EP chunk overlap ranges do not match the shape profile")
        compute_stream, comm_stream = self._streams(x_2d.device)
        return compute_stream, comm_stream, torch.cuda.current_stream(x_2d.device)

    def _forward_output_async(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        compute_stream, comm_stream, caller_stream = self._forward_streams(x_2d, ranges)

        def finish_dispatch(pending):
            chunk_idx, _, _, _, _, dispatcher, state, lease = pending
            with torch.cuda.stream(compute_stream):
                dispatched, tpe, probs = dispatcher.finish_deepep_dispatch(state)
                _validate_finished_deepep_dispatch(
                    self.workspace.key.shape_profile, state, dispatched
                )
            return chunk_idx, dispatcher, state, lease, dispatched, tpe, probs

        def run_expert(finished):
            chunk_idx, dispatcher, state, lease, dispatched, tpe, probs = finished
            with torch.cuda.stream(compute_stream):
                expert_activation_lease = self.workspace.acquire_expert_activation(
                    stream=compute_stream
                )
                fc1_input = expert_activation_lease.tensor(
                    "fc1_input", dispatched.shape, dtype=dispatched.dtype, device=dispatched.device
                )
                fc1_input.copy_(dispatched)
                expert_out = self.experts(
                    fc1_input,
                    tpe,
                    probs,
                    tokens_per_expert_list=getattr(dispatcher, "_local_tpe_list", None),
                    output_allocation=_expert_activation_output_allocation(
                        expert_activation_lease, fc1_input
                    ),
                )
                expert_ready = compute_stream.record_event()
                expert_activation_lease.release(expert_ready)
                _record_state_tensors_current_stream(state)
                state.pop("recv_hidden", None)
                state.pop("recv_indices", None)
                state.pop("recv_probs", None)
                state.pop("recv_per_expert", None)
                rank_grouped, handle = dispatcher.prepare_deepep_combine(expert_out)
                ready = compute_stream.record_event()
            del dispatched, probs, expert_out
            return chunk_idx, dispatcher, rank_grouped, handle, ready, lease

        with torch.no_grad():
            output_2d = self._run_forward_pipeline(
                x_2d,
                ranges,
                compute_stream,
                caller_stream,
                comm_stream,
                finish_dispatch,
                run_expert,
            )
        return output_2d.view(input_shape).to(input_dtype).detach()

    def _finish_backward_dispatch(self, dispatcher, state, **kwargs):
        state["recv_hidden"] = state["recv_hidden"].detach().requires_grad_(True)
        state["recv_probs"] = state["recv_probs"].detach().requires_grad_(True)
        dispatched, local_tpe, probs, metadata = dispatcher.finish_deepep_dispatch_for_backward(
            state, **kwargs
        )
        _validate_finished_deepep_dispatch(self.workspace.key.shape_profile, state, dispatched)
        return dispatched, local_tpe, probs, metadata

    def _forward_saved_context_async(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, _SavedForwardContext]:
        """Run the overlapped forward once and retain its graph for backward."""
        compute_stream, comm_stream, caller_stream = self._forward_streams(x_2d, ranges)
        saved_chunks: list[_ForwardChunkContext | None] = [None for _ in ranges]

        def finish_dispatch(pending):
            chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease = pending
            with torch.cuda.stream(compute_stream):
                dispatched, local_tpe, probs, metadata = self._finish_backward_dispatch(
                    dispatcher, state
                )
                expert_input = dispatched.detach().requires_grad_(True)
                expert_probs = None if probs is None else probs.detach().requires_grad_(True)
            return pending, expert_input, expert_probs, local_tpe, metadata

        def run_expert(finished):
            pending, expert_input, expert_probs, local_tpe, metadata = finished
            chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease = pending
            with torch.cuda.stream(compute_stream):
                expert_out = self.experts(
                    expert_input,
                    local_tpe,
                    expert_probs,
                    tokens_per_expert_list=metadata["local_tpe_list"],
                )
                _record_state_tensors_current_stream(state)
                row_id_map = metadata["manual_row_id_map"]
                prob_flat_indices = metadata["manual_prob_flat_indices"]
                if row_id_map is None or prob_flat_indices is None:
                    raise RuntimeError("EP chunk saved forward requires manual backward metadata")
                rank_grouped = unpermute(
                    expert_out,
                    row_id_map,
                    restore_shape=state["recv_hidden"].shape,
                    fused=dispatcher.moe_permute_fusion,
                )
                ready = compute_stream.record_event()
                recv_consumed_event = compute_stream.record_event()

                saved_chunks[chunk_idx] = _ForwardChunkContext.from_dispatch(
                    state,
                    metadata,
                    scores,
                    expert_out,
                    idx=chunk_idx,
                    start=start,
                    end=end,
                    x=x_chunk,
                    dispatched=expert_input,
                    probs=expert_probs,
                    dispatcher=dispatcher,
                    recv_consumed_event=recv_consumed_event,
                )
                state.clear()
            return (
                chunk_idx,
                dispatcher,
                rank_grouped,
                saved_chunks[chunk_idx].handle,
                ready,
                lease,
            )

        output_2d = self._run_forward_pipeline(
            x_2d, ranges, compute_stream, caller_stream, comm_stream, finish_dispatch, run_expert
        )
        if any(chunk is None for chunk in saved_chunks):
            raise RuntimeError("EP chunk saved forward context is incomplete")
        context = _SavedForwardContext(
            chunks=[chunk for chunk in saved_chunks if chunk is not None], input_shape=input_shape
        )
        return output_2d.view(input_shape).to(input_dtype).detach(), context

    def _run_forward_pipeline(
        self, x_2d, ranges, compute_stream, caller_stream, comm_stream, finish_dispatch, run_expert
    ):
        """Shared two-slot schedule; expert callbacks own saved-context policy."""
        input_ready = caller_stream.record_event()

        def submit_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end]
            lease = self.workspace.acquire(chunk_idx % EP_CHUNK_COUNT, stream=comm_stream)
            dispatcher = lease.dispatcher
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(input_ready)
                scores, indices = self.router(x_chunk)
                route_ready = compute_stream.record_event()
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(route_ready)
                if x_chunk.is_cuda:
                    x_chunk.record_stream(comm_stream)
                if scores.is_cuda:
                    scores.record_stream(comm_stream)
                if indices.is_cuda:
                    indices.record_stream(comm_stream)
                lease.check_active()
                state = dispatcher.submit_deepep_dispatch(
                    x_chunk, scores, indices, allocate_on_comm_stream=True
                )
            return chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease

        def submit_combine(prepared):
            chunk_idx, dispatcher, rank_grouped, handle, ready, lease = prepared
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                combine_state = dispatcher.submit_deepep_combine_prepared(
                    rank_grouped, handle, allocate_on_comm_stream=True
                )
            return chunk_idx, dispatcher, combine_state, lease

        output_2d = x_2d.new_empty(x_2d.shape)

        def finish_combine(pending) -> None:
            chunk_idx, dispatcher, state, lease = pending
            chunk_out = dispatcher.finish_deepep_combine(state)
            start, end = ranges[chunk_idx]
            output_2d[start:end].copy_(chunk_out)
            consumed = torch.cuda.current_stream(output_2d.device).record_event()
            lease.release(consumed)

        current_state = submit_dispatch(0)
        pending_combine = None
        for loop_idx in range(len(ranges)):
            finished = finish_dispatch(current_state)
            if loop_idx + 1 < len(ranges):
                if pending_combine is not None:
                    next_slot = (loop_idx + 1) % EP_CHUNK_COUNT
                    pending_slot = pending_combine[0] % EP_CHUNK_COUNT
                    if next_slot == pending_slot:
                        finish_combine(pending_combine)
                        pending_combine = None
                current_state = submit_dispatch(loop_idx + 1)
            prepared = run_expert(finished)
            # Release the permute result before the next dispatch allocates another.
            del finished
            if pending_combine is not None:
                finish_combine(pending_combine)
            pending_combine = submit_combine(prepared)

        done = compute_stream.record_event()
        caller_stream.wait_event(done)
        if pending_combine is None:
            raise RuntimeError("EP chunk combine pipeline produced no pending output")
        finish_combine(pending_combine)
        return output_2d

    def _full_recompute_fused_backward(self, x_2d: torch.Tensor, grad_2d: torch.Tensor):
        ranges = runtime_ep_chunk_ranges(x_2d.size(0), chunk_count=self._logical_chunk_count)
        router_params = tuple(self.router.parameters())
        expert_params = tuple(self.experts.parameters())
        compute_stream, comm_stream = self._streams(grad_2d.device)
        wgrad_stream = _shared_stream(grad_2d.device, "wgrad")
        input_ready = torch.cuda.current_stream(grad_2d.device).record_event()
        grad_x_chunks: list[torch.Tensor | None] = [None for _ in ranges]
        router_accum: list[Any] = [None for _ in router_params]
        pending_dispatch_bwd: list[tuple[_BackwardChunk, dict[str, Any]]] = []
        last_deepep_event: Any | None = None
        last_wgrad_done: torch.cuda.Event | None = None

        def remember_deepep_event(state: dict[str, Any]):
            nonlocal last_deepep_event
            last_deepep_event = state.get("event")
            return state

        def submit_recompute_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end].detach().requires_grad_(True)
            lease = self.workspace.acquire(chunk_idx % EP_CHUNK_COUNT, stream=comm_stream)
            dispatcher = lease.dispatcher
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(input_ready)
                scores, indices = self.router(x_chunk)
                router_ready = compute_stream.record_event()
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(router_ready)
                _event_current_stream_wait(last_deepep_event)
                lease.check_active()
                state = remember_deepep_event(
                    dispatcher.submit_deepep_dispatch(
                        x_chunk, scores, indices, allocate_on_comm_stream=True
                    )
                )
            return chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease

        def submit_combine_bwd(start: int, end: int, dispatcher: TokenDispatcher, handle: Any):
            with torch.cuda.stream(comm_stream):
                grad_chunk = grad_2d[start:end].contiguous()
                _event_current_stream_wait(last_deepep_event)
                return remember_deepep_event(
                    dispatcher.submit_deepep_combine_backward(
                        grad_chunk, handle, allocate_on_comm_stream=True
                    )
                )

        def finish_recompute_expert(dispatcher: TokenDispatcher, state: dict[str, Any]):
            with torch.cuda.stream(compute_stream):
                expert_activation_lease = self.workspace.acquire_expert_activation(
                    stream=compute_stream
                )
                expert_input, local_tpe, probs, metadata = self._finish_backward_dispatch(
                    dispatcher,
                    state,
                    output_allocation=_expert_activation_output_allocation(
                        expert_activation_lease, state["recv_hidden"]
                    ),
                )
                expert_input.requires_grad_(True)
                expert_probs = None if probs is None else probs.detach().requires_grad_(True)
                expert_out = self.experts(
                    expert_input,
                    local_tpe,
                    expert_probs,
                    tokens_per_expert_list=metadata["local_tpe_list"],
                    output_allocation=_expert_activation_output_allocation(
                        expert_activation_lease, expert_input
                    ),
                )
                _record_state_tensors_current_stream(state)
            return metadata, expert_input, expert_probs, expert_out, expert_activation_lease

        def retire_pending_dispatch_bwd() -> None:
            """Retire wgrad/dispatch leases before FC1/SwiGLU; allow an in-flight DeepEP receive."""
            if len(pending_dispatch_bwd) > 1:
                raise RuntimeError("EP chunk fused backward retained more than one pending chunk")
            if not pending_dispatch_bwd:
                return
            chunk, local_state = pending_dispatch_bwd.pop()
            with torch.cuda.stream(compute_stream):
                grad_hidden, grad_scores = chunk.dispatcher.finish_deepep_dispatch_backward(
                    local_state["dispatch_bwd_state"]
                )
                grad_x_chunks[chunk.idx] = _backward_router(
                    chunk, grad_hidden, grad_scores, router_params, router_accum
                )
                chunk.scores = None
                chunk.scores_edge = None
                consumed = compute_stream.record_event()
                chunk.workspace_lease.release(consumed)
                local_state.clear()

        with torch.enable_grad():
            next_state = submit_recompute_dispatch(len(ranges) - 1)
            for rev_idx in range(len(ranges) - 1, -1, -1):
                (chunk_idx, start, end, x_chunk, scores, dispatcher, state, workspace_lease) = (
                    next_state
                )
                # `next_state` was prefetched by the preceding iteration. Retire
                # the prior large context before this chunk can enter FC1/SwiGLU.
                retire_pending_dispatch_bwd()
                combine_state = submit_combine_bwd(start, end, dispatcher, state["handle"])
                (metadata, expert_input, expert_probs, expert_out, expert_activation_lease) = (
                    finish_recompute_expert(dispatcher, state)
                )

                row_id_map = metadata["manual_row_id_map"]
                prob_flat_indices = metadata["manual_prob_flat_indices"]
                if row_id_map is None or prob_flat_indices is None:
                    raise RuntimeError("Fused backward requires manual dgrad metadata")

                chunk = _BackwardChunk.from_dispatch(
                    state,
                    metadata,
                    scores,
                    expert_out,
                    idx=chunk_idx,
                    start=start,
                    end=end,
                    x=x_chunk,
                    dispatched=expert_input,
                    probs=expert_probs,
                    dispatcher=dispatcher,
                    workspace_lease=workspace_lease,
                    retain_output=True,
                )
                state.clear()
                del expert_out, scores
                del expert_input, expert_probs, metadata

                local_state: dict[str, Any] = {}
                with torch.cuda.stream(compute_stream):
                    grad_rank_grouped = dispatcher.finish_deepep_combine_backward(combine_state)
                    combine_state.pop("grad_rank_grouped", None)
                    combine_state.pop("event", None)
                    if chunk.expert_out is None:
                        raise RuntimeError("EP chunk fused backward lost expert output storage.")
                    local_state["grad_expert_out"] = _manual_unpermute_backward(
                        chunk, grad_rank_grouped, out=chunk.expert_out.detach()
                    )
                    del grad_rank_grouped

                if rev_idx > 0:
                    next_state = submit_recompute_dispatch(rev_idx - 1)

                with torch.cuda.stream(compute_stream):
                    # Each fused chunk flushes its aliases, so rebind the TE sink.
                    self.experts._prepare_delayed_weight_grad_sinks()
                    grad_dispatched, grad_probs, hidden_reuse_base = _backward_expert(
                        chunk, local_state.pop("grad_expert_out"), expert_activation_lease
                    )
                    dgrad_ready = compute_stream.record_event()

                with torch.cuda.stream(wgrad_stream):
                    wgrad_stream.wait_event(dgrad_ready)
                    self.experts.flush_delayed_weight_grads(num_contexts=1, stream=wgrad_stream)
                    wgrad_done = wgrad_stream.record_event()
                    for tensor in (
                        grad_dispatched,
                        grad_probs,
                        hidden_reuse_base,
                        chunk.recv_probs_base,
                        chunk.row_id_map,
                        chunk.prob_flat_indices,
                    ):
                        if tensor is not None and tensor.is_cuda:
                            tensor.record_stream(wgrad_stream)
                    grad_recv_hidden, grad_recv_probs = _dispatch_local_backward(
                        chunk, grad_dispatched, grad_probs, hidden_reuse_base=hidden_reuse_base
                    )
                    local_bwd_ready = wgrad_stream.record_event()
                    expert_activation_lease.release(local_bwd_ready)
                    del grad_dispatched, grad_probs
                last_wgrad_done = wgrad_done

                local_state["dispatch_bwd_state"] = remember_deepep_event(
                    _submit_dispatch_backward(
                        chunk,
                        grad_recv_hidden,
                        grad_recv_probs,
                        comm_stream,
                        local_bwd_ready,
                        last_deepep_event,
                    )
                )
                del grad_recv_hidden, grad_recv_probs, hidden_reuse_base

                pending_dispatch_bwd.append((chunk, local_state))
                if len(pending_dispatch_bwd) > 1:
                    raise RuntimeError("EP chunk fused backward pending queue exceeded one chunk")

        if last_wgrad_done is None:
            raise RuntimeError("EP chunk fused backward did not flush expert wgrads")
        _queue_backward_stream_wait(last_wgrad_done, grad_2d.device)

        retire_pending_dispatch_bwd()

        done = compute_stream.record_event()
        torch.cuda.current_stream(grad_2d.device).wait_event(done)

        grad_x = torch.cat(
            [
                torch.zeros_like(x_2d[start:end]) if grad is None else grad
                for (start, end), grad in zip(ranges, grad_x_chunks, strict=True)
            ],
            dim=0,
        ).view_as(grad_2d)
        router_grads_out = _materialize(router_params, router_accum)
        return grad_x, router_grads_out, [None for _ in expert_params]

    def _saved_context_backward(
        self, context: _SavedForwardContext, grad_2d: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
        """Consume saved forward graphs without rerunning router or experts."""
        router_params = tuple(self.router.parameters())
        expert_params = tuple(self.experts.parameters())
        compute_stream, comm_stream = self._streams(grad_2d.device)
        caller_stream = torch.cuda.current_stream(grad_2d.device)
        grad_ready = caller_stream.record_event()
        wgrad_stream = _shared_stream(grad_2d.device, "wgrad")
        grad_x_chunks: list[torch.Tensor | None] = [None for _ in context.chunks]
        router_accum: list[Any] = [None for _ in router_params]
        last_deepep_event: Any | None = grad_ready

        def remember_deepep_event(state: dict[str, Any]):
            nonlocal last_deepep_event
            last_deepep_event = state.get("event")
            return state

        # Retire at most two logical chunks before reusing either physical slot.
        for end in range(len(context.chunks), 0, -EP_CHUNK_COUNT):
            pending_dispatch_bwd: list[tuple[_BackwardChunk, dict[str, Any]]] = []
            self.experts._prepare_delayed_weight_grad_sinks()
            expert_activation_lease = self.workspace.acquire_expert_activation(
                stream=compute_stream
            )
            for saved in reversed(context.chunks[max(0, end - EP_CHUNK_COUNT) : end]):
                lease = self.workspace.acquire(
                    saved.idx % EP_CHUNK_COUNT, stream=comm_stream, require_dispatcher=False
                )
                chunk = _BackwardChunk(
                    **{item.name: getattr(saved, item.name) for item in fields(_ChunkContext)},
                    workspace_lease=lease,
                )
                with torch.cuda.stream(comm_stream):
                    _event_current_stream_wait(last_deepep_event)
                    combine_state = remember_deepep_event(
                        chunk.dispatcher.submit_deepep_combine_backward(
                            grad_2d[chunk.start : chunk.end].contiguous(),
                            chunk.handle,
                            allocate_on_comm_stream=True,
                        )
                    )

                local_state: dict[str, Any] = {}
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(saved.recv_consumed_event)
                    grad_rank_grouped = chunk.dispatcher.finish_deepep_combine_backward(
                        combine_state
                    )
                    grad_dispatched, grad_probs, hidden_reuse_base = _backward_expert(
                        chunk,
                        _manual_unpermute_backward(chunk, grad_rank_grouped),
                        expert_activation_lease,
                    )
                    local_state["hidden_reuse_base"] = hidden_reuse_base
                    local_state["grad_dispatched"] = grad_dispatched
                    local_state["grad_probs"] = grad_probs
                    saved.probs = None
                    saved.expert_out = None
                    saved.expert_out_edge = None
                    saved.dispatched = None
                    del hidden_reuse_base
                pending_dispatch_bwd.append((chunk, local_state))

            wgrad_ready = compute_stream.record_event()
            with torch.cuda.stream(wgrad_stream):
                wgrad_stream.wait_event(wgrad_ready)
                self.experts.flush_delayed_weight_grads(num_contexts=len(pending_dispatch_bwd))
                wgrad_done = wgrad_stream.record_event()
            _queue_backward_stream_wait(wgrad_done, grad_2d.device)

            # Delayed grouped-linear Wgrad retains FC1 input. Do not repurpose its
            # distinct local-scatter destination until that queue drains.
            for chunk, local_state in pending_dispatch_bwd:
                with torch.cuda.stream(compute_stream):
                    compute_stream.wait_event(wgrad_done)
                    hidden_reuse_base = local_state.pop("hidden_reuse_base")
                    grad_recv_hidden, grad_recv_probs = _dispatch_local_backward(
                        chunk,
                        local_state.pop("grad_dispatched"),
                        local_state.pop("grad_probs"),
                        hidden_reuse_base=hidden_reuse_base,
                    )
                    local_ready = compute_stream.record_event()

                local_state["dispatch_bwd_state"] = remember_deepep_event(
                    _submit_dispatch_backward(
                        chunk,
                        grad_recv_hidden,
                        grad_recv_probs,
                        comm_stream,
                        local_ready,
                        last_deepep_event,
                    )
                )
                del grad_recv_hidden, grad_recv_probs, hidden_reuse_base

            with torch.cuda.stream(compute_stream):
                backward_activation_done = compute_stream.record_event()
            expert_activation_lease.release(backward_activation_done)

            for chunk, local_state in pending_dispatch_bwd:
                with torch.cuda.stream(compute_stream):
                    grad_hidden, grad_scores = chunk.dispatcher.finish_deepep_dispatch_backward(
                        local_state["dispatch_bwd_state"]
                    )
                    grad_x_chunks[chunk.idx] = _backward_router(
                        chunk, grad_hidden, grad_scores, router_params, router_accum
                    )
                    consumed = compute_stream.record_event()
                    chunk.workspace_lease.release(consumed)

        done = compute_stream.record_event()
        torch.cuda.current_stream(grad_2d.device).wait_event(done)
        grad_x = torch.cat(
            [
                torch.zeros_like(chunk.x) if grad is None else grad
                for chunk, grad in zip(context.chunks, grad_x_chunks, strict=True)
            ],
            dim=0,
        ).view(context.input_shape)
        return (grad_x, _materialize(router_params, router_accum), [None for _ in expert_params])


def _outer_parameter_grads(params, grads):
    """Publish completed main-grad writes through the outer autograd/DDP edge."""
    return [
        _caller_owned_dummy_wgrad(
            param.main_grad, param, zero=getattr(param, "zero_out_wgrad", False)
        )
        if grad is None and param.requires_grad and getattr(param, "grad_added_to_main_grad", False)
        else grad
        for param, grad in zip(params, grads, strict=True)
    ]


class _SavedContextEPChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_2d: torch.Tensor,
        forward_op: "EPChunkForwardOp",
        input_shape: torch.Size,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.backward_op = forward_op.backward_op
        ctx.input_shape = x_2d.shape
        ctx.params = params
        with torch.enable_grad():
            x_graph = x_2d.detach().requires_grad_(True)
            output, saved_context = forward_op._forward_saved_context_async(
                x_graph,
                runtime_ep_chunk_ranges(
                    x_graph.size(0), chunk_count=forward_op._logical_chunk_count
                ),
                input_shape,
                x_graph.dtype,
            )
        ctx.saved_forward_context = saved_context
        return output.detach()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        with torch.enable_grad():
            grad_x, router_grads, expert_grads = ctx.backward_op.backward(
                ctx.saved_forward_context, grad_output
            )
        return (
            grad_x.reshape(ctx.input_shape),
            None,
            None,
            *_outer_parameter_grads(ctx.params, (*router_grads, *expert_grads)),
        )


class EPChunkForwardOp(_EPChunkOperationBase):
    """Chunked forward with saved-context autograd when gradients are enabled."""

    def __init__(self, *, backward_op: "EPChunkBackwardOp | None" = None, **kwargs):
        super().__init__(**kwargs)
        if backward_op is not None and (
            backward_op.router is not self.router or backward_op.experts is not self.experts
        ):
            raise RuntimeError("Saved-context EP forward/backward must share router and experts")
        self.backward_op = backward_op

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        if torch.is_grad_enabled():
            if self.backward_op is None:
                raise RuntimeError("Grad-enabled EPChunkForwardOp requires a paired backward op")
            params = tuple(self.router.parameters()) + tuple(self.experts.parameters())
            return _SavedContextEPChunkFunction.apply(x_2d, self, input_shape, *params)
        ranges = runtime_ep_chunk_ranges(x_2d.size(0), chunk_count=self._logical_chunk_count)
        return self._forward_output_async(x_2d, ranges, input_shape, x.dtype)

    __call__ = forward


class EPChunkBackwardOp(_EPChunkOperationBase):
    """Consume a saved forward context without rerunning forward compute."""

    def backward(
        self, context: _SavedForwardContext, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
        grad_2d = grad_output.contiguous().view(-1, grad_output.size(-1))
        return self._saved_context_backward(context, grad_2d)


class EPChunkFusedForwardBackwardOp(_EPChunkOperationBase):
    """Explicit recompute-forward plus backward owned by the fused workspace."""

    def forward_backward(
        self, x_saved: torch.Tensor, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
        x_2d = x_saved.view(-1, x_saved.size(-1))
        grad_2d = grad_output.contiguous().view(-1, grad_output.size(-1))
        with torch.enable_grad():
            grad_x, router_grads, expert_grads = self._full_recompute_fused_backward(x_2d, grad_2d)
        grad_x = grad_x.view_as(x_saved)
        return grad_x, router_grads, expert_grads


def _manual_unpermute_backward(
    chunk: _BackwardChunk, grad_rank_grouped: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if chunk.expert_out_shape is None or chunk.expert_out_dtype is None:
        raise RuntimeError("Missing expert output metadata.")
    row_id_map = chunk.row_id_map.reshape(-1).to(torch.long)
    expected_shape = (row_id_map.numel(), grad_rank_grouped.size(1))
    if (
        tuple(chunk.expert_out_shape) != expected_shape
        or grad_rank_grouped.dtype != chunk.expert_out_dtype
    ):
        raise RuntimeError("EP chunk unpermute metadata does not match rank-grouped gradient")
    if out is None:
        grad_expert_out = chunk.workspace_lease.tensor(
            "grad_expert_out",
            chunk.expert_out_shape,
            dtype=chunk.expert_out_dtype,
            device=grad_rank_grouped.device,
        )
    else:
        if (
            out.shape != chunk.expert_out_shape
            or out.dtype != chunk.expert_out_dtype
            or out.device != grad_rank_grouped.device
            or not out.is_contiguous()
        ):
            raise RuntimeError("Unpermute storage must be contiguous and match shape/dtype/device")
        grad_expert_out = out
    with torch.no_grad():
        torch.index_select(grad_rank_grouped.detach(), 0, row_id_map, out=grad_expert_out)
    return grad_expert_out


def _dispatch_local_backward(
    chunk: _BackwardChunk,
    grad_dispatched: torch.Tensor,
    grad_probs: torch.Tensor | None,
    *,
    hidden_reuse_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not grad_dispatched.is_contiguous():
        raise RuntimeError("EP chunk local backward gradient source must be contiguous")
    row_id_map = chunk.row_id_map.reshape(-1).to(torch.long)
    if chunk.recv_probs_base is None:
        raise RuntimeError("EP chunk backward requires retained recv probability storage")
    required_hidden_numel = math.prod(chunk.recv_hidden_shape)
    if (
        hidden_reuse_base.dtype != chunk.recv_hidden_dtype
        or hidden_reuse_base.device != grad_dispatched.device
        or not hidden_reuse_base.is_contiguous()
        or hidden_reuse_base.numel() < required_hidden_numel
    ):
        raise RuntimeError(
            "EP chunk hidden reuse needs contiguous storage, matching dtype/device/capacity"
        )
    grad_recv_hidden = (
        hidden_reuse_base.detach().view(-1)[:required_hidden_numel].view(chunk.recv_hidden_shape)
    )
    if _tensor_byte_ranges_overlap(grad_dispatched, grad_recv_hidden):
        raise RuntimeError("EP chunk local backward source and destination storage overlap")
    grad_recv_probs = chunk.recv_probs_base.detach()
    if (
        grad_recv_probs.shape != chunk.recv_probs_shape
        or grad_recv_probs.dtype != chunk.recv_probs_dtype
        or grad_recv_probs.device != grad_dispatched.device
    ):
        raise RuntimeError("Retained recv probability storage differs from saved metadata")
    grad_recv_hidden.zero_()
    grad_recv_hidden.scatter_add_(
        0,
        row_id_map.unsqueeze(1).expand(-1, grad_dispatched.size(1)),
        grad_dispatched.to(grad_recv_hidden.dtype),
    )
    grad_recv_probs.zero_()
    if grad_probs is not None:
        flat = chunk.prob_flat_indices.reshape(-1).to(grad_probs.device, torch.long)
        grad_recv_probs.reshape(-1).index_copy_(
            0, flat, grad_probs.reshape(-1).to(grad_recv_probs.dtype)
        )
    return grad_recv_hidden, grad_recv_probs


def _backward_expert(chunk, grad_output, lease):
    """Different schedules share local autograd, but own their wgrad flush."""
    output = chunk.expert_out_edge if chunk.expert_out_edge is not None else chunk.expert_out
    dispatched, probs = chunk.dispatched, chunk.probs
    if dispatched is None or output is None:
        raise RuntimeError("EP chunk expert graph was released")
    lease.check_active()
    inputs = (dispatched,) if probs is None else (dispatched, probs)
    grads = torch.autograd.grad(output, inputs, grad_output, allow_unused=True)
    grad_dispatched = grads[0] if grads[0] is not None else torch.zeros_like(dispatched)
    grad_probs = None
    if probs is not None:
        grad_probs = grads[1] if grads[1] is not None else torch.zeros_like(probs)
    # Keep FC1 input until delayed wgrad completes, then reuse it for local scatter.
    hidden_reuse_base = dispatched.detach()
    chunk.dispatched = None
    chunk.probs = None
    chunk.expert_out = None
    chunk.expert_out_edge = None
    return grad_dispatched, grad_probs, hidden_reuse_base


def _backward_router(chunk, grad_hidden, grad_scores, router_params, router_accum):
    """Combine router and expert input gradients after dispatch backward finishes."""
    if grad_scores is None:
        if chunk.scores_shape is None or chunk.scores_dtype is None:
            raise RuntimeError("Missing router score metadata.")
        grad_scores = torch.zeros(
            chunk.scores_shape, device=chunk.x.device, dtype=chunk.scores_dtype
        )
    router_output = chunk.scores_edge if chunk.scores_edge is not None else chunk.scores
    if router_output is None:
        raise RuntimeError("EP chunk overlap router graph was released.")
    if any(hasattr(param, "_capture_wgrad") for param in router_params):
        raise RuntimeError("Router weight-gradient accumulator is already leased")
    captured = [[] for _ in router_params]
    try:
        for param, parts in zip(router_params, captured, strict=True):
            param._capture_wgrad = lambda x, dy, dtype, parts=parts: parts.append(
                (chunk.start, x.detach(), dy.detach().clone(), dtype)
            )
        router_grads = torch.autograd.grad(
            router_output,
            (chunk.x, *router_params),
            grad_scores.to(chunk.scores_dtype),
            allow_unused=True,
        )
    finally:
        for param in router_params:
            if hasattr(param, "_capture_wgrad"):
                del param._capture_wgrad
    grad_score_x = router_grads[0]
    if grad_score_x is None:
        grad_score_x = torch.zeros_like(chunk.x)
    for idx, (param, grad, parts) in enumerate(
        zip(router_params, router_grads[1:], captured, strict=True)
    ):
        if parts:
            if grad is not None:
                raise RuntimeError("Router gradient was both captured and returned")
            if router_accum[idx] is None:
                router_accum[idx] = []
            router_accum[idx].extend(parts)
        elif grad is not None:
            grad = grad.to(param.dtype)
            router_accum[idx] = grad if router_accum[idx] is None else router_accum[idx].add_(grad)
    return grad_hidden.to(chunk.x.dtype) + grad_score_x


def _submit_dispatch_backward(chunk, grad_hidden, grad_probs, stream, ready, previous_event):
    """Submit on the communication stream, retaining inputs until its work finishes."""
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        _event_current_stream_wait(previous_event)
        state = chunk.dispatcher.submit_deepep_dispatch_backward(
            grad_hidden, grad_probs, chunk.handle, allocate_on_comm_stream=True
        )
        for tensor in (grad_hidden, grad_probs):
            if tensor.is_cuda:
                tensor.record_stream(stream)
        chunk.recv_probs_base = None
    return state


def _materialize(params: tuple[torch.Tensor, ...], accum: list[Any]) -> list[torch.Tensor]:
    # Match the unchunked router reduction; expert activations still retire per chunk.
    for idx, (param, parts) in enumerate(zip(params, accum, strict=True)):
        if isinstance(parts, list):
            parts.sort(key=lambda part: part[0])
            dtype = parts[0][3]
            x = torch.cat([part[1] for part in parts]).to(dtype)
            dy = torch.cat([part[2] for part in parts])
            out = (
                _te_general_gemm(x, dy, dtype, layout="NT", grad=True)
                if dtype != torch.float64
                else None
            )
            accum[idx] = (dy.t() @ x if out is None else out[0]).to(param.dtype)
    return [
        torch.zeros_like(param) if grad is None else grad
        for param, grad in zip(params, accum, strict=True)
    ]


class EPChunkExecution:
    """Compose lazy workspaces: retain_backward selects saved fwd+bwd, otherwise fwd+fused."""

    def __init__(
        self,
        *,
        router,
        experts,
        dispatcher_factory,
        max_input_rows,
        hidden_size,
        expert_intermediate_size,
        topk,
        ep_size,
        ep_group,
        chunk_count=2,
        retain_backward=False,
    ):
        profile = EPChunkShapeProfile(
            max_input_rows=max_input_rows,
            hidden_size=hidden_size,
            expert_intermediate_size=expert_intermediate_size,
            topk=topk,
            ep_size=ep_size,
            chunk_count=chunk_count,
        )

        def workspace(op):
            return get_ep_chunk_workspace(
                EPChunkWorkspaceKey(op, "cuda", None, id(ep_group), torch.bfloat16, profile),
                dispatcher_factory,
            )

        kwargs = dict(router=router, experts=experts)
        self.backward_op = (
            EPChunkBackwardOp(workspace=workspace("backward"), **kwargs)
            if retain_backward
            else None
        )
        self.forward_op = EPChunkForwardOp(
            workspace=workspace("forward"), backward_op=self.backward_op, **kwargs
        )
        self.fused_op = (
            None
            if retain_backward
            else EPChunkFusedForwardBackwardOp(
                workspace=workspace("fused_forward_backward"), **kwargs
            )
        )

    def _requirements(self, phase):
        if phase == "forward":
            return self.forward_op.workspace, True
        if phase == "backward":
            op = self.fused_op or self.backward_op
            return op.workspace, self.fused_op is not None
        raise ValueError(f"Unsupported EP chunk workspace phase {phase!r}")

    def materialize(self, *, phase="forward", device=None, expert_activation_max_rows=None):
        workspace, require_dispatcher = self._requirements(phase)
        if require_dispatcher:
            workspace.materialize(device=device)
        else:
            workspace.prepare_scratch(device=device)
        if expert_activation_max_rows is not None:
            workspace.reserve_expert_activations(
                max_expert_rows=expert_activation_max_rows, device=device
            )

    def release(self, *, stream=None):
        for phase in ("forward", "backward"):
            workspace, _ = self._requirements(phase)
            release_ep_chunk_workspace(workspace.key, stream=stream)

    def finish_forward(self, tensor):
        stream = torch.cuda.current_stream(tensor.device) if tensor.is_cuda else None
        self.forward_op.workspace.reset_tensors(stream=stream)

    def finish_backward(self, tensor):
        stream = torch.cuda.current_stream(tensor.device) if tensor.is_cuda else None
        op = self.fused_op or self.backward_op
        op.workspace.park_expert_activations(stream=stream)


class ChunkedMoE(nn.Module):
    """Parameter-owning MoE module over the three policy-free EP operations."""

    def __init__(self, *, router, experts, **execution_kwargs):
        super().__init__()
        self.router = router
        self.experts = experts
        self.chunked_ep = EPChunkExecution(router=router, experts=experts, **execution_kwargs)

    def forward(self, x):
        return self.chunked_ep.forward_op(x)


class _EPChunkCheckpoint(torch.autograd.Function):
    """Checkpoint a caller-owned prefix and residual; model/recompute policy stays outside."""

    @staticmethod
    def forward(ctx, x, prefix, execution, finish_backward, prefix_count, *params):
        ctx.prefix = prefix
        ctx.execution = execution
        ctx.finish_backward = finish_backward
        ctx.prefix_count = prefix_count
        ctx.params = params
        ctx.cpu_rng = torch.get_rng_state()
        ctx.device = x.device if x.is_cuda else None
        ctx.cuda_rng = torch.cuda.get_rng_state(ctx.device) if ctx.device else None
        ctx.save_for_backward(x.detach())
        norm, residual = prefix(x)
        return residual + execution.forward_op(norm)

    @staticmethod
    def backward(ctx, grad_output):
        (saved,) = ctx.saved_tensors
        x = saved.detach().requires_grad_(True)
        prefix_params = ctx.params[: ctx.prefix_count]
        devices = [] if ctx.device is None else [ctx.device]
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.set_rng_state(ctx.cpu_rng)
            if ctx.cuda_rng is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
            norm, residual = ctx.prefix(x)
            grad_norm, router_grads, expert_grads = ctx.execution.fused_op.forward_backward(
                norm, grad_output
            )
            if ctx.finish_backward:
                ctx.execution.finish_backward(x)
            required = tuple(p for p in (x, *prefix_params) if p.requires_grad)
            grads = torch.autograd.grad(
                (norm, residual), required, (grad_norm, grad_output), allow_unused=True
            )
        by_id = {id(p): grad for p, grad in zip(required, grads, strict=True)}
        return (
            by_id.get(id(x)),
            None,
            None,
            None,
            None,
            *_outer_parameter_grads(
                ctx.params,
                (*(by_id.get(id(p)) for p in prefix_params), *router_grads, *expert_grads),
            ),
        )


def checkpoint_ep_chunk(prefix, x, execution, prefix_params, *, finish_backward=False):
    """Use graph-free forward + fused backward without repeating MoE forward."""
    if execution.fused_op is None:
        raise ValueError("EP checkpoint requires a forward/fused composition")
    params = tuple(prefix_params)
    return _EPChunkCheckpoint.apply(
        x,
        prefix,
        execution,
        finish_backward,
        len(params),
        *params,
        *execution.forward_op.router.parameters(),
        *execution.forward_op.experts.parameters(),
    )


__all__ = [
    "ChunkedMoE",
    "checkpoint_ep_chunk",
    "EPChunkExecution",
    "EP_CHUNK_COUNT",
    "EPChunkBackwardOp",
    "EPChunkForwardOp",
    "EPChunkFusedForwardBackwardOp",
    "EPChunkShapeProfile",
    "EPChunkWorkspace",
    "EPChunkWorkspaceKey",
    "EPChunkWorkspaceRegistry",
    "get_ep_chunk_workspace",
    "release_ep_chunk_workspace",
]
