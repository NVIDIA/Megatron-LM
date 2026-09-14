# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepEP MoE EP chunk-overlap primitive."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn as nn

from megatron.lite.primitive.modules.chunked_ep_dispatcher import (
    ChunkedDispatcher as TokenDispatcher,
)
from megatron.lite.primitive.modules.chunked_ep_experts import ChunkedExperts as Experts
from megatron.lite.primitive.modules.moe_ep_chunk_overlap_policy import runtime_ep_chunk_ranges
from megatron.lite.primitive.utils.moe import unpermute

# Physical workspace capacity, deliberately independent of logical chunk count.
EP_CHUNK_COUNT = 2
EPChunkOpName = Literal["forward", "backward", "fused_forward_backward"]


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

    @classmethod
    def for_two_slot_chunked_ep(
        cls,
        *,
        max_input_rows: int,
        hidden_size: int,
        topk: int,
        ep_size: int,
        chunk_count: int = 2,
        expert_intermediate_size: int | None = None,
    ) -> "EPChunkShapeProfile":
        if ep_size <= 1:
            raise ValueError("Two-slot EP chunk profile requires EP > 1")
        if max_input_rows < 2:
            raise ValueError("Two-slot EP chunk profile requires at least two rows")
        return cls(
            max_input_rows=max_input_rows,
            hidden_size=hidden_size,
            topk=topk,
            ep_size=ep_size,
            chunk_count=chunk_count,
            expert_intermediate_size=expert_intermediate_size,
        )

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
        # DeepEP recv storage may contain every source rank's chunk on one
        # destination rank. It stores one hidden row per transported token and
        # represents its local top-k destinations in recv_probs.
        max_recv_rows = max_chunk_rows * self.ep_size
        object.__setattr__(self, "max_recv_rows", max_recv_rows)
        # Manual expert permutation can expand one received token into as many
        # as top-k local expert rows. This is a validation ceiling only: the
        # workspace still reserves the observed runtime shape lazily.
        object.__setattr__(self, "max_expert_rows", max_recv_rows * self.topk)

    def validate_input_rows(self, rows: int) -> None:
        if rows > self.max_input_rows:
            raise RuntimeError(
                f"EP chunk input rows {rows} exceeds two-slot profile "
                f"capacity {self.max_input_rows}"
            )

    def validate_recv_rows(self, rows: int) -> None:
        """Validate DeepEP rows received by one destination rank for one chunk."""
        if rows > self.max_recv_rows:
            raise RuntimeError(
                f"EP chunk recv rows {rows} exceeds two-slot profile capacity {self.max_recv_rows}"
            )

    def validate_expert_rows(self, rows: int) -> None:
        """Validate rows after one received token expands to local experts."""
        if rows > self.max_expert_rows:
            raise RuntimeError(
                f"EP chunk expert rows {rows} exceeds two-slot profile "
                f"capacity {self.max_expert_rows}"
            )

    def validate_input(self, value: torch.Tensor) -> None:
        if value.size(-1) != self.hidden_size:
            raise RuntimeError(
                f"EP chunk hidden size {value.size(-1)} does not match two-slot "
                f"profile {self.hidden_size}"
            )
        self.validate_input_rows(value.numel() // self.hidden_size)


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
    profile.validate_recv_rows(recv_hidden.size(0))
    if not torch.is_tensor(recv_probs) or recv_probs.dim() != 2:
        raise RuntimeError("EP chunk DeepEP recv_probs must be a rank-2 tensor")
    if recv_probs.size(0) != recv_hidden.size(0):
        raise RuntimeError("EP chunk recv_hidden and recv_probs rows must match")
    if recv_probs.size(1) != profile.topk:
        raise RuntimeError(
            f"EP chunk recv_probs top-k {recv_probs.size(1)} does not match "
            f"fixed profile {profile.topk}"
        )
    profile.validate_recv_rows(recv_probs.size(0))
    if dispatched.dim() != 2 or dispatched.size(1) != profile.hidden_size:
        raise RuntimeError(
            "EP chunk dispatched expert input must be rank-2 with the fixed hidden size"
        )
    profile.validate_expert_rows(dispatched.size(0))


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


@dataclass
class _EPChunkActivationTensorArena:
    """Logical activation views; eager storage uses PyTorch's default allocator."""

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


_EXPERT_ACTIVATION_SIZE_CLASS_BYTES = 8 * 1024 * 1024
_EXPERT_ACTIVATION_LOGICAL_NAMES = frozenset(
    {"fc1_input", "fc1_output", "fc2_output", "fc1_dgrad", "fc2_dgrad"}
)

# Normal saved-context backward keeps FC2 output separate because its delayed
# FC2 Wgrad remains deferred. FC2 dgrad is consumed by SwiGLU before FC1 dgrad
# is written, making that pair the only safe normal-mode raw-byte alias.
_NORMAL_EXPERT_ACTIVATION_STORAGE_SLOTS = {"fc1_dgrad": "fc2_dgrad"}

# Forward runs under no-grad. Experts finishes the FC1/SwiGLU allocation scope
# before FC2 starts, and FC2 reads only h, so its output may overwrite the
# now-dead FC1 input. Requires-grad paths deliberately do not use this mapping.
_FORWARD_EXPERT_ACTIVATION_STORAGE_SLOTS = {
    **_NORMAL_EXPERT_ACTIVATION_STORAGE_SLOTS,
    "fc2_output": "fc1_input",
}

# Fused backward enqueues overlapping FC2 Wgrad on the current stream before
# writing FC2 dgrad, then writes FC1 dgrad after SwiGLU consumes it. Same-stream
# ordering makes the overwrite safe without a CUDA synchronize and permits all
# three logical tensors to use FC2-output raw storage in this OP only.
_FUSED_EXPERT_ACTIVATION_STORAGE_SLOTS = {"fc1_dgrad": "fc2_output", "fc2_dgrad": "fc2_output"}


def _expert_activation_capacity_bytes(requested_bytes: int) -> int:
    """Round an observed activation request to its 8 MiB reuse class."""
    if requested_bytes <= 0:
        raise ValueError("EP chunk expert activation must have positive byte size")
    return (
        (requested_bytes + _EXPERT_ACTIVATION_SIZE_CLASS_BYTES - 1)
        // _EXPERT_ACTIVATION_SIZE_CLASS_BYTES
    ) * _EXPERT_ACTIVATION_SIZE_CLASS_BYTES


@dataclass(frozen=True)
class _EPChunkExpertActivationKey:
    """Physical activation ownership is shared across layers of one EP op."""

    op: EPChunkOpName
    device_type: str
    device_index: int | None
    ep_group_id: int
    dtype: torch.dtype
    shape_profile: EPChunkShapeProfile


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
    arena: _EPChunkActivationTensorArena = field(default_factory=_EPChunkActivationTensorArena)
    claimed_op: EPChunkOpName | None = None
    consumer_event: Any | None = None
    waits: int = 0
    allocations: int = 0
    grows: int = 0
    parks: int = 0
    rehydrates: int = 0
    issued_storage_slots: set[str] = field(default_factory=set)
    max_requested_bytes: dict[str, int] = field(default_factory=dict)
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
            raise ValueError(
                "EP chunk max_expert_rows must be positive and within the profile ceiling"
            )
        if expert_intermediate_size <= 0:
            raise ValueError("EP chunk expert_intermediate_size must be positive")
        capacities = {
            "fc1_input": max_expert_rows * profile.hidden_size,
            "fc1_output": max_expert_rows * 2 * expert_intermediate_size,
            "fc2_output": max_expert_rows * profile.hidden_size,
            # Normal backward aliases FC1 dgrad into FC2 dgrad storage.  The
            # shared raw slot therefore must cover the wider of SwiGLU's
            # intermediate gradient and FC1's hidden-size gradient.
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

    def acquire(self, *, op: EPChunkOpName, stream: Any | None) -> None:
        if self.claimed_op is not None:
            raise RuntimeError(
                f"EP chunk expert activation coordinator is already claimed by {self.claimed_op}"
            )
        event = self.consumer_event
        if event is not None and not (bool(event.query()) if hasattr(event, "query") else False):
            if stream is not None and hasattr(stream, "wait_event"):
                stream.wait_event(event)
            elif hasattr(event, "current_stream_wait"):
                event.current_stream_wait()
            else:
                raise RuntimeError(
                    "Pending EP chunk expert activation event is not stream-waitable"
                )
            self.waits += 1
        self.consumer_event = None
        self.claimed_op = op
        self.issued_storage_slots.clear()

    def release(self, *, op: EPChunkOpName, event: Any) -> None:
        if self.claimed_op != op:
            raise RuntimeError("EP chunk expert activation coordinator release lost owner")
        if self.frozen and self.graph_backing_owned:
            self._record_pointer_signature(
                op, {name: tensor.data_ptr() for name, tensor in self.arena.tensors.items()}
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
            raise RuntimeError(
                "EP chunk expert activation backing pointers changed after frozen rehydrate: "
                f"{details}"
            )
        previous.update(current)

    def park(self, *, stream: Any | None) -> None:
        """Park views after the consumer event without a device-wide sync.

        Eager execution releases physical backing to PyTorch's default caching
        allocator while retaining frozen capacity. CUDA graph capture/replay
        owns backing until explicit close, matching graph-private lifetime.
        """
        if self.claimed_op is not None:
            raise RuntimeError("Cannot park a leased EP chunk expert activation arena")
        event = self.consumer_event
        if event is not None and not (bool(event.query()) if hasattr(event, "query") else False):
            if stream is not None and hasattr(stream, "wait_event"):
                stream.wait_event(event)
            elif hasattr(event, "current_stream_wait"):
                event.current_stream_wait()
            else:
                raise RuntimeError(
                    "Pending EP chunk expert activation event is not stream-waitable"
                )
            self.waits += 1
        self.consumer_event = None
        if stream is not None and not self.graph_backing_owned:
            # The reset stream waits for the final expert consumer. Recording it
            # before dropping the last eager reference keeps default-allocator
            # reuse ordered.
            for tensor in self.arena.tensors.values():
                if tensor.is_cuda:
                    tensor.record_stream(stream)
        if self.arena.tensors:
            self.parks += 1
            self.arena.tensors.clear()
        if not self.graph_backing_owned:
            self.backing_tensors.clear()
            # Eager re-acquisition may receive a new allocator address; stability is
            # a CUDA graph capture/replay contract only.
            self.pointer_signatures_by_op.clear()

    def _is_current_stream_capturing(self) -> bool:
        """Read capture state without taking ownership of a graph pool."""
        return self.key.device_type == "cuda" and bool(torch.cuda.is_current_stream_capturing())

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
                f"EP chunk expert activation {name!r} trailing shape {trailing_shape} "
                f"does not match logical activation trailing shape "
                f"{previous_trailing_shape}"
            )
        existing = self.arena.tensors.get(storage_name)
        requested_numel = 1
        for dim in requested:
            requested_numel *= dim
        element_size = int(torch.empty((), dtype=dtype).element_size())
        requested_bytes = requested_numel * element_size
        incompatible = existing is not None and (
            existing.dtype != dtype or existing.device != torch.device(device)
        )
        if incompatible:
            raise RuntimeError(
                f"EP chunk expert activation {name!r} does not match colored storage "
                f"{storage_name!r}"
            )
        if requested_numel == 0:
            # A zero-token expert is a logical view, not a request for the
            # allocator's positive-size reuse class.  In particular, a parked
            # frozen reservation must not rehydrate merely to represent it.
            if previous_trailing_shape is None:
                self.logical_trailing_shapes[name] = trailing_shape
            self.max_requested_bytes[storage_name] = max(
                self.max_requested_bytes.get(storage_name, 0), 0
            )
            if existing is not None:
                return existing.narrow(0, 0, 0).view(requested).detach()
            return torch.empty(requested, dtype=dtype, device=device).detach()
        reserved_capacity_bytes = self.capacity_bytes.get(storage_name, 0)
        rehydrating = existing is None and reserved_capacity_bytes > 0
        capture_now = self.frozen and self._is_current_stream_capturing()
        if capture_now:
            # The outer graph manager owns graph_pool_handle and replay lifetime.
            self.graph_backing_owned = True
        if existing is None and self.frozen and self.graph_backing_owned:
            existing = self.backing_tensors.get(storage_name)
            if existing is not None:
                self.arena.tensors[storage_name] = existing
        growing = requested_bytes > reserved_capacity_bytes > 0
        if existing is not None:
            growing = growing or requested_bytes > existing.numel() * existing.element_size()
        if self.frozen and requested_bytes > reserved_capacity_bytes:
            raise RuntimeError(
                f"Frozen EP chunk expert activation {storage_name!r} requested "
                f"{requested_bytes} bytes over reserved {reserved_capacity_bytes}"
            )
        if growing and storage_name in self.issued_storage_slots:
            raise RuntimeError(
                "EP chunk expert activation cannot grow colored storage "
                f"{storage_name!r} during an active lease"
            )
        if existing is None or growing:
            requested_capacity_bytes = _expert_activation_capacity_bytes(requested_bytes)
            capacity_bytes = max(reserved_capacity_bytes, requested_capacity_bytes)
            ceiling_numel = 1
            for dim in ceiling:
                ceiling_numel *= dim
            ceiling_capacity_bytes = _expert_activation_capacity_bytes(ceiling_numel * element_size)
            capacity_bytes = min(capacity_bytes, ceiling_capacity_bytes)
            capacity_numel = (capacity_bytes + element_size - 1) // element_size
            # MCore supplies graph_pool_handle to the outer torch.cuda.graph;
            # eager allocation likewise uses the default caching allocator.
            existing = torch.empty((capacity_numel,), dtype=dtype, device=device)
            self.arena.tensors[storage_name] = existing
            if self.frozen and self.graph_backing_owned:
                self.backing_tensors[storage_name] = existing
            self.allocations += int(reserved_capacity_bytes == 0)
            self.grows += int(growing)
            self.rehydrates += int(rehydrating)
            self.capacity_bytes[storage_name] = capacity_bytes
        if previous_trailing_shape is None:
            self.logical_trailing_shapes[name] = trailing_shape
        self.max_requested_bytes[storage_name] = max(
            self.max_requested_bytes.get(storage_name, 0), requested_bytes
        )
        self.issued_storage_slots.add(storage_name)
        return existing.narrow(0, 0, requested_numel).view(requested).detach()


@dataclass
class _EPChunkLogicalExpertActivationOwner:
    """Per-OP lease identity layered over one profile-compatible arena."""

    key: _EPChunkExpertActivationKey
    coordinator: _EPChunkExpertActivationArenaCoordinator
    in_use: bool = False
    waits: int = 0
    allocations: int = 0
    grows: int = 0

    @property
    def arena(self) -> _EPChunkActivationTensorArena:
        return self.coordinator.arena

    @property
    def consumer_event(self) -> Any | None:
        return self.coordinator.consumer_event

    def acquire(self, *, stream: Any | None) -> None:
        if self.in_use:
            raise RuntimeError("EP chunk expert activation arena is already leased")
        waits_before = self.coordinator.waits
        self.coordinator.acquire(op=self.key.op, stream=stream)
        self.waits += self.coordinator.waits - waits_before
        self.in_use = True

    def release(self, event: Any) -> None:
        if not self.in_use:
            raise RuntimeError("EP chunk expert activation arena is not leased")
        self.coordinator.release(op=self.key.op, event=event)
        self.in_use = False

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
        storage_name: str | None = None,
    ) -> torch.Tensor:
        allocations_before = self.coordinator.allocations
        grows_before = self.coordinator.grows
        tensor = self.coordinator.tensor(
            name, shape, dtype=dtype, device=device, storage_name=storage_name
        )
        self.allocations += self.coordinator.allocations - allocations_before
        self.grows += self.coordinator.grows - grows_before
        return tensor


class _EPChunkExpertActivationLease:
    def __init__(self, workspace: "EPChunkWorkspace"):
        self.workspace = workspace
        self._active = True

    @contextmanager
    def allocate(self):
        if not self._active:
            raise RuntimeError("EP chunk expert activation lease has already been released")
        yield

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Return this lease's stable activation storage.

        The lease must remain live through every delayed TE consumer.  In
        particular, FC1's saved input is reused only after the caller records
        the wgrad completion event in ``release``.
        """
        if not self._active:
            raise RuntimeError("EP chunk expert activation lease has already been released")
        return self.workspace._expert_activation_tensor(name, shape, dtype=dtype, device=device)

    def release(self, consumer_event: Any) -> None:
        if not self._active:
            raise RuntimeError("EP chunk expert activation lease has already been released")
        if consumer_event is None:
            raise RuntimeError("EP chunk expert activation release requires a consumer event")
        self.workspace._expert_activation_owner.release(consumer_event)
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
        if not self._active:
            raise RuntimeError("EP chunk workspace lease has already been released")
        return self.workspace._lease_tensor(self.slot, name, shape, dtype=dtype, device=device)

    @contextmanager
    def deepep_recv_allocation(self):
        """Validate the active lease around a default-allocator DeepEP receive."""
        if not self._active:
            raise RuntimeError("EP chunk workspace lease has already been released")
        yield

    def release(self, consumer_event: Any) -> None:
        if not self._active:
            raise RuntimeError("EP chunk workspace lease has already been released")
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
        activation_key = _EPChunkExpertActivationKey(
            key.op, key.device_type, key.device_index, key.ep_group_id, key.dtype, key.shape_profile
        )
        self._expert_activation_owner = _EPChunkLogicalExpertActivationOwner(
            activation_key,
            _EPChunkExpertActivationArenaCoordinator(
                _EPChunkExpertActivationArenaKey(
                    key.device_type, key.device_index, key.ep_group_id, key.dtype, key.shape_profile
                )
            ),
        )
        self._allocations = 0
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
        if self._materialized:
            self._validate_bound_device(device)
            if all(slot.dispatcher is not None for slot in self._slots):
                return
            profile_device = self._bound_device
            if profile_device is None:
                raise RuntimeError("Materialized EP chunk workspace has no bound device")
        else:
            profile_device = self._bind(device)
        if self.key.device_type == "cuda":
            with torch.cuda.device(profile_device):
                dispatchers = [self._dispatcher_factory(slot) for slot in range(EP_CHUNK_COUNT)]
        else:
            dispatchers = [self._dispatcher_factory(slot) for slot in range(EP_CHUNK_COUNT)]
        if len({id(dispatcher) for dispatcher in dispatchers}) != EP_CHUNK_COUNT:
            raise RuntimeError("EP chunk workspace requires two distinct dispatchers")
        for chunk_idx, dispatcher in enumerate(dispatchers):
            if hasattr(dispatcher, "use_deepep") and not dispatcher.use_deepep:
                raise RuntimeError(f"EP chunk dispatcher {chunk_idx} has DeepEP disabled")
        for chunk_idx, dispatcher in enumerate(dispatchers):
            self._slots[chunk_idx].dispatcher = dispatcher
        self._materialized = True

    def prepare_scratch(self, *, device: torch.device | str | None = None) -> None:
        """Bind scratch ownership without constructing a dispatcher."""
        if self._materialized:
            self._validate_bound_device(device)
            return
        self._bind(device)

    def reserve_expert_activations(
        self,
        *,
        max_expert_rows: int,
        expert_intermediate_size: int | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Declare a bounded activation arena for capture or frozen measurement.

        This is opt-in: absent a caller capacity the arena preserves lazy-growth
        behavior. Frozen capacity survives eager ``reset_tensors``, but physical
        backing survives only CUDA graph capture/replay; close drops both.
        """
        if not self._materialized:
            self._bind(device)
        else:
            self._validate_bound_device(device)
        intermediate = expert_intermediate_size
        if intermediate is None:
            intermediate = self.key.shape_profile.expert_intermediate_size
        if intermediate is None:
            raise ValueError("EP chunk reservation requires expert_intermediate_size")
        self._expert_activation_owner.coordinator.reserve(
            max_expert_rows=max_expert_rows, expert_intermediate_size=intermediate
        )

    def acquire(
        self, slot: int, *, stream: Any | None = None, require_dispatcher: bool = True
    ) -> EPChunkWorkspaceLease:
        self._validate_slot(slot)
        runtime_device = getattr(stream, "device", None)
        if require_dispatcher:
            self.materialize(device=runtime_device)
        elif not self._materialized:
            self._bind(runtime_device)
        else:
            self._validate_bound_device(runtime_device)
        state = self._slots[slot]
        if state.in_use:
            raise RuntimeError(f"EP chunk workspace slot {slot} is already leased")
        event = state.consumer_event
        if event is not None:
            ready = bool(event.query()) if hasattr(event, "query") else False
            if not ready:
                if stream is not None and hasattr(stream, "wait_event"):
                    stream.wait_event(event)
                elif hasattr(event, "current_stream_wait"):
                    event.current_stream_wait()
                else:
                    raise RuntimeError("Pending EP chunk consumer event is not stream-waitable")
                self._waits += 1
        state.consumer_event = None
        state.in_use = True
        return EPChunkWorkspaceLease(self, slot, require_dispatcher=require_dispatcher)

    def acquire_expert_activation(
        self, *, stream: Any | None = None
    ) -> _EPChunkExpertActivationLease:
        runtime_device = getattr(stream, "device", None)
        if not self._materialized:
            self._bind(runtime_device)
        else:
            self._validate_bound_device(runtime_device)
        self._expert_activation_owner.acquire(stream=stream)
        return _EPChunkExpertActivationLease(self)

    def _bind(self, device: torch.device | str | None) -> torch.device:
        """Claim registry identity and bind a runtime device without allocating."""
        if self._registry is not None:
            self._registry._claim(self)
        profile_device = self._resolve_materialize_device(device)
        self._bound_device = profile_device
        self._materialized = True
        return profile_device

    def close(self, *, stream: Any | None = None) -> None:
        """Release resident state without a device-wide synchronization."""
        if not self._materialized:
            return
        self._prepare_slots_for_reset(stream=stream, operation="close")
        for slot in self._slots:
            slot.dispatcher = None
        self._allocations = 0
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
        """Park only the shared expert-activation coordinator.

        This lifecycle boundary deliberately preserves workspace slots, their
        DeepEP dispatchers, and slot scratch.  It is safe after a fused
        backward has materialized its caller-visible outputs.
        """
        if not self._materialized:
            return
        self._expert_activation_owner.coordinator.park(stream=stream)

    def _prepare_slots_for_reset(self, *, stream: Any | None, operation: str) -> None:
        coordinator = self._expert_activation_owner.coordinator
        if coordinator.claimed_op is not None:
            raise RuntimeError(
                f"Cannot {operation} in EP chunk workspace: expert activation arena is leased"
            )
        for slot_idx, slot in enumerate(self._slots):
            if slot.in_use:
                raise RuntimeError(
                    f"Cannot {operation} in EP chunk workspace: slot {slot_idx} is leased"
                )
        activation_event = coordinator.consumer_event
        if (
            activation_event is not None
            and not (
                bool(activation_event.query()) if hasattr(activation_event, "query") else False
            )
            and not (
                (stream is not None and hasattr(stream, "wait_event"))
                or hasattr(activation_event, "current_stream_wait")
            )
        ):
            raise RuntimeError(
                f"Cannot {operation} in EP chunk workspace with a pending expert activation event"
            )
        for slot in self._slots:
            event = slot.consumer_event
            if (
                event is not None
                and not (bool(event.query()) if hasattr(event, "query") else False)
                and (stream is None or not hasattr(stream, "wait_event"))
            ):
                raise RuntimeError(
                    f"Cannot {operation} in EP chunk workspace with a pending consumer event"
                )

        coordinator.park(stream=stream)
        for slot in self._slots:
            event = slot.consumer_event
            if event is not None:
                ready = bool(event.query()) if hasattr(event, "query") else False
                if not ready:
                    if stream is None or not hasattr(stream, "wait_event"):
                        raise RuntimeError(
                            f"Cannot {operation} in EP chunk workspace with a pending "
                            "consumer event"
                        )
                    stream.wait_event(event)
                    for tensor in slot.tensors.values():
                        if tensor.is_cuda:
                            tensor.record_stream(stream)
        for slot in self._slots:
            slot.consumer_event = None
            slot.tensors.clear()

    def release(self, *, stream: Any | None = None) -> None:
        """Idempotent alias for explicit lifecycle callers."""
        self.close(stream=stream)

    def _lease_tensor(
        self,
        slot: int,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        requested = tuple(int(dim) for dim in shape)
        self._validate_runtime_tensor(name, requested, dtype)
        tensor = self._reserve_tensor(slot, name, requested, dtype=dtype, device=device)
        slices = tuple(slice(0, dim) for dim in requested)
        return tensor[slices].view(requested).detach()

    def _expert_activation_tensor(
        self,
        name: str,
        shape: tuple[int, ...] | torch.Size,
        *,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Reserve caller-owned storage at the observed high-watermark.

        ``max_expert_rows`` is an input-validity ceiling, not an allocation
        target: allocating it would turn a sparse routing bound into a large,
        permanently resident activation.  A replacement is safe here because
        ``acquire_expert_activation`` has waited for the preceding consumer
        event before this method can be called.
        """
        requested = tuple(int(dim) for dim in shape)
        if self.key.op == "forward":
            storage_slots = _FORWARD_EXPERT_ACTIVATION_STORAGE_SLOTS
        elif self.key.op == "fused_forward_backward":
            storage_slots = _FUSED_EXPERT_ACTIVATION_STORAGE_SLOTS
        else:
            storage_slots = _NORMAL_EXPERT_ACTIVATION_STORAGE_SLOTS
        return self._expert_activation_owner.tensor(
            name, requested, dtype=dtype, device=device, storage_name=storage_slots.get(name, name)
        )

    def _validate_runtime_tensor(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        profile = self.key.shape_profile
        contracts = {
            "grad_expert_out": ((profile.max_expert_rows, profile.hidden_size), self.key.dtype)
        }
        contract = contracts.get(name)
        if contract is None:
            return
        capacity, expected_dtype = contract
        if (
            len(shape) != len(capacity)
            or any(want > have for want, have in zip(shape, capacity, strict=True))
            or shape[1:] != capacity[1:]
            or dtype != expected_dtype
        ):
            raise RuntimeError(
                f"EP chunk tensor {name!r} shape {shape} dtype {dtype} exceeds fixed "
                f"profile capacity {capacity} dtype {expected_dtype}"
            )

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
        if existing is None:
            existing = torch.empty(requested, dtype=dtype, device=device)
            self._slots[slot].tensors[name] = existing
            self._allocations += 1
            self._runtime_allocations += 1
            return existing
        capacity = tuple(existing.shape)
        incompatible = (
            existing.dtype != dtype
            or existing.device != torch.device(device)
            or len(capacity) != len(requested)
        )
        too_small = not incompatible and any(
            want > have for want, have in zip(requested, capacity, strict=True)
        )
        if too_small:
            existing = torch.empty(requested, dtype=dtype, device=device)
            self._slots[slot].tensors[name] = existing
            self._allocations += 1
            self._runtime_allocations += 1
            self._grows += 1
            return existing
        if incompatible or too_small:
            raise RuntimeError(
                f"EP chunk tensor {name!r} shape {requested} exceeds the fixed "
                f"workspace shape {capacity}"
            )
        return existing

    def metrics(self) -> dict[str, int]:
        return {
            "allocations": self._allocations,
            "runtime_allocations": self._runtime_allocations,
            "waits": self._waits,
            "grows": self._grows,
        }

    def evidence(self) -> dict[str, Any]:
        """Small lifecycle counters; allocator profiling belongs to the caller."""
        arena = self._expert_activation_owner.coordinator
        return {
            **self.metrics(),
            "expert_activation_allocations": arena.allocations,
            "expert_activation_grows": arena.grows,
            "expert_activation_arena_id": id(arena),
            "expert_activation_capacity_bytes": dict(arena.capacity_bytes),
        }

    def _resolve_materialize_device(self, device: torch.device | str | None) -> torch.device:
        key_device = (
            torch.device(self.key.device_type)
            if self.key.device_index is None
            else torch.device(self.key.device_type, self.key.device_index)
        )
        requested = key_device if device is None else torch.device(device)
        if requested.type != self.key.device_type:
            raise RuntimeError(
                f"EP chunk materialize device {requested} does not match workspace "
                f"device type {self.key.device_type}"
            )
        if self.key.device_index is not None and requested.index != self.key.device_index:
            raise RuntimeError(
                f"EP chunk materialize device {requested} does not match workspace "
                f"key device {key_device}"
            )
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
            raise RuntimeError(
                f"EP chunk workspace already materialized on {self._bound_device}, "
                f"cannot use stream/device {requested}"
            )

    @staticmethod
    def _validate_slot(slot: int) -> None:
        if not isinstance(slot, int) or not 0 <= slot < EP_CHUNK_COUNT:
            raise IndexError(f"EP chunk slot must be 0 or 1, got {slot!r}")


class EPChunkWorkspaceRegistry:
    def __init__(self):
        self._workspaces: dict[EPChunkWorkspaceKey, EPChunkWorkspace] = {}
        self._expert_activation_owners: dict[
            _EPChunkExpertActivationKey, _EPChunkLogicalExpertActivationOwner
        ] = {}
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
            activation_key = _EPChunkExpertActivationKey(
                key.op,
                key.device_type,
                key.device_index,
                key.ep_group_id,
                key.dtype,
                key.shape_profile,
            )
            arena_key = _EPChunkExpertActivationArenaKey(
                key.device_type, key.device_index, key.ep_group_id, key.dtype, key.shape_profile
            )
            coordinator = self._expert_activation_arenas.setdefault(
                arena_key, _EPChunkExpertActivationArenaCoordinator(arena_key)
            )
            workspace._expert_activation_owner = self._expert_activation_owners.setdefault(
                activation_key, _EPChunkLogicalExpertActivationOwner(activation_key, coordinator)
            )
            self._workspaces[key] = workspace
        return workspace

    def _claim(self, workspace: EPChunkWorkspace) -> None:
        current = self._workspaces.get(workspace.key)
        if current is not None and current is not workspace:
            raise RuntimeError(
                "Cannot rematerialize an EP chunk workspace after its key was reused"
            )
        activation_key = workspace._expert_activation_owner.key
        current_owner = self._expert_activation_owners.get(activation_key)
        if current_owner is None:
            self._expert_activation_owners[activation_key] = workspace._expert_activation_owner
        elif current_owner is not workspace._expert_activation_owner:
            raise RuntimeError(
                "Cannot rematerialize an EP chunk workspace with a replaced activation owner"
            )
        arena_key = workspace._expert_activation_owner.coordinator.key
        current_arena = self._expert_activation_arenas.get(arena_key)
        if current_arena is None:
            self._expert_activation_arenas[arena_key] = (
                workspace._expert_activation_owner.coordinator
            )
        elif current_arena is not workspace._expert_activation_owner.coordinator:
            raise RuntimeError(
                "Cannot rematerialize an EP chunk workspace with a replaced activation arena"
            )
        self._workspaces[workspace.key] = workspace

    def release(self, key: EPChunkWorkspaceKey, *, stream: Any | None = None) -> None:
        """Close and unregister one workspace; missing keys are idempotent."""
        workspace = self._workspaces.get(key)
        if workspace is None:
            return
        workspace.close(stream=stream)
        if self._workspaces.get(key) is workspace:
            del self._workspaces[key]
        activation_key = workspace._expert_activation_owner.key
        if not any(
            candidate._expert_activation_owner is workspace._expert_activation_owner
            for candidate in self._workspaces.values()
        ):
            owner = self._expert_activation_owners.pop(activation_key, None)
            if owner is not None and owner.in_use:
                raise RuntimeError("Cannot release leased EP chunk expert activation owner")
        coordinator = workspace._expert_activation_owner.coordinator
        if not any(
            candidate._expert_activation_owner.coordinator is coordinator
            for candidate in self._workspaces.values()
        ):
            event = coordinator.consumer_event
            if event is not None and not (
                bool(event.query()) if hasattr(event, "query") else False
            ):
                if stream is None or not hasattr(stream, "wait_event"):
                    raise RuntimeError(
                        "Cannot release EP chunk expert activation arena with pending event"
                    )
                stream.wait_event(event)
            if coordinator.claimed_op is not None:
                raise RuntimeError("Cannot release leased EP chunk expert activation arena")
            self._expert_activation_arenas.pop(coordinator.key, None)
            coordinator.consumer_event = None
            coordinator.arena.tensors.clear()
            coordinator.backing_tensors.clear()
            coordinator.max_requested_bytes.clear()
            coordinator.capacity_bytes.clear()
            coordinator.logical_trailing_shapes.clear()
            coordinator.pointer_signatures_by_op.clear()
            coordinator.graph_backing_owned = False
            coordinator.frozen = False
            coordinator.parks = 0
            coordinator.rehydrates = 0


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
    try:
        return torch.cuda.Stream(device=device)
    except TypeError:
        with torch.cuda.device(device):
            return torch.cuda.Stream()


_EP_CHUNK_SHARED_COMM_STREAMS: dict[int, torch.cuda.Stream] = {}
_EP_CHUNK_WGRAD_STREAMS: dict[int, torch.cuda.Stream] = {}


def _cuda_device_index(device: torch.device | int | str) -> int:
    if isinstance(device, int):
        return device
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise RuntimeError("EP chunk overlap requires CUDA tensors.")
    return torch.cuda.current_device() if cuda_device.index is None else cuda_device.index


def _shared_comm_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    device_index = _cuda_device_index(device)
    stream = _EP_CHUNK_SHARED_COMM_STREAMS.get(device_index)
    if stream is None:
        stream = _make_stream(device_index)
        _EP_CHUNK_SHARED_COMM_STREAMS[device_index] = stream
    return stream


def _shared_wgrad_stream(device: torch.device | int | str) -> torch.cuda.Stream:
    device_index = _cuda_device_index(device)
    stream = _EP_CHUNK_WGRAD_STREAMS.get(device_index)
    if stream is None:
        stream = _make_stream(device_index)
        _EP_CHUNK_WGRAD_STREAMS[device_index] = stream
    return stream


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
class _BackwardChunk:
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
    workspace_lease: EPChunkWorkspaceLease
    scores_edge: Any | None = None
    scores_shape: torch.Size | None = None
    scores_dtype: torch.dtype | None = None
    expert_out_edge: Any | None = None
    expert_out_shape: torch.Size | None = None
    expert_out_dtype: torch.dtype | None = None


@dataclass
class _ForwardChunkContext:
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
    recv_consumed_event: Any
    scores_edge: Any | None = None
    scores_shape: torch.Size | None = None
    scores_dtype: torch.dtype | None = None
    expert_out_edge: Any | None = None
    expert_out_shape: torch.Size | None = None
    expert_out_dtype: torch.dtype | None = None


@dataclass
class _SavedForwardContext:
    chunks: list[_ForwardChunkContext]
    input_shape: torch.Size


class _EPChunkOperationBase:
    """Shared schedule mechanics; each public operation owns its own workspace."""

    def __init__(
        self,
        *,
        router: nn.Module,
        experts: Experts,
        workspace: EPChunkWorkspace,
        router_forward: (
            Callable[
                [nn.Module, torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]
            ]
            | None
        ) = None,
    ):
        self._router_forward = router_forward
        self._active_routing_input: torch.Tensor | None = None
        self.router = router
        self.experts = experts
        self.workspace = workspace

    def _streams(self, device: torch.device) -> tuple[torch.cuda.Stream, torch.cuda.Stream]:
        return torch.cuda.current_stream(device), _shared_comm_stream(device)

    @property
    def _logical_chunk_count(self) -> int:
        return self.workspace.key.shape_profile.chunk_count

    @contextmanager
    def _routing_context(self, routing_input: torch.Tensor | None):
        previous = getattr(self, "_active_routing_input", None)
        self._active_routing_input = routing_input
        try:
            yield
        finally:
            self._active_routing_input = previous

    def _route(
        self, x: torch.Tensor, start: int = 0, end: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        router_forward = getattr(self, "_router_forward", None)
        if router_forward is None:
            return self.router(x)
        routing_input = self._active_routing_input
        if routing_input is not None:
            routing_input = routing_input.reshape(-1)[start:end]
        return router_forward(self.router, x, routing_input)

    def _forward_output_async(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        self.workspace.key.shape_profile.validate_input(x_2d)
        if not ranges:
            return x_2d.new_empty(input_shape).to(input_dtype)
        if len(ranges) != self._logical_chunk_count:
            raise RuntimeError("EP chunk overlap ranges do not match the shape profile")

        compute_stream, comm_stream = self._streams(x_2d.device)
        caller_stream = torch.cuda.current_stream(x_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)

        def submit_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end]
            lease = self.workspace.acquire(chunk_idx % EP_CHUNK_COUNT, stream=comm_stream)
            dispatcher = lease.dispatcher
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(input_ready)
                scores, indices = self._route(x_chunk, start, end)
                route_ready = torch.cuda.Event()
                route_ready.record(compute_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(route_ready)
                if x_chunk.is_cuda:
                    x_chunk.record_stream(comm_stream)
                if scores.is_cuda:
                    scores.record_stream(comm_stream)
                if indices.is_cuda:
                    indices.record_stream(comm_stream)
                with lease.deepep_recv_allocation():
                    state = dispatcher.submit_deepep_dispatch(
                        x_chunk, scores, indices, allocate_on_comm_stream=True, async_finish=True
                    )
            return chunk_idx, dispatcher, state, lease

        def finish_dispatch(pending):
            chunk_idx, dispatcher, state, lease = pending
            with torch.cuda.stream(compute_stream):
                dispatched, tpe, probs = dispatcher.finish_deepep_dispatch(
                    state, materialize_local_tpe=False
                )
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
                recv_hidden = state.get("recv_hidden")
                fc1_input = expert_activation_lease.tensor(
                    "fc1_input", dispatched.shape, dtype=dispatched.dtype, device=dispatched.device
                )
                fc1_input.copy_(dispatched)
                expert_out = self.experts(
                    fc1_input,
                    tpe,
                    probs,
                    tokens_per_expert_list=getattr(dispatcher, "_local_tpe_list", None),
                    activation_allocation=expert_activation_lease.allocate,
                    output_allocation=_expert_activation_output_allocation(
                        expert_activation_lease, fc1_input
                    ),
                )
                expert_ready = torch.cuda.Event()
                expert_ready.record(compute_stream)
                expert_activation_lease.release(expert_ready)
                _record_state_tensors_current_stream(state)
                state.pop("recv_hidden", None)
                state.pop("recv_indices", None)
                state.pop("recv_probs", None)
                state.pop("recv_per_expert", None)
                rank_grouped, handle = dispatcher.prepare_deepep_combine(expert_out)
                ready = torch.cuda.Event()
                ready.record(compute_stream)
            del dispatched, probs, expert_out
            return chunk_idx, dispatcher, rank_grouped, handle, ready, lease

        def submit_combine(prepared):
            chunk_idx, dispatcher, rank_grouped, handle, ready, lease = prepared
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                combine_state = dispatcher.submit_deepep_combine_prepared(
                    rank_grouped, handle, allocate_on_comm_stream=True, async_finish=True
                )
            return chunk_idx, dispatcher, combine_state, lease

        output_2d = x_2d.new_empty(x_2d.shape)

        def finish_combine(pending) -> None:
            chunk_idx, dispatcher, state, lease = pending
            chunk_out = dispatcher.finish_deepep_combine(state)
            start, end = ranges[chunk_idx]
            output_2d[start:end].copy_(chunk_out)
            consumed = torch.cuda.Event()
            consumed.record(torch.cuda.current_stream(output_2d.device))
            lease.release(consumed)

        pending_combine = None
        with torch.no_grad():
            current_state = submit_dispatch(0)
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
                # ``run_expert`` copies the permute result into fc1_input.
                # Do not retain that result through the next iteration's
                # finish_dispatch RHS, which otherwise creates a second large
                # permute allocation before the first can be retired.
                del finished
                if pending_combine is not None:
                    finish_combine(pending_combine)
                pending_combine = submit_combine(prepared)

        done = torch.cuda.Event()
        done.record(compute_stream)
        caller_stream.wait_event(done)
        if pending_combine is None:
            raise RuntimeError("EP chunk combine pipeline produced no pending output")
        finish_combine(pending_combine)
        return output_2d.view(input_shape).to(input_dtype).detach()

    def _forward_saved_context_async(
        self,
        x_2d: torch.Tensor,
        ranges: list[tuple[int, int]],
        input_shape: torch.Size,
        input_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, _SavedForwardContext]:
        """Run the overlapped forward once and retain its graph for backward."""
        self.workspace.key.shape_profile.validate_input(x_2d)
        if len(ranges) != self._logical_chunk_count:
            raise RuntimeError("EP chunk overlap ranges do not match the shape profile")

        compute_stream, comm_stream = self._streams(x_2d.device)
        caller_stream = torch.cuda.current_stream(x_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        saved_chunks: list[_ForwardChunkContext | None] = [None for _ in ranges]

        def submit_dispatch(chunk_idx: int):
            start, end = ranges[chunk_idx]
            x_chunk = x_2d[start:end]
            lease = self.workspace.acquire(chunk_idx % EP_CHUNK_COUNT, stream=comm_stream)
            dispatcher = lease.dispatcher
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(input_ready)
                scores, indices = self._route(x_chunk, start, end)
                route_ready = torch.cuda.Event()
                route_ready.record(compute_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(route_ready)
                if x_chunk.is_cuda:
                    x_chunk.record_stream(comm_stream)
                if scores.is_cuda:
                    scores.record_stream(comm_stream)
                if indices.is_cuda:
                    indices.record_stream(comm_stream)
                with lease.deepep_recv_allocation():
                    state = dispatcher.submit_deepep_dispatch(
                        x_chunk, scores, indices, allocate_on_comm_stream=True, async_finish=True
                    )
            return chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease

        def finish_dispatch(pending):
            chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease = pending
            with torch.cuda.stream(compute_stream):
                state["recv_hidden"] = state["recv_hidden"].detach().requires_grad_(True)
                state["recv_probs"] = state["recv_probs"].detach().requires_grad_(True)
                dispatched, local_tpe, probs, metadata = (
                    dispatcher.finish_deepep_dispatch_external_with_options(
                        state,
                        force_manual_map=True,
                        force_direct_permute=True,
                        materialize_local_tpe=False,
                    )
                )
                _validate_finished_deepep_dispatch(
                    self.workspace.key.shape_profile, state, dispatched
                )
                expert_input = dispatched.detach().requires_grad_(True)
                expert_probs = None if probs is None else probs.detach().requires_grad_(True)
            return (
                chunk_idx,
                start,
                end,
                x_chunk,
                scores,
                dispatcher,
                state,
                lease,
                expert_input,
                expert_probs,
                local_tpe,
                metadata,
            )

        def run_expert(finished):
            (
                chunk_idx,
                start,
                end,
                x_chunk,
                scores,
                dispatcher,
                state,
                lease,
                expert_input,
                expert_probs,
                local_tpe,
                metadata,
            ) = finished
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
                ready = torch.cuda.Event()
                ready.record(compute_stream)
                recv_consumed_event = torch.cuda.Event()
                recv_consumed_event.record(compute_stream)

                scores_edge = None
                scores_ref: torch.Tensor | None = scores
                expert_out_edge = None
                expert_out_ref: torch.Tensor | None = expert_out
                if hasattr(torch.autograd.graph, "get_gradient_edge"):
                    scores_edge = torch.autograd.graph.get_gradient_edge(scores)
                    scores_ref = None
                    expert_out_edge = torch.autograd.graph.get_gradient_edge(expert_out)
                    expert_out_ref = None
                saved_chunks[chunk_idx] = _ForwardChunkContext(
                    idx=chunk_idx,
                    start=start,
                    end=end,
                    x=x_chunk,
                    scores=scores_ref,
                    handle=state["handle"],
                    row_id_map=row_id_map.detach(),
                    prob_flat_indices=prob_flat_indices.detach(),
                    recv_hidden_shape=state["recv_hidden"].shape,
                    recv_hidden_dtype=state["recv_hidden"].dtype,
                    recv_probs_shape=state["recv_probs"].shape,
                    recv_probs_dtype=state["recv_probs"].dtype,
                    recv_probs_base=state["recv_probs"],
                    dispatched=expert_input,
                    probs=expert_probs,
                    expert_out=expert_out_ref,
                    scores_edge=scores_edge,
                    scores_shape=scores.shape,
                    scores_dtype=scores.dtype,
                    expert_out_edge=expert_out_edge,
                    expert_out_shape=expert_out.shape,
                    expert_out_dtype=expert_out.dtype,
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

        def submit_combine(prepared):
            chunk_idx, dispatcher, rank_grouped, handle, ready, lease = prepared
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(ready)
                combine_state = dispatcher.submit_deepep_combine_prepared(
                    rank_grouped, handle, allocate_on_comm_stream=True, async_finish=True
                )
            return chunk_idx, dispatcher, combine_state, lease

        output_2d = x_2d.new_empty(x_2d.shape)

        def finish_combine(pending) -> None:
            chunk_idx, dispatcher, state, lease = pending
            chunk_out = dispatcher.finish_deepep_combine(state)
            start, end = ranges[chunk_idx]
            output_2d[start:end].copy_(chunk_out)
            consumed = torch.cuda.Event()
            consumed.record(torch.cuda.current_stream(output_2d.device))
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
            if pending_combine is not None:
                finish_combine(pending_combine)
            pending_combine = submit_combine(prepared)

        done = torch.cuda.Event()
        done.record(compute_stream)
        caller_stream.wait_event(done)
        if pending_combine is None:
            raise RuntimeError("EP chunk combine pipeline produced no pending output")
        finish_combine(pending_combine)
        if any(chunk is None for chunk in saved_chunks):
            raise RuntimeError("EP chunk saved forward context is incomplete")
        context = _SavedForwardContext(
            chunks=[chunk for chunk in saved_chunks if chunk is not None], input_shape=input_shape
        )
        return output_2d.view(input_shape).to(input_dtype).detach(), context

    def _full_recompute_fused_backward(self, x_2d: torch.Tensor, grad_2d: torch.Tensor):
        ranges = runtime_ep_chunk_ranges(x_2d.size(0), chunk_count=self._logical_chunk_count)
        router_params = tuple(self.router.parameters())
        expert_params = tuple(self.experts.parameters())
        if not ranges:
            return (
                grad_2d.new_zeros(grad_2d.shape),
                [torch.zeros_like(param) for param in router_params],
                [torch.zeros_like(param) for param in expert_params],
            )

        compute_stream, comm_stream = self._streams(grad_2d.device)
        wgrad_stream = _shared_wgrad_stream(grad_2d.device)
        input_ready = torch.cuda.Event()
        input_ready.record(torch.cuda.current_stream(grad_2d.device))
        grad_x_chunks: list[torch.Tensor | None] = [None for _ in ranges]
        router_accum: list[torch.Tensor | None] = [None for _ in router_params]
        pending_dispatch_bwd: list[tuple[_BackwardChunk, dict[str, Any]]] = []
        last_deepep_event: Any | None = None
        last_wgrad_done: torch.cuda.Event | None = None

        def chain_deepep_event() -> None:
            if last_deepep_event is not None:
                _event_current_stream_wait(last_deepep_event)

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
                scores, indices = self._route(x_chunk, start, end)
                router_ready = torch.cuda.Event()
                router_ready.record(compute_stream)
            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(router_ready)
                chain_deepep_event()
                with lease.deepep_recv_allocation():
                    state = remember_deepep_event(
                        dispatcher.submit_deepep_dispatch(
                            x_chunk,
                            scores,
                            indices,
                            allocate_on_comm_stream=True,
                            async_finish=True,
                        )
                    )
            return chunk_idx, start, end, x_chunk, scores, dispatcher, state, lease

        def submit_combine_bwd(
            chunk_idx: int, start: int, end: int, dispatcher: TokenDispatcher, handle: Any
        ):
            with torch.cuda.stream(comm_stream):
                grad_chunk = grad_2d[start:end].contiguous()
                chain_deepep_event()
                return remember_deepep_event(
                    dispatcher.submit_deepep_combine_backward(
                        grad_chunk, handle, allocate_on_comm_stream=True
                    )
                )

        def finish_recompute_expert(
            chunk_idx: int,
            dispatcher: TokenDispatcher,
            state: dict[str, Any],
            workspace_lease: EPChunkWorkspaceLease,
        ):
            with torch.cuda.stream(compute_stream):
                state["recv_hidden"] = state["recv_hidden"].detach().requires_grad_(True)
                state["recv_probs"] = state["recv_probs"].detach().requires_grad_(True)
                dispatched, local_tpe, probs, metadata = (
                    dispatcher.finish_deepep_dispatch_external_with_options(
                        state,
                        force_manual_map=True,
                        force_direct_permute=True,
                        materialize_local_tpe=False,
                    )
                )
                _validate_finished_deepep_dispatch(
                    self.workspace.key.shape_profile, state, dispatched
                )
                expert_probs = None if probs is None else probs.detach().requires_grad_(True)
                expert_activation_lease = self.workspace.acquire_expert_activation(
                    stream=compute_stream
                )
                fc1_input = expert_activation_lease.tensor(
                    "fc1_input", dispatched.shape, dtype=dispatched.dtype, device=dispatched.device
                )
                with torch.no_grad():
                    fc1_input.copy_(dispatched)
                expert_input = fc1_input.requires_grad_(True)
                expert_out = self.experts(
                    expert_input,
                    local_tpe,
                    expert_probs,
                    tokens_per_expert_list=metadata["local_tpe_list"],
                    activation_allocation=expert_activation_lease.allocate,
                    output_allocation=_expert_activation_output_allocation(
                        expert_activation_lease, expert_input
                    ),
                )
                _record_state_tensors_current_stream(state)
            return (
                dispatched,
                local_tpe,
                probs,
                metadata,
                expert_input,
                expert_probs,
                expert_out,
                expert_activation_lease,
            )

        def retire_pending_dispatch_bwd() -> None:
            """Retire the previous chunk before the next large expert activation.

            The next DeepEP receive may already be in flight, but a two-slot
            workspace cannot keep a prior dispatch-backward lease and delayed
            wgrad aliases alive through another FC1/SwiGLU activation.
            """
            _retire_one_fused_dispatch_bwd(
                pending_dispatch_bwd,
                compute_stream=compute_stream,
                grad_2d=grad_2d,
                router_params=router_params,
                grad_x_chunks=grad_x_chunks,
                router_accum=router_accum,
            )

        with torch.enable_grad():
            next_state = submit_recompute_dispatch(len(ranges) - 1)
            for rev_idx in range(len(ranges) - 1, -1, -1):
                (chunk_idx, start, end, x_chunk, scores, dispatcher, state, workspace_lease) = (
                    next_state
                )
                # `next_state` was prefetched by the preceding iteration. Retire
                # the prior large context before this chunk can enter FC1/SwiGLU.
                retire_pending_dispatch_bwd()
                combine_state = submit_combine_bwd(
                    chunk_idx, start, end, dispatcher, state["handle"]
                )
                (
                    dispatched,
                    local_tpe,
                    probs,
                    metadata,
                    expert_input,
                    expert_probs,
                    expert_out,
                    expert_activation_lease,
                ) = finish_recompute_expert(chunk_idx, dispatcher, state, workspace_lease)

                row_id_map = metadata["manual_row_id_map"]
                prob_flat_indices = metadata["manual_prob_flat_indices"]
                if row_id_map is None or prob_flat_indices is None:
                    raise RuntimeError(
                        "EP chunk overlap fused backward requires manual dgrad metadata."
                    )

                scores_edge = None
                scores_ref: torch.Tensor | None = scores
                expert_out_edge = None
                expert_out_ref: torch.Tensor | None = expert_out
                if hasattr(torch.autograd.graph, "get_gradient_edge"):
                    scores_edge = torch.autograd.graph.get_gradient_edge(scores)
                    scores_ref = None
                    expert_out_edge = torch.autograd.graph.get_gradient_edge(expert_out)

                chunk = _BackwardChunk(
                    idx=chunk_idx,
                    start=start,
                    end=end,
                    x=x_chunk,
                    scores=scores_ref,
                    handle=state["handle"],
                    row_id_map=row_id_map.detach(),
                    prob_flat_indices=prob_flat_indices.detach(),
                    recv_hidden_shape=state["recv_hidden"].shape,
                    recv_hidden_dtype=state["recv_hidden"].dtype,
                    recv_probs_shape=state["recv_probs"].shape,
                    recv_probs_dtype=state["recv_probs"].dtype,
                    recv_probs_base=state["recv_probs"],
                    dispatched=expert_input,
                    probs=expert_probs,
                    expert_out=expert_out_ref,
                    scores_edge=scores_edge,
                    scores_shape=scores.shape,
                    scores_dtype=scores.dtype,
                    expert_out_edge=expert_out_edge,
                    expert_out_shape=expert_out.shape,
                    expert_out_dtype=expert_out.dtype,
                    dispatcher=dispatcher,
                    workspace_lease=workspace_lease,
                )
                state.clear()
                del dispatched, probs, expert_out, scores, local_tpe
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
                    expert_output = (
                        chunk.expert_out_edge
                        if chunk.expert_out_edge is not None
                        else chunk.expert_out
                    )
                    expert_dispatched = chunk.dispatched
                    expert_probs_input = chunk.probs
                    if expert_dispatched is None or expert_output is None:
                        raise RuntimeError("EP chunk overlap expert graph was released.")
                    expert_inputs = _expert_grad_inputs(expert_dispatched, expert_probs_input)
                    # Fused mode flushes and releases owned aliases per chunk,
                    # so each delayed TE context must rebind its selected sink.
                    self.experts._prepare_delayed_weight_grad_sinks()
                    with expert_activation_lease.allocate():
                        expert_grads = torch.autograd.grad(
                            expert_output,
                            expert_inputs,
                            local_state["grad_expert_out"],
                            allow_unused=True,
                        )
                    grad_dispatched = expert_grads[0]
                    if grad_dispatched is None:
                        grad_dispatched = torch.zeros_like(expert_dispatched)
                    if expert_probs_input is None:
                        grad_probs = None
                    else:
                        grad_probs = expert_grads[1]
                        if grad_probs is None:
                            grad_probs = torch.zeros_like(expert_probs_input)
                    # In fused mode FC1 dgrad aliases FC2 output, so autograd
                    # can have overwritten grad_expert_out. FC1 input is the
                    # delayed-Wgrad-flush local-scatter destination below.
                    hidden_reuse_base = expert_dispatched.detach()
                    local_state.pop("grad_expert_out")
                    chunk.dispatched = None
                    chunk.probs = None
                    chunk.expert_out = None
                    chunk.expert_out_edge = None
                    del expert_dispatched, expert_probs_input
                    del expert_inputs, expert_grads, expert_output
                    dgrad_ready = torch.cuda.Event()
                    dgrad_ready.record(compute_stream)

                with torch.cuda.stream(wgrad_stream):
                    wgrad_stream.wait_event(dgrad_ready)
                    self.experts.flush_delayed_weight_grads(num_contexts=1, stream=wgrad_stream)
                    wgrad_done = torch.cuda.Event()
                    wgrad_done.record(wgrad_stream)
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
                    local_bwd_ready = torch.cuda.Event()
                    local_bwd_ready.record(wgrad_stream)
                    expert_activation_lease.release(local_bwd_ready)
                    del grad_dispatched, grad_probs
                last_wgrad_done = wgrad_done

                with torch.cuda.stream(comm_stream):
                    comm_stream.wait_event(local_bwd_ready)
                    chain_deepep_event()
                    local_state["dispatch_bwd_state"] = remember_deepep_event(
                        chunk.dispatcher.submit_deepep_dispatch_backward(
                            grad_recv_hidden,
                            grad_recv_probs,
                            chunk.handle,
                            allocate_on_comm_stream=True,
                        )
                    )
                    if grad_recv_hidden.is_cuda:
                        grad_recv_hidden.record_stream(comm_stream)
                    if grad_recv_probs.is_cuda:
                        grad_recv_probs.record_stream(comm_stream)
                    chunk.recv_probs_base = None
                    del grad_recv_hidden, grad_recv_probs, hidden_reuse_base

                pending_dispatch_bwd.append((chunk, local_state))
                if len(pending_dispatch_bwd) > 1:
                    raise RuntimeError("EP chunk fused backward pending queue exceeded one chunk")

        if last_wgrad_done is None:
            raise RuntimeError("EP chunk fused backward did not flush expert wgrads")
        _queue_backward_stream_wait(last_wgrad_done, grad_2d.device)

        retire_pending_dispatch_bwd()

        done = torch.cuda.Event()
        done.record(compute_stream)
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
        grad_ready = torch.cuda.Event()
        grad_ready.record(caller_stream)
        wgrad_stream = _shared_wgrad_stream(grad_2d.device)
        grad_x_chunks: list[torch.Tensor | None] = [None for _ in context.chunks]
        router_accum: list[torch.Tensor | None] = [None for _ in router_params]
        pending_dispatch_bwd: list[tuple[_BackwardChunk, dict[str, Any]]] = []
        last_deepep_event: Any | None = grad_ready
        if context.chunks:
            self.experts._prepare_delayed_weight_grad_sinks()
        expert_activation_lease = self.workspace.acquire_expert_activation(stream=compute_stream)

        def remember_deepep_event(state: dict[str, Any]):
            nonlocal last_deepep_event
            last_deepep_event = state.get("event")
            return state

        for saved in reversed(context.chunks):
            lease = self.workspace.acquire(
                saved.idx % EP_CHUNK_COUNT, stream=comm_stream, require_dispatcher=False
            )
            chunk = _BackwardChunk(
                idx=saved.idx,
                start=saved.start,
                end=saved.end,
                x=saved.x,
                scores=saved.scores,
                handle=saved.handle,
                row_id_map=saved.row_id_map,
                prob_flat_indices=saved.prob_flat_indices,
                recv_hidden_shape=saved.recv_hidden_shape,
                recv_hidden_dtype=saved.recv_hidden_dtype,
                recv_probs_shape=saved.recv_probs_shape,
                recv_probs_dtype=saved.recv_probs_dtype,
                recv_probs_base=saved.recv_probs_base,
                dispatched=saved.dispatched,
                probs=saved.probs,
                expert_out=saved.expert_out,
                scores_edge=saved.scores_edge,
                scores_shape=saved.scores_shape,
                scores_dtype=saved.scores_dtype,
                expert_out_edge=saved.expert_out_edge,
                expert_out_shape=saved.expert_out_shape,
                expert_out_dtype=saved.expert_out_dtype,
                dispatcher=saved.dispatcher,
                workspace_lease=lease,
            )
            with torch.cuda.stream(comm_stream):
                if last_deepep_event is not None:
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
                grad_rank_grouped = chunk.dispatcher.finish_deepep_combine_backward(combine_state)
                local_state["grad_expert_out"] = _manual_unpermute_backward(
                    chunk, grad_rank_grouped
                )
                expert_output = (
                    chunk.expert_out_edge if chunk.expert_out_edge is not None else chunk.expert_out
                )
                if expert_output is None:
                    raise RuntimeError("EP chunk saved expert graph was released")
                dispatched = chunk.dispatched
                probs = chunk.probs
                if dispatched is None:
                    raise RuntimeError("EP chunk saved expert input was released")
                expert_inputs = _expert_grad_inputs(dispatched, probs)
                with expert_activation_lease.allocate():
                    expert_grads = torch.autograd.grad(
                        expert_output,
                        expert_inputs,
                        local_state["grad_expert_out"],
                        allow_unused=True,
                    )
                grad_dispatched = expert_grads[0]
                if grad_dispatched is None:
                    grad_dispatched = torch.zeros_like(dispatched)
                grad_probs = None
                if probs is not None:
                    grad_probs = expert_grads[1]
                    if grad_probs is None:
                        grad_probs = torch.zeros_like(probs)
                # FC1 dgrad aliases FC2 dgrad. Keep FC1 input as the delayed-
                # Wgrad-flush local-scatter destination instead.
                local_state["hidden_reuse_base"] = dispatched.detach()
                local_state.pop("grad_expert_out")
                local_state["grad_dispatched"] = grad_dispatched
                local_state["grad_probs"] = grad_probs
                saved.probs = None
                saved.expert_out = None
                saved.expert_out_edge = None
                saved.dispatched = None
                chunk.dispatched = None
                chunk.probs = None
                chunk.expert_out = None
                chunk.expert_out_edge = None
                del dispatched, probs, expert_inputs, expert_grads, expert_output
            pending_dispatch_bwd.append((chunk, local_state))

        wgrad_ready = torch.cuda.Event()
        wgrad_ready.record(compute_stream)
        with torch.cuda.stream(wgrad_stream):
            wgrad_stream.wait_event(wgrad_ready)
            self.experts.flush_delayed_weight_grads(num_contexts=len(pending_dispatch_bwd))
            wgrad_done = torch.cuda.Event()
            wgrad_done.record(wgrad_stream)
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
                local_ready = torch.cuda.Event()
                local_ready.record(compute_stream)

            with torch.cuda.stream(comm_stream):
                comm_stream.wait_event(local_ready)
                if last_deepep_event is not None:
                    _event_current_stream_wait(last_deepep_event)
                local_state["dispatch_bwd_state"] = remember_deepep_event(
                    chunk.dispatcher.submit_deepep_dispatch_backward(
                        grad_recv_hidden,
                        grad_recv_probs,
                        chunk.handle,
                        allocate_on_comm_stream=True,
                    )
                )
                if grad_recv_hidden.is_cuda:
                    grad_recv_hidden.record_stream(comm_stream)
                if grad_recv_probs.is_cuda:
                    grad_recv_probs.record_stream(comm_stream)
                chunk.recv_probs_base = None
                del grad_recv_hidden, grad_recv_probs, hidden_reuse_base

        with torch.cuda.stream(compute_stream):
            backward_activation_done = torch.cuda.Event()
            backward_activation_done.record(compute_stream)
        expert_activation_lease.release(backward_activation_done)

        for chunk, local_state in pending_dispatch_bwd:
            with torch.cuda.stream(compute_stream):
                grad_hidden, grad_scores = chunk.dispatcher.finish_deepep_dispatch_backward(
                    local_state["dispatch_bwd_state"]
                )
                if grad_scores is None:
                    if chunk.scores_shape is None or chunk.scores_dtype is None:
                        raise RuntimeError("Missing saved router score metadata")
                    grad_scores = torch.zeros(
                        chunk.scores_shape, device=grad_2d.device, dtype=chunk.scores_dtype
                    )
                router_output = chunk.scores_edge if chunk.scores_edge is not None else chunk.scores
                if router_output is None:
                    raise RuntimeError("EP chunk saved router graph was released")
                router_grads = torch.autograd.grad(
                    router_output,
                    (chunk.x, *router_params),
                    grad_scores.to(chunk.scores_dtype),
                    allow_unused=True,
                )
                grad_score_x = router_grads[0]
                if grad_score_x is None:
                    grad_score_x = torch.zeros_like(chunk.x)
                grad_x_chunks[chunk.idx] = grad_hidden.to(chunk.x.dtype) + grad_score_x
                _accumulate(router_accum, router_params, router_grads[1:])
                consumed = torch.cuda.Event()
                consumed.record(compute_stream)
                chunk.workspace_lease.release(consumed)

        done = torch.cuda.Event()
        done.record(compute_stream)
        torch.cuda.current_stream(grad_2d.device).wait_event(done)
        grad_x = torch.cat(
            [
                torch.zeros_like(chunk.x) if grad is None else grad
                for chunk, grad in zip(context.chunks, grad_x_chunks, strict=True)
            ],
            dim=0,
        ).view(context.input_shape)
        return (grad_x, _materialize(router_params, router_accum), [None for _ in expert_params])


class _SavedContextEPChunkFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_2d: torch.Tensor,
        routing_input: torch.Tensor | None,
        forward_op: "EPChunkForwardOp",
        input_shape: torch.Size,
        input_dtype: torch.dtype,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        del params, input_dtype
        ctx.backward_op = forward_op.backward_op
        ctx.num_router_params = len(tuple(forward_op.router.parameters()))
        with torch.enable_grad(), forward_op._routing_context(routing_input):
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
        return grad_x, None, None, None, None, *router_grads, *expert_grads


class EPChunkForwardOp(_EPChunkOperationBase):
    """Two-chunk forward with saved-context autograd when gradients are enabled."""

    def __init__(self, *, backward_op: "EPChunkBackwardOp | None" = None, **kwargs):
        super().__init__(**kwargs)
        if backward_op is not None and (
            backward_op.router is not self.router or backward_op.experts is not self.experts
        ):
            raise RuntimeError("Saved-context EP forward/backward must share router and experts")
        self.backward_op = backward_op

    def forward(self, x: torch.Tensor, routing_input: torch.Tensor | None = None) -> torch.Tensor:
        input_shape = x.shape
        x_2d = x.view(-1, x.size(-1)) if x.dim() == 3 else x
        if torch.is_grad_enabled():
            if self.backward_op is None:
                raise RuntimeError("Grad-enabled EPChunkForwardOp requires a paired backward op")
            params = tuple(self.router.parameters()) + tuple(self.experts.parameters())
            return _SavedContextEPChunkFunction.apply(
                x_2d, routing_input, self, input_shape, x.dtype, *params
            )
        ranges = runtime_ep_chunk_ranges(x_2d.size(0), chunk_count=self._logical_chunk_count)
        with self._routing_context(routing_input):
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
        self,
        x_saved: torch.Tensor,
        grad_output: torch.Tensor,
        routing_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor | None]]:
        x_2d = x_saved.view(-1, x_saved.size(-1))
        grad_2d = grad_output.contiguous().view(-1, grad_output.size(-1))
        with self._routing_context(routing_input), torch.enable_grad():
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
        raise RuntimeError(
            "EP chunk manual unpermute output metadata does not match the rank-grouped gradient"
        )
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
            raise RuntimeError(
                "EP chunk manual unpermute output storage must be contiguous "
                "with matching shape, dtype, and device"
            )
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
            "EP chunk hidden reuse storage must be contiguous with matching "
            "dtype/device and sufficient capacity"
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
        raise RuntimeError(
            "EP chunk retained recv probability storage does not match saved metadata"
        )
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


def _tensor_byte_ranges_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Whether two contiguous tensor views overlap in addressable bytes."""
    if left.numel() == 0 or right.numel() == 0:
        return False
    left_start = left.data_ptr()
    left_end = left_start + left.numel() * left.element_size()
    right_start = right.data_ptr()
    right_end = right_start + right.numel() * right.element_size()
    return left_start < right_end and right_start < left_end


def _expert_grad_inputs(
    dispatched: torch.Tensor, probs: torch.Tensor | None
) -> tuple[torch.Tensor, ...]:
    return (dispatched,) if probs is None else (dispatched, probs)


def _accumulate(
    accum: list[torch.Tensor | None],
    params: tuple[torch.Tensor, ...],
    grads: tuple[torch.Tensor | None, ...],
) -> None:
    for idx, (param, grad) in enumerate(zip(params, grads, strict=True)):
        if grad is None:
            continue
        grad = grad.to(param.dtype)
        if accum[idx] is None:
            accum[idx] = grad
        else:
            accum[idx].add_(grad)


def _retire_one_fused_dispatch_bwd(
    pending: list[tuple[_BackwardChunk, dict[str, Any]]],
    *,
    compute_stream: torch.cuda.Stream,
    grad_2d: torch.Tensor,
    router_params: tuple[torch.Tensor, ...],
    grad_x_chunks: list[torch.Tensor | None],
    router_accum: list[torch.Tensor | None],
) -> None:
    """Finish one dispatched backward chunk and release its slot lease."""
    if len(pending) > 1:
        raise RuntimeError("EP chunk fused backward retained more than one pending chunk")
    if not pending:
        return
    chunk, local_state = pending.pop()
    with torch.cuda.stream(compute_stream):
        grad_hidden, grad_scores = chunk.dispatcher.finish_deepep_dispatch_backward(
            local_state["dispatch_bwd_state"]
        )
        if grad_scores is None:
            if chunk.scores_shape is None or chunk.scores_dtype is None:
                raise RuntimeError("Missing router score metadata.")
            grad_scores = torch.zeros(
                chunk.scores_shape, device=grad_2d.device, dtype=chunk.scores_dtype
            )
        router_output = chunk.scores_edge if chunk.scores_edge is not None else chunk.scores
        if router_output is None:
            raise RuntimeError("EP chunk overlap router graph was released.")
        router_grads = torch.autograd.grad(
            router_output,
            (chunk.x, *router_params),
            grad_scores.to(chunk.scores_dtype),
            allow_unused=True,
        )
        grad_score_x = router_grads[0]
        if grad_score_x is None:
            grad_score_x = torch.zeros_like(chunk.x)
        grad_x_chunks[chunk.idx] = grad_hidden.to(chunk.x.dtype) + grad_score_x
        _accumulate(router_accum, router_params, router_grads[1:])
        chunk.scores = None
        chunk.scores_edge = None
        consumed = torch.cuda.Event()
        consumed.record(compute_stream)
        chunk.workspace_lease.release(consumed)
        local_state.clear()


def _materialize(
    params: tuple[torch.Tensor, ...], accum: list[torch.Tensor | None]
) -> list[torch.Tensor]:
    return [
        torch.zeros_like(param) if grad is None else grad
        for param, grad in zip(params, accum, strict=True)
    ]


class EPChunkExecution:
    """Model-independent OP composition; parameters remain owned by the caller.

    retain_backward selects saved-context forward/backward; otherwise expose
    graph-free forward and fused forward/backward. No recompute policy is read.
    Workspace construction stays lazy and shared through the existing registry.
    """

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
        profile = EPChunkShapeProfile.for_two_slot_chunked_ep(
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
            return ((self.forward_op.workspace, True),)
        if phase == "backward":
            op = self.fused_op or self.backward_op
            return ((op.workspace, self.fused_op is not None),)
        raise ValueError(f"Unsupported EP chunk workspace phase {phase!r}")

    def materialize(self, *, phase="forward", device=None, expert_activation_max_rows=None):
        for workspace, require_dispatcher in self._requirements(phase):
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
            for workspace, _ in self._requirements(phase):
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
    """Checkpoint a tensor prefix followed by residual + ChunkedEP.

    The prefix returns (MoE input, residual). Architecture and recompute policy
    belong to the caller; this bridge never inspects a model or its config.
    """

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
            required = tuple(p for p in (x, *prefix_params) if p.requires_grad)
            grads = torch.autograd.grad(
                (norm, residual), required, (grad_norm, grad_output), allow_unused=True
            )
            if ctx.finish_backward:
                ctx.execution.finish_backward(x)
        by_id = {id(p): grad for p, grad in zip(required, grads, strict=True)}
        return (
            by_id.get(id(x)),
            None,
            None,
            None,
            None,
            *(by_id.get(id(p)) for p in prefix_params),
            *router_grads,
            *expert_grads,
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
