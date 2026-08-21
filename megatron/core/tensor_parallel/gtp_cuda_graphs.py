# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph lifecycle support for Generalized Tensor Parallelism (GTP).

This module owns state that exists only for local CUDA-graph capture and replay:

* capture-local ownership of asynchronous GTP communication;
* persistent storage for work that outlives a local graph handoff;
* routing graph-owned allocations into the shared CUDA-graph memory pool.

``get_graph_persistent_buffer`` serves replay-invariant temporary storage whose complete lifetime
belongs to one backward runner. Eager execution records the required capacity, then capture
receives a stable view from one of a bounded number of alternating arenas. Callers never select
an arena, generation, name, or device domain.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch

_PERSISTENT_BUFFER_ALIGNMENT_BYTES = 256


def _shape_numel(shape: Iterable[int]) -> int:
    """Return the number of elements in ``shape`` after validating it."""
    shape = tuple(shape)
    numel = 1
    for dimension in shape:
        if dimension < 0:
            raise ValueError(f"Persistent buffer shape must be non-negative, got {shape}")
        numel *= dimension
    return numel


def _aligned_numel(numel: int, dtype: torch.dtype) -> int:
    """Round a persistent suballocation to a transport-safe alignment."""
    element_size = torch.empty((), dtype=dtype).element_size()
    alignment = max(1, _PERSISTENT_BUFFER_ALIGNMENT_BYTES // element_size)
    return ((numel + alignment - 1) // alignment) * alignment


# Fixed runner arenas serve requests whose number and shape are replay-invariant. An eager
# backward first measures each runner's per-dtype capacity. The plan then packs those capacities
# into alternating fixed-address arenas, and capture bump-allocates stable views from them.
@dataclass
class GraphPersistentBufferState:
    """Per-runner discovery, capture, and replay state.

    This object does not own GPU storage. ``GraphPersistentBufferPlan`` owns the alternating
    arenas and binds each buffer-using runner to one generation.

    Lifecycle: the runner creates this state, eager backward discovers its required capacity,
    graph capture bump-allocates views from its assigned arena, and replay fences arena reuse.
    """

    # Total aligned elements requested per dtype during the runner's eager discovery backward.
    capacities: dict[torch.dtype, int] = field(default_factory=dict)
    # Capture-order position, used only to identify the runner in allocation errors.
    runner_index: Optional[int] = field(default=None, init=False)
    # Alternating arena selected by the plan; runners without arena requests leave this unset.
    generation: Optional[int] = field(default=None, init=False)
    # Per-dtype bump offsets used while capturing this runner.
    _offsets: dict[torch.dtype, int] = field(default_factory=dict, init=False, repr=False)
    _plan: Optional["GraphPersistentBufferPlan"] = field(default=None, init=False, repr=False)

    def start_discovery(self) -> None:
        """Start a fresh eager capacity-discovery pass for this runner."""
        self.capacities.clear()

    def clear(self) -> None:
        """Drop all state after graphs using this runner have been deleted."""
        self.capacities.clear()
        self._offsets.clear()
        self.runner_index = None
        self.generation = None
        self._plan = None

    def record(self, shape: Iterable[int], dtype: torch.dtype) -> None:
        """Add one aligned request to this runner's discovered capacity."""
        aligned_numel = _aligned_numel(_shape_numel(shape), dtype)
        self.capacities[dtype] = self.capacities.get(dtype, 0) + aligned_numel

    def bind(
        self, plan: "GraphPersistentBufferPlan", runner_index: int, generation: Optional[int]
    ) -> None:
        """Bind discovery results to the shared plan before graph capture."""
        self._plan = plan
        self.runner_index = runner_index
        self.generation = generation

    @property
    def needs_replay_fence(self) -> bool:
        """Return whether replay needs any persistent-buffer fencing."""
        return bool(self.capacities)

    def start_capture(self) -> None:
        """Reset bump-allocation before capture."""
        self._offsets = {dtype: 0 for dtype in self.capacities}

    def allocate(self, shape: Iterable[int], dtype: torch.dtype) -> torch.Tensor:
        """Bump-allocate one stable tensor view from this runner's generation."""
        if self._plan is None or self.generation is None or self.runner_index is None:
            raise RuntimeError(f"Runner {self.runner_index} has no persistent buffer generation")
        shape = tuple(shape)
        numel = _shape_numel(shape)
        aligned_numel = _aligned_numel(numel, dtype)
        consumed = self._offsets.get(dtype, 0)
        capacity = self.capacities.get(dtype, 0)
        if consumed + aligned_numel > capacity:
            raise RuntimeError(
                f"Runner {self.runner_index} exhausted its persistent {dtype} arena: "
                f"requested {aligned_numel} aligned elements after {consumed}, "
                f"capacity {capacity}"
            )
        arena = self._plan.get_arena(self.generation, dtype)
        self._offsets[dtype] = consumed + aligned_numel
        return arena.narrow(0, consumed, numel).view(shape)

    def validate_complete(self) -> None:
        """Require capture to consume exactly the capacity measured during discovery."""
        for dtype, capacity in self.capacities.items():
            consumed = self._offsets.get(dtype, 0)
            if consumed != capacity:
                raise RuntimeError(
                    f"Runner {self.runner_index} consumed {consumed}/{capacity} persistent "
                    f"{dtype} elements during capture"
                )

    def _get_bound_plan(self) -> tuple["GraphPersistentBufferPlan", int]:
        """Return the plan and generation assigned before graph capture."""
        if self._plan is None or self.generation is None:
            raise RuntimeError(f"Runner {self.runner_index} has no persistent buffer generation")
        return self._plan, self.generation

    def wait_for_reuse(self, stream: torch.cuda.Stream) -> None:
        """Wait before overwriting this runner's arena generation."""
        if self.capacities:
            plan, generation = self._get_bound_plan()
            plan.wait_for_reuse(generation, stream)

    def mark_reusable_after(self, stream: torch.cuda.Stream) -> None:
        """Publish the graph tail for the arena generation consumed here."""
        if self.capacities:
            plan, generation = self._get_bound_plan()
            plan.mark_reusable_after(generation, stream)


class GraphPersistentBufferPlan:
    """Own the alternating fixed-address arenas shared by backward runners.

    A generation may back several non-adjacent runners. Its arena is sized to the largest runner
    assigned to it, and its event prevents a later runner from overwriting the arena before the
    previous runner's graph tail has completed. Runner states only retain their generation and
    bump-allocation offsets; all arena tensors live here.

    Lifecycle: after eager discovery completes, one plan binds all runner states and allocates the
    arenas before graph capture. It remains shared across replays and is cleared with the graphs.
    """

    def __init__(
        self,
        states: list[GraphPersistentBufferState],
        capacities: dict[tuple[int, torch.dtype], int],
        generation_count: int,
    ) -> None:
        # Runner states in backward graph-capture order.
        self._states: list[GraphPersistentBufferState] = states
        # Maximum aligned elements needed for each (generation, dtype) arena.
        self._capacities: dict[tuple[int, torch.dtype], int] = capacities
        # Fixed-address backing tensors allocated outside CUDA graph memory pools.
        self._arenas: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        # One completion event per reusable generation.
        self._ready_events: list[torch.cuda.Event] = []
        self._generation_count = generation_count

    @classmethod
    def create(
        cls, states: Iterable[GraphPersistentBufferState], *, max_inflight: int = 3
    ) -> "GraphPersistentBufferPlan":
        """Plan and allocate alternating arenas for the supplied runner states."""
        if max_inflight < 1:
            raise ValueError("max_inflight must be at least 1")

        state_list = list(states)
        buffer_runner_count = sum(bool(state.capacities) for state in state_list)
        generation_count = min(max_inflight, buffer_runner_count)
        capacities: dict[tuple[int, torch.dtype], int] = {}
        assignments = []
        buffer_runner_index = 0

        for state in state_list:
            generation = None
            if state.capacities:
                generation = buffer_runner_index % generation_count
                buffer_runner_index += 1
                for dtype, capacity in state.capacities.items():
                    key = (generation, dtype)
                    capacities[key] = max(capacities.get(key, 0), capacity)
            assignments.append(generation)

        plan = cls(state_list, capacities, generation_count)
        for runner_index, (state, generation) in enumerate(zip(state_list, assignments)):
            state.bind(plan, runner_index, generation)
        plan._allocate()
        return plan

    def _allocate(self) -> None:
        """Allocate every arena outside CUDA graph memory pools on the current device."""
        device = torch.cuda.current_device()
        for key, capacity in self._capacities.items():
            _, dtype = key
            self._arenas[key] = torch.empty(capacity, dtype=dtype, device=device)
        current_stream = torch.cuda.current_stream()
        for _ in range(self._generation_count):
            event = torch.cuda.Event(external=True)
            # Seed the event so every captured runner has the same unconditional reuse fence.
            event.record(current_stream)
            self._ready_events.append(event)

    def get_arena(self, generation: int, dtype: torch.dtype) -> torch.Tensor:
        """Return one generation's typed backing arena."""
        try:
            return self._arenas[(generation, dtype)]
        except KeyError as exc:
            raise RuntimeError(
                f"No persistent arena for generation {generation} and dtype {dtype}"
            ) from exc

    def wait_for_reuse(self, generation: int, stream: torch.cuda.Stream) -> None:
        """Wait until the prior runner using ``generation`` has completed."""
        stream.wait_event(self._ready_events[generation])

    def mark_reusable_after(self, generation: int, stream: torch.cuda.Stream) -> None:
        """Make ``generation`` reusable after ``stream`` reaches this point."""
        self._ready_events[generation].record(stream)

    def clear(self) -> None:
        """Drop persistent arenas after all referencing graphs are deleted."""
        self._arenas.clear()
        self._ready_events.clear()
        self._generation_count = 0
        for state in self._states:
            if state._plan is self:
                state._plan = None
                state.generation = None


_ACTIVE_DISCOVERY_STATE: Optional[GraphPersistentBufferState] = None
_ACTIVE_CAPTURE_STATE: Optional[GraphPersistentBufferState] = None


def set_graph_persistent_buffer_discovery(state: Optional[GraphPersistentBufferState]) -> None:
    """Select the runner whose eager backward is discovering persistent capacity."""
    global _ACTIVE_DISCOVERY_STATE
    _ACTIVE_DISCOVERY_STATE = state


def get_graph_persistent_buffer(shape: Iterable[int], dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Record one eager request or return its stable capture-time tensor view."""
    shape = tuple(shape)
    if _ACTIVE_DISCOVERY_STATE is not None:
        _ACTIVE_DISCOVERY_STATE.record(shape, dtype)
    if _ACTIVE_CAPTURE_STATE is None:
        return None
    return _ACTIVE_CAPTURE_STATE.allocate(shape, dtype)


@contextmanager
def use_graph_persistent_buffer_state(state: GraphPersistentBufferState):
    """Activate one runner's persistent-buffer state during backward graph capture."""
    global _ACTIVE_CAPTURE_STATE
    if _ACTIVE_CAPTURE_STATE is not None:
        raise RuntimeError("Nested persistent buffer states are unsupported")
    state.start_capture()
    _ACTIVE_CAPTURE_STATE = state
    try:
        yield
        state.validate_complete()
    finally:
        _ACTIVE_CAPTURE_STATE = None


@dataclass
class GTPCaptureCommState:
    """Asynchronous GTP work issued while capturing one CUDA graph."""

    params: list = field(default_factory=list)
    finalized_params: list = field(default_factory=list)
    ag_streams: list = field(default_factory=list)
    rs_streams: list = field(default_factory=list)
    _param_ids: set = field(default_factory=set)
    _ag_stream_ids: set = field(default_factory=set)
    _rs_stream_ids: set = field(default_factory=set)

    def register_comm(self, param, stream: torch.cuda.Stream, *, reduce_scatter: bool) -> None:
        """Record a parameter and side stream owned by this graph capture."""
        param_id = id(param)
        if param_id not in self._param_ids:
            self._param_ids.add(param_id)
            self.params.append(param)

        stream_id = id(stream)
        streams = self.rs_streams if reduce_scatter else self.ag_streams
        stream_ids = self._rs_stream_ids if reduce_scatter else self._ag_stream_ids
        if stream_id not in stream_ids:
            stream_ids.add(stream_id)
            streams.append(stream)

    def register_wgrad_finalize(self, param) -> None:
        """Record one DDP grad-ready occurrence produced by captured GTP finalization."""
        # A parameter used repeatedly by MTP must remain repeated here: DDP learns the expected
        # per-parameter ready count from eager execution and replay must reproduce that count.
        self.finalized_params.append(param)


_ACTIVE_CAPTURE_COMM_STATE: Optional[GTPCaptureCommState] = None


def register_capture_comm(param, stream: torch.cuda.Stream, *, reduce_scatter: bool) -> None:
    """Register communication with the active capture, if one exists."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_comm(param, stream, reduce_scatter=reduce_scatter)


def register_capture_wgrad_finalize(param) -> None:
    """Record one GTP wgrad-finalization occurrence with the active capture."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_wgrad_finalize(param)


@contextmanager
def track_gtp_capture_comms():
    """Track asynchronous GTP work owned by one CUDA-graph capture."""
    global _ACTIVE_CAPTURE_COMM_STATE

    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        raise RuntimeError("Nested GTP CUDA-graph communication tracking is unsupported")

    state = GTPCaptureCommState()
    _ACTIVE_CAPTURE_COMM_STATE = state
    try:
        yield state
    finally:
        _ACTIVE_CAPTURE_COMM_STATE = None


_CG_MEMPOOL_DEVICE = None
_CG_MEMPOOL = None


def set_cuda_graph_mempool(device, mempool) -> None:
    """Register the shared memory pool used for graph-owned GTP allocations."""
    global _CG_MEMPOOL_DEVICE, _CG_MEMPOOL
    _CG_MEMPOOL_DEVICE = device
    _CG_MEMPOOL = mempool


@contextmanager
def cuda_graph_pool_allocation(enabled: bool):
    """Route allocations in this context into the registered CUDA-graph pool."""
    if _CG_MEMPOOL is None or not enabled or torch.cuda.is_current_stream_capturing():
        yield
        return

    torch._C._cuda_beginAllocateCurrentThreadToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
    try:
        yield
    finally:
        torch._C._cuda_endAllocateToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
