# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph lifecycle support for Generalized Tensor Parallelism (GTP).

This module owns state that exists only for local CUDA-graph capture and replay:

* capture-local ownership of asynchronous GTP communication;
* persistent storage for work that outlives a local graph handoff;
* routing graph-owned allocations into the shared CUDA-graph memory pool.

``get_graph_persistent_buffer`` serves replay-invariant temporary storage whose complete lifetime
belongs to one backward runner. Eager execution records the required capacity, then capture
receives a stable view from one of a bounded number of alternating arenas. Callers never select
an arena, generation, name, or device. RS send buffers may select their registered symmetric
process-group domain; every other request uses the default allocator domain.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch

from megatron.core.tensor_parallel.gtp_symmetric_memory import gtp_symm_pool_ctx

_PERSISTENT_BUFFER_ALIGNMENT_BYTES = 256


def _persistent_buffer_key(
    dtype: torch.dtype, symmetric_group=None
) -> tuple[Optional[str], torch.dtype]:
    """Return the allocation-domain and dtype key for one persistent request."""
    group_name = symmetric_group.group_name if symmetric_group is not None else None
    return group_name, dtype


def _format_persistent_buffer_key(key: tuple[Optional[str], torch.dtype]) -> str:
    """Format an allocation-domain key for diagnostics."""
    group_name, dtype = key
    domain = "default allocator" if group_name is None else f"symmetric group {group_name}"
    return f"{dtype} in {domain}"


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
# backward first measures each runner's per-domain, per-dtype capacity. The plan then packs those
# capacities into alternating fixed-address arenas, and capture bump-allocates stable views.


@contextmanager
def preserve_gtp_prefetch_state(params: Iterable[torch.nn.Parameter]):
    """Preserve cross-graph AG handoffs while a local graph warms up.

    A local graph's first GTP weight can be prefetched and drained by the preceding forward or
    backward graph. Runner-local warmup consumes that readiness without rerunning the producer,
    and can create outgoing readiness of its own. Capture must therefore see the same forward and
    recompute-forward handoff state that existed before each warmup pass. Completed Work handles
    are intentionally not restored; the producer's external event carries the device dependency.
    """
    prefetch_state = tuple(
        (
            param,
            getattr(param, "_already_ag_drained", False),
            getattr(param, "_recompute_already_drained", False),
        )
        for param in params
        if getattr(param, "is_gtp_weight_remat", False)
    )
    completed = False
    try:
        yield
        completed = True
    finally:
        leaked_params = []
        for param, already_drained, recompute_already_drained in prefetch_state:
            if (
                getattr(param, "_prefetch_handle", None) is not None
                or getattr(param, "_recompute_prefetch_handle", None) is not None
            ):
                leaked_params.append(getattr(param, "_debug_name", "") or f"id={id(param):#x}")
            param._already_ag_drained = already_drained
            param._recompute_already_drained = recompute_already_drained

        # Do not replace an exception from the warmup body with a cleanup failure. On successful
        # warmup, however, a live Work handle would make the restored host handoff state invalid.
        if completed and leaked_params:
            raise RuntimeError("GTP warmup left undrained AG work for: " + ", ".join(leaked_params))


@dataclass
class GraphPersistentBufferState:
    """Per-runner discovery, capture, and replay state.

    This object does not own GPU storage. ``GraphPersistentBufferPlan`` owns the alternating
    arenas and binds each buffer-using runner to one generation.

    Lifecycle: the runner creates this state, eager backward discovers its required capacity,
    graph capture bump-allocates views from its assigned arena, and replay fences arena reuse.
    """

    # Total aligned elements requested per allocation domain and dtype during eager discovery.
    capacities: dict[tuple[Optional[str], torch.dtype], int] = field(default_factory=dict)
    # Capture-order position, used only to identify the runner in allocation errors.
    runner_index: Optional[int] = field(default=None, init=False)
    # Alternating arena selected by the plan; runners without arena requests leave this unset.
    generation: Optional[int] = field(default=None, init=False)
    # Exact process-group object for each symmetric allocation domain used by this runner.
    _symmetric_groups: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    # Per-domain bump offsets used while capturing this runner.
    _offsets: dict[tuple[Optional[str], torch.dtype], int] = field(
        default_factory=dict, init=False, repr=False
    )
    _plan: Optional["GraphPersistentBufferPlan"] = field(default=None, init=False, repr=False)

    def start_discovery(self) -> None:
        """Start a fresh eager capacity-discovery pass for this runner."""
        self.capacities.clear()
        self._symmetric_groups.clear()

    def clear(self) -> None:
        """Drop all state after graphs using this runner have been deleted."""
        self.capacities.clear()
        self._symmetric_groups.clear()
        self._offsets.clear()
        self.runner_index = None
        self.generation = None
        self._plan = None

    def record(self, shape: Iterable[int], dtype: torch.dtype, *, symmetric_group=None) -> None:
        """Add one aligned request to this runner's discovered allocation domain."""
        key = _persistent_buffer_key(dtype, symmetric_group)
        group_name, _ = key
        if group_name is not None:
            existing_group = self._symmetric_groups.setdefault(group_name, symmetric_group)
            if existing_group is not symmetric_group:
                raise RuntimeError(
                    f"Persistent buffer domain {group_name} refers to multiple process groups"
                )
        aligned_numel = _aligned_numel(_shape_numel(shape), dtype)
        self.capacities[key] = self.capacities.get(key, 0) + aligned_numel

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
        self._offsets = {key: 0 for key in self.capacities}

    def allocate(
        self, shape: Iterable[int], dtype: torch.dtype, *, symmetric_group=None
    ) -> torch.Tensor:
        """Bump-allocate one stable tensor view from this runner's generation."""
        if self._plan is None or self.generation is None or self.runner_index is None:
            raise RuntimeError(f"Runner {self.runner_index} has no persistent buffer generation")
        shape = tuple(shape)
        numel = _shape_numel(shape)
        aligned_numel = _aligned_numel(numel, dtype)
        key = _persistent_buffer_key(dtype, symmetric_group)
        group_name, _ = key
        if group_name is not None and self._symmetric_groups.get(group_name) is not symmetric_group:
            raise RuntimeError(
                f"Runner {self.runner_index} requested an unexpected process group for "
                f"persistent domain {group_name} during capture"
            )
        consumed = self._offsets.get(key, 0)
        capacity = self.capacities.get(key, 0)
        if consumed + aligned_numel > capacity:
            raise RuntimeError(
                f"Runner {self.runner_index} exhausted its persistent "
                f"{_format_persistent_buffer_key(key)} arena: "
                f"requested {aligned_numel} aligned elements after {consumed}, "
                f"capacity {capacity}"
            )
        arena = self._plan.get_arena(self.generation, dtype, symmetric_group=symmetric_group)
        self._offsets[key] = consumed + aligned_numel
        return arena.narrow(0, consumed, numel).view(shape)

    def validate_complete(self) -> None:
        """Require capture to consume exactly the capacity measured during discovery."""
        for key, capacity in self.capacities.items():
            consumed = self._offsets.get(key, 0)
            if consumed != capacity:
                raise RuntimeError(
                    f"Runner {self.runner_index} consumed {consumed}/{capacity} persistent "
                    f"{_format_persistent_buffer_key(key)} elements during capture"
                )

    def wait_for_reuse(self, stream: torch.cuda.Stream) -> None:
        """Wait before overwriting this runner's arena generation."""
        if self.capacities:
            self._plan.wait_for_reuse(self.generation, stream)

    def mark_reusable_after(self, stream: torch.cuda.Stream) -> None:
        """Publish the graph tail for the arena generation consumed here."""
        if self.capacities:
            self._plan.mark_reusable_after(self.generation, stream)


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
        capacities: dict[tuple[int, Optional[str], torch.dtype], int],
        symmetric_groups: dict[str, object],
        generation_count: int,
    ) -> None:
        # Runner states in backward graph-capture order.
        self._states: list[GraphPersistentBufferState] = states
        # Maximum aligned elements needed for each (generation, domain, dtype) arena.
        self._capacities: dict[tuple[int, Optional[str], torch.dtype], int] = capacities
        # Exact process-group object associated with each non-default allocation domain.
        self._symmetric_groups = symmetric_groups
        # Fixed-address backing tensors allocated outside CUDA graph memory pools.
        self._arenas: dict[tuple[int, Optional[str], torch.dtype], torch.Tensor] = {}
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
        capacities = {}
        symmetric_groups = {}
        assignments = []
        buffer_runner_index = 0

        for state in state_list:
            generation = None
            if state.capacities:
                generation = buffer_runner_index % generation_count
                buffer_runner_index += 1
                for group_name, group in state._symmetric_groups.items():
                    existing_group = symmetric_groups.setdefault(group_name, group)
                    if existing_group is not group:
                        raise RuntimeError(
                            f"Persistent buffer domain {group_name} refers to multiple "
                            "process groups"
                        )
                for (group_name, dtype), capacity in state.capacities.items():
                    key = (generation, group_name, dtype)
                    capacities[key] = max(capacities.get(key, 0), capacity)
            assignments.append(generation)

        plan = cls(state_list, capacities, symmetric_groups, generation_count)
        for runner_index, (state, generation) in enumerate(zip(state_list, assignments)):
            state.bind(plan, runner_index, generation)
        plan._allocate()
        return plan

    def _allocate(self) -> None:
        """Allocate every arena outside CUDA graph memory pools on the current device."""
        device = torch.cuda.current_device()
        for key, capacity in self._capacities.items():
            _, group_name, dtype = key
            group = self._symmetric_groups.get(group_name)
            allocation_context = gtp_symm_pool_ctx(group) if group is not None else nullcontext()
            with allocation_context:
                self._arenas[key] = torch.empty(capacity, dtype=dtype, device=device)
        current_stream = torch.cuda.current_stream()
        for _ in range(self._generation_count):
            event = torch.cuda.Event(external=True)
            # Seed the event so every captured runner has the same unconditional reuse fence.
            event.record(current_stream)
            self._ready_events.append(event)

    def get_arena(
        self, generation: int, dtype: torch.dtype, *, symmetric_group=None
    ) -> torch.Tensor:
        """Return one generation's typed backing arena."""
        group_name, _ = _persistent_buffer_key(dtype, symmetric_group)
        try:
            return self._arenas[(generation, group_name, dtype)]
        except KeyError as exc:
            raise RuntimeError(
                f"No persistent arena for generation {generation} and "
                f"{_format_persistent_buffer_key((group_name, dtype))}"
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
        self._symmetric_groups.clear()
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


def get_graph_persistent_buffer(
    shape: Iterable[int], dtype: torch.dtype, *, symmetric_group=None
) -> Optional[torch.Tensor]:
    """Record one eager request or return its stable capture-time tensor view.

    ``symmetric_group`` selects the registered NCCL allocation domain for an RS send buffer.
    Other graph-owned storage uses the default CUDA allocator. Keeping the domain explicit
    prevents dense GTP and expert GTP buffers from sharing memory registered to another group.
    """
    shape = tuple(shape)
    if _ACTIVE_DISCOVERY_STATE is not None:
        _ACTIVE_DISCOVERY_STATE.record(shape, dtype, symmetric_group=symmetric_group)
    if _ACTIVE_CAPTURE_STATE is None:
        return None
    return _ACTIVE_CAPTURE_STATE.allocate(shape, dtype, symmetric_group=symmetric_group)


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
    """GTP work and parameter-readiness dependencies owned by one capture or warmup pass."""

    params: list = field(default_factory=list)
    finalized_params: list = field(default_factory=list)
    params_to_ensure_ready: list = field(default_factory=list)
    ag_streams: list = field(default_factory=list)
    rs_streams: list = field(default_factory=list)
    _param_ids: set = field(default_factory=set)
    _param_ids_to_ensure_ready: set = field(default_factory=set)
    _ag_stream_ids: set = field(default_factory=set)
    _rs_stream_ids: set = field(default_factory=set)
    _comm_records: list[tuple[object, torch.cuda.Stream, bool]] = field(default_factory=list)

    def register_comm(self, param, stream: torch.cuda.Stream, *, reduce_scatter: bool) -> None:
        """Record a parameter and side stream owned by the active capture tracker."""
        self._comm_records.append((param, stream, reduce_scatter))
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

    def register_params_to_ensure_ready(self, params: Iterable) -> None:
        """Record parameters that must be published before this graph replays."""
        for param in params:
            param_id = id(param)
            if param_id not in self._param_ids_to_ensure_ready:
                self._param_ids_to_ensure_ready.add(param_id)
                self.params_to_ensure_ready.append(param)

    def get_comms_for_chain(self, chain_id: str) -> tuple[list, list, list]:
        """Return owned parameters, AG streams, and RS streams for one GTP chain."""
        params = []
        ag_streams = []
        rs_streams = []
        param_ids = set()
        ag_stream_ids = set()
        rs_stream_ids = set()

        for param, stream, reduce_scatter in self._comm_records:
            if getattr(param, "chain_id", None) != chain_id:
                continue
            if id(param) not in param_ids:
                param_ids.add(id(param))
                params.append(param)

            streams = rs_streams if reduce_scatter else ag_streams
            stream_ids = rs_stream_ids if reduce_scatter else ag_stream_ids
            if id(stream) not in stream_ids:
                stream_ids.add(id(stream))
                streams.append(stream)

        return params, ag_streams, rs_streams

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


def register_capture_params_to_ensure_ready(params: Iterable) -> None:
    """Record GTP parameter reads that need publication before graph replay."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_params_to_ensure_ready(params)


@contextmanager
def track_gtp_capture_comms():
    """Track asynchronous GTP work issued during one capture or warmup pass.

    This context tracks ownership, not graph-chain membership. Nested modules can issue work for
    ``UNGRAPHED`` parameters while the context is active, so callers implementing the cross-graph
    handoff must explicitly select the ``GRAPHED`` parameters and their corresponding streams.
    """
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
