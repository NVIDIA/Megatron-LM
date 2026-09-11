# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Persistent wgrad storage and CUDA-graph lifecycle support for GTP.

This module owns:

* capture-local ownership of asynchronous GTP communication;
* persistent wgrad rings shared by eager execution and local CUDA graphs;
* routing graph-owned allocations into the shared CUDA-graph memory pool.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch

from megatron.core.tensor_parallel.gtp_symmetric_memory import (
    gtp_symm_pool_ctx,
    is_gtp_symm_pool_registered,
)
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


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
class WgradRingSlot:
    """One persistent wgrad slot guarded by its reduce-scatter completion event."""

    tensor: torch.Tensor
    ready_event: torch.cuda.Event
    owner: object = None

    def wait_for_reader(self) -> None:
        """Acquire an eager writer's slot, finalizing any deferred reduction first."""
        if self.owner is not None and self.owner._wgrad_rs_handle is not None:
            self.owner._wait_reduce_scatter(finalize_grad=True)
        self.ready_event.wait()


@dataclass
class GTPCaptureCommState:
    """GTP work and parameter-readiness dependencies owned by one capture or warmup pass."""

    params: list = field(default_factory=list)
    params_to_ensure_ready: list = field(default_factory=list)
    ag_streams: list = field(default_factory=list)
    rs_streams: list = field(default_factory=list)
    wgrad_ring_slots: list = field(default_factory=list)
    _param_ids: set = field(default_factory=set)
    _param_ids_to_ensure_ready: set = field(default_factory=set)
    _ag_stream_ids: set = field(default_factory=set)
    _rs_stream_ids: set = field(default_factory=set)
    _wgrad_ring_slot_params: dict = field(default_factory=dict)
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

    def register_wgrad_ring_slot(self, slot: WgradRingSlot, param) -> None:
        """Track slots used by this graph and reject unsafe intra-graph aliasing."""
        slot_id = id(slot)
        param_id = id(param)
        prior_param_id = self._wgrad_ring_slot_params.get(slot_id)
        if prior_param_id is not None and prior_param_id != param_id:
            raise RuntimeError(
                "One CUDA graph writes the same GTP wgrad ring slot for multiple "
                "parameters; increase GTP_CONFIG.graph_wgrad_ring_size"
            )
        if prior_param_id is None:
            self._wgrad_ring_slot_params[slot_id] = param_id
            self.wgrad_ring_slots.append(slot)


_ACTIVE_CAPTURE_COMM_STATE: Optional[GTPCaptureCommState] = None


def register_capture_comm(param, stream: torch.cuda.Stream, *, reduce_scatter: bool) -> None:
    """Register communication with the active capture, if one exists."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_comm(param, stream, reduce_scatter=reduce_scatter)


def register_capture_params_to_ensure_ready(params: Iterable) -> None:
    """Record GTP parameter reads that need publication before graph replay."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_params_to_ensure_ready(params)


def register_capture_wgrad_ring_slot(slot: WgradRingSlot, param) -> None:
    """Register a ring slot with the active capture, if one exists."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_wgrad_ring_slot(slot, param)


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


# Slots live outside the shared graph pool so independently replayed graphs cannot reuse an
# in-flight reduce-scatter input as temporary workspace.
_WGRAD_RINGS: dict[tuple, WgradRingSlot] = {}


def bind_wgrad_ring_slot(param, key: tuple) -> WgradRingSlot:
    """Bind a parameter to fixed padded storage in its caller-selected scheduling domain."""
    slot = _WGRAD_RINGS.get(key)
    if slot is None:
        symm = is_gtp_symm_pool_registered(param.group)
        with gtp_symm_pool_ctx(param.group) if symm else nullcontext():
            tensor = torch.empty(
                param._unsharded_shape_padded, dtype=param.main_grad.dtype, device=param.device
            )
        if param.pad_length > 0:
            tensor[param._unsharded_shape[0] :].zero_()
        slot = _WGRAD_RINGS[key] = WgradRingSlot(
            tensor=tensor, ready_event=torch.cuda.Event(external=True)
        )
        slot.ready_event.record()
    param._gtp_wgrad_ring_slot = slot
    param._gtp_wgrad_ring_view = slot.tensor[: param._unsharded_shape[0]]
    return slot


def allocate_graph_wgrad_rings(
    params: Iterable,
    *,
    full_iteration: bool,
    async_reduction: bool,
    ring_size: int,
    graphed_chain_id: str,
    stream_key: Callable[[str, object], tuple],
) -> None:
    """Allocate bounded persistent inputs for cross-graph asynchronous reduce-scatter.

    Slots are shared across layers only within one communication scheduling domain. A two-slot
    ring retains one graph of overlap without allocating one full unsharded wgrad per layer.
    """
    if full_iteration or not async_reduction:
        return
    if ring_size < 1:
        raise ValueError("GTP_CONFIG.graph_wgrad_ring_size must be at least 1")

    params_by_key = defaultdict(list)
    seen_params = set()
    for chain_param in params:
        if not getattr(chain_param, "is_gtp_weight_remat", False):
            continue
        if chain_param.chain_id != graphed_chain_id or chain_param.prev_w is None:
            continue
        for param in chain_param._weights:
            if id(param) in seen_params or hasattr(param, "_gtp_wgrad_ring_slot"):
                continue
            seen_params.add(id(param))
            if not hasattr(param, "main_grad"):
                raise RuntimeError(
                    "GTP wgrad rings must be initialized after DDP creates param.main_grad"
                )
            key = (
                stream_key(param.chain_id, param.group),
                param._unsharded_shape,
                param._unsharded_shape_padded,
                param.main_grad.dtype,
                param.expert_idx,
            )
            params_by_key[key].append(param)

    # Symm-RS: GRAPHED chains send their persistent ring slot directly, so allocating the
    # slot from the window-registered pool puts the RS send buffer in the NCCL symmetric
    # window. This runs pre-capture and the pool routing is collective-free, so it is
    # capture-safe; slot addresses stay stable either way.
    total_bytes = 0
    buffer_count = 0
    for key, matching_params in params_by_key.items():
        slot_count = min(ring_size, len(matching_params))
        exemplar = matching_params[0]
        assert all(p.group is exemplar.group for p in matching_params), (
            "GTP wgrad ring slots are allocated from the exemplar's symmetric pool, so "
            "every param sharing a ring key must share its process group"
        )
        for param_index, param in enumerate(matching_params):
            slot = bind_wgrad_ring_slot(param, ("graph", key, param_index % slot_count))
            if param_index < slot_count:
                total_bytes += slot.tensor.numel() * slot.tensor.element_size()
                buffer_count += 1

    # Initially every slot is available. Later generations are recorded on the RS stream after NCCL
    # has finished reading the slot.
    if buffer_count:
        torch.cuda.current_stream().synchronize()

    log_single_rank(
        logger,
        logging.INFO,
        f"[GTP Wgrad Ring] allocated {buffer_count} buffers "
        f"({total_bytes / 1024**2:.1f} MB), ring_size={ring_size}",
    )


def clear_wgrad_rings(params: Iterable) -> None:
    """Drop persistent storage and parameter bindings after callers have synchronized GPU work."""
    for param in params:
        for attr in ("_gtp_wgrad_ring_slot", "_gtp_wgrad_ring_view"):
            if hasattr(param, attr):
                delattr(param, attr)
    _WGRAD_RINGS.clear()


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
    if _CG_MEMPOOL is None or not enabled:
        yield
        return

    torch._C._cuda_beginAllocateCurrentThreadToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
    try:
        yield
    finally:
        torch._C._cuda_endAllocateToPool(_CG_MEMPOOL_DEVICE, _CG_MEMPOOL)
