# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph lifecycle support for Generalized Tensor Parallelism (GTP).

This module owns state that exists only for local CUDA-graph capture and replay:

* capture-local ownership of asynchronous GTP communication;
* persistent wgrad ring buffers whose lifetime may cross graph boundaries;
* routing graph-owned allocations into the shared CUDA-graph memory pool.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import torch

from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


@dataclass
class GraphWgradRingSlot:
    """One persistent wgrad slot guarded by its reduce-scatter completion event."""

    tensor: torch.Tensor
    ready_event: torch.cuda.Event
    key: tuple
    index: int


@dataclass
class GTPCaptureCommState:
    """Asynchronous GTP work issued while capturing one CUDA graph."""

    params: list = field(default_factory=list)
    ag_streams: list = field(default_factory=list)
    rs_streams: list = field(default_factory=list)
    wgrad_ring_slots: list = field(default_factory=list)
    _param_ids: set = field(default_factory=set)
    _ag_stream_ids: set = field(default_factory=set)
    _rs_stream_ids: set = field(default_factory=set)
    _wgrad_ring_slot_params: dict = field(default_factory=dict)

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

    def register_wgrad_ring_slot(self, slot: GraphWgradRingSlot, param) -> None:
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


def register_capture_wgrad_ring_slot(slot: GraphWgradRingSlot, param) -> None:
    """Register a ring slot with the active capture, if one exists."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_wgrad_ring_slot(slot, param)


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


# Slots live outside the shared graph pool so independently replayed graphs cannot reuse an
# in-flight reduce-scatter input as temporary workspace.
_GRAPH_WGRAD_RINGS: dict[tuple, list[GraphWgradRingSlot]] = {}


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
    if full_iteration or not async_reduction or _GRAPH_WGRAD_RINGS:
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
            if id(param) in seen_params:
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

    total_bytes = 0
    buffer_count = 0
    new_slots = []
    for key, matching_params in params_by_key.items():
        slot_count = min(ring_size, len(matching_params))
        slots = []
        exemplar = matching_params[0]
        for slot_index in range(slot_count):
            tensor = torch.empty(
                exemplar._unsharded_shape_padded,
                dtype=exemplar.main_grad.dtype,
                device=exemplar.device,
                memory_format=torch.contiguous_format,
            )
            if exemplar.pad_length > 0:
                tensor.narrow(0, exemplar._unsharded_shape[0], exemplar.pad_length).zero_()
            slot = GraphWgradRingSlot(
                tensor=tensor,
                ready_event=torch.cuda.Event(external=True),
                key=key,
                index=slot_index,
            )
            slots.append(slot)
            new_slots.append(slot)
            total_bytes += tensor.numel() * tensor.element_size()
            buffer_count += 1

        _GRAPH_WGRAD_RINGS[key] = slots
        for param_index, param in enumerate(matching_params):
            slot = slots[param_index % slot_count]
            param._gtp_graph_wgrad_ring_slot = slot
            if param.pad_length > 0:
                param._gtp_graph_wgrad_ring_view = slot.tensor.narrow(
                    0, 0, param._unsharded_shape[0]
                )
            else:
                param._gtp_graph_wgrad_ring_view = slot.tensor

    # Initially every slot is available. Later generations are recorded on the RS stream after NCCL
    # has finished reading the slot.
    for slot in new_slots:
        slot.ready_event.record()
    if new_slots:
        torch.cuda.current_stream().synchronize()

    log_single_rank(
        logger,
        logging.INFO,
        f"[GTP Wgrad Ring] allocated {buffer_count} buffers "
        f"({total_bytes / 1024**2:.1f} MB), ring_size={ring_size}",
    )


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
