# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Persistent eager wgrad rings for Generalized Tensor Parallelism (GTP).

Slots share the CUDA-graph ring allocation helpers and finalize their previous
reduce-scatter reader before another eager writer reuses the storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from megatron.core.tensor_parallel.gtp_cuda_graphs import (
    GraphWgradRingSlot,
    allocate_wgrad_ring,
    wgrad_ring_key,
)


@dataclass
class EagerWgradRingSlot(GraphWgradRingSlot):
    """Persistent eager input whose previous reduce-scatter must finish before reuse."""

    rs_owner: object = None

    def wait_for_reader(self) -> None:
        """Finalize a deferred reduction before another GEMM writes into its input."""
        rs_owner = self.rs_owner
        if rs_owner is not None and rs_owner._wgrad_rs_handle is not None:
            rs_owner._wait_reduce_scatter(finalize_grad=True)
            # The previous result is accounted for; its marker must not suppress the next one.
            rs_owner._already_finalized = False
        self.ready_event.wait()


_EAGER_WGRAD_RINGS: dict[tuple, list[EagerWgradRingSlot]] = {}


def bind_eager_wgrad_ring_slot(
    param, params: Iterable, *, ring_size: int, stream_key: Callable[[str, object], tuple]
) -> EagerWgradRingSlot:
    """Initialize a domain's registered writers together on its first persistent request."""
    if ring_size < 1:
        raise ValueError("GTP_CONFIG.eager_wgrad_ring_size must be at least 1")
    key = wgrad_ring_key(param, stream_key)
    if key not in _EAGER_WGRAD_RINGS:
        matching_params = [
            weight
            for weight in params
            if weight.requires_grad
            and weight.chain_id == param.chain_id
            and wgrad_ring_key(weight, stream_key) == key
        ]
        if not any(weight is param for weight in matching_params):
            raise RuntimeError("Persistent eager wgrad writers must be registered in _GTP_PARAMS")
        slots = _EAGER_WGRAD_RINGS[key] = allocate_wgrad_ring(
            matching_params, key=key, ring_size=ring_size, slot_type=EagerWgradRingSlot
        )
        for param_index, weight in enumerate(matching_params):
            weight._gtp_eager_wgrad_ring_slot = slots[param_index % len(slots)]
    return param._gtp_eager_wgrad_ring_slot


def clear_eager_wgrad_rings() -> None:
    """Drop eager ring storage when the process-global GTP state is reset."""
    _EAGER_WGRAD_RINGS.clear()
