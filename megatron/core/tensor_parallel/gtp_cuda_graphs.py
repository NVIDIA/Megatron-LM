# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CUDA-graph capture and replay support for Generalized Tensor Parallelism (GTP)."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class GTPCaptureCommState:
    """GTP communication dependencies issued while capturing one CUDA graph."""

    gtp_param_sync_handles: list = field(default_factory=list)
    _gtp_param_sync_handle_ids: set = field(default_factory=set)

    def register_gtp_param_sync(self, handles: Iterable) -> None:
        """Record each DDP parameter-sync dependency once for this graph."""
        for handle in handles:
            handle_id = id(handle)
            if handle_id not in self._gtp_param_sync_handle_ids:
                self._gtp_param_sync_handle_ids.add(handle_id)
                self.gtp_param_sync_handles.append(handle)

    def ensure_param_sync_ready(self) -> None:
        """Finish DDP parameter gathers required before replaying this graph."""
        for handle in self.gtp_param_sync_handles:
            handle.ensure_ready()


_ACTIVE_CAPTURE_COMM_STATE: Optional[GTPCaptureCommState] = None


def register_capture_gtp_param_sync(handles: Iterable) -> None:
    """Register DDP parameter-sync dependencies with the active graph capture."""
    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        _ACTIVE_CAPTURE_COMM_STATE.register_gtp_param_sync(handles)


@contextmanager
def track_gtp_capture_comms():
    """Create isolated GTP communication state for one CUDA-graph capture."""
    global _ACTIVE_CAPTURE_COMM_STATE

    if _ACTIVE_CAPTURE_COMM_STATE is not None:
        raise RuntimeError("Nested GTP CUDA-graph communication tracking is unsupported")

    state = GTPCaptureCommState()
    _ACTIVE_CAPTURE_COMM_STATE = state
    try:
        yield state
    finally:
        _ACTIVE_CAPTURE_COMM_STATE = None
