# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Small JSONL helpers for opt-in context-parallel timing instrumentation."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

_COUNTERS: dict[str, int] = {}


def cp_timing_enabled() -> bool:
    return bool(os.getenv("MEGATRON_CP_TIMING_DIR"))


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def record_cp_timing(event: dict[str, Any]) -> None:
    out_dir = os.getenv("MEGATRON_CP_TIMING_DIR")
    if not out_dir:
        return
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    event_name = str(event.get("event", "unknown"))
    event["event_index"] = _COUNTERS.get(event_name, 0)
    _COUNTERS[event_name] = event["event_index"] + 1
    event.setdefault("rank", _rank())
    event.setdefault("ts_ns", time.time_ns())
    with (path / f"rank{event['rank']:05d}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True) + "\n")


def tensor_meta(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "element_size": int(tensor.element_size()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "device": str(tensor.device),
    }


def classify_cp_comm(cp_comm_type: str | None) -> str:
    if cp_comm_type == "p2p":
        return "ring_p2p_multi_peer"
    if cp_comm_type == "all_gather":
        return "all_gather_full_kv"
    if cp_comm_type == "a2a":
        return "all_to_all_head_sequence_exchange"
    if cp_comm_type == "a2a+p2p":
        return "hybrid_hierarchical_a2a_p2p"
    if cp_comm_type == "nvshmem_ring_p2p":
        return "nvshmem_peer_read_ring_p2p_forward_only"
    if cp_comm_type is None:
        return "none"
    return "unknown"
