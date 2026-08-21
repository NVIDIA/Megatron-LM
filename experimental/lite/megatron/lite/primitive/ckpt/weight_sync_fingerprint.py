# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Low-overhead fingerprints for online rollout weight-transfer streams."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable

import torch

_ENV_NAME = "MLITE_WEIGHT_SYNC_FINGERPRINT"
_REPORT_PREFIX = "MLITE_WEIGHT_SYNC_FINGERPRINT "


def weight_sync_fingerprint_enabled() -> bool:
    return os.getenv(_ENV_NAME, "").strip().lower() in {"1", "true", "yes", "on"}


def _sample_payload(name: str, tensor: torch.Tensor) -> tuple[bool, bytes]:
    """Sample data for a stable subset of tensors without copying full weights."""
    name_hash = hashlib.sha256(name.encode()).digest()
    important = any(
        marker in name
        for marker in ("embed_tokens", "lm_head", "attn_sink", "expert_bias", "tid2eid")
    )
    sampled = important or name_hash[0] < 16  # About 1/16 of ordinary tensors.
    if not sampled or tensor.numel() == 0:
        return sampled, b""

    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    count = min(256, raw.numel())
    if count == raw.numel():
        sample = raw
    else:
        # ``torch.linspace(..., dtype=int64)`` computes through floating point
        # on CUDA.  For multi-billion-byte expert tensors its rounded endpoint
        # can become ``numel`` and trip ScatterGatherKernel's bounds assert.
        # Integer arithmetic keeps every index in [0, numel - 1].
        indices = _sample_indices(raw.numel(), count, device=raw.device)
        sample = raw.index_select(0, indices)
    return sampled, bytes(sample.cpu().tolist())


def _sample_indices(numel: int, count: int, *, device=None) -> torch.Tensor:
    if not 1 <= count <= numel:
        raise ValueError(f"expected 1 <= count <= numel, got count={count}, numel={numel}")
    if count == 1:
        return torch.zeros(1, dtype=torch.int64, device=device)
    positions = torch.arange(count, dtype=torch.int64, device=device)
    return positions.mul_(numel - 1).floor_divide_(count - 1)


def tensor_fingerprint_record(name: str, tensor: torch.Tensor) -> dict[str, object]:
    sampled, payload = _sample_payload(name, tensor)
    metadata = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "nbytes": tensor.nbytes,
        "sampled": sampled,
    }
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    digest.update(payload)
    metadata["sha256"] = digest.hexdigest()
    return metadata


def stream_fingerprint(records: Iterable[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(records, key=lambda record: str(record["name"]))
    digest = hashlib.sha256()
    total_bytes = 0
    sampled_tensors = 0
    for record in ordered:
        digest.update(str(record["sha256"]).encode())
        total_bytes += int(record["nbytes"])
        sampled_tensors += int(bool(record["sampled"]))
    return {
        "sha256": digest.hexdigest(),
        "tensors": len(ordered),
        "sampled_tensors": sampled_tensors,
        "bytes": total_bytes,
    }


def report_stream_fingerprint(role: str, rank: int, records) -> None:
    report = {"role": role, "rank": rank, **stream_fingerprint(records)}
    print(_REPORT_PREFIX + json.dumps(report, sort_keys=True), flush=True)


__all__ = [
    "report_stream_fingerprint",
    "stream_fingerprint",
    "tensor_fingerprint_record",
    "weight_sync_fingerprint_enabled",
]
