# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lossless, token-aligned records for self-contained offline KD caches.

This module deliberately has no training/runtime dependencies so the format can
also be inspected and tested on CPU without initializing Megatron.
"""

import hashlib
import io
from typing import Any

import numpy as np
import torch
import zstandard

FORMAT_VERSION = 3
TOKEN_FIELDS = ("tokens", "labels", "position_ids", "loss_mask")
BOUNDARY_FIELDS = ("cu_seqlens", "cu_seqlens_padded")


def digest(data: bytes) -> str:
    """Return the integrity digest for an encoded member."""
    return hashlib.sha256(data).hexdigest()


def encode(payload: dict[str, Any]) -> bytes:
    """Serialize tensors into one independently compressed tar member."""
    stream = io.BytesIO()
    torch.save(payload, stream)
    return zstandard.ZstdCompressor(level=3).compress(stream.getvalue())


def decode(data: bytes, expected_digest: str) -> dict[str, Any]:
    """Validate a member before decoding its tensor-only payload."""
    if digest(data) != expected_digest:
        raise ValueError("Offline KD v3 member checksum mismatch")
    raw = zstandard.ZstdDecompressor().decompress(data)
    result = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if result.get("format_version") != FORMAT_VERSION:
        raise ValueError("Expected an offline KD v3 payload")
    return result


def int32(tensor: torch.Tensor) -> torch.Tensor:
    """Compact integers without silently wrapping values."""
    limits = torch.iinfo(torch.int32)
    if tensor.numel() and (tensor.min().item() < limits.min or tensor.max().item() > limits.max):
        raise ValueError("Offline KD input does not fit in int32")
    return tensor.detach().to(device="cpu", dtype=torch.int32).contiguous().clone()


def capture_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Own a CPU snapshot before TP filtering, THD flattening, and CP slicing."""
    shape = batch["tokens"].shape
    if len(shape) != 2:
        raise ValueError("Offline KD v3 initially requires fixed-length [batch, sequence] inputs")
    rows, length = shape
    payload = {"format_version": FORMAT_VERSION, "sample_offsets": torch.arange(rows + 1) * length}
    for key in TOKEN_FIELDS:
        tensor = batch[key]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Offline KD {key} shape does not match tokens")
        if key == "loss_mask":
            tensor = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(tensor).all() or (tensor < 0).any():
                raise ValueError("Offline KD loss weights must be finite and nonnegative")
            binary = bool(((tensor == 0) | (tensor == 1)).all())
            payload["loss_mask_encoding"] = "bool" if binary else "float32"
            payload[key] = tensor.to(torch.bool) if binary else tensor
        else:
            payload[key] = int32(tensor)
        payload[key] = payload[key].reshape(-1)
    for key in BOUNDARY_FIELDS:
        boundaries = batch.get(key)
        if boundaries is None:
            payload[key] = None
            continue
        if boundaries.ndim != 2 or boundaries.shape[0] != rows:
            raise ValueError(f"Offline KD {key} must retain its original per-sample rows")
        compact = []
        for row in boundaries:
            row = int32(row)
            # Default collate pads the metadata by repeating the final boundary.
            row = torch.unique_consecutive(row)
            if row.numel() < 2 or row[0] != 0 or row[-1] != length or (row[1:] <= row[:-1]).any():
                raise ValueError(f"Invalid offline KD {key} boundaries")
            compact.append(row)
        payload[key] = compact
    return payload


def join_inputs(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate captured samples without imposing teacher microbatch shapes."""
    if not batches:
        raise ValueError("Cannot serialize an empty offline KD iteration")
    result = {"format_version": FORMAT_VERSION}
    offsets = [0]
    for batch in batches:
        offsets.extend((batch["sample_offsets"][1:] + offsets[-1]).tolist())
    result["sample_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    for key in TOKEN_FIELDS:
        result[key] = torch.cat([batch[key] for batch in batches])
    result["loss_mask_encoding"] = "bool" if result["loss_mask"].dtype == torch.bool else "float32"
    for key in BOUNDARY_FIELDS:
        available = [batch[key] is not None for batch in batches]
        if any(available) != all(available):
            raise ValueError(f"Mixed availability of {key} within a cache iteration")
        result[key] = [row for batch in batches for row in batch[key]] if all(available) else None
    return result


def pack_targets(values: torch.Tensor, indices: torch.Tensor) -> dict[str, Any]:
    """Encode canonical [tokens, K] targets with the existing 17-bit layout."""
    if values.shape != indices.shape or values.ndim != 2:
        raise ValueError("Offline KD targets must have matching [tokens, K] shapes")
    if indices.numel() and (indices.min() < -1 or indices.max() >= (1 << 17)):
        raise ValueError("Offline KD 17-bit encoding cannot represent this vocabulary index")
    indices = indices.to(device="cpu", dtype=torch.int64)
    bits = np.packbits(((indices.reshape(-1).numpy() >> 16) & 1).astype(np.uint8))
    return {
        "format_version": FORMAT_VERSION,
        "values": values.detach().cpu().contiguous(),
        "indices_low": (indices & 0xFFFF).to(torch.uint16),
        "bit_17": torch.from_numpy(bits),
    }


def unpack_targets(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct vocabulary indices in canonical token order."""
    low = payload["indices_low"].to(torch.int64)
    high = np.unpackbits(payload["bit_17"].numpy(), count=low.numel()).copy()
    indices = low | (torch.from_numpy(high).to(torch.int64).reshape(low.shape) << 16)
    values = payload["values"]
    if values.shape != indices.shape or values.ndim != 2:
        raise ValueError("Malformed offline KD target tensors")
    return values, indices


def validate_inputs(payload: dict[str, Any]) -> None:
    """Reject malformed sample offsets and token-aligned input fields."""
    offsets = payload["sample_offsets"]
    if (
        offsets.ndim != 1
        or offsets.numel() < 2
        or offsets[0] != 0
        or (offsets[1:] <= offsets[:-1]).any()
    ):
        raise ValueError("Malformed offline KD sample offsets")
    for key in TOKEN_FIELDS:
        if payload[key].ndim != 1 or payload[key].numel() != offsets[-1]:
            raise ValueError(f"Malformed offline KD {key} tensor")
    for key in BOUNDARY_FIELDS:
        rows = payload.get(key)
        if rows is None:
            continue
        if len(rows) != offsets.numel() - 1:
            raise ValueError(f"Malformed offline KD {key} sample count")
        for i, row in enumerate(rows):
            if (
                row.ndim != 1
                or row.numel() < 2
                or row[0] != 0
                or row[-1] != offsets[i + 1] - offsets[i]
            ):
                raise ValueError(f"Malformed offline KD {key} endpoints")
            if (row[1:] <= row[:-1]).any():
                raise ValueError(f"Malformed offline KD {key} order")
