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
MAX_VOCAB_SIZE = 1 << 24
LOGPROB_SENTINEL = -1000.0
TOKEN_FIELDS = ("tokens", "labels", "position_ids", "loss_mask")
BOUNDARY_FIELDS = ("cu_seqlens", "cu_seqlens_padded")
# Settings every tar in a cache must share, whichever dump job wrote it.
SHARED_KEYS = (
    "format_version",
    "first_sample",
    "dp_size_save",
    "mbs_save",
    "gbs_save",
    "cp_size_save",
    "save_interval",
    "train_budget",
    "seq_length",
    "sft",
    "inter_document_masking",
    "reset_attention_mask",
    "tokenizer",
    "padded_vocab_size",
    "targets",
    "dataset_identity",
    "attention_mode",
    "padding_policy",
)


def index_bits_for_vocab(vocab_size: int) -> int:
    """Choose the smallest index width with a uint16 low word."""
    if type(vocab_size) is not int or not 1 <= vocab_size <= MAX_VOCAB_SIZE:
        raise ValueError("Offline KD vocabulary size must be an integer in [1, 2**24]")
    return max(16, (vocab_size - 1).bit_length())


def attention_mode_for_layout(
    *,
    sft: bool,
    inter_document_masking: bool,
    reset_attention_mask: bool,
    create_attention_mask: bool,
) -> str:
    """Describe the effective dataset attention path, excluding loss masking."""
    if sft or inter_document_masking:
        return "isolated_segments"
    if reset_attention_mask and create_attention_mask:
        return "causal_eod_reset"
    return "causal"


def validate_metadata(meta: dict[str, Any]) -> None:
    """Reject unsupported or internally inconsistent V3 encoding/context declarations."""
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError("Expected offline KD v3 metadata")
    bits = index_bits_for_vocab(meta.get("padded_vocab_size"))
    if (
        type(meta.get("targets", {}).get("index_bits")) is not int
        or meta["targets"]["index_bits"] != bits
    ):
        raise ValueError("Offline KD target index_bits disagrees with padded_vocab_size")
    mode = meta.get("attention_mode")
    if mode not in ("causal", "causal_eod_reset", "isolated_segments"):
        raise ValueError("Unsupported offline KD attention_mode")
    packed = bool(meta.get("sft") or meta.get("inter_document_masking"))
    if (mode == "isolated_segments") != packed:
        raise ValueError("Offline KD attention_mode disagrees with packing flags")
    if mode == "causal_eod_reset" and not meta.get("reset_attention_mask"):
        raise ValueError("Offline KD causal_eod_reset requires reset_attention_mask")
    if meta.get("padding_policy") != "preserve":
        raise ValueError("Offline KD v3 requires padding_policy=preserve")


def mismatched_settings(meta: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return the shared settings on which two cache descriptions disagree."""
    return [key for key in SHARED_KEYS if meta.get(key) != expected.get(key)]


def digest(data: bytes) -> str:
    """Return a stable fingerprint (used for tokenizer identity)."""
    return hashlib.sha256(data).hexdigest()


def encode(payload: dict[str, Any]) -> bytes:
    """Serialize tensors into one zstd frame carrying its own content checksum."""
    stream = io.BytesIO()
    torch.save(payload, stream)
    return zstandard.ZstdCompressor(level=3, write_checksum=True).compress(stream.getvalue())


def decode(data: bytes) -> dict[str, Any]:
    """Verify the frame checksum while decompressing, then load the tensor-only payload."""
    try:
        raw = zstandard.ZstdDecompressor().decompress(data)
    except zstandard.ZstdError as error:
        raise ValueError(f"Offline KD v3 member checksum or framing error: {error}") from None
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


def pack_targets(values: torch.Tensor, indices: torch.Tensor, *, vocab_size: int) -> dict[str, Any]:
    """Encode [tokens, K] indices as uint16 low words and packed high-bit planes."""
    bits = index_bits_for_vocab(vocab_size)
    if values.shape != indices.shape or values.ndim != 2:
        raise ValueError("Offline KD targets must have matching [tokens, K] shapes")
    if not values.is_floating_point() or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Offline KD targets require floating values and int32/int64 indices")
    values = values.detach().cpu().contiguous()
    indices = indices.detach().to(device="cpu", dtype=torch.int64)
    if indices.numel() and (
        (indices < -1).any()
        or (indices >= vocab_size).any()
        or ((indices == -1) & (values != LOGPROB_SENTINEL)).any()
    ):
        raise ValueError("Offline KD vocabulary index is out of range or has an unmasked sentinel")
    flat = indices.reshape(-1).numpy()
    high = np.empty((bits - 16, (flat.size + 7) // 8), dtype=np.uint8)
    for plane in range(bits - 16):
        high[plane] = np.packbits(((flat >> (16 + plane)) & 1).astype(np.uint8), bitorder="big")
    return {
        "format_version": FORMAT_VERSION,
        "index_bits": bits,
        "values": values,
        "indices_low": (indices & 0xFFFF).to(torch.uint16),
        "indices_high": torch.from_numpy(high),
    }


def unpack_targets(
    payload: dict[str, Any], *, vocab_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct vocabulary indices in canonical token order."""
    bits = payload.get("index_bits")
    if type(bits) is not int or not 16 <= bits <= 24:
        raise ValueError("Malformed offline KD target index_bits")
    if vocab_size is not None and bits != index_bits_for_vocab(vocab_size):
        raise ValueError("Offline KD target index_bits disagrees with padded vocabulary")
    low, high = payload["indices_low"], payload["indices_high"]
    values = payload["values"]
    if (
        values.shape != low.shape
        or values.ndim != 2
        or not values.is_floating_point()
        or low.dtype != torch.uint16
        or high.dtype != torch.uint8
        or tuple(high.shape) != (bits - 16, (low.numel() + 7) // 8)
    ):
        raise ValueError("Malformed offline KD target tensors")
    indices = low.to(torch.int64)
    for plane, packed in enumerate(high.numpy()):
        if low.numel() % 8 and packed[-1] & ((1 << (8 - low.numel() % 8)) - 1):
            raise ValueError("Malformed offline KD high-bit plane padding")
        unpacked = np.unpackbits(packed, count=low.numel(), bitorder="big").copy()
        indices |= torch.from_numpy(unpacked).to(torch.int64).reshape(low.shape) << (16 + plane)
    if vocab_size is not None and ((indices >= vocab_size) & (values != LOGPROB_SENTINEL)).any():
        raise ValueError("Offline KD target vocabulary index is out of range")
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
