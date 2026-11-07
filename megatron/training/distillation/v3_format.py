# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Lossless, token-aligned records for self-contained offline KD caches.

This module deliberately has no training/runtime dependencies so the format can
also be inspected, written, and tested on CPU without initializing Megatron, and
reused by non-Megatron producers and consumers.
"""

import itertools
from typing import Any

import torch
import zstandard
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from .format.contract import (  # noqa: F401  (re-exported for V3 callers)
    ATTENTION_MODES,
    FORMAT_VERSION,
    LOGPROB_SENTINEL,
    MASS_TOLERANCE,
    MAX_VOCAB_SIZE,
    RECORDED_KEYS,
    REPLAY_KEYS,
    REQUIRED_KEYS,
    digest,
    hf_tokenizer_identity,
    index_encoding_for_vocab,
    mismatched_settings,
    pack_targets,
    tail_logprob,
    tokenizer_fingerprint,
    unpack_targets,
    validate_metadata,
)

# Token-aligned input fields. Only ``tokens`` is required; the rest have defaults.
TOKEN_FIELDS = ("tokens", "labels", "position_ids", "loss_mask")
OPTIONAL_TOKEN_FIELDS = ("labels", "position_ids", "loss_mask")


def attention_mode_for_layout(
    *,
    sft: bool,
    inter_document_masking: bool,
    reset_attention_mask: bool,
    create_attention_mask: bool,
) -> str:
    """Describe a Megatron dataset's effective attention path, excluding loss masking."""
    if sft or inter_document_masking:
        return "isolated_segments"
    if reset_attention_mask and create_attention_mask:
        return "causal_eod_reset"
    return "causal"


def segment_padding_divisor(*, context_parallel_size: int, sequence_parallel_tp: int) -> int:
    """Megatron SFT's per-segment padding divisor (``SFTDataset._calculate_padding_divisor``)."""
    cp_pad = 2 * context_parallel_size if context_parallel_size > 1 else 1
    return cp_pad * max(1, sequence_parallel_tp)


def encode(payload: dict[str, torch.Tensor]) -> bytes:
    """Serialize a flat dict of tensors as safetensors inside one checksummed zstd frame."""
    tensors = {key: value.detach().cpu().contiguous() for key, value in payload.items()}
    return zstandard.ZstdCompressor(level=3, write_checksum=True).compress(save_tensors(tensors))


def decode(data: bytes) -> dict[str, torch.Tensor]:
    """Verify the frame checksum while decompressing, then load the safetensors payload."""
    try:
        raw = zstandard.ZstdDecompressor().decompress(data)
    except zstandard.ZstdError as error:
        raise ValueError(f"Offline KD v3 member checksum or framing error: {error}") from None
    try:
        return load_tensors(raw)
    except Exception as error:
        raise ValueError(f"Offline KD v3 member is not a safetensors payload: {error}") from None


def int32(tensor: torch.Tensor) -> torch.Tensor:
    """Compact integers without silently wrapping values."""
    limits = torch.iinfo(torch.int32)
    if tensor.numel() and (tensor.min().item() < limits.min or tensor.max().item() > limits.max):
        raise ValueError("Offline KD input does not fit in int32")
    return tensor.detach().to(device="cpu", dtype=torch.int32).contiguous().clone()


def capture_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Own a CPU snapshot of a fixed-length dataset microbatch before TP/CP slicing.

    This is the teacher-side capture layout (dataset boundaries, padding included);
    :func:`finalize_record` converts joined captures into the stored layout.
    """
    shape = batch["tokens"].shape
    if len(shape) != 2:
        raise ValueError("Offline KD v3 initially requires fixed-length [batch, sequence] inputs")
    rows, length = shape
    payload = {"sample_offsets": torch.arange(rows + 1) * length}
    for key in TOKEN_FIELDS:
        tensor = batch[key]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Offline KD {key} shape does not match tokens")
        if key == "loss_mask":
            tensor = tensor.detach().to(device="cpu", dtype=torch.float32).clone()
            if not torch.isfinite(tensor).all() or (tensor < 0).any():
                raise ValueError("Offline KD loss weights must be finite and nonnegative")
            binary = bool(((tensor == 0) | (tensor == 1)).all())
            payload[key] = tensor.to(torch.bool) if binary else tensor
        else:
            payload[key] = int32(tensor)
        payload[key] = payload[key].reshape(-1)
    boundaries = batch.get("cu_seqlens")
    payload["cu_seqlens"] = None
    if boundaries is not None:
        if boundaries.ndim != 2 or boundaries.shape[0] != rows:
            raise ValueError("Offline KD cu_seqlens must retain its original per-sample rows")
        compact = []
        for row in boundaries:
            row = int32(row)
            # Default collate pads the metadata by repeating the final boundary.
            row = torch.unique_consecutive(row)
            if row.numel() < 2 or row[0] != 0 or row[-1] != length or (row[1:] <= row[:-1]).any():
                raise ValueError("Invalid offline KD cu_seqlens boundaries")
            compact.append(row)
        payload["cu_seqlens"] = compact
    return payload


def join_inputs(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate captured samples without imposing teacher microbatch shapes."""
    if not batches:
        raise ValueError("Cannot serialize an empty offline KD iteration")
    result = {}
    offsets = [0]
    for batch in batches:
        offsets.extend((batch["sample_offsets"][1:] + offsets[-1]).tolist())
    result["sample_offsets"] = torch.tensor(offsets, dtype=torch.int64)
    for key in TOKEN_FIELDS:
        parts = [batch[key] for batch in batches]
        if len({part.dtype for part in parts}) > 1:
            parts = [part.float() for part in parts]  # Mixed binary/fractional weights.
        result[key] = torch.cat(parts)
    available = [batch["cu_seqlens"] is not None for batch in batches]
    if any(available) != all(available):
        raise ValueError("Mixed availability of cu_seqlens within a cache iteration")
    result["cu_seqlens"] = (
        [row for batch in batches for row in batch["cu_seqlens"]] if all(available) else None
    )
    return result


def default_inputs(
    tokens: torch.Tensor, segment_lengths: torch.Tensor, final_labels: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Defaults for omitted optional fields, per segment.

    ``position_ids`` count from 0 within each segment; ``labels`` are the next token
    within the segment, and ``final_labels`` (one per segment) at each segment's last
    position; ``loss_mask`` is all ones. Stored segments never end on an unscored
    position (trailing zero-weight positions are stripped), so the last position has a
    real next-token label that the stored tokens cannot provide.
    """
    total = tokens.numel()
    starts = torch.zeros(total, dtype=torch.int64)
    labels = torch.zeros(total, dtype=torch.int32)
    if total > 1:
        labels[:-1] = tokens[1:].to(torch.int32)
    if segment_lengths.numel():
        offsets = torch.cumsum(segment_lengths, 0)
        segment_of = torch.repeat_interleave(torch.arange(segment_lengths.numel()), segment_lengths)
        starts = (offsets - segment_lengths)[segment_of]
        labels[offsets - 1] = 0 if final_labels is None else final_labels.to(torch.int32)
    positions = torch.arange(total, dtype=torch.int64) - starts
    return {
        "position_ids": positions.to(torch.int32),
        "labels": labels,
        "loss_mask": torch.ones(total, dtype=torch.bool),
    }


def finalize_record(
    inputs: dict[str, Any],
    values: torch.Tensor,
    indices: torch.Tensor,
    tail: torch.Tensor,
    *,
    isolated: bool,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert joined teacher captures to the stored, padding-free record.

    Within each segment (each saved ``cu_seqlens`` segment when ``isolated``, else
    each whole sample), drop the trailing run of zero-weight positions. Under causal
    attention no kept position attends to them and they are never scored, so this is
    lossless; it removes dataset padding and trailing masked turns alike. Targets are
    kept only for the remaining positions with nonzero loss weight: context-only
    positions such as prompts need tokens but no target rows. Optional fields equal to
    their defaults are omitted.
    """
    offsets = inputs["sample_offsets"].tolist()
    weights = inputs["loss_mask"].float()
    keep = torch.zeros(weights.numel(), dtype=torch.bool)
    segment_lengths, segment_counts = [], []
    for sample in range(len(offsets) - 1):
        start, end = offsets[sample], offsets[sample + 1]
        bounds = inputs["cu_seqlens"][sample].tolist() if isolated else [0, end - start]
        count = 0
        for a, b in zip(bounds[:-1], bounds[1:]):
            scored = torch.nonzero(weights[start + a : start + b]).reshape(-1)
            if scored.numel():
                kept = int(scored[-1]) + 1
                keep[start + a : start + a + kept] = True
                segment_lengths.append(kept)
                count += 1
        segment_counts.append(count)
    lengths = torch.tensor(segment_lengths, dtype=torch.int64)
    segment_ends = list(itertools.accumulate(segment_lengths))
    sample_ends = [segment_ends[c - 1] if c else 0 for c in itertools.accumulate(segment_counts)]
    record = {
        "sample_offsets": torch.tensor([0, *sample_ends], dtype=torch.int64),
        "tokens": inputs["tokens"][keep],
        "segment_lengths": lengths.to(torch.int32),
        "segment_counts": torch.tensor(segment_counts, dtype=torch.int32),
    }
    ends = torch.cumsum(lengths, 0) - 1
    final_labels = inputs["labels"][keep][ends].to(torch.int32)
    defaults = default_inputs(record["tokens"], lengths, final_labels)
    for key in OPTIONAL_TOKEN_FIELDS:
        kept = inputs[key][keep]
        default = defaults[key]
        if kept.dtype == torch.bool or default.dtype == torch.bool:
            same = kept.dtype == default.dtype and torch.equal(kept, default)
        else:
            same = torch.equal(kept.to(default.dtype), default)
        if not same:
            record[key] = kept
    if "labels" not in record:
        record["segment_final_labels"] = final_labels
    rows = keep & (weights != 0)
    return record, values[rows], indices[rows], tail[rows]


def validate_inputs(payload: dict[str, Any]) -> None:
    """Reject malformed stored inputs (sample offsets, segments, token-aligned fields)."""
    offsets = payload["sample_offsets"]
    if (
        offsets.ndim != 1
        or offsets.numel() < 2
        or offsets[0] != 0
        or (offsets[1:] < offsets[:-1]).any()
    ):
        raise ValueError("Malformed offline KD sample offsets")
    total = int(offsets[-1])
    for key in TOKEN_FIELDS:
        tensor = payload.get(key)
        if tensor is None and key != "tokens":
            continue
        if tensor is None or tensor.ndim != 1 or tensor.numel() != total:
            raise ValueError(f"Malformed offline KD {key} tensor")
    lengths, counts = payload.get("segment_lengths"), payload.get("segment_counts")
    if payload.get("labels") is None:
        final = payload.get("segment_final_labels")
        segments = (
            lengths.numel() if lengths is not None else int((offsets[1:] > offsets[:-1]).sum())
        )
        if final is None or final.ndim != 1 or final.numel() != segments:
            raise ValueError("Offline KD records without labels need one final label per segment")
    if (lengths is None) != (counts is None):
        raise ValueError("Offline KD segment_lengths and segment_counts must appear together")
    if lengths is None:
        return
    if (
        counts.numel() != offsets.numel() - 1
        or int(counts.sum()) != lengths.numel()
        or (lengths <= 0).any()
        or (counts < 0).any()
    ):
        raise ValueError("Malformed offline KD segments")
    per_sample = torch.zeros(counts.numel(), dtype=torch.int64)
    if lengths.numel():
        owner = torch.repeat_interleave(torch.arange(counts.numel()), counts.long())
        per_sample.index_add_(0, owner, lengths.long())
    if not torch.equal(per_sample, offsets[1:] - offsets[:-1]):
        raise ValueError("Offline KD segments disagree with sample lengths")


def fill_input_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    """Return inputs with every optional field present (stored values or defaults).

    Inputs without segments treat each non-empty sample as one segment.
    """
    offsets = payload["sample_offsets"]
    sample_lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)
    if payload.get("segment_lengths") is None:
        lengths = sample_lengths[sample_lengths > 0]
        counts = (sample_lengths > 0).to(torch.int32)
    else:
        lengths = payload["segment_lengths"].to(torch.int64)
        counts = payload["segment_counts"]
    result = dict(payload)
    result["segment_lengths"], result["segment_counts"] = lengths, counts
    defaults = default_inputs(payload["tokens"], lengths, payload.get("segment_final_labels"))
    for key in OPTIONAL_TOKEN_FIELDS:
        if result.get(key) is None:
            result[key] = defaults[key]
    return result
