# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Teacher-side capture of Megatron batches into lossless, token-aligned records.

The producer-neutral parts of the format (target encodings, metadata, the Arrow file
layout) live in :mod:`.format`; this module re-exports the contract helpers for V3
callers. It has no training/runtime dependencies, so it is tested on CPU without
initializing Megatron.
"""

import itertools
from typing import Any

import torch

from .format.contract import (  # noqa: F401  (re-exported for V3 callers)
    ATTENTION_MODES,
    CORPUS_KEYS,
    FORMAT_VERSION,
    LOGPROB_SENTINEL,
    MASS_TOLERANCE,
    MAX_VOCAB_SIZE,
    RECORDED_KEYS,
    REPLAY_KEYS,
    REQUIRED_KEYS,
    check_targets,
    digest,
    hf_tokenizer_identity,
    index_encoding_for_vocab,
    join_indices,
    mismatched_settings,
    split_indices,
    tail_logprob,
    tokenizer_fingerprint,
    validate_metadata,
)

# Token-aligned input fields, all stored for every sample.
TOKEN_FIELDS = ("tokens", "labels", "position_ids", "loss_mask")


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
    positions such as prompts need tokens but no target rows.
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
        **{key: inputs[key][keep] for key in TOKEN_FIELDS},
        "segment_lengths": lengths.to(torch.int32),
        "segment_counts": torch.tensor(segment_counts, dtype=torch.int32),
    }
    rows = keep & (weights != 0)
    return record, values[rows], indices[rows], tail[rows]
