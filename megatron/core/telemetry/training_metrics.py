# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Failure-isolated processed-token metric owned by Megatron-LM."""

from __future__ import annotations

import logging
import math
import weakref
from typing import Any

NV_DL_TRAINING_TOKENS_PROCESSED = "nv.dl.training.tokens.processed"

try:
    from opentelemetry import metrics
except ImportError:
    metrics = None

_logger = logging.getLogger(__name__)
_TRAINING_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_training_instruments(meter: Any) -> dict:
    instruments = _TRAINING_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "tokens_processed": meter.create_counter(
                name=NV_DL_TRAINING_TOKENS_PROCESSED,
                unit="{token}",
                description="Global tokens processed per executed training iteration.",
            )
        }
        _TRAINING_INSTRUMENTS[meter] = instruments
    return instruments


def processed_token_increment(
    global_batch_size: int, sequence_length: int, packed_token_count: float | None = None
) -> int | None:
    """Return the global integer token increment for one executed iteration.

    Packed training passes the already-global host value materialized by the
    existing sequence-length statistics path. Invalid packed values are skipped
    so telemetry cannot affect training.
    """
    if packed_token_count is None:
        return int(global_batch_size) * int(sequence_length)
    if (
        isinstance(packed_token_count, bool)
        or not isinstance(packed_token_count, (int, float))
        or not math.isfinite(packed_token_count)
        or not float(packed_token_count).is_integer()
        or packed_token_count < 0
    ):
        _logger.warning("Skipping invalid packed processed-token count %r", packed_token_count)
        return None
    return int(packed_token_count)


def record_processed_tokens(meter: Any, token_count: int | None = None) -> None:
    """Add one global token increment to this rank's monotonic counter.

    If ``opentelemetry`` is not installed, this function is a no-op.
    """
    if metrics is None or token_count is None:
        return

    try:
        instruments = _get_training_instruments(meter)
    except Exception:
        _logger.warning("Failed to create training metric instruments", exc_info=True)
        return

    try:
        instruments["tokens_processed"].add(token_count)
    except Exception:
        _logger.warning("Failed to record processed-token metric", exc_info=True)


def record_processed_tokens_for_iteration(
    telemetry_handle: Any | None,
    *,
    global_batch_size: int,
    sequence_length: int,
    packed_token_count: float | None = None,
) -> None:
    """Record tokens for a model-work iteration committed by the trainer."""
    if telemetry_handle is None:
        return

    try:
        if not telemetry_handle.is_exporting:
            return
        meter = telemetry_handle.meter
    except Exception:
        _logger.warning("Failed to inspect telemetry handle for processed tokens", exc_info=True)
        return

    token_count = processed_token_increment(global_batch_size, sequence_length, packed_token_count)
    record_processed_tokens(meter, token_count)
