# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Training metric instruments (megatron.training.* namespace).

Self-contained copy of the metric recording logic from nemo.lens so that
Megatron-LM can emit training metrics without a hard dependency on nemo-lens.
"""

from __future__ import annotations

import logging
import weakref

# Metric name constants (mirrors nemo.lens.semconv).
MEGATRON_TRAINING_STEP_DURATION_MS = "megatron.training.step_duration_ms"
MEGATRON_TRAINING_LOSS = "megatron.training.loss"
MEGATRON_TRAINING_THROUGHPUT_TFLOPS = "megatron.training.throughput_tflops"
MEGATRON_TRAINING_GRAD_NORM = "megatron.training.grad_norm"
MEGATRON_TRAINING_SKIPPED_ITERS = "megatron.training.skipped_iters"
MEGATRON_TRAINING_LEARNING_RATE = "megatron.training.learning_rate"
MEGATRON_TRAINING_TOKENS_PER_SEC = "megatron.training.tokens_per_sec"
MEGATRON_TRAINING_MEMORY_ALLOCATED_GB = "megatron.training.memory_allocated_gb"

try:
    from opentelemetry import metrics
except ImportError:
    metrics = None

_logger = logging.getLogger(__name__)
_TRAINING_INSTRUMENTS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _get_training_instruments(meter) -> dict:
    instruments = _TRAINING_INSTRUMENTS.get(meter)
    if instruments is None:
        instruments = {
            "step_duration_ms": meter.create_histogram(
                name=MEGATRON_TRAINING_STEP_DURATION_MS,
                unit="ms",
                description="Duration of one training step in milliseconds.",
            ),
            "loss": meter.create_gauge(
                name=MEGATRON_TRAINING_LOSS,
                description="Training loss value at each log interval.",
            ),
            "throughput_tflops": meter.create_gauge(
                name=MEGATRON_TRAINING_THROUGHPUT_TFLOPS,
                description="Training throughput in TFLOP/s/GPU.",
            ),
            "grad_norm": meter.create_gauge(
                name=MEGATRON_TRAINING_GRAD_NORM,
                description="Global gradient norm.",
            ),
            "skipped_iters": meter.create_counter(
                name=MEGATRON_TRAINING_SKIPPED_ITERS,
                description="Number of training iterations skipped.",
            ),
            "learning_rate": meter.create_gauge(
                name=MEGATRON_TRAINING_LEARNING_RATE,
                description="Current learning rate.",
            ),
            "tokens_per_sec": meter.create_gauge(
                name=MEGATRON_TRAINING_TOKENS_PER_SEC,
                description="Training throughput in tokens/second.",
            ),
            "memory_allocated_gb": meter.create_gauge(
                name=MEGATRON_TRAINING_MEMORY_ALLOCATED_GB,
                description="Peak GPU memory allocated in GB.",
            ),
        }
        _TRAINING_INSTRUMENTS[meter] = instruments
    return instruments


def record_training_metrics(
    meter,
    step_duration_ms: float | None = None,
    loss: float | None = None,
    throughput_tflops: float | None = None,
    grad_norm: float | None = None,
    skipped_iters: int | None = None,
    learning_rate: float | None = None,
    tokens_per_sec: float | None = None,
    memory_allocated_gb: float | None = None,
) -> None:
    """Record training metrics to the OTel meter.

    All arguments are optional; ``None`` values are silently skipped.
    Safe to call when telemetry is disabled (meter is no-op).

    If ``opentelemetry`` is not installed, this function is a no-op.
    """
    if metrics is None:
        return

    try:
        instruments = _get_training_instruments(meter)
    except Exception:
        _logger.warning("Failed to create training metric instruments", exc_info=True)
        return

    if step_duration_ms is not None:
        instruments["step_duration_ms"].record(step_duration_ms)
    if loss is not None:
        instruments["loss"].set(loss)
    if throughput_tflops is not None:
        instruments["throughput_tflops"].set(throughput_tflops)
    if grad_norm is not None:
        instruments["grad_norm"].set(float(grad_norm))
    if skipped_iters is not None and skipped_iters > 0:
        instruments["skipped_iters"].add(skipped_iters)
    if learning_rate is not None:
        instruments["learning_rate"].set(learning_rate)
    if tokens_per_sec is not None:
        instruments["tokens_per_sec"].set(tokens_per_sec)
    if memory_allocated_gb is not None:
        instruments["memory_allocated_gb"].set(memory_allocated_gb)
