# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Auto-tuning for inference memory parameters.

Profiles activation memory and compute throughput during CUDA graph warmup,
then solves for optimal (max_requests, max_tokens) based on available GPU memory
and the memory-bound → compute-bound transition (the "elbow").

See NOTES.md for design rationale.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class AutotuneProfile:
    """Profiling data collected during CUDA graph warmup.

    Each entry corresponds to one CUDA graph batch dimension's forward pass.
    """

    # Per-batch-dimension measurements (parallel lists).
    token_counts: List[int] = field(default_factory=list)
    peak_activation_bytes: List[int] = field(default_factory=list)
    step_times_ms: List[float] = field(default_factory=list)

    # Memory accounting constants from the context.
    block_size_bytes: int = 0
    mamba_memory_per_request: int = 0
    max_kv_block_count: int = 0
    per_request_bytes: int = 0
    per_token_bytes: int = 0

    # GPU memory state captured before context allocation.
    gpu_total_bytes: int = 0
    memory_after_model_load_bytes: int = 0

    def add_sample(self, token_count: int, peak_bytes: int, time_ms: float):
        """Record one profiling sample."""
        self.token_counts.append(token_count)
        self.peak_activation_bytes.append(peak_bytes)
        self.step_times_ms.append(time_ms)


def _build_activation_interpolator(
    token_counts: List[int], peak_bytes: List[int]
) -> Dict[int, int]:
    """Deduplicate and build a sorted mapping of token_count → peak_activation_bytes.

    When multiple samples have the same token_count, keep the maximum peak_bytes.
    """
    merged: Dict[int, int] = {}
    for tc, pb in zip(token_counts, peak_bytes):
        if tc not in merged or pb > merged[tc]:
            merged[tc] = pb
    return dict(sorted(merged.items()))


def _interpolate(table: Dict[int, int], x: int) -> int:
    """Linearly interpolate or extrapolate from a sorted table."""
    keys = list(table.keys())
    vals = list(table.values())

    if x <= keys[0]:
        if len(keys) >= 2:
            # Extrapolate from first two points.
            slope = (vals[1] - vals[0]) / max(keys[1] - keys[0], 1)
            return max(0, int(vals[0] + slope * (x - keys[0])))
        return vals[0]
    if x >= keys[-1]:
        if len(keys) >= 2:
            slope = (vals[-1] - vals[-2]) / max(keys[-1] - keys[-2], 1)
            return int(vals[-1] + slope * (x - keys[-1]))
        return vals[-1]

    # Binary search for the enclosing interval.
    for i in range(len(keys) - 1):
        if keys[i] <= x <= keys[i + 1]:
            t = (x - keys[i]) / max(keys[i + 1] - keys[i], 1)
            return int(vals[i] + t * (vals[i + 1] - vals[i]))

    return vals[-1]  # fallback


def _inverse_interpolate(table: Dict[int, int], target_bytes: int) -> int:
    """Find the largest token_count whose activation memory ≤ target_bytes."""
    keys = list(table.keys())
    vals = list(table.values())

    # Walk from largest to smallest token count.
    for i in range(len(keys) - 1, -1, -1):
        if vals[i] <= target_bytes:
            # Try to extrapolate a bit beyond this sample.
            if i < len(keys) - 1:
                slope = (vals[i + 1] - vals[i]) / max(keys[i + 1] - keys[i], 1)
                if slope > 0:
                    extra = int((target_bytes - vals[i]) / slope)
                    return keys[i] + extra
            return keys[i]

    # Even the smallest sample exceeds the budget.
    return keys[0]


def _find_compute_elbow(
    token_counts: List[int], step_times_ms: List[float]
) -> int:
    """Find the token count where throughput (tokens/ms) plateaus.

    Uses a simple threshold: the elbow is the smallest token count where
    throughput reaches 90% of the maximum observed throughput.
    """
    if not token_counts:
        return 1

    throughput = {}
    for tc, t in zip(token_counts, step_times_ms):
        if t > 0:
            tp = tc / t
            if tc not in throughput or tp > throughput[tc]:
                throughput[tc] = tp

    if not throughput:
        return 1

    max_throughput = max(throughput.values())
    threshold = 0.90 * max_throughput

    # Find the smallest token count that reaches the threshold.
    for tc in sorted(throughput.keys()):
        if throughput[tc] >= threshold:
            return tc

    return max(throughput.keys())


def compute_optimal_params(
    profile: AutotuneProfile,
    tp_size: int = 1,
    request_rounder: int = 4,
    avg_sequence_length: Optional[int] = None,
    safety_margin_fraction: float = 0.10,
) -> Tuple[int, int, float]:
    """Solve for optimal (max_requests, max_tokens, buffer_size_gb) from profiling data.

    Args:
        profile: Profiling data from CUDA graph warmup.
        tp_size: Tensor parallel size (for alignment).
        request_rounder: Request count alignment (typically 4).
        avg_sequence_length: Expected average sequence length for freed-cache
            estimate. If None, defaults to max_sequence_length // 2.
        safety_margin_fraction: Fraction of free GPU memory to reserve for
            CUDA graph workspace, TMS VMM overhead, and allocator
            fragmentation. Defaults to 0.10 (10%).

    Returns:
        (max_requests, max_tokens, buffer_size_gb) tuple.
    """
    # Build interpolation table from profiling data.
    activation_table = _build_activation_interpolator(
        profile.token_counts, profile.peak_activation_bytes
    )

    if not activation_table:
        raise ValueError("No profiling data collected during CUDA graph warmup")

    # Find the compute elbow.
    elbow = _find_compute_elbow(profile.token_counts, profile.step_times_ms)

    # Available GPU memory = total - model weights (approximated by memory
    # used after model load, before context allocation).
    gpu_free_raw = profile.gpu_total_bytes - profile.memory_after_model_load_bytes
    safety_margin = int(gpu_free_raw * safety_margin_fraction)
    gpu_free = gpu_free_raw - safety_margin

    logging.info(
        "Autotune: GPU total %d MB, after model load %d MB, "
        "free %d MB (reserving %d MB safety margin = %d MB usable)",
        profile.gpu_total_bytes // (1024 ** 2),
        profile.memory_after_model_load_bytes // (1024 ** 2),
        gpu_free_raw // (1024 ** 2),
        safety_margin // (1024 ** 2),
        gpu_free // (1024 ** 2),
    )
    logging.info("Autotune: compute elbow at %d tokens", elbow)

    # Step 1: Find max_requests via decode constraint.
    # context_state(R) + activation_memory(R) ≤ gpu_free
    # context_state(R) = block_count(R) * block_size_bytes
    #                   + R * mamba_memory_per_request
    #                   + R * per_request_bytes
    #                   + R * per_token_bytes  (since max_tokens = max_requests)
    max_requests = 0
    best_block_count = 0
    # Search from high to low in steps aligned to request_rounder and tp_size.
    alignment = max(tp_size, request_rounder)
    # Upper bound: assume all gpu_free goes to blocks (ignoring everything else).
    upper_bound = gpu_free // max(profile.block_size_bytes, 1)
    upper_bound = (upper_bound // alignment) * alignment

    for candidate in range(upper_bound, 0, -alignment):
        activation_bytes = _interpolate(activation_table, candidate)
        metadata_bytes = (
            candidate * profile.per_request_bytes
            + candidate * profile.per_token_bytes  # max_tokens = max_requests
            + candidate * profile.mamba_memory_per_request
        )
        remaining_for_kv = gpu_free - activation_bytes - metadata_bytes
        if remaining_for_kv <= 0:
            continue
        block_count = remaining_for_kv // profile.block_size_bytes
        if block_count >= 2:  # need ≥ 1 active + 1 dummy
            max_requests = candidate
            best_block_count = block_count
            break

    if max_requests == 0:
        logging.warning(
            "Autotune: could not find valid max_requests. "
            "Falling back to minimum (alignment=%d).",
            alignment,
        )
        max_requests = alignment
        best_block_count = 2

    # Step 2: Determine max_tokens.
    if max_requests >= elbow:
        # Case 1: saturated — max_tokens based on freed cache from completed requests.
        if avg_sequence_length is None:
            max_kv_block_count = profile.max_kv_block_count
            avg_sequence_length = max_kv_block_count * profile.block_size_bytes // 2
        kv_bytes_per_token = profile.block_size_bytes // max(
            profile.max_kv_block_count, 1
        )
        typical_freed_cache = avg_sequence_length * kv_bytes_per_token
        decode_activation = _interpolate(activation_table, max_requests)
        max_activation_budget = decode_activation + typical_freed_cache
        max_tokens = _inverse_interpolate(activation_table, max_activation_budget)
    else:
        # Case 2: undersaturated — max_tokens at the elbow.
        max_tokens = elbow

    # Enforce invariant: max_tokens >= max_requests.
    max_tokens = max(max_tokens, max_requests)

    # Align max_tokens.
    max_tokens = (max_tokens // tp_size) * tp_size
    max_tokens = max(max_tokens, max_requests)  # re-enforce after alignment

    # Derive buffer_size_gb from the block count that fits.
    buffer_size_bytes = best_block_count * profile.block_size_bytes
    buffer_size_gb = buffer_size_bytes / (1024 ** 3)

    logging.info(
        "Autotune result: max_requests=%d, max_tokens=%d, buffer_size_gb=%.2f "
        "(%d blocks), case=%s (elbow=%d)",
        max_requests,
        max_tokens,
        buffer_size_gb,
        best_block_count,
        "saturated" if max_requests >= elbow else "undersaturated",
        elbow,
    )

    return max_requests, max_tokens, buffer_size_gb
