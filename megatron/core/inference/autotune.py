# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Auto-tuning for inference memory parameters.

Profiles activation memory and compute throughput at startup, then solves for
optimal (max_requests, max_tokens, buffer_size_gb) based on available GPU memory.

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

    # Empirically measured runtime overhead per request (sampling + logprobs).
    # Computed as (full_step_peak - forward_only_peak) / batch_size.
    runtime_overhead_per_request: int = 0

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
            return max(0, int(vals[-1] + slope * (x - keys[-1])))
        return vals[-1]

    # Binary search for the enclosing interval.
    for i in range(len(keys) - 1):
        if keys[i] <= x <= keys[i + 1]:
            t = (x - keys[i]) / max(keys[i + 1] - keys[i], 1)
            return int(vals[i] + t * (vals[i + 1] - vals[i]))

    return vals[-1]  # fallback


def compute_optimal_params(
    profile: AutotuneProfile,
    tp_size: int = 1,
    request_rounder: int = 4,
    reserved_memory_bytes: int = 0,
) -> Tuple[int, int, float]:
    """Solve for optimal (max_requests, max_tokens, buffer_size_gb) from profiling data.

    The solver maximizes max_requests (concurrent request capacity) given the
    GPU memory budget, then sets max_tokens = max_requests (no prefill
    optimization). The CUDA graph pool cost at max_requests is subtracted
    from the budget since it is permanently reserved.

    Args:
        profile: Profiling data from CUDA graph warmup.
        tp_size: Tensor parallel size (for alignment).
        request_rounder: Request count alignment (typically 4).
        reserved_memory_bytes: Bytes to reserve for non-inference use (e.g.,
            RL training optimizer states, gradients, caching allocator
            headroom). Subtracted from the available GPU memory budget.

    Returns:
        (max_requests, max_tokens, buffer_size_gb) tuple.
    """
    # Build interpolation table from profiling data.
    activation_table = _build_activation_interpolator(
        profile.token_counts, profile.peak_activation_bytes
    )

    if not activation_table:
        raise ValueError("No profiling data collected during autotune")

    # Available GPU memory = total - model weights - reserved.
    gpu_free_raw = profile.gpu_total_bytes - profile.memory_after_model_load_bytes
    gpu_free = gpu_free_raw - reserved_memory_bytes

    logging.info(
        "Autotune: GPU total %d MB, after model load %d MB, "
        "free %d MB (reserving %d MB = %d MB usable)",
        profile.gpu_total_bytes // (1024 ** 2),
        profile.memory_after_model_load_bytes // (1024 ** 2),
        gpu_free_raw // (1024 ** 2),
        reserved_memory_bytes // (1024 ** 2),
        gpu_free // (1024 ** 2),
    )
    logging.info(
        "Autotune: profile params: block_size=%d bytes, mamba/req=%d bytes (%.2f MB), "
        "per_req=%d bytes, per_token=%d bytes, max_kv_blocks=%d, "
        "blocks_per_request=%d (31.73%% of max), "
        "runtime_overhead/req=%d bytes (%.2f MB)",
        profile.block_size_bytes,
        profile.mamba_memory_per_request,
        profile.mamba_memory_per_request / (1024 ** 2),
        profile.per_request_bytes,
        profile.per_token_bytes,
        profile.max_kv_block_count,
        max(1, math.ceil(profile.max_kv_block_count * 0.3173)),
        profile.runtime_overhead_per_request,
        profile.runtime_overhead_per_request / (1024 ** 2),
    )

    # Find max_requests via decode constraint.
    # CG_pool(R) + runtime_overhead(R) + metadata(R) + kv_blocks * block_size ≤ gpu_free
    #
    # CG_pool = forward activation (permanent, inside CUDA graph)
    # runtime_overhead = empirically measured transient memory for sampling,
    #   log-probs, and other non-graphed ops (freed each step, but must fit
    #   in available memory during peak usage)
    max_requests = 0
    best_block_count = 0
    alignment = max(tp_size, request_rounder)
    # Upper bound: assume all gpu_free goes to blocks (ignoring everything else).
    upper_bound = gpu_free // max(profile.block_size_bytes, 1)
    upper_bound = (upper_bound // alignment) * alignment

    # Estimate blocks needed per request: assume typical sequence length is
    # ~1 standard deviation below the mean (31.73% of max). This favors
    # more concurrent request slots over KV block headroom: requests that
    # grow beyond this will pause until blocks free up.
    blocks_per_request = max(1, math.ceil(profile.max_kv_block_count * 0.3173))

    # The CUDA graph pool is sized by the largest graph's activation.
    # With max_tokens = max_requests, the CG pool = activation(R).
    for candidate in range(upper_bound, 0, -alignment):
        activation_bytes = max(0, _interpolate(activation_table, candidate))
        cuda_graph_pool = activation_bytes
        runtime_overhead = candidate * profile.runtime_overhead_per_request
        metadata_bytes = (
            candidate * profile.per_request_bytes
            + candidate * profile.per_token_bytes
            + candidate * profile.mamba_memory_per_request
        )
        remaining_for_kv = gpu_free - cuda_graph_pool - runtime_overhead - metadata_bytes
        if remaining_for_kv <= 0:
            continue
        block_count = remaining_for_kv // profile.block_size_bytes
        # Each request needs blocks_per_request KV blocks at typical
        # sequence length, plus 1 dummy block.
        if block_count >= candidate * blocks_per_request + 1:
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

    # max_tokens = max_requests (no prefill optimization in the base solver).
    max_tokens = max_requests

    # Align max_tokens to TOKEN_ROUNDER (and TP) so that round_up_tokens()
    # can never produce a value exceeding max_tokens: the token buffers are
    # allocated for max_tokens, and any padded count beyond that silently
    # truncates the slice, causing shape mismatches downstream.
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    token_rounder = math.ceil(DynamicInferenceContext.TOKEN_ROUNDER / tp_size) * tp_size
    # Round max_tokens UP to the next token_rounder boundary so that
    # max_tokens >= max_requests still holds (max_requests is already
    # aligned to max(tp_size, request_rounder) which may be smaller).
    max_tokens = math.ceil(max_requests / token_rounder) * token_rounder

    # Derive buffer_size_gb. The context's auto-derive path expects
    # buffer_size_gb to contain BOTH mamba states AND KV blocks combined —
    # it carves mamba out of the buffer proportionally.
    kv_bytes = best_block_count * profile.block_size_bytes
    mamba_bytes = max_requests * profile.mamba_memory_per_request
    buffer_size_bytes = kv_bytes + mamba_bytes
    buffer_size_gb = buffer_size_bytes / (1024 ** 3)

    logging.info(
        "Autotune result: max_requests=%d, max_tokens=%d, buffer_size_gb=%.2f "
        "(%d blocks)",
        max_requests,
        max_tokens,
        buffer_size_gb,
        best_block_count,
    )

    return max_requests, max_tokens, buffer_size_gb
