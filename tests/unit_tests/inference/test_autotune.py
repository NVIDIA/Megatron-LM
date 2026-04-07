# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math

import pytest

from megatron.core.inference.autotune import (
    AutotuneProfile,
    _build_activation_interpolator,
    _interpolate,
    compute_optimal_params,
)

GB = 1024 ** 3
MB = 1024 ** 2


def _make_profile(
    gpu_total_gb=80,
    model_gb=30,
    block_size_bytes=512 * 1024,
    max_kv_block_count=80,
    mamba_memory_per_request=0,
    per_request_bytes=400,
    per_token_bytes=48,
    runtime_overhead_per_request=1 * MB,
    samples=None,
):
    """Build an AutotuneProfile with synthetic data.

    Default samples: linear activation curve, 1 MB per token of batch.
    """
    p = AutotuneProfile(
        gpu_total_bytes=int(gpu_total_gb * GB),
        memory_after_model_load_bytes=int(model_gb * GB),
        block_size_bytes=block_size_bytes,
        max_kv_block_count=max_kv_block_count,
        mamba_memory_per_request=mamba_memory_per_request,
        per_request_bytes=per_request_bytes,
        per_token_bytes=per_token_bytes,
        runtime_overhead_per_request=runtime_overhead_per_request,
    )
    for tc, peak, ms in (samples or [(32, 32 * MB, 1.0), (256, 256 * MB, 4.0)]):
        p.add_sample(tc, peak, ms)
    return p


class TestAutotune:
    """Tests for the autotune solver and interpolation helpers."""

    @pytest.mark.parametrize(
        "table, x, expected",
        [
            # Exact match
            ({10: 100, 20: 200}, 10, 100),
            # Midpoint interpolation
            ({10: 100, 20: 200}, 15, 150),
            # Extrapolate below
            ({10: 100, 20: 200}, 5, 50),
            # Extrapolate above
            ({10: 100, 20: 200}, 30, 300),
            # Single point — no extrapolation possible
            ({10: 100}, 5, 100),
            ({10: 100}, 20, 100),
            # Three points, interpolate in second segment
            ({10: 100, 20: 200, 40: 600}, 30, 400),
            # Extrapolate below clamps to 0
            ({10: 100, 20: 200}, -100, 0),
        ],
    )
    def test_interpolation(self, table, x, expected):
        assert _interpolate(table, x) == expected

    @pytest.mark.parametrize(
        "desc, profile_kwargs, tp_size",
        [
            ("baseline_80gb", {}, 1),
            ("small_gpu_40gb", dict(gpu_total_gb=40), 1),
            ("large_gpu_160gb", dict(gpu_total_gb=160), 1),
            ("with_mamba", dict(mamba_memory_per_request=4 * MB), 1),
            ("heavy_mamba", dict(mamba_memory_per_request=16 * MB), 1),
            ("tp8_alignment", {}, 8),
            ("short_sequences", dict(max_kv_block_count=4), 1),
            ("long_sequences", dict(max_kv_block_count=256), 1),
            ("tiny_gpu_barely_fits", dict(gpu_total_gb=32, model_gb=30), 1),
        ],
    )
    def test_solver_constraints(self, desc, profile_kwargs, tp_size):
        """Verify the solver's output satisfies all structural constraints."""
        profile = _make_profile(**profile_kwargs)
        max_requests, max_tokens, buffer_size_gb = compute_optimal_params(
            profile, tp_size=tp_size, safety_margin_fraction=0.10,
        )

        alignment = max(tp_size, 4)
        blocks_per_request = max(1, math.ceil(profile.max_kv_block_count * 0.6827))

        # Alignment
        assert max_requests % alignment == 0, f"{desc}: max_requests not aligned"
        assert max_requests > 0, f"{desc}: max_requests must be positive"
        assert max_tokens >= max_requests, f"{desc}: max_tokens < max_requests"

        # buffer_size_gb must cover KV blocks + mamba
        buffer_bytes = buffer_size_gb * GB
        mamba_bytes = max_requests * profile.mamba_memory_per_request
        kv_bytes = buffer_bytes - mamba_bytes
        assert kv_bytes > 0, f"{desc}: no room for KV blocks"
        block_count = int(kv_bytes // profile.block_size_bytes)
        assert block_count >= max_requests * blocks_per_request + 1, (
            f"{desc}: not enough blocks ({block_count}) for "
            f"{max_requests} reqs * {blocks_per_request} blocks/req + 1"
        )

        # Budget: CG pool + runtime overhead + metadata + buffer must fit in gpu_free
        gpu_free = (profile.gpu_total_bytes - profile.memory_after_model_load_bytes) * 0.95
        activation_table = _build_activation_interpolator(
            profile.token_counts, profile.peak_activation_bytes
        )
        cg_pool = max(0, _interpolate(activation_table, max_requests))
        runtime_overhead = max_requests * profile.runtime_overhead_per_request
        metadata = max_requests * (
            profile.per_request_bytes + profile.per_token_bytes
            + profile.mamba_memory_per_request
        )
        total_used = cg_pool + runtime_overhead + metadata + buffer_bytes
        assert total_used <= gpu_free * 1.01, (  # 1% tolerance for rounding
            f"{desc}: total {total_used / GB:.2f} GB exceeds budget {gpu_free / GB:.2f} GB"
        )

    def test_sanity_checks(self):
        """Monotonicity and edge cases."""
        # More GPU memory → more requests
        r_small, _, _ = compute_optimal_params(_make_profile(gpu_total_gb=40))
        r_large, _, _ = compute_optimal_params(_make_profile(gpu_total_gb=80))
        assert r_large >= r_small

        # Mamba cost reduces capacity
        r_no_mamba, _, _ = compute_optimal_params(_make_profile(mamba_memory_per_request=0))
        r_mamba, _, _ = compute_optimal_params(_make_profile(mamba_memory_per_request=8 * MB))
        assert r_mamba <= r_no_mamba

        # Longer sequences → fewer requests
        r_short, _, _ = compute_optimal_params(_make_profile(max_kv_block_count=10))
        r_long, _, _ = compute_optimal_params(_make_profile(max_kv_block_count=200))
        assert r_long <= r_short

        # Empty profile raises
        with pytest.raises(ValueError, match="No profiling data"):
            compute_optimal_params(AutotuneProfile(
                gpu_total_bytes=80 * GB, memory_after_model_load_bytes=30 * GB,
            ))
