# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical distributed-checkpoint load spans."""

from tests.unit_tests.telemetry.mcore_span_ast import (
    FULLY_PARALLEL_PATH,
    TORCH_CHECKPOINT_PATH,
    assert_detail_span_counts,
    assert_uses_shim_not_legacy_fallbacks,
)


def test_checkpoint_load_call_sites_use_constants_and_detail_group():
    assert_detail_span_counts(
        (FULLY_PARALLEL_PATH, TORCH_CHECKPOINT_PATH),
        {
            "SPAN_CHECKPOINT_LOAD_READ_METADATA": 1,
            "SPAN_CHECKPOINT_LOAD_MCORE_TO_PYT": 1,
            "SPAN_CHECKPOINT_LOAD_FULLY_PARALLEL_PLAN": 1,
            "SPAN_CHECKPOINT_LOAD_IO": 1,
            "SPAN_CHECKPOINT_LOAD_EXCHANGE": 1,
        },
    )
    assert_uses_shim_not_legacy_fallbacks(FULLY_PARALLEL_PATH)
    assert_uses_shim_not_legacy_fallbacks(TORCH_CHECKPOINT_PATH)
