# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical DDP gradient-sync span."""

from tests.unit_tests.telemetry.mcore_span_ast import (
    DDP_PATH,
    assert_detail_span_counts,
    assert_uses_shim_not_legacy_fallbacks,
    nested_function,
)


def test_ddp_call_sites_use_constants_and_detail_group():
    assert_detail_span_counts((DDP_PATH,), {"SPAN_GRAD_SYNC_FINISH": 1})
    assert_uses_shim_not_legacy_fallbacks(DDP_PATH)


def test_ddp_start_grad_sync_is_not_instrumented():
    # Pipeline schedules own nv.mcore.grad_sync.start at their dispatch sites.
    assert not nested_function(DDP_PATH, "start_grad_sync").decorator_list
