# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical Mamba-layer spans."""

from tests.unit_tests.telemetry.mcore_span_ast import (
    MAMBA_LAYER_PATH,
    assert_detail_span_counts,
    assert_layer_forward_number,
    assert_uses_shim_not_legacy_fallbacks,
)


def test_mamba_layer_call_sites_use_constants_and_detail_group():
    assert_detail_span_counts((MAMBA_LAYER_PATH,), {"SPAN_LAYER_FORWARD": 1, "SPAN_LAYER_MAMBA": 1})
    assert_uses_shim_not_legacy_fallbacks(MAMBA_LAYER_PATH)


def test_mamba_layer_forward_span_carries_canonical_layer_number():
    assert_layer_forward_number(MAMBA_LAYER_PATH)
