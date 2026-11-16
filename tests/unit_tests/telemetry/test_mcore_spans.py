# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Canonical MCore detail span names owned by the telemetry shim."""

from tests.unit_tests.telemetry.mcore_span_ast import (
    EXPECTED_GRAD_SYNC_SITES,
    EXPECTED_MCORE_ATTRIBUTES,
    EXPECTED_MCORE_SPANS,
    TELEMETRY_PATH,
    module_string_constants,
)


def test_mcore_constants_match_canonical_names():
    constants = module_string_constants(TELEMETRY_PATH)

    assert {name: constants[name] for name in EXPECTED_MCORE_SPANS} == EXPECTED_MCORE_SPANS
    assert {
        name: constants[name] for name in EXPECTED_MCORE_ATTRIBUTES
    } == EXPECTED_MCORE_ATTRIBUTES
    assert {name: constants[name] for name in EXPECTED_GRAD_SYNC_SITES} == EXPECTED_GRAD_SYNC_SITES
