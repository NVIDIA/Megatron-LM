# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from tests.functional_tests.python_test_utils.compute_golden_statistics import (
    _aggregate_training_results,
)


def test_aggregate_training_results_accepts_precision_metadata():
    aggregated = {}
    data = {"lm loss": {"value_precision": "full", "values": {"1": 1.23456789}}}

    _aggregate_training_results(data, aggregated, run_index=0)

    assert aggregated == {"lm loss": {"1": [1.23456789]}}
