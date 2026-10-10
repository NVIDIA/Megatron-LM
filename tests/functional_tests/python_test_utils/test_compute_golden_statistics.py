# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from tests.functional_tests.python_test_utils.compute_golden_statistics import (
    _aggregate_training_results,
)


def test_aggregate_training_results_accepts_precision_metadata():
    aggregated = {}
    data = {"lm loss": {"value_precision": "full", "values": {"1": 1.23456789}}}

    _aggregate_training_results(data, aggregated, run_index=0)

    assert aggregated == {"lm loss": {"1": [1.23456789]}}


def test_aggregate_training_results_excludes_infinite_samples():
    aggregated = {}
    data = {
        'lm loss': {'values': {'1': 1.25, '2': float('inf'), '3': '-inf', '4': 'nan'}},
        'iteration-time': {'values': {'1': 0.5, '2': 'inf'}},
    }

    _aggregate_training_results(data, aggregated, run_index=0)

    assert aggregated == {
        'lm loss': {'1': [1.25]},
        'iteration-time': {'1': [0.5], '_all_values_run_0': [0.5]},
    }
