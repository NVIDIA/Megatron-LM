# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from tests.functional_tests.python_test_utils.compute_golden_statistics import (
    _aggregate_training_results,
    _extract_result_path_from_log,
)


def test_aggregate_training_results_accepts_precision_metadata():
    aggregated = {}
    data = {"lm loss": {"value_precision": "full", "values": {"1": 1.23456789}}}

    _aggregate_training_results(data, aggregated, run_index=0)

    assert aggregated == {"lm loss": {"1": [1.23456789]}}


def test_searches_later_result_markers_after_an_incomplete_run(tmp_path):
    completed_run = tmp_path / 'runs' / 'completed'
    completed_run.mkdir(parents=True)
    result = completed_run / 'golden_values.json'
    result.write_text('{}')
    log = tmp_path / 'job.out'
    log.write_text(
        'This test wrote results into /opt/megatron-lm/runs/incomplete\n'
        'This test wrote results into /opt/megatron-lm/runs/completed\n'
    )

    assert _extract_result_path_from_log(log, str(tmp_path)) == str(result)
