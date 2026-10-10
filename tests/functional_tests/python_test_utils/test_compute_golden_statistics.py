# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from tests.functional_tests.python_test_utils.compute_golden_statistics import (
    _aggregate_training_results,
    find_result_json_files,
)


def test_aggregate_training_results_accepts_precision_metadata():
    aggregated = {}
    data = {"lm loss": {"value_precision": "full", "values": {"1": 1.23456789}}}

    _aggregate_training_results(data, aggregated, run_index=0)

    assert aggregated == {"lm loss": {"1": [1.23456789]}}


def test_discovered_batch_results_do_not_count_one_run_twice(tmp_path):
    run = tmp_path / 'runs' / 'complete'
    run.mkdir(parents=True)
    result = run / 'golden_values.json'
    result.write_text('{}')
    logs = tmp_path / 'logs'
    logs.mkdir()
    for name in ['rank-0.out', 'rank-1.out']:
        (logs / name).write_text('This test wrote results into /opt/megatron-lm/runs/complete\n')

    assert find_result_json_files(str(logs), str(tmp_path)) == [str(result)]
