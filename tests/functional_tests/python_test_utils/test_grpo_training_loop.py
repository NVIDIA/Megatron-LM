# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import logging
import math
from statistics import median

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_grpo_training_loop(golden_values_path: str, test_values_path: str) -> None:

    with open(golden_values_path, 'r') as f1, open(test_values_path, 'r') as f2:
        golden_values_content = f1.read()
        tensorboard_content = f2.read()

    output_groundtruth = json.loads(golden_values_content)

    if isinstance(output_groundtruth, str):
        # Handle JSONL output, assume only one line in this case.
        output_groundtruth = json.loads(output_groundtruth)

    output_current = json.loads(tensorboard_content)
    if isinstance(output_current, str):
        # Handle JSONL output, assume only one line in this case.
        output_current = json.loads(output_current)

    assert set(output_groundtruth.keys()).issuperset(
        set(output_current.keys())
    ), f"Some IDs from groundtruth are missing in current: {output_groundtruth.keys()} vs {output_current.keys()}"
    if set(output_groundtruth.keys()) != set(output_current.keys()):
        logger.warning(
            f"Some IDs from groundtruth are missing in output, only the subset of ids in groundtruth will be tested: {output_groundtruth.keys()} vs {output_current.keys()}"
        )
    assert len(output_groundtruth) > 0, "No test performed for output"

    if "iteration-time" in output_groundtruth.keys():

        # First warmup iteration is excluded from iteration-time statistics.
        iteration_time_sampled = median(
            [l for l in output_current["iteration-time"]['values'].values()][1:]
        )
        iteration_time_golden = median(
            [l for l in output_groundtruth["iteration-time"]['values'].values()][1:]
        )

        # 10% is empirically observed to be within hardware variance.
        assert (
            0.9 * iteration_time_golden <= iteration_time_sampled <= 1.2 * iteration_time_golden
        ), (
            f"Iteration time {iteration_time_sampled} ms not within 10% below or 20% above "
            f"golden value ~{iteration_time_golden} ms. "
            f"Sampled: {output_current['iteration-time']} ms. "
            f"Please update golden values in the functional tests if this is expected."
        )

        output_groundtruth.pop('iteration-time')
