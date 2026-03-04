# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from typing import Dict

import yaml

from tests.functional_tests.python_test_utils import common, test_pretraining_regular_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_resume_checkpoint_pipeline(
    compare_approximate_results: bool,
    actual_values_first_run: Dict[str, common.GoldenValueMetric],
    actual_values_second_run: Dict[str, common.GoldenValueMetric],
    train_iters: int,
    model_config_path: str,
):
    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    checks_types = (
        model_config["METRICS"] if "METRICS" in model_config else ["lm loss", "num-zeros"]
    )
    checks = {
        metric: test_pretraining_regular_pipeline.CHECK_THRESHOLDS[metric]
        for metric in checks_types
    }

    if (
        len(
            missing_metrics := [
                golden_metric
                for golden_metric in checks.keys()
                if golden_metric not in actual_values_first_run.keys()
            ]
        )
        > 0
    ):
        logger.error(
            f"The following metrics are required but not logged during training: {', '.join(missing_metrics)}"
        )
        assert False

    # actual_values_second_run is NaN for the first 50 steps. We want to replace those
    # with the first 50 steps of actual_values_first_run

    actual_values_first_run = {
        metric_name: metric_values
        for (metric_name, metric_values) in actual_values_first_run.items()
        if metric_name in checks.keys()
    }

    actual_values_second_run = {
        metric_name: metric_values
        for (metric_name, metric_values) in actual_values_second_run.items()
        if metric_name in checks.keys()
    }

    for metric_name in checks.keys():
        actual_values_first_run[metric_name].start_step = train_iters // 2 + 1
        actual_values_first_run[metric_name].values = {
            k: v
            for k, v in actual_values_first_run[metric_name].values.items()
            if k > train_iters // 2
        }

        actual_values_second_run[metric_name].start_step = train_iters // 2 + 1
        actual_values_second_run[metric_name].values = {
            k: v
            for k, v in actual_values_second_run[metric_name].values.items()
            if k > train_iters // 2
        }

    logger.info(actual_values_first_run)
    logger.info(actual_values_second_run)

    test_pretraining_regular_pipeline.test_regular_pipeline(
        compare_approximate_results=compare_approximate_results,
        golden_values=actual_values_first_run,
        actual_values=actual_values_second_run,
        checks=checks,
        model_config_path=model_config_path,
    )
