import logging

import yaml

from tests.functional_tests.python_test_utils import common, test_pretraining_regular_pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECK_THRESHOLDS = {
    "iteration-time": [common.ApproximateTest(atol=2.0, rtol=0)],
    "mem-allocated-bytes": [
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.10), rtol=0)
    ],
    "mem-max-allocated-bytes": [
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.10), rtol=0)
    ],
    "lm loss": [
        common.DeterministicTest(),
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.05), rtol=0),
    ],
    "num-zeros": [
        common.DeterministicTest(),
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.20), rtol=0),
    ],
}


def test_resume_checkpoint_pipeline(
    compare_approximate_results: bool,
    tensorboard_path: str,
    train_iters: int,
    model_config_path: str,
):

    first_run_values = common.read_tb_logs_as_list(
        tensorboard_path, index=0, train_iters=train_iters, start_idx=(train_iters // 2) + 1
    )
    second_run_values = common.read_tb_logs_as_list(
        tensorboard_path, index=1, train_iters=train_iters, start_idx=(train_iters // 2) + 1
    )

    with open(model_config_path) as f:
        model_config = yaml.safe_load(f)

    checks_types = (
        model_config["METRICS"]
        if "METRICS" in model_config
        else ["iteration-time", "lm loss", "num-zeros"]
    )
    checks = {metric: CHECK_THRESHOLDS[metric] for metric in checks_types}

    if (
        len(
            missing_metrics := [
                golden_metric
                for golden_metric in checks.keys()
                if golden_metric not in first_run_values.keys()
            ]
        )
        > 0
    ):
        logger.error(
            f"The following metrics are required but not logged during training: {', '.join(missing_metrics)}"
        )
        assert False

    first_run_values = {
        metric_name: metric_values
        for (metric_name, metric_values) in first_run_values.items()
        if metric_name in checks.keys()
    }

    second_run_values = {
        metric_name: metric_values
        for (metric_name, metric_values) in second_run_values.items()
        if metric_name in checks.keys()
    }

    logger.info(first_run_values)
    logger.info(second_run_values)

    test_pretraining_regular_pipeline.test_regular_pipeline(
        compare_approximate_results=compare_approximate_results,
        golden_values=first_run_values,
        tensorboard_logs=second_run_values,
        checks=checks,
        model_config_path=model_config_path,
    )
