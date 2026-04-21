import logging
from typing import Dict, List, Optional

import yaml

from tests.functional_tests.python_test_utils import common

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECK_THRESHOLDS = {
    "iteration-time": [common.ApproximateTest(atol=2.0, rtol=0)],
    "mem-allocated-bytes": [
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.05), rtol=0)
    ],
    "mem-max-allocated-bytes": [
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.05), rtol=0)
    ],
    "lm loss": [
        common.DeterministicTest(),
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.05), rtol=0),
    ],
    "num-zeros": [
        common.DeterministicTest(),
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.20), rtol=0),
    ],
    "generated_tokens": [
        common.DeterministicTest(),
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.05), rtol=0),
    ],
    "logprobs": [
        common.DeterministicTest(),
        common.ApproximateTest(atol_func=common.approximate_threshold(rtol=0.05), rtol=0),
    ],
}


def test_regular_pipeline(
    compare_approximate_results: bool,
    golden_values: Dict[str, common.GoldenValueMetric],
    tensorboard_logs: Dict[str, common.GoldenValueMetric],
    model_config_path: str,
    checks: Optional[Dict[str, List[common.Test]]] = None,
):
    if checks is None:
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
                    if golden_metric not in golden_values.keys()
                ]
            )
            > 0
        ):
            logger.error(
                f"The following metrics are required but not provided in golden values: {', '.join(missing_metrics)}"
            )
            assert False

    common.pipeline(
        compare_approximate_results=compare_approximate_results,
        golden_values=golden_values,
        tensorboard_logs=tensorboard_logs,
        checks=checks,
    )
