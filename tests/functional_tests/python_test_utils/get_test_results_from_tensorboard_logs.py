import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json
import logging

import click

from tests.functional_tests.python_test_utils import common

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--logs-dir", required=True, type=str, help="Path to Tensorboard logs")
@click.option("--train-iters", required=True, type=int, help="Number of train iters")
@click.option("--output-path", required=False, type=str, help="Path to write golden values")
@click.option(
    "--is-convergence-test/--is-normal-test",
    type=bool,
    help="Tensorboard index to extract",
    default=False,
)
@click.option("--step-size", required=False, default=5, type=int, help="Step size of sampling")
def collect_train_test_metrics(
    logs_dir: str, train_iters: str, output_path: str, is_convergence_test: bool, step_size: int
):
    summaries = common.read_tb_logs_as_list(
        logs_dir,
        index=(0 if not is_convergence_test else -1),
        train_iters=train_iters,
        start_idx=1,
        step_size=step_size,
    )

    if summaries is None:
        logger.warning("No tensorboard logs found, no golden values created.")
        return

    summaries = {
        golden_value_key: golden_value
        for (golden_value_key, golden_value) in summaries.items()
        if golden_value_key
        in [
            "iteration-time",
            "mem-allocated-bytes",
            "mem-max-allocated-bytes",
            "lm loss",
            "num-zeros",
        ]
    }

    if output_path is not None:
        with open(output_path, "w") as fh:
            json.dump(
                {
                    golden_value_key: golden_values.model_dump()
                    for golden_value_key, golden_values in summaries.items()
                },
                fh,
            )


if __name__ == "__main__":
    collect_train_test_metrics()
