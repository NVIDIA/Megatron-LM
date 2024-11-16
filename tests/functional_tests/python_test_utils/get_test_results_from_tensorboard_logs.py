import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import json

import click

from tests.functional_tests.python_test_utils import common


@click.command()
@click.option("--logs-dir", required=True, type=str, help="Path to Tensorboard logs")
@click.option("--output-path", required=False, type=str, help="Path to write golden values")
@click.option(
    "--is-convergence-test/--is-normal-test",
    type=bool,
    help="Tensorboard index to extract",
    default=False,
)
def collect_train_test_metrics(logs_dir: str, output_path: str, is_convergence_test: bool):
    summaries = common.read_tb_logs_as_list(logs_dir, index=-1 if is_convergence_test else 0)

    train_metrics = {
        metric_name: {
            "start_step": 0,
            "end_step": len(metric_values),
            "step_interval": 5,
            "values": metric_values[0 : len(metric_values) : 5],
        }
        for metric_name, metric_values in summaries.items()
    }

    if output_path is not None:
        with open(output_path, "w") as fh:
            json.dump(train_metrics, fh)


if __name__ == "__main__":
    collect_train_test_metrics()
