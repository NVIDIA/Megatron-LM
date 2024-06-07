import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys

from tests.functional_tests.python_test_utils.common import read_tb_logs_as_list


def collect_train_test_metrics(logs_dir, run_name):
    summaries = read_tb_logs_as_list(logs_dir)

    train_metrics = {
        metric_name: {
            "start_step": 0,
            "end_step": len(metric_values),
            "step_interval": 5,
            "values": metric_values[0 : len(metric_values) : 5],
        }
        for metric_name, metric_values in summaries.items()
    }
    str_train_metrics = str(train_metrics).replace("'", '"')
    print(
        f"\n ----------- Store the following metrics in tests/functional_tests/test_results/jet/{run_name}.json ----------"
    )
    print(f"\n {str_train_metrics}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    logs_dir = args[0]  # eg /lustre/fsw/joc/shanmugamr/megatron/logs/
    run_name = args[1]
    collect_train_test_metrics(logs_dir, run_name)
