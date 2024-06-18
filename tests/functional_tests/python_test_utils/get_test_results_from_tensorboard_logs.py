import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys

from tests.functional_tests.python_test_utils.common import read_tb_logs_as_list


def collect_train_test_metrics(logs_dir, run_name):
    # TODO: Fetch current baseline

    # train loss
    train_loss_list = read_tb_logs_as_list(logs_dir, "lm loss")

    # num zeros
    num_zeros = read_tb_logs_as_list(logs_dir, "num-zeros")

    iteration_time = read_tb_logs_as_list(logs_dir, "iteration-time")

    # First few iterations might take a little longer. So we take the last 70 percent of the timings
    idx = len(iteration_time)//3   
    iteration_time_avg = sum(iteration_time[idx:])/len(iteration_time[idx:])

    train_metrics = {
        "lm loss": {
            "start_step": 0,
            "end_step": len(train_loss_list),
            "step_interval": 5,
            "values": train_loss_list[0:len(train_loss_list):5],
        },
        "num-zeros": {
            "start_step": 0,
            "end_step": len(num_zeros),
            "step_interval": 5,
            "values": num_zeros[0:len(num_zeros):5],
        },
        "iteration_timing_avg": iteration_time_avg,
    }
    str_train_metrics = str(train_metrics).replace("'", "\"")
    print(f"\n ----------- Store the following metrics in tests/functional_tests/test_results/jet/{run_name}.json ----------")
    print(f"\n {str_train_metrics}", flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    logs_dir = args[0] # eg /lustre/fsw/joc/shanmugamr/megatron/logs/
    run_name = args[1]
    collect_train_test_metrics(logs_dir, run_name)


