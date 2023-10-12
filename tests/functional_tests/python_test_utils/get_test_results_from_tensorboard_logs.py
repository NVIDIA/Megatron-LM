import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import glob
from tensorboard.backend.event_processing import event_accumulator


def read_tb_logs_as_list(path, summary_name):
    """Reads a TensorBoard Events file from the input path, and returns the
    summary specified as input as a list.

    Arguments:
    path: str, path to the dir where the events file is located.
    summary_name: str, name of the summary to read from the TB logs.
    Output:
    summary_list: list, the values in the read summary list, formatted as a list.
    """
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    if files:
        event_file = files[0]
        ea = event_accumulator.EventAccumulator(event_file)
        ea.Reload()
        summary = ea.Scalars(summary_name)
        summary_list = [round(x.value, 5) for x in summary]
        print(f'\nObtained the following list for {summary_name} ------------------')
        print(summary_list)
        return summary_list
    raise FileNotFoundError(f"File not found matching: {path}/events*")    

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
    model_name = run_name.split('_')[0]
    str_train_metrics = str(train_metrics).replace("'", "\"")
    print(f"\n ----------- Store the following metrics in tests/functional_tests/test_results/${model_name}/{run_name}.json ----------")
    print(f"\n {str_train_metrics}", flush=True)

if __name__ == '__main__':
    args = sys.argv[1:]
    logs_dir = args[0] # eg /lustre/fsw/joc/shanmugamr/megatron/logs/
    run_name = args[1]
    collect_train_test_metrics(logs_dir, run_name)


