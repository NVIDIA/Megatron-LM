import os
import glob
from tensorboard.backend.event_processing import event_accumulator

import enum


class TypeOfTest(enum.Enum):
    APPROX = 1
    DETERMINISTIC = 2


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
