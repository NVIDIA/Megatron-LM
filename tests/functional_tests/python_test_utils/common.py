import enum
import glob
import os

from tensorboard.backend.event_processing import event_accumulator

# By default TB tries to be smart about what to load in memory to avoid OOM
# Since we expect every step to be there when we do our comparisons, we explicitly
# set the size guidance to 0 so that we load everything. It's okay given our tests
# are small/short.
SIZE_GUIDANCE = {
    event_accumulator.TENSORS: 0,
    event_accumulator.SCALARS: 0,
}


class TypeOfTest(enum.Enum):
    APPROX = 1
    DETERMINISTIC = 2


TYPE_OF_TEST_TO_METRIC = {
    TypeOfTest.DETERMINISTIC: ["lm loss", "num-zeros"],
    TypeOfTest.APPROX: ["num-zeros"],
}


def read_tb_logs_as_list(path, summary_name, index=0):
    """Reads a TensorBoard Events file from the input path, and returns the
    summary specified as input as a list.

    Args:
        path: str, path to the dir where the events file is located.
        summary_name: str, name of the summary to read from the TB logs.

    Returns:
        summary_list: list, the values in the read summary list, formatted as a list.
    """
    files = glob.glob(f"{path}/events*tfevents*")
    files += glob.glob(f"{path}/results/events*tfevents*")
    if not files:
        raise FileNotFoundError(
            f"File not found matching: {path}/events* || {path}/results/events*"
        )
    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))

    event_file = files[index]
    ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
    ea.Reload()
    summary = ea.Scalars(summary_name)
    summary_list = [round(x.value, 5) for x in summary]
    print(f"\nObtained the following list for {summary_name} ------------------")
    print(summary_list)
    return summary_list
