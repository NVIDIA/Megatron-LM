import enum
import glob
import json
import logging
import os

from tensorboard.backend.event_processing import event_accumulator

# By default TB tries to be smart about what to load in memory to avoid OOM
# Since we expect every step to be there when we do our comparisons, we explicitly
# set the size guidance to 0 so that we load everything. It's okay given our tests
# are small/short.
SIZE_GUIDANCE = {event_accumulator.TENSORS: 0, event_accumulator.SCALARS: 0}

logger = logging.getLogger()


class TypeOfTest(enum.Enum):
    APPROX = 1
    DETERMINISTIC = 2


TYPE_OF_TEST_TO_METRIC = {
    TypeOfTest.DETERMINISTIC: ["lm loss", "num-zeros"],
    TypeOfTest.APPROX: ["lm loss", "iteration-time", "mem-allocated-bytes"],
}

METRIC_TO_THRESHOLD = {
    "iteration-time": 0.8,
    "mem-allocated-bytes": 3 * 1000 * 1000,  # 3MB
    "lm loss": 0.05,
}


def read_tb_logs_as_list(path, index=0):
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

    summaries = {}

    if not files:
        logger.info(f"File not found matching: {path}/events* || {path}/results/events*")
        return summaries

    files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)))
    accumulators = []

    if index == -1:
        for event_file in files:
            ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
            ea.Reload()
            accumulators.append(ea)
    else:
        event_file = files[index]
        ea = event_accumulator.EventAccumulator(event_file, size_guidance=SIZE_GUIDANCE)
        ea.Reload()
        accumulators.append(ea)

    for ea in accumulators:
        for scalar_name in ea.Tags()["scalars"]:
            if scalar_name in summaries:
                summaries[scalar_name] += [round(x.value, 5) for x in ea.Scalars(scalar_name)]
            else:
                summaries[scalar_name] = [round(x.value, 5) for x in ea.Scalars(scalar_name)]

            print(
                f"Extracted {len(summaries[scalar_name])} values of {scalar_name} from Tensorboard \
    logs. Here are the first 5 values: {summaries[scalar_name][:5]}"
            )

    return summaries


def load_expected_data():
    expected_metrics_file = os.getenv("EXPECTED_METRICS_FILE")

    with open(expected_metrics_file) as f:
        if os.path.exists(expected_metrics_file):
            with open(expected_metrics_file) as f:
                return json.load(f)
        else:
            print(f"File {expected_metrics_file} not found!")
