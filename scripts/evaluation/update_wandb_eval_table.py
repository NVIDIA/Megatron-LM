import math
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb


def main(entity: str, project: str, runid: str, groups: list[str]):
    # Download history.
    ignore_cols = ["evaluation/group_eval_results", "evaluation/eval_results", "eval_table"]
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{runid}")
    history = defaultdict(dict)
    for row in run.scan_history():
        if "OptimizerStep" not in row:
            continue
        optstep = row["OptimizerStep"]
        for name, value in row.items():
            if not name.startswith("_") and name not in ignore_cols and value is not None and not math.isnan(value):
                history[optstep][name] = value
    history = pd.DataFrame(list(history.values()))
    last_step = np.max(history["OptimizerStep"])
    last_row = history[history["OptimizerStep"] == last_step]
    assert last_row.shape[0] == 1
    last_row = last_row.iloc[0, :]
    last_row = last_row.dropna()

    # Get task -> list of metrics dict.
    names = list(filter(lambda col: col != "OptimizerStep", last_row.index))
    last_row = last_row[names]
    tasks = defaultdict(list)
    tasks_maybe_stranded = defaultdict(lambda: defaultdict(list))
    for col in names:
        task, metric = col.split("/")
        belongs_here = "stderr" not in metric and metric != "acc_norm"

        # We may ignore some cols if they are requested to be grouped.
        matched_groups = [group for group in groups if task.startswith(group)]
        if len(matched_groups) > 0:
            assert len(matched_groups) == 1
            group, = matched_groups
            if task != group:
                if belongs_here:
                    tasks_maybe_stranded[group][task].append(metric)
                continue

        if belongs_here:
            tasks[task].append(metric)

    # Build metrics row.
    data = {"OptimizerStep": last_step}
    for task, metrics in tasks.items():
        for metric in metrics:
            data[f"{task}/{metric}"] = last_row[f"{task}/{metric}"]

    # Deal with the maybe stranded.
    # Some benchmarks (e.g. mmlu_continuation) perform automatically the aggregation of metrics while others dont.
    # Make sure we manually aggregate the average if the automatic one isn't there already.
    for group, maybe_stranded in tasks_maybe_stranded.items():
        if group not in tasks:
            print("Manually computing average for task group", group)
            assert len(set(map(lambda metrics: tuple(sorted(metrics)), maybe_stranded.values()))) == 1  # Make sure all subtasks use the same metrics.
            for metric in next(iter(maybe_stranded.values())):
                data[f"{group}/{metric}"] = np.mean(last_row[[f"{task}/{metric}" for task in maybe_stranded]])

    # Now connect to the run in question and attempt to find the table.
    run = wandb.init(entity=entity, project=project, id=runid)
    df = pd.DataFrame([data])
    table = wandb.Table(dataframe=df)
    wandb.log({"eval_table": table})
    wandb.finish()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--runid", required=True)
    parser.add_argument("--groups", nargs="+", default=[])
    args = parser.parse_args()
    main(**vars(args))
