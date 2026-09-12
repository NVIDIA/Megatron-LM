# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Audit real TensorBoard and W&B history records from a completed acceptance run."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from examples.engram.recipe import Schedule


def tensorboard_records(directory: Path) -> tuple[dict, dict]:
    """Read scalar events, retaining duplicates instead of silently overwriting them."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(directory), size_guidance={"scalars": 0}, purge_orphaned_data=False
    )
    accumulator.Reload()
    records = defaultdict(list)
    for tag in accumulator.Tags()["scalars"]:
        for event in accumulator.Scalars(tag):
            records[(tag, event.step)].append(event.value)
    return records, {"event_files": len(list(directory.glob("events.out.tfevents.*")))}


def wandb_records(directory: Path) -> tuple[dict, dict]:
    """Read local W&B journals without starting a run or uploading any data.

    W&B's installed journal reader validates its own record framing and checksum.
    Multiple offline restart segments remain distinct local journals; matching IDs
    alone do not prove they have been merged by the online service.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    records = defaultdict(list)
    segments = []
    run_ids = set()
    for path in sorted(directory.rglob("*.wandb")):
        store = DataStore()
        store.open_for_scan(str(path))
        steps = []
        training_steps = []
        try:
            while (payload := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(payload)
                if record.HasField("run") and record.run.run_id:
                    run_ids.add(record.run.run_id)
                if not record.HasField("history"):
                    continue
                values = {}
                for item in record.history.item:
                    key = item.key or "/".join(item.nested_key)
                    values[key] = json.loads(item.value_json)
                if "_step" not in values:
                    continue
                step = int(values["_step"])
                steps.append(step)
                if "recipe/update_lr" in values or "lm loss" in values:
                    training_steps.append(step)
                checkpoint_step = values.get("final/checkpoint_step", step)
                if (
                    not isinstance(checkpoint_step, (int, float))
                    or not math.isfinite(checkpoint_step)
                    or int(checkpoint_step) != checkpoint_step
                ):
                    raise ValueError(f"Invalid final checkpoint step in {path}")
                for key, value in values.items():
                    if not key.startswith("_") and isinstance(value, (float, int)):
                        coordinate = int(checkpoint_step) if key.startswith("final/") else step
                        records[(key, coordinate)].append(value)
        finally:
            store.close()
        segments.append(
            {
                "path": str(path),
                "first_step": min(steps) if steps else None,
                "last_step": max(steps) if steps else None,
                "history_rows": len(steps),
                "first_training_step": min(training_steps) if training_steps else None,
            }
        )
    return records, {
        "run_ids": sorted(run_ids),
        "segments": segments,
        "service_merge_verified": False,
        "sync_status": "Local journals audited; online upload/merge not asserted",
    }


def compare_records(
    tensorboard: dict,
    wandb: dict,
    start_step: int,
    end_step: int,
    engram: bool = False,
    eval_interval: int | None = None,
    require_health_metrics: bool = False,
    require_experiment_metrics: bool = False,
    startup_steps: set[int] | None = None,
) -> dict:
    """Require identical scalar semantics, complete progress, and no duplicate curve points."""
    required = [
        "lm loss",
        "learning-rate",
        "grad-norm",
        "recipe/update_lr",
        "recipe/mtp_weight",
        "recipe/phase",
        "recipe/completed_samples_before_update",
        "recipe/completed_tokens_before_update",
    ]
    if engram:
        required.append("recipe/engram_update_lr")
    if require_health_metrics:
        required.extend(["optimizer-skipped-iterations", "nan-iterations", "tokens-per-second"])
    if require_experiment_metrics:
        required.extend(
            [
                "mtp_1 loss",
                "seq_load_balancing_loss",
                "batch-size",
                "mem-reserved-bytes",
                "mem-allocated-bytes",
                "mem-max-allocated-bytes",
                "mem-allocated-count",
            ]
        )
    errors = []
    comparisons = 0
    metric_steps = {key: list(range(start_step, end_step + 1)) for key in required}
    if require_experiment_metrics:
        # Native interval timing intentionally omits the startup report in each process.
        startup_steps = {start_step} if startup_steps is None else startup_steps
        metric_steps["iteration-time"] = [
            step for step in range(start_step, end_step + 1) if step not in startup_steps
        ]
    if eval_interval:
        metric_steps["lm loss validation"] = [
            step for step in range(start_step, end_step + 1) if step % eval_interval == 0
        ]
    for key, steps in metric_steps.items():
        for step in steps:
            left, right = tensorboard.get((key, step), []), wandb.get((key, step), [])
            if len(left) != 1 or len(right) != 1:
                errors.append(f"{key}@{step}: expected one point; TB={len(left)}, W&B={len(right)}")
                continue
            if not math.isfinite(left[0]) or not math.isfinite(right[0]):
                errors.append(f"{key}@{step}: non-finite value")
            elif not math.isclose(left[0], right[0], rel_tol=1e-6, abs_tol=1e-7):
                errors.append(f"{key}@{step}: TB={left[0]}, W&B={right[0]}")
            comparisons += 1
    for step in range(start_step, end_step + 1):
        expected = {
            "recipe/mtp_weight": Schedule().mtp_weight(step - 1),
            "recipe/phase": {"warmup": 0, "stable": 1, "decay": 2}[Schedule().phase(step - 1)],
            "recipe/completed_samples_before_update": (step - 1) * 64,
            "recipe/completed_tokens_before_update": (step - 1) * 64 * 4096,
        }
        for key, value in expected.items():
            actual = wandb.get((key, step), [])
            if len(actual) == 1 and not math.isclose(actual[0], value, rel_tol=1e-7, abs_tol=1e-7):
                errors.append(f"{key}@{step}: progress/schedule mismatch {actual[0]} != {value}")
        if step > start_step:
            previous = wandb.get(("learning-rate", step - 1), [])
            current = wandb.get(("recipe/update_lr", step), [])
            if len(previous) == len(current) == 1 and not math.isclose(
                previous[0], current[0], rel_tol=1e-7, abs_tol=1e-12
            ):
                errors.append(f"Update LR at {step} differs from preceding scheduler output")
    return {"passed": not errors, "scalar_comparisons": comparisons, "errors": errors}


def main() -> None:
    """Audit local records and persist a machine-readable acceptance result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--start-step", type=int, default=1)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--engram", action="store_true", help="Require the Engram table update LR")
    parser.add_argument("--eval-interval", type=int, default=64)
    parser.add_argument("--require-health-metrics", action="store_true")
    parser.add_argument("--require-experiment-metrics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tensorboard, tb_info = tensorboard_records(args.run_dir / "tensorboard")
    wandb, wb_info = wandb_records(args.run_dir / "wandb")
    report = compare_records(
        tensorboard,
        wandb,
        args.start_step,
        args.end_step,
        args.engram,
        args.eval_interval,
        args.require_health_metrics,
        args.require_experiment_metrics,
        {
            item["first_training_step"]
            for item in wb_info["segments"]
            if item["first_training_step"] is not None
        },
    )
    expected_id = (args.run_dir / "artifacts" / "wandb_run_id").read_text().strip()
    if wb_info["run_ids"] != [expected_id]:
        report["errors"].append("Journal run IDs do not match the persisted run identity")
        report["passed"] = False
    report.update({"tensorboard": tb_info, "wandb": wb_info, "run_dir": str(args.run_dir)})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "scalar_comparisons": report["scalar_comparisons"],
                "error_count": len(report["errors"]),
                "first_errors": report["errors"][:10],
                "report": str(args.output),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
