# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Verify recipe event evidence in JSONL, TensorBoard text, and actual W&B journals."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def tensorboard_events(directory: Path) -> dict:
    """Retain each text event, including duplicates and restarted sessions."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(directory), size_guidance={"tensors": 0}, purge_orphaned_data=False
    )
    accumulator.Reload()
    events = defaultdict(list)
    for tag in accumulator.Tags()["tensors"]:
        if not tag.startswith("recipe/events/"):
            continue
        for event in accumulator.Tensors(tag):
            record = json.loads(event.tensor_proto.string_val[0])
            events[(record["session"], record["event"], event.step)].append(record)
    return events


def wandb_events(directory: Path) -> tuple[dict, list]:
    """Inspect persisted config updates, native checkpoint artifacts and run exits."""
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    events, segments = defaultdict(list), []
    for path in sorted(directory.rglob("*.wandb")):
        store = DataStore()
        store.open_for_scan(str(path))
        segment = {"path": str(path), "sessions": set(), "checkpoint_steps": [], "exits": []}
        try:
            while (payload := store.scan_data()) is not None:
                record = wandb_internal_pb2.Record()
                record.ParseFromString(payload)
                updates = []
                if record.HasField("config"):
                    updates.extend(record.config.update)
                if record.HasField("run"):
                    updates.extend(record.run.config.update)
                for update in updates:
                    key = update.key or "/".join(update.nested_key)
                    if not key.startswith("recipe_event_"):
                        continue
                    value = json.loads(update.value_json)
                    identity = (value["session"], value["event"], value["step"])
                    events[identity].append(value)
                    segment["sessions"].add(value["session"])
                if record.HasField("artifact") and record.artifact.metadata:
                    metadata = json.loads(record.artifact.metadata)
                    if record.artifact.type == "model" and "iteration" in metadata:
                        segment["checkpoint_steps"].append(int(metadata["iteration"]))
                if record.HasField("exit"):
                    segment["exits"].append(int(record.exit.exit_code))
        finally:
            store.close()
        segment["sessions"] = sorted(segment["sessions"])
        segments.append(segment)
    return events, segments


def compare_events(records: list[dict], tensorboard: dict, wandb: dict, segments: list) -> dict:
    """Require actual sink evidence; an expected native artifact is not itself proof."""
    errors = []
    seen = set()
    for record in records:
        identity = (record["session"], record["event"], record["step"])
        if identity in seen:
            errors.append(f"Repeated JSONL event {identity}")
        seen.add(identity)
        if tensorboard.get(identity) != [record]:
            errors.append(f"Missing, changed or repeated TensorBoard event {identity}")
        source = record["wandb_event_source"]
        if source == "run_config":
            if not wandb.get(identity) or any(value != record for value in wandb[identity]):
                errors.append(f"Missing or changed W&B config event {identity}")
        elif source in ("native_checkpoint_artifact", "native_run_finish"):
            matching = [segment for segment in segments if record["session"] in segment["sessions"]]
            if source == "native_checkpoint_artifact":
                found = any(record["step"] in item["checkpoint_steps"] for item in matching)
            else:
                expected_exit = 0 if record["outcome"] in ("completed", "exited") else None
                found = any(
                    expected_exit in item["exits"] if expected_exit is not None else item["exits"]
                    for item in matching
                )
            if not found:
                errors.append(f"No persisted W&B native evidence for {identity}: {source}")
        else:
            errors.append(f"W&B event was not recorded for {identity}: {source}")
    return {
        "passed": bool(records) and not errors,
        "events_compared": len(records),
        "errors": errors,
    }


def main() -> None:
    """Audit completed local records without contacting W&B or changing a run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in (args.run_dir / "artifacts" / "events.jsonl").read_text().splitlines()
    ]
    tensorboard = tensorboard_events(args.run_dir / "tensorboard")
    wandb, segments = wandb_events(args.run_dir / "wandb")
    result = compare_events(records, tensorboard, wandb, segments)
    result["wandb_segments"] = segments
    result["service_merge_verified"] = False
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
