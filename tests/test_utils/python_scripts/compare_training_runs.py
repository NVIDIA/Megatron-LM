# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# TensorBoard does not publish typing metadata.
# mypy: disable-error-code=import-untyped

"""Compare original and refactored training runs, including resumed event files."""

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_TAGS = ("lm loss", "learning-rate", "batch-size", "grad-norm")


def read_training_scalars(
    log_dir: Path, *, start_step: int, end_step: int, tags: tuple[str, ...] = DEFAULT_TAGS
) -> dict[str, list[tuple[int, float]]]:
    """Read a complete finite scalar series for every requested metric.

    Args:
        log_dir: TensorBoard directory, possibly containing multiple resume phases.
        start_step: First required step, inclusive.
        end_step: Last required step, inclusive.
        tags: Required TensorBoard scalar tags.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if start_step < 0 or end_step < start_step:
        raise ValueError("Expected 0 <= start_step <= end_step")
    if not tags or len(set(tags)) != len(tags):
        raise ValueError("Required metric tags must be nonempty and unique")
    if not log_dir.is_dir():
        raise FileNotFoundError(log_dir)
    events = EventAccumulator(
        str(log_dir), size_guidance={"scalars": 0}, purge_orphaned_data=False
    ).Reload()
    expected_steps = list(range(start_step, end_step + 1))
    records = {}
    for tag in tags:
        if tag not in events.Tags()["scalars"]:
            raise ValueError(f"{log_dir}: missing metric {tag!r}")
        values = [
            (event.step, event.value)
            for event in events.Scalars(tag)
            if start_step <= event.step <= end_step
        ]
        if [step for step, _ in values] != expected_steps:
            raise ValueError(
                f"{log_dir}: {tag!r} requires exactly one ordered value per step "
                f"{start_step}..{end_step}; observed {[step for step, _ in values]}"
            )
        if not all(math.isfinite(value) for _, value in values):
            raise ValueError(f"{log_dir}: {tag!r} contains non-finite values")
        records[tag] = values
    return records


def compare_training_runs(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    start_step: int,
    end_step: int,
    tags: tuple[str, ...] = DEFAULT_TAGS,
) -> dict:
    """Compare complete scalar series exactly, without rounding or tolerances.

    Args:
        baseline_dir: Original training run's TensorBoard directory.
        candidate_dir: Refactored training run's TensorBoard directory.
        start_step: First required step, inclusive.
        end_step: Last required step, inclusive.
        tags: Required scalar tags; runtime timing metrics are intentionally excluded.

    Returns:
        JSON-serializable comparison evidence containing both observed series.
    """
    baseline = read_training_scalars(
        baseline_dir, start_step=start_step, end_step=end_step, tags=tags
    )
    candidate = read_training_scalars(
        candidate_dir, start_step=start_step, end_step=end_step, tags=tags
    )
    for tag in tags:
        for expected, actual in zip(baseline[tag], candidate[tag], strict=True):
            if expected != actual:
                raise AssertionError(
                    f"{tag!r} at step {expected[0]}: baseline={expected[1]!r}, "
                    f"candidate={actual[1]!r}"
                )
    return {
        "status": "passed",
        "comparison": "exact_logged_scalars",
        "baseline": baseline,
        "candidate": candidate,
    }


def main() -> None:
    """Compare two TensorBoard directories over an explicitly required step window."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--end-step", type=int, required=True)
    parser.add_argument("--tags", nargs="+", default=DEFAULT_TAGS)
    args = parser.parse_args()
    result = compare_training_runs(
        args.baseline_dir,
        args.candidate_dir,
        start_step=args.start_step,
        end_step=args.end_step,
        tags=tuple(args.tags),
    )
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
