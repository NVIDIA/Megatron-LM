#!/usr/bin/env python3
"""Validate exact train/rollout agreement from VERL's JSONL logger."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate(path: Path, expected_steps: int) -> None:
    steps = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        data = record.get("data", {})
        if (
            "training/rollout_probs_diff_valid" in data
            or "training/rollout_logprob_bitwise_equal_fraction" in data
        ):
            steps[int(record["step"])] = data
    if len(steps) != expected_steps:
        raise ValueError(
            f"expected {expected_steps} alignment steps, found {sorted(steps)} in {path}"
        )

    bad = []
    for step, data in sorted(steps.items()):
        if "training/rollout_probs_diff_valid" in data:
            observed = (
                data["training/rollout_probs_diff_valid"],
                data["training/rollout_probs_diff_max"],
                data["training/rollout_probs_diff_mean"],
                data["training/rollout_probs_diff_std"],
                data["rollout_corr/k3_kl"],
            )
            if observed != (1, 0.0, 0.0, 0.0, 0.0):
                bad.append((step, "upstream_probs", observed))
        else:
            observed = (
                data["training/rollout_logprob_bitwise_equal_fraction"],
                data["training/rollout_logprob_abs_diff_max"],
                data["rollout_corr/k3_kl"],
            )
            if observed != (1.0, 0.0, 0.0):
                bad.append((step, "legacy_logprob", observed))
    if bad:
        raise ValueError(f"DS4 train/infer alignment gate failed: {bad}")


if __name__ == "__main__":
    validate(Path(sys.argv[1]), int(sys.argv[2]))
    print(f"DS4_4L_ALIGNMENT_EXACT steps={sys.argv[2]} probs_diff=0.0 k3_kl=0.0")
