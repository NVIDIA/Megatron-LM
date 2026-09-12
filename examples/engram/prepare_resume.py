# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare an isolated 128 -> 129 -> 130 checkpoint/logging acceptance branch."""

import argparse
import json
import shlex
import shutil
import uuid
from pathlib import Path

from examples.engram.check_checkpoint import inspect_checkpoint


def prepare(reference: Path, destination: Path, engram: bool) -> dict:
    """Copy the complete immutable step-128 checkpoint and allocate a distinct log identity."""
    source = reference / "checkpoints" / "iter_0000128"
    evidence = inspect_checkpoint(source, 128, engram)
    if not evidence["passed"]:
        raise RuntimeError(
            f"The reference checkpoint failed completeness checks: {evidence['errors']}"
        )
    if destination.exists():
        raise FileExistsError(f"Resume acceptance requires a fresh directory: {destination}")
    checkpoints = destination / "checkpoints"
    checkpoints.mkdir(parents=True)
    shutil.copytree(source, checkpoints / source.name)
    (checkpoints / "latest_checkpointed_iteration.txt").write_text("128\n")
    artifacts = destination / "artifacts"
    artifacts.mkdir()
    run_id = uuid.uuid4().hex[:12]
    (artifacts / "wandb_run_id").write_text(run_id + "\n")
    for filename in ("data_audit.json", "actual_parameters.json"):
        source_artifact = reference / "artifacts" / filename
        if source_artifact.exists():
            shutil.copy2(source_artifact, artifacts / filename)
    wrapper = "pretrain_moe_engram.sh" if engram else "pretrain_moe_baseline.sh"
    commands = []
    for step in (129, 130):
        commands.append(
            shlex.join(
                [
                    "env",
                    f"RUN_ROOT={destination.parent}",
                    f"EXPERIMENT_NAME={destination.name}",
                    f"LOAD_PATH={checkpoints}",
                    f"EXIT_INTERVAL={step}",
                    "SAVE_INTERVAL=1",
                    "EVAL_INTERVAL=64",
                    "bash",
                    f"examples/engram/{wrapper}",
                ]
            )
        )
    manifest = {
        "reference_run": str(reference),
        "resume_run": str(destination),
        "checkpoint_copied": str(source),
        "starting_iteration": 128,
        "starting_consumed_samples": 128 * 64,
        "wandb_run_id": run_id,
        "steps": [129, 130],
        "commands_in_order": commands,
        "reference_checkpoint_inspection": evidence,
        "purpose": "Compare continuous step 130 with a 128->129->130 restored trajectory",
    }
    (artifacts / "resume_branch.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    """Prepare the branch only; GPU execution stays with the experiment coordinator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--resume-run-dir", type=Path, required=True)
    parser.add_argument("--engram", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference_run_dir, args.resume_run_dir, args.engram), indent=2))


if __name__ == "__main__":
    main()
