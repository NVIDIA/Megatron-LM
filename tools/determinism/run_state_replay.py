# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run fresh processes, checkpoint resume and an omitted-restore sensitivity control."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from tools.determinism.training_state import compare_runs

ROOT = Path(__file__).resolve().parents[2]


def _run_worker(command: list[str], env: dict, log) -> int:
    """Give torchrun time to terminate its workers if a phase times out."""
    with subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    ) as child:
        try:
            return child.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise


def run_protocol(
    output: Path, *, backend: str, world_size: int, steps: int, checkpoint_step: int, control: str
) -> dict:
    """Run two independent trainings, a resume, and a deliberately broken resume.

    The output must be new. Raw state, command lines, checkpoint identity and
    worker logs remain available when a comparison or child process fails.
    """
    if backend not in ("cpu", "mcore_gpt") or world_size not in (1, 2, 4, 8):
        raise ValueError("Unsupported backend or world size")
    if backend == "cpu" and world_size != 1:
        raise ValueError("The CPU harness validation recipe uses one process")
    if not 0 < checkpoint_step < steps or control not in (
        "rng",
        "optimizer",
        "scheduler",
        "dataloader",
    ):
        raise ValueError("Require a real resume interval and a supported sensitivity control")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {
        "kind": "megatron_training_state_protocol_v1",
        "status": "not_verified",
        "backend": backend,
        "world_size": world_size,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "control_injection": f"omit_restore_{control}",
    }
    worker_env = dict(os.environ)
    for key in list(worker_env):
        if key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ) or key.startswith("TORCHELASTIC_"):
            worker_env.pop(key)
    try:
        for name in ("reference", "repeat", "resume", "control"):
            command = [sys.executable]
            if backend == "mcore_gpt":
                command += [
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    f"--nproc-per-node={world_size}",
                    "--max-restarts=0",
                ]
            command += [
                "-m",
                "tools.determinism.state_replay_worker",
                "--backend",
                backend,
                "--output",
                str(output / name),
                "--run-id",
                str(uuid.uuid4()),
                "--steps",
                str(steps),
                "--checkpoint-step",
                str(checkpoint_step),
            ]
            if name in ("resume", "control"):
                command += ["--resume", str(output / "reference")]
            if name == "control":
                command += ["--omit-restore", control]
            (output / f"{name}-command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (output / f"{name}.log").open("w") as log:
                exit_code = _run_worker(command, worker_env, log)
            if exit_code:
                result.update(reason=f"{name} worker failed", exit_code=exit_code)
                return result
        result["fresh"] = compare_runs(
            output / "reference",
            output / "repeat",
            steps=list(range(1, steps + 1)),
            world_size=world_size,
            comparison="fresh",
        )
        for name in ("resume", "control"):
            result[name] = compare_runs(
                output / "reference",
                output / name,
                steps=list(range(checkpoint_step + 1, steps + 1)),
                world_size=world_size,
                comparison="resume",
            )
        observed = (
            result["fresh"].get("comparison_status") == "equal"
            and result["resume"].get("comparison_status") == "equal"
            and result["control"].get("comparison_status") == "different"
        )
        result["expected_observations"] = observed
        if any(result[name]["status"] == "not_verified" for name in ("fresh", "resume", "control")):
            result["reason"] = "At least one comparison has incomplete or ineligible evidence"
        elif observed:
            result["status"] = "passed"
        else:
            result.update(status="failed", reason="Replay/resume mismatch or insensitive control")
        return result
    except (OSError, subprocess.TimeoutExpired) as error:
        result["reason"] = str(error)
        return result
    finally:
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> int:
    """Run the protocol and return a gate status; unverified evidence cannot pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "mcore_gpt"), required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint-step", type=int, default=2)
    parser.add_argument(
        "--control", choices=("rng", "optimizer", "scheduler", "dataloader"), default="rng"
    )
    args = parser.parse_args()
    result = run_protocol(
        args.output,
        backend=args.backend,
        world_size=args.world_size,
        steps=args.steps,
        checkpoint_step=args.checkpoint_step,
        control=args.control,
    )
    return {"passed": 0, "failed": 1, "not_verified": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
