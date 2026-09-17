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
from itertools import product
from pathlib import Path

from tools.determinism.training_state import (
    UnverifiedState,
    capture_configuration,
    compare_runs,
    snapshot_path,
)

ROOT = Path(__file__).resolve().parents[2]


def _validate_pipeline_size(backend: str, pipeline_size: int) -> None:
    if (
        type(pipeline_size) is not int
        or pipeline_size not in (1, 2)
        or (pipeline_size != 1 and backend != "megatron_gpt")
    ):
        raise ValueError("Only the Megatron training adapter supports PP=2")


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


def _verify_stop_point(directory: Path, world_size: int, capture: dict) -> None:
    """Require only the target snapshot, not a subset of per-step instrumentation."""
    expected = {
        snapshot_path(directory, capture["stop_step"], rank).with_suffix(suffix)
        for rank in range(world_size)
        for suffix in (".json", ".bin")
    }
    actual = {path for path in directory.glob("step-*/*") if path.is_file()}
    if actual != expected or any(path.is_symlink() for path in expected):
        raise UnverifiedState(f"Stop-point snapshot inventory differs: {directory}")
    for rank in range(world_size):
        completion = json.loads((directory / f"complete-rank-{rank:06d}.json").read_text())
        if (
            completion["steps"] != [capture["stop_step"]]
            or completion["provenance"]["recipe"].get("capture") != capture
        ):
            raise UnverifiedState(f"Missing or incompatible stop-point declaration: {directory}")


def _verify_megatron_layout(directory: Path, world_size: int, pipeline_size: int) -> None:
    """Require each initialized TP/PP/DP coordinate and its loader owner once."""
    data_size = world_size // (2 * pipeline_size)
    expected = set(product(range(2), range(pipeline_size), range(data_size)))
    observed = set()
    for rank in range(world_size):
        completion = json.loads((directory / f"complete-rank-{rank:06d}.json").read_text())
        layout = completion["provenance"]["rank_layout"]
        if (
            layout["global_rank"] != rank
            or layout["world_size"] != world_size
            or layout["sizes"] != {"TP": 2, "PP": pipeline_size, "DP": data_size, "CP": 1}
            or layout["CP"] != 0
            or layout["VPP"] is not None
            or layout["owns_loader"] is not (layout["TP"] == 0)
        ):
            raise UnverifiedState(f"Unexpected Megatron rank layout: {directory}, rank {rank}")
        observed.add((layout["TP"], layout["PP"], layout["DP"]))
    if observed != expected:
        raise UnverifiedState(f"Missing or duplicated Megatron TP/PP/DP coordinates: {directory}")


def run_protocol(
    output: Path,
    *,
    backend: str,
    world_size: int,
    steps: int,
    checkpoint_step: int,
    control: str,
    stop_step: int | None = None,
    pipeline_size: int = 1,
) -> dict:
    """Run two independent trainings, a resume, and a deliberately broken resume.

    The output must be new. Raw state, command lines, checkpoint identity and
    worker logs remain available when a comparison or child process fails.
    """
    capture = capture_configuration(steps, checkpoint_step, stop_step)
    if backend not in ("cpu", "mcore_gpt", "megatron_gpt") or world_size not in (1, 2, 4, 8):
        raise ValueError("Unsupported backend or world size")
    if backend == "cpu" and world_size != 1:
        raise ValueError("The CPU harness validation recipe uses one process")
    if backend == "megatron_gpt" and (world_size not in (4, 8) or control != "rng"):
        raise ValueError("Megatron training uses four/eight ranks and an omitted-RNG control")
    _validate_pipeline_size(backend, pipeline_size)
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
        "pipeline_size": pipeline_size,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "control_injection": f"omit_restore_{control}",
        "capture": capture,
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
            if backend != "cpu":
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
                (
                    "tools.determinism.megatron_state_worker"
                    if backend == "megatron_gpt"
                    else "tools.determinism.state_replay_worker"
                ),
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
            if stop_step is not None:
                command += ["--stop-step", str(stop_step)]
            if backend == "megatron_gpt":
                command += ["--pipeline-size", str(pipeline_size)]
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
        if stop_step is not None:
            for name in ("reference", "repeat", "resume", "control"):
                _verify_stop_point(output / name, world_size, capture)
        if backend == "megatron_gpt":
            for name in ("reference", "repeat", "resume", "control"):
                _verify_megatron_layout(output / name, world_size, pipeline_size)
        result["fresh"] = compare_runs(
            output / "reference",
            output / "repeat",
            steps=[stop_step] if stop_step is not None else list(range(1, steps + 1)),
            world_size=world_size,
            comparison="fresh",
        )
        for name in ("resume", "control"):
            result[name] = compare_runs(
                output / "reference",
                output / name,
                steps=(
                    [stop_step]
                    if stop_step is not None
                    else list(range(checkpoint_step + 1, steps + 1))
                ),
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
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        result["reason"] = str(error)
        return result
    finally:
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")


def run_stop_points(
    output: Path,
    *,
    backend: str,
    world_size: int,
    steps: int,
    checkpoint_step: int,
    control: str,
    stop_steps: list[int],
    pipeline_size: int = 1,
) -> dict:
    """Run a separate four-launch protocol for every selected target step."""
    _validate_pipeline_size(backend, pipeline_size)
    if not stop_steps or len(set(stop_steps)) != len(stop_steps):
        raise ValueError("Require a nonempty list of unique stop steps")
    for step in stop_steps:
        capture_configuration(steps, checkpoint_step, step)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {
        "kind": "megatron_training_state_stop_points_v1",
        "status": "not_verified",
        "backend": backend,
        "world_size": world_size,
        "pipeline_size": pipeline_size,
        "steps": steps,
        "checkpoint_step": checkpoint_step,
        "stop_steps": sorted(stop_steps),
        "control_injection": f"omit_restore_{control}",
        "targets": [],
    }
    try:
        for step in sorted(stop_steps):
            target = run_protocol(
                output / f"stop-{step:08d}",
                backend=backend,
                world_size=world_size,
                steps=steps,
                checkpoint_step=checkpoint_step,
                control=control,
                stop_step=step,
                pipeline_size=pipeline_size,
            )
            result["targets"].append(target)
        statuses = {target["status"] for target in result["targets"]}
        if "not_verified" not in statuses:
            result["status"] = "passed" if statuses == {"passed"} else "failed"
    finally:
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    """Run the protocol and return a gate status; unverified evidence cannot pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "mcore_gpt", "megatron_gpt"), required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--pipeline-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint-step", type=int, default=2)
    parser.add_argument(
        "--stop-steps", type=int, nargs="+", help="Separate runs capturing only each target step"
    )
    parser.add_argument(
        "--control", choices=("rng", "optimizer", "scheduler", "dataloader"), default="rng"
    )
    args = parser.parse_args()
    run = run_protocol if args.stop_steps is None else run_stop_points
    result = run(
        args.output,
        backend=args.backend,
        world_size=args.world_size,
        pipeline_size=args.pipeline_size,
        steps=args.steps,
        checkpoint_step=args.checkpoint_step,
        control=args.control,
        **({} if args.stop_steps is None else {"stop_steps": args.stop_steps}),
    )
    return {"passed": 0, "failed": 1, "not_verified": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
