# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Produce one complete same-allocation collective CI dataset, retaining failed stages."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import collective_baseline
from benchmark import _source
from collective_workload import MAPPINGS

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = Path(__file__).resolve().parent
MAX_BYTES = 64 * 1024 * 1024
RECIPE_ID = "ci-tp-sp-collectives-v1"
DEPENDENCIES = (
    "megatron/determinism/__init__.py",
    "tools/determinism/capture_recipe.py",
    "tools/determinism/collective_capture.py",
    "tools/determinism/replay_collectives.py",
    "tools/determinism/coverage.py",
    "tools/determinism/recipe_coverage.py",
    "tests/unit_tests/determinism/kernels/test_captured_collectives.py",
)


def _write(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Capture, replay, check accuracy, inventory and time before accepting artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, choices=(4, 8), required=True)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args(argv)
    if args.pairs < 3 or min(args.warmup, args.steps) < 1:
        parser.error("A baseline needs at least three pairs and positive warmup/measured steps")
    if (
        "TORCHELASTIC_RUN_ID" in os.environ
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1
        or int(os.environ.get("RANK", "0")) != 0
    ):
        parser.error("Launch the pipeline parent outside torchrun on one node")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Use an empty output directory; preserve previous attempts")
    output.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "schema_version": 1,
        "kind": "determinism_collective_producer",
        "status": "incomplete",
        "gpus": args.gpus,
        "recipe_id": RECIPE_ID,
        "stages": [],
    }
    report_path = output / "producer.json"
    _write(report_path, report)
    try:
        if Path.cwd().resolve() != ROOT:
            raise ValueError("Run the pipeline from its source checkout")
        missing = [name for name in DEPENDENCIES if not (ROOT / name).is_file()]
        if missing:
            raise ValueError(
                "Collective capture/replay and early startup dependencies are missing: "
                + ", ".join(missing)
            )
        source = _source(ROOT)
        if source["dirty"]:
            raise ValueError("The producer requires a clean source checkout")
        report["source"] = source
        bindings = output / "bindings.json"
        _write(
            bindings,
            [
                {
                    "target": "megatron.core.tensor_parallel.mappings:" + name,
                    "op_id": "tensor_parallel_mappings",
                    "implementation": "mcore:" + name,
                    "adapter": "tensor_parallel_collective",
                }
                for name in MAPPINGS.values()
            ],
        )
        environment = {**os.environ, "PYTHONHASHSEED": "0"}
        python = [sys.executable]
        distributed = python + [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.gpus}",
        ]
        capture, coverage, timing = output / "capture", output / "coverage.json", output / "timing"
        commands = [
            (
                "capture",
                distributed
                + [
                    "-m",
                    "tools.determinism.capture_recipe",
                    "--bindings",
                    str(bindings),
                    "--output",
                    str(output / "inventory"),
                    "--recipe-id",
                    RECIPE_ID,
                    "--max-signatures",
                    "128",
                    "--collective-capture",
                    str(capture),
                    "--max-collective-bytes",
                    str(MAX_BYTES),
                    "--",
                    str(SCRIPTS / "collective_workload.py"),
                    "--deterministic-mode",
                ],
            ),
            (
                "replay",
                distributed
                + [
                    "-m",
                    "tools.determinism.replay_collectives",
                    "--capture",
                    str(capture),
                    "--evidence",
                    str(output / "shards"),
                    "--max-bytes",
                    str(MAX_BYTES),
                ],
            ),
            (
                "coverage",
                python
                + [
                    "-m",
                    "tools.determinism.coverage",
                    str(output / "shards"),
                    "--revision",
                    source["revision"],
                    "--output",
                    str(coverage),
                    "--require-verified",
                    "--require-case",
                    "*test_captured_collective_replay*",
                ],
            ),
            (
                "recipe-join",
                python
                + [
                    "-m",
                    "tools.determinism.recipe_coverage",
                    str(output / "inventory"),
                    "--evidence",
                    str(coverage),
                    "--output",
                    str(output / "recipe-report.json"),
                    "--strict",
                ],
            ),
            (
                "timing",
                python
                + [
                    str(SCRIPTS / "benchmark_collectives.py"),
                    "--capture",
                    str(capture),
                    "--evidence",
                    str(coverage),
                    "--output",
                    str(timing),
                    "--pairs",
                    str(args.pairs),
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--max-bytes",
                    str(MAX_BYTES),
                ],
            ),
        ]
        for name, command in commands:
            stage: dict = {
                "name": name,
                "command": command,
                "status": "incomplete",
                "log": name + ".log",
            }
            report["stages"].append(stage)
            _write(report_path, report)
            print(f"Collective producer: {name}", flush=True)
            with (output / stage["log"]).open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            stage.update(
                exit_code=result.returncode, status="passed" if result.returncode == 0 else "failed"
            )
            _write(report_path, report)
            if result.returncode != 0:
                raise ValueError(f"{name} failed with exit {result.returncode}; see {stage['log']}")
            if _source(ROOT) != source:
                raise ValueError("Source changed during collective evidence production")
        manifest, _ = collective_baseline.snapshot(
            capture,
            coverage,
            timing / "benchmark.json",
            source["revision"],
            "producer:" + RECIPE_ID,
        )
        evidence = manifest["collective_evidence"]
        if (
            evidence["ranks"] != args.gpus
            or evidence["events"] != 96
            or evidence["status"] != "not_gated"
        ):
            raise ValueError("Incomplete pilot matrix or unexpected performance gate")
        report.update(status="complete", collective_evidence=evidence)
    except (
        OSError,
        AttributeError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.SubprocessError,
    ) as error:
        report["error"] = str(error)
        _write(report_path, report)
        print(json.dumps({"status": "incomplete", "error": str(error)}))
        return 1
    _write(report_path, report)
    print(
        json.dumps(
            {"status": report["status"], "evidence": report["collective_evidence"]}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
