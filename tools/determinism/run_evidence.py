# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Collect measured replay evidence for the determinism tests on demand.

Runs the kernel or model replay tests once under ``torchrun`` with the evidence
plugin, then aggregates the per-rank shards and applies the coverage gates::

    python -m tools.determinism.run_evidence --scope kernel --output /tmp/evidence \\
        --nproc-per-node 8

Arguments after ``--`` replace the default test directory (for example a single
test file or ``-k`` selection). Nothing here runs as part of the regular unit
test buckets.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Library and runtime settings for deterministic execution. They must be in
# the environment before Torch, NCCL, cuBLAS or Transformer Engine initialize.
POLICY_ENVIRONMENT = {
    "NCCL_ALGO": "Ring",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "MAMBA_DETERMINISTIC": "1",
    "CAUSAL_CONV1D_DETERMINISTIC": "1",
}

SCOPES = {
    "kernel": ("tests/unit_tests/determinism/kernels", []),
    "model": ("tests/unit_tests/determinism/correctness", ["--require-parallelism"]),
}


def child_environment(parent: dict, max_connections: int) -> dict:
    """Return the test environment: the parent plus the deterministic policy."""
    environment = {**parent, **POLICY_ENVIRONMENT}
    environment["CUDA_DEVICE_MAX_CONNECTIONS"] = str(max_connections)
    # Cached autotuning needs a shared cache directory; use pinned configurations.
    environment.pop("TRITON_CACHE_AUTOTUNING", None)
    return environment


def commands(args: argparse.Namespace, revision: str) -> tuple[list[str], list[str]]:
    """Return the pytest collection command and the coverage gate command."""
    shards = args.output / "shards"
    targets = args.pytest_args or [SCOPES[args.scope][0]]
    collect = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        str(args.nproc_per_node),
        "-m",
        "pytest",
        "-p",
        "tools.determinism.pytest_plugin",
        f"--determinism-evidence-scope={args.scope}",
        "--determinism-branch-coverage",
        f"--determinism-evidence-dir={shards}",
        f"--determinism-evidence-run-id={args.run_id or args.output.name}",
        *targets,
    ]
    gate = [
        sys.executable,
        "-m",
        "tools.determinism.coverage",
        str(shards),
        "--output",
        str(args.output / "coverage.json"),
        "--revision",
        revision,
        "--require-verified",
        "--forbid-nondeterministic",
        "--require-branches",
        *SCOPES[args.scope][1],
    ]
    for pattern in args.require_case:
        gate += ["--require-case", pattern]
    return collect, gate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scope", choices=sorted(SCOPES), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, empty directory")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument(
        "--cuda-device-max-connections",
        type=int,
        default=1,
        help="1 serializes side-stream work; larger values allow scheduling contention",
    )
    parser.add_argument("--run-id", help="Shard run identifier (default: the output name)")
    parser.add_argument(
        "--require-case",
        action="append",
        default=[],
        help="Also require verified evidence for every case matching this pattern",
    )
    parser.add_argument("pytest_args", nargs="*", help="Test selection (after --)")
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use a new, empty --output directory for each collection run")
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    collect, gate = commands(args, revision)
    environment = child_environment(dict(os.environ), args.cuda_device_max_connections)
    collected = subprocess.run(collect, cwd=REPO_ROOT, env=environment).returncode
    # Aggregate even after test failures so the report shows what was observed.
    gated = subprocess.run(gate, cwd=REPO_ROOT).returncode
    return collected or gated


if __name__ == "__main__":
    sys.exit(main())
