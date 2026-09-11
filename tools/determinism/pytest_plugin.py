# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in pytest producer for per-rank replay evidence.

Load with ``-p tools.determinism.pytest_plugin --determinism-evidence-dir DIR``.
Only tests marked ``determinism_case(op_id=..., implementation=...)`` contribute
to the denominator. Numerical outcomes come from the replay harness, not xfail.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from tools.determinism.coverage import SCHEMA_VERSION, collect_observations

ENVIRONMENT_KEYS = (
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUBLAS_WORKSPACE_CONFIG",
    "NCCL_ALGO",
    "NCCL_PROTO",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO",
    "MAMBA_DETERMINISTIC",
    "CAUSAL_CONV1D_DETERMINISTIC",
)


def _context(root: Path) -> dict:
    import torch

    versions: dict[str, str | None] = {}
    for name in (
        "torch",
        "triton",
        "transformer-engine",
        "causal-conv1d",
        "mamba-ssm",
        "flash-attn",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root
        )
    )
    driver: list[str] | None
    try:
        driver = sorted(
            set(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
                )
                .strip()
                .splitlines()
            )
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        driver = None
    return {
        "revision": revision,
        "dirty": dirty,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "python": platform.python_version(),
        "versions": versions,
        "cuda": torch.version.cuda,
        "driver": driver,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "capability": (
            list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        ),
        "environment": {key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
    }


class EvidencePlugin:
    """Persist planned cases before execution so interruptions retain unknowns."""

    def __init__(self, directory: Path, root: Path):
        self.directory = directory
        self.root = root
        self.data = None
        self.path = directory / f"rank-{os.environ.get('RANK', '0')}-{os.getpid()}.json"

    def _write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def pytest_collection_finish(self, session):
        cases = {}
        for item in session.items:
            marker = item.get_closest_marker("determinism_case")
            if marker is None:
                continue
            declaration = dict(marker.kwargs)
            if (
                marker.args
                or set(declaration) != {"op_id", "implementation"}
                or not all(isinstance(value, str) and value for value in declaration.values())
            ):
                raise pytest.UsageError(
                    "determinism_case requires op_id and implementation strings"
                )
            cases[item.nodeid] = {
                "declaration": declaration,
                "observations": [],
                "test_complete": False,
            }
        if not cases:
            return
        manifest_path = self.root / "tests/unit_tests/determinism/kernels/manifest.py"
        spec = importlib.util.spec_from_file_location(
            "_determinism_evidence_manifest", manifest_path
        )
        manifest = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = manifest
        spec.loader.exec_module(manifest)
        inventory = {
            entry.name: {"sources": entry.sources, "exempt_reason": entry.exempt_reason}
            for entry in manifest.KERNELS
        }
        unknown = {case["declaration"]["op_id"] for case in cases.values()} - inventory.keys()
        if unknown:
            raise pytest.UsageError(f"Unregistered determinism operation IDs: {sorted(unknown)}")
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "run_id": os.environ.get("DETERMINISM_EVIDENCE_RUN_ID", "local"),
            "rank": int(os.environ.get("RANK", "0")),
            "context": _context(self.root),
            "complete": False,
            "inventory": inventory,
            "cases": cases,
        }
        self._write()

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        if self.data is None or item.nodeid not in self.data["cases"]:
            return (yield)
        case = self.data["cases"][item.nodeid]

        def record(observation):
            observation["signature"].update(case["declaration"])
            case["observations"].append(observation)
            self._write()

        with collect_observations(record):
            return (yield)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item, call):
        report = yield
        if self.data is None or item.nodeid not in self.data["cases"]:
            return report
        case = self.data["cases"][item.nodeid]
        if report.when == "call":
            case["test_complete"] = report.passed
        if report.failed or report.skipped:
            case["test_complete"] = False
            case["reason"] = f"{report.when}: {report.outcome}"
        self._write()
        return report

    def pytest_sessionfinish(self, session, exitstatus):
        if self.data is not None:
            self.data["complete"] = int(exitstatus) in (0, 1)
            self.data["exit_status"] = int(exitstatus)
            self._write()


def pytest_addoption(parser):
    """Register the opt-in output path."""
    parser.addoption("--determinism-evidence-dir", type=Path, default=None)


def pytest_configure(config):
    """Register the marker and install the producer only when requested."""
    config.addinivalue_line(
        "markers", "determinism_case(op_id, implementation): measured kernel replay case"
    )
    directory = config.getoption("--determinism-evidence-dir")
    if directory is not None:
        config.pluginmanager.register(
            EvidencePlugin(directory, Path(config.rootpath)), "determinism-evidence"
        )
