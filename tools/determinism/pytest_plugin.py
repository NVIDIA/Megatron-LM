# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in pytest producer for per-rank replay evidence.

Load with ``-p tools.determinism.pytest_plugin --determinism-evidence-dir DIR``.
Kernel reports include only ``determinism_case(op_id=..., implementation=...)``.
``--determinism-evidence-scope model`` selects ``determinism_model(model_id=...)``
instead and produces a separate report kind. Numerical outcomes come from replay
comparisons, not pytest passes or xfails.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from tools.determinism.branch_coverage import BranchRecorder
from tools.determinism.checks import collect_checks
from tools.determinism.coverage import SCHEMA_VERSION, collect_observations, triton_signature
from tools.determinism.parallelism import normalize_parallelism

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
        "environment": {
            **{key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
            **triton_signature(),
        },
    }


class EvidencePlugin:
    """Persist planned cases before execution so interruptions retain unknowns."""

    def __init__(
        self,
        directory: Path,
        root: Path,
        scope: str = "kernel",
        branch_sources: list[str] | None = None,
    ):
        self.directory = directory
        self.root = root
        self.scope = scope
        self.data = None
        self.branch_sources = branch_sources
        self.branches = None
        self.path = directory / f"rank-{os.environ.get('RANK', '0')}-{os.getpid()}.json"

    def _write(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.path)

    def pytest_collection_finish(self, session):
        cases = {}
        for item in session.items:
            marker_name = "determinism_model" if self.scope == "model" else "determinism_case"
            marker = item.get_closest_marker(marker_name)
            if marker is None:
                continue
            declaration = dict(marker.kwargs)
            plan = None
            if self.scope == "model":
                plan = declaration.pop("parallelism", None)
                parameters = getattr(getattr(item, "callspec", None), "params", {})
                if "parallelism" in parameters:
                    if plan is not None and normalize_parallelism(plan) != normalize_parallelism(
                        parameters["parallelism"]
                    ):
                        raise pytest.UsageError("Marker and parametrized parallelism plans differ")
                    plan = parameters["parallelism"]
                if plan is not None:
                    try:
                        plan = normalize_parallelism(plan)
                    except ValueError as error:
                        raise pytest.UsageError(str(error)) from error
            required = {"model_id"} if self.scope == "model" else {"op_id", "implementation"}
            if (
                marker.args
                or set(declaration) != required
                or not all(isinstance(value, str) and value for value in declaration.values())
            ):
                raise pytest.UsageError(f"{marker_name} requires {sorted(required)} strings")
            if self.scope == "model":
                model_id = declaration["model_id"]
                declaration = {"op_id": model_id, "implementation": "model:" + model_id}
            cases[item.nodeid] = {
                "declaration": declaration,
                "observations": [],
                "checks": [],
                "test_complete": False,
                **({"parallelism_plan": plan} if plan is not None else {}),
            }
        if not cases:
            return
        inventory = self._inventory(cases)
        unknown = {case["declaration"]["op_id"] for case in cases.values()} - inventory.keys()
        if unknown:
            raise pytest.UsageError(f"Unregistered determinism operation IDs: {sorted(unknown)}")
        self.data = {
            "schema_version": SCHEMA_VERSION,
            "evidence_scope": self.scope,
            "run_id": os.environ.get("DETERMINISM_EVIDENCE_RUN_ID", "local"),
            "rank": int(os.environ.get("RANK", "0")),
            "context": _context(self.root),
            "complete": False,
            "inventory": inventory,
            "cases": cases,
        }
        self._write()
        if self.branch_sources is not None:
            self.data["branches"] = {
                "complete": False,
                "reason": "Branch collection did not finish",
            }
            self._write()
            try:
                self.branches = BranchRecorder(
                    self.root, self.branch_sources, self.directory / "branch-data" / self.path.name
                )
            except (ImportError, ValueError) as error:
                raise pytest.UsageError(str(error)) from error

    def _inventory(self, cases):
        if self.scope == "model":
            return {case["declaration"]["op_id"]: {} for case in cases.values()}
        manifest_path = self.root / "tests/unit_tests/determinism/kernels/manifest.py"
        spec = importlib.util.spec_from_file_location(
            "_determinism_evidence_manifest", manifest_path
        )
        manifest = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = manifest
        spec.loader.exec_module(manifest)
        return {
            entry.name: {
                "sources": entry.sources,
                "exempt_reason": entry.exempt_reason,
                "author_tests": list(entry.author_tests),
            }
            for entry in manifest.KERNELS
        }

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_call(self, item):
        if self.data is None or item.nodeid not in self.data["cases"]:
            return (yield)
        case = self.data["cases"][item.nodeid]

        def record(observation):
            observation["signature"].update(case["declaration"])
            case["observations"].append(observation)
            self._write()

        def record_check(check):
            check["signature"].update(case["declaration"])
            case["checks"].append(check)
            self._write()

        branch_context = (
            self.branches.case(item.nodeid) if self.branches else contextlib.nullcontext()
        )
        with branch_context, collect_observations(record), collect_checks(record_check):
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
            case["reason"] = f"{report.when}: {report.outcome}: {report.longrepr}"
        self._write()
        return report

    def pytest_sessionfinish(self, session, exitstatus):
        if self.data is not None:
            self.data["complete"] = int(exitstatus) in (0, 1)
            self.data["exit_status"] = int(exitstatus)
            if self.branches is not None:
                try:
                    self.data["branches"], arcs = self.branches.finish()
                    for case_id, branches in arcs.items():
                        self.data["cases"][case_id]["branch_arcs"] = branches
                except Exception as error:
                    self.data["branches"] = {
                        "complete": False,
                        "reason": f"{type(error).__name__}: {error}",
                    }
            self._write()


def pytest_addoption(parser):
    """Register the opt-in output path."""
    parser.addoption("--determinism-evidence-dir", type=Path, default=None)
    parser.addoption("--determinism-evidence-scope", choices=("kernel", "model"), default="kernel")
    parser.addoption("--determinism-branch-coverage", action="store_true", default=False)
    parser.addoption(
        "--determinism-branch-source",
        action="append",
        default=[],
        help="Repository Python path; repeat for multiple sources (default: megatron/core)",
    )


def pytest_configure(config):
    """Register the marker and install the producer only when requested."""
    config.addinivalue_line(
        "markers", "determinism_case(op_id, implementation): measured kernel replay case"
    )
    config.addinivalue_line(
        "markers", "determinism_model(model_id, parallelism=None): model output/gradient replay"
    )
    directory = config.getoption("--determinism-evidence-dir")
    branches = config.getoption("--determinism-branch-coverage")
    sources = config.getoption("--determinism-branch-source")
    if (branches or sources) and directory is None:
        raise pytest.UsageError("Branch collection requires --determinism-evidence-dir")
    if sources and not branches:
        raise pytest.UsageError(
            "--determinism-branch-source requires --determinism-branch-coverage"
        )
    if directory is not None:
        config.pluginmanager.register(
            EvidencePlugin(
                directory,
                Path(config.rootpath),
                config.getoption("--determinism-evidence-scope"),
                (sources or ["megatron/core"]) if branches else None,
            ),
            "determinism-evidence",
        )
