# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for the on-demand evidence runner's commands and exit status."""

import sys
from types import SimpleNamespace

import pytest

from tools.determinism import run_evidence


class Launches(list):
    """Recorded subprocess calls plus the return codes to hand back in order."""

    codes: list


@pytest.fixture
def launches(monkeypatch):
    calls = Launches()
    calls.codes = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        return SimpleNamespace(returncode=calls.codes.pop(0))

    monkeypatch.setattr(run_evidence.subprocess, "run", run)
    return calls


@pytest.mark.parametrize(
    "scope,views", [("kernel", ["--require-author-checks"]), ("model", ["--require-parallelism"])]
)
def test_runner_collects_one_session_then_applies_the_scope_gates(
    launches, tmp_path, monkeypatch, scope, views
):
    monkeypatch.setenv("TRITON_CACHE_AUTOTUNING", "1")
    launches.codes = [0, 0]
    output = tmp_path / "run"
    status = run_evidence.main(
        ["--scope", scope, "--output", str(output), "--require-case", "*mxfp8*"]
    )
    assert status == 0
    (_, rev), (collect, collect_kwargs), (gate, gate_kwargs) = launches
    assert collect[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "tools.determinism.pytest_plugin" in collect
    assert f"--determinism-evidence-scope={scope}" in collect
    assert f"--determinism-evidence-dir={output / 'shards'}" in collect
    assert "--determinism-evidence-run-id=run" in collect
    assert collect[-1] == run_evidence.SCOPES[scope][0]
    environment = collect_kwargs["env"]
    assert environment["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert environment["NCCL_ALGO"] == "Ring"
    assert "TRITON_CACHE_AUTOTUNING" not in environment
    assert gate[2:4] == ["tools.determinism.coverage", str(output / "shards")]
    assert gate[gate.index("--revision") + 1] == "a" * 40
    for flag in ["--require-verified", "--forbid-nondeterministic", "--require-branches", *views]:
        assert flag in gate
    assert gate[-2:] == ["--require-case", "*mxfp8*"]


def test_explicit_selection_replaces_the_default_directory(launches, tmp_path):
    launches.codes = [0, 0]
    selection = ["tests/unit_tests/determinism/kernels/test_fused_activations.py", "-k", "swiglu"]
    assert (
        run_evidence.main(["--scope", "kernel", "--output", str(tmp_path / "o"), "--", *selection])
        == 0
    )
    collect = launches[1][0]
    assert collect[-3:] == selection


@pytest.mark.parametrize("codes,expected", [([1, 0], 1), ([0, 2], 2), ([1, 2], 1)])
def test_gate_runs_after_failed_collection_and_failures_propagate(
    launches, tmp_path, codes, expected
):
    launches.codes = list(codes)
    assert run_evidence.main(["--scope", "model", "--output", str(tmp_path / "o")]) == expected
    assert len(launches) == 3


def test_reused_output_directory_is_rejected(launches, tmp_path):
    (tmp_path / "old-shard.json").write_text("{}")
    with pytest.raises(SystemExit):
        run_evidence.main(["--scope", "kernel", "--output", str(tmp_path)])
    assert launches == []
