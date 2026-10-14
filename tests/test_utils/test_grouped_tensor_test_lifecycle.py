# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the GPU test fixture's pytest lifecycle without requiring CUDA or TE."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).parents[1]
    / "unit_tests/transformer/moe/test_grouped_tensor_dispatcher_numerics.py"
)
CLASS = "TestGroupedTensorDispatcherNumerics"
PARITY = "test_hybridep_grouped_tensor_moe_parity"
LAST_CASE = f"{CLASS}::{PARITY}[single-weight-single-bias]"
FIRST_CASE = f"{CLASS}::{PARITY}[discrete-weight-no-bias]"

_STUB_RUNTIME = '''
import atexit
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

events = []
state = {"groups": False, "buffer": False}
mcore_config = SimpleNamespace(ENABLE_EXPERIMENTAL=False)

def synchronize():
    events.append("synchronize")

def reset_hybrid_ep_buffer():
    state["buffer"] = False
    events.append("release_buffer")

class Utils:
    world_size = 8

    @staticmethod
    def initialize_model_parallel(**kwargs):
        assert not state["groups"]
        assert kwargs == {"tensor_model_parallel_size": 1, "expert_model_parallel_size": 8}
        state["groups"] = True
        events.append("initialize_groups")

    @staticmethod
    def initialize_distributed():
        assert state["groups"]

    @staticmethod
    def destroy_model_parallel():
        assert not state["buffer"], "Release buffers before their process group"
        assert os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == "original"
        assert not mcore_config.ENABLE_EXPERIMENTAL
        state["groups"] = False
        events.append("destroy_groups")

torch = SimpleNamespace(
    distributed=SimpleNamespace(is_available=lambda: True, barrier=lambda: events.append("barrier")),
    cuda=SimpleNamespace(synchronize=synchronize),
)

def nccl_ep_release_context():
    assert events[-2:] == ["synchronize", "barrier"]
    events.append("release_nccl_context")

def run_case(dispatcher):
    assert state["groups"]
    assert os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] == "1"
    assert not mcore_config.ENABLE_EXPERIMENTAL, "Restore experimental mode between cases"
    mcore_config.ENABLE_EXPERIMENTAL = True
    if dispatcher == "hybridep" and not state["buffer"]:
        state["buffer"] = True
        events.append("initialize_buffer")
    events.append("case")
    if os.environ["CASE_OUTCOME"] == "fail":
        pytest.fail("Injected numerical assertion failure")
    if os.environ["CASE_OUTCOME"] == "skip":
        pytest.skip("Injected unavailable backend")

def _run_numerical_parity_case(dispatcher, **kwargs):
    run_case(dispatcher)

def _run_padding_lifecycle_case(dispatcher, monkeypatch):
    run_case(dispatcher)

atexit.register(lambda: Path("events.json").write_text(json.dumps(events)))
'''


def _run_cases(tmp_path: Path, selected: list[str], outcome: str = "pass"):
    # Execute the real fixture, xunit setup/teardown, decorators, and test methods.
    # Only numerical helpers/imports are replaced: importing the full module itself
    # would require Transformer Engine and CUDA on this CPU-only harness.
    tree = ast.parse(SOURCE.read_text())
    definitions = [
        node
        for node in tree.body
        if getattr(node, "name", None) in {"grouped_tensor_parallel", CLASS}
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_PARAMETER_LAYOUTS"
                for target in node.targets
            )
        )
    ]
    module = tmp_path / "test_cases.py"
    module.write_text(_STUB_RUNTIME + "\n" + ast.unparse(ast.Module(definitions, [])))
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = timeout: GPU case timeout\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-x",
            *([f"{module}::{case}" for case in selected] if selected else [str(module)]),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "NVTE_GROUPED_LINEAR_SINGLE_PARAM": "original",
            "CASE_OUTCOME": outcome,
        },
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result, json.loads((tmp_path / "events.json").read_text())


@pytest.mark.parametrize("selected", [[], [LAST_CASE], [LAST_CASE, FIRST_CASE]])
def test_selected_grouped_tensor_cases_own_their_resources(tmp_path: Path, selected: list[str]):
    result, events = _run_cases(tmp_path, selected)
    assert result.returncode == 0, result.stdout + result.stderr
    case_count = len(selected) or 28
    assert f"{case_count} passed" in result.stdout
    assert events.count("initialize_groups") == 1
    assert events.count("initialize_buffer") == 1
    assert events.count("case") == case_count
    assert events.count("release_nccl_context") == case_count
    assert events[-3:] == ["synchronize", "release_buffer", "destroy_groups"]


@pytest.mark.parametrize("outcome", ["fail", "skip"])
def test_grouped_tensor_resources_are_released_after_failure_or_skip(tmp_path: Path, outcome: str):
    result, events = _run_cases(tmp_path, [LAST_CASE], outcome)
    assert result.returncode == (1 if outcome == "fail" else 0), result.stdout + result.stderr
    assert events.count("case") == 1
    assert events.count("release_nccl_context") == 1
    assert events[-3:] == ["synchronize", "release_buffer", "destroy_groups"]
