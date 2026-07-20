# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TEST_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = TEST_ROOT / "_test_harness"
REPO_ROOT = TEST_ROOT.parents[2]


def _load_internal_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_internal_module("mlite_local_test_runner", HARNESS_ROOT / "runner.py")
markers = importlib.import_module("markers")


class _FakeCuda:
    def __init__(self, capabilities: list[tuple[int, int]]) -> None:
        self._capabilities = capabilities

    def is_available(self) -> bool:
        return bool(self._capabilities)

    def device_count(self) -> int:
        return len(self._capabilities)

    def get_device_capability(self, index: int) -> tuple[int, int]:
        return self._capabilities[index]


class _FakeTorch:
    def __init__(self, capabilities: list[tuple[int, int]]) -> None:
        self.cuda = _FakeCuda(capabilities)
        self.distributed = SimpleNamespace(
            is_available=lambda: True, is_nccl_available=lambda: True
        )
        self.__version__ = "0.test"
        self.version = SimpleNamespace(cuda="0.test")


class _FakeItem:
    def __init__(self, **named_markers) -> None:
        self._markers = named_markers

    def iter_markers(self, name: str):
        return iter(self._markers.get(name, []))


def _marker(*args, **kwargs):
    return SimpleNamespace(args=args, kwargs=kwargs)


def _case(
    nodeid: str,
    *,
    gpus: int = 0,
    architecture: str | None = None,
    environment: tuple[tuple[str, str | None], ...] = (),
    timeout: int = 1800,
    optional: bool = False,
) -> runner.TestCase:
    return runner.TestCase(
        nodeid=nodeid,
        gpus=gpus,
        min_architecture=architecture,
        environment=environment,
        timeout_seconds=timeout,
        optional=optional,
    )


def _selection(profile: str, architecture: str, count: int) -> runner.HardwareSelection:
    return runner.HardwareSelection(
        profile=profile,
        architecture=architecture,
        gpu_count=count,
        compute_capability=markers.ARCHITECTURE_CAPABILITIES[architecture],
    )


def _suite(
    name: str = "test-suite",
    *,
    gpus: int = 0,
    environment: tuple[tuple[str, str | None], ...] = (),
    timeout: int = 1800,
    targets: tuple[str, ...] = (
        "experimental/lite/tests/runtime/test_example.py::test_case",
    ),
) -> runner.Suite:
    return runner.Suite(
        name=name,
        gpus=gpus,
        environment=environment,
        timeout_seconds=timeout,
        targets=targets,
    )


def _counts(**overrides: int) -> dict[str, int]:
    counts = {
        "passed": 1,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "error": 0,
    }
    counts.update(overrides)
    return counts


def _rank_report(
    rank: int,
    outcome_digest: str = "a" * 64,
    status: str = "PASS",
    **count_overrides: int,
) -> dict[str, object]:
    return {
        "rank": rank,
        "status": status,
        "counts": _counts(**count_overrides),
        "outcome_digest": outcome_digest,
    }


def _passing_result(suite: runner.Suite) -> runner.SuiteResult:
    return runner.SuiteResult(
        name=suite.name,
        gpus=suite.gpus,
        duration_seconds=0.1,
        status="PASS",
        ranks_reported=max(1, suite.gpus),
        passed=1,
        failed=0,
        skipped=0,
        xfailed=0,
        xpassed=0,
        errors=0,
    )


def test_stable_configuration_is_code_owned_and_does_not_enumerate_tests():
    assert set(runner.TEST_ROOTS) == {
        "experimental/lite/tests/model",
        "experimental/lite/tests/primitive",
        "experimental/lite/tests/runtime",
        "experimental/lite/tests/examples",
    }
    assert runner.EXCLUDE_GLOBS == (
        "experimental/lite/tests/examples/test_verl_*.py",
    )
    assert "--strict-markers" in runner.PYTEST_ARGS
    assert "--disable-warnings" in runner.PYTEST_ARGS
    assert runner.PYTEST_ARGS[:2] == ("-o", "addopts=")
    assert runner.HARDWARE_PROFILES == {
        "standard": (8, "hopper"),
        "blackwell": (4, "blackwell"),
    }
    assert not (HARNESS_ROOT / "manifest.json").exists()


def test_marker_defaults_describe_cpu_hopper_and_1800_seconds():
    cpu = markers.execution_for_item(_FakeItem())
    gpu = markers.execution_for_item(_FakeItem(gpus=[_marker(2)]))

    assert cpu.gpus == 0
    assert cpu.min_architecture is None
    assert cpu.environment == ()
    assert cpu.timeout_seconds == 1800
    assert gpu.gpus == 2
    assert gpu.min_architecture == "hopper"


def test_marker_scopes_merge_environment_and_closest_values_win():
    item = _FakeItem(
        gpus=[
            _marker(4, min_architecture="blackwell"),
            _marker(2),
        ],
        env=[
            _marker(CUDA_DEVICE_MAX_CONNECTIONS=None),
            _marker(CUDA_DEVICE_MAX_CONNECTIONS="1"),
        ],
        timeout=[_marker(seconds=45), _marker(seconds=3600)],
        optional=[_marker()],
    )

    execution = markers.execution_for_item(item)

    assert execution.gpus == 4
    assert execution.min_architecture == "blackwell"
    assert execution.environment == (("CUDA_DEVICE_MAX_CONNECTIONS", None),)
    assert execution.timeout_seconds == 45
    assert execution.optional


@pytest.mark.parametrize(
    "item",
    [
        _FakeItem(gpus=[_marker()]),
        _FakeItem(gpus=[_marker(0)]),
        _FakeItem(gpus=[_marker(1, min_architecture="future")]),
        _FakeItem(env=[_marker(UNKNOWN="1")]),
        _FakeItem(env=[_marker(CUDA_DEVICE_MAX_CONNECTIONS=1)]),
        _FakeItem(timeout=[_marker(10)]),
        _FakeItem(timeout=[_marker(seconds=0)]),
    ],
)
def test_invalid_execution_markers_are_rejected(item):
    with pytest.raises(markers.MarkerError):
        markers.execution_for_item(item)


def test_default_targets_are_all_four_roots_and_new_test_files_need_no_registration():
    targets = runner.resolve_targets([], REPO_ROOT)

    assert set(targets) == set(runner.TEST_ROOTS)
    for target in targets:
        assert (REPO_ROOT / target).is_dir()


def test_path_targets_resolve_from_caller_and_deduplicate_descendants():
    targets = runner.resolve_targets(
        ["model", "model/test_qwen_config_unit.py", "runtime/test_checkpoint_unit.py"],
        TEST_ROOT,
    )

    assert targets == [
        "experimental/lite/tests/model",
        "experimental/lite/tests/runtime/test_checkpoint_unit.py",
    ]


@pytest.mark.parametrize("target", ["--maxfail=1", "../lite", "README.md"])
def test_path_targets_reject_options_outside_paths_and_non_test_files(target):
    with pytest.raises(runner.SelectionError):
        runner.resolve_targets([target], TEST_ROOT)


def test_collection_command_uses_shared_args_exclusions_and_generated_plan(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        captured["timeout"] = kwargs["timeout"]
        plan = {
            "tests": [
                {
                    "nodeid": "experimental/lite/tests/runtime/test_example.py::test_case",
                    "gpus": 0,
                    "min_architecture": None,
                    "environment": {},
                    "timeout_seconds": 1800,
                    "optional": False,
                }
            ]
        }
        Path(kwargs["env"]["MLITE_TEST_PLAN_PATH"]).write_text(
            json.dumps(plan), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    cases = runner.collect_tests(list(runner.TEST_ROOTS), tmp_path)

    assert len(cases) == 1
    assert cases[0].gpus == 0
    assert captured["command"][-4:] == list(runner.TEST_ROOTS)
    assert f"--rootdir={REPO_ROOT}" in captured["command"]
    assert "--collect-only" in captured["command"]
    assert (
        "--ignore-glob=experimental/lite/tests/examples/test_verl_*.py"
        in captured["command"]
    )
    assert captured["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert captured["environment"]["MLITE_TEST_HARNESS"] == "1"
    assert captured["timeout"] == runner.COLLECTION_TIMEOUT_SECONDS


def test_profile_selection_uses_default_and_explicit_subset_semantics():
    cpu = _case("experimental/lite/tests/runtime/test_a.py::test_cpu")
    hopper = _case(
        "experimental/lite/tests/model/test_a.py::test_hopper",
        gpus=1,
        architecture="hopper",
    )
    blackwell = _case(
        "experimental/lite/tests/model/test_b.py::test_blackwell",
        gpus=4,
        architecture="blackwell",
    )
    optional = _case(
        "experimental/lite/tests/model/test_c.py::test_optional",
        gpus=1,
        architecture="hopper",
        optional=True,
    )
    cases = [cpu, hopper, blackwell, optional]

    assert runner.select_cases(cases, _selection("standard", "hopper", 8), False) == [
        cpu,
        hopper,
    ]
    assert runner.select_cases(
        cases, _selection("blackwell", "blackwell", 4), False
    ) == [blackwell]
    assert runner.select_cases(
        cases, _selection("subset", "blackwell", 4), True
    ) == cases
    with pytest.raises(runner.SelectionError):
        runner.select_cases(
            [blackwell], _selection("subset", "hopper", 8), True
        )


def test_suites_group_gpu_functions_by_file_and_execution_contract():
    cpu_a = _case("experimental/lite/tests/runtime/test_a.py::test_one")
    cpu_b = _case("experimental/lite/tests/model/test_b.py::test_two")
    gpu_a = _case(
        "experimental/lite/tests/model/test_gpu.py::test_matrix_a",
        gpus=2,
        architecture="hopper",
        environment=(("CUDA_DEVICE_MAX_CONNECTIONS", "1"),),
        timeout=3600,
    )
    gpu_b = _case(
        "experimental/lite/tests/model/test_gpu.py::test_matrix_b",
        gpus=2,
        architecture="hopper",
        environment=(("CUDA_DEVICE_MAX_CONNECTIONS", "1"),),
        timeout=3600,
    )
    gpu_other_file = _case(
        "experimental/lite/tests/model/test_other.py::test_matrix",
        gpus=2,
        architecture="hopper",
        environment=(("CUDA_DEVICE_MAX_CONNECTIONS", "1"),),
        timeout=3600,
    )

    suites = runner.build_suites([cpu_a, gpu_b, cpu_b, gpu_a, gpu_other_file])

    assert len(suites) == 3
    cpu = next(suite for suite in suites if suite.gpus == 0)
    gpu = next(
        suite
        for suite in suites
        if suite.targets == tuple(sorted([gpu_a.nodeid, gpu_b.nodeid]))
    )
    assert cpu.name == "cpu"
    assert cpu.targets == tuple(sorted([cpu_a.nodeid, cpu_b.nodeid]))
    assert gpu.environment == (("CUDA_DEVICE_MAX_CONNECTIONS", "1"),)
    assert gpu.timeout_seconds == 3600


@pytest.mark.parametrize(
    ("capabilities", "expected_profile"),
    [([(9, 0)] * 8, "standard"), ([(10, 0)] * 4, "blackwell")],
)
def test_default_hardware_detection_selects_supported_profile(
    capabilities, expected_profile
):
    selection = runner.detect_default_hardware(_FakeTorch(capabilities))

    assert selection.profile == expected_profile
    assert selection.gpu_count == len(capabilities)
    assert selection.compute_capability == capabilities[0]


@pytest.mark.parametrize(
    "capabilities",
    [[], [(9, 0)] * 4, [(10, 0)] * 8, [(9, 0)] * 7 + [(10, 0)]],
)
def test_default_hardware_detection_rejects_unsupported_or_mixed_topology(
    capabilities,
):
    with pytest.raises(runner.UnsupportedHardware):
        runner.detect_default_hardware(_FakeTorch(capabilities))


@pytest.mark.parametrize(
    "capabilities",
    [[(9, 0)], [(9, 0)] * 2, [(10, 0)], [(10, 0)] * 3],
)
def test_subset_hardware_accepts_any_positive_homogeneous_supported_gpu_count(
    capabilities,
):
    selection = runner.detect_subset_hardware(_FakeTorch(capabilities))

    assert selection.profile == "subset"
    assert selection.gpu_count == len(capabilities)


def test_suite_environment_sanitizes_then_applies_declared_variables(tmp_path):
    base = {
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c",
        "CUDA_DEVICE_MAX_CONNECTIONS": "8",
        "RANK": "7",
        "WORLD_SIZE": "8",
        "MLITE_FORCE_TOPO": "1,1,1,1,1",
        "QWEN35_HF_DIR": "/external/checkpoint",
        "PYTEST_ADDOPTS": "-k incomplete-selection",
        "PYTEST_PLUGINS": "untrusted_plugin",
        "PYTHONPATH": "/untrusted/path",
    }
    suite = _suite(
        "two-gpu",
        gpus=2,
        environment=(("CUDA_DEVICE_MAX_CONNECTIONS", "1"),),
    )

    environment = runner.build_suite_environment(suite, tmp_path, base)

    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert environment["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert "RANK" not in environment
    assert "WORLD_SIZE" not in environment
    assert "MLITE_FORCE_TOPO" not in environment
    assert environment["QWEN35_HF_DIR"] == "/external/checkpoint"
    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert environment["PYTHONPATH"] == os.pathsep.join(
        (str(REPO_ROOT), str(TEST_ROOT.parent))
    )
    assert environment["MLITE_TEST_HARNESS"] == "1"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"

    unset_suite = _suite(
        environment=(("CUDA_DEVICE_MAX_CONNECTIONS", None),),
    )
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in runner.build_suite_environment(
        unset_suite, tmp_path, base
    )


def test_suite_commands_use_direct_worker_or_isolated_torchrun():
    cpu = _suite()
    distributed = _suite(
        gpus=2,
        targets=("experimental/lite/tests/model/test_b.py::test_b",),
    )

    cpu_command = runner.build_suite_command(cpu)
    distributed_command = runner.build_suite_command(distributed)

    assert "torch.distributed.run" not in cpu_command
    assert "torch.distributed.run" in distributed_command
    assert "--standalone" in distributed_command
    assert "--nproc-per-node=2" in distributed_command
    assert "--disable-warnings" in distributed_command
    assert f"--rootdir={REPO_ROOT}" in cpu_command
    assert f"--rootdir={REPO_ROOT}" in distributed_command


def test_source_revision_marks_untracked_changes_without_exposing_paths(monkeypatch):
    revision = "a" * 40

    def fake_run(command, **_kwargs):
        stdout = (
            revision + "\n" if command[1] == "rev-parse" else "?? private-local-name\n"
        )
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner._source_revision() == revision + "+dirty"


def test_rank_report_integrity_accepts_matching_clean_ranks():
    suite = _suite("two-rank", gpus=2)
    result = runner.evaluate_suite_reports(
        suite, [_rank_report(0), _rank_report(1)], process_exit_code=0, timed_out=False
    )

    assert result.status == "PASS"
    assert result.ranks_reported == 2
    assert result.passed == 1


def test_rank_report_integrity_rejects_same_counts_for_different_node_outcomes():
    suite = _suite("two-rank", gpus=2)
    result = runner.evaluate_suite_reports(
        suite,
        [
            _rank_report(0, outcome_digest="a" * 64),
            _rank_report(1, outcome_digest="b" * 64),
        ],
        process_exit_code=0,
        timed_out=False,
    )

    assert result.status == "FAIL"


@pytest.mark.parametrize(
    "reports",
    [
        [_rank_report(0)],
        [_rank_report(0), _rank_report(1, status="FAIL", skipped=1)],
        [_rank_report(0), _rank_report(1, outcome_digest="invalid")],
        [_rank_report(0), _rank_report(0)],
    ],
)
def test_rank_report_integrity_rejects_missing_failed_malformed_or_duplicate_ranks(
    reports,
):
    suite = _suite("two-rank", gpus=2)
    result = runner.evaluate_suite_reports(
        suite, reports, process_exit_code=0, timed_out=False
    )

    assert result.status == "FAIL"


@pytest.mark.parametrize(
    ("test_source", "expected_exit", "outcome"),
    [
        ("def test_ok():\n    assert True\n", 0, "passed"),
        (
            "import pytest\ndef test_skip():\n    pytest.skip('not allowed')\n",
            1,
            "skipped",
        ),
        (
            "import pytest\npytest.importorskip('mlite_module_that_does_not_exist')\n",
            1,
            "skipped",
        ),
        (
            "import pytest\n"
            "@pytest.fixture\n"
            "def skip_teardown():\n"
            "    yield\n"
            "    pytest.skip('not allowed')\n"
            "def test_teardown_skip(skip_teardown):\n"
            "    pass\n",
            1,
            "skipped",
        ),
        (
            "import pytest\n@pytest.mark.xfail(strict=False)\n"
            "def test_unexpected_pass():\n    assert True\n",
            1,
            "xpassed",
        ),
    ],
)
def test_worker_enforces_skip_and_xpass_policy(
    tmp_path, test_source, expected_exit, outcome
):
    test_file = tmp_path / "test_worker_case.py"
    report_dir = tmp_path / "reports"
    test_file.write_text(test_source, encoding="utf-8")
    environment = dict(os.environ)
    environment.update(
        {
            "MLITE_TEST_REPORT_DIR": str(report_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(HARNESS_ROOT / "pytest_worker.py"),
            "-q",
            "-p",
            "no:cacheprovider",
            str(test_file),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads((report_dir / "rank-0.json").read_text(encoding="utf-8"))

    assert completed.returncode == expected_exit
    assert report["counts"][outcome] == 1
    assert report["status"] == ("PASS" if expected_exit == 0 else "FAIL")
    assert len(report["outcome_digest"]) == 64


def test_explicit_cpu_subset_does_not_import_torch_or_probe_hardware(
    monkeypatch, capsys
):
    target = "experimental/lite/tests/runtime/test_local_validation_runner.py"
    case = _case(f"{target}::test_synthetic_cpu")
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setattr(runner, "collect_tests", lambda *_args, **_kwargs: [case])
    monkeypatch.setattr(
        runner,
        "_load_torch",
        lambda: pytest.fail("CPU subset must not import torch in the runner"),
    )
    monkeypatch.setattr(runner, "_print_header", lambda *_args: None)
    monkeypatch.setattr(runner, "run_suite", lambda suite, _temporary: _passing_result(suite))

    assert runner.main([target]) == 0
    assert "overall=PASS" in capsys.readouterr().out


def test_internal_runner_rejects_non_path_arguments_without_hardware_probe(capsys):
    assert runner.main(["--maxfail=1"]) == 2
    output = capsys.readouterr().out
    assert "invalid_test_target" in output
    assert "exit_code=2" in output
