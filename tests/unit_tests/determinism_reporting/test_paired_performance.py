# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of timing integrity, paired comparisons, and the real launcher loop."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "tests/performance_tests/shell_test_utils/determinism"


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load_module("benchmark")
launcher = load_module("run_training")


def write_log(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            f"iteration {i}/ 5 | elapsed time per iteration (ms): {value} |" for i, value in values
        )
    )


def test_discards_warmup_and_reads_last_rank(tmp_path):
    path = tmp_path / "attempt_0/7/stdout.log"
    write_log(path, [(1, 800), (2, 900), (3, 10), (4, 12), (5, 11)])
    samples, source = benchmark.read_step_times(tmp_path, 2, 3)
    assert samples == [10, 12, 11]
    assert source == str(path)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nan", "inf", "zero", "two_logs"])
def test_incomplete_or_ambiguous_logs_are_errors(tmp_path, problem):
    values = [(1, 10), (2, 11)]
    if problem == "missing":
        values.pop()
    elif problem == "duplicate":
        values.append((2, 11))
    elif problem in ("nan", "inf", "zero"):
        values[1] = (2, {"nan": "nan", "inf": "inf", "zero": "0"}[problem])
    write_log(tmp_path / "rank/stdout.log", values)
    if problem == "two_logs":
        write_log(tmp_path / "another/stdout.log", values)
    with pytest.raises(ValueError):
        benchmark.read_step_times(tmp_path, 1, 1)


def test_inherited_deterministic_environment_does_not_contaminate_default():
    parent = {**benchmark.DET_ENV, "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_PROTO": "Simple"}
    default = benchmark.mode_environment(parent, "default")
    assert "NCCL_ALGO" not in default
    assert "CUBLAS_WORKSPACE_CONFIG" not in default
    assert default["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "1"
    assert default["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert benchmark.mode_environment(parent, "det")["NCCL_ALGO"] == "Ring"
    assert parent["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"


@pytest.mark.parametrize(
    "values,status",
    [
        ([1.1] * 3, "pass"),
        ([1.5] * 3, "fail"),
        ([1.1, 1.4, 1.2], "inconclusive"),
        ([1.0], "inconclusive"),
    ],
)
def test_uncertainty_and_sample_count_affect_gate(values, status):
    assert benchmark.ratio_summary(values, 1.35)["status"] == status


def test_shared_slowdown_is_detected_by_revision_comparison():
    runs = []
    for pair in range(3):
        for label, factor in (("base", 1), ("head", 1.2)):
            for mode, timing in (("default", 10), ("det", 12)):
                runs.append(
                    {
                        "pair": pair,
                        "revision_label": label,
                        "mode": mode,
                        "median_ms": timing * factor,
                        "status": "complete",
                    }
                )
    result = benchmark.summarize(runs, 3, True, 1.35, 1.05)
    assert result["head_overhead"]["status"] == "pass"
    assert result["det_regression"]["status"] == "fail"
    assert result["default_regression"]["status"] == "fail"
    with pytest.raises(ValueError, match="Missing"):
        benchmark.summarize(runs[:-1], 3, True, 1.35, 1.05)


@pytest.mark.parametrize("recipe", ["dense", "moe", "hybrid"])
def test_training_arguments_differ_only_by_policy_flag(recipe):
    environment = {
        "DETERMINISM_PERF_RECIPE": recipe,
        "DETERMINISM_PERF_LOG_DIR": "/tmp/logs",
        "DETERMINISM_PERF_GPUS": "4",
        "DETERMINISM_PERF_MODE": "default",
    }
    default = launcher.training_command(environment)
    deterministic = launcher.training_command({**environment, "DETERMINISM_PERF_MODE": "det"})
    assert deterministic == default + ["--deterministic-mode"]
    assert "--profile" not in default
    assert "--seed" in default


@pytest.mark.parametrize("incomplete", [False, True])
def test_subprocess_measurements_and_artifacts(tmp_path, monkeypatch, incomplete):
    monkeypatch.setattr(
        benchmark,
        "_source",
        lambda checkout: {"revision": "a" * 40, "dirty": False, "checkout": str(checkout)},
    )
    monkeypatch.setattr(benchmark, "_machine", lambda: {"gpus": ["test-only fixture"]})
    program = tmp_path / "fixture.py"
    program.write_text("""
import json, os
from pathlib import Path
root = Path(os.environ['DETERMINISM_PERF_LOG_DIR'])
(root / 'env.json').write_text(json.dumps(dict(os.environ)))
timing = 12 if os.environ['DETERMINISM_PERF_MODE'] == 'det' else 10
count = int(os.environ['DETERMINISM_PERF_TRAIN_ITERS'])
if os.environ.get('FIXTURE_INCOMPLETE'):
    count -= 1
(root / 'stdout.log').write_text(''.join(f'iteration {i}/ 4 | elapsed time per iteration (ms): {timing} |\\n' for i in range(1, count + 1)))
""")
    if incomplete:
        monkeypatch.setenv("FIXTURE_INCOMPLETE", "1")
    output = tmp_path / "output"
    rc = benchmark.main(
        [
            "--output",
            str(output),
            "--pairs",
            "3",
            "--warmup",
            "2",
            "--steps",
            "2",
            "--",
            sys.executable,
            str(program),
        ]
    )
    report = json.loads((output / "benchmark.json").read_text())
    assert (output / "benchmark.md").is_file()
    if incomplete:
        assert rc == 1
        assert report["status"] == "error"
        assert report["runs"][0]["status"] == "incomplete"
        assert "Missing iterations" in report["error"]
    else:
        assert rc == 0
        assert report["status"] == "pass"
        assert [run["mode"] for run in report["runs"]] == [
            "default",
            "det",
            "det",
            "default",
            "default",
            "det",
        ]
        assert report["comparisons"]["head_overhead"]["median_ratio"] == 1.2
        assert all(len(run["step_times_ms"]) == 2 for run in report["runs"])
    with pytest.raises(SystemExit):
        benchmark.main(["--output", str(output)])
