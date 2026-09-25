# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests of timing integrity, paired comparisons, and the real launcher loop."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
kernel = load_module("run_kernel")


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
    parent = {
        **benchmark.DET_ENV,
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_PROTO": "Simple",
        "TRITON_CACHE_AUTOTUNING": "1",
        "TRITON_CACHE_DIR": "/shared/cache",
        "TRITON_AUTOTUNE_BLOCK_SIZE_M": "64",
    }
    default = benchmark.mode_environment(parent, "default")
    assert "NCCL_ALGO" not in default
    assert "CUBLAS_WORKSPACE_CONFIG" not in default
    assert default["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "1"
    assert default["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"
    assert benchmark.mode_environment(parent, "det")["NCCL_ALGO"] == "Ring"
    assert parent["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == "0"
    for mode in ("det", "default"):
        environment = benchmark.mode_environment(parent, mode)
        assert "TRITON_CACHE_AUTOTUNING" not in environment
        assert environment["TRITON_CACHE_DIR"] == "/shared/cache"
        assert environment["TRITON_AUTOTUNE_BLOCK_SIZE_M"] == "64"


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


def test_kernel_timings_need_calibration_for_a_performance_pass():
    assert benchmark.ratio_summary([2.0] * 3, None)["status"] == "not_gated"
    assert benchmark.ratio_summary([1.0], None)["status"] == "inconclusive"


def test_kernel_compilation_cannot_be_counted_as_steady_state(tmp_path):
    with pytest.raises(SystemExit):
        benchmark.main(
            [
                "--output",
                str(tmp_path / "cold"),
                "--kernel-case",
                "bias_swiglu",
                "--gpus",
                "1",
                "--warmup",
                "0",
            ]
        )
    assert not (tmp_path / "cold").exists()


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
@pytest.mark.parametrize("report_only", [False, True])
def test_subprocess_measurements_and_artifacts(tmp_path, monkeypatch, incomplete, report_only):
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
            *(["--report-only", "--max-overhead-ratio", "1.1"] if report_only else []),
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
        assert report["status"] == ("fail" if report_only else "pass")
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


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_kernel_phase_timing_excludes_setup_and_warmup(phase):
    calls = []

    class Event:
        def __init__(self, **kwargs):
            self.index = 0

        def record(self):
            calls.append("event")

        def synchronize(self):
            calls.append("end_sync")

        def elapsed_time(self, other):
            self.index += 1
            return float(self.index)

    def forward():
        calls.append("forward")
        return "output"

    def backward(output, inputs, grad_outputs):
        assert (output, inputs, grad_outputs) == ("output", ("input",), "gradient")
        calls.append("backward")
        return ("grad",)

    torch = SimpleNamespace(
        cuda=SimpleNamespace(Event=Event, synchronize=lambda: calls.append("setup_sync")),
        ones_like=lambda output: "gradient",
        autograd=SimpleNamespace(grad=backward),
    )
    assert kernel.measure(torch, forward, ("input",), phase, warmup=2, steps=3) == [3, 4, 5]
    expected = ["setup_sync", "event", phase, "event", "end_sync"]
    if phase == "backward":
        # One setup forward supplies the upstream gradient's shape.
        assert calls.pop(0) == "forward"
        expected.insert(0, "forward")
    assert calls == expected * 5


@pytest.mark.parametrize("problem", ["missing", "nan", "zero", "mode", "phase", "policy"])
def test_invalid_kernel_measurements_are_rejected(tmp_path, problem):
    measurement = dict(
        kernel_case="weighted_swiglu",
        phase="backward",
        tokens=4096,
        hidden_size=8192,
        dtype="bfloat16",
        warmup=2,
        steps=3,
    )
    result = {
        "measurement": dict(measurement),
        "mode": "det",
        "deterministic_algorithms": True,
        "samples_ms": [1.0] * 3,
    }
    if problem == "missing":
        result["samples_ms"].pop()
    elif problem in ("nan", "zero"):
        result["samples_ms"][0] = float("nan") if problem == "nan" else 0
    elif problem == "phase":
        result["measurement"]["phase"] = "forward"
    elif problem == "policy":
        result["deterministic_algorithms"] = False
    else:
        result["mode"] = "default"
    path = tmp_path / "kernel.json"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        kernel.read_result(path, measurement, "det")


def test_kernel_driver_uses_real_subprocesses_and_preserves_raw_samples(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.setattr(benchmark, "_source", lambda path: {"revision": "a" * 40, "dirty": False})
    monkeypatch.setattr(benchmark, "_machine", lambda: {"gpus": ["CPU fixture only"]})
    fixture = tmp_path / "kernel_fixture.py"
    fixture.write_text("""
import json, os, sys
from pathlib import Path
options = dict(zip((arg[2:].replace('-', '_') for arg in sys.argv[1::2]), sys.argv[2::2]))
for key in ('tokens', 'hidden_size', 'warmup', 'steps'):
    options[key] = int(options[key])
mode = os.environ['DETERMINISM_PERF_MODE']
result = dict(measurement=options, mode=mode, deterministic_algorithms=mode == 'det',
              samples_ms=[12.0 if mode == 'det' else 10.0] * options['steps'])
(Path(os.environ['DETERMINISM_PERF_LOG_DIR']) / 'kernel.json').write_text(json.dumps(result))
""")
    real_run = subprocess.run
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda command, **kwargs: real_run([sys.executable, str(fixture), *command[2:]], **kwargs),
    )
    output = tmp_path / "kernel-output"
    assert (
        benchmark.main(
            [
                "--output",
                str(output),
                "--kernel-case",
                "weighted_swiglu",
                "--phase",
                "backward",
                "--gpus",
                "1",
                "--pairs",
                "3",
                "--warmup",
                "2",
                "--steps",
                "3",
            ]
        )
        == 0
    )
    report = json.loads((output / "benchmark.json").read_text())
    assert report["status"] == "reported"
    assert "diagnostic_only" not in report["measurement"]
    assert report["kind"] == "determinism_kernel_performance"
    assert report["measurement"]["timing"] == "cuda_event_ms"
    assert "recipe" not in report["measurement"]
    assert report["comparisons"]["head_overhead"]["median_ratio"] == 1.2
    assert all(len(run["samples_ms"]) == 3 for run in report["runs"])
    assert all(Path(run["timing_log"]).is_file() for run in report["runs"])


def test_leaderboard_keeps_failed_rows_and_runs_remaining_cases(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    leaderboard = load_module("kernel_leaderboard")
    calls = []

    def run(args):
        options = dict(zip(args[::2], args[1::2]))
        calls.append(options["--kernel-case"])
        output = Path(options["--output"])
        output.mkdir()
        report = {
            "measurement": {
                "kernel_case": options["--kernel-case"],
                "phase": options["--phase"],
                "dtype": options["--dtype"],
            },
            "status": "error" if len(calls) == 1 else "reported",
            "runs": [],
        }
        (output / "benchmark.json").write_text(json.dumps(report))
        return 1 if len(calls) == 1 else 0

    monkeypatch.setattr(benchmark, "main", run)
    output = tmp_path / "leaderboard"
    assert leaderboard.main(["--output", str(output)]) == 1
    assert len(calls) == 12
    assert len(json.loads((output / "leaderboard.json").read_text())) == 12
    text = (output / "leaderboard.md").read_text()
    assert "error" in text and "weighted_squared_relu" in text
    with pytest.raises(SystemExit):
        leaderboard.main(["--output", str(output)])
