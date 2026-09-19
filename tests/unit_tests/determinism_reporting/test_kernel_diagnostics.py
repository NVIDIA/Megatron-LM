# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contracts for diagnostic isolation, trace boundaries and partial evidence."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit_tests.determinism_reporting.test_paired_performance import (
    benchmark,
    kernel,
    load_module,
)

diagnostics = load_module("kernel_diagnostics")


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_profiles_exclude_backward_setup_and_preserve_all_kernels(tmp_path, phase):
    active = []
    calls = []
    graphs = []

    class Profile:
        def __enter__(self):
            active.append([])
            return self

        def __exit__(self, *_args):
            calls.append(active.pop())

        def export_chrome_trace(self, path):
            # Two kernels per operator: the diagnostic must not assume one fusion.
            events = [
                {"cat": "kernel", "name": name + str(index), "dur": 1}
                for name in calls[-1]
                for index in range(2)
            ]
            Path(path).write_text(json.dumps({"traceEvents": events}))

    def forward():
        if active:
            active[-1].append("forward")
        graph = object()
        graphs.append(graph)
        return graph

    used = []

    def backward(output, inputs, grad_outputs):
        assert output not in used and output in graphs
        assert inputs == ("input",) and grad_outputs == "gradient"
        used.append(output)
        active[-1].append("backward")

    torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: None),
        ones_like=lambda output: "gradient",
        autograd=SimpleNamespace(grad=backward),
        profiler=SimpleNamespace(
            profile=lambda **_kwargs: Profile(),
            ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
        ),
    )
    result = diagnostics.profile_calls(torch, forward, ("input",), phase, tmp_path)
    assert calls == [[phase]] * 3
    assert len(graphs) == (4 if phase == "backward" else 3)
    assert all(
        [event["name"] for event in call["kernels"]] == [phase + "0", phase + "1"]
        for call in result
    )
    assert len(list(tmp_path.glob("profile-*.json"))) == 3


@pytest.mark.parametrize(
    "change", ["unchanged", "modified", "added", "removed", "unset", "empty", "symlink"]
)
def test_cache_identity_never_treats_missing_evidence_as_equal(tmp_path, monkeypatch, change):
    for name in diagnostics.CACHE_ENV:
        root = tmp_path / name
        root.mkdir()
        (root / "kernel.py").write_text("compiled-kernel-fixture")
        monkeypatch.setenv(name, str(root))
    before = diagnostics.cache_snapshot()
    root = tmp_path / diagnostics.CACHE_ENV[0]
    if change == "modified":
        (root / "kernel.py").write_text("different-compiled-kernel")
    elif change == "added":
        (root / "extra.ptx").write_text("another-kernel")
    elif change in ("removed", "empty"):
        (root / "kernel.py").unlink()
        if change == "removed":
            (root / "other.py").write_text("still-has-an-inventory")
    elif change == "unset":
        monkeypatch.delenv(diagnostics.CACHE_ENV[0])
    elif change == "symlink":
        (root / "alias.py").symlink_to(root / "kernel.py")
    after = diagnostics.cache_snapshot()
    result = diagnostics.compare_caches(before, after)
    expected = None if change in ("unset", "empty", "symlink") else change == "unchanged"
    assert result["compiler_cache_unchanged"] is expected


@pytest.mark.parametrize("failure", [None, "timing", "profile", "telemetry"])
def test_timing_precedes_profiles_and_sampler_stops_on_errors(tmp_path, monkeypatch, failure):
    events = []

    class Sampler:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append("terminate")
            self.returncode = -15

        def wait(self, timeout):
            assert timeout == 5
            events.append("wait")

    def start(command, stdout, stderr):
        assert "--id=GPU-fixture" in command
        events.append("sampler")
        if failure == "telemetry":
            raise FileNotFoundError("nvidia-smi unavailable")
        stdout.write("timestamp,GPU-fixture,P0,123,456,100,50,90\n")
        return Sampler()

    def measure(*_args):
        events.append("timing")
        if failure == "timing":
            raise ValueError("timing failed")
        return [1.0, 2.0, 3.0]

    def profile(*args):
        assert events.index("timing") < len(events)
        events.append("profile")
        (args[-1] / "profile-0.json").write_text("retained partial profile")
        if failure == "profile":
            raise ValueError("profiling failed")
        return [{"kernels": [{"name": "synthetic-kernel"}]}]

    monkeypatch.setattr(diagnostics.subprocess, "Popen", start)
    monkeypatch.setattr(diagnostics, "profile_calls", profile)
    for name in diagnostics.CACHE_ENV:
        monkeypatch.delenv(name, raising=False)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(uuid="fixture")),
        get_num_threads=lambda: 1,
        get_num_interop_threads=lambda: 1,
    )
    directory = tmp_path / "diagnostics"
    args = (measure, torch, None, (), "forward", 2, 3, directory)
    if failure in ("timing", "profile"):
        with pytest.raises(ValueError, match="failed"):
            diagnostics.measure_with_diagnostics(*args)
    else:
        samples, metadata = diagnostics.measure_with_diagnostics(*args)
        assert samples == [1.0, 2.0, 3.0]
        assert (
            metadata["sha256"]
            == hashlib.sha256((directory / "diagnostics.json").read_bytes()).hexdigest()
        )
    report = json.loads((directory / "diagnostics.json").read_text())
    assert report["diagnostic_only"] is True
    assert report["status"] == ("error" if failure in ("timing", "profile") else "observed")
    if failure != "telemetry":
        assert events[-2:] == ["terminate", "wait"]
        assert report["telemetry_available"] is True
    else:
        assert report["telemetry_available"] is False
        assert "unavailable" in report["telemetry_error"]
    if failure != "timing":
        assert (directory / "timing.json").is_file()
        assert (directory / "profile-0.json").is_file()
        assert events.index("timing") < events.index("profile")
    if failure not in ("timing", "profile"):
        assert report["compiler_cache_unchanged"] is None


def test_sampler_is_killed_if_termination_times_out(tmp_path, monkeypatch):
    events = []

    class Sampler:
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            if self.returncode is None:
                raise subprocess.TimeoutExpired("fixture", timeout)
            events.append("wait")

        def kill(self):
            events.append("kill")
            self.returncode = -9

    monkeypatch.setattr(diagnostics.subprocess, "Popen", lambda *_args, **_kwargs: Sampler())
    torch = SimpleNamespace(
        cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(uuid="fixture")),
        get_num_threads=lambda: 1,
        get_num_interop_threads=lambda: 1,
    )

    def fail(*_args):
        raise ValueError("timing failed")

    with pytest.raises(ValueError, match="timing failed"):
        diagnostics.measure_with_diagnostics(
            fail, torch, None, (), "forward", 2, 3, tmp_path / "diagnostics"
        )
    assert events == ["terminate", "kill", "wait"]


@pytest.mark.parametrize(
    "options",
    [[], ["--", "custom"], ["--max-overhead-ratio", "1.2"], ["--max-regression-ratio", "1.1"]],
)
def test_diagnostic_cli_rejects_training_custom_launchers_and_gates(tmp_path, options):
    output = tmp_path / "output"
    arguments = ["--output", str(output), "--diagnostics"]
    if options:
        arguments += ["--kernel-case", "bias_swiglu", "--gpus", "1", *options]
    with pytest.raises(SystemExit):
        benchmark.main(arguments)
    assert not output.exists()


@pytest.mark.parametrize(
    "problem", ["raw_marker", "report_marker", "raw_metadata", "false_marker", "incomplete"]
)
def test_diagnostic_markers_must_match_raw_and_report(problem):
    measurement = dict(
        kernel_case="bias_swiglu",
        phase="forward",
        tokens=4,
        hidden_size=8,
        dtype="float32",
        warmup=2,
        steps=3,
    )
    result = dict(
        measurement=dict(measurement),
        mode="det",
        deterministic_algorithms=True,
        samples_ms=[1, 2, 3],
    )
    if problem == "raw_marker":
        result["measurement"]["diagnostic_only"] = True
    elif problem == "report_marker":
        measurement["diagnostic_only"] = True
    elif problem == "raw_metadata":
        result["diagnostics"] = {"status": "observed"}
    else:
        measurement["diagnostic_only"] = result["measurement"]["diagnostic_only"] = (
            problem != "false_marker"
        )
        result["diagnostics"] = {"status": "error" if problem == "incomplete" else "observed"}
    with pytest.raises(ValueError):
        kernel.validate_result(result, measurement, "det")
