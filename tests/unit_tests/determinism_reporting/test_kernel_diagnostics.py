# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Profiled artifacts must remain ineligible after removing the optional profiler."""

import pytest

from tests.unit_tests.determinism_reporting.test_paired_performance import benchmark, kernel


@pytest.mark.parametrize("options", [[], ["--kernel-case", "bias_swiglu", "--gpus", "1"]])
def test_optional_profiler_is_not_part_of_benchmark(tmp_path, options):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--output", str(output), "--diagnostics", *options])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "problem",
    ["raw_marker", "report_marker", "raw_metadata", "false_marker", "incomplete", "complete"],
)
def test_profiled_artifacts_cannot_supply_kernel_timings(problem):
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
