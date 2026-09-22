# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU orchestration and selection checks; collective execution is GPU-validated separately."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.unit_tests.determinism_reporting.test_paired_performance import (
    ROOT,
    SCRIPTS,
    load_module,
)

STAGES = ["capture", "replay", "coverage", "recipe-join", "timing"]


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    module = load_module("collective_pipeline")
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(module, "DEPENDENCIES", ())
    monkeypatch.setattr(module, "_source", lambda root: {"revision": "a" * 40, "dirty": False})
    calls = []

    def run(command, **kwargs):
        calls.append((Path(kwargs["stdout"].name).stem, command, kwargs))
        kwargs["stdout"].write("retained stage output\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", run)
    evidence = {"ranks": 4, "events": 96, "status": "not_gated"}
    monkeypatch.setattr(
        module.collective_baseline,
        "snapshot",
        lambda *args: ({"collective_evidence": evidence.copy()}, {}),
    )
    return module, calls, evidence


@pytest.mark.parametrize("gpus", [4, 8])
def test_one_source_all_rank_pipeline_requires_every_stage(pipeline, tmp_path, gpus):
    module, calls, evidence = pipeline
    evidence["ranks"] = gpus
    output = tmp_path / "output"
    assert module.main(["--output", str(output), "--gpus", str(gpus)]) == 0
    report = json.loads((output / "producer.json").read_text())
    assert report["status"] == "complete" and report["collective_evidence"]["status"] == "not_gated"
    assert [stage["name"] for stage in report["stages"]] == STAGES
    assert [name for name, _, _ in calls] == STAGES
    assert all(stage["exit_code"] == 0 for stage in report["stages"])
    for name, command, settings in calls:
        assert settings["cwd"] == ROOT and settings["env"]["PYTHONHASHSEED"] == "0"
        assert (output / (name + ".log")).read_text() == "retained stage output\n"
        if name in ("capture", "replay"):
            assert f"--nproc-per-node={gpus}" in command and "--standalone" in command
        if name == "coverage":
            assert command[command.index("--revision") + 1] == "a" * 40
            assert "--require-verified" in command and "--require-case" in command
        if name == "recipe-join":
            assert "--strict" in command
        if name == "timing":
            assert command[command.index("--pairs") + 1] == "3"
            assert command[command.index("--warmup") + 1] == "20"
            assert command[command.index("--steps") + 1] == "50"
            assert not {"--max-overhead-ratio", "--max-regression-ratio"}.intersection(command)
    bindings = json.loads((output / "bindings.json").read_text())
    assert len(bindings) == 6 and all(
        row["adapter"] == "tensor_parallel_collective" for row in bindings
    )


@pytest.mark.parametrize("failed", STAGES)
def test_failed_stage_keeps_evidence_and_stops_before_later_work(
    pipeline, monkeypatch, tmp_path, failed
):
    module, calls, _ = pipeline
    original = module.subprocess.run

    def run(command, **kwargs):
        result = original(command, **kwargs)
        if calls[-1][0] == failed:
            result.returncode = 9
        return result

    monkeypatch.setattr(module.subprocess, "run", run)
    assert module.main(["--output", str(tmp_path), "--gpus", "4"]) == 1
    assert [name for name, _, _ in calls] == STAGES[: STAGES.index(failed) + 1]
    report = json.loads((tmp_path / "producer.json").read_text())
    assert report["status"] == "incomplete" and report["stages"][-1]["exit_code"] == 9
    assert report["stages"][-1]["status"] == "failed"
    assert (tmp_path / (failed + ".log")).read_text() == "retained stage output\n"


@pytest.mark.parametrize(
    "problem", ["dirty", "changed_source", "missing_dependency", "wrong_checkout", "invalid_bundle"]
)
def test_source_and_final_bundle_contracts_cannot_yield_success(
    pipeline, monkeypatch, tmp_path, problem
):
    module, calls, _ = pipeline
    if problem == "dirty":
        monkeypatch.setattr(module, "_source", lambda root: {"revision": "a" * 40, "dirty": True})
    elif problem == "changed_source":
        monkeypatch.setattr(
            module,
            "_source",
            lambda root: {"revision": ("b" if calls else "a") * 40, "dirty": False},
        )
    elif problem == "missing_dependency":
        monkeypatch.setattr(module, "DEPENDENCIES", ("missing-collective-dependency.py",))
    elif problem == "wrong_checkout":
        monkeypatch.chdir(tmp_path)
    else:

        def reject(*args):
            raise ValueError("all-rank accuracy evidence is incomplete")

        monkeypatch.setattr(module.collective_baseline, "snapshot", reject)
    assert module.main(["--output", str(tmp_path / "output"), "--gpus", "4"]) == 1
    report = json.loads((tmp_path / "output/producer.json").read_text())
    assert report["status"] == "incomplete" and report["error"]
    expected = 5 if problem == "invalid_bundle" else 1 if problem == "changed_source" else 0
    assert len(calls) == expected


@pytest.mark.parametrize("key,value", [("ranks", 2), ("events", 95), ("status", "passed")])
def test_pilot_needs_complete_matrix_and_retains_report_only_policy(pipeline, tmp_path, key, value):
    module, _, evidence = pipeline
    evidence[key] = value
    assert module.main(["--output", str(tmp_path), "--gpus", "4"]) == 1
    assert json.loads((tmp_path / "producer.json").read_text())["status"] == "incomplete"


@pytest.mark.parametrize(
    "option,value", [("--pairs", "2"), ("--warmup", "0"), ("--steps", "0"), ("--gpus", "2")]
)
def test_invalid_protocol_is_rejected_before_artifact_creation(pipeline, tmp_path, option, value):
    module, calls, _ = pipeline
    with pytest.raises(SystemExit) as error:
        module.main(["--output", str(tmp_path / "output"), "--gpus", "4", option, value])
    assert error.value.code == 2 and not calls and not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("TORCHELASTIC_RUN_ID", "fixture"),
        ("RANK", "1"),
        ("WORLD_SIZE", "4"),
        ("LOCAL_WORLD_SIZE", "4"),
    ],
)
def test_parent_rejects_nested_distributed_launch(pipeline, monkeypatch, tmp_path, name, value):
    module, calls, _ = pipeline
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as error:
        module.main(["--output", str(tmp_path / "output"), "--gpus", "4"])
    assert error.value.code == 2 and not calls


def test_previous_attempt_is_never_reused_or_deleted(pipeline, tmp_path):
    module, calls, _ = pipeline
    previous = tmp_path / "producer.json"
    previous.write_text("previous failed attempt")
    with pytest.raises(SystemExit):
        module.main(["--output", str(tmp_path), "--gpus", "4"])
    assert previous.read_text() == "previous failed attempt" and not calls


@pytest.mark.parametrize("layout", ["offset", "strided"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_pilot_inputs_retain_values_layout_and_explicit_distinct_gradients(
    monkeypatch, layout, dtype
):
    import torch

    monkeypatch.syspath_prepend(str(SCRIPTS))
    workload = load_module("collective_workload")
    values = workload._tensor([17, 2, 64], getattr(torch, dtype), 1700, 2, layout, "cpu")
    gradient = workload._tensor([17, 2, 64], getattr(torch, dtype), 9100, 2, layout, "cpu")
    expected = torch.randn(
        (17, 2, 64), dtype=torch.float64, generator=torch.Generator().manual_seed(1702)
    ).to(getattr(torch, dtype))
    expected[0, 0, 0] = 3
    assert torch.equal(values, expected) and not torch.equal(values, gradient)
    assert (values.storage_offset() > 0) == (layout == "offset")
    assert values.is_contiguous() == (layout == "offset")


@pytest.mark.parametrize("platform,gpus", [("dgx_h100", 8), ("dgx_gb200", 4)])
@pytest.mark.parametrize(
    "cadence,expected", [("pr", 0), ("mergegroup", 0), ("nightly", 1), (None, 1)]
)
def test_actual_recipe_selection_keeps_pilot_nightly_or_explicit_bypass(
    platform, gpus, cadence, expected
):
    from tests.test_utils.python_scripts.recipe_parser import load_workloads

    workloads = load_workloads(
        container_tag="fixture",
        environment="dev",
        platform=platform,
        test_case="determinism_collective_perf",
        scope="L1",
        cadence=cadence,
    )
    workloads = [workload for workload in workloads if workload.type == "basic"]
    assert len(workloads) == expected
    if workloads:
        spec = workloads[0].spec
        assert spec["gpus"] == gpus and spec["nodes"] == 1 and spec["n_repeat"] == 1
        assert "collective_pipeline.py" in spec["script"]
        assert "--gpus {gpus}" in spec["script"]
