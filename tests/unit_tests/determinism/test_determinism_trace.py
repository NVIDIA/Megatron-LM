# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for rank-local determinism tracing and offline comparison."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from megatron.training.determinism_trace import DigestMode, RankLocalTrace, TraceConfig
from tools.determinism.trace_comparison import (
    TraceValidationError,
    compare_traces,
    load_trace,
)


def _records(trace: RankLocalTrace) -> list[dict]:
    return [
        json.loads(line)
        for line in trace.output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _event_trace(path: Path, names: list[str]) -> None:
    with RankLocalTrace(TraceConfig(output_dir=path, rank=0, append=False)) as trace:
        for name in names:
            trace.record_event(name, iteration=1, phase="forward")


def _tensor_trace(path: Path, tensor: torch.Tensor, mode: DigestMode = DigestMode.FULL) -> None:
    with RankLocalTrace(
        TraceConfig(output_dir=path, rank=0, mode=mode, append=False)
    ) as trace:
        trace.record_tensor("hidden", tensor, iteration=1, phase="forward")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": -1},
        {"rank": 0, "sample_count": 0},
        {"rank": 0, "flush_every": -1},
        {"rank": 0, "rank_spec": "3-1"},
        {"rank": 0, "iteration_spec": "1,,2"},
        {"rank": 0, "mode": "invalid"},
    ],
)
def test_config_rejects_invalid_values(tmp_path, kwargs):
    with pytest.raises(ValueError):
        TraceConfig(output_dir=tmp_path, **kwargs)


def test_rank_and_iteration_selection(tmp_path):
    disabled = RankLocalTrace(
        TraceConfig(output_dir=tmp_path / "disabled", rank=3, rank_spec="0-2", append=False)
    )
    disabled.record_event("ignored", iteration=2)
    disabled.close()
    assert not disabled.output_path.exists()

    with RankLocalTrace(
        TraceConfig(
            output_dir=tmp_path / "selected",
            rank=2,
            rank_spec="0,2-3",
            iteration_spec="2,4-5",
            append=False,
        )
    ) as trace:
        trace.record_event("ignored", iteration=1)
        trace.record_event("kept", iteration=2)

    assert [record["name"] for record in _records(trace)] == ["kept"]


def test_metadata_mode_does_not_observe_values(tmp_path):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    with RankLocalTrace(
        TraceConfig(output_dir=tmp_path, rank=0, mode="metadata", append=False)
    ) as trace:
        trace.record_tensor("activation", tensor, iteration=7, phase="forward")

    evidence = _records(trace)[0]["tensor"]
    assert evidence["shape"] == [4, 3]
    assert evidence["stride"] == [1, 4]
    assert evidence["value_observed"] is False
    assert "sha256" not in evidence


def test_summary_mode_handles_empty_and_nonfinite_tensors(tmp_path):
    with RankLocalTrace(
        TraceConfig(output_dir=tmp_path, rank=0, mode="summary", append=False)
    ) as trace:
        trace.record_tensor("empty", torch.empty(0), iteration=1)
        trace.record_tensor("nonfinite", torch.tensor([1.0, float("inf")]), iteration=1)

    empty, nonfinite = [record["tensor"]["summary"] for record in _records(trace)]
    assert empty == {
        "all_finite": True,
        "l2_norm": 0.0,
        "maximum": None,
        "mean": None,
        "minimum": None,
    }
    assert nonfinite["all_finite"] is False
    assert nonfinite["maximum"] is None
    assert nonfinite["mean"] is None


def test_sampled_mode_uses_stable_indices_and_digest(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    tensor = torch.arange(10, dtype=torch.float32)
    for output in (left, right):
        with RankLocalTrace(
            TraceConfig(
                output_dir=output,
                rank=0,
                mode="sampled",
                sample_count=4,
                append=False,
            )
        ) as trace:
            trace.record_tensor("activation", tensor, iteration=1)

    left_evidence = load_trace(left)[0]["tensor"]
    right_evidence = load_trace(right)[0]["tensor"]
    assert left_evidence["sample_indices"] == [0, 3, 6, 9]
    assert left_evidence["captured_numel"] == 4
    assert left_evidence["sha256"] == right_evidence["sha256"]
    assert left_evidence["exact_value_certificate"] is False


def test_full_mode_snapshots_before_mutation(tmp_path):
    original = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4).transpose(0, 1)
    mutable = original.clone()

    with RankLocalTrace(
        TraceConfig(output_dir=tmp_path / "left", rank=0, mode="full", append=False)
    ) as trace:
        trace.record_tensor("hidden", mutable, iteration=1, phase="forward")
        mutable.add_(100)

    _tensor_trace(tmp_path / "right", original, DigestMode.FULL)
    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["match"] is True
    assert report["match_strength"] == "full_tensor_certificate"


def test_scalars_and_occurrences_are_json_safe(tmp_path):
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=False)) as trace:
        trace.record_scalar("metric", float("nan"), iteration=1)
        trace.record_scalar("metric", 3.5, iteration=1)
        trace.record_scalar("tensor_metric", torch.tensor(2.0), iteration=1)

    records = _records(trace)
    assert [record["occurrence"] for record in records[:2]] == [0, 1]
    assert records[0]["fields"] == {"finite": False, "nonfinite": "nan", "value": None}
    assert records[2]["tensor"]["exact_value_certificate"] is True


def test_create_only_and_append_continuation(tmp_path):
    _event_trace(tmp_path, ["first"])
    with pytest.raises(FileExistsError):
        RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=False))

    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=True)) as trace:
        trace.record_event("first", iteration=1, phase="forward")

    records = _records(trace)
    assert [record["sequence"] for record in records] == [0, 1]
    assert [record["occurrence"] for record in records] == [0, 1]


def test_append_rejects_malformed_existing_trace(tmp_path):
    output = tmp_path / "rank_000000.jsonl"
    output.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid existing trace JSON"):
        RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=True))


def test_comparator_reports_first_content_divergence(tmp_path):
    _tensor_trace(tmp_path / "left", torch.tensor([1.0, 2.0]))
    _tensor_trace(tmp_path / "right", torch.tensor([1.0, 3.0]))

    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["match"] is False
    assert report["first_divergence"]["kind"] == "content_mismatch"
    assert report["compared_records_before_divergence"] == 0
    assert report["first_divergence"]["key"]["name"] == "hidden"


def test_comparator_reports_missing_event(tmp_path):
    _event_trace(tmp_path / "left", ["a", "b"])
    _event_trace(tmp_path / "right", ["a"])

    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["first_divergence"] == {
        "kind": "missing_event",
        "key": {
            "event": "event",
            "iteration": 1,
            "microbatch": None,
            "name": "b",
            "occurrence": 0,
            "phase": "forward",
            "rank": 0,
        },
        "missing_from": "right",
    }


def test_comparator_reports_event_order_mismatch(tmp_path):
    _event_trace(tmp_path / "left", ["a", "b"])
    _event_trace(tmp_path / "right", ["b", "a"])

    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["first_divergence"]["kind"] == "event_order_mismatch"


@pytest.mark.parametrize(
    "contents,error",
    [
        ("", "contains no records"),
        ("{not-json}\n", "invalid JSON"),
        (
            json.dumps(
                {
                    "event": "event",
                    "fields": {},
                    "iteration": 1,
                    "microbatch": None,
                    "name": "x",
                    "occurrence": 0,
                    "phase": "forward",
                    "rank": 1,
                    "schema_version": 1,
                    "sequence": 0,
                }
            ),
            "does not match filename rank",
        ),
    ],
)
def test_loader_rejects_invalid_traces(tmp_path, contents, error):
    (tmp_path / "rank_000000.jsonl").write_text(contents, encoding="utf-8")
    with pytest.raises(TraceValidationError, match=error):
        load_trace(tmp_path)


def test_loader_rejects_invalid_tensor_evidence(tmp_path):
    record = {
        "event": "tensor",
        "iteration": 1,
        "metadata": {},
        "microbatch": None,
        "name": "hidden",
        "occurrence": 0,
        "phase": "forward",
        "rank": 0,
        "schema_version": 1,
        "sequence": 0,
        "tensor": {"mode": "unknown", "shape": [2]},
    }
    (tmp_path / "rank_000000.jsonl").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(TraceValidationError, match="invalid tensor evidence mode"):
        load_trace(tmp_path)


def test_loader_rejects_nonfinite_json(tmp_path):
    record = {
        "event": "event",
        "fields": {"value": float("nan")},
        "iteration": 1,
        "microbatch": None,
        "name": "loss",
        "occurrence": 0,
        "phase": "forward",
        "rank": 0,
        "schema_version": 1,
        "sequence": 0,
    }
    (tmp_path / "rank_000000.jsonl").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(TraceValidationError, match="non-finite number"):
        load_trace(tmp_path)


def test_comparison_cli_exit_codes_and_report(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    report_path = tmp_path / "report.json"
    _event_trace(left, ["same"])
    _event_trace(right, ["same"])
    repository_root = Path(__file__).parents[3]
    command = [
        sys.executable,
        str(repository_root / "tools/determinism/compare_traces.py"),
        str(left),
        str(right),
        "--output",
        str(report_path),
    ]

    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "match"
    assert json.loads(report_path.read_text(encoding="utf-8"))["match"] is True
