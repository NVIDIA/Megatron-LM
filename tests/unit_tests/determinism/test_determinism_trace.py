# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for rank-local determinism tracing and offline comparison."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from megatron.core.determinism import trace as trace_module
from megatron.core.determinism.trace import DigestMode, RankLocalTrace, TraceConfig
from tools.determinism.trace_comparison import TraceValidationError, compare_traces, load_trace


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
    with RankLocalTrace(TraceConfig(output_dir=path, rank=0, mode=mode, append=False)) as trace:
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


@pytest.mark.parametrize("spec", ["4-", "-4", "word", "1-2-3"])
def test_range_specs_report_labeled_errors(tmp_path, spec):
    with pytest.raises(ValueError, match=rf"invalid iteration_spec range: {spec!r}"):
        TraceConfig(output_dir=tmp_path, rank=0, iteration_spec=spec)


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


def test_restricted_iteration_selection_requires_value_capture_iteration(tmp_path):
    with RankLocalTrace(
        TraceConfig(output_dir=tmp_path, rank=0, mode="full", iteration_spec="2-4")
    ) as trace:
        trace.record_event("lifecycle", iteration=None)
        trace.record_tensor("metadata", torch.ones(1), iteration=None, mode="metadata")
        with pytest.raises(ValueError, match="iteration is required for value-bearing captures"):
            trace.record_tensor("value", torch.ones(1), iteration=None)
        with pytest.raises(ValueError, match="iteration is required for value-bearing captures"):
            trace.record_scalar("python_scalar", 1.0, iteration=None)
        with pytest.raises(ValueError, match="iteration is required for value-bearing captures"):
            trace.record_scalar("tensor_scalar", torch.tensor(1.0), iteration=None)

    assert [record["name"] for record in _records(trace)] == ["lifecycle", "metadata"]


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


def test_complex_summary_uses_element_magnitudes(tmp_path):
    with RankLocalTrace(
        TraceConfig(output_dir=tmp_path, rank=0, mode="summary", append=False)
    ) as trace:
        trace.record_tensor("complex", torch.tensor([3 + 4j, 0 + 2j]), iteration=1)

    summary = _records(trace)[0]["tensor"]["summary"]
    assert summary["minimum"] == 2.0
    assert summary["maximum"] == 5.0
    assert summary["mean"] == 3.5
    assert summary["l2_norm"] == pytest.approx(29**0.5)


def test_sampled_mode_uses_stable_indices_and_digest(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    tensor = torch.arange(10, dtype=torch.float32)
    for output in (left, right):
        with RankLocalTrace(
            TraceConfig(output_dir=output, rank=0, mode="sampled", sample_count=4, append=False)
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


@pytest.mark.parametrize(
    "value,expected", [(np.int64(3), 3), (np.float32(1.25), 1.25), (np.float64(2.5), 2.5)]
)
def test_numpy_scalars_are_normalized_to_json_numbers(tmp_path, value, expected):
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=False)) as trace:
        trace.record_scalar("metric", value, iteration=1)

    assert _records(trace)[0]["fields"] == {"value": expected}


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

    with pytest.raises(ValueError, match="invalid JSON"):
        RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=True))


def test_comparator_reports_first_content_divergence(tmp_path):
    _tensor_trace(tmp_path / "left", torch.tensor([1.0, 2.0]))
    _tensor_trace(tmp_path / "right", torch.tensor([1.0, 3.0]))

    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["match"] is False
    assert report["rank_results"]["0"]["first_divergence"]["kind"] == "content_mismatch"
    assert report["rank_results"]["0"]["compared_records_before_divergence"] == 0
    assert report["rank_results"]["0"]["first_divergence"]["key"]["name"] == "hidden"


def test_comparator_reports_missing_event(tmp_path):
    _event_trace(tmp_path / "left", ["a", "b"])
    _event_trace(tmp_path / "right", ["a"])

    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["rank_results"]["0"]["first_divergence"] == {
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
        "left_sequence": 1,
        "right_sequence": None,
    }


def test_comparator_reports_event_order_mismatch(tmp_path):
    _event_trace(tmp_path / "left", ["a", "b"])
    _event_trace(tmp_path / "right", ["b", "a"])

    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["rank_results"]["0"]["first_divergence"]["kind"] == "event_order_mismatch"


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
        command, cwd=repository_root, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "match"
    assert json.loads(report_path.read_text(encoding="utf-8"))["match"] is True

    module_completed = subprocess.run(
        [sys.executable, "-m", "tools.determinism.compare_traces", str(left), str(right)],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert module_completed.returncode == 0, module_completed.stderr
    assert json.loads(module_completed.stdout)["status"] == "match"


def test_atexit_cuda_graph_flush_failure_emits_incomplete_note(monkeypatch, capsys):
    def fail_during_capture():
        raise RuntimeError("flush of value-bearing traces is not CUDA-graph-capture safe")

    monkeypatch.setattr(trace_module, "close_determinism_trace", fail_during_capture)
    trace_module._close_determinism_trace_at_exit()
    assert "determinism trace incomplete" in capsys.readouterr().err


def test_atexit_does_not_hide_unrelated_runtime_errors(monkeypatch):
    def fail_for_other_reason():
        raise RuntimeError("unrelated failure")

    monkeypatch.setattr(trace_module, "close_determinism_trace", fail_for_other_reason)
    with pytest.raises(RuntimeError, match="unrelated failure"):
        trace_module._close_determinism_trace_at_exit()


@pytest.mark.parametrize("mode", ["summary", "sampled", "full"])
def test_all_value_modes_snapshot_noncontiguous_inputs(tmp_path, mode):
    original = torch.arange(24, dtype=torch.float64).reshape(4, 6).transpose(0, 1)
    mutable = original.clone()
    with RankLocalTrace(TraceConfig(output_dir=tmp_path / "left", rank=0, mode=mode)) as trace:
        trace.record_tensor("hidden", mutable, iteration=1, phase="forward")
        mutable.add_(100)
    _tensor_trace(tmp_path / "right", original, DigestMode(mode))
    assert compare_traces(tmp_path / "left", tmp_path / "right")["match"]


@pytest.mark.parametrize(
    "tensor", [torch.tensor(3.0), torch.empty(0), torch.ones(1, 4).expand(3, 4)]
)
def test_sampled_scalar_empty_and_expanded_inputs(tmp_path, tensor):
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, mode="sampled")) as trace:
        trace.record_tensor("input", tensor, iteration=1)
    evidence = load_trace(tmp_path)[0]["tensor"]
    assert evidence["captured_numel"] == tensor.numel()


def test_metadata_tensor_capture_does_not_invoke_value_operations(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("metadata capture must not read tensor values")

    tensor = torch.ones(4)
    for name in ("clone", "cpu", "item", "tolist", "numpy", "detach"):
        monkeypatch.setattr(torch.Tensor, name, unexpected)
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0)) as trace:
        trace.record_tensor("input", tensor, iteration=1)
    assert load_trace(tmp_path)[0]["tensor"]["value_observed"] is False


def test_scalar_override_is_explicit_in_metadata_trace(tmp_path):
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, mode="metadata")) as trace:
        trace.record_scalar("loss", torch.tensor(2.0), iteration=1)
    evidence = load_trace(tmp_path)[0]["tensor"]
    assert evidence["mode"] == "full"
    assert evidence["value_observed"] is True


def test_append_preserves_a_valid_final_line_without_newline(tmp_path):
    _event_trace(tmp_path, ["first"])
    path = tmp_path / "rank_000000.jsonl"
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=True)) as trace:
        trace.record_event("second", iteration=1, phase="forward")
    assert [record["name"] for record in load_trace(tmp_path)] == ["first", "second"]


@pytest.mark.parametrize("mutation", ["duplicate", "missing_fields", "invalid_evidence"])
def test_append_uses_the_same_contract_as_the_offline_loader(tmp_path, mutation):
    if mutation == "invalid_evidence":
        _tensor_trace(tmp_path, torch.ones(2))
    else:
        _event_trace(tmp_path, ["first"])
    path = tmp_path / "rank_000000.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "duplicate":
        extra = dict(record, sequence=1)
        text = json.dumps(record) + "\n" + json.dumps(extra) + "\n"
    else:
        if mutation == "missing_fields":
            record.pop("fields")
        else:
            record["tensor"]["captured_numel"] = 1
        text = json.dumps(record) + "\n"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0, append=True))
    with pytest.raises(TraceValidationError):
        load_trace(tmp_path)
    assert path.read_text(encoding="utf-8") == text


def test_failed_capture_does_not_consume_occurrence_or_sequence(tmp_path):
    with RankLocalTrace(TraceConfig(output_dir=tmp_path, rank=0)) as trace:
        with pytest.raises(TypeError):
            trace.record_tensor("same", object(), iteration=1)
        trace.record_tensor("same", torch.ones(1), iteration=1)
    record = load_trace(tmp_path)[0]
    assert record["sequence"] == record["occurrence"] == 0


@pytest.mark.parametrize(
    "kwargs", [{"rank": True}, {"rank": 0, "sample_count": 1.5}, {"rank": 0, "flush_every": False}]
)
def test_config_rejects_non_integer_counters(tmp_path, kwargs):
    with pytest.raises(ValueError):
        TraceConfig(output_dir=tmp_path, **kwargs)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_full_capture_preserves_supported_logical_dtypes(tmp_path, dtype):
    tensor = torch.arange(8).to(dtype).reshape(2, 4).t()
    _tensor_trace(tmp_path / "left", tensor)
    _tensor_trace(tmp_path / "right", tensor.clone())
    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["match_strength"] == "full_tensor_certificate"


def test_full_capture_distinguishes_signed_zero(tmp_path):
    _tensor_trace(tmp_path / "left", torch.tensor([0.0]))
    _tensor_trace(tmp_path / "right", torch.tensor([-0.0]))
    assert not compare_traces(tmp_path / "left", tmp_path / "right")["match"]
