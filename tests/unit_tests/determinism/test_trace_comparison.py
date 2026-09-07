# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline trace contracts, requiring only pytest and the Python standard library."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.determinism.trace_comparison import TraceValidationError, compare_traces, load_trace


def _event(name="boundary", sequence=0, rank=0, value=0):
    return {
        "event": "event",
        "fields": {"value": value},
        "iteration": 1,
        "microbatch": None,
        "name": name,
        "occurrence": 0,
        "phase": "forward",
        "rank": rank,
        "schema_version": 1,
        "sequence": sequence,
    }


def _full_tensor():
    record = _event()
    record.pop("fields")
    record.update(
        event="tensor",
        metadata={},
        tensor={
            "mode": "full",
            "shape": [2],
            "stride": [1],
            "numel": 2,
            "dtype": "float32",
            "layout": "strided",
            "device_type": "cpu",
            "requires_grad": False,
            "value_observed": True,
            "exact_value_certificate": True,
            "sha256": "0" * 64,
            "captured_numel": 2,
            "all_finite": True,
        },
    )
    return record


def _write(directory, records, filename="rank_000000.jsonl"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _compare(tmp_path, left, right):
    _write(tmp_path / "left", left)
    _write(tmp_path / "right", right)
    return compare_traces(tmp_path / "left", tmp_path / "right")["rank_results"]["0"]


def test_first_content_mismatch_uses_execution_order(tmp_path):
    result = _compare(
        tmp_path,
        [_event("z_first", 0), _event("a_later", 1)],
        [_event("z_first", 0, value=1), _event("a_later", 1, value=1)],
    )
    assert result["first_divergence"]["key"]["name"] == "z_first"
    assert result["first_divergence"]["left_sequence"] == 0
    assert result["compared_records_before_divergence"] == 0


def test_order_mismatch_precedes_later_content_mismatch(tmp_path):
    result = _compare(
        tmp_path,
        [_event("a", 0), _event("b", 1), _event("c", 2)],
        [_event("b", 0), _event("a", 1), _event("c", 2, value=1)],
    )
    assert result["first_divergence"]["kind"] == "event_order_mismatch"
    assert result["first_divergence"]["left_sequence"] == 0
    assert result["compared_records_before_divergence"] == 0


def test_missing_event_precedes_later_content_mismatch(tmp_path):
    result = _compare(
        tmp_path, [_event("z_missing", 0), _event("a_later", 1)], [_event("a_later", 0, value=1)]
    )
    divergence = result["first_divergence"]
    assert divergence["kind"] == "missing_event"
    assert divergence["key"]["name"] == "z_missing"
    assert divergence["missing_from"] == "right"
    assert (divergence["left_sequence"], divergence["right_sequence"]) == (0, 0)


def test_different_events_report_both_execution_candidates(tmp_path):
    divergence = _compare(tmp_path, [_event("z")], [_event("a")])["first_divergence"]
    assert divergence["kind"] == "event_mismatch"
    assert divergence["left"]["name"] == "z"
    assert divergence["right"]["name"] == "a"


def test_sequence_offsets_do_not_change_identity(tmp_path):
    result = _compare(
        tmp_path, [_event("a", 10), _event("b", 20)], [_event("a", 0), _event("b", 1)]
    )
    assert result["match"]
    assert result["compared_records_before_divergence"] == 2


def test_occurrences_and_iteration_order_follow_sequence(tmp_path):
    left = [_event(sequence=0), dict(_event(sequence=1), occurrence=1, iteration=0)]
    right = copy.deepcopy(left)
    right[0]["fields"]["value"] = 1
    right[1]["fields"]["value"] = 2
    result = _compare(tmp_path, left, right)
    assert result["first_divergence"]["left_sequence"] == 0


def test_each_rank_has_its_own_first_divergence(tmp_path):
    for side, value in (("left", 0), ("right", 1)):
        _write(tmp_path / side, [_event("late", 99, value=value)])
        _write(tmp_path / side, [_event("early", 0, rank=1, value=value)], "rank_000001.jsonl")
    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert not report["match"]
    assert "first_divergence" not in report
    assert "compared_records_before_divergence" not in report
    assert report["rank_results"]["0"]["first_divergence"]["left_sequence"] == 99
    assert report["rank_results"]["1"]["first_divergence"]["left_sequence"] == 0


def test_missing_rank_is_reported(tmp_path):
    for side in ("left", "right"):
        _write(tmp_path / side, [_event()])
    _write(tmp_path / "left", [_event(rank=1)], "rank_000001.jsonl")
    report = compare_traces(tmp_path / "left", tmp_path / "right")
    assert report["rank_results"]["0"]["match"]
    assert report["rank_results"]["1"]["first_divergence"]["missing_from"] == "right"
    assert report["left"]["ranks"] == [0, 1]
    assert report["right"]["ranks"] == [0]


@pytest.mark.parametrize(
    "filename", ["rank_backup.jsonl", "rank_backup_rank_0.jsonl", "rank_-1.jsonl"]
)
def test_directory_rejects_malformed_rank_names(tmp_path, filename):
    _write(tmp_path, [_event()], filename)
    with pytest.raises(TraceValidationError, match="expected filename"):
        load_trace(tmp_path)


def test_directory_rejects_rank_aliases(tmp_path):
    _write(tmp_path, [_event()])
    _write(tmp_path, [_event("other")], "rank_0.jsonl")
    with pytest.raises(TraceValidationError, match="duplicate trace file"):
        load_trace(tmp_path)


def test_directory_rejects_an_empty_rank_among_valid_ranks(tmp_path):
    _write(tmp_path, [_event()])
    _write(tmp_path, [], "rank_000001.jsonl")
    with pytest.raises(TraceValidationError, match="contains no records"):
        load_trace(tmp_path)


def test_explicit_arbitrary_filename_still_requires_one_rank(tmp_path):
    path = _write(tmp_path, [_event()], "arbitrary.jsonl")
    assert len(load_trace(path)) == 1
    _write(tmp_path, [_event(), _event(sequence=1, rank=1)], path.name)
    with pytest.raises(TraceValidationError, match="does not match filename rank"):
        load_trace(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("rank", False),
        ("sequence", 1.0),
        ("occurrence", []),
        ("iteration", True),
        ("microbatch", -1),
        ("event", []),
        ("phase", ""),
    ],
)
def test_loader_rejects_wrong_record_types(tmp_path, field, value):
    record = _event()
    record[field] = value
    _write(tmp_path, [record])
    with pytest.raises(TraceValidationError):
        load_trace(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", []),
        ("mode", {}),
        ("shape", [True]),
        ("numel", 3),
        ("captured_numel", 1),
        ("captured_numel", True),
        ("dtype", None),
        ("stride", []),
        ("requires_grad", 0),
        ("value_observed", False),
        ("exact_value_certificate", False),
        ("sha256", "invalid"),
        ("all_finite", None),
    ],
)
def test_loader_rejects_contradictory_full_evidence(tmp_path, field, value):
    record = _full_tensor()
    record["tensor"][field] = value
    _write(tmp_path, [record])
    with pytest.raises(TraceValidationError):
        load_trace(tmp_path)


@pytest.mark.parametrize("indices", [[], [0, 0], [-1, 1], [0, 2], [True, 1], [1, 0]])
def test_loader_rejects_invalid_sample_indices(tmp_path, indices):
    record = _full_tensor()
    record["tensor"].update(mode="sampled", exact_value_certificate=False, sample_indices=indices)
    _write(tmp_path, [record])
    with pytest.raises(TraceValidationError):
        load_trace(tmp_path)


def test_loader_rejects_duplicate_json_fields(tmp_path):
    path = _write(tmp_path, [_event()])
    path.write_text(
        path.read_text(encoding="utf-8").replace('"rank": 0', '"rank": 1, "rank": 0'),
        encoding="utf-8",
    )
    with pytest.raises(TraceValidationError, match="duplicate JSON field"):
        load_trace(tmp_path)


@pytest.mark.parametrize(
    "records,error",
    [
        ([_event(), _event("other")], "sequence must be strictly increasing"),
        ([_event(), _event(sequence=1)], "duplicate semantic event key"),
    ],
)
def test_loader_rejects_invalid_streams(tmp_path, records, error):
    _write(tmp_path, records)
    with pytest.raises(TraceValidationError, match=error):
        load_trace(tmp_path)


@pytest.mark.parametrize(
    "case,code,status",
    [
        ("match", 0, "match"),
        ("diverged", 1, "diverged"),
        ("invalid_mode", 2, "invalid_trace"),
        ("invalid_utf8", 2, "invalid_trace"),
    ],
)
def test_cli_exit_codes(tmp_path, case, code, status):
    left = _write(tmp_path / "left", [_event()])
    right = _write(tmp_path / "right", [_event(value=int(case == "diverged"))])
    if case == "invalid_mode":
        record = _full_tensor()
        record["tensor"]["mode"] = []
        right = _write(tmp_path / "right", [record])
    elif case == "invalid_utf8":
        right.write_bytes(b"\xff\xfe")
    report_path = tmp_path / "report.json"
    cli = Path(__file__).parents[3] / "tools/determinism/compare_traces.py"
    result = subprocess.run(
        [sys.executable, "-B", str(cli), str(left), str(right), "--output", str(report_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == code, result.stderr
    assert json.loads(result.stdout)["status"] == status
    assert json.loads(report_path.read_text(encoding="utf-8")) == json.loads(result.stdout)
