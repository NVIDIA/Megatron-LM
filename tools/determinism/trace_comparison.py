# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline comparison for rank-local determinism traces."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

TRACE_SCHEMA_VERSION = 1

_RANK_FILE_RE = re.compile(r"rank_(\d+)\.jsonl$")
_REQUIRED_FIELDS = {
    "event",
    "iteration",
    "microbatch",
    "name",
    "occurrence",
    "phase",
    "rank",
    "schema_version",
    "sequence",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TENSOR_MODES = {"metadata", "summary", "sampled", "full"}


class TraceValidationError(ValueError):
    """Raised when a trace does not satisfy the comparison schema."""


def _validate_json_tree(value: Any, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise TraceValidationError(f"{location}: JSON contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _validate_json_tree(item, location)
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TraceValidationError(f"{location}: JSON object keys must be strings")
            _validate_json_tree(item, location)


def _validate_tensor_evidence(tensor: Any, location: str) -> None:
    if not isinstance(tensor, dict):
        raise TraceValidationError(f"{location}: tensor event is missing tensor evidence")
    mode = tensor.get("mode")
    if mode not in _TENSOR_MODES:
        raise TraceValidationError(f"{location}: invalid tensor evidence mode: {mode!r}")
    if not isinstance(tensor.get("shape"), list) or not all(
        isinstance(dimension, int) and dimension >= 0 for dimension in tensor["shape"]
    ):
        raise TraceValidationError(f"{location}: tensor shape must contain non-negative integers")
    if mode in {"sampled", "full"} and not _SHA256_RE.fullmatch(str(tensor.get("sha256", ""))):
        raise TraceValidationError(f"{location}: {mode} tensor evidence requires a SHA-256 digest")
    if mode == "summary" and not isinstance(tensor.get("summary"), dict):
        raise TraceValidationError(f"{location}: summary tensor evidence is missing its summary")


def _trace_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise TraceValidationError(f"trace path does not exist: {path}")
    files = sorted(path.glob("rank_*.jsonl"))
    if not files:
        raise TraceValidationError(f"no rank_*.jsonl files found under {path}")
    return files


def _validate_record(record: Any, path: Path, line_number: int, expected_rank: int | None) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    if not isinstance(record, dict):
        raise TraceValidationError(f"{location}: record must be a JSON object")
    _validate_json_tree(record, location)
    missing = sorted(_REQUIRED_FIELDS - set(record))
    if missing:
        raise TraceValidationError(f"{location}: missing fields: {missing}")
    if record["schema_version"] != TRACE_SCHEMA_VERSION:
        raise TraceValidationError(
            f"{location}: unsupported schema_version={record['schema_version']!r}"
        )
    for field in ("rank", "sequence", "occurrence"):
        if not isinstance(record[field], int) or record[field] < 0:
            raise TraceValidationError(f"{location}: {field} must be a non-negative integer")
    for field in ("iteration", "microbatch"):
        if record[field] is not None and (
            not isinstance(record[field], int) or record[field] < 0
        ):
            raise TraceValidationError(f"{location}: {field} must be null or a non-negative integer")
    for field in ("event", "name", "phase"):
        if not isinstance(record[field], str) or not record[field]:
            raise TraceValidationError(f"{location}: {field} must be a non-empty string")
    if record["event"] not in {"event", "tensor"}:
        raise TraceValidationError(f"{location}: unsupported event type: {record['event']!r}")
    if record["event"] == "tensor":
        _validate_tensor_evidence(record.get("tensor"), location)
    if record["event"] == "event" and not isinstance(record.get("fields"), dict):
        raise TraceValidationError(f"{location}: semantic event is missing fields")
    if expected_rank is not None and record["rank"] != expected_rank:
        raise TraceValidationError(
            f"{location}: record rank {record['rank']} does not match filename rank {expected_rank}"
        )
    return record


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate every rank-local JSONL record at ``path``."""
    root = Path(path)
    records: list[dict[str, Any]] = []
    ranks_seen: set[int] = set()
    for trace_file in _trace_files(root):
        match = _RANK_FILE_RE.search(trace_file.name)
        expected_rank = int(match.group(1)) if match else None
        if expected_rank is not None and expected_rank in ranks_seen:
            raise TraceValidationError(f"duplicate trace file for rank {expected_rank}")
        if expected_rank is not None:
            ranks_seen.add(expected_rank)

        previous_sequence = -1
        with trace_file.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TraceValidationError(f"{trace_file}:{line_number}: invalid JSON") from exc
                record = _validate_record(decoded, trace_file, line_number, expected_rank)
                if record["sequence"] <= previous_sequence:
                    raise TraceValidationError(
                        f"{trace_file}:{line_number}: sequence must be strictly increasing"
                    )
                previous_sequence = record["sequence"]
                records.append(record)
    if not records:
        raise TraceValidationError(f"trace contains no records: {path}")
    return records


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["rank"],
        record["iteration"],
        record["microbatch"],
        record["phase"],
        record["event"],
        record["name"],
        record["occurrence"],
    )


def _sortable_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
    rank, iteration, microbatch, phase, event, name, occurrence = key
    return (
        -1 if iteration is None else iteration,
        -1 if microbatch is None else microbatch,
        rank,
        phase,
        event,
        name,
        occurrence,
    )


def _key_json(key: tuple[Any, ...]) -> dict[str, Any]:
    rank, iteration, microbatch, phase, event, name, occurrence = key
    return {
        "event": event,
        "iteration": iteration,
        "microbatch": microbatch,
        "name": name,
        "occurrence": occurrence,
        "phase": phase,
        "rank": rank,
    }


def _semantic_record(record: dict[str, Any]) -> dict[str, Any]:
    semantic = dict(record)
    semantic.pop("sequence", None)
    return semantic


def _different_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    fields = set(left) | set(right)
    return sorted(field for field in fields if left.get(field) != right.get(field))


def _index_records(records: Iterable[dict[str, Any]], label: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = _record_key(record)
        if key in indexed:
            raise TraceValidationError(f"{label}: duplicate semantic event key: {_key_json(key)}")
        indexed[key] = record
    return indexed


def _mode_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        record["tensor"].get("mode", "unknown")
        for record in records
        if record["event"] == "tensor"
    )
    return dict(sorted(counts.items()))


def compare_traces(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    """Compare two trace directories and report their first semantic divergence.

    Sequence numbers are validated for monotonicity but are not part of event
    identity. Event ordering is checked separately so insertion of one record
    does not turn every following event into a content mismatch.
    """
    left_records = load_trace(left_path)
    right_records = load_trace(right_path)
    left = _index_records(left_records, "left")
    right = _index_records(right_records, "right")

    first_divergence: dict[str, Any] | None = None
    compared = 0
    for key in sorted(set(left) | set(right), key=_sortable_key):
        left_record = left.get(key)
        right_record = right.get(key)
        if left_record is None or right_record is None:
            first_divergence = {
                "kind": "missing_event",
                "key": _key_json(key),
                "missing_from": "left" if left_record is None else "right",
            }
            break
        left_semantic = _semantic_record(left_record)
        right_semantic = _semantic_record(right_record)
        if left_semantic != right_semantic:
            first_divergence = {
                "differing_fields": _different_fields(left_semantic, right_semantic),
                "key": _key_json(key),
                "kind": "content_mismatch",
                "left": left_semantic,
                "right": right_semantic,
            }
            break
        compared += 1

    if first_divergence is None:
        left_order = [
            _record_key(record)
            for record in sorted(left_records, key=lambda item: (item["rank"], item["sequence"]))
        ]
        right_order = [
            _record_key(record)
            for record in sorted(right_records, key=lambda item: (item["rank"], item["sequence"]))
        ]
        if left_order != right_order:
            mismatch_index = next(
                index
                for index, values in enumerate(zip(left_order, right_order))
                if values[0] != values[1]
            )
            first_divergence = {
                "index": mismatch_index,
                "kind": "event_order_mismatch",
                "left": _key_json(left_order[mismatch_index]),
                "right": _key_json(right_order[mismatch_index]),
            }

    mode_counts = _mode_counts(left_records)
    if mode_counts.get("full", 0) == sum(mode_counts.values()) and mode_counts:
        match_strength = "full_tensor_certificate"
    elif mode_counts.get("metadata", 0) == sum(mode_counts.values()) and mode_counts:
        match_strength = "structure_only"
    elif mode_counts:
        match_strength = "diagnostic_tensor_match"
    else:
        match_strength = "event_match"

    return {
        "compared_records_before_divergence": compared,
        "first_divergence": first_divergence,
        "left": {
            "path": str(left_path),
            "records": len(left_records),
            "tensor_modes": mode_counts,
        },
        "match": first_divergence is None,
        "match_strength": match_strength if first_divergence is None else "none",
        "right": {
            "path": str(right_path),
            "records": len(right_records),
            "tensor_modes": _mode_counts(right_records),
        },
        "schema_version": TRACE_SCHEMA_VERSION,
    }
