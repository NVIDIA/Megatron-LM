# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Standard-library-only trace contract shared by the recorder and offline tools."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

TRACE_SCHEMA_VERSION = 1
_RANK_FILE_RE = re.compile(r"rank_([0-9]+)\.jsonl")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_TENSOR_MODES = {"metadata", "summary", "sampled", "full"}
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


class TraceValidationError(ValueError):
    """A trace is malformed or inconsistent with its declared evidence strength."""


def _require(condition: bool, location: str, message: str) -> None:
    if not condition:
        raise TraceValidationError(f"{location}: {message}")


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _validate_json_tree(value: Any, location: str) -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), location, "JSON contains a non-finite number")
    elif isinstance(value, list):
        for item in value:
            _validate_json_tree(item, location)
    elif isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), location, "JSON object keys must be strings")
            _validate_json_tree(item, location)


def _validate_tensor(tensor: Any, location: str) -> None:
    _require(isinstance(tensor, dict), location, "tensor event is missing tensor evidence")
    mode = tensor.get("mode")
    _require(
        isinstance(mode, str) and mode in _TENSOR_MODES,
        location,
        f"invalid tensor evidence mode: {mode!r}",
    )
    shape = tensor.get("shape")
    _require(
        isinstance(shape, list) and all(_nonnegative_int(x) for x in shape),
        location,
        "tensor shape must contain non-negative integers",
    )
    numel = tensor.get("numel")
    expected_numel = 1
    for dimension in shape:
        expected_numel *= dimension
    _require(
        _nonnegative_int(numel) and numel == expected_numel,
        location,
        "tensor numel must equal the product of shape",
    )
    for field in ("dtype", "layout", "device_type"):
        _require(
            isinstance(tensor.get(field), str) and bool(tensor[field]),
            location,
            f"tensor {field} must be a non-empty string",
        )
    _require(
        type(tensor.get("requires_grad")) is bool,
        location,
        "tensor requires_grad must be a boolean",
    )
    if tensor["layout"] == "strided":
        stride = tensor.get("stride")
        _require(
            isinstance(stride, list)
            and len(stride) == len(shape)
            and all(_nonnegative_int(x) for x in stride),
            location,
            "invalid tensor stride",
        )
    _require(
        tensor.get("value_observed") is (mode != "metadata"),
        location,
        "value_observed contradicts tensor mode",
    )
    _require(
        tensor.get("exact_value_certificate") is (mode == "full"),
        location,
        "exact_value_certificate contradicts tensor mode",
    )
    if mode == "metadata":
        _require(
            not (
                {"sha256", "summary", "captured_numel", "sample_indices", "all_finite"}
                & tensor.keys()
            ),
            location,
            "metadata evidence must not contain values",
        )
        return
    _require(
        tensor["layout"] == "strided" and tensor["device_type"] != "meta",
        location,
        "value-bearing evidence requires a materialized strided tensor",
    )
    if mode == "summary":
        summary = tensor.get("summary")
        _require(
            isinstance(summary, dict), location, "summary tensor evidence is missing its summary"
        )
        _require(type(summary.get("all_finite")) is bool, location, "invalid summary all_finite")
        for field in ("minimum", "maximum", "mean", "l2_norm"):
            _require(
                field in summary
                and (summary[field] is None or type(summary[field]) in (int, float)),
                location,
                f"invalid summary {field}",
            )
        _require(
            not ({"sha256", "captured_numel", "sample_indices"} & tensor.keys()),
            location,
            "summary evidence must not contain a tensor digest",
        )
        return
    digest = tensor.get("sha256")
    _require(
        isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
        location,
        f"{mode} tensor evidence requires a SHA-256 digest",
    )
    captured = tensor.get("captured_numel")
    _require(_nonnegative_int(captured) and captured <= numel, location, "invalid captured_numel")
    _require(type(tensor.get("all_finite")) is bool, location, "invalid tensor all_finite")
    _require("summary" not in tensor, location, "digest evidence must not contain a summary")
    if mode == "full":
        _require(
            captured == numel and "sample_indices" not in tensor,
            location,
            "full evidence must capture every element without sampling",
        )
    else:
        indices = tensor.get("sample_indices")
        _require(
            isinstance(indices, list)
            and len(indices) == captured
            and all(_nonnegative_int(x) and x < numel for x in indices),
            location,
            "invalid sample_indices",
        )
        _require(
            (numel == 0 and captured == 0) or (numel > 0 and captured > 0),
            location,
            "sampled evidence must capture non-empty tensors",
        )
        expected = (
            []
            if captured == 0
            else (
                [0]
                if captured == 1
                else [(i * (numel - 1)) // (captured - 1) for i in range(captured)]
            )
        )
        _require(
            indices == expected,
            location,
            "sample_indices must be deterministic evenly spaced indices",
        )


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return semantic identity, excluding the rank-local execution sequence."""
    return tuple(
        record[field]
        for field in ("rank", "iteration", "microbatch", "phase", "event", "name", "occurrence")
    )


def validate_record(record: Any, location: str, expected_rank: int | None = None) -> dict[str, Any]:
    """Validate a decoded record, including the strength of tensor evidence."""
    _require(isinstance(record, dict), location, "record must be a JSON object")
    _validate_json_tree(record, location)
    missing = sorted(_REQUIRED_FIELDS - record.keys())
    _require(not missing, location, f"missing fields: {missing}")
    _require(
        type(record["schema_version"]) is int and record["schema_version"] == TRACE_SCHEMA_VERSION,
        location,
        f"unsupported schema_version={record['schema_version']!r}",
    )
    for field in ("rank", "sequence", "occurrence"):
        _require(
            _nonnegative_int(record[field]), location, f"{field} must be a non-negative integer"
        )
    for field in ("iteration", "microbatch"):
        _require(
            record[field] is None or _nonnegative_int(record[field]),
            location,
            f"{field} must be null or a non-negative integer",
        )
    for field in ("event", "name", "phase"):
        value = record[field]
        _require(
            isinstance(value, str)
            and 0 < len(value) <= 512
            and all(ord(char) >= 32 for char in value),
            location,
            f"invalid {field}",
        )
    _require(record["event"] in {"event", "tensor"}, location, "unsupported event type")
    if record["event"] == "tensor":
        _require(isinstance(record.get("metadata"), dict), location, "invalid tensor metadata")
        _validate_tensor(record.get("tensor"), location)
    else:
        _require(
            isinstance(record.get("fields"), dict), location, "semantic event is missing fields"
        )
    _require(
        expected_rank is None or record["rank"] == expected_rank,
        location,
        f"record rank {record['rank']} does not match filename rank {expected_rank}",
    )
    return record


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TraceValidationError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_rank_trace(
    path: Path, expected_rank: int | None = None, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    """Read one rank stream; arbitrary explicit filenames still require a single rank."""
    records: list[dict[str, Any]] = []
    previous_sequence = -1
    keys: set[tuple[Any, ...]] = set()
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                location = f"{path}:{line_number}"
                try:
                    record = validate_record(
                        json.loads(line, object_pairs_hook=_unique_object), location, expected_rank
                    )
                except (ValueError, RecursionError) as exc:
                    raise TraceValidationError(
                        f"{location}: invalid JSON or trace record: {exc}"
                    ) from exc
                if expected_rank is None:
                    expected_rank = record["rank"]
                _require(
                    record["sequence"] > previous_sequence,
                    location,
                    "sequence must be strictly increasing",
                )
                key = record_key(record)
                _require(key not in keys, location, "duplicate semantic event key")
                keys.add(key)
                previous_sequence = record["sequence"]
                records.append(record)
    except (OSError, UnicodeError) as exc:
        raise TraceValidationError(f"{path}: cannot read trace: {exc}") from exc
    _require(bool(records) or allow_empty, str(path), "trace contains no records")
    return records


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty rank streams, rejecting malformed names and duplicate ranks."""
    root = Path(path)
    if root.is_file():
        match = _RANK_FILE_RE.fullmatch(root.name)
        return read_rank_trace(root, int(match.group(1)) if match else None)
    _require(root.is_dir(), str(root), "trace path does not exist")
    files = sorted(root.glob("rank_*.jsonl"))
    _require(bool(files), str(root), "no rank_*.jsonl files found")
    records: list[dict[str, Any]] = []
    ranks_seen: set[int] = set()
    for trace_file in files:
        match = _RANK_FILE_RE.fullmatch(trace_file.name)
        _require(match is not None, str(trace_file), "expected filename rank_<digits>.jsonl")
        rank = int(match.group(1))
        _require(rank not in ranks_seen, str(trace_file), f"duplicate trace file for rank {rank}")
        ranks_seen.add(rank)
        records.extend(read_rank_trace(trace_file, rank))
    return records
