# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline comparison for rank-local determinism traces."""

from __future__ import annotations

import importlib.util
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

# Load the shared, standard-library-only contract without executing
# megatron.core.__init__ (which imports PyTorch and optional GPU extensions).
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "megatron/core/determinism/trace_schema.py"
_SCHEMA_SPEC = importlib.util.spec_from_file_location("_determinism_trace_schema", _SCHEMA_PATH)
_SCHEMA = importlib.util.module_from_spec(_SCHEMA_SPEC)
_SCHEMA_SPEC.loader.exec_module(_SCHEMA)
TRACE_SCHEMA_VERSION = _SCHEMA.TRACE_SCHEMA_VERSION
TraceValidationError = _SCHEMA.TraceValidationError
load_trace = _SCHEMA.load_trace
_record_key = _SCHEMA.record_key


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


def _mode_counts(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        record["tensor"].get("mode", "unknown") for record in records if record["event"] == "tensor"
    )
    return dict(sorted(counts.items()))


def _compare_rank(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    """Find the end of a rank's common execution prefix without aligning by name."""
    left_keys = {_record_key(record) for record in left}
    right_keys = {_record_key(record) for record in right}
    compared = 0
    divergence = None
    for left_record, right_record in zip_longest(left, right):
        left_key = _record_key(left_record) if left_record is not None else None
        right_key = _record_key(right_record) if right_record is not None else None
        positions = {
            "left_sequence": left_record["sequence"] if left_record is not None else None,
            "right_sequence": right_record["sequence"] if right_record is not None else None,
        }
        if left_key != right_key:
            left_missing = left_key is not None and left_key not in right_keys
            right_missing = right_key is not None and right_key not in left_keys
            if left_missing != right_missing:
                divergence = {
                    "kind": "missing_event",
                    "key": _key_json(left_key if left_missing else right_key),
                    "missing_from": "right" if left_missing else "left",
                }
            else:
                # Both sides may introduce different events at this boundary.
                # Report both candidates rather than impose an order between runs.
                divergence = {
                    "kind": "event_mismatch" if left_missing else "event_order_mismatch",
                    "left": _key_json(left_key) if left_key is not None else None,
                    "right": _key_json(right_key) if right_key is not None else None,
                }
            divergence.update(positions)
            break
        left_semantic = _semantic_record(left_record)
        right_semantic = _semantic_record(right_record)
        if left_semantic != right_semantic:
            divergence = {
                "kind": "content_mismatch",
                "key": _key_json(left_key),
                "differing_fields": _different_fields(left_semantic, right_semantic),
                "left": left_semantic,
                "right": right_semantic,
                **positions,
            }
            break
        compared += 1
    return {
        "compared_records_before_divergence": compared,
        "first_divergence": divergence,
        "match": divergence is None,
    }


def compare_traces(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    """Report the first execution divergence independently for every observed rank.

    Semantic keys identify events; sequence orders them within each rank only.
    Numeric sequence offsets between runs do not themselves imply divergence.
    There is deliberately no global first-divergence field or cross-rank order.
    """
    left_records = load_trace(left_path)
    right_records = load_trace(right_path)
    left_by_rank = defaultdict(list)
    right_by_rank = defaultdict(list)
    for record in left_records:
        left_by_rank[record["rank"]].append(record)
    for record in right_records:
        right_by_rank[record["rank"]].append(record)
    rank_results = {}
    for rank in sorted(left_by_rank.keys() | right_by_rank.keys()):
        left = sorted(left_by_rank.get(rank, []), key=lambda record: record["sequence"])
        right = sorted(right_by_rank.get(rank, []), key=lambda record: record["sequence"])
        rank_results[str(rank)] = _compare_rank(left, right)

    match = all(result["match"] for result in rank_results.values())
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
        "rank_results": rank_results,
        "left": {
            "path": str(left_path),
            "records": len(left_records),
            "ranks": sorted(left_by_rank),
            "tensor_modes": mode_counts,
        },
        "match": match,
        "match_strength": match_strength if match else "none",
        "right": {
            "path": str(right_path),
            "records": len(right_records),
            "ranks": sorted(right_by_rank),
            "tensor_modes": _mode_counts(right_records),
        },
        "schema_version": TRACE_SCHEMA_VERSION,
        "report_version": 2,
    }
