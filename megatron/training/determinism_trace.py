# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Rank-local tensor tracing for cross-run determinism investigations.

The tracer deliberately performs no distributed collectives. Each rank writes
an append-only JSONL stream that can be compared offline. Tensor capture is
progressive: metadata-only records are cheap, while summary, sampled, and full
captures trade increasing cost for stronger evidence.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import torch

TRACE_SCHEMA_VERSION = 1


class DigestMode(str, Enum):
    """Tensor evidence strength recorded by :class:`RankLocalTrace`."""

    METADATA = "metadata"
    SUMMARY = "summary"
    SAMPLED = "sampled"
    FULL = "full"


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for one rank-local determinism trace.

    Args:
        output_dir: Directory containing one ``rank_XXXXXX.jsonl`` file per rank.
        rank: Global rank written into every record and the output filename.
        mode: Default tensor capture mode.
        sample_count: Maximum number of evenly spaced elements in sampled mode.
        flush_every: Flush after this many buffered records. Zero means explicit
            flushing only.
        append: Append to an existing rank file. If false, fail when the file
            already exists instead of overwriting it.
        rank_spec: Comma-separated ranks or inclusive ranges, or ``all``.
        iteration_spec: Comma-separated iterations or inclusive ranges, or
            ``all``.
    """

    output_dir: str | os.PathLike[str]
    rank: int
    mode: DigestMode | str = DigestMode.METADATA
    sample_count: int = 256
    flush_every: int = 0
    append: bool = False
    rank_spec: str = "all"
    iteration_spec: str = "all"

    def __post_init__(self) -> None:
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.flush_every < 0:
            raise ValueError("flush_every must be non-negative")
        object.__setattr__(self, "mode", DigestMode(self.mode))
        _parse_range_spec(self.rank_spec, "rank_spec")
        _parse_range_spec(self.iteration_spec, "iteration_spec")


@dataclass
class _PendingTensor:
    metadata: dict[str, Any]
    mode: DigestMode
    payload: torch.Tensor | None = None
    sample_indices: list[int] | None = None


def _parse_range_spec(spec: str, label: str) -> tuple[tuple[int, int], ...] | None:
    normalized = spec.strip().lower()
    if normalized in {"", "all", "*"}:
        return None

    ranges: list[tuple[int, int]] = []
    for item in normalized.split(","):
        item = item.strip()
        if not item:
            raise ValueError(f"{label} contains an empty item")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(item)
        if start < 0 or end < start:
            raise ValueError(f"invalid {label} range: {item!r}")
        ranges.append((start, end))
    return tuple(ranges)


def _selected(value: int, ranges: tuple[tuple[int, int], ...] | None) -> bool:
    return ranges is None or any(start <= value <= end for start, end in ranges)


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON values") from exc


def _validate_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError(f"{label} must be a non-empty string of at most 512 characters")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device_type": tensor.device.type,
        "dtype": _dtype_name(tensor),
        "layout": str(tensor.layout).removeprefix("torch."),
        "numel": tensor.numel(),
        "requires_grad": tensor.requires_grad,
        "shape": list(tensor.shape),
    }
    if tensor.layout == torch.strided:
        metadata["stride"] = list(tensor.stride())
    return metadata


def _sample_indices(numel: int, sample_count: int) -> list[int]:
    count = min(numel, sample_count)
    if count == 0:
        return []
    if count == 1:
        return [0]
    return [(index * (numel - 1)) // (count - 1) for index in range(count)]


def _capture_tensor(tensor: torch.Tensor, mode: DigestMode, sample_count: int) -> _PendingTensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("record_tensor requires a torch.Tensor")

    metadata = _tensor_metadata(tensor)
    if mode == DigestMode.METADATA:
        return _PendingTensor(metadata=metadata, mode=mode)
    if tensor.layout != torch.strided:
        raise ValueError(f"{mode.value} capture requires a strided tensor")
    if tensor.device.type == "meta":
        raise ValueError(f"{mode.value} capture does not support meta tensors")
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("value-bearing determinism traces are not CUDA-graph-capture safe")

    detached = tensor.detach()
    if mode == DigestMode.FULL:
        return _PendingTensor(metadata=metadata, mode=mode, payload=detached.contiguous().clone())
    if mode == DigestMode.SAMPLED:
        indices = _sample_indices(detached.numel(), sample_count)
        if indices:
            index_tensor = torch.tensor(indices, dtype=torch.int64, device=detached.device)
            payload = detached.reshape(-1).index_select(0, index_tensor).clone()
        else:
            payload = detached.new_empty((0,))
        return _PendingTensor(
            metadata=metadata,
            mode=mode,
            payload=payload,
            sample_indices=indices,
        )

    if detached.numel() == 0:
        payload = torch.empty((0,), dtype=torch.float64, device=detached.device)
    else:
        values = detached.abs() if detached.is_complex() else detached
        values = values.to(torch.float64)
        finite = torch.isfinite(values).all().to(torch.float64)
        payload = torch.stack(
            (
                finite,
                values.min(),
                values.max(),
                values.sum(),
                torch.linalg.vector_norm(values),
            )
        )
    return _PendingTensor(metadata=metadata, mode=mode, payload=payload)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu_tensor = tensor.detach().contiguous().cpu()
    return cpu_tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _resolve_tensor(pending: _PendingTensor) -> dict[str, Any]:
    result = dict(pending.metadata)
    result["mode"] = pending.mode.value
    result["value_observed"] = pending.mode != DigestMode.METADATA
    result["exact_value_certificate"] = pending.mode == DigestMode.FULL

    if pending.mode == DigestMode.METADATA:
        return result
    assert pending.payload is not None

    if pending.mode == DigestMode.SUMMARY:
        if pending.metadata["numel"] == 0:
            result["summary"] = {
                "all_finite": True,
                "l2_norm": 0.0,
                "maximum": None,
                "mean": None,
                "minimum": None,
            }
            return result
        finite, minimum, maximum, total, l2_norm = pending.payload.cpu().tolist()
        count = pending.metadata["numel"]
        result["summary"] = {
            "all_finite": bool(finite),
            "l2_norm": _finite_or_none(float(l2_norm)),
            "maximum": _finite_or_none(float(maximum)),
            "mean": _finite_or_none(float(total) / count),
            "minimum": _finite_or_none(float(minimum)),
        }
        return result

    raw = _tensor_bytes(pending.payload)
    hash_header: dict[str, Any] = {
        "dtype": pending.metadata["dtype"],
        "mode": pending.mode.value,
        "schema_version": TRACE_SCHEMA_VERSION,
        "shape": pending.metadata["shape"],
    }
    if pending.mode == DigestMode.SAMPLED:
        hash_header["sample_indices"] = pending.sample_indices
    hasher = hashlib.sha256()
    hasher.update(json.dumps(hash_header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(raw)
    result["sha256"] = hasher.hexdigest()
    result["captured_numel"] = pending.payload.numel()
    if pending.mode == DigestMode.SAMPLED:
        result["sample_indices"] = pending.sample_indices
    if pending.payload.is_floating_point() or pending.payload.is_complex():
        result["all_finite"] = bool(torch.isfinite(pending.payload).all().item())
    else:
        result["all_finite"] = True
    return result


class RankLocalTrace:
    """Append-only JSONL recorder for one distributed rank.

    Tensor payloads are snapshotted when ``record_tensor`` is called and are
    materialized on the host only when ``flush`` runs. The class never invokes
    a distributed collective.
    """

    def __init__(self, config: TraceConfig) -> None:
        self.config = config
        self._rank_ranges = _parse_range_spec(config.rank_spec, "rank_spec")
        self._iteration_ranges = _parse_range_spec(config.iteration_spec, "iteration_spec")
        self._enabled = _selected(config.rank, self._rank_ranges)
        self._closed = False
        self._lock = threading.RLock()
        self._pending: list[tuple[dict[str, Any], _PendingTensor | None]] = []
        self._occurrences: dict[tuple[Any, ...], int] = {}
        self._sequence = 0

        self.output_path = Path(config.output_dir) / f"rank_{config.rank:06d}.jsonl"
        if self._enabled:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            if config.append:
                if self.output_path.exists():
                    self._resume_append_state()
                self.output_path.touch(exist_ok=True)
            else:
                self.output_path.touch(exist_ok=False)

    @property
    def enabled(self) -> bool:
        """Whether this rank was selected for tracing."""
        return self._enabled and not self._closed

    def record_event(
        self,
        name: str,
        *,
        iteration: int | None,
        microbatch: int | None = None,
        phase: str = "unspecified",
        fields: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a JSON-only semantic event."""
        with self._lock:
            record = self._new_record("event", name, iteration, microbatch, phase)
            if record is None:
                return
            record["fields"] = _json_copy(dict(fields or {}), "fields")
            self._enqueue(record, None)

    def record_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        iteration: int | None,
        microbatch: int | None = None,
        phase: str = "unspecified",
        mode: DigestMode | str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Snapshot a tensor boundary for deferred rank-local serialization."""
        with self._lock:
            record = self._new_record("tensor", name, iteration, microbatch, phase)
            if record is None:
                return
            record["metadata"] = _json_copy(dict(metadata or {}), "metadata")
            capture_mode = self.config.mode if mode is None else DigestMode(mode)
            pending = _capture_tensor(tensor, capture_mode, self.config.sample_count)
            self._enqueue(record, pending)

    def record_scalar(
        self,
        name: str,
        value: bool | int | float | torch.Tensor | None,
        *,
        iteration: int | None,
        microbatch: int | None = None,
        phase: str = "unspecified",
    ) -> None:
        """Record a scalar with exact value evidence.

        Scalar tensors use full mode regardless of the configured default; the
        payload is only one element and is therefore inexpensive to preserve.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("record_scalar requires a one-element tensor")
            self.record_tensor(
                name,
                value,
                iteration=iteration,
                microbatch=microbatch,
                phase=phase,
                mode=DigestMode.FULL,
                metadata={"semantic_type": "scalar"},
            )
            return

        scalar_fields: dict[str, Any] = {"value": value}
        if isinstance(value, float) and not math.isfinite(value):
            scalar_fields = {
                "finite": False,
                "nonfinite": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf"),
                "value": None,
            }
        self.record_event(
            name,
            iteration=iteration,
            microbatch=microbatch,
            phase=phase,
            fields=scalar_fields,
        )

    def flush(self) -> None:
        """Resolve buffered tensor evidence and append complete JSONL records."""
        with self._lock:
            if not self._pending:
                return
            serialized: list[str] = []
            for record, pending in self._pending:
                resolved = dict(record)
                if pending is not None:
                    resolved["tensor"] = _resolve_tensor(pending)
                serialized.append(
                    json.dumps(resolved, allow_nan=False, separators=(",", ":"), sort_keys=True)
                )
            with self.output_path.open("a", encoding="utf-8") as output:
                output.write("\n".join(serialized) + "\n")
                output.flush()
            self._pending.clear()

    def close(self) -> None:
        """Flush pending records and reject subsequent writes."""
        with self._lock:
            if self._closed:
                return
            self.flush()
            self._closed = True

    def __enter__(self) -> "RankLocalTrace":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _new_record(
        self,
        event: str,
        name: str,
        iteration: int | None,
        microbatch: int | None,
        phase: str,
    ) -> dict[str, Any] | None:
        if self._closed:
            raise RuntimeError("determinism trace is closed")
        if not self._enabled:
            return None
        if iteration is not None:
            if iteration < 0:
                raise ValueError("iteration must be non-negative")
            if not _selected(iteration, self._iteration_ranges):
                return None
        if microbatch is not None and microbatch < 0:
            raise ValueError("microbatch must be non-negative")
        name = _validate_name(name, "name")
        phase = _validate_name(phase, "phase")

        occurrence_key = (event, name, iteration, microbatch, phase)
        occurrence = self._occurrences.get(occurrence_key, 0)
        self._occurrences[occurrence_key] = occurrence + 1
        record = {
            "event": event,
            "iteration": iteration,
            "microbatch": microbatch,
            "name": name,
            "occurrence": occurrence,
            "phase": phase,
            "rank": self.config.rank,
            "schema_version": TRACE_SCHEMA_VERSION,
            "sequence": self._sequence,
        }
        self._sequence += 1
        return record

    def _enqueue(self, record: dict[str, Any], pending: _PendingTensor | None) -> None:
        self._pending.append((record, pending))
        if self.config.flush_every and len(self._pending) >= self.config.flush_every:
            self.flush()

    def _resume_append_state(self) -> None:
        previous_sequence = -1
        with self.output_path.open(encoding="utf-8") as existing:
            for line_number, line in enumerate(existing, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{self.output_path}:{line_number}: invalid existing trace JSON"
                    ) from exc
                required = {
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
                if not isinstance(record, dict) or not required.issubset(record):
                    raise ValueError(
                        f"{self.output_path}:{line_number}: invalid existing trace record"
                    )
                sequence = record["sequence"]
                if (
                    record["rank"] != self.config.rank
                    or record["schema_version"] != TRACE_SCHEMA_VERSION
                    or not isinstance(sequence, int)
                    or sequence <= previous_sequence
                ):
                    raise ValueError(
                        f"{self.output_path}:{line_number}: incompatible existing trace record"
                    )
                occurrence = record["occurrence"]
                if not isinstance(occurrence, int) or occurrence < 0:
                    raise ValueError(
                        f"{self.output_path}:{line_number}: invalid existing occurrence"
                    )
                occurrence_key = (
                    record["event"],
                    record["name"],
                    record["iteration"],
                    record["microbatch"],
                    record["phase"],
                )
                self._occurrences[occurrence_key] = max(
                    self._occurrences.get(occurrence_key, 0), occurrence + 1
                )
                previous_sequence = sequence
        self._sequence = previous_sequence + 1


_ACTIVE_TRACE: RankLocalTrace | None = None
_ACTIVE_TRACE_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def initialize_determinism_trace(config: TraceConfig) -> RankLocalTrace | None:
    """Initialize the process-local tracer, returning None on unselected ranks."""
    global _ACTIVE_TRACE, _ATEXIT_REGISTERED
    with _ACTIVE_TRACE_LOCK:
        if _ACTIVE_TRACE is not None:
            raise RuntimeError("determinism trace is already initialized")
        trace = RankLocalTrace(config)
        if not trace.enabled:
            trace.close()
            return None
        _ACTIVE_TRACE = trace
        if not _ATEXIT_REGISTERED:
            atexit.register(close_determinism_trace)
            _ATEXIT_REGISTERED = True
        return trace


def get_determinism_trace() -> RankLocalTrace | None:
    """Return the active process-local trace, if one was initialized."""
    return _ACTIVE_TRACE


def close_determinism_trace() -> None:
    """Flush and clear the active process-local trace."""
    global _ACTIVE_TRACE
    with _ACTIVE_TRACE_LOCK:
        trace = _ACTIVE_TRACE
        _ACTIVE_TRACE = None
    if trace is not None:
        trace.close()
