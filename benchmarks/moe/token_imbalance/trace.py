#!/usr/bin/env python3
"""B1 trace v1: a persisted, checksummed logical routing workload.

Design rules taken from the B1 contract:

  * The trace fixes the LOGICAL routing only (token ids, expert ids, gates). It never
    prescribes a physical plan, so the same trace can be executed by different dispatchers,
    placements or EP/ETP layouts.
  * ``schema_version``, dtypes, shapes, per-file checksums, expert-id range, top-k
    distinctness and gate finiteness/sum are all validated on load. A validation failure
    exits; nothing is silently truncated, reordered or padded.
  * Loading refuses arbitrary pickle: tensors are stored with ``torch.save`` and read back
    with ``weights_only=True``, which only reconstructs plain tensors and primitive
    containers.

Format (one directory per trace):

    manifest.json
    tensors/token_ids.pt        [T, K] int64
    tensors/gates.pt            [T, K] float32   (row sums to 1)
    tensors/source_rank.pt      [T]    int32     (which rank contributed the row)
    tensors/expert_ids.pt       [T, K] int64     (== token_ids for this harness)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
PRODUCER = "b1_benchmark.py/trace-v1"
TENSOR_FILES = ("token_ids", "gates", "source_rank", "expert_ids")


class TraceValidationError(Exception):
    """Raised when a trace fails any structural or semantic check."""


@dataclass
class TraceManifest:
    schema_version: int
    producer_commit: str
    seed: int
    logical_layer_id: int
    step_id: int
    microbatch_id: int
    logical_num_tokens: int
    num_experts: int
    topk: int
    hidden_size: int
    world_size: int
    original_partition: str
    workload_id: str
    workload_checksum: str
    tensors: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_trace(
    out_dir: Path,
    token_ids: torch.Tensor,
    gates: torch.Tensor,
    source_rank: torch.Tensor,
    *,
    num_experts: int,
    topk: int,
    hidden_size: int,
    world_size: int,
    seed: int,
    workload_id: str,
    workload_checksum: str,
    producer_commit: str,
    step_id: int = 0,
    logical_layer_id: int = 0,
    microbatch_id: int = 0,
    original_partition: str = "rank-major, contiguous per source rank",
) -> TraceManifest:
    """Persist one logical routing workload. Refuses to overwrite an existing trace dir."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise TraceValidationError(f"refusing to overwrite existing trace dir {out_dir}")
    (out_dir / "tensors").mkdir(parents=True, exist_ok=True)

    token_ids = token_ids.to(torch.int64).cpu()
    expert_ids = token_ids.clone()
    gates = gates.to(torch.float32).cpu()
    source_rank = source_rank.to(torch.int32).cpu()

    tensors = {
        "token_ids": token_ids,
        "gates": gates,
        "source_rank": source_rank,
        "expert_ids": expert_ids,
    }
    manifest_tensors: dict[str, Any] = {}
    for name, t in tensors.items():
        p = out_dir / "tensors" / f"{name}.pt"
        torch.save(t, p)
        manifest_tensors[name] = {
            "file": f"tensors/{name}.pt",
            "sha256": _sha256(p),
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
        }

    man = TraceManifest(
        schema_version=SCHEMA_VERSION,
        producer_commit=producer_commit,
        seed=int(seed),
        logical_layer_id=int(logical_layer_id),
        step_id=int(step_id),
        microbatch_id=int(microbatch_id),
        logical_num_tokens=int(token_ids.shape[0]),
        num_experts=int(num_experts),
        topk=int(topk),
        hidden_size=int(hidden_size),
        world_size=int(world_size),
        original_partition=original_partition,
        workload_id=workload_id,
        workload_checksum=workload_checksum,
        tensors=manifest_tensors,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(man.as_json(), indent=2), encoding="utf-8"
    )
    return man


def load_trace(trace_dir: Path) -> tuple[TraceManifest, dict[str, torch.Tensor]]:
    """Load and fully validate a trace. Raises TraceValidationError on any problem."""
    trace_dir = Path(trace_dir)
    manifest_path = trace_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TraceValidationError(f"missing manifest.json in {trace_dir}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TraceValidationError(f"manifest.json is not valid JSON: {exc}") from exc

    if raw.get("schema_version") != SCHEMA_VERSION:
        raise TraceValidationError(
            f"unsupported schema_version {raw.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )
    for key in ("seed", "logical_num_tokens", "num_experts", "topk", "world_size",
                "workload_checksum", "tensors"):
        if key not in raw:
            raise TraceValidationError(f"manifest is missing required field {key!r}")

    man = TraceManifest(**{k: v for k, v in raw.items() if k in TraceManifest.__annotations__})

    tensors: dict[str, torch.Tensor] = {}
    for name in TENSOR_FILES:
        meta = raw["tensors"].get(name)
        if meta is None:
            raise TraceValidationError(f"manifest does not list tensor {name!r}")
        p = trace_dir / meta["file"]
        if not p.is_file():
            raise TraceValidationError(f"missing tensor file {p}")
        actual = _sha256(p)
        if actual != meta["sha256"]:
            raise TraceValidationError(
                f"checksum mismatch for {name}: manifest {meta['sha256'][:16]} != file "
                f"{actual[:16]}"
            )
        try:
            # weights_only=True: reconstruct plain tensors only, never arbitrary objects
            t = torch.load(p, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise TraceValidationError(f"cannot load {name} safely: {exc}") from exc
        if not isinstance(t, torch.Tensor):
            raise TraceValidationError(f"{name} did not load as a plain tensor")
        if list(t.shape) != list(meta["shape"]):
            raise TraceValidationError(
                f"{name} shape {list(t.shape)} != manifest {meta['shape']}"
            )
        if str(t.dtype).replace("torch.", "") != meta["dtype"]:
            raise TraceValidationError(f"{name} dtype {t.dtype} != manifest {meta['dtype']}")
        tensors[name] = t

    _validate_semantics(man, tensors)
    return man, tensors


def _validate_semantics(man: TraceManifest, tensors: dict[str, torch.Tensor]) -> None:
    """Semantic checks that a checksum cannot catch (a trace can be intact but nonsense)."""
    ids = tensors["expert_ids"]
    gates = tensors["gates"]
    src = tensors["source_rank"]
    T, K = ids.shape

    if man.topk != K:
        raise TraceValidationError(f"manifest topk {man.topk} != expert_ids width {K}")
    if man.logical_num_tokens != T:
        raise TraceValidationError(
            f"manifest logical_num_tokens {man.logical_num_tokens} != expert_ids rows {T}"
        )
    if gates.shape != ids.shape:
        raise TraceValidationError(f"gates shape {tuple(gates.shape)} != expert_ids {tuple(ids.shape)}")
    if src.shape != (T,):
        raise TraceValidationError(f"source_rank shape {tuple(src.shape)} != ({T},)")

    if T == 0:
        raise TraceValidationError("trace has zero logical tokens")

    lo, hi = int(ids.min()), int(ids.max())
    if lo < 0 or hi >= man.num_experts:
        raise TraceValidationError(
            f"expert id out of range [{lo}, {hi}] for num_experts {man.num_experts}"
        )
    if man.num_experts < man.topk:
        raise TraceValidationError(
            f"num_experts {man.num_experts} < topk {man.topk}: distinct experts impossible"
        )

    # every row must carry exactly K DISTINCT experts
    sorted_ids, _ = torch.sort(ids, dim=1)
    dup = (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any()
    if bool(dup):
        bad = int((sorted_ids[:, 1:] == sorted_ids[:, :-1]).any(dim=1).sum())
        raise TraceValidationError(
            f"{bad} row(s) repeat an expert in the top-k slots; the contract requires distinct"
        )

    if not bool(torch.isfinite(gates).all()):
        raise TraceValidationError("gates contain NaN or Inf")
    if bool((gates < 0).any()):
        raise TraceValidationError("gates contain negative values")
    row_sums = gates.sum(dim=1)
    if not bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4, rtol=1e-4)):
        worst = float((row_sums - 1.0).abs().max())
        raise TraceValidationError(f"gate rows do not sum to 1 (max deviation {worst:.6g})")

    if int(src.min()) < 0 or int(src.max()) >= man.world_size:
        raise TraceValidationError(
            f"source_rank out of range for world_size {man.world_size}"
        )
    counts = torch.bincount(src, minlength=man.world_size)
    if int(counts.sum()) != T:
        raise TraceValidationError("source_rank does not account for every token")

    if not torch.equal(tensors["token_ids"], ids):
        raise TraceValidationError("token_ids and expert_ids disagree")


def trace_digest(trace_dir: Path) -> str:
    """A digest over the manifest and tensor checksums, for use as a run identifier."""
    raw = json.loads((Path(trace_dir) / "manifest.json").read_text(encoding="utf-8"))
    h = hashlib.sha256()
    h.update(json.dumps(raw, sort_keys=True).encode())
    for name in TENSOR_FILES:
        h.update(raw["tensors"][name]["sha256"].encode())
    return h.hexdigest()[:16]
