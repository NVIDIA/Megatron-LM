# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Capture and compare declared training state byte for byte.

Snapshots contain a JSON index and raw logical tensor/scalar bytes. Comparison
streams the bytes, so it needs neither GPU memory nor pickle deserialization.
Equality covers the adapter's captured state, not an arbitrary model or recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import socket
import struct
import uuid
from collections.abc import Mapping
from itertools import zip_longest
from pathlib import Path

COMPONENTS = ("model", "gradients", "optimizer", "precision", "rng", "scheduler", "dataloader")
FORMAT = "megatron_training_state_v1"
PROCESS_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"


class UnverifiedState(ValueError):
    """The declared state cannot support a complete comparison."""


def _write_tree(value, path: list, stream, index: list) -> None:
    """Write one value at a time; retain only the small index in memory."""
    import numpy as np
    import torch

    item: dict = {"path": path}
    index.append(item)
    if isinstance(value, Mapping):
        if any(type(key) not in (str, int) for key in value):
            raise UnverifiedState(f"{path}: mapping keys must be strings or integers")
        keys = sorted(value, key=lambda key: (type(key).__name__, key))
        item.update(kind="mapping", keys=keys)
        for key in keys:
            _write_tree(value[key], [*path, key], stream, index)
        return
    if type(value) in (list, tuple):
        item.update(kind=type(value).__name__, length=len(value))
        for position, child in enumerate(value):
            _write_tree(child, [*path, position], stream, index)
        return
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        if value.layout != torch.strided or value.is_quantized or value.device.type == "meta":
            raise UnverifiedState(f"{path}: requires a dense, materialized tensor adapter")
        tensor = value.detach().resolve_conj().resolve_neg().contiguous()
        # View as bytes before the device transfer: no floating-point conversion.
        data = tensor.reshape(-1).view(torch.uint8).cpu().numpy()
        item.update(kind="tensor", dtype=str(value.dtype), shape=list(value.shape))
        item.update(elements=value.numel(), itemsize=value.element_size())
    elif isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.fields:
            raise UnverifiedState(f"{path}: object/structured NumPy state needs an adapter")
        data = np.ascontiguousarray(array).tobytes()
        item.update(kind="numpy", dtype=array.dtype.str, shape=list(array.shape))
        item.update(elements=array.size, itemsize=array.itemsize)
    elif type(value) in (bool, int, str, type(None)):
        item.update(kind=type(value).__name__, value=value)
        return
    elif type(value) is float:
        item.update(kind="float")
        data = struct.pack("!d", value)
    elif type(value) is complex:
        item.update(kind="complex")
        data = struct.pack("!dd", value.real, value.imag)
    elif type(value) is bytes:
        item.update(kind="bytes")
        data = value
    elif type(value) is io.BytesIO:
        item.update(kind="bytes_io", position=value.tell())
        data = value.getbuffer()
    else:
        raise UnverifiedState(f"{path}: unsupported state type {type(value).__qualname__}")
    item.update(offset=stream.tell(), nbytes=len(data))
    stream.write(data)


def _validate_index(index: list, nbytes: int) -> dict:
    """Require every component and nonempty model/gradient tensor comparisons."""
    counts = {name: {"tensors": 0, "elements": 0, "values": 0} for name in COMPONENTS}
    roots = set()
    paths = set()
    expected_paths = {json.dumps([name]) for name in COMPONENTS}
    offset = 0
    for item in index:
        path = item["path"]
        if not path or path[0] not in counts:
            raise UnverifiedState("Unknown or missing state component")
        identity = json.dumps(path)
        if identity in paths:
            raise UnverifiedState(f"Duplicate state path: {path}")
        paths.add(identity)
        if identity not in expected_paths:
            raise UnverifiedState(f"State path has no declared parent: {path}")
        kind = item["kind"]
        if kind == "mapping":
            children = item["keys"]
            if any(type(key) not in (str, int) for key in children):
                raise UnverifiedState(f"Unsupported mapping key at {path}")
            if len(set((type(key), key) for key in children)) != len(children):
                raise UnverifiedState(f"Duplicate mapping keys at {path}")
        elif kind in ("list", "tuple"):
            if type(item["length"]) is not int or item["length"] < 0:
                raise UnverifiedState(f"Invalid sequence length at {path}")
            children = range(item["length"])
        else:
            children = ()
        expected_paths.update(json.dumps([*path, child]) for child in children)
        if kind not in (
            "mapping",
            "list",
            "tuple",
            "tensor",
            "numpy",
            "bool",
            "int",
            "str",
            "NoneType",
            "float",
            "complex",
            "bytes",
            "bytes_io",
        ):
            raise UnverifiedState(f"Unsupported indexed state kind at {path}")
        binary_kind = kind in ("tensor", "numpy", "float", "complex", "bytes", "bytes_io")
        if ("offset" in item) != binary_kind or ("nbytes" in item) != binary_kind:
            raise UnverifiedState(f"Missing or unexpected byte range at {path}")
        if kind in ("bool", "int", "str", "NoneType"):
            expected_type = {"bool": bool, "int": int, "str": str, "NoneType": type(None)}[kind]
            if type(item["value"]) is not expected_type:
                raise UnverifiedState(f"Invalid scalar type at {path}")
        if kind in ("float", "complex") and item["nbytes"] != (8 if kind == "float" else 16):
            raise UnverifiedState(f"Invalid scalar byte count at {path}")
        if len(path) == 1 and item["kind"] != "NoneType":
            roots.add(path[0])
        component = counts[path[0]]
        if "offset" in item:
            if item["offset"] != offset or type(item["nbytes"]) is not int or item["nbytes"] < 0:
                raise UnverifiedState(f"Invalid byte range at {path}")
            offset += item["nbytes"]
        if item["kind"] in ("tensor", "numpy"):
            elements = math.prod(item["shape"])
            if (
                any(type(size) is not int or size < 0 for size in item["shape"])
                or elements != item["elements"]
                or type(item["itemsize"]) is not int
                or item["itemsize"] <= 0
                or elements * item["itemsize"] != item["nbytes"]
            ):
                raise UnverifiedState(f"Invalid array extent at {path}")
            component["tensors"] += int(elements > 0)
            component["elements"] += elements
        elif item["kind"] not in ("mapping", "list", "tuple", "NoneType"):
            if kind in ("bytes", "bytes_io"):
                component["values"] += int(item["nbytes"] > 0)
            elif kind == "str":
                component["values"] += int(bool(item["value"]))
            else:
                component["values"] += 1
    if offset != nbytes:
        raise UnverifiedState("Snapshot binary length does not match its index")
    if paths != expected_paths:
        raise UnverifiedState("Snapshot omits declared child state")
    if roots != set(COMPONENTS):
        raise UnverifiedState("Every required state component must be present and non-null")
    for name, count in counts.items():
        if not count["elements"] and not count["values"]:
            raise UnverifiedState(f"Empty state component: {name}")
    for name in ("model", "gradients"):
        if not counts[name]["tensors"]:
            raise UnverifiedState(f"{name} requires nonempty tensors")
    return counts


def snapshot_path(directory: Path, step: int, rank: int) -> Path:
    """Return the canonical index path for a completed training step and rank."""
    return Path(directory) / f"step-{step:08d}" / f"rank-{rank:06d}.json"


def write_snapshot(
    directory: Path,
    state: Mapping,
    *,
    step: int,
    rank: int,
    world_size: int,
    run_id: str,
    provenance: dict,
    resume_from: dict | None = None,
) -> Path:
    """Record complete state after an optimizer/scheduler step, before zeroing.

    The adapter must finish asynchronous work, enumerate all model chunks and
    provide every component in ``COMPONENTS``. Represent disabled features
    explicitly (for example ``precision={"mode": "fp32"}``), not as empty state.
    Existing snapshots are never overwritten. Unsupported values leave a
    ``not_verified`` index and raise; they are never silently omitted.
    """
    if (
        any(type(value) is not int for value in (step, rank, world_size))
        or step < 1
        or world_size < 1
        or not 0 <= rank < world_size
        or not run_id
    ):
        raise ValueError("Invalid step/rank/world size/run ID")
    path = snapshot_path(directory, step, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    metadata: dict = {
        "format": FORMAT,
        "status": "not_verified",
        "step": step,
        "rank": rank,
        "world_size": world_size,
        "run_id": run_id,
        "process_id": PROCESS_ID,
        "provenance": provenance,
        "resume_from": resume_from,
        "boundary": "after_optimizer_and_scheduler_before_zero_grad",
        "index": [],
    }
    # Validate JSON provenance before creating the binary file.
    json.dumps(metadata, allow_nan=False)
    stream = path.with_suffix(".bin").open("xb")
    try:
        with stream:
            if set(state) != set(COMPONENTS):
                raise UnverifiedState(f"Expected exactly these components: {COMPONENTS}")
            for component in COMPONENTS:
                _write_tree(state[component], [component], stream, metadata["index"])
            metadata["counts"] = _validate_index(metadata["index"], stream.tell())
        metadata["status"] = "complete"
    except UnverifiedState as error:
        metadata["reason"] = str(error)
        raise
    finally:
        # The final index is the completion marker; incomplete binary writes
        # cannot become a passing snapshot when a job stops halfway through.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    return path


def _read_snapshot(path: Path, step: int, rank: int, world_size: int) -> dict:
    metadata = json.loads(path.read_text())
    if (
        metadata["format"] != FORMAT
        or metadata["status"] != "complete"
        or any(type(metadata[key]) is not int for key in ("step", "rank", "world_size"))
        or (metadata["step"], metadata["rank"], metadata["world_size"]) != (step, rank, world_size)
        or metadata["boundary"] != "after_optimizer_and_scheduler_before_zero_grad"
        or not metadata["run_id"]
        or not metadata["process_id"]
    ):
        raise UnverifiedState(f"Incomplete or incompatible snapshot: {path}")
    counts = _validate_index(metadata["index"], path.with_suffix(".bin").stat().st_size)
    if counts != metadata["counts"]:
        raise UnverifiedState(f"Incorrect component counts: {path}")
    return metadata


def complete_rank(
    directory: Path, *, rank: int, world_size: int, run_id: str, steps: list[int], provenance: dict
) -> None:
    """Publish completion after successful training and a final provenance check.

    The adapter must check that its source/runtime provenance stayed unchanged.
    Every listed snapshot must be complete and belong to this same process.
    """
    if not steps or len(set(steps)) != len(steps):
        raise UnverifiedState("Completion requires a nonempty, unique step list")
    for step in steps:
        record = _read_snapshot(snapshot_path(directory, step, rank), step, rank, world_size)
        if (
            record["process_id"] != PROCESS_ID
            or record["run_id"] != run_id
            or record["provenance"] != provenance
        ):
            raise UnverifiedState("Rank completion does not match its snapshots")
    record = {
        "format": FORMAT,
        "status": "complete",
        "rank": rank,
        "world_size": world_size,
        "run_id": run_id,
        "process_id": PROCESS_ID,
        "steps": sorted(steps),
        "provenance": provenance,
    }
    path = Path(directory) / f"complete-rank-{rank:06d}.json"
    with path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)


def _read_completion(directory: Path, rank: int, world_size: int, steps: list[int]) -> dict:
    path = Path(directory) / f"complete-rank-{rank:06d}.json"
    record = json.loads(path.read_text())
    if (
        record["format"] != FORMAT
        or record["status"] != "complete"
        or record["rank"] != rank
        or record["world_size"] != world_size
        or not set(steps).issubset(record["steps"])
    ):
        raise UnverifiedState(f"Missing or incompatible rank completion: {path}")
    return record


def _first_difference(left: Path, right: Path, a: dict, b: dict) -> dict | None:
    with left.with_suffix(".bin").open("rb") as x, right.with_suffix(".bin").open("rb") as y:
        for item, other in zip_longest(a["index"], b["index"]):
            if item != other:
                return {
                    "path": (item or other)["path"],
                    "reason": "structure_dtype_shape_or_scalar",
                }
            remaining = item.get("nbytes", 0)
            offset = 0
            while remaining:
                size = min(remaining, 1024 * 1024)
                xb, yb = x.read(size), y.read(size)
                if len(xb) != size or len(yb) != size:
                    raise UnverifiedState("Snapshot truncated during comparison")
                if xb != yb:
                    byte = offset + next(i for i, (v, w) in enumerate(zip(xb, yb)) if v != w)
                    result = {"path": item["path"], "reason": "bytes", "byte_offset": byte}
                    if "itemsize" in item:
                        result["flat_element_index"] = byte // item["itemsize"]
                    return result
                remaining -= size
                offset += size
    return None


def checkpoint_path(directory: Path, step: int, rank: int) -> Path:
    """Return the pilot's checkpoint path; rank files are separate."""
    return Path(directory) / "checkpoints" / f"step-{step:08d}" / f"rank-{rank:06d}.pt"


def file_sha256(path: Path) -> str:
    """Fingerprint checkpoint identity, never as a replacement for state comparison."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_checkpoint(directory: Path, *, step: int, rank: int, run_id: str) -> dict:
    """Record the actual checkpoint file that a subsequent fresh process loads."""
    path = checkpoint_path(directory, step, rank)
    record = {"run_id": run_id, "step": step, "rank": rank, "checkpoint_sha256": file_sha256(path)}
    with path.with_suffix(".json").open("x") as stream:
        json.dump(record, stream, indent=2)
    return record


def compare_runs(
    reference: Path, candidate: Path, *, steps: list[int], world_size: int, comparison: str
) -> dict:
    """Compare every declared rank/step and report the first observed divergence.

    ``comparison`` is ``fresh`` or ``resume``. For resume, the candidate must
    identify a checkpoint from the reference run before the compared steps.
    Incompatible/missing evidence is unverified. Dirty-source observations are
    retained for debugging but cannot pass the gate. Process identities and
    provenance are evidence supplied by the capture protocol, not attestation.
    """
    result: dict = {"kind": FORMAT, "status": "not_verified", "comparison": comparison}
    if not steps or any(type(step) is not int or step < 1 for step in steps):
        return result | {"reason": "A nonempty list of positive steps is required"}
    if (
        len(set(steps)) != len(steps)
        or type(world_size) is not int
        or world_size < 1
        or comparison not in ("fresh", "resume")
    ):
        return result | {"reason": "Invalid comparison/step/rank denominator"}
    result.update(
        steps=sorted(steps), world_size=world_size, required_snapshots=len(steps) * world_size
    )
    try:
        pairs = []
        identities: list[set[str]] = [set(), set()]
        run_ids: list[set[str]] = [set(), set()]
        provenance_by_rank: dict[int, dict] = {}
        process_by_rank: dict[tuple[int, int], str] = {}
        checkpoint_records: dict[tuple[int, int], dict] = {}
        resume_by_rank: dict[int, dict] = {}
        resume_step = None
        global_provenance = None
        dirty = False
        completions = [
            [_read_completion(root, rank, world_size, steps) for rank in range(world_size)]
            for root in (reference, candidate)
        ]
        for step in sorted(steps):
            for rank in range(world_size):
                paths = [snapshot_path(root, step, rank) for root in (reference, candidate)]
                records = [_read_snapshot(path, step, rank, world_size) for path in paths]
                a, b = records
                for i, record in enumerate(records):
                    completion = completions[i][rank]
                    for key in ("process_id", "run_id", "provenance"):
                        if record[key] != completion[key]:
                            raise UnverifiedState(f"Snapshot differs from rank completion: {key}")
                    identities[i].add(record["process_id"])
                    run_ids[i].add(record["run_id"])
                    process_key = (i, rank)
                    if (
                        process_key in process_by_rank
                        and process_by_rank[process_key] != record["process_id"]
                    ):
                        raise UnverifiedState(f"Process changed within run/rank {process_key}")
                    process_by_rank[process_key] = record["process_id"]
                if a["provenance"] != b["provenance"]:
                    raise UnverifiedState(f"Provenance differs at step {step}, rank {rank}")
                provenance = a["provenance"]
                required = ("source_revision", "source_dirty", "software", "hardware", "recipe")
                if any(key not in provenance for key in required):
                    raise UnverifiedState("Missing source/software/hardware/recipe provenance")
                if not all(provenance[key] for key in required if key != "source_dirty"):
                    raise UnverifiedState("Empty provenance field")
                if type(provenance["source_dirty"]) is not bool:
                    raise UnverifiedState("Missing source cleanliness evidence")
                dirty |= provenance["source_dirty"]
                shared = {key: provenance[key] for key in required if key != "hardware"}
                if global_provenance is not None and global_provenance != shared:
                    raise UnverifiedState("Source/software/recipe differs across ranks or steps")
                global_provenance = shared
                if rank in provenance_by_rank and provenance_by_rank[rank] != provenance:
                    raise UnverifiedState(f"Provenance changed within rank {rank}")
                provenance_by_rank[rank] = provenance
                if a["resume_from"] is not None:
                    raise UnverifiedState("Reference must be an uninterrupted fresh run")
                resumed = b["resume_from"]
                if comparison == "fresh" and resumed is not None:
                    raise UnverifiedState("Fresh replay unexpectedly used a checkpoint")
                if comparison == "resume" and (
                    not resumed
                    or resumed["run_id"] != a["run_id"]
                    or not 0 < resumed["step"] < min(steps)
                    or not resumed.get("checkpoint_sha256")
                ):
                    raise UnverifiedState("Resume must identify the reference checkpoint")
                if comparison == "resume":
                    if type(resumed["step"]) is not int:
                        raise UnverifiedState("Checkpoint step must be an integer")
                    if resume_step is not None and resumed["step"] != resume_step:
                        raise UnverifiedState("Checkpoint step differs across ranks or snapshots")
                    resume_step = resumed["step"]
                    if rank in resume_by_rank and resume_by_rank[rank] != resumed:
                        raise UnverifiedState("Checkpoint identity changed within a resumed rank")
                    resume_by_rank[rank] = resumed
                    checkpoint_key = (resumed["step"], rank)
                    if checkpoint_key not in checkpoint_records:
                        checkpoint = checkpoint_path(reference, resumed["step"], rank)
                        record = json.loads(checkpoint.with_suffix(".json").read_text())
                        if record["checkpoint_sha256"] != file_sha256(checkpoint):
                            raise UnverifiedState("Reference checkpoint changed after recording")
                        checkpoint_records[checkpoint_key] = record
                    if resumed != checkpoint_records[checkpoint_key] or resumed["rank"] != rank:
                        raise UnverifiedState("Resume checkpoint identity does not match reference")
                pairs.append((step, rank, paths, records))
        if any(len(ids) != 1 for ids in run_ids) or run_ids[0] == run_ids[1]:
            raise UnverifiedState("Snapshots must belong to two distinct runs")
        if identities[0] & identities[1]:
            raise UnverifiedState("Replay requires independent fresh processes")
        if any(len(ids) != world_size for ids in identities):
            raise UnverifiedState("Every rank must be captured by its own process")
        result.update(
            provenance_by_rank={str(rank): value for rank, value in provenance_by_rank.items()},
            reference_run_id=next(iter(run_ids[0])),
            candidate_run_id=next(iter(run_ids[1])),
        )
        compared = 0
        difference = None
        for step, rank, paths, records in pairs:
            difference = _first_difference(paths[0], paths[1], records[0], records[1])
            compared += 1
            if difference:
                difference.update(step=step, rank=rank)
                break
        observed = "different" if difference else "equal"
        result.update(comparison_status=observed, compared_snapshots=compared)
        result.update(status="not_verified" if dirty else observed, first_difference=difference)
        if dirty:
            result["reason"] = "Source tree is dirty; observation is diagnostic only"
        result["scope"] = "Only the declared adapter state, steps, ranks and matched provenance"
        return result
    except (OSError, ValueError, KeyError, TypeError) as error:
        return result | {"reason": str(error)}


def main() -> int:
    """Compare saved state; exit 0 for equality, 1 for difference, 2 for unverified."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--comparison", choices=("fresh", "resume"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(
        args.reference,
        args.candidate,
        steps=args.steps,
        world_size=args.world_size,
        comparison=args.comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return {"equal": 0, "different": 1, "not_verified": 2}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
