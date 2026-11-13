# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Plan a blend of corpus sources into a fixed, DP-independent sample schedule.

A blend config names a corpus directory, sources with weights in samples, a seed, and a
lane count ``L``::

    {"corpus": "/data/corpus", "sources": {"web_crawl": 0.7, "math_sft": 0.3},
     "seed": 1234, "lanes": 64}

Every source must be complete. The planner picks each source's rows (whole passes over
its blocks, then a seeded subset), spreads every source evenly along one stream, and
cuts the stream into ``L`` equal lanes. Global sample ``p`` is row ``p // L`` of lane
``p % L``, so the schedule does not depend on the student's data-parallel size; when
``L`` is a multiple of DP x MBS, each lane is read by exactly one student rank.

A plan is compact and in memory: numeric file and block tables, one Arrow schema per
distinct column layout (no per-file footers), and per-lane ``segments`` of
``(block, first row, end row)``. :func:`rank_plan` keeps only the lanes, blocks, files
and schemas one data-parallel rank reads.
"""

import hashlib
import json
import math
import os
from typing import Any

import numpy as np

from ..v3_storage import Storage
from .arrow import block_schema
from .contract import blend_contract, mismatched_corpus_order, mismatched_settings
from .corpus import check_chunk, relative_path

MANIFEST_VERSION = 4
DEFAULTS = {"seed": 1234, "lanes": 64}
# 32 bytes per block: byte ranges are within one file, and rows within one block.
BLOCK_DTYPE = np.dtype(
    [
        ("file", "<i4"),
        ("first_row", "<i4"),
        ("rows", "<i4"),
        ("supervised", "<i4"),
        ("offset", "<i8"),
        ("length", "<i4"),
        ("input_length", "<i4"),
    ]
)
# One piece of a lane: rows [first, end) of a block.
SEGMENT_DTYPE = np.dtype([("block", "<i4"), ("first", "<i4"), ("end", "<i4")])
_BLOCK_KEYS = BLOCK_DTYPE.names[1:]
_INT32 = np.iinfo(np.int32).max


def load_config(path: str) -> dict[str, Any]:
    """Read and complete a blend config."""
    with open(path) as stream:
        config = {**DEFAULTS, **json.load(stream)}
    if not isinstance(config.get("corpus"), str) or not config.get("sources"):
        raise ValueError("A corpus blend config needs a corpus directory and weighted sources")
    if any(not isinstance(w, (int, float)) or w <= 0 for w in config["sources"].values()):
        raise ValueError("Corpus blend weights must be positive numbers")
    if type(config["lanes"]) is not int or config["lanes"] < 1:
        raise ValueError("Corpus blend lanes must be a positive integer")
    return config


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _part_blocks(index: dict, file_id: int, label: str) -> np.ndarray:
    values = np.array(
        [[entry[key] for key in _BLOCK_KEYS] for entry in index["blocks"]], dtype=np.int64
    ).reshape(-1, len(_BLOCK_KEYS))
    table = np.empty(len(values), dtype=BLOCK_DTYPE)
    table["file"] = file_id
    for column, key in enumerate(_BLOCK_KEYS):
        if BLOCK_DTYPE[key] == np.int32 and values[:, column].max(initial=0) > _INT32:
            raise ValueError(f"Corpus part {label} exceeds the plan's 32-bit {key} field")
        table[key] = values[:, column]
    return table


def _read_source(
    root: str, name: str, source_id: int, files: dict, schemas: dict
) -> tuple[dict, np.ndarray]:
    """Validate one source's chunk indexes (one at a time) into its block table."""
    storage = Storage(os.path.join(root, name))
    paths = sorted(storage.list("chunk-*/index.json"))
    if not paths:
        raise ValueError(f"Corpus source {name} has no published chunks")
    metadata, tables, intervals = None, [], []
    for path in paths:
        folder = relative_path(os.path.dirname(path))
        chunk = json.loads(storage.read(path))
        current = check_chunk(chunk)
        producer = chunk.get("producer") or {}
        if "start_iteration" in producer and "end_iteration" in producer:
            intervals.append((producer["start_iteration"], producer["end_iteration"], folder))
        if metadata is not None:
            differing = mismatched_settings(current, metadata)[0]
            differing += mismatched_corpus_order(current, metadata)
            if differing:
                raise ValueError(f"Corpus source {name} mixes settings: {', '.join(differing)}")
        metadata = current
        for part in chunk["parts"]:
            index = part["index"]
            part_path = relative_path(f"{folder}/{part['path']}")
            schema = schemas.setdefault(block_schema(index), len(schemas))
            file_id = len(files["path"])
            files["source"].append(source_id)
            files["path"].append(part_path)
            files["size"].append(index["size"])
            files["schema"].append(schema)
            tables.append(_part_blocks(index, file_id, f"{name}/{part_path}"))
    # Matching row totals cannot reveal one range dumped twice and another never.
    intervals.sort()
    for (_, end, first), (start, _, second) in zip(intervals, intervals[1:]):
        if start < end:
            raise ValueError(f"Corpus source {name} has overlapping chunks {first} and {second}")
    blocks = np.concatenate(tables)
    available = int(blocks["rows"].sum(dtype=np.int64))
    expected = metadata["contract"]["num_samples"]
    if available != expected:
        raise ValueError(
            f"Corpus source {name} covers {available} of {expected} samples; finish its dump "
            "first and check for duplicate chunks"
        )
    return metadata, blocks


def _select(rows: np.ndarray, target: int, seed: int, name: str) -> tuple[np.ndarray, ...]:
    """Take ``target`` rows: whole seeded passes over the blocks, then a seeded subset.

    Returns the chosen ``(block, rows taken)``; each block's rows start at its first.
    """
    empty = np.zeros(0, dtype=np.int64)  # A tiny weight may round to no samples.
    available, chosen, takes, remaining, passes = int(rows.sum()), [empty], [empty], target, 0
    while remaining > 0:
        order = np.random.RandomState(_seed(seed, name, passes) % (1 << 32)).permutation(len(rows))
        if remaining >= available:
            chosen.append(order)
            takes.append(rows[order])
            remaining -= available
        else:
            ends = np.cumsum(rows[order])
            count = int(np.searchsorted(ends, remaining)) + 1  # The last one may be partial.
            take = rows[order[:count]].copy()
            take[-1] -= int(ends[count - 1]) - remaining
            chosen.append(order[:count])
            takes.append(take)
            remaining = 0
        passes += 1
    return np.concatenate(chosen), np.concatenate(takes).astype(np.int64)


def _hash(plan: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    names = [source["name"] for source in plan["sources"]]
    files = plan["files"]
    table = [
        [names[source], path, int(size)]
        for source, path, size in zip(files["source"], files["path"], files["size"])
    ]
    for value in (plan["contract"], table, plan["lanes"], plan["lane_samples"]):
        digest.update(json.dumps(value, sort_keys=True).encode())
    for key in ("blocks", "lane_starts", "segments"):
        digest.update(np.ascontiguousarray(plan[key]).tobytes())
    return digest.hexdigest()


def plan(config: dict[str, Any], *, budget: int) -> dict[str, Any]:
    """Plan a blend of at least ``budget`` samples on one process.

    Returns the whole schedule in compact NumPy tables: ``files`` (source, path, size,
    schema), ``blocks`` (:data:`BLOCK_DTYPE`), ``schemas`` (serialized Arrow schemas),
    and lane ``lane_ids[i]``'s pieces ``segments[lane_starts[i]:lane_starts[i + 1]]``.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("A corpus plan requires a positive sample budget")
    root, seed, lanes = config["corpus"], config["seed"], config["lanes"]
    weights = config["sources"]
    total = lanes * -(-budget // lanes)  # Whole lanes, at least the budget.
    weight_sum = sum(weights.values())
    targets = {name: int(total * weight / weight_sum) for name, weight in weights.items()}
    for name in sorted(weights)[: total - sum(targets.values())]:
        targets[name] += 1  # Distribute the rounding remainder.
    files: dict[str, list] = {"source": [], "path": [], "size": [], "schema": []}
    schemas: dict[bytes, int] = {}
    tables, sources, streams, reference = [], [], [], None
    first_block = 0
    for source_id, name in enumerate(sorted(weights)):
        meta, blocks = _read_source(root, name, source_id, files, schemas)
        shared = blend_contract(meta["contract"])
        if reference is None:
            reference = shared
        differing = [key for key in shared if shared[key] != reference[key]]
        if differing:
            raise ValueError(f"Corpus source {name} differs on {', '.join(differing)}")
        rows = blocks["rows"].astype(np.int64)
        chosen, take = _select(rows, targets[name], seed, name)
        supervised = blocks["supervised"][chosen] * take / rows[chosen]
        # Spread each source evenly along the stream: sort by position within its source.
        spread = np.random.RandomState(_seed(seed, name, "spread") % (1 << 32))
        cursor = np.cumsum(take) - take
        keys = (cursor + spread.random_sample(len(take)) * take) / targets[name]
        streams.append((keys, chosen + first_block, take))
        tables.append(blocks)
        first_block += len(blocks)
        sources.append(
            {
                "name": name,
                "samples": targets[name],
                "available": int(rows.sum()),
                "passes": -(-targets[name] // int(rows.sum())),
                "supervised": float(supervised.sum()),
                "teacher": meta.get("recorded", {}).get("teacher"),
            }
        )
    blocks = np.concatenate(tables)
    if len(blocks) > _INT32:
        raise ValueError("A corpus plan holds at most 2**31 - 1 blocks")
    keys, block_ids, take = (np.concatenate(parts) for parts in zip(*streams))
    order = np.lexsort((block_ids, keys))
    block_ids, take = block_ids[order], take[order]
    # Cut the stream into equal lanes; a piece crossing a lane boundary is split.
    per_lane = total // lanes
    ends = np.cumsum(take)
    starts = ends - take
    cuts = np.union1d(starts, np.arange(0, total, per_lane))
    piece = np.searchsorted(ends, cuts, side="right")
    first = cuts - starts[piece]  # Every chosen piece starts at its block's first row.
    segments = np.empty(len(cuts), dtype=SEGMENT_DTYPE)
    segments["block"] = block_ids[piece]
    segments["first"] = first
    segments["end"] = first + np.diff(np.append(cuts, total))
    lane_starts = np.searchsorted(cuts // per_lane, np.arange(lanes + 1)).astype(np.int64)
    scored = sum(source["supervised"] for source in sources) or 1.0
    for source in sources:
        source["share"] = source["samples"] / total
        source["supervised_share"] = source["supervised"] / scored
    result = {
        "manifest_version": MANIFEST_VERSION,
        "config": config,
        "budget": budget,
        "lanes": lanes,
        "lane_samples": per_lane,
        "contract": reference,
        "sources": sources,
        "schemas": list(schemas),
        "files": {
            "source": np.asarray(files["source"], dtype=np.int32),
            "path": files["path"],
            "size": np.asarray(files["size"], dtype=np.int64),
            "schema": np.asarray(files["schema"], dtype=np.int32),
        },
        "blocks": blocks,
        "lane_ids": np.arange(lanes, dtype=np.int64),
        "lane_starts": lane_starts,
        "segments": segments,
        "rank": None,
    }
    result["hash"] = _hash(result)
    return result


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Identify a schedule: contract, file table, lanes, lane length and pieces.

    Only a whole plan can be hashed; a :func:`rank_plan` carries its plan's hash.
    """
    if manifest.get("rank") is not None:
        raise ValueError("A rank-local corpus plan carries, but cannot recompute, its hash")
    return _hash(manifest)


def rank_lanes(
    lanes: int, total: int, *, dp_rank: int, dp_size: int, micro_batch_size: int, start: int = 0
) -> np.ndarray:
    """Lanes that DP rank ``r`` reads from sample ``start`` (the resumed cursor) on.

    Its positions are ``start + (m * D + r) * mbs + j`` modulo ``L``: every global batch,
    ramp-up included, is a multiple of ``D * mbs``, so they keep that offset. A resume
    that changes DP or micro-batch size therefore needs its own cursor here.
    """
    step = dp_size * micro_batch_size
    # Lane patterns repeat every ``L / gcd(step, L)`` global steps.
    steps = min(-(-max(total - start, 0) // step), lanes // math.gcd(step, lanes))
    positions = (
        start
        + np.arange(steps, dtype=np.int64)[:, None] * step
        + dp_rank * micro_batch_size
        + np.arange(micro_batch_size, dtype=np.int64)[None, :]
    ).reshape(-1)
    return np.unique(positions[positions < total] % lanes)


def rank_plan(
    manifest: dict[str, Any], *, dp_rank: int, dp_size: int, micro_batch_size: int, start: int = 0
) -> dict[str, Any]:
    """Keep only the lanes one DP rank reads from ``start`` on, and the blocks, files and
    schemas they use."""
    if not 0 <= dp_rank < dp_size or micro_batch_size < 1:
        raise ValueError("Invalid data-parallel layout for a corpus plan")
    lanes = rank_lanes(
        manifest["lanes"],
        manifest["lanes"] * manifest["lane_samples"],
        dp_rank=dp_rank,
        dp_size=dp_size,
        micro_batch_size=micro_batch_size,
        start=start,
    )
    if not np.isin(lanes, manifest["lane_ids"]).all():
        raise ValueError("This corpus plan does not hold every lane the rank reads")
    local = np.searchsorted(manifest["lane_ids"], lanes)
    starts = manifest["lane_starts"]
    counts = starts[local + 1] - starts[local]
    lane_starts = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    rows = np.arange(lane_starts[-1], dtype=np.int64) + np.repeat(
        starts[local] - lane_starts[:-1], counts
    )
    segments = manifest["segments"][rows]
    used_blocks, segments["block"] = np.unique(segments["block"], return_inverse=True)
    blocks = manifest["blocks"][used_blocks]
    used_files, blocks["file"] = np.unique(blocks["file"], return_inverse=True)
    files = manifest["files"]
    used_schemas, schema_ids = np.unique(files["schema"][used_files], return_inverse=True)
    return {
        **manifest,
        "schemas": [manifest["schemas"][i] for i in used_schemas],
        "files": {
            "source": files["source"][used_files],
            "path": [files["path"][i] for i in used_files],
            "size": files["size"][used_files],
            "schema": schema_ids.astype(np.int32),
        },
        "blocks": blocks,
        "lane_ids": lanes,
        "lane_starts": lane_starts,
        "segments": segments,
        "rank": {
            "dp_rank": dp_rank,
            "dp_size": dp_size,
            "micro_batch_size": micro_batch_size,
            "start": start,
        },
    }


def describe(manifest: dict[str, Any]) -> str:
    """One line per source: requested weight, realized share, passes."""
    lines = [
        f"Offline KD corpus blend {manifest['hash'][:12]}: {manifest['lanes']} lanes x "
        f"{manifest['lane_samples']} samples"
    ]
    for source in manifest["sources"]:
        weight = manifest["config"]["sources"][source["name"]]
        lines.append(
            f"  {source['name']}: weight {weight}, {source['samples']} samples "
            f"({source['share']:.1%}; {source['supervised_share']:.1%} of supervised tokens) "
            f"from {source['available']} available, {source['passes']} pass(es)"
        )
    return "\n".join(lines)


__all__ = ["describe", "load_config", "manifest_hash", "plan", "rank_lanes", "rank_plan"]
