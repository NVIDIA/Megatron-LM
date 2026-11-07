# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Plan a blend of corpus sources into a fixed, DP-independent sample schedule.

A blend config names a corpus directory, sources with weights in samples, the split,
a seed, and a lane count ``L``::

    {"corpus": "/data/corpus", "sources": {"web_crawl": 0.7, "math_sft": 0.3},
     "split": "train", "seed": 1234, "lanes": 64, "allow_partial": []}

The planner picks each source's rows (whole passes over its blocks, then a seeded
subset), spreads every source evenly along one stream, and cuts the stream into ``L``
equal lanes. Global sample ``p`` is row ``p // L`` of lane ``p % L``, so the schedule
does not depend on the student's data-parallel size; when ``L`` is a multiple of DP ×
MBS, each lane is read by exactly one student rank. Each lane is a list of pieces
``(file, block, first row, end row)``, so a reader fetches whole blocks and uses the
pieces' rows.
"""

import hashlib
import json
import os
from typing import Any

import numpy as np

from ..v3_storage import (
    FileName,
    Storage,
    complete_ranges,
    contiguous_end,
    list_published,
    read_index,
)
from .arrow import check_index, file_meta
from .contract import REQUIRED_KEYS, mismatched_settings, validate_metadata

MANIFEST_VERSION = 1
DEFAULTS = {"split": "train", "seed": 1234, "lanes": 64, "allow_partial": []}


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


def _source_rows(storage: Storage, name: str, partial: bool) -> tuple[dict, list, list, list]:
    """Return a source's settings, files, file sizes, and pieces of real (non-filler) rows."""
    files = list_published(storage)
    if not files:
        raise ValueError(f"Corpus source {name} has no published files")
    indexes = []
    meta = None
    for file in files:
        index = read_index(storage, file)
        check_index(index)
        file_settings = file_meta(index)
        validate_metadata(file_settings, corpus=True)
        if meta is None:
            meta = file_settings
        elif mismatched_settings(file_settings, meta)[0]:
            raise ValueError(f"Corpus source {name} mixes files written with different settings")
        indexes.append(index)
    contract = meta["contract"]
    writers, mbs, total = contract["writers"], contract["writer_mbs"], contract["num_samples"]
    covered = contiguous_end(complete_ranges(files, writers), 0)
    if covered < total and not partial:
        raise ValueError(
            f"Corpus source {name} covers samples 0-{covered} of {total}; finish its dump, or "
            "list it under allow_partial to blend what is published"
        )
    limit = min(covered, total)
    pieces = []  # (file id, block, first row, end row, supervised rows)
    for file_id, (file, index) in enumerate(zip(files, indexes)):
        if file.start >= limit:
            continue
        for block, entry in enumerate(index["blocks"]):
            rows = np.arange(entry["first_row"], entry["first_row"] + entry["rows"])
            # The writer layout maps each row of this writer's file back to its sample.
            samples = file.start + (rows // mbs) * mbs * writers + file.dp_rank * mbs + rows % mbs
            real = samples < limit
            if not real.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(([0], real.astype(np.int8), [0]))))
            per_row = entry["supervised"] / entry["rows"]
            for a, b in zip(edges[0::2], edges[1::2]):
                pieces.append((file_id, block, int(a), int(b), per_row * (b - a)))
    if not pieces:
        raise ValueError(f"Corpus source {name} has no samples to blend")
    return meta, files, [index["size"] for index in indexes], pieces


def _seed(*parts: Any) -> int:
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _select(pieces: list, target: int, seed: int, name: str) -> tuple[list, int]:
    """Take ``target`` rows: whole seeded passes over the pieces, then a seeded subset."""
    total = sum(b - a for _, _, a, b, _ in pieces)
    chosen, remaining, passes = [], target, 0
    while remaining > 0:
        order = np.random.RandomState(_seed(seed, name, passes) % (1 << 32)).permutation(
            len(pieces)
        )
        for i in order:
            file_id, block, a, b, supervised = pieces[i]
            take = min(b - a, remaining)
            chosen.append((file_id, block, a, a + take, supervised * take / (b - a)))
            remaining -= take
            if not remaining:
                break
        passes += 1
    return chosen, -(-target // total) if total else 0


def plan(config: dict[str, Any], *, budget: int) -> dict[str, Any]:
    """Plan a blend of at least ``budget`` samples into a manifest.

    The manifest holds the settings every file shares, the files it reads, per-source
    shares, and ``segments``: rows of ``(lane, file id, block, first row, end row)``.
    """
    root, split, seed, lanes = config["corpus"], config["split"], config["seed"], config["lanes"]
    weights = config["sources"]
    total = lanes * -(-budget // lanes)  # Whole lanes, at least the budget.
    files, sources, streams, reference = [], [], [], None
    weight_sum = sum(weights.values())
    targets = {name: int(total * weight / weight_sum) for name, weight in weights.items()}
    for name in sorted(weights)[: total - sum(targets.values())]:
        targets[name] += 1  # Distribute the rounding remainder.
    for name in sorted(weights):
        folder = f"{name}/{split}"
        storage = Storage(os.path.join(root, folder))
        meta, source_files, sizes, pieces = _source_rows(
            storage, name, name in config.get("allow_partial", [])
        )
        if reference is None:
            reference = meta
        differing = [
            key for key in REQUIRED_KEYS if meta["contract"][key] != reference["contract"][key]
        ]
        if differing:
            raise ValueError(f"Corpus source {name} differs on {', '.join(differing)}")
        offset = len(files)
        files += [
            {"folder": folder, "dp_rank": f.dp_rank, "start": f.start, "end": f.end, "size": size}
            for f, size in zip(source_files, sizes)
        ]
        chosen, passes = _select(pieces, targets[name], seed, name)
        # Spread each source evenly along the stream: sort by position within its source.
        rng = np.random.RandomState(_seed(seed, name, "spread") % (1 << 32))
        cursor = 0
        for file_id, block, a, b, supervised in chosen:
            key = (cursor + rng.random_sample() * (b - a)) / targets[name]
            streams.append((key, file_id + offset, block, a, b, len(sources), supervised))
            cursor += b - a
        sources.append(
            {
                "name": name,
                "weight": weights[name],
                "samples": targets[name],
                "available": sum(b - a for _, _, a, b, _ in pieces),
                "passes": passes,
                "supervised": float(sum(piece[4] for piece in chosen)),
                "teacher": meta.get("recorded", {}).get("teacher"),
            }
        )
    streams.sort()
    # Cut the stream into equal lanes; a piece crossing a lane boundary is split.
    per_lane = total // lanes
    segments, lane, used = [], 0, 0
    for _, file_id, block, a, b, _, _ in streams:
        while a < b:
            take = min(b - a, per_lane - used)
            segments.append((lane, file_id, block, a, a + take))
            a += take
            used += take
            if used == per_lane:
                lane, used = lane + 1, 0
    supervised = sum(source["supervised"] for source in sources) or 1.0
    for source in sources:
        source["share"] = source["samples"] / total
        source["supervised_share"] = source["supervised"] / supervised
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "config": config,
        "budget": budget,
        "lanes": lanes,
        "lane_samples": per_lane,
        "contract": {key: reference["contract"][key] for key in REQUIRED_KEYS},
        "files": files,
        "sources": sources,
        "segments": np.asarray(segments, dtype=np.int64).reshape(-1, 5),
    }
    manifest["hash"] = manifest_hash(manifest)
    return manifest


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Identify a manifest's schedule: its files, contract, lanes and segments."""
    digest = hashlib.sha256()
    for key in ("contract", "files", "lanes", "lane_samples"):
        digest.update(json.dumps(manifest[key], sort_keys=True).encode())
    digest.update(np.ascontiguousarray(manifest["segments"]).tobytes())
    return digest.hexdigest()


def describe(manifest: dict[str, Any]) -> str:
    """One line per source: requested weight, realized share, passes."""
    lines = [
        f"Offline KD corpus blend {manifest['hash'][:12]}: {manifest['lanes']} lanes x "
        f"{manifest['lane_samples']} samples"
    ]
    for source in manifest["sources"]:
        lines.append(
            f"  {source['name']}: weight {source['weight']}, {source['samples']} samples "
            f"({source['share']:.1%}; {source['supervised_share']:.1%} of supervised tokens) "
            f"from {source['available']} available, {source['passes']} pass(es)"
        )
    return "\n".join(lines)


__all__ = ["FileName", "describe", "load_config", "manifest_hash", "plan"]
