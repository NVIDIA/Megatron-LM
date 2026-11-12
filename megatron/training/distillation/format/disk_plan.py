# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Disk-backed corpus inventories and deterministic external-sort blend plans.

Only a small descriptor crosses process boundaries. SQLite's bounded page cache and
file-backed sorter keep both index discovery and schedule construction independent
of corpus size in RAM. Artifacts live on the caller's shared POSIX plan directory.
"""

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import uuid
import zlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..v3_storage import Storage
from .arrow import _open
from .contract import blend_contract, mismatched_corpus_order, mismatched_settings
from .corpus import check_chunk, relative_path

MANIFEST_VERSION = 3
_BATCH = 4096


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256(_json(parts).encode()).digest()[:8], "little")


def _connection(path: str, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        database = sqlite3.connect(Path(path).as_uri() + "?mode=ro&immutable=1", uri=True)
    else:
        database = sqlite3.connect(path)
    database.execute("PRAGMA cache_size=-8192")
    database.execute("PRAGMA temp_store=FILE")
    database.execute("PRAGMA mmap_size=0")
    return database


def plan(config: dict[str, Any], *, budget: int, cache_dir: str | None = None) -> dict[str, Any]:
    """Build a bounded-memory plan and atomically publish its immutable SQLite catalog.

    ``cache_dir`` must be shared POSIX storage for distributed consumers. Omitting it
    creates a persistent temporary directory for a standalone HF/local reader.
    """
    if type(budget) is not int or budget < 1:
        raise ValueError("A corpus plan requires a positive sample budget")
    if cache_dir is not None and "://" in cache_dir:
        raise ValueError("Corpus plans require a POSIX cache directory")
    directory = os.path.abspath(cache_dir or tempfile.mkdtemp(prefix="offline-kd-plan-"))
    if "://" in directory:
        raise ValueError("Corpus plans require a POSIX cache directory")
    os.makedirs(directory, exist_ok=True)
    staging = os.path.join(directory, f".building-{uuid.uuid4().hex}.sqlite")
    db = _connection(staging)
    try:
        return _build(db, staging, directory, config, budget)
    except BaseException:
        db.close()
        for suffix in ("", "-journal"):
            try:
                os.unlink(staging + suffix)
            except FileNotFoundError:
                pass
        raise


def _build(db: sqlite3.Connection, staging: str, directory: str, config: dict, budget: int) -> dict:
    db.executescript("""
        CREATE TABLE schemas(id INTEGER PRIMARY KEY, fingerprint TEXT UNIQUE, dictionary BLOB);
        CREATE TABLE files(id INTEGER PRIMARY KEY, folder TEXT, path TEXT, size INTEGER,
                           schema_id INTEGER, footer BLOB, UNIQUE(folder,path));
        CREATE TABLE blocks(id INTEGER PRIMARY KEY, source INTEGER, file INTEGER, block INTEGER,
                            first_row INTEGER, rows INTEGER, offset INTEGER, length INTEGER,
                            input_length INTEGER, supervised INTEGER, UNIQUE(file,block));
        CREATE INDEX source_blocks ON blocks(source,id);
        CREATE TABLE streams(sort_key REAL, file INTEGER, block INTEGER, a INTEGER, b INTEGER,
                             source INTEGER, supervised REAL);
        CREATE TABLE segments(lane INTEGER, ordinal INTEGER, start INTEGER, end INTEGER,
                              file INTEGER, block INTEGER, a INTEGER, b INTEGER,
                              PRIMARY KEY(lane,ordinal));
        CREATE INDEX lane_offsets ON segments(lane,end);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT);
    """)
    lanes, seed = config["lanes"], config["seed"]
    total = lanes * -(-budget // lanes)
    weights = config["sources"]
    weight_sum = sum(weights.values())
    targets = {name: int(total * weight / weight_sum) for name, weight in weights.items()}
    for name in sorted(weights)[: total - sum(targets.values())]:
        targets[name] += 1
    reference, summaries = None, []
    digest = hashlib.sha256()
    digest.update(
        _json(
            {
                "version": MANIFEST_VERSION,
                "seed": seed,
                "lanes": lanes,
                "total": total,
                "budget": budget,
            }
        ).encode()
    )
    for source_id, name in enumerate(sorted(weights)):
        storage = Storage(os.path.join(config["corpus"], name))
        # One listing per source; only one chunk's JSON is retained at a time.
        paths = storage.list("chunk-*/index.json")
        if not paths:
            raise ValueError(f"Corpus source {name} has no published chunks")
        metadata, available = None, 0
        for path in paths:
            chunk = json.loads(storage.read(path))
            current = check_chunk(chunk)
            if metadata is not None:
                differing = mismatched_settings(current, metadata)[0] + mismatched_corpus_order(
                    current, metadata
                )
                if differing:
                    raise ValueError(f"Corpus source {name} mixes settings: {', '.join(differing)}")
            else:
                metadata = current
            folder = os.path.dirname(path)
            for part in chunk["parts"]:
                index = part["index"]
                part_name = relative_path(f"{folder}/{part['path']}")
                schema = _open(index, {}).schema.serialize().to_pybytes()
                fingerprint = hashlib.sha256(schema).hexdigest()
                footer = base64.b64decode(index["footer"])
                found = db.execute(
                    "SELECT id,dictionary FROM schemas WHERE fingerprint=?", (fingerprint,)
                ).fetchone()
                if found is None:
                    schema_id = db.execute(
                        "INSERT INTO schemas(fingerprint,dictionary) VALUES(?,?)",
                        (fingerprint, schema),
                    ).lastrowid
                    dictionary = schema
                else:
                    schema_id, dictionary = found
                compressor = zlib.compressobj(zdict=dictionary[-32768:])
                compressed = compressor.compress(footer) + compressor.flush()
                file_id = db.execute(
                    "INSERT INTO files(folder,path,size,schema_id,footer) VALUES(?,?,?,?,?)",
                    (name, part_name, index["size"], schema_id, compressed),
                ).lastrowid
                digest.update(_json([name, part_name, index["size"], fingerprint]).encode())
                digest.update(footer)
                for block, entry in enumerate(index["blocks"]):
                    values = tuple(
                        entry[key]
                        for key in (
                            "first_row",
                            "rows",
                            "offset",
                            "length",
                            "input_length",
                            "supervised",
                        )
                    )
                    db.execute(
                        "INSERT INTO blocks(source,file,block,first_row,rows,offset,length,input_length,supervised) VALUES(?,?,?,?,?,?,?,?,?)",
                        (source_id, file_id, block, *values),
                    )
                    available += entry["rows"]
                    digest.update(np.asarray(values, dtype="<i8").tobytes())
            del chunk
        expected = metadata["contract"]["num_samples"]
        if available != expected:
            raise ValueError(
                f"Corpus source {name} covers {available} of {expected} samples; finish its dump first and check for duplicate chunks"
            )
        shared = blend_contract(metadata["contract"])
        if reference is None:
            reference = shared
            digest.update(_json(reference).encode())
        elif shared != reference:
            differing = [key for key in shared if shared[key] != reference[key]]
            raise ValueError(f"Corpus source {name} differs on {', '.join(differing)}")
        target, remaining, pass_number, cursor = targets[name], targets[name], 0, 0
        supervised = 0.0
        spread = np.random.RandomState(_seed(seed, name, "spread") % (1 << 32))
        while remaining:
            db.execute("DROP TABLE IF EXISTS shuffle")
            db.execute("CREATE TEMP TABLE shuffle(sort_key INTEGER, block_id INTEGER)")
            rng = np.random.RandomState(_seed(seed, name, pass_number) % (1 << 32))
            pieces = db.execute("SELECT id FROM blocks WHERE source=? ORDER BY id", (source_id,))
            while batch := pieces.fetchmany(_BATCH):
                keys = rng.randint(0, np.iinfo(np.int64).max, size=len(batch), dtype=np.int64)
                db.executemany(
                    "INSERT INTO shuffle VALUES(?,?)",
                    ((int(key), row[0]) for key, row in zip(keys, batch)),
                )
            selected = db.execute(
                "SELECT b.file,b.block,b.rows,b.supervised FROM shuffle s JOIN blocks b ON b.id=s.block_id ORDER BY s.sort_key,s.block_id"
            )
            while remaining and (batch := selected.fetchmany(_BATCH)):
                for file_id, block, rows, scored in batch:
                    take = min(rows, remaining)
                    score = scored * take / rows
                    key = (cursor + spread.random_sample() * take) / target
                    db.execute(
                        "INSERT INTO streams VALUES(?,?,?,?,?,?,?)",
                        (key, file_id, block, 0, take, source_id, score),
                    )
                    supervised += score
                    cursor += take
                    remaining -= take
                    if not remaining:
                        break
            selected.close()
            pass_number += 1
        summaries.append(
            {
                "name": name,
                "samples": target,
                "available": available,
                "passes": -(-target // available),
                "supervised": supervised,
                "teacher": metadata.get("recorded", {}).get("teacher"),
            }
        )
        db.commit()
    db.execute("DROP TABLE IF EXISTS shuffle")
    per_lane = total // lanes
    lane = used = ordinal = 0
    stream = db.execute(
        "SELECT file,block,a,b FROM streams ORDER BY sort_key,file,block,a,b,source,supervised"
    )
    while batch := stream.fetchmany(_BATCH):
        for file_id, block, a, b in batch:
            while a < b:
                take = min(b - a, per_lane - used)
                row = (lane, ordinal, used, used + take, file_id, block, a, a + take)
                db.execute("INSERT INTO segments VALUES(?,?,?,?,?,?,?,?)", row)
                digest.update(np.asarray(row, dtype="<i8").tobytes())
                a += take
                used += take
                ordinal += 1
                if used == per_lane:
                    lane, used, ordinal = lane + 1, 0, 0
    stream.close()
    scored_total = sum(source["supervised"] for source in summaries) or 1.0
    for source in summaries:
        source["share"] = source["samples"] / total
        source["supervised_share"] = source["supervised"] / scored_total
    identity = digest.hexdigest()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "config": config,
        "budget": budget,
        "lanes": lanes,
        "lane_samples": per_lane,
        "contract": reference,
        "sources": summaries,
        "hash": identity,
        "catalog": os.path.join(directory, identity + ".sqlite"),
    }
    db.execute("DROP TABLE streams")
    db.execute("INSERT INTO metadata VALUES('manifest',?)", (_json(manifest),))
    db.commit()
    db.execute("VACUUM")
    db.close()
    # Replace only after the complete database is closed; readers never see staging.
    os.replace(staging, manifest["catalog"])
    return manifest


class Catalog:
    """Thread-local read-only access to a compact catalog, with an 8 MiB page cache."""

    def __init__(self, manifest: dict):
        self.manifest = manifest
        self.local = threading.local()
        db = self.connection()
        saved = json.loads(
            db.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0]
        )
        if saved["manifest_version"] != MANIFEST_VERSION or any(
            saved[key] != manifest[key]
            for key in ("hash", "manifest_version", "contract", "lanes", "lane_samples", "budget")
        ):
            self.close_current()
            raise ValueError("Corpus catalog differs from the checkpoint's plan identity")

    def connection(self) -> sqlite3.Connection:
        if not hasattr(self.local, "database"):
            self.local.database = _connection(
                os.path.abspath(self.manifest["catalog"]), readonly=True
            )
        return self.local.database

    def close_current(self) -> None:
        """Close the caller's connection; worker connections close with their threads."""
        database = getattr(self.local, "database", None)
        if database is not None:
            database.close()
            del self.local.database

    def file(self, file_id: int) -> dict:
        row = (
            self.connection()
            .execute("SELECT folder,path,size,schema_id,footer FROM files WHERE id=?", (file_id,))
            .fetchone()
        )
        if row is None:
            raise ValueError(f"Unknown corpus file {file_id}")
        folder, path, size, schema_id, compressed = row
        dictionary = (
            self.connection()
            .execute("SELECT dictionary FROM schemas WHERE id=?", (schema_id,))
            .fetchone()[0]
        )
        decompressor = zlib.decompressobj(zdict=dictionary[-32768:])
        footer = decompressor.decompress(compressed) + decompressor.flush()
        return {
            "folder": folder,
            "path": path,
            "index": {
                "index_version": 1,
                "size": size,
                "footer": base64.b64encode(footer).decode(),
                "blocks": _Blocks(self, file_id),
            },
        }

    def piece(self, lane: int, offset: int) -> tuple:
        row = (
            self.connection()
            .execute(
                "SELECT ordinal,start,file,block,a,b FROM segments WHERE lane=? AND end>? ORDER BY end LIMIT 1",
                (lane, offset),
            )
            .fetchone()
        )
        if row is None:
            raise ValueError(f"Corpus lane {lane} has no sample at {offset}")
        return row

    def lane(self, lane: int, start: int) -> Iterator[tuple]:
        cursor = self.connection().execute(
            "SELECT ordinal,file,block,a,b FROM segments WHERE lane=? AND ordinal>=? ORDER BY ordinal",
            (lane, start),
        )
        while rows := cursor.fetchmany(128):
            yield from rows


class _Blocks:
    def __init__(self, catalog: Catalog, file_id: int):
        self.catalog, self.file_id = catalog, file_id

    def __getitem__(self, block: int) -> dict:
        values = (
            self.catalog.connection()
            .execute(
                "SELECT first_row,rows,offset,length,input_length,supervised FROM blocks WHERE file=? AND block=?",
                (self.file_id, block),
            )
            .fetchone()
        )
        if values is None:
            raise IndexError(block)
        return dict(
            zip(("first_row", "rows", "offset", "length", "input_length", "supervised"), values)
        )
