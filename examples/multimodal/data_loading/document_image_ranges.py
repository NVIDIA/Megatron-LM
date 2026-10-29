# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Local byte-range access and compact catalog resolution, without loader dependencies.

This module implements storage-format contracts only. It does not decode images,
select training examples, canonicalize corpus namespaces, or register a loader.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import struct
import tarfile
from contextlib import closing
from functools import lru_cache
from pathlib import Path

_SEGMENT = struct.Struct(">QQQI4xQ")
_MAX_INDEX_BYTES = 64 << 20


def extract_tar_image_payload(payload: bytes) -> bytes:
    """Read one uncompressed member extent, with optional terminal zero padding.

    A logical image range can end directly after its payload rather than at the
    next TAR block. Iterating the archive would incorrectly require that padding.
    Let tarfile parse metadata, but validate the complete payload before reading.
    """
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        member = archive.next()
        if member is None or not member.isfile() or member.size <= 0:
            raise ValueError("image range must contain one nonempty regular TAR member")
        end = member.offset_data + member.size
        if end > len(payload):
            raise ValueError("image TAR payload is truncated")
        if any(payload[end:]):
            raise ValueError("image range contains extra members or nonzero TAR padding")
        with archive.extractfile(member) as source:
            result = source.read()
        if len(result) != member.size:
            raise ValueError("image TAR payload is truncated")
        return result


class LocalImageRanges:
    """Read contained regular files positionally, without persistent file handles."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("image range root must be a directory")

    def _path(self, relative: str) -> Path:
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise ValueError("image range paths must be contained relative paths")
        candidate = self.root / relative
        if candidate.is_symlink():
            raise ValueError("image range files must not be symlinks")
        path = candidate.resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("image range escapes its root or is not a regular file")
        return path

    @staticmethod
    def _read(path: Path, offset: int, size: int) -> bytes:
        if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
            raise ValueError("image extents require a nonnegative offset and positive size")
        with path.open("rb", buffering=0) as source:
            if offset + size > os.fstat(source.fileno()).st_size:
                raise ValueError("image extent exceeds file size")
            data = os.pread(source.fileno(), size, offset)
        if len(data) != size:
            raise OSError("short positional image read")
        return data


class CompactImageCatalog:
    """Resolve original logical extents before coalescing physical pack reads.

    The catalog and its readiness receipt must be immutable once published.
    Index checksums are verified on first use, with bounded caches. Each SQLite
    connection is local to one lookup, so no connection crosses a fork or thread.
    Payload SHA verification belongs to the delivery audit; this reader checks
    index integrity, mapping uniqueness, pack extents, and read lengths.
    """

    def __init__(
        self, catalog: Path, dataset_id: str, *, validation_run_id: str | None = None
    ) -> None:
        self.catalog = Path(catalog).resolve(strict=True)
        receipt_path = self.catalog.with_suffix(".ready.json")
        if receipt_path.stat().st_size > 1 << 20:
            raise ValueError("compact readiness receipt is oversized")
        receipt = json.loads(receipt_path.read_text())
        if validation_run_id is None:
            ready = (
                receipt.get("full_coverage") is True
                and receipt.get("production_use_forbidden") is not True
            )
        else:
            ready = (
                isinstance(validation_run_id, str)
                and bool(validation_run_id)
                and receipt.get("canary_run_id") == validation_run_id
                and dataset_id == f"canary:{validation_run_id}"
                and receipt.get("full_coverage") is False
                and receipt.get("production_use_forbidden") is True
                and receipt.get("bounded_subset_not_dataset_completion") is True
            )
        if not (
            isinstance(dataset_id, str)
            and dataset_id
            and type(receipt.get("version")) is int
            and receipt["version"] == 1
            and ready
            and receipt.get("dataset_id") == dataset_id
        ):
            raise ValueError("compact catalog readiness or dataset identity differs")
        if self.catalog.stat().st_size > 1 << 30:
            raise ValueError("compact catalog exceeds the bounded metadata budget")
        digest = hashlib.sha256()
        with self.catalog.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != receipt.get("catalog_sha256"):
            raise ValueError("compact catalog checksum differs from readiness receipt")
        self.payloads = LocalImageRanges(receipt["payload_root"])
        with closing(self._connect()) as db:
            identity = db.execute("SELECT value FROM metadata WHERE key='dataset_id'").fetchone()
        if identity != (dataset_id,):
            raise ValueError("compact catalog and readiness identities differ")

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.catalog.as_uri() + "?mode=ro&immutable=1", uri=True)
        db.execute("PRAGMA mmap_size=0")
        db.execute("PRAGMA cache_size=-2048")
        return db

    @lru_cache(maxsize=128)
    def _candidates(self, logical_path: str) -> tuple[int, list[tuple]]:
        with closing(self._connect()) as db:
            source = db.execute(
                "SELECT id,shard FROM tars WHERE path=?", (logical_path,)
            ).fetchone()
            if source is None:
                raise ValueError(f"image coordinate is outside compact catalog: {logical_path}")
            ident, shard = source
            rows = db.execute(
                "SELECT index_path,index_sha256,index_bytes,pack_path,pack_bytes,"
                "first_tar,first_offset,last_tar,last_end FROM packs "
                "WHERE shard=? AND first_tar<=? AND last_tar>=? "
                "ORDER BY first_tar,first_offset LIMIT 129",
                (shard, ident, ident),
            ).fetchall()
        if len(rows) > 128:
            raise ValueError("compact coordinate exceeds the bounded candidate budget")
        return int(ident, 16), rows

    @lru_cache(maxsize=2)
    def _index(
        self, relative: str, expected_hash: str, expected_size: int, pack_size: int
    ) -> bytes:
        if (
            type(expected_size) is not int
            or not 0 < expected_size <= _MAX_INDEX_BYTES
            or expected_size % _SEGMENT.size
            or type(pack_size) is not int
            or pack_size <= 0
        ):
            raise ValueError("invalid compact segment index size or pack size")
        path = self.payloads._path(relative)
        if path.stat().st_size != expected_size:
            raise ValueError("compact segment index size differs")
        raw = LocalImageRanges._read(path, 0, expected_size)
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError("compact segment index checksum differs")
        previous, number, physical_end = None, None, 0
        for ident, begin, end, pack, physical in _SEGMENT.iter_unpack(raw):
            if (
                begin >= end
                or physical != physical_end
                or (previous is not None and (ident, begin) < previous)
                or (number is not None and pack != number)
            ):
                raise ValueError("invalid, overlapping or unsorted compact segment index")
            previous, number, physical_end = (ident, end), pack, physical + end - begin
        if physical_end != pack_size:
            raise ValueError("compact segments do not account for the complete physical pack")
        return raw

    def locate(
        self, logical_path: str, offset: int, size: int, tar_member: bool = False
    ) -> tuple[str, int, int, int]:
        """Require exactly one segment containing the complete logical image extent."""
        if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
            raise ValueError("invalid compact image extent")
        if type(tar_member) is not bool:
            raise ValueError("compact image TAR flag must be boolean")
        with closing(self._connect()) as db:
            excluded = db.execute(
                "SELECT status FROM exclusions WHERE path=? AND offset=? AND size=? AND flags=?",
                (logical_path, offset, size, int(tar_member)),
            ).fetchone()
        if excluded is not None:
            raise ValueError(f"image extent is explicitly excluded: {excluded[0]}")
        ident, candidates = self._candidates(logical_path)
        matches = []
        for index, digest, nbytes, pack, total, first, start, last, end in candidates:
            if not (int(first, 16), start) <= (ident, offset) < (int(last, 16), end):
                continue
            raw = self._index(index, digest, nbytes, total)
            low, high = 0, len(raw) // _SEGMENT.size
            while low < high:
                middle = (low + high) // 2
                row = _SEGMENT.unpack_from(raw, middle * _SEGMENT.size)
                if (row[0], row[1]) <= (ident, offset):
                    low = middle + 1
                else:
                    high = middle
            if low:
                source, begin, stop, _, physical = _SEGMENT.unpack_from(
                    raw, (low - 1) * _SEGMENT.size
                )
                if source == ident and begin <= offset and offset + size <= stop:
                    matches.append((pack, total, physical + offset - begin, size))
        if len(matches) != 1:
            raise ValueError(f"expected exactly one compact image mapping, got {len(matches)}")
        return matches[0]

    def read_many(self, logical_path: str, extents: list[tuple]) -> list[bytes]:
        """Coalesce relocated physical spans only, preserving request order/duplicates."""
        locations = [self.locate(logical_path, *extent) for extent in extents]
        groups, output = {}, [b""] * len(locations)
        for index, (relative, total, offset, size) in enumerate(locations):
            groups.setdefault((relative, total), []).append((offset, size, index))
        for (relative, total), ranges in groups.items():
            path = self.payloads._path(relative)
            with path.open("rb", buffering=0) as source:
                if os.fstat(source.fileno()).st_size != total:
                    raise ValueError("compact physical pack size differs")
                spans = []
                for offset, size, index in sorted(ranges):
                    if size > _MAX_INDEX_BYTES or offset < 0 or offset + size > total:
                        raise ValueError(
                            "compact physical image extent exceeds read budget or pack"
                        )
                    if (
                        not spans
                        or offset - spans[-1][1] > 512 << 10
                        or offset + size - spans[-1][0] > _MAX_INDEX_BYTES
                    ):
                        spans.append([offset, offset + size, []])
                    span = spans[-1]
                    span[1] = max(span[1], offset + size)
                    span[2].append((offset, size, index))
                for begin, end, items in spans:
                    raw = os.pread(source.fileno(), end - begin, begin)
                    if len(raw) != end - begin:
                        raise OSError("short compact physical pack read")
                    for offset, size, index in items:
                        output[index] = raw[offset - begin : offset - begin + size]
        return output
