# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Publication and listing of offline KD v3 files.

A cache is a flat directory of per-writer files, one data file and one index per
writer and flush (save-interval) range::

    dp{rank}__{start}-{end}.arrow        # samples and their targets (see format.arrow)
    dp{rank}__{start}-{end}.index.json   # block byte ranges and a footer copy

The index is written after its data file is complete, so its existence publishes the
range for that writer: local writes are staged and renamed, and remote objects only
become visible once their upload completes. Data files without an index are ignored
and replaced by a retry.
"""

import glob
import importlib
import json
import os
import re
import uuid
from typing import Any, NamedTuple

_FILE_NAME = re.compile(r"^dp(\d+)__(\d+)-(\d+)\.(arrow|index\.json)$")
_STAGED_NAME = re.compile(r"^(.+)\.[0-9a-f]{32}\.tmp$")


class FileName(NamedTuple):
    """A parsed v3 file name."""

    dp_rank: int
    start: int
    end: int

    @property
    def data(self) -> str:
        """The data file's name."""
        return data_name(self.dp_rank, self.start, self.end)

    @property
    def index(self) -> str:
        """The index's name."""
        return index_name(self.dp_rank, self.start, self.end)


def data_name(dp_rank: int, start: int, end: int) -> str:
    """Name the data file holding one writer's flush range."""
    return f"dp{dp_rank}__{start}-{end}.arrow"


def index_name(dp_rank: int, start: int, end: int) -> str:
    """Name the index published after that data file."""
    return f"dp{dp_rank}__{start}-{end}.index.json"


def parse_name(name: str) -> FileName | None:
    """Parse a v3 data file or index name, or return None for other objects."""
    match = _FILE_NAME.match(name)
    if match is None:
        return None
    dp_rank, start, end, _ = match.groups()
    return FileName(int(dp_rank), int(start), int(end))


class Storage:
    """Use local storage or Megatron's configured MSC backend."""

    def __init__(self, root: str):
        self.root = root
        self.remote = root.startswith("msc://")
        self.backend = None
        if self.remote:
            from megatron.core.msc_utils import MultiStorageClientFeature

            self.backend = MultiStorageClientFeature.import_package()

    def path(self, name: str) -> str:
        """Resolve a cache-relative filename."""
        return os.path.join(self.root, name)

    def open(self, name: str, mode: str = "rb") -> Any:
        """Open a cache object."""
        path = self.path(name)
        return self.backend.open(path, mode) if self.remote else open(path, mode)

    def exists(self, name: str) -> bool:
        """Check whether a cache object exists."""
        return (
            self.backend.os.path.exists(self.path(name))
            if self.remote
            else os.path.exists(self.path(name))
        )

    def list(self, pattern: str) -> list[str]:
        """List fresh object names; never cache a growing-cache listing."""
        paths = (
            self.backend.glob(self.path(pattern)) if self.remote else glob.glob(self.path(pattern))
        )
        return sorted(os.path.basename(str(path)) for path in paths)

    def remove(self, name: str) -> None:
        """Delete an object."""
        if self.remote:
            self.backend.delete(self.path(name))
        else:
            os.unlink(self.path(name))

    def _client(self, name: str) -> tuple[Any, str]:
        return self.backend.resolve_storage_client(self.path(name))

    def size(self, name: str) -> int:
        """Return an object's size in bytes."""
        if not self.remote:
            return os.path.getsize(self.path(name))
        client, path = self._client(name)
        return int(client.info(path).content_length)

    def read(self, name: str) -> bytes:
        """Read a whole (small) object with one request."""
        with self.open(name) as stream:
            return stream.read()

    def read_range(self, name: str, offset: int, size: int) -> bytes:
        """Read ``size`` bytes at ``offset`` with exactly one request."""
        if not self.remote:
            with open(self.path(name), "rb") as stream:
                stream.seek(offset)
                data = stream.read(size)
        else:
            client, path = self._client(name)
            byte_range = importlib.import_module("multistorageclient.types").Range(
                offset=offset, size=size
            )
            data = client.read(path, byte_range=byte_range)
            data = data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)
        if len(data) != size:
            raise IOError(f"Short ranged read from offline KD object {name}")
        return data

    def write(self, name: str, data: bytes) -> None:
        """Write an object; local writes are staged and renamed into place."""
        if self.remote:
            with self.open(name, "wb") as stream:
                stream.write(data)
            return
        os.makedirs(self.root, exist_ok=True)
        staged = f"{name}.{uuid.uuid4().hex}.tmp"
        with self.open(staged, "wb") as stream:
            stream.write(data)
        os.replace(self.path(staged), self.path(name))


def publish(storage: Storage, name: FileName, data: bytes, index: dict[str, Any]) -> None:
    """Publish an immutable range: the data file first, then its index.

    A published range is never replaced. A data file without an index (an interrupted
    earlier attempt) is overwritten.
    """
    if storage.exists(name.index):
        raise RuntimeError(f"Refusing to replace published offline KD v3 file {name.data}")
    storage.write(name.data, data)
    storage.write(name.index, json.dumps(index, separators=(",", ":")).encode())


def read_index(storage: Storage, name: FileName) -> dict[str, Any]:
    """Read a published range's index."""
    return json.loads(storage.read(name.index))


def read_meta(storage: Storage, name: FileName) -> dict[str, Any]:
    """Read a published file's settings from its index (one small request)."""
    from .format.arrow import check_index, file_meta

    index = read_index(storage, name)
    check_index(index)
    return file_meta(index)


def has_legacy_tars(storage: Storage) -> bool:
    """Whether the directory holds V1/V2 tar dumps, which V3 never mixes with."""
    return bool(storage.list("*.tar"))


def list_published(storage: Storage) -> list[FileName]:
    """List published ranges (those with an index), ignoring staged and other objects."""
    return sorted(
        parsed
        for parsed in map(parse_name, storage.list("dp*__*.index.json"))
        if parsed is not None
    )


def complete_ranges(published: list[FileName], writers: int) -> list[tuple[int, int]]:
    """Return flush ranges published by every writer."""
    present: dict[tuple[int, int], set[int]] = {}
    for name in published:
        present.setdefault((name.start, name.end), set()).add(name.dp_rank)
    return sorted(span for span, found in present.items() if set(range(writers)) <= found)


def contiguous_end(ranges: list[tuple[int, int]], start: int, end: int | None = None) -> int:
    """Return the end of the contiguous run of ``ranges`` beginning at ``start``."""
    by_start = dict(ranges)
    cursor = start
    while cursor in by_start and (end is None or cursor < end):
        cursor = by_start[cursor]
    return cursor


def discard_unpublished(storage: Storage, published_through: int, end: int) -> list[str]:
    """Delete a resuming job's partial tail in ``[published_through, end)``.

    Only this job's own range is touched, so parallel dump jobs never interfere.
    Readers cannot have consumed these objects: they lie beyond a hole.
    """
    removed = []
    # Indexes first: an interrupted cleanup then leaves only unpublished data files.
    for name in sorted(storage.list("dp*__*"), key=lambda name: ".index.json" not in name):
        match = _STAGED_NAME.match(name)
        parsed = parse_name(match.group(1) if match else name)
        if parsed is not None and published_through <= parsed.start < end:
            if match is None or not storage.remote:
                storage.remove(name)
                removed.append(name)
    return removed
