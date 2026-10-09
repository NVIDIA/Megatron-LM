# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Publication and streaming tar reads for offline KD v3.

A cache is a flat directory of per-teacher-DP-rank tars, one inputs tar and one
targets tar per flush (save-interval) range::

    dp{rank}__{start}-{end}.inputs.tar
    dp{rank}__{start}-{end}.targets.tar

Each tar begins with a ``_meta.json`` member describing the cache settings, the
writing job, and the tar itself, followed by one member per teacher iteration.
A tar is published once its object exists: local writes are staged and renamed,
and remote objects only become visible once their upload completes.
"""

import glob
import importlib
import io
import json
import os
import re
import tarfile
import uuid
from typing import Any, Iterator, NamedTuple

META_MEMBER = "_meta.json"
KINDS = ("inputs", "targets")
DEFAULT_CHUNK_BYTES = 64 << 20
_TAR_NAME = re.compile(r"^dp(\d+)__(\d+)-(\d+)\.(inputs|targets)\.tar$")
_STAGED_NAME = re.compile(r"^(.+\.tar)\.[0-9a-f]{32}\.tmp$")


class TarName(NamedTuple):
    """A parsed v3 tar name."""

    dp_rank: int
    start: int
    end: int
    kind: str

    @property
    def name(self) -> str:
        """Return the object name."""
        return tar_name(self.dp_rank, self.start, self.end, self.kind)


def tar_name(dp_rank: int, start: int, end: int, kind: str) -> str:
    """Name the tar holding one teacher DP rank's flush range of one kind."""
    return f"dp{dp_rank}__{start}-{end}.{kind}.tar"


def member_name(start: int, end: int, kind: str) -> str:
    """Name the member holding one teacher iteration."""
    return f"{start}-{end}.{kind}.pt.zst"


def parse_tar_name(name: str) -> TarName | None:
    """Parse a v3 tar name, or return None for other objects."""
    match = _TAR_NAME.match(name)
    if match is None:
        return None
    dp_rank, start, end, kind = match.groups()
    return TarName(int(dp_rank), int(start), int(end), kind)


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

    def read_range(self, name: str, offset: int, size: int) -> bytes:
        """Read ``size`` bytes at ``offset`` with exactly one request."""
        if not self.remote:
            with open(self.path(name), "rb") as stream:
                stream.seek(offset)
                return stream.read(size)
        client, path = self._client(name)
        byte_range = importlib.import_module("multistorageclient.types").Range(
            offset=offset, size=size
        )
        data = client.read(path, byte_range=byte_range)
        return data.to_bytes() if hasattr(data, "to_bytes") else bytes(data)


class ChunkedReader:
    """Sequential file-like view of a remote object, fetched in large ranged reads.

    ``tarfile`` reads in small records; serving them from large chunks keeps the
    number of storage requests at ``ceil(size / chunk_bytes)`` per object, with
    no multipart fan-out.
    """

    def __init__(self, storage: Storage, name: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
        self.storage = storage
        self.name = name
        self.chunk_bytes = max(1, chunk_bytes)
        self.size = storage.size(name)
        self.position = 0  # Object offset of the next chunk to fetch.
        self.chunk = b""
        self.offset = 0  # Read offset within the current chunk.

    def read(self, size: int = -1) -> bytes:
        """Return up to ``size`` bytes, fetching the next chunk when the current one is spent.

        The chunk is never rebuilt: each read copies only the bytes it returns.
        """
        if self.offset == len(self.chunk) and self.position < self.size:
            length = min(self.chunk_bytes, self.size - self.position)
            self.chunk = self.storage.read_range(self.name, self.position, length)
            if len(self.chunk) != length:
                raise IOError(f"Short ranged read from offline KD v3 object {self.name}")
            self.position += length
            self.offset = 0
        end = len(self.chunk)
        if size is not None and size >= 0:
            end = min(self.offset + size, end)
        data = self.chunk[self.offset : end]
        self.offset = end
        return data

    def close(self) -> None:
        """Release the buffered chunk."""
        self.chunk = b""
        self.offset = 0


def _open_stream(storage: Storage, name: str, chunk_bytes: int) -> Any:
    return ChunkedReader(storage, name, chunk_bytes) if storage.remote else storage.open(name)


def iter_tar(
    storage: Storage, name: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES
) -> Iterator[tuple[str, bytes]]:
    """Stream ``(member name, bytes)`` pairs in order, without random access."""
    stream = _open_stream(storage, name, chunk_bytes)
    try:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for info in archive:
                member = archive.extractfile(info)
                if member is None:
                    raise ValueError(f"Unexpected non-file member {info.name} in {name}")
                yield info.name, member.read()
    finally:
        stream.close()


def read_meta(storage: Storage, name: str, chunk_bytes: int = 1 << 16) -> dict[str, Any]:
    """Read a tar's leading ``_meta.json`` with one small request."""
    members = iter_tar(storage, name, chunk_bytes)
    try:
        member, data = next(members, (None, None))
    finally:
        members.close()
    if member != META_MEMBER:
        raise ValueError(f"Offline KD v3 tar {name} does not begin with {META_MEMBER}")
    return json.loads(data)


def write_tar(
    storage: Storage, name: str, meta: dict[str, Any], members: list[tuple[str, bytes]]
) -> None:
    """Publish an immutable tar: ``_meta.json`` first, then the given members.

    Local writes are staged under a unique temporary name and renamed into place.
    Remote objects only become visible once their upload completes.
    """
    if storage.exists(name):
        raise RuntimeError(f"Refusing to replace published offline KD v3 tar {name}")
    staged = name if storage.remote else f"{name}.{uuid.uuid4().hex}.tmp"
    if not storage.remote:
        os.makedirs(storage.root, exist_ok=True)
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    with storage.open(staged, "wb") as stream:
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for member, raw in [(META_MEMBER, meta_bytes), *members]:
                info = tarfile.TarInfo(member)
                info.size = len(raw)
                archive.addfile(info, io.BytesIO(raw))
    if not storage.remote:
        os.replace(storage.path(staged), storage.path(name))


def list_tars(storage: Storage) -> list[TarName]:
    """List published v3 tars, ignoring staged and unrelated objects."""
    return sorted(
        parsed for parsed in map(parse_tar_name, storage.list("dp*__*.tar")) if parsed is not None
    )


def complete_ranges(tars: list[TarName], dp_size: int) -> list[tuple[int, int]]:
    """Return flush ranges whose inputs and targets tars exist for every teacher DP rank."""
    present: dict[tuple[int, int], set[tuple[int, str]]] = {}
    for tar in tars:
        present.setdefault((tar.start, tar.end), set()).add((tar.dp_rank, tar.kind))
    required = {(rank, kind) for rank in range(dp_size) for kind in KINDS}
    return sorted(span for span, found in present.items() if required <= found)


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
    for tar in list_tars(storage):
        if published_through <= tar.start < end:
            storage.remove(tar.name)
            removed.append(tar.name)
    if not storage.remote:
        for staged in storage.list("dp*__*.tar.*.tmp"):
            match = _STAGED_NAME.match(staged)
            target = parse_tar_name(match.group(1)) if match else None
            if target is not None and published_through <= target.start < end:
                storage.remove(staged)
                removed.append(staged)
    return removed
