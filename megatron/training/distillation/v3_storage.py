# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Storage and atomic replay-group publication for offline KD V3.

A replay flush has independent ``replay-START-END.part-N.arrow`` files and one
``replay-START-END.index.json`` published after every upload succeeds. Its explicit
part paths and per-group writer layout reconstruct the teacher order without fixing
the whole cache to one teacher DP size. Unindexed parts are ignored on discovery.
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
_REPLAY_GROUP = re.compile(r"^replay-(\d+)-(\d+)\.index\.json$")
_REPLAY_OBJECT = re.compile(
    r"^replay-(\d+)-(\d+)\.(?:index\.json|part-\d+\.arrow)(?:\.[0-9a-f]{32}\.tmp)?$"
)


class GroupName(NamedTuple):
    """An atomically published replay flush, addressed in global sample positions."""

    start: int
    end: int

    @property
    def index(self) -> str:
        """The consolidated index published after all writer uploads."""
        return f"replay-{self.start}-{self.end}.index.json"


def replay_data_name(writer: int, start: int, end: int) -> str:
    """Name one independent replay part; consumers use its explicit index path."""
    return f"replay-{start}-{end}.part-{writer:06d}.arrow"


class FileName(NamedTuple):
    """A parsed v3 file name."""

    dp_rank: int
    start: int
    end: int

    @property
    def data(self) -> str:
        """The data file's name."""
        return replay_data_name(self.dp_rank, self.start, self.end)

    @property
    def index(self) -> str:
        """The index's name."""
        return GroupName(self.start, self.end).index


def data_name(dp_rank: int, start: int, end: int) -> str:
    """Name the data file holding one writer's flush range."""
    return replay_data_name(dp_rank, start, end)


def index_name(dp_rank: int, start: int, end: int) -> str:
    """Name the index published after that data file."""
    return GroupName(start, end).index


def parse_name(name: str) -> FileName | None:
    """Parse a v3 data file or index name, or return None for other objects."""
    match = re.fullmatch(r"replay-(\d+)-(\d+)\.part-(\d+)\.arrow", name)
    if match is not None:
        start, end, writer = match.groups()
        return FileName(int(writer), int(start), int(end))
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
        """List fresh object names relative to the root; never cache a growing listing."""
        paths = (
            self.backend.glob(self.path(pattern)) if self.remote else glob.glob(self.path(pattern))
        )
        prefix = self.path("")
        return sorted(
            (
                str(path)[len(prefix) :]
                if str(path).startswith(prefix)
                else os.path.basename(str(path))
            )
            for path in paths
        )

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
        os.makedirs(os.path.dirname(self.path(name)), exist_ok=True)
        staged = f"{name}.{uuid.uuid4().hex}.tmp"
        with self.open(staged, "wb") as stream:
            stream.write(data)
        os.replace(self.path(staged), self.path(name))


def read_index(storage: Storage, name: FileName | GroupName) -> dict[str, Any]:
    """Read a group, or select one writer's Arrow index from the group."""
    group = read_group(storage, GroupName(name.start, name.end))
    return group["parts"][name.dp_rank]["index"] if isinstance(name, FileName) else group


def read_group(storage: Storage, name: GroupName) -> dict[str, Any]:
    """Read and validate one complete replay group with one GET, without HEADs."""
    from .format.arrow import check_index, file_meta
    from .format.contract import mismatched_settings, validate_metadata

    group = json.loads(storage.read(name.index))
    if group.get("group_version") != 1 or (group.get("start"), group.get("end")) != tuple(name):
        raise ValueError(f"Invalid replay group header {name.index}")
    writers, mbs = group.get("writers"), group.get("writer_mbs")
    if type(writers) is not int or writers < 1 or type(mbs) is not int or mbs < 1:
        raise ValueError(f"Invalid replay writer layout in {name.index}")
    if name.start < 0 or name.end <= name.start or (name.end - name.start) % (writers * mbs):
        raise ValueError(f"Replay group {name.index} is not whole writer microbatches")
    if len(group.get("parts", [])) != writers:
        raise ValueError(f"Replay group {name.index} is missing writer parts")
    parts, seen, reference = [None] * writers, set(), None
    for part in group["parts"]:
        writer, path, index = part.get("writer"), part.get("path"), part.get("index")
        if type(writer) is not int or not 0 <= writer < writers or parts[writer] is not None:
            raise ValueError(f"Invalid or duplicate replay writer in {name.index}")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ":" in path
            or "\\" in path
            or any(p in ("", ".", "..") for p in path.split("/"))
            or path in seen
        ):
            raise ValueError(f"Invalid or duplicate replay part path in {name.index}")
        seen.add(path)
        check_index(index)
        cursor, byte_end = 0, 0
        for block in index["blocks"]:
            if block["first_row"] != cursor or block["rows"] <= 0:
                raise ValueError(f"Invalid replay block rows in {path}")
            if (
                block["offset"] < byte_end
                or block["length"] <= 0
                or block["offset"] + block["length"] > index["size"]
                or not 0 < block["input_length"] <= block["length"]
            ):
                raise ValueError(f"Invalid replay byte ranges in {path}")
            cursor += block["rows"]
            byte_end = block["offset"] + block["length"]
        if cursor != (name.end - name.start) // writers:
            raise ValueError(f"Wrong row count in replay part {path}")
        metadata = file_meta(index)
        validate_metadata(metadata)
        if reference is not None and mismatched_settings(metadata, reference)[0]:
            raise ValueError(f"Replay group {name.index} mixes incompatible file contracts")
        reference = metadata
        parts[writer] = part
    group["parts"] = parts
    return group


def read_meta(storage: Storage, name: GroupName | FileName) -> dict[str, Any]:
    """Read a replay group's shared metadata from its first part footer."""
    from .format.arrow import check_index, file_meta

    if isinstance(name, GroupName):
        return file_meta(read_group(storage, name)["parts"][0]["index"])
    index = read_index(storage, name)
    check_index(index)
    return file_meta(index)


def has_legacy_tars(storage: Storage) -> bool:
    """Whether the directory holds V1/V2 tar dumps, which V3 never mixes with."""
    return bool(storage.list("*.tar"))


def list_published(storage: Storage, *, objects: list[str] | None = None) -> list[GroupName]:
    """List atomically published groups; incomplete writer uploads have no index."""
    groups = []
    for path in storage.list("replay-*.index.json") if objects is None else objects:
        match = _REPLAY_GROUP.fullmatch(path)
        if match is not None:
            groups.append(GroupName(*(int(value) for value in match.groups())))
    return sorted(groups)


def complete_ranges(
    published: list[GroupName], writers: int | None = None
) -> list[tuple[int, int]]:
    """A group index already establishes that all of its writers completed."""
    return [(name.start, name.end) for name in published]


def validate_published_ranges(
    published: list[GroupName], writers: int | None = None, writer_mbs: int | None = None
) -> list[tuple[int, int]]:
    """Reject overlapping groups; each group's own index validates its writer layout."""
    previous = None
    for name in sorted(published):
        if (
            name.start < 0
            or name.end <= name.start
            or previous is not None
            and name.start < previous
        ):
            raise ValueError(f"Invalid or overlapping replay group {name.index}")
        previous = name.end
    return complete_ranges(published)


def contiguous_end(ranges: list[tuple[int, int]], start: int, end: int | None = None) -> int:
    """Return the end of the contiguous run beginning at start."""
    by_start = dict(ranges)
    cursor = start
    while cursor in by_start and (end is None or cursor < end):
        cursor = by_start[cursor]
    return cursor


def object_ranges(storage: Storage, *, objects: list[str] | None = None) -> list[tuple[int, int]]:
    """Ranges of replay group indexes, parts and incomplete staged writes."""
    spans = set()
    for name in storage.list("replay-*") if objects is None else objects:
        match = _REPLAY_OBJECT.fullmatch(name)
        if match is not None:
            spans.add(tuple(int(value) for value in match.groups()))
    return sorted(spans)


def discard_range(
    storage: Storage, start: int, end: int, *, objects: list[str] | None = None
) -> list[str]:
    """Unpublish selected groups first, then remove their parts before replacement."""
    removed = []
    for name in sorted(
        storage.list("replay-*") if objects is None else objects,
        key=lambda name: not name.endswith(".index.json"),
    ):
        match = _REPLAY_OBJECT.fullmatch(name)
        if match is not None and start <= int(match[1]) < end:
            storage.remove(name)
            removed.append(name)
    return removed
