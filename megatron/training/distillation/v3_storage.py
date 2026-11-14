# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Storage and flush publication for offline KD V3.

Both dump modes publish one directory per flush, named by its teacher iteration range:
``<kind>-<start>-<end>/`` (12-digit iterations; ``replay`` or ``chunk``) holds one
``part-<rank>.arrow`` per teacher DP writer and an ``index.json`` written only after
every part has been uploaded. A missing or corrupt part names the DP rank that wrote it.
"""

import glob
import importlib
import json
import os
import re
import uuid
from typing import Any

REPLAY = "replay"
CHUNK = "chunk"
INDEX = "index.json"
_FLUSH_DIR = re.compile(r"^(replay|chunk)-(\d{12})-(\d{12})$")


def flush_dir(kind: str, start: int, end: int) -> str:
    """Name a flush directory by its teacher iteration range ``[start, end)``."""
    return f"{kind}-{start:012d}-{end:012d}"


def part_name(rank: int) -> str:
    """Name a teacher DP writer's part within its flush directory."""
    return f"part-{rank:06d}.arrow"


def parse_flush_dir(name: str, kind: str) -> tuple[int, int] | None:
    """Return the iteration range of a ``kind`` flush directory name, or None."""
    match = _FLUSH_DIR.fullmatch(name)
    if match is None or match[1] != kind:
        return None
    return int(match[2]), int(match[3])


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


def flush_objects(storage: Storage, kind: str) -> list[str]:
    """List every object in ``kind`` flush directories, published or not (one listing)."""
    return storage.list(f"{kind}-*/*")


def object_spans(objects: list[str], kind: str) -> list[tuple[int, int]]:
    """Iteration ranges of the flush directories that hold any of ``objects``."""
    spans = {parse_flush_dir(path.split("/", 1)[0], kind) for path in objects if "/" in path}
    return sorted(span for span in spans if span is not None)


def published_spans(objects: list[str], kind: str) -> list[tuple[int, int]]:
    """Iteration ranges of published flushes (those with an index), checked for overlap."""
    spans = sorted(
        span
        for path in objects
        if path.count("/") == 1
        and path.endswith("/" + INDEX)
        and (span := parse_flush_dir(path.split("/", 1)[0], kind)) is not None
    )
    previous = 0
    for start, end in spans:
        if end <= start or start < previous:
            raise ValueError(
                f"Invalid or overlapping offline KD flush {flush_dir(kind, start, end)}"
            )
        previous = end
    return spans


def discard_range(
    storage: Storage, kind: str, objects: list[str], start: int, end: int
) -> list[tuple[int, int]]:
    """Unpublish and delete the flushes starting in ``[start, end)`` before rewriting them.

    Indexes go first, so an interrupted cleanup leaves only unpublished parts. Only this
    job's range is touched, so parallel dump jobs never interfere.
    """
    spans = [span for span in object_spans(objects, kind) if start <= span[0] < end]
    doomed = {flush_dir(kind, *span) for span in spans}
    victims = [path for path in objects if path.split("/", 1)[0] in doomed]
    for path in sorted(victims, key=lambda path: not path.endswith("/" + INDEX)):
        storage.remove(path)
    if not storage.remote:
        for directory in doomed:
            try:
                os.rmdir(storage.path(directory))
            except OSError:
                pass
    return spans


def read_group(storage: Storage, span: tuple[int, int]) -> dict[str, Any]:
    """Read and validate one published replay flush index with one GET."""
    from .format.arrow import check_index, file_meta
    from .format.contract import mismatched_settings, validate_metadata

    name = f"{flush_dir(REPLAY, *span)}/{INDEX}"
    group = json.loads(storage.read(name))
    if group.get("group_version") != 1 or (
        group.get("start_iteration"),
        group.get("end_iteration"),
    ) != tuple(span):
        raise ValueError(f"Invalid replay flush index {name}")
    writers, mbs = group.get("writers"), group.get("writer_mbs")
    if type(writers) is not int or writers < 1 or type(mbs) is not int or mbs < 1:
        raise ValueError(f"Invalid replay writer layout in {name}")
    if len(group.get("parts", [])) != writers:
        raise ValueError(f"Replay flush {name} does not list one part per writer")
    parts, reference = [None] * writers, None
    for part in group["parts"]:
        writer, index = part.get("writer"), part.get("index")
        if type(writer) is not int or not 0 <= writer < writers or parts[writer] is not None:
            raise ValueError(f"Invalid or duplicate replay writer in {name}")
        if part.get("path") != part_name(writer):
            raise ValueError(
                f"Replay flush {name} lists an unexpected part path for writer {writer}"
            )
        check_index(index)
        cursor, byte_end = 0, 0
        for block in index["blocks"]:
            if (
                block["first_row"] != cursor
                or block["rows"] <= 0
                or block["offset"] < byte_end
                or block["length"] <= 0
                or block["offset"] + block["length"] > index["size"]
                or not 0 < block["input_length"] <= block["length"]
            ):
                raise ValueError(f"Invalid block rows or byte ranges for writer {writer} in {name}")
            cursor += block["rows"]
            byte_end = block["offset"] + block["length"]
        metadata = file_meta(index)
        validate_metadata(metadata)
        if reference is not None and mismatched_settings(metadata, reference)[0]:
            raise ValueError(f"Replay flush {name} mixes incompatible file contracts")
        reference = metadata
        gbs = metadata["contract"]["global_batch_size"]
        rows = (span[1] - span[0]) * gbs
        # Each teacher iteration splits evenly into the writers' microbatches.
        if gbs % (writers * mbs) or sum(b["rows"] for b in index["blocks"]) != rows // writers:
            raise ValueError(f"Replay part {name} writer {writer} has the wrong number of samples")
        parts[writer] = part
    group["parts"] = parts
    return group


def read_meta(storage: Storage, span: tuple[int, int]) -> dict[str, Any]:
    """Read a replay flush's file settings from its first part's footer copy."""
    from .format.arrow import file_meta

    return file_meta(read_group(storage, span)["parts"][0]["index"])


def has_legacy_tars(storage: Storage) -> bool:
    """Whether the directory holds V1/V2 tar dumps, which V3 never mixes with."""
    return bool(storage.list("*.tar"))
