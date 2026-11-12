# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent corpus parts, published together by a consolidated chunk index.

Consumers know paths and Arrow rows, never the producer's rank layout. A snapshot
lists once and reads each chunk index once; all subsequent planning and decoding
uses those cached indexes. Only producers interpret the optional resume information.
"""

import json
import os
from pathlib import PurePosixPath
from typing import Any

from ..v3_storage import Storage
from .arrow import check_index, file_meta
from .contract import mismatched_corpus_order, mismatched_settings, validate_metadata
from .inventory import ChunkInventory

CHUNK_VERSION = 1


def chunk_path(iteration: int) -> str:
    """Name a Megatron chunk by its starting teacher iteration."""
    return f"chunk-{iteration:012d}"


def part_path(rank: int) -> str:
    """Name a part; the ordinal has no rank semantics for consumers."""
    return f"part-{rank:06d}.arrow"


def relative_path(path: str) -> str:
    """Reject paths that could escape a corpus source or alias another object."""
    if (
        not isinstance(path, str)
        or not path
        or "\\" in path
        or ":" in path
        or PurePosixPath(path).is_absolute()
        or any(piece in ("", ".", "..") for piece in path.split("/"))
    ):
        raise ValueError(f"Invalid corpus relative path {path!r}")
    return path


def check_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Validate all part indexes and return their common file metadata."""
    if chunk.get("chunk_version") != CHUNK_VERSION or not chunk.get("parts"):
        raise ValueError("Corpus chunk needs a supported chunk_version and nonempty parts")
    reference, paths = None, set()
    for part in chunk["parts"]:
        path = relative_path(part["path"])
        if path in paths:
            raise ValueError(f"Duplicate corpus part {path}")
        paths.add(path)
        index = part["index"]
        check_index(index)
        cursor = 0
        for block in index["blocks"]:
            if block["first_row"] != cursor or block["rows"] <= 0:
                raise ValueError(f"Invalid corpus block rows in {path}")
            cursor += block["rows"]
            if (
                block["offset"] < 0
                or block["length"] <= 0
                or block["offset"] + block["length"] > index["size"]
                or not 0 < block["input_length"] <= block["length"]
            ):
                raise ValueError(f"Invalid corpus block byte range in {path}")
        if not cursor:
            raise ValueError(f"Empty corpus part {path}")
        metadata = file_meta(index)
        validate_metadata(metadata, replay=False, corpus=True)
        if reference is not None:
            differing = mismatched_settings(metadata, reference)[0]
            differing += mismatched_corpus_order(metadata, reference)
            if differing:
                raise ValueError(f"Corpus chunk mixes settings: {', '.join(differing)}")
        reference = metadata
    return reference


def publish_chunk(storage: Storage, directory: str, chunk: dict[str, Any]) -> None:
    """Publish an index only after every referenced part has been written."""
    check_chunk(chunk)
    storage.write(
        f"{relative_path(directory)}/index.json",
        json.dumps(chunk, sort_keys=True, separators=(",", ":")).encode(),
    )


class CorpusInventory:
    """One explicit snapshot of a source's objects and consolidated indexes.

    Producer metadata is spooled to local scratch storage and never broadcast.
    Mutating a producer's inventory never refreshes a student view.
    """

    def __init__(self, storage: Storage, snapshot: dict[str, Any] | None = None):
        self.storage = storage
        if snapshot is None:
            objects = storage.list("chunk-*/*")
            chunks = ChunkInventory()
            for path in objects:
                if path.endswith("/index.json"):
                    chunks[os.path.dirname(path)] = json.loads(storage.read(path))
            snapshot = {"objects": objects, "chunks": chunks}
        self.snapshot = snapshot
        self.chunks = snapshot["chunks"]
        self.metadata = None
        for directory in self.chunks:
            chunk = self.chunks[directory]
            relative_path(directory)
            metadata = check_chunk(chunk)
            if self.metadata is not None:
                differing = mismatched_settings(metadata, self.metadata)[0]
                differing += mismatched_corpus_order(metadata, self.metadata)
                if differing:
                    raise ValueError(f"Corpus source mixes settings: {', '.join(differing)}")
            self.metadata = metadata

    def parts(self) -> list[dict[str, Any]]:
        """Return source-relative paths and cached indexes in stable storage order."""
        result, seen = [], set()
        for directory in sorted(self.chunks):
            chunk = self.chunks[directory]
            for part in chunk["parts"]:
                path = relative_path(f"{directory}/{part['path']}")
                if path in seen:
                    raise ValueError(f"Duplicate corpus part {path}")
                seen.add(path)
                result.append({"path": path, "index": part["index"]})
        return result

    def add(self, directory: str, chunk: dict[str, Any]) -> None:
        """Record our own successful publication without listing or rereading it."""
        self.metadata = check_chunk(chunk)
        new = directory not in self.chunks
        self.chunks[directory] = chunk
        if new:
            self.snapshot["objects"].extend(
                [
                    f"{directory}/index.json",
                    *[f"{directory}/{part['path']}" for part in chunk["parts"]],
                ]
            )

    def invalidate(self, directories: set[str]) -> None:
        """Unpublish a producer's selected tail before any part is overwritten.

        Old data can remain: readers only follow indexes, and replacements overwrite
        known paths. Unreferenced parts from interrupted attempts are harmless.
        """
        for directory in sorted(directories):
            if directory in self.chunks:
                self.storage.remove(f"{directory}/index.json")
                del self.chunks[directory]
        self.snapshot["objects"] = [
            path for path in self.snapshot["objects"] if os.path.dirname(path) not in directories
        ]
