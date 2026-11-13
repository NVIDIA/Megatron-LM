# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Independent corpus parts, published together by a consolidated chunk index.

Megatron names chunks ``chunk-<start>-<end>`` by teacher iteration and parts
``part-<rank>.arrow`` by the writer's DP rank (see ``v3_storage``). Consumers follow
the explicit paths in the index and never interpret that layout. A snapshot
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

CHUNK_VERSION = 1


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
    """A producer's view of its own source, listed once and read once at startup.

    ``objects`` is every object under its chunk directories (published or not);
    ``chunks`` maps each published chunk directory to its index. Students plan from the
    chunk indexes directly (see :mod:`.blend`).
    """

    def __init__(self, storage: Storage):
        self.storage = storage
        self.objects = storage.list("chunk-*/*")
        self.chunks = {
            os.path.dirname(path): json.loads(storage.read(path))
            for path in self.objects
            if path.count("/") == 1 and path.endswith("/index.json")
        }
        self.metadata = None
        for directory, chunk in sorted(self.chunks.items()):
            relative_path(directory)
            metadata = check_chunk(chunk)
            if self.metadata is not None:
                differing = mismatched_settings(metadata, self.metadata)[0]
                differing += mismatched_corpus_order(metadata, self.metadata)
                if differing:
                    raise ValueError(f"Corpus source mixes settings: {', '.join(differing)}")
            self.metadata = metadata

    def add(self, directory: str, chunk: dict[str, Any]) -> None:
        """Record our own successful publication without listing or rereading it."""
        self.metadata = check_chunk(chunk)
        self.chunks[directory] = chunk
