# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Neutral local document-relative image ranges through Energon's aux-store API.

Offsets are read positionally, with no fork-inherited mutable file pointer. This
module reads an existing binary offset-table format; it never builds or scans it.
"""

import json
import struct
from pathlib import PurePosixPath

from megatron.energon import AuxFileStoreReference, FileStore, SourceInfo
from megatron.energon.epathlib import EPath
from megatron.energon.recipe import register_aux_filestore_protocol

from .document_image_ranges import LocalImageRanges, extract_tar_image_payload


class DocumentImageStore(LocalImageRanges, FileStore):
    """Read image bytes under a local root; Energon's decoder owns pixel decoding."""

    def _coordinates(self, ref: dict) -> tuple[str, int, int]:
        relative = ref["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise ValueError("image coordinates must be contained relative paths")
        offset, size = ref["offset"], ref["size"]
        if type(offset) is not int or type(size) is not int or offset < 0 or size <= 0:
            raise ValueError("image extents require a nonnegative offset and positive size")
        if "document_index" in ref:
            index = ref["document_index"]
            if type(index) is not int or index < 0:
                raise ValueError("document index must be a nonnegative integer")
            index_path = self._path(ref["path"] + ".msidx")
            if index_path.stat().st_size % 8:
                raise ValueError("document offset table has an invalid size")
            base = struct.unpack("<Q", self._read(index_path, index * 8, 8))[0]
            if base == 0:
                raise ValueError("document has no indexed image payload")
            offset += base
        return relative, offset, size

    def _payload(self, ref: dict) -> bytes:
        relative, offset, size = self._coordinates(ref)
        return self._read(self._path(relative), offset, size)

    def __getitem__(self, key: str):
        ref = json.loads(key)
        payload = self._payload(ref)
        wrapped = ref.get("tar_member", False)
        if type(wrapped) is not bool:
            raise ValueError("tar_member must be a boolean")
        if wrapped:
            payload = extract_tar_image_payload(payload)
        return payload, SourceInfo(
            dataset_path=self.get_path(), index=key, shard_name=ref["path"], file_names=(key,)
        )

    def get_path(self) -> str:
        return str(self.root)


class DocumentImagesReference(AuxFileStoreReference):
    """Recipe reference for ``document-images://<local-root>``."""

    def __init__(self, root) -> None:
        self.root = root

    def _resolve_path(self, recipe_path: EPath | None) -> EPath:
        if not isinstance(self.root, EPath):
            if recipe_path is None:
                raise ValueError("relative image root requires the recipe path")
            self.root = recipe_path.parent / self.root
        if not self.root.is_local():
            raise ValueError("document image store currently supports local roots only")
        return self.root

    def get_file_store(self) -> FileStore:
        return DocumentImageStore(str(self.root))

    def get_traversed_path(self) -> EPath:
        return self.root


def register_document_image_store() -> None:
    """Register only when the interleaved provider is selected; no import-time registration."""
    from .compact_document_images import CompactImagesReference

    register_aux_filestore_protocol("document-images", DocumentImagesReference, override=True)
    register_aux_filestore_protocol(
        "compact-document-images", CompactImagesReference, override=True
    )
