# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Energon adapters for an immutable local compact-range catalog.

The SQLite/binary layouts are existing storage formats documented in
MULTIMODAL_INTERLEAVED.md. Namespace aliases, corpus constants and loader policy are
not part of this reader. Logical coordinates are explicit recipe configuration.
"""

import json
from pathlib import Path

from megatron.energon import AuxFileStoreReference
from megatron.energon.epathlib import EPath

from .document_image_ranges import CompactImageCatalog
from .document_image_store import DocumentImageStore


class CompactDocumentImageStore(DocumentImageStore):
    """Use local document-offset sidecars but read images only from compact packs."""

    def __init__(self, config_path: Path, *, validation_run_id: str | None = None) -> None:
        if config_path.stat().st_size > 64 << 10:
            raise ValueError("compact image configuration is oversized")
        config = json.loads(config_path.read_text())
        base = config_path.resolve(strict=True).parent
        super().__init__(base / config["offset_root"])
        self.prefix = config["logical_prefix"]
        if self.prefix and (
            self.prefix.startswith("/")
            or not self.prefix.endswith("/")
            or any(part in ("", ".", "..") for part in self.prefix[:-1].split("/"))
        ):
            raise ValueError("compact logical prefix must be empty or a safe relative prefix")
        self.ranges = CompactImageCatalog(
            base / config["catalog"], config["dataset_id"], validation_run_id=validation_run_id
        )

    def _payload(self, ref):
        relative, offset, size = self._coordinates(ref)
        return self.ranges.read_many(
            self.prefix + relative, [(offset, size, ref.get("tar_member", False))]
        )[0]


class CompactImagesReference(AuxFileStoreReference):
    """Recipe protocol ``compact-document-images://<local-config.json>``."""

    def __init__(self, path) -> None:
        self.path = path

    def _resolve_path(self, recipe_path):
        if not isinstance(self.path, EPath):
            if recipe_path is None:
                raise ValueError("compact configuration requires a recipe path")
            self.path = recipe_path.parent / self.path
        if not self.path.is_local():
            raise ValueError("compact image configuration must be local")
        return self.path

    def get_traversed_path(self):
        return self.path

    def get_file_store(self):
        return CompactDocumentImageStore(Path(str(self.path)))
