# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Expose the native indexed-JSONL reader as a companion-metadata auxiliary."""

from megatron.energon.epathlib import EPath
from megatron.energon.flavors.jsonl.file_store import JsonlFileStore
from megatron.energon.recipe.recipe import AuxFilesystemReference


class IndexedJsonlMetadataReference(AuxFilesystemReference):
    """Recipe protocol ``indexed-jsonl://<prepared-metadata.jsonl>``."""

    def get_file_store(self):
        if not isinstance(self.fs_path, EPath):
            raise ValueError("indexed JSONL metadata requires a resolved recipe path")
        return JsonlFileStore(self.fs_path)
