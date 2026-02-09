# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import os
from typing import Dict, Union

from torch.distributed.checkpoint import FileSystemReader, Metadata


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is shared across all reader instances that use the same checkpoint
    directory (same path), since the loaded metadata is identical.

    Attributes:
        _metadata_cache (Dict[str, Metadata]): Class-level cache keyed by checkpoint path.
    """

    _metadata_cache: Dict[str, Metadata] = {}

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cache_key = os.path.abspath(os.fspath(path))

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory is the same.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cache_key not in CachedMetadataFileSystemReader._metadata_cache:
            CachedMetadataFileSystemReader._metadata_cache[
                self._cache_key
            ] = super().read_metadata()
        return CachedMetadataFileSystemReader._metadata_cache[self._cache_key]
