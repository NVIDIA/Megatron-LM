# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import os
from typing import Union

from torch.distributed.checkpoint import FileSystemReader, Metadata


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Attributes:
        _cached_metadata (Metadata or None): Cached metadata from the file system.
    """

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cached_metadata = None

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cached_metadata is None:
            self._cached_metadata = super().read_metadata()
        return self._cached_metadata
