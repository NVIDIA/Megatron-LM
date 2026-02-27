# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import logging
import os
import pickle
from typing import Iterable, Union

import msgpack
from torch.distributed.checkpoint import FileSystemReader, Metadata

logger = logging.getLogger(__name__)


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

        If a .metadata_offset file exists alongside .metadata, it will be loaded
        and combined with the base metadata. This allows for faster checkpoint saving
        by splitting the metadata into a reusable base part and a smaller offset part.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cached_metadata is None:
            self._cached_metadata = super().read_metadata()

            # Check if .metadata_offset exists
            metadata_offset_path = os.path.join(self.path, ".metadata_offset")
            if os.path.exists(metadata_offset_path):
                # Load the offset metadata (now stored as storage_md dict)
                storage_md = None
                with open(metadata_offset_path, "rb") as f:
                    # Use msgpack to deserialize the wrapper structure
                    # Note: use_bin_type is only for packb(), not unpackb()
                    # raw=False tells msgpack to decode bytes to str (default in msgpack 1.0+)
                    data = msgpack.unpackb(f.read())

                    # Handle both chunked and non-chunked formats
                    if isinstance(data, Iterable) and len(data) == 2 and data[0] == 'chunked':
                        # Chunked format: ('chunked', [pickled_chunk1, pickled_chunk2, ...])
                        logger.debug(f"chunked format")
                        _, pickled_chunks = data
                        storage_md = {}
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=len(pickled_chunks)) as chunk_executor:
                            chunks = list(chunk_executor.map(pickle.loads, pickled_chunks))
                        for chunk in chunks:
                            storage_md.update(chunk)
                    else:
                        # Non-chunked format: direct dictionary
                        storage_md = data
                # Combine storage_md with the base metadata's storage_data
                if hasattr(self._cached_metadata, 'storage_data'):
                    # Update existing storage_data with the offset
                    if self._cached_metadata.storage_data is None:
                        self._cached_metadata.storage_data = storage_md
                    else:
                        self._cached_metadata.storage_data.update(storage_md)
                else:
                    # Set storage_data if it doesn't exist
                    self._cached_metadata.storage_data = storage_md

        return self._cached_metadata
