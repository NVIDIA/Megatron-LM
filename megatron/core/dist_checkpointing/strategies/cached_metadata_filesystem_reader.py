# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Share deserialized local checkpoint metadata between successive readers."""

import copy
import gc
import os
from threading import Lock
from typing import Any

from torch.distributed.checkpoint import FileSystemReader, Metadata

_FileIdentity = tuple[str, int, int, int, int, int]


class CachedMetadataFileSystemReader(FileSystemReader):
    """Cache the most recently read metadata file in each process.

    Common-state reads, shape lookups, and tensor loads can reuse the same deserialization.
    File identity and timestamps are checked on every read. Only one file is retained to bound
    memory use, and a file changed during a read is not cached.

    Each call returns a shallow copy, including independent storage metadata, so reader load IDs
    and save-side attribute assignments do not mutate the cache. Large metadata dictionaries and
    any extension attributes remain shared and must be treated as read-only by callers.
    """

    _cached_metadata: tuple[_FileIdentity, Metadata] | None = None
    _cache_lock = Lock()

    def _file_identity(self, rank: int | None) -> _FileIdentity | None:
        """Identify the metadata file, including rank-local files on newer PyTorch versions."""
        if hasattr(self, "_get_metadata_path"):
            path = self._get_metadata_path(rank)
        else:
            path = self.fs.concat_path(self.path, ".metadata")
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return (
            os.path.abspath(path),
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    def read_metadata(self, *args: Any, **kwargs: Any) -> Metadata:
        """Read metadata once while its file identity remains unchanged.

        Cyclic garbage collection is paused during deserialization to avoid repeatedly scanning
        the growing object graph. Its previous state is restored even if deserialization fails.
        """
        cls = CachedMetadataFileSystemReader
        with cls._cache_lock:
            identity = self._file_identity(kwargs.get("rank"))
            entry = cls._cached_metadata
            if identity is not None and entry is not None and entry[0] == identity:
                metadata = entry[1]
            else:
                # Release the previous checkpoint before allocating another large object graph.
                cls._cached_metadata = None
                del entry
                gc_enabled = gc.isenabled()
                try:
                    gc.disable()
                    metadata = super().read_metadata(*args, **kwargs)
                finally:
                    if gc_enabled:
                        gc.enable()
                if identity is not None and identity == self._file_identity(kwargs.get("rank")):
                    cls._cached_metadata = (identity, metadata)

            result = copy.copy(metadata)
            if getattr(metadata, "storage_meta", None) is not None:
                result.storage_meta = copy.copy(metadata.storage_meta)
                result.storage_meta.load_id = self.load_id
            return result

    @classmethod
    def clear_metadata_cache(cls, path: str | os.PathLike | None = None) -> None:
        """Release cached metadata, optionally only for the given checkpoint directory."""
        with cls._cache_lock:
            entry = cls._cached_metadata
            if path is None or (
                entry is not None and os.path.dirname(entry[0][0]) == os.path.abspath(path)
            ):
                cls._cached_metadata = None
