# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import os
import io
from typing import Any, Callable, cast, IO, Optional, Union
import torch
from torch import Tensor
from torch.distributed.checkpoint import FileSystemReader, Metadata
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
)
from torch.distributed._shard._utils import narrow_tensor_by_index

from torch.futures import Future
from concurrent.futures import ThreadPoolExecutor
import threading


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
    #   read_data_multi_thread
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        planner_lock = threading.Lock()

        def _read_worker(relative_path: str, reqs: list[ReadItem]) -> None:
            new_path = self.fs.concat_path(self.path, relative_path)
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(stream, item_md)
                    transform_from = self.transforms.transform_load_stream(
                        req,
                        # This field wasn't present in older
                        # implementations so provide a fallback.
                        item_md.transform_descriptors or (),
                        file_slice,
                    )

                    if req.type == LoadItemType.BYTE_IO:
                        read_bytes = io.BytesIO(transform_from.read(-1))
                        read_bytes.seek(0)
                        with planner_lock:
                            planner.load_bytes(req, read_bytes)
                    else:
                        if transform_from.seekable():
                            seekable = transform_from
                        else:
                            # torch.load requires a seekable input, so read the transform
                            # stream now and store the output if needed
                            seekable = io.BytesIO(transform_from.read(-1))
                            seekable.seek(0)

                        tensor = cast(
                            Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )

                        with planner_lock:
                            target_tensor = planner.resolve_tensor(req).detach()

                            assert target_tensor.size() == tensor.size(), (
                                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            )
                            target_tensor.copy_(tensor)
                            planner.commit_tensor(req, target_tensor)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(_read_worker, relative_path, reqs) for relative_path, reqs in per_file.items()]
            for future in futures:
                future.result()  # Raise any exceptions that occurred in the threads

        fut: Future = Future()
        fut.set_result(None)
        return fut