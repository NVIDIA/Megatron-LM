# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

""" FS Reader with metadata cached support. """

import io
import os
from typing import Dict, Union, cast

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import FileSystemReader, Metadata
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.planner import LoadItemType, LoadPlan, LoadPlanner, ReadItem
from torch.futures import Future


class CachedMetadataFileSystemReader(FileSystemReader):
    """
    Extends FileSystemReader to cache metadata for improved performance.

    Metadata is shared across all reader instances that use the same checkpoint
    directory (same path), since the loaded metadata is identical.

    Attributes:
        _metadata_cache (Dict[str, Metadata]): Class-level cache keyed by checkpoint path.
    """

    _metadata_cache: Dict[str, Metadata] = {}

    def __init__(self, path: Union[str, os.PathLike], cache_metadata: bool = True) -> None:
        """
        Initialize with file system path.

        Args:
            path (Union[str, os.PathLike]): Path to the checkpoint directory or file.
        """
        super().__init__(path=path)
        self._cache_key = os.path.abspath(os.fspath(path)) if cache_metadata else None

    def read_metadata(self) -> Metadata:
        """
        Read metadata from file system, caching for subsequent calls.
        Shared across instances when the checkpoint directory is the same.

        Returns:
            Metadata: Checkpoint metadata.
        """
        if self._cache_key not in CachedMetadataFileSystemReader._metadata_cache:
            CachedMetadataFileSystemReader._metadata_cache[self._cache_key] = (
                super().read_metadata()
            )
        return CachedMetadataFileSystemReader._metadata_cache[self._cache_key]

    @classmethod
    def clear_metadata_cache(cls):
        """
        Clear the metadata cache.
        """
        cls._metadata_cache.clear()

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: dict[str, list[ReadItem]] = {}
        for read_item in plan.items:
            item_md: _StorageInfo = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        rank = int(os.environ.get('RANK', 0))

        # Per-rank disk-I/O accounting for the local-replica change. The
        # PR only relocates *which* file each rank reads from, not *how
        # many* bytes — these counters let an operator confirm exactly
        # that on a real run by diff'ing two log lines (legacy load vs
        # local-read load) over the same on-disk checkpoint. We sum the
        # ``length`` field of every ``_StorageInfo`` we are about to
        # consume; tensors are counted as one per ``ReadItem`` regardless
        # of underlying type (BYTE_IO or tensor). No collectives — each
        # rank prints only what it sees.
        local_total_bytes = 0
        local_total_items = 0
        for read_items in per_file.values():
            for req in read_items:
                local_total_bytes += self.storage_data[req.storage_index].length
                local_total_items += 1
        print(
            f"[DEBUG-TP-REP] [Rank {rank}] read_data: "
            f"items={local_total_items} bytes={local_total_bytes} "
            f"files={len(per_file)}"
        )

        for relative_path, reqs in per_file.items():
            new_path = self.fs.concat_path(self.path, relative_path)
            # Extract the rank number from the checkpoint shard filename (e.g., "__1_0.distcp" --> 1)
            # Assumes the path ends with "__<rank>_<tp>.distcp"
            import re
            match = re.search(r"__(\d+)_\d+\.distcp$", str(new_path))
            file_path_rank = int(match.group(1)) if match else None
            
            if file_path_rank != rank:
                print(f"[DEBUG-TP-REP] [Rank {rank}] Cross read data from {new_path} (rank {file_path_rank})")
            if len(reqs) == 0:
                continue
            #print(f"[DEBUG-TP-REP] [Rank {rank}] Reading data from {new_path}")
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO sort by offset and cache the reading
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    #if file_path_rank != rank:
                    print(f"[DEBUG-TP-REP] [Rank {rank}] Reading item {req.storage_index} from {file_path_rank} ({new_path}) (type: {req.type})")
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
                            torch.Tensor,
                            torch.load(
                                seekable,
                                map_location="cpu",
                                weights_only=True,
                            ),
                        )
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        target_tensor = planner.resolve_tensor(req).detach()

                        if target_tensor.size() != tensor.size():
                            raise AssertionError(
                                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            )
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut
