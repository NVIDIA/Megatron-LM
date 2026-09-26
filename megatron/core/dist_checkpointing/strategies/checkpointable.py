# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from itertools import chain

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    TensorProperties,
)
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType

from ..mapping import ShardedTensor


class CheckpointableShardedTensor(torch.Tensor):
    """ShardedTensor extension compatible with PyTorch DCP checkpointing library.

    Implements the torch.distributed._checkpointable._Checkpointable protocol.

    ShardedTensors with prepended axes (``prepend_axis_num > 0``, e.g. the expert axis of grouped
    MoE weights) are supported: the DCP chunk is described in global-tensor dimensions, with one
    singleton dimension per prepended axis, and the data is exposed through a view of that shape.
    This is exactly the chunk layout the torch ``ShardedTensor`` translation
    (``sharded_tensor_to_torch_sharded_tensor``) produces, so checkpoints written through either
    path are interchangeable.
    """

    def __new__(cls, data: torch.Tensor, sh_ten: ShardedTensor):
        return torch.Tensor._make_wrapper_subclass(cls, torch.Size(sh_ten.global_shape))

    def __init__(self, data: torch.Tensor, sh_ten: ShardedTensor):
        self._data = data
        self._sh_ten = sh_ten

    def _chunk_shape(self) -> torch.Size:
        """Local shard shape in global-tensor dimensions: prepended singleton axes + local_shape."""
        sh_ten = self._sh_ten
        return torch.Size((1,) * sh_ten.prepend_axis_num + tuple(sh_ten.local_shape))

    def _shard_tensor(self) -> torch.Tensor:
        """Local data as DCP expects it: a view with the prepended singleton axes (same storage)."""
        sh_ten = self._sh_ten
        data = sh_ten.data
        assert data.size() == torch.Size(sh_ten.local_shape), (data.size(), sh_ten.local_shape)
        if sh_ten.prepend_axis_num == 0:
            return data
        if not data.is_contiguous():
            sh_ten.data = data = data.contiguous()
        return data.view(self._chunk_shape())

    def __create_write_items__(
        self, fqn: str, sh_ten: 'CheckpointableShardedTensor', index: int = None
    ) -> list[WriteItem]:
        """Simple translation from ShardedTensor offsets into DCP offsets.

        Args:
            fqn (str): tensor FQN.
            sh_ten (CheckpointableShardedTensor): same as `self`
            index (int): specifies index within the LocalShardsContainer.
                This is an optimization hint used in DCP.

        Returns:
            List[WriteItem]: list of DCP WriteItem metadata objects.
        """
        offsets = torch.Size(sh_ten._sh_ten.global_offset)
        global_shape = torch.Size(sh_ten._sh_ten.global_shape)
        chunk_size = sh_ten._chunk_shape()
        assert torch.Size(sh_ten._sh_ten.local_shape) == sh_ten._sh_ten.data.size()

        return [
            WriteItem(
                index=MetadataIndex(fqn, offsets, index),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(offsets=offsets, sizes=chunk_size),
                    properties=TensorProperties.create_from_tensor(sh_ten._sh_ten.data),
                    size=global_shape,
                ),
            )
        ]

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """Simple translation from ShardedTensor offsets into DCP offsets.

        Returns:
            List[ChunkStorageMetadata]: list of DCP ChunkStorageMetadata metadata objects.
        """
        offsets = torch.Size(self._sh_ten.global_offset)
        chunk_size = self._chunk_shape()
        assert torch.Size(self._sh_ten.local_shape) == self._sh_ten.data.size()

        return [ChunkStorageMetadata(offsets=offsets, sizes=chunk_size)]

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """Yields the underlying data tensor, viewed with the prepended singleton axes if any.

        Args:
            index (MetadataIndex): unused

        Returns:
            Tensor: the underlying data tensor (a view sharing its storage when axes are prepended)
        """
        return self._shard_tensor()

    @classmethod
    def from_sh_ten(cls, sh_ten: ShardedTensor) -> 'CheckpointableShardedTensor':
        """Constructor which turns a ShardedTensor into CheckpointableShardedTensor

        Args:
            sh_ten (ShardedTensor): a sharded tensor to wrap

        Returns:
            CheckpointableShardedTensor: wrapped ShardedTensor
        """
        assert isinstance(sh_ten, ShardedTensor)
        return cls(sh_ten.data, sh_ten)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Placeholder implementation."""
        raise NotImplementedError(
            f"{cls.__name__}.__torch_dispatch__ not implemented."
            f" {cls.__name__} shouldn't be used with Tensor operations."
        )

    def __repr__(self):
        return f'{self.__class__.__name__}({self._sh_ten.__repr__()})'


class LocalShardsContainer(torch.Tensor):
    """DCP compatible container for local shards.

    PyTorch DCP requires a single tensor per rank for a given global tensor FQN.
    This class acts as a container allowing multiple checkpointable shards per rank.

    Implements the torch.distributed._checkpointable._Checkpointable protocol.
    """

    @staticmethod
    def __new__(cls, local_shards: list[torch.Tensor]) -> "LocalShardsContainer":
        assert len(local_shards) > 0
        # This assumes local shard already has correct size info
        return torch.Tensor._make_wrapper_subclass(cls, local_shards[0].size())

    def __init__(self, local_shards: list[torch.Tensor]):
        for local_shard in local_shards:
            # this is needed only for __get_tensor_shard__
            assert isinstance(local_shard, CheckpointableShardedTensor)
        self._local_shards = local_shards

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Placeholder implementation."""
        raise NotImplementedError(
            f"{cls.__name__}.__torch_dispatch__ not implemented."
            f" {cls.__name__} shouldn't be used with Tensor operations."
        )

    def __create_write_items__(
        self, fqn: str, local_shards_cont: 'LocalShardsContainer'
    ) -> list[object]:
        """Delegates creating write items to local shards.

        Args:
            fqn (str): tensor FQN.
            local_shards_cont (LocalShardsContainer): same as `self`

        Returns:
            List[WriteItem]: list of DCP WriteItem metadata objects.
        """
        return list(
            chain.from_iterable(
                shard.__create_write_items__(fqn, shard, index=index)
                for index, shard in enumerate(local_shards_cont._local_shards)
            )
        )

    def __create_chunk_list__(self) -> list[ChunkStorageMetadata]:
        """Delegates creating chunk items to local shards.

        Returns:
            List[ChunkStorageMetadata]: list of DCP ChunkStorageMetadata metadata objects.
        """
        return list(
            chain.from_iterable(shard.__create_chunk_list__() for shard in self._local_shards)
        )

    def __get_tensor_shard__(self, index: MetadataIndex) -> torch.Tensor:
        """Performs shard matching lookup based on index hint or offset.

        Args:
            index (MetadataIndex): metadata specifying the offset of the queried shard.
                Optionally provides an index hint which speeds up the lookup.

        Returns:
            Tensor: the matching shard data tensor
        """
        if index.offset is None:
            raise ValueError(
                f"Cannot lookup {index.fqn} for a LocalShardsContainer without an offset"
            )

        shards = self._local_shards
        # index hint direct lookup
        if index.index is not None:
            if (
                len(shards) > index.index
                and torch.Size(shards[index.index]._sh_ten.global_offset) == index.offset
            ):
                return shards[index.index].__get_tensor_shard__(index)

        # slow linear search
        for shard in shards:
            if torch.Size(shard._sh_ten.global_offset) == index.offset:
                return shard.__get_tensor_shard__(index)
        raise ValueError(f"Could not find shard at '{index.offset}' for FQN: '{index.fqn}'")

    def __repr__(self):
        return f'{self.__class__.__name__}({self._local_shards.__repr__()})'
