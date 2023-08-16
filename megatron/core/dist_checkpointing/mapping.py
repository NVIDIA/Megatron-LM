# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Core library classes. """

from dataclasses import dataclass, replace
from itertools import chain
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .core import CheckpointingException

# These type definitions are just hints to differentiate a plain model state
#  dict (StateDict) from a state dict with tensors replaced with ShardedTensors
#  (ShardedStateDict).
StateDict = Dict[str, Any]
ShardedStateDict = Dict[str, Any]
ReplicaId = Union[int, Tuple[int, ...]]


@dataclass
class ShardedTensor:
    """Represents a mapping between a local tensor and a global tensor.

    Global tensor is assumed to consist of many local tensors distributed
    between different processes.

    Attributes:
        key: unique identifier of a global tensor
        data: local tensor data. Can be None only for consistency validation
        dtype: tensor dtype
        local_shape: local tensor shape
        global_shape: global tensor shape
        global_offset: offset of a local tensor in a global tensor, specified
            in number of tensor elements
        axis_fragmentations: global tensor fragmentation of each axis
        replica_id: indicates given local tensor's replication wrt. local
            tensors in different processes
        prepend_axis_num: number of axes prepended to the local tensor
            to reflect global tensor shape.
            The behavior is similar to unsqueezing the local tensor.
        allow_shape_mismatch: if True, during loading, the global shape of a
            stored tensor does not have to match the expected global shape.
            Useful for representing tensors with flexible shape, e.g. padded.
        flattened_range: specifies a slice that should be applied to a flattened
            tensor with `local_shape` in order to get the tensor stored as `data`
    """

    key: str
    data: Optional[torch.Tensor]
    dtype: torch.dtype
    local_shape: Tuple[int, ...]
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    axis_fragmentations: Optional[Tuple[int, ...]]
    replica_id: ReplicaId = 0
    prepend_axis_num: int = 0
    allow_shape_mismatch: bool = False
    flattened_range: Optional[slice] = None

    def global_slice(self) -> Tuple[Union[int, slice], ...]:
        assert len(self.global_offset) == len(self.local_shape) + self.prepend_axis_num
        return tuple(
            chain(
                (off for off in self.global_offset[: self.prepend_axis_num]),
                (
                    slice(off, off + sh)
                    for off, sh in zip(
                        self.global_offset[self.prepend_axis_num :], self.local_shape
                    )
                ),
            )
        )

    def global_coordinates(self) -> Tuple[np.ndarray, ...]:
        if self.flattened_range is None:
            raise CheckpointingException(
                f'`global_coordinates` is undefined for'
                f' {self.__class__.__name__} without `flattened_range`'
            )

        local_coords = self.local_coordinates()
        assert len(local_coords) + self.prepend_axis_num == len(self.global_offset), (
            len(local_coords),
            self,
        )
        global_coords = tuple(
            c + off
            for c, off in zip((0,) * self.prepend_axis_num + local_coords, self.global_offset)
        )
        return global_coords

    def local_coordinates(self) -> Tuple[np.ndarray, ...]:
        if self.flattened_range is None:
            raise CheckpointingException(
                f'`local_coordinates` is undefined for'
                f' {self.__class__.__name__} without `flattened_range`'
            )

        # TODO: np.unravel_index?
        mask = np.zeros(np.product(self.local_shape), dtype=bool)
        mask[self.flattened_range] = True
        return np.nonzero(mask.reshape(self.local_shape))

    def max_allowed_chunks(self) -> Tuple[int, ...]:
        chunks = []
        for axis_sh, axis_fragm in zip(self.global_shape, self.axis_fragmentations):
            if not self.allow_shape_mismatch and axis_sh % axis_fragm != 0:
                raise CheckpointingException(
                    f'Axis shape ({axis_sh}) not divisible' f' by axis fragmentation ({axis_fragm}'
                )
            axis_chunk_size = axis_sh // axis_fragm
            chunks.append(axis_chunk_size)
        return tuple(chunks)

    def without_data(self):
        return replace(self, data=None)

    @classmethod
    def from_rank_offsets(
        cls,
        key: str,
        data: torch.Tensor,
        *rank_offsets: Tuple[int, int, int],
        replica_id: ReplicaId = 0,
        prepend_axis_num: int = 0,
        allow_shape_mismatch: bool = False,
    ):
        """Allows to construct the ShardedTensor given offset specified in process ranks.
        Arguments:
            key: unique key
            data: local tensor data
            rank_offsets: each tuple (axis, axis_rank_offset, axis_fragm)
                says that if global tensor is divided into `axis_fragm`
                 fragment along `axis` axis, then local tensor data
                 corresponds to the `axis_rank_offset` chunk.
            replica_id: see ShardedTensor
            prepend_axis_num: see ShardedTensor
            allow_shape_mismatch: see ShardedTensor
        """
        global_offset = [0] * (data.ndim + prepend_axis_num)
        global_shape = ([1] * prepend_axis_num) + list(data.shape)
        axis_fragmentations = [1] * (data.ndim + prepend_axis_num)
        _seen_axis = set()
        for axis, axis_rank_offset, axis_fragm in rank_offsets:
            assert axis >= 0 and axis_rank_offset >= 0 and axis_fragm >= 0, (
                axis,
                axis_rank_offset,
                axis_fragm,
            )
            assert (
                axis_rank_offset < axis_fragm
            ), 'Rank offset must be lower than axis fragmentation'
            if axis in _seen_axis:
                raise CheckpointingException('Duplicated axis specified')
            _seen_axis.add(axis)

            local_axis_shape = 1 if axis < prepend_axis_num else data.shape[axis - prepend_axis_num]
            global_shape[axis] = axis_fragm * local_axis_shape
            global_offset[axis] = axis_rank_offset * local_axis_shape
            axis_fragmentations[axis] = axis_fragm

        return cls(
            key,
            data,
            data.dtype,
            tuple(data.shape),
            tuple(global_shape),
            tuple(global_offset),
            tuple(axis_fragmentations),
            replica_id,
            prepend_axis_num,
            allow_shape_mismatch,
        )

    def __str__(self):
        return f'{self.__class__.__name__}(key=\'{self.key}\')'


def is_main_replica(replica_id):
    if isinstance(replica_id, int):
        return replica_id == 0
    return all(r == 0 for r in replica_id)


class LocalNonpersitentObject:
    """Object that should not be stored in a checkpoint, but restored locally.

    Wrapping any object inside the state dict with LocalNonpersitentObject
    will result in:
    - during saving, this object will *not* be stored in the checkpoint
    - during loading, a local version of this object will be placed in a state dict
    """

    def __init__(self, obj):
        self.obj = obj

    def unwrap(self):
        return self.obj


@dataclass
class ShardedObject:
    """Represents a mapping between a local object and a global object.

    Global object is assumed to consist of many local objects distributed
    between different processes.

    NOTE: Contrary to ShardedTensor, it's impossible to change global object
    sharding. Conceptually, ShardedObject is a fully-sharded ShardedTensor
    with atomic arbitrary typed elements.

    Attributes:
        key: unique identifier of a global tensor
        data: local object data. Can be None only for consistency validation
        global_shape: global object shape
        global_offset: offset of a local object in a global object, specified
            in number of shards
        replica_id: indicates local object replication wrt. local
            objects in different processes
    """

    key: str
    data: object
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    replica_id: ReplicaId = 0

    def without_data(self):
        return replace(self, data=None)

    @property
    def unique_key(self):
        return f'{self.key}/shard_{".".join(map(str, self.global_offset))}_{".".join(map(str, self.global_shape))}'

    def __str__(self):
        return f'{self.__class__.__name__}(key=\'{self.key}\')'
