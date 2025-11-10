# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Core library classes for representing sharding of tensors and objects.

The main expected usage is wrapping torch.Tensors in state dicts with
ShardedTensor class (mostly with the ShardedTensor.from_rank_offsets classmethod).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .core import CheckpointingException
from .dict_utils import dict_list_map_inplace

logger = logging.getLogger(__name__)

# These type definitions are just hints to differentiate a plain model state
#  dict (StateDict) from a state dict with tensors replaced with ShardedTensors
#  (ShardedStateDict).
StateDict = Dict[str, Any]
CommonStateDict = Dict[str, Any]
ShardedStateDict = Dict[str, Any]
ReplicaId = Union[int, Tuple[int, ...]]


_logged_deprecations = {}


class ShardedBase(ABC):
    """Base class for ShardedTensor and ShardedStateDict."""

    key: str
    data: object
    replica_id: ReplicaId

    @abstractmethod
    def validate_metadata_integrity(self):
        """Codifies the constraints on metadata attributes."""

    @abstractmethod
    def without_data(self) -> "ShardedBase":
        """Returns a new ShardedBase instance with data=None."""
        raise NotImplementedError


@dataclass
class ShardedTensor(ShardedBase):
    """Represents a mapping between a local tensor and a global tensor.

    Global tensor is assumed to consist of many local tensors distributed
    between different processes.

    Args:
        key: unique identifier of a global tensor
        data: local tensor data. Can be None only for consistency validation
        dtype: tensor dtype
        local_shape: local tensor shape
        global_shape: global tensor shape
        global_offset: offset of a local tensor in a global tensor,
            specified in number of tensor elements
        axis_fragmentations: global tensor fragmentation of each axis
        replica_id: indicates given local tensor's replication wrt.
            local tensors in different processes
        prepend_axis_num: number of axes prepended to the local tensor to
            reflect global tensor shape. The behavior is similar to
            unsqueezing the local tensor.
        allow_shape_mismatch: if True, during loading, the global shape of
            a stored tensor does not have to match the expected global shape.
            Useful for representing tensors with flexible shape,
            e.g. padded.
        flattened_range: specifies a slice that should be applied to a
            flattened tensor with `local_shape` in order to get
            the tensor stored as `data`
    """

    key: str
    data: Optional[torch.Tensor] = field(repr=False)
    dtype: torch.dtype
    local_shape: Tuple[int, ...]
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    axis_fragmentations: Optional[Tuple[int, ...]]
    replica_id: ReplicaId = 0
    prepend_axis_num: int = 0
    allow_shape_mismatch: bool = False
    flattened_range: Optional[slice] = None

    def __post_init__(self):
        self.validate_metadata_integrity()

    def validate_metadata_integrity(self) -> None:
        """Codifies the constraints on metadata attributes.

        Meeting those constraints is guaranteed when instantiating a ShardedTensor
        class with `from_rank_offsets` or `from_rank_offsets_flat` constructors.

        Returns:
            None
        """
        has_flattened_range = self.flattened_range is not None
        if self.data is not None:
            if self.data.dtype != self.dtype:
                raise CheckpointingException(
                    f"Data dtype should match `dtype` attribute for {self}"
                )
            if not has_flattened_range and self.data.shape != self.local_shape:
                raise CheckpointingException(
                    f"Data shape should match `local_shape` attribute for {self}"
                )

        if len(self.global_shape) != len(self.global_offset):
            raise CheckpointingException(
                f"Global offset dimensions should be equal to global shape dimensions for {self}"
            )
        if len(self.local_shape) + self.prepend_axis_num != len(self.global_shape):
            raise CheckpointingException(
                f"Local shape together with `prepend_axis_num` dimensions should be "
                f"equal to global shape dimensions for {self}"
            )

        if self.axis_fragmentations is not None:
            for off, sh in zip(self.global_offset[self.prepend_axis_num :], self.local_shape):
                if sh != 0 and off % sh != 0:
                    raise CheckpointingException(
                        f"Global offset ({off}) must be divisible by local shape ({sh}) for {self}."
                    )

        if self.flattened_range is not None:
            raise CheckpointingException("ShardedTensor.flattened_range is not supported.")

    @property
    def has_regular_grid(self):
        """Alias for having a regular sharding grid."""
        return self.axis_fragmentations is not None

    def global_slice(self) -> Tuple[Union[int, slice], ...]:
        """
        Returns a tuple of int and slice objects representing a slice of the
        global tensor that this ShardedTensor corresponds to.
        """
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

    def local_chunk_offset_in_global(self) -> Tuple[int, ...]:
        """Offset of a local chunk in a global array of chunks.

        Returns:
            Tuple[int, ...]: the offset of the whole local chunk in a global array of chunks.
        """
        assert len(self.global_offset) == len(self.local_shape) + self.prepend_axis_num
        chunk_offset = list(self.global_offset[: self.prepend_axis_num])
        for off, sh in zip(self.global_offset[self.prepend_axis_num :], self.local_shape):
            assert off % sh == 0, str(self)
            chunk_offset.append(off // sh)
        return tuple(chunk_offset)

    def max_allowed_chunks(self) -> Tuple[int, ...]:
        """
        Returns the maximum allowed chunks for this ShardedTensor.
        """
        chunks = []
        for axis_sh, axis_fragm in zip(self.global_shape, self.axis_fragmentations):
            if not self.allow_shape_mismatch and axis_sh % axis_fragm != 0:
                raise CheckpointingException(
                    f"Axis shape ({axis_sh}) not divisible by axis fragmentation ({axis_fragm}"
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
        flattened_range: None = None,
        **init_kwargs,
    ):
        """Allows to construct the ShardedTensor given offset specified in process ranks.

        Args:
            key (str): unique key
            data (torch.Tensor): local tensor data
            rank_offsets (Tuple[int, int, int]): each tuple
                (axis, axis_rank_offset, axis_fragm) says that if
                global tensor is divided into `axis_fragm` fragment along `axis`
                axis, then local tensor data corresponds to the `axis_rank_offset` chunk.
            replica_id (ReplicaId): see ShardedTensor
            prepend_axis_num (int): see ShardedTensor
            flattened_range (None): must be None when using this constructor
            init_kwargs: passed to ShardedTensor.__init__
        """
        if flattened_range is not None:
            raise ValueError(
                "Cannot instantiate a flat ShardedTensor with `from_rank_offsets` method."
                " Use `from_rank_offsets_flat` instead"
            )
        global_offset = [0] * (data.ndim + prepend_axis_num)
        global_shape = ([1] * prepend_axis_num) + list(data.shape)
        axis_fragmentations = [1] * (data.ndim + prepend_axis_num)
        _seen_axis = set()
        for axis, axis_rank_offset, axis_fragm in rank_offsets:
            if axis < 0 or axis_rank_offset < 0 or axis_fragm < 1 or axis_rank_offset >= axis_fragm:
                raise CheckpointingException(f"Invalid rank offsets: {rank_offsets} for key {key}.")
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
            flattened_range=flattened_range,
            **init_kwargs,
        )

    def init_data(self, device: Union[str, torch.device], init_fn=torch.empty):
        """
        Initialize the tensor data of this ShardedTensor.

        Only called if `data` attribute is None.

        Args:
            device (Union[str, torch.device]): device to place the tensor on
            init_fn (Callable, optional): function to use to initialize the tensor.
                Defaults to `torch.empty`.
        """
        if self.data is not None:
            return
        self.data = init_fn(self.local_shape, dtype=self.dtype, device=device)

    def narrow(self, dim: int, start: int, length: int) -> List["ShardedTensor"]:
        """This is an analogue of torch.narrow for ShardedTensors.

        Narrowing assumes that we narrow a local tensor on each rank.
        This has consequences on local_shape, global_shape, global_offset, etc.

        Args:
            dim (int): dimension to narrow. Doesn't include prepended axes.
            start (int): start element
            length (int): length of the slice

        Returns:
            List[ShardedTensor]: narrowed ShardedTensors. For non-flat tensors,
                the list will always have 1 element. For flat ShardedTensors the number of
                elements varies depending on `dim` and on overlap, because flat
                tensors must be contiguous. In particular the list can be empty.
        """
        prepended_dim = dim + self.prepend_axis_num
        local_length_along_dim = self.local_shape[dim]

        def _update_tuple(x, ind, val):
            x = list(x)
            x[ind] = val
            return tuple(x)

        def _safe_div(x, y):
            assert x % y == 0, (x, y)
            return x // y

        # Decrease global shape and global offset by `length / local_length_along_dim`
        assert (
            self.global_shape[prepended_dim] % local_length_along_dim == 0
        ), f"Only regular grid of local tensors is supported for narrowing, got: {self}"
        assert (
            self.global_offset[prepended_dim] % local_length_along_dim == 0
        ), f"Only regular grid of local tensors is supported for narrowing, got: {self}"
        global_shape = _update_tuple(
            self.global_shape,
            prepended_dim,
            _safe_div(self.global_shape[prepended_dim] * length, local_length_along_dim),
        )
        global_offset = _update_tuple(
            self.global_offset,
            prepended_dim,
            _safe_div(self.global_offset[prepended_dim] * length, local_length_along_dim),
        )

        new_data = self.data.narrow(dim, start, length)
        # always a single result tensor
        return [
            replace(
                self,
                data=new_data,
                local_shape=new_data.shape,
                global_shape=global_shape,
                global_offset=global_offset,
            )
        ]


def is_main_replica(replica_id: ReplicaId):
    """Checks if given `replica_id` is considered as main.

    "Main" replica is:
    - integer 0
    - or an iterable with all 0 elements

    It is the application responsibility to set correct replicas for sharded tensors.

    Args:
        replica_id (Union[int, Tuple[int, ...]]): replica id

    Returns:
        (bool): True for a "main" replica
    """
    if isinstance(replica_id, int):
        return replica_id == 0
    return all(r == 0 for r in replica_id)


class LocalNonpersistentObject:
    """Object that should not be stored in a checkpoint, but restored locally.

    Wrapping any object inside the state dict with LocalNonpersistentObject
    will result in:
    - during saving, this object will *not* be stored in the checkpoint
    - during loading, a local version of this object will be placed in a state dict
    """

    def __init__(self, obj):
        self.obj = obj

    def unwrap(self):
        """Returns the original object."""
        return self.obj


@dataclass
class ShardedObject(ShardedBase):
    """Represents a mapping between a local object and a global object.

    Global object is assumed to consist of many local objects distributed
    between different processes.

    NOTE: Contrary to ShardedTensor, it's impossible to change global object
    sharding. Conceptually, ShardedObject is a fully-sharded ShardedTensor
    with atomic arbitrary typed elements.

    Args:
        key: unique identifier of a global tensor
        data: local object data. Can be None only for consistency validation
        global_shape: global object shape
        global_offset: offset of a local object in a global object, specified in number of shards
        replica_id: indicates local object replication wrt. local objects in different processes
    """

    key: str
    data: object
    global_shape: Tuple[int, ...]
    global_offset: Tuple[int, ...]
    replica_id: ReplicaId = 0

    def __post_init__(self):
        self.validate_metadata_integrity()

    def validate_metadata_integrity(self):
        if len(self.global_shape) != len(self.global_offset):
            raise CheckpointingException(
                f"Global offset dimensions should be equal to global shape dimensions for {self}"
            )

    def without_data(self):
        return replace(self, data=None)

    @property
    def unique_key(self):
        """returns a unique key for this object"""
        return (
            f"{self.key}/shard_"
            f"{'.'.join(map(str, self.global_offset))}_"
            f"{'.'.join(map(str, self.global_shape))}"
        )

    def __str__(self):
        return f"{self.__class__.__name__}(key='{self.key}')"

    @classmethod
    def empty_from_unique_key(cls, unique_key, replica_id: ReplicaId = 0) -> "ShardedObject":
        """Instantiates a ShardedObject from a unique key.

        Args:
            unique_key: a string of the form
                <key>/shard_<global_offset>_<global_shape>
            replica_id: indicates local object replication wrt.
                local objects in different processes

        Returns:
            a ShardedObject with data=None
        """
        key, shard_key = unique_key.split("/")
        shard_str, offset, shape = shard_key.split("_")
        assert shard_str == "shard"
        offset = tuple(map(int, offset.split(".")))
        shape = tuple(map(int, shape.split(".")))
        if len(shape) + 1 == len(offset):
            # This is a backward-compatible fix. We don't know the last
            # element of global shape so set it to -1.
            shape += (-1,)
        return cls(key, None, shape, offset, replica_id)


FactoryBuildFn = Callable[[str, torch.Tensor, ReplicaId, Optional[slice]], ShardedStateDict]
FactoryMergeFn = Callable[[StateDict], torch.Tensor]


@dataclass
class ShardedTensorFactory(ShardedBase):
    """Allows to apply transformations to tensors before/after serialization.

    The essence of those transformations is that they can be applied to
    optimizer states the same way they are applied to the model params.
    The ultimate state dict with sharded tensors must depend functionally on
    `build_fn` arguments (key, data, replica_id, flattened_range),
    which will be provided by the optimizer.

    Builder creates a sub-state-dict out of a tensor before saving, and merger
    merges the corresponding state dict after loading.

    Args:
        key (str): unique identifier of the factory
        data (torch.Tensor): original model parameter that will be further
            transformed by this factory
        build_fn (callable): function that transforms the original tensor
            to a sharded state dict
        merge_fn (callable): function that transforms loaded subtree back
            into a single tensor (inverse of `build_fn`)
        replica_id (ReplicaId): indicates factory replication wrt.
            factories in different processes
        flattened_range (slice, optional): indicates additional flattening
            applied to the ShardedTensors produced by the factory
    """

    key: str
    data: torch.Tensor
    build_fn: FactoryBuildFn
    merge_fn: FactoryMergeFn
    replica_id: ReplicaId = 0
    flattened_range: Optional[slice] = None

    def build(self):
        """Builds a ShardedStateDict from the original tensor"""
        return self.build_fn(self.key, self.data, self.replica_id, self.flattened_range)

    def validate_metadata_integrity(self):
        """No reasonable checks can be applied"""
        pass

    def without_data(self):
        return replace(self, data=None)


def apply_factories(sharded_state_dict: ShardedStateDict):
    """Turn ShardedTensorFactories into ShardedTensors *in-place*.

    Args:
        sharded_state_dict (ShardedStateDict): state dict possibly
            containing ShardedTensorFactory objects

    Returns:
        None: state dict is modified in place
    """

    def apply(x):
        if isinstance(x, ShardedTensorFactory):
            x = x.build()
        return x

    dict_list_map_inplace(apply, sharded_state_dict)


def apply_factory_merges(
    x1: StateDict, x2: ShardedStateDict, key: Tuple[str, ...] = ()
) -> StateDict:
    """Apply merges defined by ShardedTensorFactories *in-place*.

    Args:
        x1 (StateDict): state dict loaded from the checkpoint
        x2 (ShardedStateDict): subset of `x1` (in terms of dict keys)
            with ShardedTensorFactory
            as (possibly nested) values that define how to
            merge objects from the `x1` state dict
        key (Tuple[str, ...]): current key in a recursive call.
            Used only for reporting meaningful errors

    Returns:
        StateDict: `x1` modified in-place
    """
    if isinstance(x2, ShardedTensorFactory):
        return x2.merge_fn(x1)

    # There rest is almost the same as the `merge` function from `dict_utils`
    if isinstance(x1, dict) and isinstance(x2, dict):
        for k, v2 in x2.items():
            if k not in x1:
                raise ValueError(
                    f"Different dict keys encountered in `apply_factory_merges` "
                    f"({x1.keys()} vs {x2.keys()})"
                )
            else:
                x1[k] = apply_factory_merges(x1[k], v2, key=key + (k,))
    elif isinstance(x1, list) and isinstance(x2, list):
        if len(x1) != len(x2):
            err_msg = (
                f"Cannot merge two lists with different lengths "
                f"({len(x1)} and {len(x2)}, encountered at key {key})"
            )
            logger.error(err_msg + f"\nx1: {x1}\nx2: {x2}")
            raise ValueError(err_msg)
        for i, v2 in enumerate(x2):
            x1[i] = apply_factory_merges(x1[i], v2, key=key + (i,))
    elif isinstance(x1, list) and isinstance(x2, dict):
        for k, v2 in x2.items():
            if not isinstance(k, int):
                raise ValueError(
                    f"Invalid dict key {k} non-integer type encountered "
                    f"in a list-dict merge at level {key}"
                )
            if k >= len(x1):
                raise ValueError(
                    f"Dict key {k} out of bound for list of length"
                    f"{len(x1)} (encountered at level {key})"
                )
            x1[k] = apply_factory_merges(x1[k], v2, key=key + (k,))
    else:
        raise ValueError(
            f"Duplicate non-dict and non-list values encountered: `{x1}` and `{x2} (at key {key})`"
        )
    return x1
