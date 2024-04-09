# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using TensorStore to load and save Zarr arrays. """

from functools import partial
from itertools import starmap
from logging import getLogger
from pathlib import Path

import tensorstore as ts
import torch

from ..core import CheckpointingException
from ..dict_utils import dict_list_map_inplace
from ..mapping import ShardedStateDict, ShardedTensor
from .base import LoadShardedStrategy, StrategyAction, default_strategies
from .zarr import (
    load_zarr_based_sharded_metadata,
    numpy_to_torch_dtype_dict,
    postprocess_numpy_array,
)

_import_trigger = None

logger = getLogger(__name__)


class TensorStoreLoadShardedStrategy(LoadShardedStrategy):
    def __init__(self, load_directly_on_device: bool = False):
        super().__init__()
        self.load_directly_on_device = load_directly_on_device

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        if torch.distributed.get_rank() == 0:
            print(f'Loading distributed checkpoint with {self.__class__.__name__}')
            if self.load_directly_on_device:
                print(f'Loading distributed checkpoint directly on the GPU')
        load_fn = partial(
            _load_from_array,
            checkpoint_dir=checkpoint_dir,
            load_directly_on_device=self.load_directly_on_device,
        )
        dict_list_map_inplace(load_fn, sharded_state_dict)
        return sharded_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path):
        def get_ts_shape_dtype(path):
            arr = open_ts_array(path)
            return arr.shape, arr.dtype.numpy_dtype

        return load_zarr_based_sharded_metadata(checkpoint_dir, get_ts_shape_dtype)

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO


def merge_global_slice_with_shape(global_slice, actual_shape, key):
    def _merge_slice(dim_slice, dim_size):
        if isinstance(dim_slice, slice):
            assert (
                dim_slice.start < dim_size
            ), f'Got empty slice for ShardedTensor {key} ({dim_slice}, {dim_size})'
            if dim_slice.stop > dim_size:
                dim_slice = slice(dim_slice.start, dim_size, dim_slice.step)
        return dim_slice

    assert len(global_slice) == len(actual_shape), (global_slice, actual_shape, key)
    return tuple(starmap(_merge_slice, zip(global_slice, actual_shape)))


def _load_from_array(
    sharded_tensor: ShardedTensor,
    checkpoint_dir: Path,
    load_directly_on_device: bool = False,
    apply_flattened_range: bool = True,
):
    x = _load_regular_chunk(sharded_tensor, checkpoint_dir)
    ten = postprocess_numpy_array(x, sharded_tensor, apply_flattened_range)
    if load_directly_on_device:
        sharded_tensor.data.data.copy_(ten)
        return sharded_tensor.data
    else:
        return ten


def _load_regular_chunk(sharded_tensor: ShardedTensor, checkpoint_dir: Path):
    assert isinstance(sharded_tensor, ShardedTensor), type(sharded_tensor)
    arr = open_ts_array(checkpoint_dir / sharded_tensor.key)
    if sharded_tensor.global_shape == arr.shape:
        x = (
            arr[sharded_tensor.global_slice()].read().result()
        )  # flattened tensors loading is delayed
    elif sharded_tensor.allow_shape_mismatch:
        global_slice = merge_global_slice_with_shape(
            sharded_tensor.global_slice(), arr.shape, sharded_tensor.key
        )
        x = arr[global_slice].read().result()  # flattened tensors loading is delayed
    else:
        _msg = (
            f'Global shape mismatch for loaded ({arr.shape})'
            f' and expected ({sharded_tensor.global_shape}) tensor'
            f' for key {sharded_tensor.key}'
        )
        raise CheckpointingException(_msg)
    return x


def open_ts_array(arr_path: Path):
    """Opens a Zarr file array with Tensorstore with basic setting.

    Args:
        arr_path (Path): path to a Zarr (Tensorstore) array
    """
    spec = {'driver': 'zarr', 'metadata_key': '.zarray', 'kvstore': {}}
    spec['kvstore'] = {
        'driver': 'file',
        'path': str(arr_path),
    }
    try:
        arr = ts.open(ts.Spec(spec), open=True).result()
    except Exception as e:
        raise CheckpointingException(f'Array {arr_path} could not be loaded. Error: {e}') from e
    return arr


default_strategies[StrategyAction.LOAD_SHARDED.value][
    ('zarr', 1)
] = TensorStoreLoadShardedStrategy()
