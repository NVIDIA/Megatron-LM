# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using Zarr as an underlying format. """
import os
from functools import partial
from logging import getLogger
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import zarr

from ..core import CheckpointingException
from ..dict_utils import dict_list_map_inplace
from ..mapping import ShardedStateDict, ShardedTensor, is_main_replica
from .base import LoadShardedStrategy, SaveShardedStrategy, StrategyAction, default_strategies

numpy_to_torch_dtype_dict = {
    np.dtype('bool'): torch.bool,
    np.dtype('uint8'): torch.uint8,
    np.dtype('int8'): torch.int8,
    np.dtype('int16'): torch.int16,
    np.dtype('int32'): torch.int32,
    np.dtype('int64'): torch.int64,
    np.dtype('float16'): torch.float16,
    np.dtype('float32'): torch.float32,
    np.dtype('float64'): torch.float64,
    np.dtype('complex64'): torch.complex64,
    np.dtype('complex128'): torch.complex128,
}

torch_to_numpy_dtype_dict = {v: k for k, v in numpy_to_torch_dtype_dict.items()}


try:
    import tensorstore

    HAS_BFLOAT16 = True
    numpy_to_torch_dtype_dict[np.dtype('bfloat16')] = torch.bfloat16
    torch_to_numpy_dtype_dict[torch.bfloat16] = np.dtype('bfloat16')
except ImportError:
    HAS_BFLOAT16 = False

_import_trigger = None

logger = getLogger(__name__)


class ZarrSaveShardedStrategy(SaveShardedStrategy):
    def save(self, sharded_tensors: List[ShardedTensor], checkpoint_dir: Path):
        arrays = _create_or_open_zarr_arrays(sharded_tensors, checkpoint_dir)
        for ten, arr in zip(sharded_tensors, arrays):
            _save_to_existing_array(ten, arr)
        torch.distributed.barrier()


def _create_or_open_zarr_arrays(
    sharded_tensors: List[ShardedTensor], checkpoint_dir: Path
) -> List[Optional[zarr.Array]]:
    """ Returns list of zarr arrays corresponding to given tensors.

    For a sharded tensors that:
    a) is main replica and represents the first chunk (all offsets 0), creates the Zarr array
    b) is main replica but not the first chunk, opens the arrays created in (a) (possibly by other process)
    c) otherwise, sets the corresponding array to None since it won't be used

    Args:
        sharded_tensors (List[ShardedTensor]): sharded tensors from a given rank that will be saved to checkpoint
        checkpoint_dir (Path): checkpoint in which the arrays will be created
    """
    arrays = []
    for ten in sharded_tensors:
        arr = _create_zarr_array(ten, checkpoint_dir) if _should_create_array(ten) else None
        arrays.append(arr)

    torch.distributed.barrier()
    # Open arrays created above by other processes
    for arr_idx, ten in enumerate(sharded_tensors):
        if arrays[arr_idx] is not None:
            # array created by this process
            assert _should_create_array(ten), ten
            continue
        if not is_main_replica(ten.replica_id):
            # this array won't be needed for saving and can stay None
            continue
        open_kwargs = {}
        if ten.flattened_range is not None:
            open_kwargs['synchronizer'] = zarr.ProcessSynchronizer(
                str(checkpoint_dir / f'{ten.key}.sync')
            )
        arrays[arr_idx] = zarr.open(checkpoint_dir / ten.key, 'r+', **open_kwargs)
    return arrays


def _should_create_array(ten: ShardedTensor):
    return (
        is_main_replica(ten.replica_id)
        and set(ten.global_offset) == {0}
        and (ten.flattened_range is None or ten.flattened_range.start == 0)
    )


def _save_to_existing_array(sharded_tensor: ShardedTensor, arr: Optional[zarr.Array]):
    if not is_main_replica(sharded_tensor.replica_id):
        return
    assert arr is not None
    x = sharded_tensor.data
    x = x.detach().cpu()
    torch.cuda.synchronize()
    if x.dtype == torch.bfloat16:
        x = x.float()
        x = x.numpy()
        x = x.astype('bfloat16')
    else:
        x = x.numpy()

    if sharded_tensor.flattened_range is None:
        arr[sharded_tensor.global_slice()] = x
    else:
        arr.set_coordinate_selection(sharded_tensor.global_coordinates(), x)


def _create_zarr_array(sharded_tensor: ShardedTensor, checkpoint_dir: Path):
    np_dtype = torch_to_numpy_dtype_dict[sharded_tensor.dtype]
    try:
        arr = zarr.create(
            sharded_tensor.global_shape,
            dtype=np_dtype,
            store=checkpoint_dir / sharded_tensor.key,
            chunks=sharded_tensor.max_allowed_chunks(),
            compressor=None,
            fill_value=None,
            write_empty_chunks=True,
        )
    except zarr.errors.ContainsArrayError as e:
        raise CheckpointingException(
            f'Array {checkpoint_dir / sharded_tensor.key} already exists'
        ) from e

    if HAS_BFLOAT16 and np_dtype == np.dtype('bfloat16'):
        arr._dtype = np_dtype
        zarray = arr.store['.zarray']
        arr.store['.zarray'] = zarray.replace(b'<V2', b'bfloat16')
    return arr


class ZarrLoadShardedStrategy(LoadShardedStrategy):
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        dict_list_map_inplace(
            partial(_load_from_array, checkpoint_dir=checkpoint_dir), sharded_state_dict
        )
        return sharded_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path):
        def get_zarr_shape_dtype(path):
            arr = zarr.open(path, 'r')
            return arr.shape, arr.dtype

        return load_zarr_based_sharded_metadata(checkpoint_dir, get_zarr_shape_dtype)

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO


def _load_from_array(sharded_tensor: ShardedTensor, checkpoint_dir: Path):
    assert isinstance(sharded_tensor, ShardedTensor), type(sharded_tensor)
    try:
        arr = zarr.open(checkpoint_dir / sharded_tensor.key, 'r')
    except zarr.errors.PathNotFoundError as e:
        raise CheckpointingException(
            f'Array {checkpoint_dir / sharded_tensor.key} not found'
        ) from e

    if not sharded_tensor.allow_shape_mismatch and sharded_tensor.global_shape != arr.shape:
        _msg = (
            f'Global shape mismatch for loaded ({arr.shape})'
            f' and expected ({sharded_tensor.global_shape}) tensor'
            f' for key {sharded_tensor.key}'
        )
        raise CheckpointingException(_msg)

    x = arr[sharded_tensor.global_slice()]  # flattened tensors loading is delayed
    return postprocess_numpy_array(x, sharded_tensor)


def postprocess_numpy_array(loaded_array, sharded_tensor, apply_flattened_range=True):
    x = loaded_array
    if HAS_BFLOAT16 and x.dtype == np.dtype('bfloat16'):
        x = x.astype(np.dtype('float32'))
        x = torch.from_numpy(x)
        x = x.bfloat16()
    else:
        x = torch.from_numpy(x)
    # TODO: consider some other consistency checks
    if x.shape != sharded_tensor.local_shape:
        if sharded_tensor.allow_shape_mismatch:
            x = pad_to_expected_shape(x, sharded_tensor)
        else:
            _msg = (
                f'Local shape mismatch for loaded ({x.shape})'
                f' and expected ({sharded_tensor.local_shape}) tensor'
                f' for key {sharded_tensor.key}'
            )
            raise CheckpointingException(_msg)

    if apply_flattened_range and sharded_tensor.flattened_range is not None:
        x = flatten_range(sharded_tensor, x)

    # TODO: consider cuda() tensors support
    return x


def flatten_range(sharded_tensor, x):
    return x.flatten()[sharded_tensor.flattened_range]


def pad_to_expected_shape(x: torch.Tensor, expected_sharded_ten: ShardedTensor):
    pad_args = []
    assert len(x.shape) == len(expected_sharded_ten.local_shape)
    # Reversed iteration order because F.pad expects so
    for x_sh, exp_sh, axis_fragm in reversed(
        list(
            zip(x.shape, expected_sharded_ten.local_shape, expected_sharded_ten.axis_fragmentations)
        )
    ):
        if x_sh == exp_sh:
            pad_args.extend((0, 0))
        elif x_sh > exp_sh:
            assert (
                False
            ), f'Expected shape ({exp_sh}) smaller than actual ({x_sh}) for {repr(expected_sharded_ten)}'
        else:
            pad_args.extend((0, exp_sh - x_sh))
    # TODO: behavior control with envvar is for testing purposes only, remove it
    if not int(os.environ.get('DIST_CKPT_PAD_REPLICATE', 0)):
        return torch.nn.functional.pad(x, pad_args)

    # unsqueeze and squeeze to get shapes supported by cudnn
    print(f'Replicating last row for {expected_sharded_ten.key}')
    if x.dtype == torch.bfloat16:
        return (
            torch.nn.functional.pad(x.float().unsqueeze(0), pad_args, mode='replicate')
            .squeeze(0)
            .bfloat16()
        )
    return torch.nn.functional.pad(x.unsqueeze(0), pad_args, mode='replicate').squeeze(0)


def load_zarr_based_sharded_metadata(
    checkpoint_dir: Path, get_shape_dtype_fn: Callable[[str], Tuple[Tuple[int], np.dtype]]
) -> ShardedStateDict:
    """Load metadata of Zarr arrays.

    Arguments:
        checkpoint_dir (str): checkpoint root directory
        get_shape_dtype_fn (str -> ((int, ...), np.dtype)): a function returning
            an array shape and dtype for a given Zarr array path
    """
    sharded_state_dict = {}
    for subdir in checkpoint_dir.iterdir():
        if not subdir.is_dir() or not (subdir / '.zarray').exists():
            continue
        key = subdir.name
        arr_shape, arr_dtype = get_shape_dtype_fn(str(subdir))

        sharded_state_dict[key] = ShardedTensor(
            key,
            None,
            numpy_to_torch_dtype_dict[arr_dtype],
            arr_shape,
            arr_shape,
            tuple(0 for _ in arr_shape),
            tuple(1 for _ in arr_shape),
        )
    return sharded_state_dict


# default_strategies[StrategyAction.LOAD_SHARDED.value][('zarr', 1)] = ZarrLoadShardedStrategy()
default_strategies[StrategyAction.SAVE_SHARDED.value][('zarr', 1)] = ZarrSaveShardedStrategy(
    'zarr', 1
)
