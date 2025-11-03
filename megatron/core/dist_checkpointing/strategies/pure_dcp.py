# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using PyTorch distributed.checkpoint as an underlying format. """
import os
import pickle
import warnings
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast

import torch
from torch.distributed import checkpoint, DeviceMesh
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    FileSystemReader,
    FileSystemWriter,
    TensorStorageMetadata
)
from torch.distributed.checkpoint.metadata import Metadata

from ...utils import is_torch_min_version
from ..dict_utils import dict_list_map_inplace
from ..mapping import (
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
    CheckpointableShardedTensor,
)
from .async_utils import AsyncRequest
from .base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
)

try:
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

_metadata_fn: str = ".metadata"


logger = getLogger(__name__)

SH_TEN_MODE = os.getenv('MCORE_SH_TEN_MODE', 'ckptable')  # dtensor, ckptable, ?legacy?


# NOTE: this function will be gone
def translate_state_dict_to_dcp_compatible(sharded_state_dict: ShardedStateDict,
                                           sh_ten_mode: str = 'dtensor') -> StateDict:
    num_dtensor = 0
    num_ckptable = 0
    def sh_ten_to_dtensor(x: Union[ShardedTensor, Any]) -> Union[Any, DTensor]:
        nonlocal num_dtensor
        nonlocal num_ckptable

        if isinstance(x, ShardedTensor):
            if sh_ten_mode == 'dtensor':
                x = x.to_dtensor()
                num_dtensor += 1
            elif sh_ten_mode == 'ckptable':
                x = x.to_checkpointable()
                num_ckptable += 1
            elif sh_ten_mode == 'hybrid':
                if x.dtensor_ckpt_device_mesh is not None:
                    x = x.to_dtensor()
                    num_dtensor += 1
                else:
                    x = x.to_checkpointable()
                    num_ckptable += 1
            else:
                raise NotImplementedError(f'sh_ten_mode: {sh_ten_mode}')
        elif isinstance(x, ShardedObject):
            if not all(dim_size == 1 for dim_size in x.global_shape):
                raise RuntimeError("INTERNAL WARNING: ShardedObjects with non-trivial sharding won't be supported")
        return x

    dict_list_map_inplace(sh_ten_to_dtensor, sharded_state_dict)
    print(f'Translations: DTensor: {num_dtensor}, ShTen(_Checkpointable): {num_ckptable}.')
    if num_dtensor > 0 and num_ckptable > 0:
        print(f'Coexisting DTensors ({num_dtensor}) and ShTen(_Checkpointable) tensors ({num_ckptable}) discovered!.')
    return sharded_state_dict


# This stays, but could be simplified to a non-nested case if current code stays
def unwrap_dtensors_and_sh_ten(state_dict: StateDict) -> StateDict:
    def dtensor_to_ten(x: Union[DTensor, Any]) -> Union[Any, torch.Tensor]:
        if isinstance(x, DTensor):
            x = x.to_local()
        elif isinstance(x, CheckpointableShardedTensor):
            x = x._sh_ten.data
        elif isinstance(x, ShardedObject):
            x = x.data
        return x

    dict_list_map_inplace(dtensor_to_ten, state_dict)
    return state_dict

@dataclass
class PlaceholderValue:
    key: str


def inject_placeholders(sharded_state_dict: ShardedStateDict) -> Dict[str, Any]:
    """Replaces values in state dict with ValuePlaceholders.

    Extracts all values from a given state dict to a flat dict, injecting
    placeholders instead to allow later recovery with `fill_placeholders`.
    """
    # TODO: in order to handle arbitrary DTensors (without `.key` attribute)
    #  an additional step computing FQNs might be needed (`traverse_state_dict`?)
    #  which computes DTensor.key
    extracted_values = {}

    def _replace_with_placeholder(x: Union[ShardedBase, DTensor]):
        if isinstance(x, DTensor):
            if not hasattr(x, 'key'):
                raise NotImplementedError(f'DTensors currently require `key` attribute, got: {x}')
        elif not isinstance(x, ShardedBase):
            raise RuntimeError(f'Unexpected type {x} during placeholders injection')

        if x.key in extracted_values:
            raise RuntimeError(f'Duplicated sharded key encountered: {x.key}')
        extracted_values[x.key] = x
        return PlaceholderValue(x.key)

    dict_list_map_inplace(_replace_with_placeholder, sharded_state_dict)
    return extracted_values


def fill_placeholders(sharded_state_dict: ShardedStateDict, loaded_values: Dict[str, Any]) -> None:
    """Inverse of `inject_placeholders`. """
    def _fill_placeholder(x: PlaceholderValue):
        assert isinstance(x, PlaceholderValue)
        return loaded_values[x.key]
    dict_list_map_inplace(_fill_placeholder, sharded_state_dict)


class TorchDistSaveShardedStrategy(AsyncSaveShardedStrategy):
    """TODO. """

    def __init__(
        self,
        backend: str,
        version: int,
        keep_only_main_replica: bool = True,
        sh_ten_mode: Optional[str] = None
    ):
        """Adds parameters specific to PyT Distributed format
        Args:
            backend (str): format backend string
            version (int): format version
            keep_only_main_replica (bool, optional): PyT Distributed has a mechanism
                for deduplication, but replica_id aware deduplication is more coherent.
                Default is True (recommended to keep it).
            thread_count (int, optional): threads to use during saving.
                Affects the number of files in the checkpoint (saving ranks * num_threads).
        """
        super().__init__(backend, version)
        self.keep_only_main_replica = keep_only_main_replica
        if sh_ten_mode is None:
            sh_ten_mode = SH_TEN_MODE
        self.sh_ten_mode = sh_ten_mode

    def async_save(
            self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> AsyncRequest:
        values_to_save = inject_placeholders(sharded_state_dict)

        # NOTE: this translation won't be needed if _Checkpointable or DTensor in state dict by default
        values_to_save = translate_state_dict_to_dcp_compatible(values_to_save, self.sh_ten_mode)

        torch.distributed.checkpoint.save(values_to_save, checkpoint_id=checkpoint_dir)

        # Everything already done synchronously
        # TODO: how to implement async?
        return AsyncRequest(lambda: None, (), [])

    @property
    def can_handle_sharded_objects(self):
        return True


class TorchDistLoadShardedStrategy(LoadShardedStrategy):
    """TODO. """

    def __init__(self, sh_ten_mode: Optional[str] = None):
        super().__init__()
        if sh_ten_mode is None:
            sh_ten_mode = SH_TEN_MODE
        self.sh_ten_mode = sh_ten_mode

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path):
        values_to_load = inject_placeholders(sharded_state_dict)

        # NOTE: this translation won't be needed if _Checkpointable or DTensor in state dict by default
        values_to_load = translate_state_dict_to_dcp_compatible(values_to_load, self.sh_ten_mode)

        torch.distributed.checkpoint.load(values_to_load, checkpoint_id=checkpoint_dir)
        unwrap_dtensors_and_sh_ten(values_to_load)

        fill_placeholders(sharded_state_dict, values_to_load)

        return sharded_state_dict


    def load_tensors_metadata(self, checkpoint_dir: Path, metadata: Metadata = None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            fs_reader = FileSystemReader(checkpoint_dir)
            metadata = fs_reader.read_metadata()

        mcore_data = getattr(metadata, 'mcore_data', {})
        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, TensorStorageMetadata):
                continue  # load only tensors

            nd_orig_global_shape = mcore_data.get(k, {}).get('nd_reformulated_orig_global_shape')
            if nd_orig_global_shape is None:
                # Regular tensor
                sharded_metadata[k] = ShardedTensor.from_rank_offsets(
                    k, torch.empty(tp.size, **tp.properties.__dict__, device='meta'),
                    dtensor_ckpt_device_mesh=DeviceMesh.from_group(torch.distributed.GroupMember.WORLD, "cuda"),
                ).without_data()
            else:
                # N-D flattened tensor
                unflat_ten = torch.empty(
                    nd_orig_global_shape, **tp.properties.__dict__, device='meta'
                )
                flat_ten = unflat_ten.flatten()
                sharded_metadata[k] = ShardedTensor.from_rank_offsets_flat(
                    k,
                    flat_ten,
                    unflat_ten.shape,
                    flattened_range=slice(0, unflat_ten.numel()),  # whole slice
                ).without_data()

        return sharded_metadata

    def load_sharded_metadata(self, checkpoint_dir: Path) -> ShardedStateDict:
        """Uses tensors and objects metadata stored in the metadata file."""
        fs_reader = FileSystemReader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_key(metadata_key)
            sharded_metadata[sh_obj.unique_key] = sh_obj

        sharded_metadata.update(self.load_tensors_metadata(checkpoint_dir, metadata))
        return sharded_metadata

    def remove_sharded_tensors(self, checkpoint_dir: str, key_prefix: str):
        """Removes checkpoint files whose keys have the given prefix.

        Performs the following steps:
        1. checks whether there are files that start with the key_prefix
        2. loads metadata
        3. removes all entries from the metadata that start with the key_prefix
        4. resaves the new metadata and removes the old metadata
        5. removes the relevant files
        """

        assert is_torch_min_version(
            "2.3.0"
        ), f'torch >= 2.3.0 is required for remove_sharded_tensors'

        distckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith("distcp")]
        files_to_remove = [f for f in distckpt_files if f.startswith(key_prefix)]

        if not files_to_remove:
            warnings.warn(
                f'There are no files in {checkpoint_dir} that begin with "{key_prefix}".'
                f' Skipping removal.'
            )
            return

        fs_reader = FileSystemReader(checkpoint_dir)
        original_metadata = fs_reader.read_metadata()

        new_state_dict_metadata = {}
        new_planner_data = {}
        new_storage_data = {}
        for k in original_metadata.state_dict_metadata.keys():
            if k.startswith(key_prefix):
                continue
            new_state_dict_metadata[k] = original_metadata.state_dict_metadata[k]
        for k in original_metadata.planner_data.keys():
            if k.startswith(key_prefix):
                continue
            new_planner_data[k] = original_metadata.planner_data[k]
        for k in original_metadata.storage_data.keys():
            if k.fqn.startswith(key_prefix):
                continue
            new_storage_data[k] = original_metadata.storage_data[k]
        metadata = Metadata(
            state_dict_metadata=new_state_dict_metadata,
            planner_data=new_planner_data,
            storage_data=new_storage_data,
        )
        fs_writer = FileSystemWriter(checkpoint_dir)
        metadata_filename = cast(Path, fs_writer.fs.concat_path(fs_writer.path, _metadata_fn))
        tmp_path = cast(
            metadata_filename, fs_writer.fs.concat_path(fs_writer.path, f"{_metadata_fn}.tmp")
        )
        old_path = cast(
            metadata_filename, fs_writer.fs.concat_path(fs_writer.path, f"{_metadata_fn}.bck")
        )
        ## save the new metadata
        with fs_writer.fs.create_stream(tmp_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            try:
                os.fsync(metadata_file.fileno())
            except AttributeError:
                os.sync()
        ## move the old metadata
        fs_writer.fs.rename(fs_writer.metadata_path, old_path)
        try:
            ## rename the new metadata
            fs_writer.fs.rename(tmp_path, fs_writer.metadata_path)

            ## finally, remove the files we want to drop
            for f in files_to_remove:
                fs_writer.fs.rm_file(checkpoint_dir / f)
        except Exception as e:
            fs_writer.fs.rename(old_path, fs_writer.metadata_path)
            raise e
        else:
            fs_writer.fs.rm_file(old_path)

    def can_handle_sharded_objects(self):
        return True

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO
