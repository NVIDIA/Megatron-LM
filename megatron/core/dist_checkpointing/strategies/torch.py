# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using PyTorch distributed.checkpoint as an underlying format. """
import dataclasses
import io
import itertools
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from torch.distributed import checkpoint
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import Shard, ShardedTensorMetadata, TensorProperties
from torch.distributed._sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    LoadPlan,
    SavePlan,
    TensorStorageMetadata,
    WriteItem,
)
from torch.distributed.checkpoint._nested_dict import FLATTEN_MAPPING, unflatten_state_dict
from torch.distributed.checkpoint._traverse import OBJ_PATH, traverse_state_dict
from torch.distributed.checkpoint.default_planner import create_default_local_save_plan
from torch.distributed.checkpoint.planner_helpers import _create_write_items

from ..core import CheckpointingException
from ..dict_utils import nested_values
from ..mapping import (
    ShardedBase,
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
    is_main_replica,
)
from .async_utils import AsyncRequest
from .base import (
    AsyncSaveShardedStrategy,
    LoadShardedStrategy,
    SaveShardedStrategy,
    StrategyAction,
    default_strategies,
)
from .filesystem_async import FileSystemWriterAsync
from .state_dict_saver import save_state_dict_async_finalize, save_state_dict_async_plan

_import_trigger = None

logger = getLogger(__name__)


def flatten_state_dict(
    state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, Dict[str, OBJ_PATH]]:
    """ Flattens state dict into a single level dict.

    It's a copy of torch.distributed.checkpoint._nested_dict.flatten_state_dict
    which also accepts ShardedBase tensors as terminal objects

    Args:
        state_dict (ShardedStateDict): state dict to be flattened

    Returns (tuple): flattened state dict and a mapping allowing to recreate the original one

    """
    flattened = {}
    mappings = {}

    def flat_copy(path: OBJ_PATH, value: Any) -> None:
        new_fqn = ".".join(map(str, path))
        if new_fqn in flattened:
            raise ValueError(f"duplicated flatten key {new_fqn}")
        flattened[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy, lambda x: isinstance(x, (torch.Tensor, ShardedBase)))
    return flattened, mappings


def sharded_tensor_to_torch_sharded_tensor(
    sh_tens: List[ShardedTensor], rank: Optional[int] = None
) -> TorchShardedTensor:
    """Convert MCore ShardedTensor to PyT ShardedTensor. PyT requires information about all chunks.

    NOTE: this function assumes regular (grid) sharding of the MCore ShardedTensor.
    The only local irregularities could be introduced with a `flattened_range` attribute.

    NOTE: `flattened_range` is currently supported only for 1D tensors.

    This function follows the logic of torch.distributed.fsdp._shard_utils._create_chunk_sharded_tensor.
    Additionally, it saves `prepend_axis_num` (specific to MCore) as an attribute
    for further restoration in `_unwrap_pyt_sharded_tensor`.

    Args:
        sh_tens (List[ShardedTensor]): list of sharded tensors to convert
        rank (int, optional): current process rank passed to PyT ShardedTensor.
            If None, assumes rank in the default pg.

    Returns (TorchShardedTensor): PyT ShardedTensor containing all passed shards.

    """
    if rank is None:
        rank = torch.distributed.get_rank()

    some_sh_ten = sh_tens[0]
    has_flattened_range = some_sh_ten.flattened_range is not None

    prepend_axis_num = sh_tens[0].prepend_axis_num
    # Determine local shards
    if has_flattened_range:
        if prepend_axis_num:
            raise NotImplementedError(
                '`prepend_axis_num` attribute of ShardedTensor not supported'
                'together with `flattened_range` for PyT Distributed format'
            )
        for sh_ten in sh_tens:
            assert sh_ten.flattened_range is not None
            assert len(sh_ten.global_offset) == 1, sh_ten

        local_shards = [
            Shard.from_tensor_and_offsets(
                sh_ten.data, [sh_ten.global_offset[0] + sh_ten.flattened_range.start], rank
            )
            for sh_ten in sh_tens
        ]
        offsets_shape = some_sh_ten.local_shape  # used to determine local offsets
    else:
        # Apply extra axes `prepend_axis_num` with a view
        for sh_ten in sh_tens:
            assert sh_ten.flattened_range is None, sh_ten.flattened_range
            if prepend_axis_num:
                sh_ten.data = sh_ten.data.view((1,) * prepend_axis_num + sh_ten.local_shape)

        local_shards = [
            Shard.from_tensor_and_offsets(sh_ten.data, list(sh_ten.global_offset), rank)
            for sh_ten in sh_tens
        ]
        offsets_shape = some_sh_ten.data.shape  # includes prepended axes

    local_global_offsets = {}
    for sh_ten in sh_tens:
        local_global_offsets.setdefault(sh_ten.global_offset, []).append(sh_ten)

    # Create a ShardedTensor without invoking communication. Determine global shards
    shard_metadata = []
    # NOTE: here we assume a regular grid of shards
    for fragment_offsets in itertools.product(*map(range, some_sh_ten.axis_fragmentations)):
        offset = tuple(map(lambda x: x[0] * x[1], zip(fragment_offsets, offsets_shape)))
        if offset in local_global_offsets:
            # local shard
            placement = f"rank:{rank}/cuda"
            for sh_ten in local_global_offsets[offset]:
                if has_flattened_range:
                    offset = (sh_ten.global_offset[0] + sh_ten.flattened_range.start,)
                size = sh_ten.data.shape
                shard_metadata.append(ShardMetadata(offset, size, placement))

        else:
            # for shards from other ranks we provide simplistic data - this information will be discarded
            # during TorchShardedTensor._init_from_local_shards_and_global_metadata call
            shard_metadata.append(ShardMetadata(offset, offsets_shape, "cuda"))

    tensor = some_sh_ten.data
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=torch.Size(some_sh_ten.global_shape),
        tensor_properties=TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        ),
    )
    pyt_sh_ten = TorchShardedTensor._init_from_local_shards_and_global_metadata(
        local_shards, sharded_tensor_metadata=sharded_tensor_metadata, process_group=None
    )
    pyt_sh_ten.prepend_axis_num = prepend_axis_num
    return pyt_sh_ten


def mcore_to_pyt_state_dict(
    state_dict: Dict[str, List[ShardedBase]],
    is_loading: bool = False,
    init_device: torch.device = torch.device("cpu"),
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
    """Turn state dict with ShardedTensors and ShardedObjects to state dict compatible with PyT Dist format.

    Operates in-place and returns the original state dict.

    Args:
        state_dict (Dict[str, List[ShardedBase]]): flattened state dict, where values
            are lists of either ShardedTensor or ShardedObjects.
        is_loading (bool, optional): flag indicating if loading or saving. Defaults to False.
        init_device (torch.device, optional): device to initialize potentially missing tensors
            during loading. Defaults to 'cpu'.

    Returns (Dict[str, Union[TorchShardedTensor, io.BytesIO]]): original dictionary with values
        converted either into PyT ShardedTensors or io.BytesIO.

    """
    rank = torch.distributed.get_rank()
    pyt_state_dict = {}

    def _mcore_to_torch_sharded_tensor(sh_tens: List[ShardedTensor]) -> TorchShardedTensor:
        """Build a PyT ShardedTensor from given shards.

        During loading:
        - if data is None, initialize it with an empty tensor (will be used to copy the data into)
        - if `allow_shape_mismatch` is True, the data is initialized with zeros
            prior to loading (not all parts of the tensor will be read from the checkpoint)
        """
        assert all(isinstance(sh_ten, ShardedTensor) for sh_ten in sh_tens), sh_tens
        for sh_ten in sh_tens:
            if sh_ten.data is None:
                if is_loading:
                    sh_ten.init_data(
                        init_device,
                        init_fn=torch.zeros if sh_ten.allow_shape_mismatch else torch.empty,
                    )
                else:
                    raise CheckpointingException(f'`data` attr is None for {sh_ten}')
            else:
                sh_ten.data = sh_ten.data.detach()
                if sh_ten.allow_shape_mismatch and is_loading:
                    sh_ten.data.zero_()

        torch_sh_ten = sharded_tensor_to_torch_sharded_tensor(sh_tens, rank)
        torch_sh_ten.key = sh_tens[0].key
        return torch_sh_ten

    def _mcore_to_torch_sharded_object(sh_objs: List[ShardedObject]) -> io.BytesIO:
        """Build io.BytesIO from given sharded objects data."""
        assert all(isinstance(sh_obj, ShardedObject) for sh_obj in sh_objs), sh_objs
        serialized_data = io.BytesIO()
        torch.save([sh_obj.data for sh_obj in sh_objs], serialized_data)
        return serialized_data

    for k, v in state_dict.items():
        if isinstance(v[0], ShardedTensor):
            v = cast(List[ShardedTensor], v)
            pyt_state_dict[k] = _mcore_to_torch_sharded_tensor(v)
        else:
            v = cast(List[ShardedObject], v)
            pyt_state_dict[k] = _mcore_to_torch_sharded_object(v)

    return pyt_state_dict


def _unwrap_pyt_sharded_tensor(sh_ten: TorchShardedTensor) -> List[torch.Tensor]:
    """ Unwrap tensor from PyT ShardedTensor instance.

    If `prepend_axis_num` was non-zero (which is specific to MCore ShardedTensor)
    then the tensor has additional singleton dimensions which should be squeezed.
    """
    prepend_axis_num = getattr(sh_ten, 'prepend_axis_num', 0)
    if prepend_axis_num == 0:
        return [sh.tensor for sh in sh_ten.local_shards()]
    ret_tensors = []
    for sh in sh_ten.local_shards():
        ten = sh.tensor
        for _ in range(prepend_axis_num):
            ten = ten.squeeze(0)
        ret_tensors.append(ten)
    return ret_tensors


def _replace_state_dict_keys_with_sharded_keys(
    sharded_state_dict: ShardedStateDict, keep_only_main_replica: bool = False
) -> Tuple[Dict[str, List[ShardedBase]], FLATTEN_MAPPING, Dict[str, List[str]]]:
    """Group ShardedBase objects by keys and return mappings required for recreating the original dict. """
    flat_sd, flat_mapping = flatten_state_dict(sharded_state_dict)
    rename_mapping = defaultdict(list)
    new_flat_sd = defaultdict(list)
    for k, sh_base in flat_sd.items():
        assert isinstance(sh_base, ShardedBase), type(sh_base)
        key = sh_base.unique_key if isinstance(sh_base, ShardedObject) else sh_base.key
        if is_main_replica(sh_base.replica_id) or not keep_only_main_replica:
            rename_mapping[key].append(k)
            new_flat_sd[key].append(sh_base)
    return new_flat_sd, flat_mapping, rename_mapping


def _replace_sharded_keys_with_state_dict_keys(
    state_dict: Dict[str, List[Union[torch.Tensor, io.BytesIO]]],
    flat_mapping: FLATTEN_MAPPING,
    rename_mapping: Dict[str, List[str]],
):
    """ Inverse of _replace_state_dict_keys_with_sharded_keys. """
    recovered_sd = {}
    for k, tensors in state_dict.items():
        assert len(tensors) == len(rename_mapping[k])
        for ten, recovered_k in zip(tensors, rename_mapping[k]):
            recovered_sd[recovered_k] = ten

    return unflatten_state_dict(recovered_sd, flat_mapping)


def _restore_dict_types(x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):
    """ Recursively update `x` keys, based on `keys_template`. """
    if isinstance(keys_template, dict):
        assert isinstance(x, dict), type(x)
        for k, v in keys_template.items():
            if not isinstance(k, str):
                assert str(k) in x, (k, x.keys)
                x[k] = x.pop(str(k))
            _restore_dict_types(x[k], v)
    elif isinstance(keys_template, list):
        assert isinstance(x, list), type(x)
        for x_val, templ_val in zip(x, keys_template):
            _restore_dict_types(x_val, templ_val)


class MCoreSavePlanner(DefaultSavePlanner):
    """Differs with the default planner by saving BytesIO objects on all ranks.

    In the integration of MCore with PyT Distributed format, BytesIO objects
    come from ShardedObjects, which should be treated as separate objects on each rank
    (not common on all ranks).

    Also, the objects are already packed in io.BytesIO, so no need to redo it
    in transform_object.
    """

    def create_local_plan(self) -> SavePlan:
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        self._add_non_coordinator_iobytes_request(plan)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan

        return self.plan

    def _add_non_coordinator_iobytes_request(self, plan):
        if self.is_coordinator:
            return
        for fqn, obj in self.state_dict.items():
            if isinstance(obj, io.BytesIO):
                plan.items.extend(_create_write_items(fqn, obj))

    def transform_object(self, write_item: WriteItem, object: Any):
        return object


class MCoreLoadPlanner(DefaultLoadPlanner):
    """Adds global shape validation to the default planner.

    If global shape validation can be ignored (shouldn't!), the default
    load planner can be used.
    """

    def __init__(
        self, *args, shapes_validation_sharded_tensors: Iterable[ShardedTensor] = (), **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shapes_validation_sharded_tensors = shapes_validation_sharded_tensors

    def _validate_global_shapes(self, metadata, sharded_tensors):
        for sh_ten in sharded_tensors:
            loaded_shape = metadata.state_dict_metadata[sh_ten.key].size
            if loaded_shape != sh_ten.global_shape:
                _msg = (
                    f'Global shape mismatch for loaded ({loaded_shape})'
                    f' and expected ({sh_ten.global_shape}) tensor'
                    f' for key {sh_ten.key}'
                )
                raise CheckpointingException(_msg)

    def create_local_plan(self) -> LoadPlan:
        self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)
        return super().create_local_plan()


class TorchDistSaveShardedStrategy(AsyncSaveShardedStrategy):
    """Async save strategy for the PyT Distributed format.

    The idea is to translate MCore ShardedTensors into PyT ShardedTensors
    and use the async-adjusted torch.distributed.checkpoint saving mechanism
    provided by the FileSystemWriterAsync writer.
    """

    def __init__(
        self, backend: str, version: int, keep_only_main_replica: bool = True, thread_count: int = 2
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
        self.thread_count = thread_count

    def async_save(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> AsyncRequest:
        """ Translates MCore ShardedTensors to PyT ShardedTensors and saves in PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint directory

        Returns: None
        """
        # Translate the state dict
        (
            sharded_state_dict,
            flat_mapping,
            rename_mapping,
        ) = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict, self.keep_only_main_replica
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        # Use PyT saving mechanism
        writer = FileSystemWriterAsync(checkpoint_dir, thread_count=self.thread_count)

        save_state_dict_ret = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            planner=MCoreSavePlanner(dedup_replicated_tensors=not self.keep_only_main_replica),
        )
        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)

    def _get_save_and_finalize_callbacks(self, writer, save_state_dict_ret) -> AsyncRequest:
        save_fn_args = writer.get_save_function_and_args()
        save_fn, save_args = save_fn_args

        def finalize_fn():
            save_state_dict_async_finalize(*save_state_dict_ret)
            torch.distributed.barrier()

        return AsyncRequest(save_fn, save_args, [finalize_fn])

    def can_handle_sharded_objects(self):
        return True


class TorchDistLoadShardedStrategy(LoadShardedStrategy):
    """Basic load strategy for the PyT Distributed format. """

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors and loads from PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (
            sharded_state_dict,
            flat_mapping,
            rename_mapping,
        ) = _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, True)
        # Load PyT Distributed format
        checkpoint.load_state_dict(
            pyt_state_dict,
            FileSystemReader(checkpoint_dir),
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors
            ),
        )
        pyt_state_dict = cast(
            Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
        )
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        return mcore_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path):
        """Uses tensors metadata stored in the metadata file."""
        fs_reader = FileSystemReader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        return {
            k: ShardedTensor.from_rank_offsets(
                k, torch.empty(tp.size, **tp.properties.__dict__, device='meta')
            ).without_data()
            for k, tp in metadata.state_dict_metadata.items()
            if isinstance(tp, TensorStorageMetadata)
        }

    def can_handle_sharded_objects(self):
        return True

    def check_backend_compatibility(self, loaded_version):
        pass  # TODO

    def check_version_compatibility(self, loaded_version):
        pass  # TODO


default_strategies[StrategyAction.LOAD_SHARDED.value][
    ('torch_dist', 1)
] = TorchDistLoadShardedStrategy()
default_strategies[StrategyAction.SAVE_SHARDED.value][
    ('torch_dist', 1)
] = TorchDistSaveShardedStrategy('torch_dist', 1)
