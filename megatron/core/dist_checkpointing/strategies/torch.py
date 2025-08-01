# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Strategies using PyTorch distributed.checkpoint as an underlying format. """
import io
import os
import pickle
import warnings
from collections import ChainMap, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import product
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from packaging.version import Version as PkgVersion
from torch.distributed import checkpoint
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import Shard
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed._shard.sharded_tensor import ShardedTensorMetadata, TensorProperties
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    DefaultLoadPlanner,
    DefaultSavePlanner,
    FileSystemReader,
    FileSystemWriter,
    LoadPlan,
    Metadata,
    ReadItem,
    SavePlan,
    TensorStorageMetadata,
    WriteItem,
)
from torch.distributed.checkpoint._nested_dict import FLATTEN_MAPPING, unflatten_state_dict
from torch.distributed.checkpoint._traverse import OBJ_PATH, traverse_state_dict
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner_helpers import _create_write_items

from ...utils import get_torch_version, is_torch_min_version
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
    StrategyAction,
    register_default_strategy,
)
from .cached_metadata_filesystem_reader import CachedMetadataFileSystemReader
from .filesystem_async import FileSystemWriterAsync
from .resharding import (
    TensorReformulationMetadata,
    apply_nd_flattened_tensors_reformulation,
    is_nd_flattened_tensor,
    nd_flattened_tensor_reformulated_global_shape,
    restore_nd_flattened_tensors_formulation,
)
from .state_dict_saver import save_state_dict_async_finalize, save_state_dict_async_plan

try:
    if not torch.cuda.is_available():
        raise ImportError
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

try:
    from torch.distributed._tensor import DTensor

    HAVE_DTENSOR = True
except ImportError:
    HAVE_DTENSOR = False

from megatron.core.msc_utils import MultiStorageClientFeature

MSC_PREFIX = "msc://"

_metadata_fn: str = ".metadata"


def register_default_torch_strategies():
    """Register default strategies related to PyT Distributed backend."""
    register_default_strategy(
        StrategyAction.LOAD_SHARDED, 'torch_dist', 1, TorchDistLoadShardedStrategy()
    )
    register_default_strategy(
        StrategyAction.SAVE_SHARDED, 'torch_dist', 1, TorchDistSaveShardedStrategy('torch_dist', 1)
    )


logger = getLogger(__name__)


def flatten_state_dict(
    state_dict: ShardedStateDict,
) -> Tuple[ShardedStateDict, Dict[str, OBJ_PATH]]:
    """Flattens state dict into a single level dict.

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
    sh_tens: List[ShardedTensor],
    rank: Optional[int] = None,
    load_legacy_1d_flatten_tensors: bool = False,
) -> TorchShardedTensor:
    """Convert MCore ShardedTensor to PyT ShardedTensor. PyT requires information about all chunks.

    On high-level, this function follows the logic of
    torch.distributed.fsdp._shard_utils._create_chunk_sharded_tensor.
    Additionally, it saves `prepend_axis_num` and `has_flattened_range` (specific to MCore)
    as attributes for further restoration in `_unwrap_pyt_sharded_tensor`.

    NOTE: this function assumes regular (grid) sharding of the MCore ShardedTensor.
    The only local irregularities could be introduced with a `flattened_range` attribute.

    This function handles 2 different type of ShardedTensors:
    1. Non-flat regular ShardedTensors (`not has_flattened_range`)
    2. N-D flattened ShardedTensors (`has_flattened_range`)

    (1) type are saved according to their original shape.
    Type (2) however requires global shape adjustment for efficiency:
    we treat [X, Y, Z] global shape tensor with local shape [x, y, z]
    as a [X // x, Y // y, Z // z, x * y * z] tensor with last axis
    partitioned according to `flattened_range` slices.
    This will need special handling while resharding.

    Args:
        sh_tens (List[ShardedTensor]): list of sharded tensors to convert
        rank (int, optional): current process rank passed to PyT ShardedTensor.
            If None, assumes rank in the default pg.
        load_legacy_1d_flatten_tensors (bool, optional): flag indicating if 1-D flattened tensors
            should be loaded in a legacy way. Defaults to False.

    Returns (TorchShardedTensor): PyT ShardedTensor containing all passed shards.

    """
    if rank is None:
        rank = torch.distributed.get_rank()

    some_sh_ten = sh_tens[0]
    has_flattened_range = some_sh_ten.flattened_range is not None

    for sh_ten in sh_tens:
        assert (sh_ten.flattened_range is not None) == has_flattened_range, sh_tens
        if not sh_ten.data.is_contiguous():
            sh_ten.data = sh_ten.data.contiguous()

    if load_legacy_1d_flatten_tensors and len(some_sh_ten.global_shape) == 1:
        # Legacy 1-D flattened tensors are loaded as non-flat regular ShardedTensors
        has_flattened_range = False

    local_global_offsets = {}

    prepend_axis_num = sh_tens[0].prepend_axis_num
    # Determine local shards according to tensor type (see docs)
    if has_flattened_range:
        # Type (3) case: N-D flattened ShardedTensors
        for sh_ten in sh_tens:
            local_global_offsets.setdefault(sh_ten.local_chunk_offset_in_global(), []).append(
                sh_ten
            )
            assert sh_ten.data.ndim == 1, sh_ten
            sh_ten.data = sh_ten.data.view((1,) * len(sh_ten.global_shape) + (-1,))

        # Global shape reformulation:
        global_shape = nd_flattened_tensor_reformulated_global_shape(some_sh_ten)
        offsets_shape = (1,) * len(
            some_sh_ten.global_shape
        )  # reformulated global shape has shape equal ti number of local chunks

        local_shards = [
            Shard.from_tensor_and_offsets(
                sh_ten.data,
                list(
                    sh_ten.local_chunk_offset_in_global() + (sh_ten.flattened_range.start,)
                ),  # additional flattened offset
                rank,
            )
            for sh_ten in sh_tens
        ]
    else:
        # Type (1) case: non-flat regular ShardedTensors
        for sh_ten in sh_tens:
            local_global_offsets.setdefault(sh_ten.global_offset, []).append(sh_ten)
            sh_ten.data = sh_ten.data.view(
                (1,) * prepend_axis_num + sh_ten.local_shape
            )  # adjust to prepended_axis_num

        global_shape = some_sh_ten.global_shape
        offsets_shape = some_sh_ten.data.shape  # includes prepended axes

        local_shards = [
            Shard.from_tensor_and_offsets(
                sh_ten.data, list(sh_ten.global_offset), rank  # simple case
            )
            for sh_ten in sh_tens
        ]

    # Create a ShardedTensor without invoking communication. Determine global shards
    world_size = torch.distributed.get_world_size()
    shard_metadata = []
    # NOTE: here we assume a regular grid of shards
    for fragment_offsets in product(*map(range, some_sh_ten.axis_fragmentations)):
        offset = tuple(map(lambda x: x[0] * x[1], zip(fragment_offsets, offsets_shape)))
        if offset in local_global_offsets:
            # local shard
            placement = f"rank:{rank}/cuda"
            for sh_ten in local_global_offsets[offset]:
                if has_flattened_range:
                    assert offset == sh_ten.local_chunk_offset_in_global()
                    # This is not an actual offset, but an offset of the whole shard
                    # This is needed for a PyT Dist internal integrity check
                    offset = sh_ten.local_chunk_offset_in_global() + (0,)
                    size = (1,) * len(offsets_shape) + global_shape[-1:]
                else:
                    size = sh_ten.data.shape
                shard_metadata.append(ShardMetadata(offset, size, placement))

        else:
            # pylint: disable=line-too-long
            # for shards from other ranks we provide simplistic data - this information will be discarded
            # during TorchShardedTensor._init_from_local_shards_and_global_metadata call.
            # Due to a bug in PyT 24.05 container we must specify some concrete rank within a world size.
            # The exact rank doesn't matter as long as it's different than my rank - hence (rank + 1) % WS.
            placement = f"rank:{(rank + 1) % world_size}/cuda"
            if has_flattened_range:
                offset = offset + (0,)
                size = (1,) * len(offsets_shape) + global_shape[-1:]
            else:
                size = offsets_shape
            shard_metadata.append(ShardMetadata(offset, size, placement))

    tensor = some_sh_ten.data
    sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=shard_metadata,
        size=torch.Size(global_shape),
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
    # Store MCore related data as PyTShardedTensor attribute.
    # This won't be stored in the checkpoint, only for runtime purposes
    pyt_sh_ten.mcore_sh_ten = sh_ten.without_data()
    pyt_sh_ten.mcore_metadata = {}
    if has_flattened_range:
        pyt_sh_ten.mcore_metadata['nd_reformulated_orig_global_shape'] = sh_ten.global_shape
    return pyt_sh_ten


def mcore_to_pyt_state_dict(
    state_dict: Dict[str, List[ShardedBase]],
    is_loading: bool = False,
    init_device: torch.device = torch.device("cpu"),
    load_legacy_1d_flatten_tensors: bool = False,
) -> Dict[str, Union[TorchShardedTensor, io.BytesIO]]:
    """Convert state dict with ShardedTensors and ShardedObjects
    to state dict compatible with PyT Dist format.

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

        torch_sh_ten = sharded_tensor_to_torch_sharded_tensor(
            sh_tens, rank, load_legacy_1d_flatten_tensors
        )
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
    """Unwrap tensor from PyT ShardedTensor instance.

    If `prepend_axis_num` was non-zero (which is specific to MCore ShardedTensor)
    then the tensor has additional singleton dimensions which should be squeezed.
    """
    mcore_sh_ten = sh_ten.mcore_sh_ten
    ret_tensors = []
    for sh in sh_ten.local_shards():
        ten = sh.tensor
        if mcore_sh_ten.flattened_range is not None:
            assert ten.shape[:-1] == (1,) * (len(ten.shape) - 1), ten.shape
            ten = ten.view(-1)
        else:
            for _ in range(mcore_sh_ten.prepend_axis_num):
                assert ten.size(0) == 1
                ten = ten[0]  # NOTE: ten.squeeze(0) uses more memory for FP8 tensors
        ret_tensors.append(ten)
    return ret_tensors


def _replace_state_dict_keys_with_sharded_keys(
    sharded_state_dict: ShardedStateDict, keep_only_main_replica: bool = False
) -> Tuple[Dict[str, List[ShardedBase]], FLATTEN_MAPPING, Dict[str, List[str]]]:
    """Group ShardedBase objects by keys and
    return mappings required for recreating the original dict."""
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
    """Inverse of _replace_state_dict_keys_with_sharded_keys."""
    recovered_sd = {}
    for k, tensors in state_dict.items():
        assert len(tensors) == len(rename_mapping[k])
        for ten, recovered_k in zip(tensors, rename_mapping[k]):
            recovered_sd[recovered_k] = ten

    return unflatten_state_dict(recovered_sd, flat_mapping)


def _restore_dict_types(x: Union[dict, list, Any], keys_template: Union[dict, list, Any]):
    """Recursively update `x` keys, based on `keys_template`."""
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


@dataclass(frozen=True)
class MCoreSavePlan(SavePlan):
    """SavePlan with MCore specific data."""

    mcore_data: Optional[Dict[str, Dict[str, Any]]] = None  # Mcore related data about each tensor


class MCoreSavePlanner(DefaultSavePlanner):
    """Differs with the default planner by saving BytesIO objects on all ranks.

    In the integration of MCore with PyT Distributed format, BytesIO objects
    come from ShardedObjects, which should be treated as separate objects on each rank
    (not common on all ranks).

    Also, the objects are already packed in io.BytesIO, so no need to redo it
    in transform_object.
    """

    def __init__(
        self,
        *args,
        dedup_replicated_tensors: Optional[bool] = None,
        nd_flattened_global_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        can_run_decentralized_global_plan: bool = True,
        **kwargs,
    ) -> None:
        # `dedup_replicated_tensors` was deprecated in 2.3; this check avoids warnings
        # during saving.
        if get_torch_version() <= PkgVersion("2.2"):
            kwargs['dedup_replicated_tensors'] = dedup_replicated_tensors
        super().__init__(*args, **kwargs)
        self.nd_flattened_global_shapes = nd_flattened_global_shapes or {}
        self.can_run_decentralized_global_plan = can_run_decentralized_global_plan
        if can_run_decentralized_global_plan:
            assert (
                not dedup_replicated_tensors
            ), 'Cannot run decentralized plan with dedup_replicated_tensors=True'
            assert (
                not self.flatten_state_dict
            ), 'Cannot run decentralized plan with flatten_state_dict=True'

    def create_local_plan(self) -> SavePlan:
        """Adds IOBytes write request on non-coordinator ranks."""

        # NOTE: for PyT 2.4.0a0 we can't rely on `create_default_local_save_plan` because
        # some alpha versions (specifically 2.4.0a0+f70bd71a48 in 24.06 NGC PyTorch container)
        # add iobytes request only on coordinator ranks and some alpha versions
        # (specifically 2.4.0a0+3bcc3cddb5 in 24.07 NGC PyTorch container)
        # add those requests on all ranks. We inline a simplified version of this method below.
        write_items = []
        for fqn, obj in self.state_dict.items():
            assert not HAVE_DTENSOR or not isinstance(
                obj, DTensor
            )  # translation from MCore ShardedTensors shouldn't result in DTensors
            # Create write requests for tensor and bytes values.
            # For MCore, these should be already non-duplicates.
            write_items += _create_write_items(fqn, obj)

        self.plan = MCoreSavePlan(
            items=write_items,
            planner_data=self.mappings,
            mcore_data={
                k: sh_ten.mcore_metadata
                for k, sh_ten in self.state_dict.items()
                if isinstance(sh_ten, TorchShardedTensor)
            },
        )
        return self.plan

    def create_global_plan(self, all_plans: List[MCoreSavePlan]) -> Tuple[List[SavePlan], Metadata]:
        """Merges MCore data for all plans."""
        global_plan, metadata = super().create_global_plan(all_plans)
        metadata.mcore_data = dict(
            ChainMap(*(plan.mcore_data for plan in all_plans))  # type: ignore[arg-type]
        )
        return global_plan, metadata

    def create_decentralized_global_plan(self, local_plan: SavePlan) -> SavePlan:
        """Nothing to do, just some checks.

        Args:
            local_plan (SavePlan): local plan to turn to a global plan
                (without interactions with other ranks)

        Returns:
            SavePlan - locally transformed plan equivalent to the plan that would be
                created by the coordinator
        """
        assert (
            not self.flatten_state_dict
        ), 'Cannot run decentralized plan with flatten_state_dict=True'
        assert not local_plan.planner_data, 'Planner data should be empty with decentralized plan'
        return local_plan

    def transform_object(self, write_item: WriteItem, object: Any):
        """Make no transformations - bytes objects are already serialized."""
        return object


class MCoreLoadPlanner(DefaultLoadPlanner):
    """Adds global shape validation to the default planner.

    If global shape validation can be ignored (shouldn't!), the default
    load planner can be used.
    """

    def __init__(
        self,
        *args,
        shapes_validation_sharded_tensors: Iterable[ShardedTensor] = (),
        allow_shape_mismatch_sharded_tensors: Optional[Dict[str, ShardedTensor]] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.shapes_validation_sharded_tensors = shapes_validation_sharded_tensors
        self.allow_shape_mismatch_sharded_tensors = allow_shape_mismatch_sharded_tensors
        self._intermediate_read_item_and_target: Optional[Tuple[ReadItem, torch.Tensor]] = None

    @staticmethod
    def _expected_shape(sh_ten):
        return (
            nd_flattened_tensor_reformulated_global_shape(sh_ten)
            if is_nd_flattened_tensor(sh_ten)
            else sh_ten.global_shape
        )

    def _validate_global_shapes(self, metadata, sharded_tensors):
        for sh_ten in sharded_tensors:
            if sh_ten.key not in metadata.state_dict_metadata:
                raise KeyError(
                    f"{sh_ten.key} from model not in state dict:"
                    f" {sorted(metadata.state_dict_metadata.keys())}"
                )
            loaded_shape = metadata.state_dict_metadata[sh_ten.key].size
            expected_shape = self._expected_shape(sh_ten)
            if loaded_shape != expected_shape:
                if is_nd_flattened_tensor(sh_ten) and len(sh_ten.global_shape) == 1:
                    # Handle legacy 1-D flattened tensors checkpoint format
                    # where the global shape is not stored in the metadata
                    expected_shape = sh_ten.global_shape
                    if loaded_shape == expected_shape:
                        continue
                _msg = (
                    f'Global shape mismatch for loaded ({loaded_shape})'
                    f' and expected ({expected_shape}) tensor'
                    f' for key {sh_ten.key}'
                )
                raise CheckpointingException(_msg)

    @contextmanager
    def _temporarily_bypass_shape_validation(self):
        """
        Temporarily set the size of tensors to their expected shapes to bypass DCP shape validation.
        This is used when validating the shapes during local plan creation.
        """
        if not self.allow_shape_mismatch_sharded_tensors:
            yield
            return

        tensor_metadata = self.metadata.state_dict_metadata
        metadata_with_sizes = [
            (tensor_metadata[key], tensor_metadata[key].size, sharded_tensor)
            for key, sharded_tensor in self.allow_shape_mismatch_sharded_tensors.items()
        ]
        try:
            # Temporarily set sizes to expected shapes
            for md, _, sharded_tensor in metadata_with_sizes:
                md.size = self._expected_shape(sharded_tensor)
            yield
        finally:
            # Restore original sizes after yield
            for md, size, _ in metadata_with_sizes:
                md.size = size

    def create_local_plan(self) -> LoadPlan:
        """Runs additional shapes validation."""
        self._validate_global_shapes(self.metadata, self.shapes_validation_sharded_tensors)

        with self._temporarily_bypass_shape_validation():
            local_plan = super().create_local_plan()

        return local_plan

    def resolve_tensor(self, read_item: ReadItem):
        """Override to add FP8 support.

        Narrowing the Float8Tensor can create incontiguous tensors and there are
        no `copy` kernels for such cases. This method creates a contiguous FP8
        tensors so that the subsequent `copy_` in FileSystemReader succeeds.
        Note that this requires tracking the original tensor
        (as `self._intermediate_read_item_and_target` attribute)
        and restoring it in `commit_tensor` method.
        """
        target_tensor = super().resolve_tensor(read_item)
        if (
            not target_tensor.is_contiguous()
            and HAVE_TE
            and isinstance(target_tensor, Float8Tensor)
        ):
            self._intermediate_read_item_and_target = (read_item, target_tensor)
            target_tensor = Float8Tensor.make_like(
                target_tensor, data=target_tensor._data.contiguous()
            )
        return target_tensor

    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None:
        """Restores the original FP8 tensor saved in `resolve_tensor`."""
        if self._intermediate_read_item_and_target is not None:
            interm_read_item, target_tensor = self._intermediate_read_item_and_target
            assert (
                interm_read_item is read_item
            ), '`commit_tensor` method should be called right after `resolve_tensor`'
            target_tensor.copy_(tensor)
            tensor = target_tensor
            self._intermediate_read_item_and_target = None
        return super().commit_tensor(read_item, tensor)


class TorchDistSaveShardedStrategy(AsyncSaveShardedStrategy):
    """Async save strategy for the PyT Distributed format.

    The idea is to translate MCore ShardedTensors into PyT ShardedTensors
    and use the async-adjusted torch.distributed.checkpoint saving mechanism
    provided by the FileSystemWriterAsync writer.
    """

    def __init__(
        self,
        backend: str,
        version: int,
        keep_only_main_replica: bool = True,
        thread_count: int = 2,
        cached_metadata: bool = False,
        separation_hint: Optional[str] = None,
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
            cached_metadata (bool, optional): Enables using cached global metadata to avoid
                gathering local metadata every checkpointing invocation
            separation_hint(str, optional): If provided, all tensors whose keys have this
                prefix will be saved to a separate file.
        """
        super().__init__(backend, version)
        self.keep_only_main_replica = keep_only_main_replica
        self.thread_count = thread_count

        # Cached SavePlans to skip plan in `save_state_dict_async_plan`
        # cached outcome of `SavePlan.prepare_global_plan`,
        # which aggregates local plans from all ranks
        self.cached_central_plan: SavePlan = None
        # cached outcome of `SavePlan.prepare_local_plan` describes how local state_dict is written
        self.cached_local_plan: SavePlan = None
        # Cached global metadata, only `coordinator` for dist-ckpt holds
        # if central plans are consistent over iters
        self.cached_global_metadata: Metadata = None
        # This variable records if the ckpt structures are consistent
        # so the following checkpoint savings reuse `cached_global_metadata`
        self.validated_cache_reuse: bool = False
        # The knob to enable cached metadata communication in saving
        self.use_cached_ckpt_structure: bool = cached_metadata

        self.separation_hint = separation_hint

        self.validated_loaded_metadata_reuse = False

    def async_save(
        self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
    ) -> AsyncRequest:
        """Translates MCore ShardedTensors to PyT ShardedTensors & saves in PyT Distributed format.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict to save
            checkpoint_dir (Path): checkpoint directory

        Returns: None
        """
        # Translate the state dict
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(
                sharded_state_dict, self.keep_only_main_replica
            )
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, False)
        # Use PyT saving mechanism
        writer = FileSystemWriterAsync(
            checkpoint_dir,
            separation_hint=self.separation_hint,
            thread_count=self.thread_count,
            use_msc=MultiStorageClientFeature.is_enabled(),
        )
        # This should be set differently if we run in a smaller process group than the default
        coordinator = 0
        # Try twice to validate the generated `central_plan` is the same across iterations
        # If so, reuse `cached_central_plan` and `cached_global_metadata`
        # From the 3rd iteration, `save_state_dict_async_plan` will not generate `global_metadata`
        # (return None) so `self.cached_global_metadata` is reused
        args_cached_plans = None
        loaded_all_plans = None
        if self.use_cached_ckpt_structure:
            loaded_all_plans = getattr(self.cached_global_metadata, "all_local_plans", None)
            if loaded_all_plans is None:
                logger.debug(
                    "no all_local_plans in metadata - can't verify global metadata reuse..."
                )

            args_cached_plans = (
                self.cached_central_plan,
                self.cached_local_plan,
                self.validated_cache_reuse,
            )

        (
            save_state_dict_ret,
            self.cached_central_plan,
            self.cached_local_plan,
            self.validated_cache_reuse,
            self.validated_loaded_metadata_reuse,
        ) = save_state_dict_async_plan(
            pyt_state_dict,
            writer,
            None,
            coordinator,
            planner=MCoreSavePlanner(
                dedup_replicated_tensors=not self.keep_only_main_replica, flatten_state_dict=False
            ),
            cached_ckpt_structure=args_cached_plans,
            loaded_all_plans=loaded_all_plans,
        )
        rank = torch.distributed.get_rank()
        if self.use_cached_ckpt_structure:
            if (
                loaded_all_plans
                and self.cached_global_metadata
                and self.validated_loaded_metadata_reuse
            ):
                if coordinator == rank:
                    logger.debug(
                        f"rank: {rank}, reuse global metadata from loaded"
                        f" .metadata, {save_state_dict_ret[1]}"
                    )
                    save_state_dict_ret = list(save_state_dict_ret)
                    save_state_dict_ret[1] = self.cached_global_metadata

            elif self.validated_cache_reuse:
                logger.debug(f"rank: {rank}, cache validated")
                if save_state_dict_ret[1]:  # when global_metadata is not cached
                    self.cached_global_metadata = save_state_dict_ret[1]  # Cache Metadata
                # Only Coordinator rank holds cached global_metadata
                # (None is returned for global_metadata)
                elif coordinator == rank:
                    logger.debug(
                        f"rank: {rank}, reuse global metadata cached from previous"
                        f" save iteration, {save_state_dict_ret[1]}"
                    )
                    save_state_dict_ret = list(save_state_dict_ret)
                    save_state_dict_ret[1] = self.cached_global_metadata

        return self._get_save_and_finalize_callbacks(writer, save_state_dict_ret)

    def _get_save_and_finalize_callbacks(self, writer, save_state_dict_ret) -> AsyncRequest:
        save_fn_args = writer.get_save_function_and_args()
        save_fn, preload_fn, save_args = save_fn_args

        def finalize_fn():
            save_state_dict_async_finalize(*save_state_dict_ret)
            torch.distributed.barrier()

        return AsyncRequest(save_fn, save_args, [finalize_fn], preload_fn=preload_fn)

    def can_handle_sharded_objects(self):
        return True


def _get_filesystem_reader(
    checkpoint_dir: Union[str, Path], cache_metadata: bool = False
) -> FileSystemReader:
    if MultiStorageClientFeature.is_enabled():
        msc = MultiStorageClientFeature.import_package()
        return msc.torch.MultiStorageFileSystemReader(checkpoint_dir, thread_count=2)

    if cache_metadata:
        return CachedMetadataFileSystemReader(checkpoint_dir)

    return FileSystemReader(checkpoint_dir)


def get_reformulation_metadata(
    sharded_state_dict: ShardedStateDict, checkpoint_dir: Path
) -> Dict[str, TensorReformulationMetadata]:
    """Reads MCore data for N-D flattened tensors from checkpoint metadata during ckpt load.

    Args:
        sharded_state_dict (ShardedStateDict): sharded state dict to load
        checkpoint_dir (Path): checkpoint directory

    Returns:
        Dict[str, TensorReformulationMetadata] - dictionary that maps keys of every
            N-D flattened tensor from the sharded_state_dict to its original global shape
            as stored in `mcore_data` in the checkpoint.
    """
    fs_reader = _get_filesystem_reader(checkpoint_dir)
    ckpt_metadata = fs_reader.read_metadata()
    reformulation_metadata = {}
    for sh_ten in nested_values(sharded_state_dict):
        if not is_nd_flattened_tensor(sh_ten):
            continue
        try:
            ckpt_global_shape = ckpt_metadata.mcore_data[sh_ten.key][
                'nd_reformulated_orig_global_shape'
            ]
        except KeyError as e:
            if len(sh_ten.global_shape) == 1:
                warnings.warn(
                    f'Legacy checkpoint format detected for 1-D flattened tensor {sh_ten}. '
                    'Skip metadata reformulation.'
                )
                continue
            raise CheckpointingException(
                f'Cannot find global shape metadata for N-D flattened tensor {sh_ten} '
                f'in checkpoint metadata: {ckpt_metadata.mcore_data}'
            ) from e

        reformulation_metadata[sh_ten.key] = TensorReformulationMetadata(
            ckpt_global_shape, ckpt_metadata.state_dict_metadata[sh_ten.key].size
        )
    return reformulation_metadata


class TorchDistLoadShardedStrategy(LoadShardedStrategy):
    """Basic load strategy for the PyT Distributed format."""

    def __init__(self):
        self.cached_global_metadata: Optional[Metadata] = None
        super().__init__()

    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Path) -> StateDict:
        """Translates MCore ShardedTensors to PyT ShardedTensors & loads from PyT Distributed fmt.

        Args:
            sharded_state_dict (ShardedStateDict): sharded state dict with mapping
                information to instruct loading
            checkpoint_dir (Path): checkpoint directory

        Returns: loaded state dict
        """
        # Apply N-D tensors resharding
        reformulation_metadata = get_reformulation_metadata(sharded_state_dict, checkpoint_dir)
        sharded_state_dict, formulation_restore_data = apply_nd_flattened_tensors_reformulation(
            sharded_state_dict, reformulation_metadata
        )

        # Check if there are legacy 1-D flattened tensors in the checkpoint
        has_legacy_1d_flattened_tensors = False
        for sh_ten in nested_values(sharded_state_dict):
            if is_nd_flattened_tensor(sh_ten) and sh_ten.key not in reformulation_metadata:
                has_legacy_1d_flattened_tensors = True
                break

        flexible_shape_sharded_tensors = [
            sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and not sh_ten.allow_shape_mismatch
        ]
        allow_shape_mismatch_sharded_tensors = {
            sh_ten.key: sh_ten
            for sh_ten in nested_values(sharded_state_dict)
            if isinstance(sh_ten, ShardedTensor) and sh_ten.allow_shape_mismatch
        }

        orig_sharded_state_dict = sharded_state_dict
        # MCore state dict to PyT Distributed compatible
        (sharded_state_dict, flat_mapping, rename_mapping) = (
            _replace_state_dict_keys_with_sharded_keys(sharded_state_dict)
        )
        pyt_state_dict = mcore_to_pyt_state_dict(
            sharded_state_dict, True, load_legacy_1d_flatten_tensors=has_legacy_1d_flattened_tensors
        )
        # Load PyT Distributed format
        fsr = _get_filesystem_reader(checkpoint_dir, cache_metadata=True)
        checkpoint.load_state_dict(
            pyt_state_dict,
            fsr,
            planner=MCoreLoadPlanner(
                shapes_validation_sharded_tensors=flexible_shape_sharded_tensors,
                allow_shape_mismatch_sharded_tensors=allow_shape_mismatch_sharded_tensors,
            ),
        )

        self.cached_global_metadata = (
            fsr.read_metadata()
        )  # no storage interaction thanks to caching

        pyt_state_dict = cast(
            Dict[str, Union[TorchShardedTensor, List[io.BytesIO]]], pyt_state_dict
        )
        # Unwrap ShardedTensors and return to original state dict
        mcore_state_dict = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mcore_state_dict = _replace_sharded_keys_with_state_dict_keys(
            mcore_state_dict, flat_mapping, rename_mapping  # type: ignore[arg-type]
        )
        _restore_dict_types(mcore_state_dict, orig_sharded_state_dict)
        # Apply N-D tensors resharding postprocessing
        mcore_state_dict = restore_nd_flattened_tensors_formulation(
            mcore_state_dict, formulation_restore_data
        )
        return mcore_state_dict

    def load_tensors_metadata(self, checkpoint_dir: Path, metadata: Metadata = None):
        """Uses tensors metadata stored in the metadata file."""
        if metadata is None:
            fs_reader = _get_filesystem_reader(checkpoint_dir)
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
                    k, torch.empty(tp.size, **tp.properties.__dict__, device='meta')
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
        fs_reader = _get_filesystem_reader(checkpoint_dir)
        metadata = fs_reader.read_metadata()

        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
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
        original_planner_data = original_metadata.planner_data
        if original_planner_data is not None:
            for k in original_planner_data.keys():
                if k.startswith(key_prefix):
                    continue
                new_planner_data[k] = original_metadata.planner_data[k]
        original_storage_data = original_metadata.storage_data
        if original_storage_data is not None:
            for k in original_storage_data.keys():
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
            metadata_filename,  # type: ignore[valid-type]
            fs_writer.fs.concat_path(fs_writer.path, f"{_metadata_fn}.tmp"),
        )
        old_path = cast(
            metadata_filename,  # type: ignore[valid-type]
            fs_writer.fs.concat_path(fs_writer.path, f"{_metadata_fn}.bck"),
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
