# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

""" Utilities for transforming state_dict, including a tensor-aware implementation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from nvidia_resiliency_ext.checkpointing.local.base_state_dict import TensorAwareStateDict

from .dict_utils import dict_list_map_inplace, dict_list_map_outplace, merge, nested_values
from .exchange_utils import (
    ShardDistribution,
    determine_main_replica_uniform_distribution,
    exchange_by_distribution,
)
from .mapping import ShardedObject, ShardedStateDict, ShardedTensor, StateDict, apply_factory_merges
from .state_dict_utils import load_preprocess, save_preprocess
from .utils import (
    _sharded_object_id,
    _sharded_tensor_shard_id,
    debug_time,
    extract_sharded_base,
    zip_strict,
)
from .validation import (
    StrictHandling,
    determine_global_metadata,
    parse_strict_flag,
    validate_integrity_and_strict_load,
)

logger = logging.getLogger(__name__)


@dataclass
class MCoreTensorAwareStateDict(TensorAwareStateDict):
    """
    MCore-specific class defining the interface between the MCore state dict and checkpoint manager.

    This class distinguishes between raw objects, the common state dict, and sharded state dicts
    (tensor parts). It also handles optional metadata needed for fully parallel save/load.
    """

    common: StateDict
    sharded_state_dict: ShardedStateDict
    _is_hollow: bool = False

    @staticmethod
    def _validate_params(algo):
        if algo != 'atomic' and algo != 'fully_parallel':
            raise NotImplementedError(
                'Only "atomic" and "fully_parallel" sharding algorithms are supported.'
            )

    @staticmethod
    def _get_distribution(
        fully_parallel, sharded_part, parallelization_group, cached_distribution=None
    ):
        if fully_parallel:
            if cached_distribution is None:
                distribution = determine_main_replica_uniform_distribution(
                    sharded_part, parallelization_group, True
                )
                logger.debug(f'MCore_TASD._get_distribution calculated distribution')
            else:
                distribution = cached_distribution
                logger.debug(f'MCore_TASD._get_distribution used cache')
        else:
            distribution = (None, None, None, None)
            logger.debug(f'MCore_TASD._get_distribution returned empty distribution')
        return distribution

    @staticmethod
    def _remove_redundant_data(
        fully_parallel, sharded_part, shard_to_saving_rank, parallelization_group
    ):
        if parallelization_group is None:
            parallelization_group = torch.distributed.group.WORLD
        if fully_parallel:
            for sh_base in nested_values(sharded_part):
                # TODO remove redundant objects as well
                if isinstance(sh_base, ShardedTensor):
                    shard_id = _sharded_tensor_shard_id(sh_base)
                    if shard_to_saving_rank[shard_id] != parallelization_group.rank():
                        sh_base.data = None

    @classmethod
    @debug_time("from_state_dict", logger)
    def from_state_dict(
        cls,
        sharded_state_dict: ShardedStateDict,
        algo: str = 'fully_parallel',
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        cached_metadata: ShardDistribution = None,
    ) -> Tuple[TensorAwareStateDict, ShardDistribution]:
        """
        Constructs a TensorAwareStateDict from a sharded state dictionary.

        This method preprocesses the input `sharded_state_dict`, validates parameters,
        and extracts the necessary data to create an instance of `MCoreTensorAwareStateDict`.

        Args:
            sharded_state_dict: The input sharded state dictionary to be converted.
            algo (str, optional): Initialization algorithm. Defaults to 'fully_parallel'.
                - 'fully_parallel' enables fully parallel initialization.
            parallelization_group (Optional): A distributed process group for parallelization.
            cached_metadata (Optional): Precomputed metadata from previous saves.
                - Reuses data that doesn't need recalculation, optimizing the creation process.

        Returns:
            TensorAwareStateDict: An instance initialized with the provided sharded state dictionary
            and optional cached metadata.
            - The metadata is stored in memory to speed up future saves.
        """
        with debug_time("_get_distribution", logger):
            cls._validate_params(algo)
            fully_parallel = algo == 'fully_parallel'
            sharded_part, common_state_dict = save_preprocess(
                sharded_state_dict, cached_metadata is None
            )
            cacheable_distribution = cls._get_distribution(
                fully_parallel, sharded_part, parallelization_group, cached_metadata
            )
        if cacheable_distribution is not None:
            shard_to_saving_rank, _, _, _ = cacheable_distribution
            cls._remove_redundant_data(
                fully_parallel, sharded_part, shard_to_saving_rank, parallelization_group
            )

        return (
            MCoreTensorAwareStateDict(common=common_state_dict, sharded_state_dict=sharded_part),
            cacheable_distribution,
        )

    @property
    def is_hollow(self):
        """
        True iff tensors had been extracted and have not been inserted back yet.
        """
        return self._is_hollow

    @property
    def _sharded_tensors(self):
        # Three possible states for sharded_tensor:
        # 1. sharded_tensor with data (.data = tensor)
        # 2. sharded_tensor hollow (.data = None, .orig_device = orig_device)
        # 3. removed sharded_tensor (.data = None, no device information)
        # TODO: Consider simplifying by removing the entire sharded_tensor instead of just the data
        if self.is_hollow:
            for sh_base in nested_values(self.sharded_state_dict):
                # FIXME: Hacky way to store the original device of the popped tensor
                if isinstance(sh_base, ShardedTensor) and hasattr(sh_base, 'orig_device'):
                    yield sh_base
        else:
            for sh_base in nested_values(self.sharded_state_dict):
                if isinstance(sh_base, ShardedTensor) and sh_base.data is not None:
                    yield sh_base

    @property
    def tensors(self) -> Iterator[torch.Tensor]:
        """
        Get the tensor data from the state dict.
        """
        assert not self.is_hollow  # TODO raise exception
        return map(lambda sh_ten: sh_ten.data, self._sharded_tensors)

    @property
    def common_state_dict(self) -> Dict:
        """
        Get the common state dict from the state dict.
        """
        return self.common

    def pop_tensors(self) -> List[torch.Tensor]:
        """
        Extracts the tensor data from the wrapped state dict, preserving metadata.

        Replaces the tensor data in sharded_tensors with device type of extracted tensors.
        After this operation, the state dictionary is "hollow", containing no tensor data.
        Further calls to `pop_tensor` will raise an error.

        @return List of extracted tensors
        """
        assert not self.is_hollow  # TODO raise exception
        result = []
        for sh_ten in self._sharded_tensors:
            result.append(sh_ten.data)
            # FIXME: Hacky way to store the original device, which is not included in the metadata
            setattr(sh_ten, 'orig_device', sh_ten.data.device.type)
            sh_ten.data = None
        self._is_hollow = True
        return result

    def insert_tensors(self, tensor_data: Iterable[torch.Tensor]):
        """
        Reverse of `pop_tensors`. Replaces device type in sharded_tensors with actual values
        Value of `self` is considered to be the same after:
            ```
            self.insert_tensors(self.pop_tensors())
            ```
        """
        assert self.is_hollow  # TODO raise exception
        for sh_ten, ten in zip_strict(self._sharded_tensors, tensor_data):
            # FIXME: Hacky way to store the original device
            if sh_ten.orig_device == ten.device.type:
                delattr(sh_ten, 'orig_device')
            # Tensor might be on non-original device
            sh_ten.data = ten
        self._is_hollow = False

    def init_tensors(self):
        """
        Initializes empty tensors with the same properties as the original tensors.

        This function should only be called after the original tensors have been popped.
        It ensures that the newly created empty tensors match the shape,
        dtype, and device of the originals, but contain no data.
        """
        assert self.is_hollow  # TODO raise exception
        for sh_ten in self._sharded_tensors:
            # Hacky way to retrieve the original device
            sh_ten.init_data(sh_ten.orig_device)
            delattr(sh_ten, 'orig_device')
        self._is_hollow = False

    def copy_tensors_to_cpu(self, non_blocking=False):
        """
        Stores CPU copies of tensors in the state_dict, replacing the originals,
        but without destroying them.
        The original devices are remembered for restoration with restore_tensor_device().
        Using non_blocking=True allows for asynchronous copying.
        """
        assert not self.is_hollow  # TODO raise exception
        for sh_ten in self._sharded_tensors:
            if sh_ten.data.device.type == 'cpu':
                # Skip cloning if it's already confirmed to be a copy
                if not hasattr(sh_ten, 'orig_device'):
                    sh_ten.data = sh_ten.data.clone()
            else:
                # FIXME: Hacky way to store the original device
                if not hasattr(sh_ten, 'orig_device'):
                    setattr(sh_ten, 'orig_device', sh_ten.data.device.type)
                sh_ten.data = sh_ten.data.detach().to("cpu", non_blocking=non_blocking)

    def restore_tensor_device(self, non_blocking=True):
        """
        Restores all tensors to their original devices, if a move is required.
        Using non_blocking=True allows for asynchronous copying.
        """
        assert not self.is_hollow  # TODO raise exception
        for sh_ten in self._sharded_tensors:
            # FIXME: Hacky way to store the original device
            if hasattr(sh_ten, 'orig_device'):
                sh_ten.data = sh_ten.data.to(sh_ten.orig_device, non_blocking=non_blocking)
                delattr(sh_ten, 'orig_device')

    def _insert_sharded_data(
        self, fully_parallel, sharded_part, parallelization_group, exchange_algo
    ):
        loaded_tensors = {}
        for sh_ten in self._sharded_tensors:
            loaded_tensors[_sharded_tensor_shard_id(sh_ten)] = sh_ten.data
        if fully_parallel:
            with debug_time("_get_distribution", logger):
                distribution = self._get_distribution(
                    fully_parallel, sharded_part, parallelization_group
                )
            if distribution is not None:
                unloaded_shards = {}
                for sh_base in nested_values(sharded_part):
                    # TODO retrieve redundant ShardedObjects once removed in _remove_redundant_data
                    if isinstance(sh_base, ShardedTensor):
                        shard_id = _sharded_tensor_shard_id(sh_base)
                        if shard_id not in loaded_tensors:
                            unloaded_shards[shard_id] = sh_base

                with debug_time("exchange_by_distribution", logger):
                    loaded_tensors = exchange_by_distribution(
                        loaded_tensors,
                        unloaded_shards,
                        distribution,
                        parallelization_group,
                        exchange_algo,
                    )
                    torch.cuda.synchronize()
        loaded_objects = {}
        for sh_base in nested_values(self.sharded_state_dict):
            if not isinstance(sh_base, ShardedTensor):
                assert isinstance(sh_base, ShardedObject)
                loaded_objects[_sharded_object_id(sh_base)] = sh_base.data

        def load_sharded_base(x: Any):
            if isinstance(x, ShardedTensor):
                shard_id = _sharded_tensor_shard_id(x)
                assert shard_id in loaded_tensors, (x, shard_id, loaded_tensors.keys())
                x = loaded_tensors[shard_id]
            if isinstance(x, ShardedObject):
                object_id = _sharded_object_id(x)
                assert object_id in loaded_objects, (x, object_id, loaded_objects.keys())
                x = loaded_objects[object_id]
            return x

        dict_list_map_inplace(load_sharded_base, sharded_part)

    @debug_time("to_state_dict", logger)
    def to_state_dict(
        self,
        sharded_state_dict: ShardedStateDict,
        algo: str = 'atomic',
        exchange_algo: str = 'broadcast',
        validate_access_integrity: bool = True,
        parallelization_group: Optional[torch.distributed.ProcessGroup] = None,
        strict: StrictHandling = StrictHandling.ASSUME_OK_UNEXPECTED,
        return_mismatch_keys: bool = False,
    ):
        """
        Convert tensor-aware dict back to the original state_dict
        """
        with debug_time("load_preprocess_and_state_dict_manipulations", logger):
            assert not self.is_hollow  # TODO raise exception
            self._validate_params(algo)
            fully_parallel = algo == 'fully_parallel'

            # __adding__ common part
            recreated_state_dict = dict_list_map_outplace(lambda x: x, self.common)

            if not sharded_state_dict:
                return recreated_state_dict
            # TODO validate self.sharded_state_dict"] and sharded_state_dict are compatible

            sharded_state_dict, nonpersistent_state_dict, sh_ten_factories = load_preprocess(
                sharded_state_dict
            )
            # __adding__ nonpersistent part
            merge(recreated_state_dict, nonpersistent_state_dict)

            sharded_part, _ = extract_sharded_base(sharded_state_dict)

        # Strictness
        ckpt_sharded_metadata = None
        local_metadata, global_metadata = None, None
        strict = parse_strict_flag(strict)

        if StrictHandling.requires_explicit_ckpt_mismatch_check(strict):
            ckpt_sharded_metadata = {
                sh_base.key: sh_base.without_data()
                for sh_base in nested_values(self.sharded_state_dict)
            }

        if validate_access_integrity or StrictHandling.requires_global_app_metadata(strict):
            local_metadata, global_metadata = determine_global_metadata(sharded_part)

        sharded_state_dict, missing_keys, unexpected_keys = validate_integrity_and_strict_load(
            sharded_part,
            strict,
            validate_access_integrity,
            local_metadata,
            global_metadata,
            ckpt_sharded_metadata,
        )

        # load sharded tensors and sharded objects to sharded_part
        with debug_time("_insert_sharded_data", logger):
            self._insert_sharded_data(
                fully_parallel, sharded_part, parallelization_group, exchange_algo
            )
        with debug_time("apply_factory_merges", logger):
            sharded_part = apply_factory_merges(sharded_part, sh_ten_factories)
            # __adding__ sharded_part
            merge(recreated_state_dict, sharded_part)

        if return_mismatch_keys:
            return recreated_state_dict, missing_keys, unexpected_keys
        else:
            return recreated_state_dict
