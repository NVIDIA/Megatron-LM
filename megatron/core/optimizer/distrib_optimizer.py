# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Megatron distributed optimizer."""


import itertools
from logging import getLogger
from typing import Callable, Dict, List, Optional, Tuple

import torch
from apex.optimizers import FusedAdam as Adam

from .. import parallel_state, tensor_parallel
from ..dist_checkpointing import ShardedTensor
from ..dist_checkpointing.mapping import LocalNonpersitentObject, ShardedObject, ShardedStateDict
from ..distributed import ParamAndGradBuffer, shard_buffer
from .grad_scaler import MegatronGradScaler
from .optimizer import MixedPrecisionOptimizer, _zero_grad_group_helper
from .optimizer_config import OptimizerConfig

logger = getLogger(__name__)


class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0):
        return Range(start, start + self.size)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

    def __len__(self):
        return self.end - self.start


class DistributedOptimizer(MixedPrecisionOptimizer):
    @classmethod
    def _build_model_gbuf_param_range_map(
        cls,
        param_world_index_map: Dict[torch.nn.Parameter, Tuple],
        gbuf_world_range: Range,
        bucket_offset: int,
    ):
        """
        Build mapping from param reference to grad buffer shard ranges.

        This method builds a mapping from parameter references to grad
        buffer shard ranges, specific to each data-parallel (DP) rank's
        set of 'owned' parameters. Each grad buffer (padded to be an even
        multiple of DP-world-size) is conceptually divided into DP-world-size
        contiguous regions, where each DP rank 'owns' a contiguous regions.
        Ownership in this sense means DP rank is responsible for reducing
        the relevant subset of grads, and updating the relevant subset of
        params.

        This conceptual partitioning of the grad buffer does NOT respect
        parameter boundaries, and as such it is assumed that each created
        range references a shard (or subset) of the full parameter. It is
        easiest to think of each DP rank as operating (i.e., reducing,
        gathering) purely on views into the grad buffer, for all model-to-
        main & main-to-model operations.

        This method creates four ranges:
        - The param's range within the entire grad buffer (i.e., world index).
        - The param's range within the relevant grad bucket's buffer.
        - The param's range within the DP rank's local view of the grad buffer.
        - The param's range within itself (i.e., its shard).
        """

        # Param range map.
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start
                )
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset, param_world_range.end - bucket_offset
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }

        return param_range_map

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer: ParamAndGradBuffer, bucket_index: int):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = torch.distributed.get_rank(param_and_grad_buffer.data_parallel_group)
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel()
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get each param's ranges.
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )

        # Group into dict.
        data = {
            "param_map": param_range_map,
        }

        return data

    @classmethod
    def _build_gbuf_range_map(cls, param_and_grad_buffer: ParamAndGradBuffer):
        """
        Build mapping between params and their grad buffers. These mappings are
        partitioned according to data type.

        Iterate through all buckets of grad buffer to construct param ranges
        that this rank "owns" (the dp_rank'th shard of each bucket, where each
        shard is 1/dp_world_size of the bucket).

        Arguments:
            param_and_grad_buffer (ParamAndGradBuffer): buffer to build mapping for.
        """
        return {
            (param_and_grad_buffer.param_dtype, param_and_grad_buffer.grad_dtype): [
                cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index)
                for bucket_index in range(len(param_and_grad_buffer.buckets))
            ]
        }

    @classmethod
    def _build_model_param_gbuf_map(
        cls, gbuf_ranges: List[Dict]
    ) -> Dict[torch.nn.Parameter, Tuple]:
        """
        Create a reverse of the gbuf_ranges, for referencing in opposite direction.
        """
        param_gbuf_map = {}
        for gbuf_index, gbuf_range_map in enumerate(gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param, _ in gbuf_range_map["param_map"].items():
                        assert (
                            param not in param_gbuf_map
                        ), "Param should not be in param_gbuf_map; each param only belongs to a single bucket"
                        param_gbuf_map[param] = (gbuf_index, dtype, bucket_index)
        return param_gbuf_map

    @classmethod
    def _build_optimizer_group_ranges(cls, param_groups: List[Dict], gbuf_ranges: List[Dict]):
        """
        Create optimizer groups.

        Given the set of parameter shard ranges that are owned by the current
        data-parallel (DP) rank, gather the set of parameters that will be
        used (in the method below) to create the current DP's optimizer
        groups.
        """

        # Param group map.
        # World param group map.
        # - Store a mapping of <model_parameter:group_index> for all parameters
        #   across all DP ranks. This is necessary because it is our first
        #   cross reference between the DDP mappings and the optimizer group
        #   parameters. This mapping only for use in the next step of building
        #   the local mapping over this DP rank's parameters.
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        # Optimizer group ranges & param-group mapping.
        # - Build a mapping from groups to their contained parameters, and also
        #   from parameters to their containing group index and order within
        #   the group. The group index and order are particularly important for
        #   saving and loading checkpoints.
        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        for gbuf_range_map in gbuf_ranges:
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_map.items():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (group_index, len(group_range["params"]) - 1)

        # Squeeze zero-size group ranges.
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges

    @classmethod
    def _build_model_and_main_param_groups(
        cls,
        gbuf_ranges: List[Dict],
        param_gbuf_map: Dict[torch.nn.Parameter, Tuple],
        opt_group_ranges: List,
    ):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_fp32_from_float16_groups: fp32 copy of float16 parameters
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_fp32_from_float16_groups = []

        # Allocate (or slice) each group's param shard.
        for group_range in opt_group_ranges:

            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_fp32_from_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_fp32_from_float16_groups.append(shard_fp32_from_float16_params_this_group)

            for model_param in group_range["params"]:

                assert model_param.requires_grad

                gbuf_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = gbuf_ranges[gbuf_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:

                    # Clone model -> main.
                    shard_model_param = model_param.detach().view(-1)[
                        param_range.start : param_range.end
                    ]
                    shard_main_param = shard_model_param.clone().float()
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_main_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_fp32_from_float16_params_this_group.append(shard_main_param)

                # fp32 params.
                elif model_param.type() == 'torch.cuda.FloatTensor':
                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    tensor_parallel.copy_tensor_model_parallel_attributes(
                        shard_model_param, model_param
                    )
                    if hasattr(model_param, 'shared'):
                        shard_model_param.shared = model_param.shared

                else:
                    raise TypeError(
                        'Wrapped parameters must be one of '
                        'torch.cuda.FloatTensor,  '
                        'torch.cuda.HalfTensor, or '
                        'torch.cuda.BFloat16Tensor. '
                        'Received {}'.format(model_param.type())
                    )

            # Update optimizer's params.
            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_fp32_from_float16_params_this_group,
            ]

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_fp32_from_float16_groups,
        )

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        per_model_buffers: Dict[int, List[ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: torch.distributed.ProcessGroup,
        data_parallel_group_idx: int,
    ):
        """
        Distributed optimizer, for all data types (fp16, bf16, and fp32).

        The steps in this method create the core mapping between param and grad buffers,
        parameters, and parameter shard ranges, that is needed for converting between model
        param indexes and main parameter shard indexes. This method also updates the optimizer
        parameter groups with the newly created shards.

        Arguments:
            optimizer (torch.optim.Optimizer): base optimizer such as Adam or SGD.
            config (OptimizerConfig): configuration object for optimizer.
            grad_scaler (MegatronGradScaler): used for scaling gradients. Note that
                this can be None. This case happens when `bf16 = True` and we don't
                use any loss scale. Note that for `bf16 = True`, we can have
                a constant gradient scaler. Also for `bf16 = False`, we
                always require a grad scaler.
            init_state_fn (Callable, optional): function to initialize state in the optimizer.
            per_model_buffers (Dict[int, List[ParamAndGradBuffer]]): the implementation of the
                distributed optimizer is centered on using a contiguous buffer for
                communicating grads & params between the model state and the optimizer state.
                You can find a more detailed description in
                https://github.com/NVIDIA/Megatron-LM/blob/main/docs/source/distrib_optimizer.md.
            data_parallel_group (torch.distributed.ProcessGroup): data-parallel group to use to
                all-gather params after optimizer.step().
            data_parallel_group_gloo (torch.distributed.ProcessGroup): gloo data-parallel group
                (used in checkpoint loading and saving).
            data_parallel_group_idx (int): index in data-parallel group (used by
                distributed checkpointing logic).
        """

        super().__init__(
            optimizer, config, grad_scaler, init_state_fn,
        )

        assert isinstance(
            optimizer, Adam
        ), "Only Adam currently supported, due to checkpointing requirements."

        # Model grad buffer ranges.
        assert per_model_buffers is not None, "per_model_buffers must be provided"
        self.buffers = list(itertools.chain(*per_model_buffers.values()))
        self.per_model_buffers = per_model_buffers
        self.data_parallel_group = data_parallel_group
        self.data_parallel_group_gloo = data_parallel_group_gloo
        self.data_parallel_group_idx = data_parallel_group_idx
        self.gbuf_idx_to_model_idx_map = {}
        gbuf_idx = 0
        for model_idx, buffers in self.per_model_buffers.items():
            for _ in buffers:
                self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
                gbuf_idx += 1
        self.gbuf_ranges = []
        self.per_bucket_numel = []
        self.per_bucket_numel_unpadded = []
        for buffer in self.buffers:

            self.per_bucket_numel.append(
                {
                    (buffer.param_dtype, buffer.grad_dtype): [
                        bucket.grad_data.numel() for bucket in buffer.buckets
                    ]
                }
            )
            self.per_bucket_numel_unpadded.append(
                {
                    (buffer.param_dtype, buffer.grad_dtype): [
                        bucket.numel_unpadded for bucket in buffer.buckets
                    ]
                }
            )
            self.gbuf_ranges.append(self._build_gbuf_range_map(buffer))
        self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

        # Optimizer ranges.
        (
            self.model_param_group_index_map,
            self.opt_group_ranges,
        ) = self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)

        # Allocate main param shards.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_fp32_from_float16_groups,
        ) = self._build_model_and_main_param_groups(
            self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
        )

        # Now construct data structures to manage all-gather handles.
        self.all_gather_handles = []
        self.all_gather_handle_index_to_bucket_index_map = []
        self.model_index_to_all_gather_handle_index_map = {}
        self.all_gather_handle_indices = []
        self.param_to_all_gather_handle_index_map = {}

        self.pbuf_view_items = self._get_model_param_buffer_dp_views()
        for (gbuf_index, dtype, bucket_index, _, _) in self.pbuf_view_items:
            self.all_gather_handle_index_to_bucket_index_map.append(
                (gbuf_index, dtype, bucket_index)
            )
            all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1
            self.all_gather_handles.append(None)

            # Store all all_gather_handle_indices.
            model_idx = self.gbuf_idx_to_model_idx_map[gbuf_index]
            if model_idx not in self.model_index_to_all_gather_handle_index_map:
                self.model_index_to_all_gather_handle_index_map[model_idx] = []
            self.model_index_to_all_gather_handle_index_map[model_idx].append(
                all_gather_handle_index
            )

            for param in self.buffers[gbuf_index].buckets[bucket_index].params_list:
                self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
        self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

        self.overlap_param_gather = self.config.overlap_param_gather
        self.remove_pre_hook_handle = None
        if self.overlap_param_gather:
            self.enable_pre_hook()

        self.update_successful = False

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

    def enable_pre_hook(self):
        """
        Enable forward pre-hook needed for param all-gather overlap with forward compute.
        """
        assert self.remove_pre_hook_handle is None
        self.remove_pre_hook_handle = torch.nn.modules.module.register_module_forward_pre_hook(
            self._make_forward_pre_hook()
        )

    def disable_pre_hook(self):
        """
        Disable forward pre-hook needed for param all-gather overlap with forward compute.
        """
        assert self.remove_pre_hook_handle is not None
        self.remove_pre_hook_handle.remove()
        self.remove_pre_hook_handle = None

        # Make sure all-gathers are completed as needed.
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=True)

    def _get_model_param_range_map(self, param: torch.nn.Parameter):
        """
        Given a model param, get the index sub-range of the param that this
        data-parallel rank owns.
        """
        gbuf_index, dtype, bucket_index = self.model_param_gbuf_map[param]
        gbuf_range_map = self.gbuf_ranges[gbuf_index][dtype][bucket_index]
        param_range_map = gbuf_range_map["param_map"][param]
        return param_range_map

    def get_model_parallel_group(self) -> torch.distributed.ProcessGroup:
        """
        With the distributed optimizer, the model parallel group is the
        entire world.
        """
        return None

    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """

        state_dict = {}

        # Optimizer state (do not store parameter state here).
        state_dict['optimizer'] = {
            k: v for k, v in self.optimizer.state_dict().items() if k != "state"
        }
        for param_group in state_dict["optimizer"]["param_groups"]:
            del param_group["params"]

        # Grad scaler state.
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        """Load the state dict.

        As detailed in state_dict(), the state dict contains all non-
        parameter-related variables. This method is notably longer than
        state_dict(), because the Torch optimizers state has yet to be
        allocated at this point, and so we must do a cross referencing between
        the optimizers state (and the ordering it expects for parameter state)
        and this DP rank's shards. The optimizer at this point does not contain
        any tensor dimension information, so we must get these dimensions from
        the DP shards mapped during DistributedOptimizer.__init__().

        The tensor parameter state is loaded via load_parameter_state(), and
        so this method also must populate the loaded state dict with dummy
        tensor data (i.e., via torch.empty() below). This will be overwritten
        during load_parameter_state().

        ** Note: Torch optimizer's state structure. **
        The Torch optimizer stores its state in two levels. The top level is a
        list of groups, where each group contains a list of integer indexes
        (corresponding to parameters) that index into a master parameter list
        that is shared by all groups. As such, three values are necessary for
        maintaining this ordering:

        - group_index : The group to which a parameter belongs.
        - group_order : The index of a parameter within its group.
        - state_order : The index of a parameter within the shared parameter
            list.
        """

        # Get the Torch optimizer's state dict.
        # - This 'inner' optimizer at this point is unallocated, and only
        #   contains an integer odering of parameters within each group, and
        #   the ordering of parameters within its flattened parameter state
        #   list.
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [
            {**group, "params": list(inner_state_dict["param_groups"][idx]["params"]),}
            for idx, group in enumerate(state_dict["optimizer"]["param_groups"])
        ]

        # Allocate 'dummy' data for optimizer state (i.e., torch.empty() below)
        # - Real data is overwritten during load_parameter_state().
        state_dict_state = []
        for gbuf_range_maps in self.gbuf_ranges:
            for gbuf_range_map_for_all_buckets in gbuf_range_maps.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Get parameter ordering information (see method docstring
                        # for details).
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        state_order = inner_state_dict["param_groups"][group_index]["params"][
                            group_order
                        ]

                        # Allocate dummy tensors.
                        numel = len(param_range_map["gbuf_world"])
                        init_shard = lambda: torch.empty(
                            (numel,), dtype=torch.float32, device=torch.cuda.current_device()
                        )

                        state_dict_state.append(
                            (state_order, {"exp_avg": init_shard(), "exp_avg_sq": init_shard(),})
                        )

        # Sort by state order (see method docstring for details).
        state_dict_state.sort(key=lambda s: s[0])
        state_dict_state = {s[0]: s[1] for s in state_dict_state}

        # Optimizer.
        self.optimizer.load_state_dict(
            {"state": state_dict_state, "param_groups": state_dict_param_groups,}
        )

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info(
                    '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

        if 'param_state' in state_dict:
            assert 'param_state_sharding_type' in state_dict, state_dict.keys()
            param_state = state_dict['param_state']
            sharding_type = state_dict['param_state_sharding_type']
            logger.info(f'Loading distributed optimizer sharded state of type {sharding_type}')
            if sharding_type == 'dp_zero_gather_scatter':
                self.load_parameter_state_from_dp_zero(param_state)
            elif sharding_type == 'fully_sharded_bucket_space':
                self.load_parameter_state_from_fs_bucket_space(param_state)
            else:
                raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

    def get_parameter_state_fs_bucket_space(self):
        """Get internal representation of parameter state without any copies and modifications.

        This is referred to as "fully sharded bucket space" because the optimizer state is
        fully sharded (e.g. no gather involved) and bucket-centric (the state
        follows the internal structure of the Distributed Optimizer buckets)
        as opposed to model-centric (typical structure of PyT optimizers)
        """
        state = {
            "per_bucket_numel": self.per_bucket_numel,
            "per_bucket_numel_unpadded": self.per_bucket_numel_unpadded,
        }
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buckets_state = []
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = []
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param": main_param,
                            **optim_state,
                            "gbuf_local_start": param_range_map["gbuf_local"].start,
                            "gbuf_local_end": param_range_map["gbuf_local"].end,
                        }
                        bucket_state.append(tensors)
                    buckets_state.append(bucket_state)
                dtype_state[dtype] = buckets_state
            state[gbuf_idx] = dtype_state
        return state

    def get_parameter_state_dp_zero(self):
        """Get parameter state (i.e., parameter & optimizer tensors).

        This method performs two steps:
        - For each DP rank, copy param & optimizer shards to contiguous CPU
          buffers (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        - Gather contiguous buffers on DP rank 0 and concatenate to world
          buffers.
        """

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Collect param states.
        state = {
            "per_bucket_numel": self.per_bucket_numel,
            "per_bucket_numel_unpadded": self.per_bucket_numel_unpadded,
        }
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

            # Iterate grad buffers (by data type).
            dtype_state = {}
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                world_tensors = {}
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                    local_shards = {
                        key: torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    # Build contiguous DP rank shards (for param + optim states).
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param": main_param,
                            **optim_state,
                        }

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                                tensors[key].detach().cpu()
                            )

                    # Gather contiguous shards on DP rank 0.
                    for key, send_tensor in local_shards.items():

                        # Gather tensor list.
                        if data_parallel_rank == 0:
                            recv_tensors = [
                                torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                                for _ in range(data_parallel_world_size)
                            ]
                        else:
                            recv_tensors = None

                        # Gather.
                        torch.distributed.gather(
                            send_tensor,
                            recv_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                        # Concatenate.
                        if data_parallel_rank == 0:
                            if key not in world_tensors:
                                world_tensors[key] = []
                            world_tensors[key].append(torch.cat(recv_tensors))

                # Collect world state.
                dtype_state[dtype] = world_tensors
            state[gbuf_idx] = dtype_state

        return state

    def save_parameter_state(self, filename: str):
        """Save the distributed parameter state on DP rank 0.

        Arguments:
            filename (str): path to save parameter state to.
        """

        state_dict = self.get_parameter_state_dp_zero()
        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            torch.save(state_dict, filename)

    def sharded_state_dict(
        self,
        model_sharded_state_dict: ShardedStateDict,
        is_loading: bool = False,
        sharding_type: str = 'fully_sharded_bucket_space',
    ):
        """
        Chooses between 3 param state sharding implementations as requested by `sharding_type`.

        Regular state dict parameters are saved on DP rank 0 and loaded on all ranks.
        """

        state_dict = {
            k: ShardedObject(
                f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.{k}',
                v,
                (1,),
                (0,),
                replica_id=torch.distributed.get_rank(self.data_parallel_group),
            )
            for k, v in self.state_dict().items()
        }

        if is_loading:
            self.init_state_fn(self.optimizer)

        if sharding_type == 'fully_sharded_bucket_space':
            param_state = self.sharded_param_state_fs_bucket_space(
                model_sharded_state_dict, is_loading
            )
        elif sharding_type == 'dp_zero_gather_scatter':
            param_state = self.sharded_param_state_dp_zero(model_sharded_state_dict, is_loading)
        elif sharding_type == 'fully_sharded_model_space':
            # In this approach the tensors could be directly related to model parameters
            # by linking them with metadata from `model_sharded_state_dict`.
            # This would allow changing TP and PP while using DistOpt (as with other optimizers).
            # This implementation is more involved and left out for now.
            raise NotImplementedError(
                f'The fully sharded model space version for'
                f' {self.__class__.__name__}.sharded_state_dict'
                f' not implemented.'
            )
        else:
            raise NotImplementedError(f'Unknown sharding_type: {sharding_type}')

        state_dict['param_state'] = param_state
        state_dict['param_state_sharding_type'] = sharding_type
        return state_dict

    def sharded_param_state_dp_zero(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        """Naive implementation which reuses gather/scatter from the legacy ckpt format.

        During saving, gathers the parameters state on DP rank 0 and saves a ShardedObject
        with fixed TPxPP structure. During loading, loads the saved data on DP rank 0
        (None on other ranks). Relies on the parameters scatter done in load_state_dict.
        """
        if is_loading:
            param_state_data = None
        else:
            # Gather on rank 0
            param_state_data = self.get_parameter_state_dp_zero()

        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            # Fixed TPxPP. Save on DP rank 0 only
            param_state = ShardedObject(
                f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.param_state',
                param_state_data,
                (1,),
                (0,),
            )
        else:
            # DP ranks > 0 don't save. During loading, the param_state needs to be None.
            param_state = LocalNonpersitentObject(None)

        return param_state

    def sharded_param_state_fs_bucket_space(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False
    ):
        """Sharded state dict where each noncontiguous buffer is a separate ShardedTensor.

        Results in fully parallel save and load without any inter-process
        communication or intermediate buffers/copies.
        """
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group)
        data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)

        state = self.get_parameter_state_fs_bucket_space()
        # per_bucket_numel metadata is saved separately for each TPxPP domain.
        for per_bucket_key in ('per_bucket_numel', 'per_bucket_numel_unpadded'):
            state[per_bucket_key] = ShardedObject(
                f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.{per_bucket_key}',
                state[per_bucket_key],
                (1,),
                (0,),
                replica_id=data_parallel_rank,
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in state[gbuf_idx].items():
                for bucket_idx, bucket_state in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    sharded_bucket_key = f'optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}.gbuf_idx_{gbuf_idx}.dtype_{dtype}.bucket_idx_{bucket_idx}'

                    # The global ckpt tensors must be fully covered.
                    # We add extra empty padding if necessary
                    assert bucket_state, 'empty bucket encountered'
                    if bucket_state[-1]['gbuf_local_end'] != gbuf_local_numel:
                        assert (
                            data_parallel_rank == data_parallel_world_size - 1
                        ), 'encountered padding on non-last DP rank'
                        pad_tensors = {
                            k: torch.empty(
                                gbuf_local_numel - bucket_state[-1]['gbuf_local_end'],
                                dtype=v.dtype,
                                device=v.device,
                            )
                            for k, v in bucket_state[-1].items()
                            if isinstance(v, torch.Tensor)
                        }
                        bucket_state.append(
                            {
                                **pad_tensors,
                                'gbuf_local_start': bucket_state[-1]['gbuf_local_end'],
                                'gbuf_local_end': gbuf_local_numel,
                            }
                        )

                    # Each tensor is mapped to a slice (`flattened_range`)
                    # of a DP-local shard of size `gbuf_local_numel`.
                    for bucket_params_idx in range(len(bucket_state)):
                        tensors = bucket_state[bucket_params_idx]
                        gbuf_local_start = tensors.pop('gbuf_local_start')
                        gbuf_local_end = tensors.pop('gbuf_local_end')

                        for key in tensors:
                            assert tensors[key].shape == (gbuf_local_end - gbuf_local_start,), (
                                tensors[key].shape,
                                gbuf_local_start,
                                gbuf_local_end,
                            )

                            tensors[key] = ShardedTensor(
                                f'{sharded_bucket_key}.{key}',
                                tensors[key],
                                tensors[key].dtype,
                                (gbuf_local_numel,),
                                (data_parallel_world_size * gbuf_local_numel,),
                                (data_parallel_rank * gbuf_local_numel,),
                                axis_fragmentations=(data_parallel_world_size,),
                                flattened_range=slice(gbuf_local_start, gbuf_local_end),
                                allow_shape_mismatch=True,
                            )
        return state

    def load_parameter_state_from_fs_bucket_space(self, state_dict):
        """ Loads the parameter state from an internal representation.

        Inverse of the `get_parameter_state_internal_repr` method.
        """
        if state_dict is not None and "per_bucket_numel_unpadded" in state_dict:
            per_bucket_numel_unpadded_in_checkpoint = state_dict["per_bucket_numel_unpadded"]
            assert self.per_bucket_numel_unpadded == per_bucket_numel_unpadded_in_checkpoint, (
                f"Number of unpadded elements in each bucket need to be the same in current run "
                f"({self.per_bucket_numel_unpadded}) and checkpoint "
                f"({per_bucket_numel_unpadded_in_checkpoint})"
            )

        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            assert len(gbuf_range_maps) == 1, "single dtype supported, for now."
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    bucket_state = state_dict[gbuf_idx][dtype][bucket_idx]

                    # State dict bucket state can be 1 entry longer in case of padding
                    assert len(bucket_state) in (
                        len(gbuf_range_map["param_map"]),
                        len(gbuf_range_map["param_map"]) + 1,
                    ), (len(bucket_state), len(gbuf_range_map["param_map"]))
                    for src_tensors, (model_param, param_range_map) in zip(
                        bucket_state, gbuf_range_map["param_map"].items()
                    ):
                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        dst_tensors = {
                            "param": main_param,
                            **optim_state,
                        }
                        for key in dst_tensors:
                            dst_tensors[key].copy_(src_tensors[key])

    def load_parameter_state_from_dp_zero(self, state_dict):
        """Load parameter state (i.e., parameter & optimizer tensors) from DP 0 rank.

        This method performs the reverse of get_parameter_state_dp_zero():
        - Scatter contiguous buffers from DP rank 0 to each DP rank (each DP
          rank receives its relevant subset of the world buffers).
        - For each DP rank, copy param & optimizer shards from contiguous CPU
          buffers. (e.g., one buffer each for main_param, exp_avg, and
          exp_avg_sq).
        """
        if state_dict is not None and "per_bucket_numel_unpadded" in state_dict:
            per_bucket_numel_unpadded_in_checkpoint = state_dict["per_bucket_numel_unpadded"]
            assert self.per_bucket_numel_unpadded == per_bucket_numel_unpadded_in_checkpoint, (
                f"Number of unpadded elements in each bucket need to be the same in current run "
                f"({self.per_bucket_numel_unpadded}) and checkpoint "
                f"({per_bucket_numel_unpadded_in_checkpoint})"
            )

        # Data parallelism variables.
        data_parallel_world_size = self.data_parallel_group_gloo.size()
        data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
        data_parallel_group_gloo = self.data_parallel_group_gloo
        data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
            self.data_parallel_group_gloo
        )

        # Scatter tensors to all DP ranks.
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    assert gbuf_world_numel == self.per_bucket_numel[gbuf_idx][dtype][bucket_idx]
                    assert gbuf_world_numel % data_parallel_world_size == 0
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    # Contiguous local shards (received from DP rank 0).
                    local_shards = {
                        key: torch.empty((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    # Scatter local shards from DP rank 0.
                    for key, recv_tensor in local_shards.items():

                        # Scatter tensor list.
                        if data_parallel_rank == 0:
                            world_tensor_for_all_buckets = state_dict[gbuf_idx][dtype][key]
                            if not isinstance(world_tensor_for_all_buckets, list):
                                world_tensor_for_all_buckets = [world_tensor_for_all_buckets]
                            assert bucket_idx < len(world_tensor_for_all_buckets), (
                                f"Trying to load state for bucket_id {bucket_idx} (out of "
                                f"{len(gbuf_range_map_for_all_buckets)} buckets) from checkpoint; "
                                f"checkpoint only has {len(world_tensor_for_all_buckets)} bucket(s)"
                            )
                            # This tensor might be bigger or smaller than expected (depending on
                            # relative sizes of per_bucket_numel_in_checkpoint and self.per_bucket_numel).
                            world_tensor = world_tensor_for_all_buckets[bucket_idx]
                            if "per_bucket_numel" in state_dict:
                                numel_in_checkpoint = state_dict["per_bucket_numel"][gbuf_idx][
                                    dtype
                                ][bucket_idx]
                                numel = self.per_bucket_numel[gbuf_idx][dtype][bucket_idx]
                                numel_unpadded = self.per_bucket_numel_unpadded[gbuf_idx][dtype][
                                    bucket_idx
                                ]
                                assert world_tensor.numel() == numel_in_checkpoint
                                assert numel_unpadded <= world_tensor.numel(), (
                                    "True number of elements should be fewer than number of elements in "
                                    "checkpoint tensor"
                                )
                                if world_tensor.numel() > numel:
                                    # Truncate extra values, which are padding anyway.
                                    logger.info(
                                        f"Truncating extra values from checkpoint (numel_in_checkpoint={numel_in_checkpoint}, "
                                        f"numel={numel}, numel_unpadded={numel_unpadded})"
                                    )
                                    world_tensor = world_tensor[:numel]
                                elif world_tensor.numel() < numel:
                                    # In this case, numel > world_tensor.numel() (which is numel_in_checkpoint).
                                    # Create new tensor with right number of values, then copy and use new tensor.
                                    logger.info(
                                        f"Expanding tensor from checkpoint (numel_in_checkpoint={numel_in_checkpoint}, "
                                        f"numel={numel}, numel_unpadded={numel_unpadded})"
                                    )
                                    world_tensor_reshaped = torch.empty(
                                        (numel,),
                                        dtype=world_tensor.dtype,
                                        device=world_tensor.device,
                                    )
                                    world_tensor_reshaped[:numel_in_checkpoint].copy_(world_tensor)
                                    world_tensor = world_tensor_reshaped
                            else:
                                logger.info(
                                    "***WARNING*** Using older checkpoint so skipping padding checks"
                                )
                            gbuf_start_idxs = list(range(0, gbuf_world_numel, gbuf_local_numel))
                            send_tensors = [
                                world_tensor[i : (i + gbuf_local_numel)] for i in gbuf_start_idxs
                            ]
                        else:
                            send_tensors = None

                        # Scatter.
                        torch.distributed.scatter(
                            recv_tensor,
                            send_tensors,
                            data_parallel_global_ranks[0],
                            data_parallel_group_gloo,
                        )

                    # Copy local contiguous shards to param/optim shards.
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():

                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]

                        tensors = {
                            "param": main_param,
                            **optim_state,
                        }

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            tensors[key].data.copy_(
                                local_shards[key][gbuf_local_start:gbuf_local_end]
                            )

    def load_parameter_state(self, filename: str):
        """Load the distributed parameter state from disk.

        Arguments:
            filename (str): path to load parameter state from.
        """
        state_dict = None
        if torch.distributed.get_rank(self.data_parallel_group) == 0:
            state_dict = torch.load(filename)

        self.load_parameter_state_from_dp_zero(state_dict)

    def zero_grad(self, set_to_none: bool = True):
        """
        Zeroes grads for the model related parameters, i.e., model_float16_groups
        and model_fp32_groups. We additionally zero the remaining groups as a
        memory optimization to reduce fragmentation; in the case of
        set_to_none==True, the space used by this field can be safely deallocated.

        Arguments:
            set_to_none (bool): if true, set grads to None.
        """
        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,  # grad empty/unused here?
            self.shard_fp32_groups,  # throws grad-access warning
            self.shard_fp32_from_float16_groups,
        ):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)

        # If overlapping param all-gather with forward compute, launch all-gather
        # for first accessed bucket here before forward compute is initiated.
        # The all-gather for the next bucket will be launched in the forward
        # pre-hook when this all-gather finishes (to ensure that the communication
        # kernels don't head-of-line block the compute kernels since we run with
        # CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence parallelism).
        if self.overlap_param_gather:
            self._dispatch_gather_model_params(all_gather_handle_index=0)

    def _get_model_param_buffer_dp_views(self):
        """
        Get shard views of each of the param buffers.

        In this nested list, the top level is grouped by the virtual model
        index and the buffer's data type. The sub-level is a list of
        shards of that buffer, where each shard in the list represents
        a contiguous view of the buffer, that is owned by a data-parallel
        rank. The shard boundary does not respect parameter boundaries, and
        so the elements of some parameters are split across data parallel
        ranks.

        Additionally, return references to the entire buffers, for use
        in _all_gather_base.
        """

        # Buffer views.
        # Add in reverse order in each model chunk since buckets start from the end of the model but we want
        # all-gathers to run first for the start of the model (same order as forward pass).
        # We keep the view_items in model chunk order since we want to still first run all_gather and
        # all_gather_handle.wait() for the first model chunk.
        # In all cases, we want all_gather and all_gather_handle.wait() to be called in the same order,
        # and all_gather_handle.wait() needs to be called just before the corresponding forward pass.
        view_items = []
        for gbuf_index, buffer in enumerate(self.buffers):
            view_items_per_model_chunk = []
            dtype = self.buffers[gbuf_index].param_dtype
            for bucket_index, bucket in enumerate(buffer.buckets):
                data_parallel_world_size = torch.distributed.get_world_size(
                    self.data_parallel_group
                )
                buf_views = shard_buffer(bucket.param_data, data_parallel_world_size)
                view_items_per_model_chunk.insert(
                    0, (gbuf_index, dtype, bucket_index, bucket.param_data, buf_views)
                )
            view_items.extend(view_items_per_model_chunk)

        return view_items

    def _dispatch_gather_model_params(self, all_gather_handle_index: int, force_sync: bool = False):
        """
        All-gather updated model params.

        When using the distributed optimizer, the params are already laid out in a contiguous
        buffer (see mcore/distributed/param_and_grad_buffer.py for details), and so the
        all-gather will put the results in the right region of memory.
        """
        async_op = self.overlap_param_gather and not force_sync
        if self.update_successful:
            data_parallel_group = self.data_parallel_group
            data_parallel_rank = torch.distributed.get_rank(data_parallel_group)

            # All-gather updated main params.
            # All param_buf views are guaranteed to have the same number of elements
            # across all data-parallel ranks, due to padding done in
            # param_and_grad_buffer.py). Thus, all sub-views will have consistent
            # start / end indexes across data-parallel ranks.
            (gbuf_index, dtype, bucket_index, pbuf, pbuf_views) = self.pbuf_view_items[
                all_gather_handle_index
            ]
            assert all_gather_handle_index < len(self.all_gather_handles)
            all_gather_handle = torch.distributed._all_gather_base(
                pbuf, pbuf_views[data_parallel_rank], group=data_parallel_group, async_op=async_op,
            )
            self.all_gather_handles[all_gather_handle_index] = all_gather_handle
            assert self.all_gather_handle_index_to_bucket_index_map[all_gather_handle_index] == (
                gbuf_index,
                dtype,
                bucket_index,
            )

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather)
        and then copy the results from the param_buffer into model_params.
        """

        def hook(module, *unused):
            assert (
                self.overlap_param_gather
            ), "Should use pre-hook only when overlap_param_gather is True"

            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters that don't require grad.
                if not param.requires_grad:
                    continue

                # Some params might be handled in another DistributedOptimizer instance; for
                # example, we use separate DistributedOptimizer instances for expert and
                # non-expert params.
                if param in self.param_to_all_gather_handle_index_map:
                    all_gather_handle_index = self.param_to_all_gather_handle_index_map[param]
                    self._finish_param_sync_helper(all_gather_handle_index)

        return hook

    def finish_param_sync(self, model_index: int, *unused):
        """
        Finishes all necessary param syncs for the model_index'th model chunk.

        Arguments:
            model_index (int): index of model chunk to synchronize params.
        """
        if model_index not in self.model_index_to_all_gather_handle_index_map:
            return

        all_gather_handle_indices = self.model_index_to_all_gather_handle_index_map[model_index]
        for all_gather_handle_index in all_gather_handle_indices:
            self._finish_param_sync_helper(all_gather_handle_index)

    def _finish_param_sync_helper(self, all_gather_handle_index: int):
        """
        Waits on all_gather_handle if necessary, then dispatches the next all-gather
        as necessary.
        """

        # First check if there is an outstanding all-gather handle for this param.
        # If so, wait on the handle to ensure the communication is finished.
        assert all_gather_handle_index < len(self.all_gather_handles)
        all_gather_handle = self.all_gather_handles[all_gather_handle_index]
        if all_gather_handle is not None:
            all_gather_handle.wait()
            self.all_gather_handles[all_gather_handle_index] = None

            # Launch the all-gather for the next bucket now.
            # We can't pre-launch all-gathers for all buckets at once since we don't
            # want to head-of-line block the compute kernels with communication kernels
            # (since we run with CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence
            # parallelism).
            next_all_gather_handle_index = all_gather_handle_index + 1
            if next_all_gather_handle_index < self.num_all_gather_handles:
                self._dispatch_gather_model_params(next_all_gather_handle_index)

    def _collect_main_grad_data_for_unscaling(self):
        """
        Note: this should be equivalent to the float-16 optimizer's method,
        but written differently, so the two should be combined.
        """
        return [
            param.grad.data for group in self.optimizer.param_groups for param in group["params"]
        ]

    def _get_model_and_main_params_data_float16(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(
            self.shard_float16_groups, self.shard_fp32_from_float16_groups
        ):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                    shard_main_param.grad = shard_model_grad.float()

        # Copy model groups to shard groups.
        copy_group_grads(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data

                    shard_model_param = model_param_buffer.view(-1)[
                        world_range.start : world_range.end
                    ]

                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_fp32_from_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)

    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):

                    param_range_map = self._get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_fp32_from_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)

    def _reset_metadata_and_sync_gather_all_model_params(self, force_sync: bool):
        """
        Reset metadata needed to track results of all-gathers.
        """
        self.all_gather_handles = [None for _ in range(len(self.all_gather_handles))]

        # Launch synchronous all-gather if --overlap-param-gather is turned on or if force_sync
        # is explicitly set to True (e.g., if we are going to turn off all-gather overlapping for
        # validation / test iterations).
        if not self.overlap_param_gather or force_sync:
            for all_gather_handle_index in range(self.num_all_gather_handles):
                self._dispatch_gather_model_params(all_gather_handle_index, force_sync=force_sync)

    @torch.no_grad()
    def step(self):
        """
        Step optimizer.
        Under the hood, either launch synchronous param all-gathers or get ready to launch
        asynchorous all-gathers that get overlapped with the next forward pass.
        """
        self.update_successful, grad_norm, num_zeros_in_grad = super().step()

        timers = self.config.timers
        if timers is not None:
            timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)
        # If not overlapping all-gather for parameters, launch synchronous all-gather
        # communication calls here. If overlapping all-gather for parameters, the following
        # call to _gather_all_model_params is a no-op: the first all-gather is launched
        # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
        # are launched in the forward pre-hook.
        self._reset_metadata_and_sync_gather_all_model_params(force_sync=False)
        if timers is not None:
            timers('params-all-gather').stop()

        return self.update_successful, grad_norm, num_zeros_in_grad
