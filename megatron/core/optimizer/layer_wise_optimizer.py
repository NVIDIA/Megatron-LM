# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from typing import Callable, List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size

from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from .optimizer_config import OptimizerConfig

logger = logging.getLogger(__name__)


class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Experimental distributed optimizer wrapper that distributes weight to DP ranks by layer.
    Implemented as ChainedOptimizer to support multiple optimizers (e.g. muon + adamW)
    When using, keep all megatron distributed-optimizer related options OFF.

    How LayerWiseDistributedOptimizer work:
    1. weights are splited into lists and each rank only keep its shard in its optimizer
    2. Megatron DDP handle allreduce grad, note that each rank have full model and grad
    3. optimizer is already modified so only param belong to this DP rank is updated
    4. grad_norm and zero counting will reduce metrics globally in step function
    5. Do regular update with chained optimizers, modified optimizer only update shard
    6. allgather updated params to every rank
    """

    def __init__(
        self,
        optimizers: List[MegatronOptimizer],
        config: OptimizerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        init_state_fn_list: Optional[List[Callable]] = None,
    ) -> None:
        """
        Initialize LayerWiseDistributedOptimizer.

        Args:
            optimizers: List of MegatronOptimizers.
            config: OptimizerConfig.
            pg_collection: ProcessGroupCollection.
            init_state_fn_list: List of init state functions.
        """

        self.pg_collection = pg_collection
        self.shard_params(optimizers)
        if init_state_fn_list:
            assert len(init_state_fn_list) == len(
                optimizers
            ), "init_state_fn_list must be the same length as optimizers if provided"

        # wrap optimizer after sharding to avoid unnecessary master weight creation
        # for higher precision, optimizers are wrapped with megatron already
        if config.bf16:
            # unwrap FP32 optimizer, possibly from reusing get_megatron_optimizer for adam
            for i in range(len(optimizers)):
                opt = optimizers[i]
                if isinstance(opt, Float16OptimizerWithFloat16Params):
                    raise TypeError(
                        'LayerWiseDistributedOptimizer received Float16 optimizer already.'
                    )
                # unwrap FP32 optimizer from reusing get_megatron_optimizer for adam
                if isinstance(opt, FP32Optimizer):
                    opt = opt.optimizer
                optimizers[i] = Float16OptimizerWithFloat16Params(
                    opt, config, None, init_state_fn_list[i] if init_state_fn_list else None
                )

        super().__init__(optimizers)

        # TODO(kunlun, deyuf): potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in
        # contiguous gbuf. So instead of shard param by layer randomly, we can shard by
        # buf range but keep some "extras" to keep boundary weight not sharded.
        # This way each rank do some duplicated work but allgather_v is no longer needed
        # All current distopt optimization can also be potentially applied

    def shard_params(self, optimizers):
        """Shard all params into lists by rank."""
        # list of parameter are sorted by numel and assigned to ranks in ping-pong style
        # example of 4 ranks and 10 parameters p0-p9 after sorting, then dp_cp_params_list will be
        # [[p0, p7, p8], [p1, p6, p9], [p2, p5], [p3, p4]]

        # simplify when dp_cp group size is 1
        if get_pg_size(self.pg_collection.dp_cp) == 1:
            self.dp_cp_params_list = None
            self.expt_dp_params_list = None
            return

        dp_cp_idx, expt_dp_idx = 0, 0
        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)
        # create ping-pong style loop so memory is more balanced
        dp_cp_loop = list(range(dp_cp_size)) + list(range(dp_cp_size))[::-1]
        expt_dp_loop = list(range(expt_dp_size)) + list(range(expt_dp_size))[::-1]
        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]
        # get all param groups
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups

        # sort param in all groups by param numel and assign to each rank evenly
        param_list = []
        for group_index, group in enumerate(param_groups):
            for p in group["params"]:
                param_list.append((p, group_index))
        param_list.sort(key=lambda x: x[0].numel())
        param_groups_this_rank = [[] for g in param_groups]

        # assign params to rank in ping-pong style loop
        for p, group_index in param_list:
            if param_groups[group_index].get("is_expert_parallel", False):
                if expt_dp_loop[expt_dp_idx] == get_pg_rank(self.pg_collection.expt_dp):
                    param_groups_this_rank[group_index].append(p)
                self.expt_dp_params_list[expt_dp_loop[expt_dp_idx]].append(p)
                expt_dp_idx = (expt_dp_idx + 1) % len(expt_dp_loop)
            else:
                if dp_cp_loop[dp_cp_idx] == get_pg_rank(self.pg_collection.dp_cp):
                    param_groups_this_rank[group_index].append(p)
                self.dp_cp_params_list[dp_cp_loop[dp_cp_idx]].append(p)
                dp_cp_idx = (dp_cp_idx + 1) % len(dp_cp_loop)

        # now we modify the group to only handle local params
        for groups, params in zip(param_groups, param_groups_this_rank):
            groups["params"] = params

        # simplify when expt_dp group size is 1 or expert parallel is off
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    @torch.no_grad()
    def allgather_params(self) -> None:
        """All-gather updated params from all ranks."""

        # helper function to flatten local params, allgather, unflatten and copy to model params
        def _allgather_helper(params_list, group):
            # flatten this rank's params and create empty tensor output list
            device = params_list[0][0].device
            dtype = params_list[0][0].dtype
            rank = get_pg_rank(group)
            # for rank without params create empty tensor and participate in allgather
            src = (
                _flatten_dense_tensors(params_list[rank])
                if len(params_list[rank]) > 0
                else torch.empty(0, device=device, dtype=dtype)
            )
            output_list = [
                torch.empty(sum([p.numel() for p in params]), device=device, dtype=dtype)
                for params in params_list
            ]
            # single all_gather_v to collect all updated params
            torch.distributed.all_gather(output_list, src, group=group)
            # unflatten and copy gathered params for each rank i
            for idx, (flat_params, params) in enumerate(zip(output_list, params_list)):
                # skip local params and empty tensors
                if len(params) == 0 or idx == rank:
                    continue
                updated_params = _unflatten_dense_tensors(flat_params, params)
                for updated_p, model_p in zip(updated_params, params):
                    model_p.data.copy_(updated_p)

        if self.pg_collection is None:
            return
        if self.dp_cp_params_list:
            _allgather_helper(self.dp_cp_params_list, self.pg_collection.dp_cp)
        if self.expt_dp_params_list:
            _allgather_helper(self.expt_dp_params_list, self.pg_collection.expt_dp)

    @torch.no_grad()
    def broadcast_params(self):
        """All rank broadcast updated local params."""
        # Broadcast linear layer weights to all other ranks. Kept as reference test.
        if self.dp_cp_params_list is None:
            return
        for i, params in enumerate(self.dp_cp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.dp_cp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.dp_cp)
        if self.expt_dp_params_list is None:
            return
        for i, params in enumerate(self.expt_dp_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.pg_collection.expt_dp, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.pg_collection.expt_dp)

    @torch.no_grad()
    def get_grad_norm(self):
        # similar to dist opt, always aggregate globally
        grads_for_norm = []
        for optimizer in self.chained_optimizers:
            grads_for_norm += optimizer.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(grads_for_norm, grad_stats_parallel_group=None)
        return grad_norm

    @torch.no_grad()
    def count_zeros(self):
        params = []
        for optimizer in self.chained_optimizers:
            params += optimizer.get_parameters()
        return count_zeros_fp32(
            params,
            grad_stats_parallel_group=None,
            use_decoupled_grad=self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8,
        )

    @torch.no_grad()
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer."""
        update_successful, grad_norm, num_zeros_in_grad = super().step()

        # All gather updated params.
        self.allgather_params()

        return update_successful, grad_norm, num_zeros_in_grad

    # TODO(deyuf): need to improve dist checkpointing design to properly handle this
    # fp32_from_fp16_params is list, each sub list could be empty if group is empty
    # this breaks dist checkpointing assumption since extract_sharded_base drop list structure
    # for now, we convert it to dict with index as key and convert back in load_state_dict
    def load_state_dict(self, state_dict):
        if len(self.chained_optimizers) == 1:
            wrapped_state_dict = {1: state_dict}
        else:
            wrapped_state_dict = state_dict
        for sd in wrapped_state_dict.values():
            if 'fp32_from_fp16_params' in sd and isinstance(sd['fp32_from_fp16_params'], dict):
                logger.info('[layerwise] converting fp32_from_fp16_params from dict to list')
                sd['fp32_from_fp16_params'] = [
                    v for k, v in sorted(sd['fp32_from_fp16_params'].items())
                ]
        super().load_state_dict(state_dict)

    def sharded_state_dict(
        self, model_sharded_state_dict: ShardedStateDict, is_loading: bool = False, **kwargs
    ):
        """
        Sharded state dict for torch_dist format checkpointing.
        For fixed DP usage only, set replica_id to 0 for all ShardedTensor.
        """
        sharded_state_dict = super().sharded_state_dict(
            model_sharded_state_dict, is_loading, **kwargs
        )

        # for fixed DP usage only
        for sh_base in nested_values(sharded_state_dict):
            if hasattr(sh_base, 'replica_id'):
                assert (
                    isinstance(sh_base.replica_id, int) or len(sh_base.replica_id) == 3
                ), f'Expected replica_id as int or (PP, TP, DP), got: {sh_base}'
                sh_base.replica_id = (
                    0 if isinstance(sh_base.replica_id, int) else (*sh_base.replica_id[:2], 0)
                )

        # later code assume list but chained optimizer fallback to non-list if there's only one
        if len(self.chained_optimizers) == 1:
            wrapped_sharded_state_dict = {1: sharded_state_dict}
        else:
            wrapped_sharded_state_dict = sharded_state_dict

        # Adjust dict rank 0 output correct global metadata into common_dict
        for sd in wrapped_sharded_state_dict.values():
            # wrap empty containers into LocalNonpersistentObject so it won't be saved/loaded
            # params is already wrapped, we only need to handle fp32_from_fp16_params and state
            # more details in load_state_dict comment
            if 'fp32_from_fp16_params' in sd:
                sd['fp32_from_fp16_params'][:] = [
                    group if group else LocalNonpersistentObject(group)
                    for group in sd['fp32_from_fp16_params']
                ]
                sd['fp32_from_fp16_params'] = {
                    i: v for i, v in enumerate(sd['fp32_from_fp16_params'])
                }
            # state is a single dict and will be empty if optimizer is fully empty
            if not sd['optimizer']['state']:
                sd['optimizer']['state'] = LocalNonpersistentObject(sd['optimizer']['state'])
            # group keys(e.g. 'step') might be missing or not updated
            for i, group in enumerate(sd['optimizer']['param_groups']):
                # keep local param tensor so we only gather metadata
                local_params = group.pop('params')
                # save whether this group is empty, so we can use non-empty rank for metadata
                group['params'] = bool(local_params.unwrap())
                all_rank_groups = [None for _ in range(torch.distributed.get_world_size())]
                torch.distributed.all_gather_object(all_rank_groups, group)
                # find first non-empty group if it exists
                nonempty_rank_group = next((g for g in all_rank_groups if g['params']), group)
                nonempty_rank_group['params'] = local_params
                sd['optimizer']['param_groups'][i] = nonempty_rank_group
        return sharded_state_dict

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer. For torch format only.
        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer. For torch format only."""
        super().load_state_dict(torch.load(filename))
