# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size

from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, MegatronOptimizer
from .optimizer_config import OptimizerConfig


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
    6. allgather updated params to every rank(currently through broadcast loop)
    """

    def __init__(
        self,
        optimizers: List[MegatronOptimizer],
        config: OptimizerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ) -> None:
        self.pg_collection = pg_collection
        self.shard_params(optimizers)
        # wrap optimizer after sharding to avoid unnecessary master weight creation
        # TODO(deyuf): check if underlying optimizer.config need to fixed
        if config.bf16:
            if isinstance(optimizers[0], Float16OptimizerWithFloat16Params):
                raise TypeError('LayerWiseDistributedOptimizer received Float16 optimizer already.')
            optimizers = [
                Float16OptimizerWithFloat16Params(optim, config, None, None) for optim in optimizers
            ]
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

        if self.dp_cp_params_list:
            _allgather_helper(self.dp_cp_params_list, self.pg_collection.dp_cp)
        if self.expt_dp_params_list:
            _allgather_helper(self.expt_dp_params_list, self.pg_collection.expt_dp)

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

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer.

        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer."""
        super().load_state_dict(torch.load(filename))
