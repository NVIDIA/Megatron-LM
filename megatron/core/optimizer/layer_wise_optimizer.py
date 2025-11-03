# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, List, Optional

import torch

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.dict_utils import nested_values
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_rank, get_pg_size

from .clip_grads import count_zeros_fp32, get_grad_norm_fp32
from .optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params, MegatronOptimizer
from .optimizer_config import OptimizerConfig


class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    This is a experimental distributed optimizer wrapper that distributes weight to DP ranks
    by full layer. Implemented as ChainedOptimizer to support different weights use different
    optimizers (e.g. muon+adam). When using, keep all megatron distributed optimizer related
    options OFF.

    How LayerWiseDistributedOptimizer work:
    1. weights are splited into lists and each rank only keep its shard in its optimizer
    2. Megatron DDP handle allreduce grad for all params, note that each rank have full model
    and grad.
    3. optimizer is already modified so only param belong to this DP rank is updated
    3. grad_norm and zero counting will reduce metrics globally in step function
    4. Do regular update with chained optimizers, optimizer is already modified so partial update
    happens.
    5. allgather updated params to every rank(currently through broadcast loop)
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
        # wrap optimizer after sharding to avoid unnecessary master weight creation
        # TODO(deyuf): check if underlying optimizer.config need to fixed and if so can use
        # that instead of passing
        if init_state_fn_list is None:
            init_state_fn_list = [None] * len(optimizers)
        else:
            assert len(init_state_fn_list) == len(optimizers), (
                "init_state_fn_list must be the " "same length as optimizers if provided"
            )

        if config.bf16:
            if isinstance(optimizers[0], Float16OptimizerWithFloat16Params):
                raise TypeError('LayerWiseDistributedOptimizer received Float16 optimizer already.')
            optimizers = [
                Float16OptimizerWithFloat16Params(optim, config, None, init_state_fn_list[idx])
                for idx, optim in enumerate(optimizers)
            ]
        super().__init__(optimizers)

        # TODO(kunlun, deyuf): potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in contiguous
        # gbuf, so instead of shard param by layer randomly, we can still shard by buf range but
        # keep some "extras" to keep boundary weight not sharded. This way each rank do some
        # duplicated work but we can call single allgather later and all current distopt
        # optimization can be applied.

    def shard_params(self, optimizers):
        """Shard all params into lists by rank."""
        # We'll optimize sharding later if there is perf issue. should be ok since linear are
        # grouped already.
        # Key is to create separate sharding for dp/expt parallel, saved in dp_cp_params_list,
        # expt_dp_params_list.
        # Example of 4 dp rank and 10 non-expert parameters p0-p9, then dp_cp_params_list will
        # look like: [[p0, p4, p8], [p1, p5, p9], [p2, p6], [p3, p7]]

        # simplify when dp_cp group size is 1
        if get_pg_size(self.pg_collection.dp_cp) == 1:
            self.dp_cp_params_list = None
            self.expt_dp_params_list = None
            return

        dp_cp_idx, expt_dp_idx = 0, 0
        dp_cp_size = get_pg_size(self.pg_collection.dp_cp)
        expt_dp_size = get_pg_size(self.pg_collection.expt_dp)
        self.dp_cp_params_list = [[] for _ in range(dp_cp_size)]
        self.expt_dp_params_list = [[] for _ in range(expt_dp_size)]
        # get all param groups, this is called before init so cannot rely on
        # Chained optimizer method
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups
        for group in param_groups:
            params_this_rank = []
            if group.get("is_expert_parallel", False):
                for p in group["params"]:
                    if expt_dp_idx == get_pg_rank(self.pg_collection.expt_dp):
                        params_this_rank.append(p)
                    self.expt_dp_params_list[expt_dp_idx].append(p)
                    expt_dp_idx = (expt_dp_idx + 1) % expt_dp_size
            else:
                for p in group["params"]:
                    if dp_cp_idx == get_pg_rank(self.pg_collection.dp_cp):
                        params_this_rank.append(p)
                    self.dp_cp_params_list[dp_cp_idx].append(p)
                    dp_cp_idx = (dp_cp_idx + 1) % dp_cp_size
            # now we modify the group to only handle local params
            group["params"] = params_this_rank

        # simplify when expt_dp group size is 1 or expert parallel is off
        if expt_dp_size == 1 or len(self.expt_dp_params_list[0]) == 0:
            self.expt_dp_params_list = None

    @torch.no_grad()
    def broadcast_params(self):
        """All rank broadcast updated local params(allgatherv)."""
        # Broadcast linear layer weights to all other ranks.
        # This may not be slower than PyTorch allgatherv which calls broadcast internally.
        # TODO(skyw): Profile and implement more efficient version.
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
        self.broadcast_params()

        return update_successful, grad_norm, num_zeros_in_grad

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
            if isinstance(sh_base, ShardedTensor):
                assert (
                    len(sh_base.replica_id) == 3
                ), f'Expected replica_id format (PP, TP, DP), got: {sh_base}'
                sh_base.replica_id = (*sh_base.replica_id[:2], 0)

        return sharded_state_dict
