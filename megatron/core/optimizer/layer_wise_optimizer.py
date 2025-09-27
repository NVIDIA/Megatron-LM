from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .optimizer import ChainedOptimizer, MegatronOptimizer, Float16OptimizerWithFloat16Params
from .optimizer_config import OptimizerConfig
from .clip_grads import clip_grad_by_total_norm_fp32, count_zeros_fp32, get_grad_norm_fp32

class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    This is a experimental distributed optimizer wrapper that distributes weight to DP ranks by full layer.
    It is implemented as a Chained optimizer to support cases where different weight group uses different
    optimizers (adam + muon) for example.
    When using, keep distributed optimizer related options OFF.
    """
    def __init__(
        self,
        optimizers: List[MegatronOptimizer],
        grad_comm_pg: torch.distributed.ProcessGroup,
        ep_grad_comm_pg: torch.distributed.ProcessGroup,
        config: OptimizerConfig,
    ) -> None:
        self.grad_comm_pg = grad_comm_pg
        self.ep_grad_comm_pg = ep_grad_comm_pg
        self.shard_params(optimizers)
        # wrap optimizer after sharding to avoid unnecessary master weight creation
        if config.bf16:
            if isinstance(optimizers[0], Float16OptimizerWithFloat16Params):
                raise TypeError('LayerWiseDistributedOptimizer received Float16 optimizer.')
            optimizers = [Float16OptimizerWithFloat16Params(optim, config, None, None) for optim in optimizers]
        super().__init__(optimizers)

        # how LayerWiseDistributedOptimizer work:
        # 1. Megatron DDP handle allreduce grad for all params
        # 2. optimizer is modified so only param belong to this DP rank is kept
        # 3. Do regular update with chained optimizers, get_gran_norm and count_zeros are overwritten
        # 4. allgather updated params to every rank(currently through broadcast loop)

        # TODO: (kunlun, deyu) potential future perf optimization
        # since allreduce is unchanged and handled by megatron DDP, they're already in contiguous gbuf
        # so instead of shard param by layer randomly, we can still shard by buf range but keep some "extras"
        # to keep boundary weight not sharded. This way each rank do some duplicated work but we can call
        # single allgather later and all current distopt optimization can be applied

    def shard_params(self, optimizers):
        """Shard all params into lists by rank. """
        # keep logic simple now since we'll optimize sharding later. should be ok since linear are separate already
        # separate dp, ep_dp
        dp_idx, ep_idx = 0, 0
        dp_size = self.grad_comm_pg.size()
        dp_rank = self.grad_comm_pg.rank()
        ep_size = self.ep_grad_comm_pg.size()
        ep_rank = self.ep_grad_comm_pg.rank()
        self.shard_params_list = [[] for _ in range(dp_size)]
        self.ep_params_list = [[] for _ in range(ep_size)]
        # simplify when DP size is 1
        if dp_size == 1:
            self.shard_params_list = []
            self.ep_params_list = []
            return

        # get all param groups, this is called before init so cannot rely on Chained optimizer method
        param_groups = []
        for optimizer in optimizers:
            param_groups += optimizer.param_groups
        for group in param_groups:
            params_this_rank = []
            if group["is_expert_parallel"]:
                for p in group["params"]:
                    if ep_idx == ep_rank:
                        params_this_rank.append(p)
                    self.ep_params_list[ep_idx].append(p)
                    ep_idx = (ep_idx + 1) % ep_size
            else:
                for p in group["params"]:
                    if dp_idx == dp_rank:
                        params_this_rank.append(p)
                    self.shard_params_list[dp_idx].append(p)
                    dp_idx = (dp_idx + 1) % dp_size
            # now we modify the group to only handle local params
            group["params"] = params_this_rank

        # simplify when EP size is 1 or EP is off
        if ep_size == 1 or len(self.ep_params_list[0]) == 0:
            self.ep_params_list = []

    @torch.no_grad()
    def broadcast_params(self):
        """All rank broadcast updated local params(allgatherv). """
        # Broadcast linear layer weights to all other ranks.
        # This may not be slower than PyTorch allgatherv which calls broadcast internally.
        # TODO(skyw): Profile and implement more efficient version.
        for i, params in enumerate(self.shard_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.grad_comm_pg, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.grad_comm_pg)
        for i, params in enumerate(self.ep_params_list):
            src_global_rank = torch.distributed.get_global_rank(self.ep_grad_comm_pg, i)
            for p in params:
                torch.distributed.broadcast(p, src_global_rank, self.ep_grad_comm_pg)

    @torch.no_grad()
    def get_grad_norm(self):
        # similar to dist opt, always aggregate globally
        grads_for_norm = []
        for optimizer in self.chained_optimizers:
            grads_for_norm += optimizer.get_main_grads_for_grad_norm()
        grad_norm = get_grad_norm_fp32(
            grads_for_norm, grad_stats_parallel_group=None
        )
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

    def save_state_dict_to_file(self, filename: str) -> None:
        """Save the parameter state of the optimizer.

        Args:
            filename: The filename to save the parameter state.
        """
        torch.save(super().state_dict(), filename)

    def load_state_dict_from_file(self, filename: str) -> None:
        """Load the parameter state of the optimizer."""
        super().load_state_dict(torch.load(filename))


