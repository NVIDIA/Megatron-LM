from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from .optimizer import ChainedOptimizer, MegatronOptimizer
from .clip_grads import clip_grad_by_total_norm_fp32, count_zeros_fp32, get_grad_norm_fp32

class LayerWiseDistributedOptimizer(ChainedOptimizer):
    """Layer-wise distributed optimizer for Megatron-core models.

    Warning:
        This function is implemented based on assumptions that won't generalize. It is meant for running
        experiments only, not merge to the main codebase. A more generalized implementation will be designed
        after MCore decentralized process group management refactoring is done.

    This is a experimental optimizer that distributes optimizers of linear and embedding layers to different ranks
    with the option to use different optimizer for different layer types.

    (deyu)
    Current assume dist-opt off, DDP on and list of [muon, adam] passed in with params already separated
    During modified step function:
    1. Megatron DDP handle allreduce grad for all params
    2. update subset of params(shard by rank) and drop grad for rest params
    3. broadcast updated params
    """
    def __init__(
        self, 
        chained_optimizers: List[MegatronOptimizer],
        grad_comm_pg: torch.distributed.ProcessGroup = None,
        ep_grad_comm_pg: torch.distributed.ProcessGroup = None,
    ) -> None:
        super().__init__(chained_optimizers)

        # TODO(deyu, kunlun suggested): shard params into ranks by gbuf range(like distopt) but keep "extra"
        # on boundaries so muon weights are not sharded. later drop "extra" and call single allgather
        # essentially it's same as kimi PR but change the "getting full muon weight grad" part into global 
        # allreduce then drop, which we do anyway with current layerwise impl through DDP

        # for initial functional test, evenly shard base on chain order and removed logic duplicating norm param
        self.grad_comm_pg = grad_comm_pg
        self.ep_grad_comm_pg = ep_grad_comm_pg
        # TODO(deyu): support dp/ep 0 cases
        self.shard_params()

    def shard_params(self):
        """Shard all params into lists by rank. """
        # keep logic simple now since we'll optimize sharding later. should be ok since linear are separate already
        # separate dp, ep_dp
        dp_idx, ep_idx = 0, 0
        dp_size = self.grad_comm_pg.size()
        ep_size = self.ep_grad_comm_pg.size()
        self.shard_params_list = [[] for _ in range(dp_size)]
        self.ep_params_list = [[] for _ in range(ep_size)]
        for group in self.param_groups:
            if group["is_expert_parallel"]:
                for p in group["params"]:
                    self.ep_params_list[ep_idx].append(p)
                    ep_idx = (ep_idx + 1) % ep_size
            else:
                for p in group["params"]:
                    self.shard_params_list[dp_idx].append(p)
                    dp_idx = (dp_idx + 1) % dp_size

    def drop_grads(self):
        """Drop grads of params belong to other ranks. """
        for i, params in enumerate(self.shard_params_list):
            if self.grad_comm_pg.rank() != i:
                for p in params:
                    p.grad = None
        for i, params in enumerate(self.ep_params_list):
            if self.ep_grad_comm_pg.rank() != i:
                for p in params:
                    p.grad = None

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
    def step(self):  # type: ignore[no-untyped-def]
        """step function for layer-wise optimizer."""
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # now that we have grad norm, nuke grads belong to other rank
        # clip_grad_by_total_norm_fp32, count_zeros and step_with_ready_grads below all checks
        # if grad is None and will only update params belong to this rank
        self.drop_grads()

        # Begin unchanged chained optimizer code
        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, "is_stub_optimizer") and optimizer.is_stub_optimizer:
                continue
            parameters = optimizer.get_parameters()
            if len(parameters) == 0:
                continue
            if optimizer.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    parameters,
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=(
                        optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                    ),
                )

        # # Count the zeros in the grads.
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else None

        update_successful = self.step_with_ready_grads()
        # End unchanged chained optimizer code

        # All gather updated params.
        self.broadcast_params()
        # TODO(deyu): need to all gather model param instead of main param above. temp fix
        for optim in self.chained_optimizers:
            if hasattr(optim, "_copy_main_params_to_model_params"):
                optim._copy_main_params_to_model_params()

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


