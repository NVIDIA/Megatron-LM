from typing import List
from megatron.core.optimizer.optimizer import MegatronOptimizer
import torch
import math
from .optimizer import ChainedOptimizer, multi_tensor_applier, multi_tensor_scale_impl
from .offload_distrib_optimizer import OffloadDistributedOptimizer

class ChainedOffloadOptimizer(ChainedOptimizer):

    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        for optimizer in chained_optimizers:
            if not isinstance(optimizer, OffloadDistributedOptimizer):
                raise ValueError(
                    "ChainedOffloadOptimizer should only be used with OffloadDistributedOptimizer!"
                )
        self.chained_optimizers: List[OffloadDistributedOptimizer] = chained_optimizers

    @torch.no_grad()
    def prepare_grads(self, mem_stats) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        found_inf_flag = False
        for optimizer in self.chained_optimizers:
            optimizer._mem_stats = mem_stats
            found_inf_flag |= optimizer.prepare_grads()

        return found_inf_flag
    
    @torch.no_grad()
    def step(self, mem_stats=None):
        """ChainedOptimizer will step all optimizers one by one."""
        found_inf_flag = self.prepare_grads(mem_stats)
        if found_inf_flag:
            return False, None, None

        # Get grad norm.
        grad_norms = []
        for optimizer in self.chained_optimizers:
            _grad_norm = optimizer.get_grad_norm()
            grad_norms += [_grad_norm if _grad_norm else 0.0]
        grad_norm = math.sqrt(sum([x**2 for x in grad_norms]))

        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if optimizer.config.clip_grad > 0.0:
                grads = []
                for g in optimizer._main_grads:
                    assert g.type() == 'torch.cuda.FloatTensor'
                    grads.append(g.detach())

                # Scale.
                clip_coeff = optimizer.config.clip_grad / (grad_norm + 1.0e-6)
                if clip_coeff < 1.0:
                    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
                    multi_tensor_applier(
                        multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], clip_coeff
                    )

        # Count the zeros in the grads.
        num_zeros_in_grad = 0
        for optimizer in self.chained_optimizers:
            num_zeros_in_grad += (
                optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
            )

        update_successful = self.step_with_ready_grads()

        return update_successful, grad_norm, num_zeros_in_grad