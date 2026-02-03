# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Optimizer for MIMO models with heterogeneous parallelism."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.process_groups_config import ProcessGroupCollection


@dataclass
class ModuleOptimizerInfo:
    """Optimizer info for a single module."""

    optimizer: Optional[MegatronOptimizer]
    grid: Any  # HyperCommGrid
    pg_collection: Optional[ProcessGroupCollection]
    is_active: bool


class MimoOptimizer(MegatronOptimizer):
    """
    Optimizer for MimoModel with heterogeneous parallelism.

    Each module gets its own optimizer. Global gradient norm is computed
    across all modules via all_reduce MAX.
    """

    def __init__(
        self,
        module_infos: Dict[str, ModuleOptimizerInfo],
        config: OptimizerConfig,
    ):
        self.module_infos = module_infos
        self.config = config
        self._active_optimizers: List[MegatronOptimizer] = [
            info.optimizer
            for info in module_infos.values()
            if info.is_active and info.optimizer is not None
        ]
        self.is_stub_optimizer = len(self._active_optimizers) == 0
        self.optimizer = None  # Base class compat

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        found_inf = False
        for opt in self._active_optimizers:
            found_inf |= opt.prepare_grads()
        return found_inf

    @torch.no_grad()
    def get_grad_norm(self) -> float:
        """Compute global gradient norm across all modules via all_reduce MAX."""
        num_modules = len(self.module_infos)
        norm_sq = torch.zeros(num_modules, device="cuda", dtype=torch.float32)

        for i, (name, info) in enumerate(sorted(self.module_infos.items())):
            if info.is_active and info.optimizer:
                module_norm = info.optimizer.get_grad_norm() or 0.0
                norm_sq[i] = module_norm**2

        torch.distributed.all_reduce(norm_sq, op=torch.distributed.ReduceOp.MAX)
        return torch.sqrt(norm_sq.sum()).item()

    @torch.no_grad()
    def step(self) -> Tuple[bool, Optional[float], Optional[int]]:
        found_inf = self.prepare_grads()
        if found_inf:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # Clip with global norm
        for opt in self._active_optimizers:
            if getattr(opt, "is_stub_optimizer", False):
                continue
            params = opt.get_parameters()
            if params and opt.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    params,
                    max_norm=opt.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=opt.config.use_precision_aware_optimizer,
                )

        num_zeros = self.count_zeros() if self.config.log_num_zeros_in_grad else None
        success = self.step_with_ready_grads()

        return success, grad_norm, num_zeros

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        success = True
        for opt in self._active_optimizers:
            success &= opt.step_with_ready_grads()
        return success

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._active_optimizers:
            opt.zero_grad(set_to_none)

    def get_loss_scale(self) -> torch.Tensor:
        if self._active_optimizers:
            return self._active_optimizers[0].get_loss_scale()
        return torch.tensor([1.0], dtype=torch.float32, device="cuda")

    def count_zeros(self) -> int:
        return sum(opt.count_zeros() for opt in self._active_optimizers)

    @property
    def param_groups(self) -> List[dict]:
        groups = []
        for opt in self._active_optimizers:
            groups.extend(opt.param_groups)
        return groups

    # Checkpointing

    def state_dict(self):
        return {
            name: info.optimizer.state_dict() if info.is_active and info.optimizer else None
            for name, info in self.module_infos.items()
        }

    def load_state_dict(self, state_dict: Dict):
        for name, info in self.module_infos.items():
            if info.is_active and info.optimizer and state_dict.get(name):
                info.optimizer.load_state_dict(state_dict[name])

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        sharded_state = {}
        for name, info in self.module_infos.items():
            if info.is_active and info.optimizer:
                sharded_state[name] = info.optimizer.sharded_state_dict(
                    model_sharded_state_dict, is_loading, **kwargs
                )
        return sharded_state

    def reload_model_params(self, state_dict=None):
        for opt in self._active_optimizers:
            opt.reload_model_params(state_dict)


def _get_pg_collection_from_grid(grid) -> ProcessGroupCollection:
    """Create ProcessGroupCollection from HyperCommGrid."""
    import torch.distributed as dist

    from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage

    pg = ProcessGroupCollection()
    pg.tp = grid.get_pg("tp")
    pg.cp = grid.get_pg("cp")
    pg.pp = grid.get_pg("pp")
    pg.ep = grid.get_pg("ep")
    pg.dp = grid.get_pg("dp")
    pg.dp_cp = grid.get_pg(["dp", "cp"])

    # Embedding groups
    if pg.pp:
        pp_ranks = sorted(dist.get_process_group_ranks(pg.pp))
        pos_embd_ranks = [pp_ranks[0]]
        embd_ranks = [pp_ranks[0]]
        if pp_ranks[-1] != pp_ranks[0]:
            embd_ranks.append(pp_ranks[-1])

        pos_embd_pg = dist.new_group(ranks=pos_embd_ranks)
        embd_pg = dist.new_group(ranks=embd_ranks)

        pg.pos_embd = pos_embd_pg if is_pp_first_stage(pg.pp) else None
        pg.embd = embd_pg if (is_pp_last_stage(pg.pp) or is_pp_first_stage(pg.pp)) else None

    pg.mp = grid.get_pg("tp")
    return pg


def get_mimo_optimizer(
    mimo_model: "MimoModel",
    config: OptimizerConfig,
) -> MimoOptimizer:
    """Create optimizer for MimoModel with heterogeneous parallelism."""
    from megatron.core.optimizer import get_megatron_optimizer

    grid_map = mimo_model.mimo_config.module_to_grid_map
    lang_key = mimo_model.mimo_config.language_module_key

    module_infos: Dict[str, ModuleOptimizerInfo] = {}

    for module_name, grid in grid_map.items():
        is_active = grid.is_rank_in_grid()

        optimizer = None
        pg_collection = None

        if is_active:
            if module_name == lang_key:
                module = mimo_model.language_model
            else:
                module = mimo_model.modality_submodules.get(module_name)

            if module is not None:
                pg_collection = _get_pg_collection_from_grid(grid)
                optimizer = get_megatron_optimizer(
                    config=config,
                    model_chunks=[module],
                    pg_collection=pg_collection,
                )

        module_infos[module_name] = ModuleOptimizerInfo(
            optimizer=optimizer,
            grid=grid,
            pg_collection=pg_collection,
            is_active=is_active,
        )

    return MimoOptimizer(module_infos, config)
