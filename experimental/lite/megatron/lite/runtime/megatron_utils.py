# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Megatron-Core utilities aligned with VERL's megatron_utils.

All functions here are ported from VERL and could theoretically be replaced
by ``from verl.utils.megatron_utils import ...`` if Megatron Lite ever depends on verl.

Sources:
  - verl/utils/megatron_utils.py  (offload/load, register_megatron_training_hooks)
  - verl/utils/megatron/optimizer.py  (get_megatron_last_lr)
"""

from __future__ import annotations

import gc
from typing import Any

import torch

# ======================================================================
# Rank utilities
# ======================================================================


def is_mp_src_rank_with_outputs() -> bool:
    """True on the rank that holds the final model output (loss).

    Only last PP stage, first TP rank, first CP rank has the output.
    VERL: MegatronEngine.is_mp_src_rank_with_outputs
    """
    from megatron.core import parallel_state as mpu

    return (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank()
        == mpu.get_pipeline_model_parallel_world_size() - 1
        and mpu.get_context_parallel_rank() == 0
    )


# ======================================================================
# Training hooks — register_megatron_training_hooks
# ======================================================================


def register_training_hooks(model_list: list, optimizer) -> None:
    """Register megatron training callbacks on model config.

    Ref: megatron/training/training.py (core_v0.15.0rc7, L2039-L2057)
    """
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import finalize_model_grads
    from megatron.core.utils import get_model_config

    for one_model in model_list:
        config = get_model_config(one_model)
        if optimizer is not None:
            config.grad_scale_func = optimizer.scale_loss
        config.finalize_model_grads_func = finalize_model_grads

        optimizer_config = getattr(optimizer, "config", None)
        overlap_param_gather = getattr(optimizer_config, "overlap_param_gather", False)
        overlap_grad_reduce = getattr(one_model.ddp_config, "overlap_grad_reduce", False)
        align_grad_reduce = True
        align_param_gather = getattr(one_model.ddp_config, "align_param_gather", False)

        if isinstance(model_list[0], DDP) and overlap_grad_reduce:
            config.no_sync_func = [m.no_sync for m in model_list]
            if len(model_list) == 1:
                config.no_sync_func = config.no_sync_func[0]
            if align_grad_reduce:
                config.grad_sync_func = [m.start_grad_sync for m in model_list]
                if len(model_list) == 1:
                    config.grad_sync_func = config.grad_sync_func[0]
        if overlap_param_gather and align_param_gather:
            config.param_sync_func = [m.start_param_sync for m in model_list]
            if len(model_list) == 1:
                config.param_sync_func = config.param_sync_func[0]


# ======================================================================
# Model offload / load — offload_megatron_model_to_cpu / load_megatron_model_to_gpu
# ======================================================================


def offload_model_to_cpu(model_list: list) -> None:
    """Offload DDP model to CPU via buffer-resize (zero-copy on GPU side)."""
    from megatron.core.distributed import DistributedDataParallel as DDP

    for model_chunk in model_list:
        if isinstance(model_chunk, DDP):
            all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in all_buffers:
                for buffer in buffers:
                    if buffer.param_data.storage().size() > 0:
                        buffer.param_data.cpu_data = buffer.param_data.data.cpu().pin_memory()
                        buffer.param_data_size = buffer.param_data.storage().size()
                        buffer.param_data.storage().resize_(0)

                    if buffer.grad_data.storage().size() > 0:
                        buffer.grad_data_size = buffer.grad_data.storage().size()
                        buffer.grad_data.storage().resize_(0)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type != "cpu":
                    param.data = param.data.to("cpu", non_blocking=True)
        else:
            model_chunk.to("cpu")


def load_model_to_gpu(model_list: list, load_grad: bool = True) -> None:
    """Load DDP model back to GPU from pinned CPU copy."""
    from megatron.core.distributed import DistributedDataParallel as DDP

    for model_chunk in model_list:
        if isinstance(model_chunk, DDP):
            all_buffers = [model_chunk.buffers, model_chunk.expert_parallel_buffers]
            for buffers in all_buffers:
                for buffer in buffers:
                    if load_grad and hasattr(buffer, "grad_data_size"):
                        current_size = buffer.grad_data.storage().size()
                        if current_size == 0 or current_size == buffer.grad_data_size:
                            buffer.grad_data.storage().resize_(buffer.grad_data_size)
                            buffer.grad_data.zero_()
                        else:
                            buffer.grad_data.zero_()

                    if buffer.param_data.storage().size() == 0:
                        buffer.param_data.storage().resize_(buffer.param_data_size)
                        buffer.param_data.copy_(buffer.param_data.cpu_data, non_blocking=True)

            for param in model_chunk.module.parameters():
                if not param.requires_grad and param.device.type == "cpu":
                    param.data = param.data.to("cuda", non_blocking=True)
        else:
            model_chunk.to("cuda")


# ======================================================================
# Optimizer offload / load — offload_megatron_optimizer / load_megatron_optimizer
# ======================================================================


def offload_optimizer(optimizer) -> None:
    """Offload optimizer states to CPU."""
    from megatron.core.optimizer import ChainedOptimizer

    for _opt in _iter_opts(optimizer, ChainedOptimizer):
        if _opt.optimizer is not None:
            hdo = _opt.optimizer
            if all(
                hasattr(hdo, a) for a in ("sub_optimizers", "inner_param_to_orig_param", "state")
            ):
                for sub_opt in hdo.sub_optimizers:
                    for param, state in sub_opt.state.items():
                        for k, v in state.items():
                            if not isinstance(v, torch.Tensor):
                                continue
                            orig_param = hdo.inner_param_to_orig_param.get(param, param)
                            hdo.state[orig_param][k] = state[k] = v.to("cpu")
            else:
                for v in _opt.optimizer.state.values():
                    if "exp_avg" in v:
                        v["exp_avg"] = v["exp_avg"].to("cpu", non_blocking=True)
                    if "exp_avg_sq" in v:
                        v["exp_avg_sq"] = v["exp_avg_sq"].to("cpu", non_blocking=True)

    gc.collect()
    torch.cuda.empty_cache()


def load_optimizer(optimizer) -> None:
    """Load optimizer states back to GPU."""
    from megatron.core.optimizer import ChainedOptimizer

    for _opt in _iter_opts(optimizer, ChainedOptimizer):
        if _opt.optimizer is not None:
            if hasattr(_opt.optimizer, "_move_new_state_to_right_device"):
                _opt.optimizer._move_new_state_to_right_device()
            else:
                for v in _opt.optimizer.state.values():
                    if "exp_avg" in v:
                        v["exp_avg"] = v["exp_avg"].to("cuda", non_blocking=True)
                    if "exp_avg_sq" in v:
                        v["exp_avg_sq"] = v["exp_avg_sq"].to("cuda", non_blocking=True)

    gc.collect()
    torch.cuda.empty_cache()


def _iter_opts(optimizer, chained_cls):
    if isinstance(optimizer, chained_cls):
        return optimizer.chained_optimizers
    return [optimizer]


# ======================================================================
# Checkpoint helpers
# ======================================================================


def build_sharded_state_dict(
    model_list: list, optimizer: Any = None, lr_scheduler: Any = None
) -> dict[str, Any]:
    """Build sharded state dict for model + optimizer + lr_scheduler.

    Uses ``model0.`` / ``model1.`` prefix for VPP (multiple model chunks).
    """
    sharded_state_dict: dict[str, Any] = {}

    for i, model_chunk in enumerate(model_list):
        prefix = f"model{i}." if len(model_list) > 1 else "model."
        chunk_sd = model_chunk.sharded_state_dict(prefix=prefix)
        sharded_state_dict.update(chunk_sd)

    if optimizer is not None:
        opt_sd = optimizer.sharded_state_dict(model_sharded_state_dict=sharded_state_dict)
        sharded_state_dict.update(opt_sd)

    if lr_scheduler is not None:
        sharded_state_dict["lr_scheduler"] = lr_scheduler.state_dict()

    return sharded_state_dict
