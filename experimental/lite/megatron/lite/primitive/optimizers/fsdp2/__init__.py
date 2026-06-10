"""FSDP2 primitive surface."""

from __future__ import annotations

from megatron.lite.primitive.optimizers.fsdp2.grad_clip import (
    all_reduce_scalar_,
    clip_grads_with_sharded_norm_,
    resolve_torch_dtype,
    sharded_grad_abs_max,
    sharded_grad_norm,
    sharded_grad_sq_sum,
)
from megatron.lite.primitive.optimizers.fsdp2.optimizer import (
    BACKEND,
    FSDP2Optimizer,
    FSDP2OptimizerBackend,
    build_fsdp2_adamw,
    build_fsdp2_training_optimizer,
)
from megatron.lite.primitive.optimizers.fsdp2.wrap import (
    FSDP2Config,
    build_fsdp2_device_mesh,
    build_fsdp2_process_group_mesh,
    build_fsdp2_shard_placement_fn,
    fsdp2_available,
    promote_fsdp2_trainable_params_to_fp32,
    set_fsdp2_requires_gradient_sync,
    wrap_fsdp2,
    wrap_fsdp2_module,
)

__all__ = [
    "BACKEND",
    "FSDP2Config",
    "FSDP2Optimizer",
    "FSDP2OptimizerBackend",
    "all_reduce_scalar_",
    "build_fsdp2_adamw",
    "build_fsdp2_training_optimizer",
    "build_fsdp2_device_mesh",
    "build_fsdp2_process_group_mesh",
    "build_fsdp2_shard_placement_fn",
    "clip_grads_with_sharded_norm_",
    "fsdp2_available",
    "promote_fsdp2_trainable_params_to_fp32",
    "resolve_torch_dtype",
    "set_fsdp2_requires_gradient_sync",
    "sharded_grad_abs_max",
    "sharded_grad_norm",
    "sharded_grad_sq_sum",
    "wrap_fsdp2",
    "wrap_fsdp2_module",
]
