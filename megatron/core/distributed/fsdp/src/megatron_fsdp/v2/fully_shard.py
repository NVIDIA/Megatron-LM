# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Public fully_shard API for Megatron-FSDP2.

The implementation is split across:
- fsdp_module.py: FSDPModule behavior and runtime state
- hooks.py: forward/backward hook registration
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from .allocator import StorageFreeingBucketAllocator, TracePoolAllocator
from .fsdp_module import FSDPModule
from .hooks import (
    _register_backward_hook,
    _register_backward_pre_hook,
    _register_forward_hook,
    _register_forward_pre_hook,
)
from .mixed_precision import MixedPrecisionPolicy
from .utils import _init_default_fully_shard_mesh

__all__ = ["FSDPModule", "fully_shard"]


def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Optional[bool | int] = None,  # TODO: implement
    shard_placement_fn: Optional[
        Callable[[nn.Parameter], Optional[Shard]]
    ] = None,  # TODO: implement
    mp_policy: Optional[MixedPrecisionPolicy] = None,
    offload_policy: Optional["OffloadPolicy"] = None,  # TODO: implement
    ignored_params: Optional[set[nn.Parameter]] = None,
    # --- Megatron-FSDP specific options ---
    enable_unshard_prefetch: bool = True,
    enable_async_reduce_grad: bool = True,
    gradient_scaling_factor: Optional[float] = None,
    enable_trace_pool: bool = False,
    sharding_strategy: str = "optim_grads_params",
    enable_cuda_graph: bool = False,
    fine_grained_hooks: bool = False,
    skip_backward_callback: bool = False,  # Skip autograd RegisterFSDPBackwardFunction.
    skip_final_backward_callback: bool = False,
) -> nn.Module:
    """
    Wrap a module with FSDP sharding semantics.

    This function:
    1. Converts the module class to FSDPModule dynamically (mixin pattern)
    2. Groups parameters by (device, dtype, requires_grad)
    3. Creates ParameterGroup for each group with dedicated buffers
    4. Registers forward/backward hooks for unshard/reshard/reduce
    5. Replaces module parameters with DTensor representations

    Args:
        fine_grained_hooks: If ``True``, register pre-forward/backward hooks
            on every sub-module (for EP-overlap / 1F1B schedules).
        skip_backward_callback: If ``True``, skip the autograd post-backward
            hook (``_register_backward_hook``).  Per-layer reshard + reduce_grad
            still fires via ``set_fsdp_reshard_hooks`` / ``mfsdp_post_backward_hook``
            for the TransformerLayer; remaining modules (e.g., TEGroupedMLP) are
            handled by ``mfsdp_post_backward_final_callback`` after all
            ``backward_dw()`` calls complete.  Set when ``delay_wgrad_compute=True``
            because weight gradients are not ready during autograd backward.
        skip_final_backward_callback: If ``True``, do not auto-enqueue
            ``_register_post_backward_final_callback`` during the backward
            pre-hook.  The caller must invoke the final callback manually
            (used by the 1F1B EP overlap schedule).
    """
    if isinstance(module, FSDPModule):
        raise ValueError(
            "The input module has already been fully sharded. "
            "Please do not call fully_shard on the same module more than once."
        )
    mesh = mesh or _init_default_fully_shard_mesh()

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy()

    cls = module.__class__
    new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), {})
    module.__class__ = new_cls

    use_trace_pool = (
        enable_trace_pool
        or enable_cuda_graph
        or any(
            getattr(m._fsdp_state, "enable_cuda_graph", False)
            for m in module.modules()
            if isinstance(m, FSDPModule) and m is not module
        )
    ) and sharding_strategy in (
        "optim",
        "optim_grads",
        "optim_grads_params",
    )
    bucket_allocator = TracePoolAllocator() if use_trace_pool else StorageFreeingBucketAllocator()

    module._init_named_param_groups(
        mesh,
        ignored_params,
        mp_policy=mp_policy,
        gradient_scaling_factor=gradient_scaling_factor,
        sharding_strategy=sharding_strategy,
    )
    module._init_fsdp_state(
        enable_unshard_prefetch=enable_unshard_prefetch,
        enable_async_reduce_grad=enable_async_reduce_grad,
        bucket_allocator=bucket_allocator,
        enable_cuda_graph=enable_cuda_graph,
    )
    module._init_param_main_grad_func()

    _register_forward_pre_hook(
        module,
        fine_grained=(
            fine_grained_hooks
            or mp_policy.fine_grained_forward_hooks_required(module._fsdp_param_groups)
        ),
    )
    _register_forward_hook(module)
    _register_backward_pre_hook(
        module,
        fine_grained=fine_grained_hooks,
        skip_final_callback=skip_final_backward_callback,
    )
    # When delay_wgrad_compute is enabled, skip the autograd post-backward
    # hook.  Per-layer reshard+reduce_grad still fires via set_fsdp_reshard_hooks
    # for TransformerLayer; remaining modules (TEGroupedMLP, root) are handled
    # by mfsdp_post_backward_final_callback after all backward_dw() calls.
    #
    # When disabled, the autograd hook does both inline.
    if not skip_backward_callback:
        _register_backward_hook(module)

    module.reshard()

    return module
