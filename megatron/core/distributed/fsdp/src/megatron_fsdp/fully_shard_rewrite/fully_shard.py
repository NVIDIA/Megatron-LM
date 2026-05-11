"""
Public fully_shard API for the Megatron-FSDP rewrite path.

The implementation is split across:
- fsdp_module.py: FSDPModule behavior and runtime state
- hooks.py: forward/backward hook registration
"""

from typing import Callable, Optional

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from .fsdp_module import FSDPModule
from .hooks import (
    _register_backward_hook,
    _register_backward_pre_hook,
    _register_forward_hook,
    _register_forward_pre_hook,
)
from .mixed_precision import FullyShardMixedPrecisionPolicy

__all__ = ["FSDPModule", "fully_shard"]


def fully_shard(
    module: nn.Module,
    *,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Optional[bool | int] = None,  # TODO: implement
    shard_placement_fn: Optional[
        Callable[[nn.Parameter], Optional[Shard]]
    ] = None,  # TODO: implement
    mp_policy: Optional[FullyShardMixedPrecisionPolicy] = None,  # TODO: implement
    offload_policy: Optional["OffloadPolicy"] = None,  # TODO: implement
    ignored_params: Optional[set[nn.Parameter]] = None,
    # --- Megatron-FSDP specific options ---
    enable_unshard_prefetch: bool = True,
    enable_async_reduce_grad: bool = True,
    gradient_scaling_factor: Optional[float] = None,
) -> nn.Module:
    """
    Wrap a module with FSDP sharding semantics.

    This function:
    1. Converts the module class to FSDPModule dynamically (mixin pattern)
    2. Groups parameters by (device, dtype, requires_grad)
    3. Creates ParameterGroup for each group with dedicated buffers
    4. Registers forward/backward hooks for unshard/reshard/reduce
    5. Replaces module parameters with DTensor representations
    """
    if isinstance(module, FSDPModule):
        raise ValueError(
            "The input module has already been fully sharded. "
            "Please do not call fully_shard on the same module more than once."
        )
    if mp_policy is None:
        mp_policy = FullyShardMixedPrecisionPolicy()

    cls = module.__class__
    new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), {})
    module.__class__ = new_cls

    module._init_named_param_groups(
        mesh,
        ignored_params,
        mp_policy=mp_policy,
        gradient_scaling_factor=gradient_scaling_factor,
    )
    module._init_fsdp_state(
        enable_unshard_prefetch=enable_unshard_prefetch,
        enable_async_reduce_grad=enable_async_reduce_grad,
    )
    module._init_param_main_grad_func()

    _register_forward_pre_hook(module)
    _register_forward_hook(module)
    _register_backward_pre_hook(module)
    _register_backward_hook(module)

    module.reshard()

    return module
