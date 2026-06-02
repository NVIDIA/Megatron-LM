# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Minimal Megatron-FSDP fully_shard entrypoint."""

from collections.abc import Iterable

import torch
from torch import nn
from torch.distributed import DeviceMesh

from ..mixed_precision import MixedPrecisionPolicy
from .fsdp_module import DelayedRelease, FsdpContext, FsdpModule, _mesh_device
from .placement import Placements


def fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    placements: Placements,
    mixed_precision_policy: MixedPrecisionPolicy | None = None,
) -> None:
    """Shard one module as a per-module FSDP unit.

    Args:
        module: Module whose currently unowned parameters become this FSDP unit.
        mesh: Device mesh used for sharding.
        placements: Parameter, gradient, and optimizer placements.
        mixed_precision_policy: Optional precision policy. Defaults to FP32 main weights
            and parameter-dtype main gradients.
    """
    if isinstance(module, FsdpModule):
        raise ValueError("This module is already managed by FSDP.")

    mixed_precision_policy = mixed_precision_policy or MixedPrecisionPolicy()
    descendant_fsdp_modules = tuple(_iter_descendant_fsdp_modules(module))
    fsdp_context = _select_fsdp_context(descendant_fsdp_modules, _mesh_device(mesh))

    original_cls = module.__class__
    _attach_mixin(module)
    try:
        assert isinstance(module, FsdpModule)
        FsdpModule.__init__(
            module,
            mesh=mesh,
            placements=placements,
            mixed_precision_policy=mixed_precision_policy,
            fsdp_context=fsdp_context,
        )
    except Exception:
        if isinstance(module, FsdpModule):
            module._fsdp_context.remove_module(module)
        module.__class__ = original_cls
        raise
    for descendant_module in descendant_fsdp_modules:
        descendant_module._assign_fsdp_context(fsdp_context)


def _iter_descendant_fsdp_modules(module: nn.Module) -> Iterable[FsdpModule]:
    for child_module in module.modules():
        if child_module is module:
            continue
        if isinstance(child_module, FsdpModule):
            yield child_module


def _select_fsdp_context(
    descendant_fsdp_modules: tuple[FsdpModule, ...], device: torch.device
) -> FsdpContext:
    for descendant_module in descendant_fsdp_modules:
        return descendant_module._fsdp_context
    return FsdpContext(device=device)


def _attach_mixin(module: nn.Module) -> None:
    if isinstance(module, FsdpModule):
        return
    module_cls = module.__class__
    fsdp_cls = type(f"ExperimentalFsdp{module_cls.__name__}", (FsdpModule, module_cls), {})
    module.__class__ = fsdp_cls
