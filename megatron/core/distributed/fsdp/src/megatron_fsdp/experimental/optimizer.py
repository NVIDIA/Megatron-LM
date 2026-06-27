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

"""Optimizer adapter for the minimal Megatron-FSDP path."""

from dataclasses import dataclass
from typing import Any, TypeVar

import torch
from torch import nn

from .parameter_group import contained_in_parameter_group

_OptimizerT = TypeVar("_OptimizerT", bound=torch.optim.Optimizer)


def fully_shard_optimizer(optimizer: _OptimizerT) -> None:
    """Attach FSDP-aware step hooks to an optimizer instance.

    The adapted optimizer preserves its existing parameter groups and only adds
    temporary gradient casting around optimizer steps for FSDP sharded
    parameters whose data dtype differs from their grad dtype.

    Args:
        optimizer: Optimizer instance to adapt in place.

    """
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "fully_shard_optimizer expected a torch.optim.Optimizer instance, "
            f"got {optimizer!r}."
        )

    @dataclass
    class CastedGrad:
        parameter: nn.Parameter
        original_grad: torch.Tensor

    casted_grads: list[CastedGrad] = []

    def step_pre_hook(
        hooked_optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        closure = kwargs.get("closure")
        if closure is None and len(args) > 1:
            closure = args[1]
        if closure is not None:
            # Step hooks run outside the base optimizer step, but closures run inside it.
            # We need to cast grads after the closure materializes them and before the
            # optimizer consumes them, which this hook-only adapter cannot intercept.
            raise NotImplementedError(
                "fully_shard_optimizer does not support optimizer.step closures."
            )
        assert not casted_grads
        for group in hooked_optimizer.param_groups:
            for parameter in group["params"]:
                if not isinstance(parameter, nn.Parameter):
                    raise TypeError(
                        "fully_shard_optimizer expected optimizer param groups to contain "
                        f"nn.Parameter values, got {type(parameter)!r}."
                    )
                if not contained_in_parameter_group(parameter):
                    continue
                if parameter.grad is None:
                    continue
                if parameter.grad.dtype == parameter.dtype:
                    continue

                original_grad = parameter.grad
                casted_grads.append(CastedGrad(parameter, original_grad))

                # Clear the existing grad before switching grad_dtype; the sharded
                # parameter cannot advertise a new grad dtype while the old grad
                # object with the previous dtype is still attached.
                parameter.grad = None
                parameter.grad_dtype = parameter.dtype
                parameter.grad = original_grad.to(dtype=parameter.dtype)

    def step_post_hook(
        hooked_optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del hooked_optimizer, args, kwargs
        for casted_grad in casted_grads:
            parameter = casted_grad.parameter
            original_grad = casted_grad.original_grad
            parameter.grad = None
            parameter.grad_dtype = original_grad.dtype
            parameter.grad = original_grad
        casted_grads.clear()

    optimizer.register_step_pre_hook(step_pre_hook)
    optimizer.register_step_post_hook(step_post_hook)
