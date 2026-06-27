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

from typing import Any, NamedTuple

import torch
from torch import nn

from .parameter_group import contained_in_parameter_group


def fully_shard_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Attach FSDP-aware step hooks to an optimizer instance.

    The adapted optimizer preserves its existing parameter groups and only adds
    temporary gradient casting around optimizer steps for FSDP sharded
    parameters whose data dtype differs from their grad dtype.

    Alternatives considered:
        - Monkey-patching optimizer methods directly on the instance. This is
          more invasive and harder to compose than hooks.
        - Generating an FSDP-specific subclass per ``torch.optim.Optimizer``.
          This adds extra class-generation machinery, but would let us
          instrument ``zero_grad`` and ``__init__`` as well as ``step`` if needed.

    Args:
        optimizer: Optimizer instance to adapt in place.
    """
    class CastedGrad(NamedTuple):
        parameter: nn.Parameter
        original_grad: torch.Tensor

    def set_grad(parameter: nn.Parameter, grad: torch.Tensor) -> None:
        """Install a grad with matching grad_dtype on a sharded parameter."""
        # Clear the existing grad before switching grad_dtype; the sharded
        # parameter cannot advertise a new grad dtype while the old grad
        # object with the previous dtype is still attached.
        parameter.grad = None
        parameter.grad_dtype = grad.dtype
        parameter.grad = grad

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

                casted_grads.append(CastedGrad(parameter, parameter.grad))
                set_grad(parameter, parameter.grad.to(dtype=parameter.dtype))

    def step_post_hook(
        hooked_optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del hooked_optimizer, args, kwargs
        for parameter, original_grad in casted_grads:
            set_grad(parameter, original_grad)
        casted_grads.clear()

    optimizer.register_step_pre_hook(step_pre_hook)
    optimizer.register_step_post_hook(step_post_hook)
