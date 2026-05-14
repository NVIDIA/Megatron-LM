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

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class ParamGroupIdx:
    """Immutable identifier for a ParameterGroup: (module_id, index)."""

    module_id: int
    index: int


class RegisterFSDPBackwardFunction(torch.autograd.Function):
    """
    Autograd Function for registering post-backward hooks.

    This Function simply passes inputs through in forward, but its
    backward calls the post_backward_hook to perform reshard and
    gradient reduction after gradients are computed.
    """

    @staticmethod
    def forward(ctx, post_backward: Callable, *inputs: torch.Tensor):
        ctx.post_backward = post_backward
        return inputs

    @staticmethod
    def backward(ctx, *grads: torch.Tensor):
        ctx.post_backward()
        return (None,) + grads


def _replace_module_parameter(module: nn.Module, name: str, new_param: nn.Parameter):
    """
    Replace a module's parameter while preserving module hierarchy.

    Example:
        If name="layers.0.linear1.weight", this finds module.layers[0].linear1
        and replaces its weight parameter.
    """
    parts = name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_param)
