from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class ParamGroupIdx:
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
