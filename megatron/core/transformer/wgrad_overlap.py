# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Release points for weight gradients deferred by ``overlap_wgrad_compute``."""

from typing import Callable

import torch


class _RunAfterBackward(torch.autograd.Function):
    """Identity whose backward runs a callback: placed on a module's input, the callback runs
    right after the module's backward pass."""

    @staticmethod
    def forward(ctx, callback: Callable[[], None], tensor: torch.Tensor):
        """Store the callback and pass the tensor through."""
        ctx.callback = callback
        return tensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        """Run the callback once the gradient of the module's input is complete."""
        ctx.callback()
        ctx.callback = None
        return None, grad


def release_deferred_wgrads_after(module: torch.nn.Module, callback: Callable[[], None]) -> None:
    """Run ``callback`` right after ``module``'s backward pass, by a forward pre-hook that wraps
    the module's input (its first positional argument or ``hidden_states``).

    The callback issues the weight gradient computations the module's backward pass deferred
    (``backward_dw``), so they run behind the data gradients that produced their inputs.
    """

    def wrap(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
            return _RunAfterBackward.apply(callback, tensor)
        return tensor

    def pre_hook(_, args, kwargs):
        if args:
            return (wrap(args[0]),) + tuple(args[1:]), kwargs
        if 'hidden_states' in kwargs:
            kwargs = {**kwargs, 'hidden_states': wrap(kwargs['hidden_states'])}
        return args, kwargs

    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
