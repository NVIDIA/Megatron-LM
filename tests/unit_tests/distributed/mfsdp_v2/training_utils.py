# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training helpers for the standalone MFSDP v2 tests."""

from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import FsdpContext, microbatch


def train_with_mse(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    microbatches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    context: FsdpContext,
    num_steps: int = 5,
    set_to_none: bool = True,
    reduce_grads: bool = False,
) -> list[torch.Tensor]:
    """Train on fixed MSE microbatches and return detached, unscaled losses.

    Average the backward loss over microbatches. For an unsharded reference with
    rank-dependent inputs, reduce_grads averages gradients across ranks before
    each optimizer step; FSDP models perform their own gradient synchronization.
    """
    losses = []
    num_microbatches = len(microbatches)
    for _ in range(num_steps):
        optimizer.zero_grad(set_to_none=set_to_none)
        for index, (inputs, targets) in enumerate(microbatches):
            with microbatch(context, is_last=index == num_microbatches - 1):
                loss = torch.nn.functional.mse_loss(model(inputs), targets)
                losses.append(loss.detach())
                (loss / num_microbatches).backward()
        if reduce_grads:
            for parameter in model.parameters():
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
        optimizer.step()
    return losses
