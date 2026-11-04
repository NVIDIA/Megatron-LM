# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training helpers for the standalone MFSDP v2 tests."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import FsdpContext, microbatch


@dataclass(frozen=True, kw_only=True)
class MSETrainer:
    """Settings for training parity tests with fixed MSE microbatches.

    For an unsharded reference with rank-dependent inputs, reduce_grads averages
    gradients across ranks before each optimizer step. FSDP models perform their
    own gradient synchronization.
    """

    num_steps: int = 5
    set_to_none: bool = True
    reduce_grads: bool = False

    def train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        microbatches: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        context: FsdpContext,
    ) -> list[torch.Tensor]:
        """Return detached losses, averaging the backward loss over microbatches."""
        losses = []
        num_microbatches = len(microbatches)
        for _ in range(self.num_steps):
            optimizer.zero_grad(set_to_none=self.set_to_none)
            for index, (inputs, targets) in enumerate(microbatches):
                with microbatch(context, is_last=index == num_microbatches - 1):
                    loss = torch.nn.functional.mse_loss(model(inputs), targets)
                    losses.append(loss.detach())
                    (loss / num_microbatches).backward()
            if self.reduce_grads:
                for parameter in model.parameters():
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            optimizer.step()
        return losses
