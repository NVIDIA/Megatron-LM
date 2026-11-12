# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training helpers for the MFSDP v2 MCore integration tests."""

import contextlib

import torch.distributed as dist
from torch.distributed.tensor import DTensor

from megatron.core.distributed.data_parallel_base import _BaseDataParallel


# Emulate MCore's training loop without its global state.
# See https://github.com/NVIDIA/Megatron-LM/issues/7223.
def run_forward_backward_on_microbatches(
    model, forward_one_microbatch, *, num_microbatches=1, loss_scale=1.0, delayed_wgrad_compute=None
):
    """Run a step's microbatches and return detached losses with gradients ready to use.

    Like MCore's schedules, accumulate all but the last microbatch under no_sync,
    then finish gradient synchronization after all backward work, including delayed wgrad.
    Keep the optimizer step separate so tests can inspect gradients or capture it separately.
    """
    is_data_parallel = isinstance(model, _BaseDataParallel)
    losses = []
    for index in range(num_microbatches):
        sync_context = (
            model.no_sync()
            if is_data_parallel and index < num_microbatches - 1
            else contextlib.nullcontext()
        )
        with sync_context:
            loss = forward_one_microbatch()
            (loss * loss_scale).backward()
            if delayed_wgrad_compute is not None:
                delayed_wgrad_compute()
        losses.append(loss.detach())
    if is_data_parallel:
        model.finish_grad_sync()
    return losses


def run_training_steps(
    model, forward_one_microbatch, optimizer, *, num_steps, data_parallel_group=None
):
    """Run independent training steps and snapshot losses, gradients, and updated weights.

    Unwrapped reference models can supply a data-parallel group to average gradients.
    Wrapped models use the synchronization in ``run_forward_backward_on_microbatches``.
    Snapshots gather DTensors into full tensors for comparisons with unsharded references.
    """

    def snapshot(tensor):
        value = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
        return value.detach().clone()

    history = []
    for _ in range(num_steps):
        optimizer.zero_grad()
        losses = run_forward_backward_on_microbatches(model, forward_one_microbatch)
        if data_parallel_group is not None:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, group=data_parallel_group)
        gradients = tuple(
            snapshot(parameter.grad) if parameter.grad is not None else None
            for parameter in model.parameters()
        )
        optimizer.step()
        parameters = tuple(snapshot(parameter) for parameter in model.parameters())
        history.append(dict(losses=losses, gradients=gradients, parameters=parameters))
    return history
