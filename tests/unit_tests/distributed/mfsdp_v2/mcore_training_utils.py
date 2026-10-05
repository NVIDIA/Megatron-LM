# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training helpers for the MFSDP v2 MCore integration tests."""

import contextlib

from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallelV2


def _forward_backward(
    model, microbatches, forward_step, *, loss_scale=1.0, delayed_wgrad_compute=None
):
    """Run a step's microbatches and return detached losses with gradients ready to use.

    Like MCore's schedules, accumulate all but the last microbatch under no_sync,
    then finish gradient synchronization after all backward work, including delayed wgrad.
    Keep the optimizer step separate so tests can inspect gradients or capture it separately.
    """
    is_fsdp = isinstance(model, FullyShardedDataParallelV2)
    losses = []
    for index, batch in enumerate(microbatches):
        sync_context = (
            model.no_sync()
            if is_fsdp and index < len(microbatches) - 1
            else contextlib.nullcontext()
        )
        with sync_context:
            loss = forward_step(model, batch)
            (loss * loss_scale).backward()
            if delayed_wgrad_compute is not None:
                delayed_wgrad_compute()
        losses.append(loss.detach())
    if is_fsdp:
        model.finish_grad_sync()
    return losses
