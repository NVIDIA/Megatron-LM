# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Training helpers for the MFSDP v2 MCore integration tests."""

import contextlib

import torch

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


def run_combined_1f1b_on_microbatches(model, config, microbatches):
    """Run a step's microbatches through MCore's combined 1F1B EP-overlap schedule.

    The counterpart of run_forward_backward_on_microbatches for
    ``config.overlap_moe_expert_parallel_comm``: the schedule interleaves one microbatch's
    forward with another's backward. Each microbatch holds GPTModel forward arguments
    including labels; its loss is the sum of the per-token losses, which the schedule scales
    by the number of microbatches before backward. Return the unscaled detached losses with
    gradients ready to use.
    """
    from megatron.core.pipeline_parallel.combined_1f1b import (
        combined_1f1b_schedule_for_no_pipelining,
    )

    def loss_func(output_tensor):
        loss = output_tensor.float().sum()
        return loss, {"loss": loss}

    def forward_step(data_iterator, model, return_schedule_plan=False):
        data = next(data_iterator)
        if return_schedule_plan:
            return model.build_schedule_plan(**data), loss_func
        return model(**data), loss_func

    num_microbatches = len(microbatches)
    forward_data_store = []
    combined_1f1b_schedule_for_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=iter(microbatches),
        model=model,
        num_microbatches=num_microbatches,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=forward_data_store,
        config=config,
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=False,
        no_sync_func=model.no_sync,
        total_num_tokens=torch.zeros([], dtype=torch.int, device="cuda"),
        check_first_val_step=lambda cond: cond,
    )
    model.finish_grad_sync()
    return [store["loss"].detach() * num_microbatches for store in forward_data_store]
