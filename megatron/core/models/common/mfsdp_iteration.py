# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Iteration boundary shared by MFSDP v2 pipeline schedule entry points."""

from contextlib import ExitStack
from functools import wraps
from inspect import signature

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.iteration import iteration
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule


def with_mfsdp_iteration(schedule):
    """Record/replay hooks across the complete schedule, including every VPP chunk.

    Nested combined calls join the outer scope. Non-MFSDP models and CUDA graph
    schedules retain their existing behavior; graph replay bypasses Python hooks
    and needs a separate recording lifecycle.
    """
    parameters = signature(schedule)

    @wraps(schedule)
    def wrapped(*args, **kwargs):
        arguments = parameters.bind(*args, **kwargs)
        arguments.apply_defaults()
        values = arguments.arguments
        models = values["model"]
        chunks = models if isinstance(models, list) else [models]
        contexts = dict.fromkeys(
            module.context
            for chunk in chunks
            for module in chunk.modules()
            if isinstance(module, FsdpModule)
        )
        if not contexts:
            return schedule(*args, **kwargs)
        config = chunks[0].config
        if getattr(config, "cuda_graph_impl", "none") != "none":
            return schedule(*args, **kwargs)
        if any(context.iteration_order.active for context in contexts):
            if not all(context.iteration_order.active for context in contexts):
                raise RuntimeError("All FSDP contexts must join the same outer iteration scope.")
            return schedule(*args, **kwargs)
        key = (
            schedule.__name__,
            values.get("forward_only", False),
            torch.is_grad_enabled(),
            values["num_microbatches"],
            len(chunks),
            values.get("seq_length"),
            values.get("micro_batch_size"),
            getattr(config, "microbatch_group_size_per_vp_stage", None),
            getattr(config, "overlap_moe_expert_parallel_comm", False),
        )
        with ExitStack() as stack:
            for context in contexts:
                stack.enter_context(iteration(context, key=key))
            return schedule(*args, **kwargs)

    return wrapped
