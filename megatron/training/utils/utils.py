# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import logging
import os
import inspect
from typing import Callable
from functools import partial

from megatron.training.state import GlobalState
import torch

from megatron.core._rank_utils import safe_get_rank
from megatron.training.config import ProfilingConfig
from megatron.training.utils.common_utils import print_rank_0

logger = logging.getLogger(__name__)


def start_memory_history_recording(profiling: ProfilingConfig | None) -> None:
    """Enable the CUDA caching allocator trace so memory snapshots contain history.

    ``torch.cuda.memory._snapshot()`` only includes allocation/free events and
    Python stack context after ``_record_memory_history()`` has been enabled.
    Without this call, dumped snapshots contain only the current live
    allocations — no timeline, no call sites.

    Must be invoked before model construction so every tensor allocation is
    captured. Guarded by ``profile_ranks`` so only ranks that will dump a
    snapshot pay the recording overhead.
    """
    if profiling is None or not profiling.record_memory_history:
        return
    if len(profiling.profile_ranks) != 0:
        if safe_get_rank() not in profiling.profile_ranks:
            return

    torch.cuda.memory._record_memory_history(
        True,
        # Retain up to 100k alloc/free events.
        trace_alloc_max_entries=100_000,
        # Record the Python stack at each event — lets memory_viz show call sites.
        trace_alloc_record_context=True,
    )

    def _oom_observer(
        device: int, alloc: int, device_alloc: int, device_free: int
    ) -> None:
        """Dump a snapshot on OOM so we can inspect what was live at the failure."""
        rank = safe_get_rank()
        base, ext = os.path.splitext(profiling.memory_snapshot_path)
        filename = f"{base}_oom_rank-{rank}{ext}"
        torch.cuda.memory._dump_snapshot(filename)
        # logger.info so the message reaches stderr on any profiled rank, not just rank 0.
        logger.info(f"[OOM] rank {rank} saved memory snapshot to {filename}")

    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
    print_rank_0(
        f"Memory history recording enabled (rank {safe_get_rank()}); "
        f"snapshots will be written to '{profiling.memory_snapshot_path}'."
    )


def prepare_forward_step_func(forward_step_func: Callable, state: GlobalState) -> Callable:
    """Convenience function to check and inject GlobalState in one call.

    This combines needs_global_state_injection() and maybe_inject_state() for cleaner code.
    Call this once at the beginning of train() or evaluate() to prevent creating new
    partial objects every iteration.

    Wrapping once is safe since:
    - functools.partial stores a reference to the state object, not a copy
    - When state.train_state.step or other fields change, the partial sees those changes
    - No staleness issues because GlobalState is mutable and passed by reference

    Functor support:
    - Works with both regular functions (def forward_step(...)) and callable classes
    - For functors: inspect.signature() inspects the __call__ method
    - For functors: partial(functor_instance, state) preserves functor's internal state
    - Example: If functor has self.call_count, it still increments correctly

    Args:
        forward_step_func: The original forward step function or functor
        state: The GlobalState object to inject if needed

    Returns:
        The wrapped function (if injection needed) or original function
    """
    needs_injection = needs_global_state_injection(forward_step_func)
    return maybe_inject_state(forward_step_func, state, needs_injection=needs_injection)


def needs_global_state_injection(forward_step_func: Callable) -> bool:
    """Check if a forward step function needs GlobalState injection.

    This function does the signature inspection once to determine if state should be injected.
    It's more efficient than repeated signature inspection in the training loop.

    Detection logic:
    1. First checks for GlobalState type annotation in any parameter
    2. Falls back to checking if first parameter is named 'state' or 'global_state'

    Args:
        forward_step_func: The forward step function to inspect.

    Returns:
        True if GlobalState should be injected, False otherwise.
    """
    signature = inspect.signature(forward_step_func)
    parameters = signature.parameters
    param_names = list(parameters.keys())

    # Check for GlobalState type annotation in any parameter
    for param_name, param in parameters.items():
        if param.annotation != inspect.Parameter.empty:
            # Handle both direct GlobalState and string annotations
            if (
                param.annotation == GlobalState
                or (isinstance(param.annotation, str) and "GlobalState" in param.annotation)
                or (hasattr(param.annotation, "__name__") and param.annotation.__name__ == "GlobalState")
            ):
                # Found GlobalState annotation - needs injection
                return True

    # Fallback: Check if the first parameter is named 'state' or 'global_state'
    return param_names and param_names[0] in ("state", "global_state")


def maybe_inject_state(
    forward_step_func: Callable, state: GlobalState, needs_injection: bool | None = None
) -> Callable:
    """Optionally inject GlobalState into forward_step functions that expect it.

    Determines whether to inject state by inspecting function signature:
    1. First checks for GlobalState type annotation in any parameter
    2. Falls back to checking if first parameter is named 'state'
    3. Otherwise assumes the function doesn't expect state

    Supported signatures:
    - (data_iterator, model) → no injection
    - (data_iterator, model, return_schedule_plan) → no injection
    - (state: GlobalState, data_iterator, model) → inject state
    - (state: GlobalState, data_iterator, model, return_schedule_plan) → inject state
    - (state, data_iterator, model) → inject state (fallback to name-based detection)

    Args:
        forward_step_func: The original forward step function.
        state: The GlobalState object to potentially inject.
        needs_injection: Whether injection is needed (optional, will be inspected if None).
                        Pass this to avoid repeated signature inspection in training loops.

    Returns:
        The original function or a partial function with GlobalState injected.
    """
    if needs_injection is None:
        needs_injection = needs_global_state_injection(forward_step_func)

    if needs_injection:
        return partial(forward_step_func, state)
    else:
        return forward_step_func
