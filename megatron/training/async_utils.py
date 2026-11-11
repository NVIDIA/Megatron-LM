# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""
import logging
import time
from abc import ABC
from typing import TYPE_CHECKING, Any

from megatron.core.dist_checkpointing.strategies.nvrx import make_nvrx_async_request
from megatron.training import get_args
from megatron.training.global_vars import get_run_config
from megatron.training.utils import print_rank_0


def _tag_current_span_call_idx(call_idx):
    """Stamp the async request's call_idx onto whatever OTel span is currently
    active -- the shared join key that ties an async checkpoint's three
    separate-in-time spans together (dispatch's megatron.checkpoint.save subtree,
    the worker's nvrx.checkpoint.* with nvrx.call_idx, and finalize's
    megatron.checkpoint.save.finalize), since they can't share a parent span. No-op
    when telemetry is off (get_current_span returns a non-recording span) or
    nemo-lens is absent. call_idx may be an int (dispatch) or a list (finalize
    may complete more than one call)."""
    if call_idx is None or call_idx == []:
        return
    try:
        from nemo.lens.helpers import safe_set_span_attributes as _set
        from opentelemetry import trace as _ot

        _set(_ot.get_current_span(), {'nvrx.call_idx': call_idx})
    except Exception:  # noqa: BLE001 -- telemetry must never break checkpointing
        pass

if TYPE_CHECKING:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
else:
    AsyncRequest = Any

try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import _results_queue
except (ImportError, ModuleNotFoundError):
    _results_queue = None


logger = logging.getLogger(__name__)

# Singleton manager of async calls
_async_calls_queue = None


def _get_async_calls_queue():
    """Get or lazily initialize the async calls queue."""
    global _async_calls_queue

    if _async_calls_queue is None:
        from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue

        _async_calls_queue = AsyncCallsQueue(
            persistent=get_run_config().checkpoint.use_persistent_ckpt_worker
        )

    return _async_calls_queue


def init_persistent_async_worker(rank: int, mp_mode: str = 'spawn'):
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import (
        get_write_results_queue,
    )

    global _async_calls_queue
    checkpoint = get_run_config().checkpoint
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    time_start = time.time()
    if rank == 0:
        print(f"init_persistent_async_worker: {rank}, Starting Async Caller", flush=True)
    _async_calls_queue = AsyncCallsQueue(
        persistent=True, cpu_shm_mode=checkpoint.async_ckpt_use_cpu_shm
    )
    # initialize the persistent caller with QoS priorities from args
    AsyncCallsQueue.warmup_persistent_caller(
        rank,
        cpu_priority=checkpoint.async_ckpt_cpu_priority,
        io_priority=checkpoint.async_ckpt_io_priority,
        cpu_shm_mode=checkpoint.async_ckpt_use_cpu_shm,
    )
    # initialize ckpt write results queue
    get_write_results_queue(mp_mode="fork")
    if rank == 0:
        print(f"init_persistent_async_worker: rank {rank}, Async Caller Started in {time.time() - time_start} seconds", flush=True)


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    call_idx = _get_async_calls_queue().schedule_async_request(async_request)
    _tag_current_span_call_idx(call_idx)


def maybe_finalize_async_save(blocking: bool = False, terminate=False):
    """Finalizes active async save calls and cleans up deletion processes.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    if not get_run_config().checkpoint.async_save:
        return

    if blocking and not is_empty_async_queue():
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    async_calls_queue = _async_calls_queue
    if async_calls_queue is not None:
        finalized = async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)
        # Tag the active finalize span (megatron.checkpoint.save.finalize) with the
        # call_idx(es) that actually completed this call, so it groups with the
        # matching dispatch + worker spans.
        _tag_current_span_call_idx(finalized)

    if terminate and async_calls_queue is not None:
        async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Returns:
        bool: True if there is any ongoing async call.
    """
    return _async_calls_queue is None or _async_calls_queue.get_num_unfinalized_calls() == 0


def reset_persistent_async_worker():
    from nvidia_resiliency_ext.checkpointing.async_ckpt.cached_metadata_filesystem_reader import (  # pylint: disable=line-too-long
        CachedMetadataFileSystemReader,
    )

    global _async_calls_queue, _results_queue

    if _async_calls_queue is not None:
        _async_calls_queue.close(abort=True)
        del _async_calls_queue
    if _results_queue is not None:
        _results_queue._manager.shutdown()
        del _results_queue
    _results_queue = None
    _async_calls_queue = None
    CachedMetadataFileSystemReader.clear_metadata_cache()


def get_save_and_finalize_callbacks(writer, pending_save) -> AsyncRequest:
    """Creates an async save request for fsdp_dtensor & torch_dcp with a finalize function."""
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest
    from nvidia_resiliency_ext.checkpointing.async_ckpt.state_dict_saver import (
        save_state_dict_async_finalize,  # pylint: disable=line-too-long
    )

    save_fn, preload_fn, save_args = writer.get_save_function_and_args()

    def finalize_fn():
        """Finalizes async checkpointing and synchronizes processes."""
        save_state_dict_async_finalize(pending_save)

    return make_nvrx_async_request(
        AsyncRequest,
        save_fn,
        save_args,
        [finalize_fn],
        async_fn_kwargs={},
        preload_fn=preload_fn,
    )
