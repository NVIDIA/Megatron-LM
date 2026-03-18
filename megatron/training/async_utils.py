# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""
import logging
import time

from abc import ABC

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.torch import get_async_strategy
from megatron.training import get_args
from megatron.training.utils import print_rank_0

try:
    from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncRequest as NVRxAsyncRequest
    from nvidia_resiliency_ext.checkpointing.async_ckpt.filesystem_async import _results_queue
except (ImportError, ModuleNotFoundError):
    from megatron.core.dist_checkpointing.strategies.filesystem_async import _results_queue

    NVRxAsyncRequest = ABC

logger = logging.getLogger(__name__)

# Singleton manager of async calls
_async_calls_queue = None


def init_persistent_async_worker(rank: int, mp_mode: str = 'spawn'):
    global _async_calls_queue
    args = get_args()
    async_strategy, async_modules = get_async_strategy(args.async_strategy)
    AsyncCallsQueue = async_modules["AsyncCallsQueue"]
    get_write_results_queue = async_modules["get_write_results_queue"]
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    time_start = time.time()
    if rank == 0:
        print(f"init_persistent_async_worker: {rank}, Starting Async Caller", flush=True)
    _async_calls_queue = AsyncCallsQueue(persistent=True)
    # initialize the persistent caller with QoS priorities from args
    kwargs = {}
    if async_strategy == "mcore":
        # Note: nvidia-resiliency-ext uses is_daemon instead of mp_mode (always spawns)
        kwargs["mp_mode"] = mp_mode
    AsyncCallsQueue.warmup_persistent_caller(
        rank,
        cpu_priority=args.async_ckpt_cpu_priority,
        io_priority=args.async_ckpt_io_priority,
        **kwargs,
    )
    # initialize ckpt write results queue
    get_write_results_queue('fork')
    if rank == 0:
        print(f"init_persistent_async_worker: rank {rank}, Async Caller Started in {time.time() - time_start} seconds", flush=True)


def schedule_async_save(async_request: AsyncRequest | NVRxAsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest | NVRxAsyncRequest): the async save request.
    """
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(blocking: bool = False, terminate=False):
    """Finalizes active async save calls and cleans up deletion processes.

    Args:
        blocking (bool, optional): if True, will wait until all active requests
            are done. Otherwise, finalizes only the async request that already
            finished. Defaults to False.
        terminate (bool, optional): if True, the asynchronous queue will
                be closed as the last action of this function.
    """
    args = get_args()
    if not args.async_save:
        return

    if blocking and not is_empty_async_queue():
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    _async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)

    # Clean up finished deletion processes to prevent zombies
    # Import here to avoid circular dependency
    from .checkpointing import finalize_deletion_processes
    finalize_deletion_processes(blocking=blocking or terminate)

    if terminate:
        _async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Returns:
        bool: True if there is any ongoing async call.
    """
    return _async_calls_queue.get_num_unfinalized_calls() == 0


def reset_persistent_async_worker(async_strategy):
    global _async_calls_queue, _results_queue
    
    if _async_calls_queue is not None:
        _async_calls_queue.close(abort=True)
        del _async_calls_queue
    if _results_queue is not None:
        _results_queue._manager.shutdown()
        del _results_queue
    _results_queue = None
    _async_calls_queue = None
    _, module = get_async_strategy(async_strategy, "CachedMetadataFileSystemReader")
    module.clear_metadata_cache()
