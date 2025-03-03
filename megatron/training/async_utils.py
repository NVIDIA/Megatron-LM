# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
This module provides a singleton instance of AsyncCallsQueue which manages
the async checkpoint save calls.
"""
import logging

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue, AsyncRequest
from megatron.training import get_args
from megatron.training.utils import print_rank_0
from megatron.core import mpu

logger = logging.getLogger(__name__)

# Singleton manager of async calls
_async_calls_queue = None


def init_persistent_async_worker():
    global _async_calls_queue
    # Recreate the async_calls_queue for persistent worker
    # This duplicate step is for backward compatiblity
    _async_calls_queue = AsyncCallsQueue(persistent=True, process_group=mpu.get_default_process_group())


def schedule_async_save(async_request: AsyncRequest):
    """Schedule the async save request.

    Args:
        async_request (AsyncRequest): the async save request.
    """
    global _async_calls_queue
    if _async_calls_queue is None:
        _async_calls_queue = AsyncCallsQueue(process_group=mpu.get_default_process_group())
    _async_calls_queue.schedule_async_request(async_request)


def maybe_finalize_async_save(blocking: bool = False, terminate=False):
    """Finalizes active async save calls.

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

    global _async_calls_queue
    if _async_calls_queue is None:
        _async_calls_queue = AsyncCallsQueue(process_group=mpu.get_default_process_group())

    if blocking and not is_empty_async_queue():
        print_rank_0('Unfinalized async checkpoint saves. Finalizing them synchronously now.')

    _async_calls_queue.maybe_finalize_async_calls(blocking, no_dist=False)

    if terminate:
        _async_calls_queue.close()


def is_empty_async_queue() -> bool:
    """Check if async calls queue is empty. This result is consistent across ranks.

    Returns:
        bool: True if there is any ongoing async call.
    """
    global _async_calls_queue
    return _async_calls_queue is None or _async_calls_queue.get_num_unfinalized_calls() == 0
