# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Asynchronous checkpoint saves, finalized on the training thread once durable on all ranks."""

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Iterable, List, Optional

import torch

from megatron.training.ckpt import native as native_ckpt


class AsyncRequest:
    """Background work of an asynchronous save and the functions that finalize it."""

    def __init__(
        self, futures: Iterable[Future] = (), finalize_fns: Iterable[Callable[[], None]] = ()
    ):
        self.futures = list(futures)
        self.finalize_fns = list(finalize_fns)

    def add_finalize_fn(self, fn: Callable[[], None]) -> None:
        """Appends a function to run once the work of the request completed on all ranks."""
        self.finalize_fns.append(fn)


_requests: List[AsyncRequest] = []
_executor: Optional[ThreadPoolExecutor] = None


def init_persistent_async_worker(rank: int, mp_mode: str = 'spawn'):
    """No-op: saves hand off to the native checkpoint daemons instead of a worker process."""


def run_in_background(fn: Callable, *args) -> Future:
    """Runs `fn(*args)` on the checkpoint background thread."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ckpt-background')
    return _executor.submit(fn, *args)


def schedule_async_save(async_request: AsyncRequest):
    """Queues the async save request for `maybe_finalize_async_save`."""
    _requests.append(async_request)


def maybe_finalize_async_save(blocking: bool = False, terminate: bool = False):
    """Finalizes, in order, the queued saves whose work completed on all ranks. Collective.

    Args:
        blocking (bool, optional): if True, waits for all queued saves. Defaults to False.
        terminate (bool, optional): if True, afterwards stops the checkpoint daemons.
            Defaults to False.
    """
    while _requests:
        done = blocking or all(future.done() for future in _requests[0].futures)
        done = torch.tensor([done], dtype=torch.int32, device='cuda')
        torch.distributed.all_reduce(done, op=torch.distributed.ReduceOp.MIN)
        if not done.item():
            break
        request = _requests.pop(0)
        for future in request.futures:
            future.result()
        for finalize_fn in request.finalize_fns:
            finalize_fn()
    if terminate:
        native_ckpt.spin_down()


def is_empty_async_queue() -> bool:
    """Whether no async save awaits finalization. This result is consistent across ranks."""
    return not _requests


def reset_persistent_async_worker():
    """Drops the queued saves without finalizing them, and the registered checkpoint state."""
    global _executor
    _requests.clear()
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None
    native_ckpt.reset()
