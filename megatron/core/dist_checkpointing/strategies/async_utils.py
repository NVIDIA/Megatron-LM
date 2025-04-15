# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""
This module provides an async utilities which allow to start
a checkpoint save process in the background.
"""
import gc
import logging
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from queue import Empty
from time import sleep, time
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import torch
from torch import multiprocessing as mp

from ..utils import debug_time

logger = logging.getLogger(__name__)


@contextmanager
def _disable_gc():
    """Temporarily disables GC."""
    gc_enabled = gc.isenabled()
    try:
        if gc_enabled:
            gc.disable()
        yield
    finally:
        if gc_enabled:
            gc.enable()


class AsyncRequest(NamedTuple):
    """Represents an async request that needs to be scheduled for execution.

    Args:
        async_fn (Callable, optional): async function to call. None represents noop.
        async_fn_args (Tuple): args to pass to `async_fn`.
        finalize_fns (List[Callable]): list of functions to call to finalize the request.
            These functions will be called synchronously after `async_fn` is done
            *on all ranks*.
        async_fn_kwargs (Tuple): kwargs to pass to `async_fn`.
        preload_fn (Callable): preload function to stage tensors from GPU to Host.
            This should be self-contained with a proper list of arguments with  `partial`.
        is_frozen (Bool): a flag to indicate this async request can be modified or not.
        call_idx (int): index variable used to order async requests for synchronization
                        in preloading and writing tensors on the async caller

    """

    async_fn: Optional[Callable]
    async_fn_args: Tuple
    finalize_fns: List[Callable]
    async_fn_kwargs: Dict = {}
    preload_fn: Callable = None
    is_frozen: bool = False
    call_idx: int = 0

    def add_finalize_fn(self, fn: Callable) -> None:
        """Adds a new finalize function to the request.

        Args:
            fn (Callable): function to add to the async request. This function
                will be called *after* existing finalization functions.

        Returns:
            None
        """
        if self.is_frozen:
            raise RuntimeError('Cannot add finalization functions to a frozen AsyncRequest')
        self.finalize_fns.append(fn)

    def execute_sync(self) -> None:
        """Helper to synchronously execute the request.

        This logic is equivalent to what should happen in case of the async call.
        """
        if self.async_fn is not None:
            self.async_fn(*self.async_fn_args)
        torch.distributed.barrier()
        for finalize_fn in self.finalize_fns:
            finalize_fn()

    def freeze(self) -> 'AsyncRequest':
        """Freezes the async request, disallowing adding new finalization functions.

        Returns:
            AsyncRequest: new async request with all same fields except for the
                `is_frozen` flag.
        """
        return self._replace(is_frozen=True)


class AsyncCaller(ABC):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    @abstractmethod
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Schedule `async_req` with some process forking or reusing
           persistent worker

        This method must be called on all ranks.

        Args:
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       start async process
        """
        raise NotImplementedError("This should be implemented")

    @abstractmethod
    def is_current_async_call_done(self, blocking: bool, no_dist: bool) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.

        """
        raise NotImplementedError("This should be implemented")

    def sync_all_async_calls(self, is_alive: int) -> bool:
        """Check if all ranks have completed async checkpoint writing

        Args:
            is_alive (bool): if True, the current async request is not completed

        Returns:
            bool: True if all ranks are done, False if at least one rank is still active.

        """
        ten = torch.tensor([is_alive], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(ten)
        return ten[0] == 0

    @abstractmethod
    def close(self):
        """Terminate the async caller at exit of an application or some termination conditions"""
        logger.info(f"AsyncCaller: {torch.distributed.get_rank()}, Destroying Async Caller")

    def __del__(self):
        self.close()


class TemporalAsyncCaller(AsyncCaller):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        self.process: Optional[mp.Process] = None
        self.start_time: Optional[float] = None

    @_disable_gc()
    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Spawn a process with `async_fn` as the target.

        This method must be called on all ranks.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       start async process
        """
        if async_req.async_fn is None:
            return  # nothing to do

        async_fn_args = list(async_req.async_fn_args)
        if async_req.preload_fn:
            # If there's a preload_fn in `async_req`, we call this func
            # to do the defined action in `async_req.preload_fn` to
            # stage GPU tensors to its defined destination
            async_fn_args[1] = async_req.preload_fn()

        rank = torch.distributed.get_rank()
        start_sync = time()
        torch.cuda.synchronize()
        end_sync = time()
        logger.debug(f"rank: {rank}, takes {end_sync - start_sync} to finish D2H ")

        ctx = mp.get_context('fork')
        self.start_time = time()
        self.process = ctx.Process(
            target=async_req.async_fn, args=async_fn_args, kwargs=async_req.async_fn_kwargs
        )
        self.process.start()
        init_time = time()
        logger.debug(f"rank: {rank}, takes {init_time - self.start_time} to schedule async ckpt ")

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.
        """
        # The following takes the same overhead
        # as torch.distributed.barrier (single integer all-reduce)
        is_alive = int(self.process.is_alive()) if self.process is not None else 0
        is_done = not is_alive if no_dist else self.sync_all_async_calls(is_alive)

        if not is_done and blocking:
            self.close()
            is_done = True
        return is_done

    def close(self):
        if self.process:
            logger.debug(f"rank: {torch.distributed.get_rank()}, joining self.process")
            self.process.join()
            self.process = None
            logger.debug(
                "TemporalAsyncCaller: Async process join finished "
                f"after {time() - self.start_time:.2f}s from forking"
            )
            self.start_time = None


class PersistentAsyncCaller(AsyncCaller):
    """Wrapper around mp.Process that ensures correct semantic of distributed finalization.

    Starts process asynchronously and allows checking if all processes on all ranks are done.
    """

    def __init__(self):
        self.process: mp.Process = None
        self.start_time: Optional[float] = None
        ctx = mp.get_context('spawn')
        # main queue to deliver `AsyncRequest` from host to the ckpt worker
        self.queue: mp.JoinableQueue = ctx.JoinableQueue()
        # Queue used to synchronize for the completion of preloading tensors to host
        # between a trainer and ckpt worker
        self.preload_q: mp.JoinableQueue = ctx.JoinableQueue()
        # Queue used to inform trainer when the saving is completed
        self.comp_q: mp.Queue = ctx.Queue()
        self.cur_item: int = None
        self.cur_idx: int = -1

    def schedule_async_call(self, async_req: AsyncRequest) -> None:
        """Put `AsyncRequest` to the Persistent Async Caller

        This method must be called on all ranks.

        Args:
            async_fn (Callable, optional): async function to call. If None,
                no process will be started.
            async_req (AsyncRequest): `AsyncRequest` object containing to
                                       schedule a checkpointing request
        """
        if async_req.async_fn is None:
            return  # nothing to do

        start_sync = end_sync = None

        self.start_time = time()
        if self.process is None:
            ctx = mp.get_context('spawn')
            logger.info(
                f"PersistentAsyncCaller: {torch.distributed.get_rank()}, Starting Async Caller"
            )
            self.process: mp.Process = ctx.Process(
                target=PersistentAsyncCaller.async_loop,
                args=(
                    torch.distributed.get_rank(),
                    self.queue,
                    self.preload_q,
                    self.comp_q,
                    logger.getEffectiveLevel(),
                ),
            )
            self.process.start()
            logger.info(
                f"PersistentAsyncCaller: {torch.distributed.get_rank()}, Started Async Caller"
            )

        if async_req.preload_fn:
            self.preload_q.put(async_req.call_idx)
        self.queue.put(async_req)
        logger.debug(f"rank: {torch.distributed.get_rank()}, put {async_req.call_idx}")

        if async_req.preload_fn:
            start_sync = time()
            # Synchronize for pre-staging tensors
            self.preload_q.join()
            end_sync = time()
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, "
                f"takes {end_sync - start_sync} to finish D2H "
            )

        init_time = time()
        logger.debug(
            f"rank: {torch.distributed.get_rank()}, takes {init_time - self.start_time} "
            "to schedule async ckpt "
        )

    def is_current_async_call_done(self, blocking: bool = False, no_dist: bool = False) -> bool:
        """Check if async save is finished on all ranks.

        For semantic correctness, requires rank synchronization in each check.
        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until the call is done
                on all ranks. Otherwise, returns immediately if at least one rank
                is still active. Defaults to False.
            no_dist (bool, Optional): if True, training ranks simply check its
                asynchronous checkpoint writer without synchronization.

        Returns:
            bool: True if all ranks are done (immediately of after active wait
                if `blocking` is True), False if at least one rank is still active.
        """

        is_alive: bool = False

        if self.process:
            while self.cur_item is None:
                try:
                    # Retrieve comp call_idx without waiting
                    self.cur_item = self.comp_q.get_nowait()
                except Empty:
                    # This method is called after any `AsyncRequest` is pushed to the main loop
                    # So, the background writing is still active
                    # before the worker put call_idx to `comp_q`
                    if not blocking:
                        is_alive = True
                        break
                    sleep(0.1)

        if self.cur_item is not None:
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, item: {self.cur_item}"
                f" is completed, {is_alive}"
            )

        is_done = not is_alive if no_dist else self.sync_all_async_calls(is_alive)
        # This is set to False when blocking == False so this routine is called again
        # to simply call `sync_all_async_calls` to check if other ranks complete the writing
        if is_done:
            # The current request is completed globally. Reset the current item for polling.
            logger.debug(
                f"rank: {torch.distributed.get_rank()}, item: {self.cur_item}"
                f" is completed globally, {is_done}"
            )
            self.cur_item = None

        return is_done

    def close(self):
        logger.info(
            f"PersistentAsyncCaller: {torch.distributed.get_rank()}, Destroying Async Caller"
        )
        if self.process:
            self.queue.put('DONE')
            self.queue.join()
            self.process.join()
            self.process = None

    @staticmethod
    @_disable_gc()
    def async_loop(
        rank: int,
        queue: mp.JoinableQueue,
        preload_q: mp.JoinableQueue,
        comp_q: mp.Queue,
        log_level: int = logging.INFO,
    ):
        """Main function for the persistent checkpoint worker

        The persisent worker is created once and terminated at exit or
        when application calls `close()` explictily

        This routine receives `AsyncRequest` and does `preload_fn` first and
        put the integer value in `preload_q` to inform the trainer to proceed.
        When the `async_fn` from the request` is completed (background saving is done),
        it puts a integer value to `comp_q` to notify the trainer the completion.

        Args:
            rank (int): the rank of the trainer where the persistent worker is created.
            queue (mp.JoinableQueue): the main queue used to receive `AsyncRequest
                                      from the training rank
            preload_q (mp.JoinableQueue): a queue to inform trainer that preloading of tensors
                                          from GPU to Host or dedicated location is completed
            comp_q (mp.Queue): a queue to inform the training rank the completion of scheduled
                               async checkpoint request
            log_level (int, Optional): an integer to set log-level in this spawned process
                                       to get aligned with the training rank's logging level

        """
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        logger.info(f"PersistentAsyncCaller: persistent ckpt worker for {rank} has started")
        while True:
            item = queue.get()
            if isinstance(item, str) and item == 'DONE':
                queue.task_done()
                break
            elif isinstance(item, AsyncRequest):
                async_fn_args = list(item.async_fn_args)
                if item.preload_fn:
                    call_idx = preload_q.get()
                    # the 2nd arg is state dict
                    async_fn_args[1] = item.preload_fn()
                    logger.debug(f"{rank} has completed D2H of {call_idx}")
                    preload_q.task_done()
                item.async_fn(*async_fn_args, **item.async_fn_kwargs)
                logger.debug(f"{rank} has completed saving {item.call_idx}")
                comp_q.put(item.call_idx)
                queue.task_done()

        logger.info(f"PersistentAsyncCaller: persistent ckpt worker for {rank}  has terminated")


class _ActiveAsyncRequest(NamedTuple):
    """Helper to represent an active async call.

    Args:
        idx (int): index of the call (starting from 0)
        async_caller (DistributedAsyncCaller): async caller instance that represents
            the async process handling the async request
        async_request (AsyncRequest):  async request that is being called
    """

    idx: int
    async_caller: AsyncCaller
    async_request: AsyncRequest


class AsyncCallsQueue:
    """Manages a queue of async calls.

    Allows adding a new async call with `schedule_async_request` and finalizing
    active calls with `maybe_finalize_async_calls`.
    """

    def __init__(self, persistent: bool = False):
        self.async_calls: deque[_ActiveAsyncRequest] = deque([])
        self.call_idx: int = -1
        self.persistent: bool = persistent
        self.persistent_caller: AsyncCaller = None

    def _get_async_caller(self):
        if not self.persistent:
            return TemporalAsyncCaller()
        if self.persistent_caller is None:
            self.persistent_caller = PersistentAsyncCaller()
        return self.persistent_caller

    def schedule_async_request(self, async_request: AsyncRequest) -> int:
        """Start a new async call and add it to a queue of active async calls.

        This method must be called on all ranks.

        Args:
            async_request (AsyncRequest): async request to start.

        Returns:
            int: index of the async call that was started.
                This can help the user keep track of the async calls.
        """
        self.call_idx += 1
        async_caller = self._get_async_caller()
        # Backward compatibility for local checkpointing built with the old AsyncRequest
        if len(async_request._fields) != len(AsyncRequest._fields):
            async_request = AsyncRequest(**async_request._asdict())

        async_request = async_request._replace(call_idx=self.call_idx)
        finalize_fns = async_request.finalize_fns
        async_request = async_request._replace(finalize_fns=None)
        async_request = async_request.freeze()
        async_caller.schedule_async_call(async_request)
        self.async_calls.append(_ActiveAsyncRequest(self.call_idx, async_caller, finalize_fns))
        return self.call_idx

    def maybe_finalize_async_calls(self, blocking=False, no_dist=False) -> List[int]:
        """Finalizes all available calls.

        This method must be called on all ranks.

        Args:
            blocking (bool, optional): if True, will wait until all active requests
                are done. Otherwise, finalizes only the async request that already
                finished. Defaults to False.
        Returns:
            List[int]: list of indices (as returned by `schedule_async_request`)
                of async calls that have been successfully finalized.
        """
        call_idx_finalized = []
        while self.async_calls:
            next_async_done = self.async_calls[0].async_caller.is_current_async_call_done(
                blocking, no_dist
            )
            if not next_async_done:
                break
            with debug_time("finalize", logger):
                call_idx, _, finalize_fns = self.async_calls.popleft()
                ten = torch.tensor([call_idx], dtype=torch.int, device=torch.cuda.current_device())
                torch.distributed.all_reduce(ten, op=torch.distributed.ReduceOp.MAX)
                assert ten.item() == call_idx, 'Unmatched async calls. '
                'That probably means not all ranks are participating in async finalization'
                for finalize_fn in finalize_fns:
                    finalize_fn()
                call_idx_finalized.append(call_idx)
        return call_idx_finalized

    def get_num_unfinalized_calls(self):
        """Get the number of active async calls."""
        return len(self.async_calls)

    def close(self):
        """Finalize all calls upon closing."""
        self.maybe_finalize_async_calls(blocking=True)
        if self.persistent and self.persistent_caller:
            self.persistent_caller.close()
