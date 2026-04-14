# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import collections
import contextlib
import ctypes
import sys
from typing import Optional

import torch

_libcudart: Optional[ctypes.CDLL] = None
_CUDA_HOST_FN_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def _get_cudart() -> ctypes.CDLL:
    """Lazily load and configure the CUDA runtime library."""
    global _libcudart
    if _libcudart is None:
        cuda_major = torch.version.cuda.split('.')[0]
        _libcudart = ctypes.CDLL(f"libcudart.so.{cuda_major}")
        _libcudart.cudaLaunchHostFunc.restype = ctypes.c_int
        _libcudart.cudaLaunchHostFunc.argtypes = [
            ctypes.c_void_p,  # cudaStream_t
            _CUDA_HOST_FN_T,  # cudaHostFn_t
            ctypes.c_void_p,  # void* userData
        ]
    return _libcudart


class GpuFuture:
    """Awaitable that resolves when all preceding work on a CUDA stream completes.

    If the event loop is an `ExclusiveTaskEventLoop` and exclusive mode is active,
    awaiting this future temporarily releases exclusivity so other tasks can run while the GPU is
    busy, then re-acquires it on completion. Pass `yield_exclusive=False` to suppress this behavior.

    Usage:
        gpu_done = GpuFuture(loop)
        launch_gpu_work(...)
        gpu_done.record()
        launch_more_gpu_work(...)
        await gpu_done          # releases exclusivity while waiting, if active
    """

    # Prevent garbage-collection of live ctypes callbacks.
    # Keyed by id() because ctypes function pointers are not hashable.
    _prevent_gc: dict = {}

    def __init__(self, loop: asyncio.AbstractEventLoop, yield_exclusive: bool = True):
        self._loop = loop
        self._future: asyncio.Future = loop.create_future()
        self._yield_exclusive = yield_exclusive

    def record(self, stream: Optional[torch.cuda.Stream] = None) -> None:
        """Enqueue a host callback that resolves this future.

        Args:
            stream: CUDA stream to attach to. Defaults to the current stream.
        """
        if stream is None:
            stream = torch.cuda.current_stream()

        # This closure prevents the ctypes wrapper from being collected
        # while the callback is in the stream.
        prevent_gc_ref: Optional[object] = None

        def _host_fn(_user_data: ctypes.c_void_p) -> None:
            # Runs on CUDA's internal callback thread.
            # MUST NOT call any CUDA API.
            try:
                self._loop.call_soon_threadsafe(self._future.set_result, None)
            except RuntimeError:
                # Event loop closed; nothing to do.
                pass
            GpuFuture._prevent_gc.pop(id(prevent_gc_ref), None)

        c_fn = _CUDA_HOST_FN_T(_host_fn)
        prevent_gc_ref = c_fn
        GpuFuture._prevent_gc[id(c_fn)] = c_fn

        err = _get_cudart().cudaLaunchHostFunc(
            ctypes.c_void_p(stream.cuda_stream), c_fn, ctypes.c_void_p(0)
        )
        if err != 0:
            # Allow the callback to be garbage-collected on failure.
            GpuFuture._prevent_gc.pop(id(c_fn), None)
            raise RuntimeError(f"cudaLaunchHostFunc failed with CUDA error {err}")

    def __await__(self):
        ready = getattr(self._loop, '_ready', None)
        exclusive_task = getattr(ready, '_exclusive_task', None) if ready else None
        should_yield = self._yield_exclusive and exclusive_task is not None

        if should_yield:
            ready.clear_exclusive()
        result = yield from self._future.__await__()
        if should_yield:
            ready.set_exclusive(exclusive_task)
        return result


def _verify_task_callback_valid(loop: asyncio.AbstractEventLoop) -> None:
    """Verify that Task step callbacks expose __self__ on this Python runtime.

    This currently works on all Python versions, but is not guaranteed to be stable.
    Internal API changes can break the mechanism at any point.

    This function creates a throwaway task, inspects its callback in `_ready`,
    and raises `RuntimeError` if the internal API pattern no longer holds.
    """

    async def _canary():
        pass

    task = loop.create_task(_canary())

    # The task's __step should now be in _ready.  Inspect without running it.
    found = False
    for handle in loop._ready:
        cb = handle._callback
        owner = getattr(cb, '__self__', None)
        if owner is task:
            found = True
            break

    # Cancel the canary — it will be cleaned up on the next loop iteration.
    task.cancel()

    if not found:
        raise RuntimeError(
            f"ExclusiveTaskEventLoop cannot identify Task ownership of callbacks "
            f"on this Python runtime (Python {sys.version}).  "
            f"Task step callbacks do not expose __self__.  "
            f"Exclusive-task scheduling requires this for correctness."
        )


class _PriorityReadyQueue:  # pylint: disable=missing-function-docstring
    """Drop-in replacement for the event loop's `_ready` deque.

    When an exclusive task is set, callbacks are routed into two categories at `append` time:

    - Allowed: the exclusive task's own callbacks. These enter the live `_queue`.
    - Deferred: every other task's callbacks.
      These accumulate in `_deferred` and are drained back into `_queue` when exclusive mode ends.

    The event loop's `run_once` only ever sees `_queue` via the standard deque interface,
    so no event-loop internals need to be patched.
    """

    __slots__ = ('_queue', '_deferred', '_exclusive_task')

    def __init__(self) -> None:
        self._queue: collections.deque = collections.deque()
        self._deferred: collections.deque = collections.deque()
        self._exclusive_task: Optional[asyncio.Task] = None

    def set_exclusive(self, task: asyncio.Task) -> None:
        """Activate exclusive mode for `task`."""
        assert self._exclusive_task is None, "Exclusive task already set"
        self._exclusive_task = task

    def clear_exclusive(self) -> None:
        """Deactivate exclusive mode and drain deferred callbacks."""
        self._exclusive_task = None
        if self._deferred:
            self._queue.extend(self._deferred)
            self._deferred.clear()

    def _is_allowed(self, handle) -> bool:
        if self._exclusive_task is None:
            return True

        cb = handle._callback
        owner = getattr(cb, '__self__', None)

        # The exclusive task's own __step callback.
        if owner is self._exclusive_task:
            return True

        # Non-Task callbacks: Future resolution, I/O dispatch, signal handlers, call_soon_threadsafe
        # Infrastructure that the exclusive task depends on to function.
        if not isinstance(owner, asyncio.Task):
            return True

        # A different task's __step; defer it.
        return False

    def append(self, handle) -> None:
        if self._is_allowed(handle):
            self._queue.append(handle)
        else:
            self._deferred.append(handle)

    def appendleft(self, handle) -> None:
        if self._is_allowed(handle):
            self._queue.appendleft(handle)
        else:
            self._deferred.appendleft(handle)

    def popleft(self):
        return self._queue.popleft()

    def extend(self, iterable) -> None:
        for item in iterable:
            self.append(item)

    def remove(self, item) -> None:
        try:
            self._queue.remove(item)
        except ValueError:
            self._deferred.remove(item)

    def clear(self) -> None:
        self._queue.clear()
        self._deferred.clear()

    def __len__(self) -> int:
        return len(self._queue)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def __iter__(self):
        return iter(self._queue)

    def __contains__(self, item) -> bool:
        return item in self._queue or item in self._deferred


class ExclusiveTaskEventLoop(asyncio.SelectorEventLoop):
    """Event loop with exclusive-task support.

    When `set_exclusive_task` is called, only the designated tasks' callbacks remain visible.
    This gives the exclusive task immediate, contention-free access to the event loop.

    Usage:
        loop = ExclusiveTaskEventLoop()
        asyncio.set_event_loop(loop)
        [...]
        # Inside the critical coroutine (i.e. GPU forward pass loop):
        async with loop.exclusive():
            while True:
                gpu_done = GpuFuture(loop)
                launch_gpu_work_1()         # exclusive — no other task can interleave
                gpu_done.record()
                launch_gpu_work_2()         # still exclusive
                await gpu_done              # yields exclusivity while GPU is busy
                                            # re-acquires when GPU work 1 completes
        # Exclusive mode is always released on exit, even on exception.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ready = _PriorityReadyQueue()

        # Verify that Task callbacks expose __self__ before we rely on it in the ready queue.
        _verify_task_callback_valid(self)

    @contextlib.asynccontextmanager
    async def exclusive(self, task: Optional[asyncio.Task] = None):
        """Async context manager for exclusive-task mode."""
        if task is None:
            task = asyncio.current_task()
        self.set_exclusive_task(task)
        try:
            yield
        finally:
            self.set_exclusive_task(None)

    def set_exclusive_task(self, task: Optional[asyncio.Task]) -> None:
        """Set or clear the exclusive task.

        Prefer the :meth:`exclusive` context manager over calling this directly,
        to ensure exclusive mode is always released.

        Args:
            task: The `asyncio.Task` that should have exclusive access,
                or `None` to end the exclusive section and drain deferred callbacks.
        """
        if task is None:
            self._ready.clear_exclusive()
        else:
            self._ready.set_exclusive(task)
