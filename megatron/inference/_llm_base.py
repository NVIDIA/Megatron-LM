# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Internal building blocks for the Megatron inference high-level API.

This module hosts private helpers shared by the future ``MegatronLLM`` and
``MegatronAsyncLLM`` classes. In Stage 1 only ``_EventLoopManager`` is
defined; the coordinator runtime and the base class are added in later
stages.
"""

import asyncio
import concurrent.futures
import threading
import time
from typing import Coroutine


class _EventLoopManager:
    """Per-instance background daemon thread + persistent asyncio event loop.

    Bridges sync and async user-thread callers to coroutines that run on the
    background loop via ``asyncio.run_coroutine_threadsafe``. Mirrors the
    pattern used by NeMo RL's inference worker.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._started: bool = False
        self._stopped: bool = False

    def start(self) -> None:
        """Spawn the daemon thread and start the event loop. Idempotent."""
        if self._started:
            return

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            loop.run_forever()

        self._thread = threading.Thread(target=_run_loop, daemon=True)
        self._thread.start()

        # Wait for the loop to be created and running before returning so
        # callers can use ``submit`` immediately. Mirrors NeMo RL's polling
        # approach.
        while self._loop is None or not self._loop.is_running():
            time.sleep(0.001)

        self._started = True

    def submit(self, coro: Coroutine) -> "concurrent.futures.Future":
        """Schedule ``coro`` on the background loop and return its future.

        The caller decides how to wait on the returned future (e.g.
        ``.result()`` for blocking sync, ``asyncio.wrap_future(...)`` for
        awaiting from another loop).
        """
        if not self._started or self._loop is None:
            raise RuntimeError("_EventLoopManager.start() must be called before submit().")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def run_sync(self, coro: Coroutine):
        """Schedule ``coro`` on the background loop and block on its result."""
        return self.submit(coro).result()

    async def run_async(self, coro: Coroutine):
        """Schedule ``coro`` on the background loop and await it from any loop."""
        return await asyncio.wrap_future(self.submit(coro))

    def stop(self) -> None:
        """Stop the event loop and join the background thread. Idempotent."""
        if not self._started or self._stopped:
            return
        assert self._loop is not None
        assert self._thread is not None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._stopped = True
        self._started = False
