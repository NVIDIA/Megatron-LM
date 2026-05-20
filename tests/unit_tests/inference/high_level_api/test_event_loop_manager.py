# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.apis._llm_base import _EventLoopManager


class TestEventLoopManager:
    def test_run_sync_raises_when_called_from_background_loop(self):
        """``run_sync`` from a coroutine running on ``self._loop`` would
        deadlock (the same loop would have to dispatch the new coroutine
        while it's blocked waiting on the caller). Calling from any other
        running loop is allowed and just stalls the caller until the
        background loop returns the result."""
        mgr = _EventLoopManager()
        mgr.start()
        try:

            async def inner():
                return 42

            async def deadlock_attempt():
                # Running on mgr._loop; run_sync would schedule on the same
                # loop we're on -> deadlock.
                mgr.run_sync(inner())

            with pytest.raises(RuntimeError, match="background loop"):
                mgr.submit(deadlock_attempt()).result()
        finally:
            mgr.stop()
