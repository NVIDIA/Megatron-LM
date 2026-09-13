# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading

import pytest

from megatron.core.inference.apis._llm_base import _EventLoopManager


class TestEventLoopManager:
    def test_start_propagates_cuda_device_to_daemon_thread(self, monkeypatch):
        """The daemon thread must inherit the spawning thread's CUDA device:
        without this, NCCL ops scheduled on the runtime loop hit GPU 0
        regardless of the rank's actual device (heterogeneous mapping under
        torchrun)."""
        import torch

        recorded = {}
        spawning_thread_id = threading.get_ident()

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)

        def fake_set_device(device):
            recorded["device"] = device
            recorded["thread_id"] = threading.get_ident()

        monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)

        mgr = _EventLoopManager()
        mgr.start()
        try:
            # set_device must be called with the spawning thread's device...
            assert recorded.get("device") == 7
            # ...and from the daemon thread, not the spawning thread.
            assert recorded.get("thread_id") != spawning_thread_id
        finally:
            mgr.stop()

    def test_stop_terminates_thread_and_loop(self):
        """``stop()`` must shut down the loop and join the daemon thread so
        no background work outlives the manager. Second ``stop()`` is a
        no-op (idempotent)."""
        mgr = _EventLoopManager()
        mgr.start()
        thread = mgr._thread
        loop = mgr._loop
        assert thread.is_alive()
        assert loop.is_running()

        mgr.stop()
        assert not thread.is_alive()
        assert not loop.is_running()

        mgr.stop()  # idempotent

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
