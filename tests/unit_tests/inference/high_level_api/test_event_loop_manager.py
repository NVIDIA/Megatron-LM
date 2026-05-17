# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import threading

import pytest

from megatron.inference._llm_base import _EventLoopManager


class TestEventLoopManager:
    def test_loop_property_raises_before_start(self):
        mgr = _EventLoopManager()
        with pytest.raises(RuntimeError, match="start"):
            _ = mgr.loop

    def test_start_is_idempotent_and_loop_accessible(self):
        mgr = _EventLoopManager()
        mgr.start()
        try:
            loop1 = mgr.loop
            mgr.start()  # double-start should be a no-op
            loop2 = mgr.loop
            assert loop1 is loop2
            assert loop1.is_running()
        finally:
            mgr.stop()

    def test_submit_run_sync_run_async(self):
        mgr = _EventLoopManager()
        mgr.start()
        try:

            async def coro():
                return 42

            # submit returns a concurrent future
            fut = mgr.submit(coro())
            assert fut.result() == 42

            # run_sync blocks on result
            assert mgr.run_sync(coro()) == 42

            # run_async awaits the future from another loop
            import asyncio

            assert asyncio.run(mgr.run_async(coro())) == 42
        finally:
            mgr.stop()

    def test_stop_is_idempotent_and_joins_thread(self):
        mgr = _EventLoopManager()
        mgr.start()
        thread = mgr._thread
        assert thread is not None
        mgr.stop()
        assert not thread.is_alive()
        # Second stop should be a no-op
        mgr.stop()

    def test_start_propagates_cuda_device_to_daemon_thread(self, monkeypatch):
        """The bug-fix invariant: daemon thread must inherit the spawning
        thread's CUDA device via ``torch.cuda.set_device``."""
        import torch

        recorded = {}
        main_thread_id = threading.get_ident()

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)

        def fake_set_device(device):
            recorded["device"] = device
            recorded["thread_id"] = threading.get_ident()

        monkeypatch.setattr(torch.cuda, "set_device", fake_set_device)

        mgr = _EventLoopManager()
        mgr.start()
        try:
            assert recorded.get("device") == 7
            # Must have run on the daemon thread, not on the spawning thread.
            assert recorded.get("thread_id") != main_thread_id
        finally:
            mgr.stop()

    def test_start_skips_set_device_when_cuda_unavailable(self, monkeypatch):
        import torch

        called = []

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "set_device", lambda d: called.append(d))

        mgr = _EventLoopManager()
        mgr.start()
        try:
            assert called == []
        finally:
            mgr.stop()
