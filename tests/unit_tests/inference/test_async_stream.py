# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

import pytest

from megatron.core.inference.async_stream import STOP_ITERATION, AsyncStream


def _run(coro):
    """Run an async coroutine to completion."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestAsyncStream:

    def test_init_sets_initial_state(self):
        """Newly constructed stream is not finished."""
        loop = asyncio.new_event_loop()
        try:
            s = AsyncStream(request_id=1, cancel=lambda: None, loop=loop)
            assert s.finished is False
            assert s._request_id == 1
        finally:
            loop.close()

    def test_finish_sets_finished_flag(self):
        """finish() flips the finished property to True."""
        loop = asyncio.new_event_loop()
        try:
            s = AsyncStream(request_id=2, cancel=lambda: None, loop=loop)
            s.finish()
            assert s.finished is True
        finally:
            loop.close()

    def test_finish_is_idempotent(self):
        """Calling finish twice does not change state."""
        loop = asyncio.new_event_loop()
        try:
            s = AsyncStream(request_id=2, cancel=lambda: None, loop=loop)
            s.finish()
            s.finish()
            assert s.finished is True
        finally:
            loop.close()

    def test_put_after_finish_is_no_op(self):
        """put() on a finished stream is silently ignored."""

        async def scenario():
            s = AsyncStream(request_id=3, cancel=lambda: None)
            s.finish()
            s.put("ignored")
            # Drain whatever is in the queue and check that "ignored" never appears.
            seen = []
            while not s._queue.empty():
                seen.append(await s._queue.get())
            assert "ignored" not in seen

        _run(scenario())

    def test_generator_yields_items_until_stop(self):
        """The async generator yields each put item then stops on finish()."""

        async def scenario():
            s = AsyncStream(request_id=4, cancel=lambda: None)
            s.put("a")
            s.put("b")
            s.finish()
            results = []
            async for item in s.generator():
                results.append(item)
            return results

        results = _run(scenario())
        assert results == ["a", "b"]

    def test_generator_raises_when_finish_passes_exception(self):
        """finish(exception) routes the exception through the generator."""

        async def scenario():
            s = AsyncStream(request_id=5, cancel=lambda: None)
            s.finish(ValueError("boom"))
            with pytest.raises(ValueError, match="boom"):
                async for _ in s.generator():
                    pass

        _run(scenario())

    def test_finish_with_non_exception_uses_stop_iteration(self):
        """finish(non-exception) falls back to STOP_ITERATION sentinel."""

        async def scenario():
            s = AsyncStream(request_id=6, cancel=lambda: None)
            s.finish("not-an-exception")
            results = []
            async for item in s.generator():
                results.append(item)
            assert results == []

        _run(scenario())

    def test_is_raisable_recognizes_exception_instances(self):
        """_is_raisable returns True for BaseException instances and subclasses."""
        assert AsyncStream._is_raisable(ValueError("x")) is True
        assert AsyncStream._is_raisable(ValueError) is True
        assert AsyncStream._is_raisable(STOP_ITERATION) is True

    def test_is_raisable_rejects_non_exceptions(self):
        """_is_raisable returns False for arbitrary objects and types."""
        assert AsyncStream._is_raisable("string") is False
        assert AsyncStream._is_raisable(42) is False
        assert AsyncStream._is_raisable(int) is False
        assert AsyncStream._is_raisable(None) is False

    def test_generator_calls_cancel_on_generator_exit(self):
        """Closing the generator early invokes the cancel callback."""

        called = []

        def cancel():
            called.append(True)

        async def scenario():
            s = AsyncStream(request_id=7, cancel=cancel)
            s.put("first")
            gen = s.generator()
            # consume one item
            await gen.__anext__()
            # close the generator early; the generator re-raises CancelledError
            # after calling cancel(), which we swallow here.
            try:
                await gen.aclose()
            except asyncio.CancelledError:
                pass

        _run(scenario())
        assert called == [True]
