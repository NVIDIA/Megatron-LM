# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

import pytest

from megatron.core.inference.async_stream import AsyncStream

pytestmark = pytest.mark.asyncio


class TestAsyncStream:

    async def test_generator_yields_items_until_finish(self):
        """The async generator yields each `put` item in FIFO order, then
        terminates cleanly on `finish()` without delivering the sentinel.

        This is the load-bearing contract — every other consumer of AsyncStream
        relies on the generator faithfully replaying everything `put` between
        construction and `finish()`, no more, no less."""
        s = AsyncStream(request_id=1, cancel=lambda: None)
        s.put("a")
        s.put("b")
        # `put` after `finish` is silently dropped — the queue is frozen.
        s.finish()
        s.put("ignored")

        items = [item async for item in s.generator()]
        assert items == ["a", "b"]
        assert s.finished is True

    @pytest.mark.parametrize(
        "exception_arg,expect_raise",
        [
            (ValueError("boom"), True),
            (ValueError, True),  # raising a class also counts
            ("not-an-exception", False),  # non-raisable falls through to STOP_ITERATION
            (42, False),
            (None, False),
        ],
    )
    async def test_finish_routes_exception_or_terminates(self, exception_arg, expect_raise):
        """`finish(x)` raises `x` through the generator iff `x` is a
        BaseException instance or subclass; anything else is treated as a
        plain "no more items" signal."""
        s = AsyncStream(request_id=1, cancel=lambda: None)
        s.finish(exception_arg)

        if expect_raise:
            with pytest.raises((ValueError, BaseException)):
                async for _ in s.generator():
                    pass
        else:
            items = [item async for item in s.generator()]
            assert items == []

    async def test_generator_close_invokes_cancel_callback(self):
        """Closing the generator early (consumer drops it) propagates as
        GeneratorExit inside `generator()`, which invokes the cancel callback
        and re-raises `CancelledError` to the caller."""
        called = []
        s = AsyncStream(request_id=1, cancel=lambda: called.append(True))
        s.put("first")
        gen = s.generator()
        await gen.__anext__()  # consume one item, then close mid-stream.
        with pytest.raises(asyncio.CancelledError):
            await gen.aclose()
        assert called == [True]
