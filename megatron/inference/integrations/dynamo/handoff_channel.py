# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent handoff control requests over one connection to a source coordinator."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import msgpack
import zmq
import zmq.asyncio

from megatron.core.inference.headers import Headers


class HandoffChannelClosed(ConnectionError):
    """The channel closed before a waiting request received its reply."""


def _reply_key(reply: Any) -> tuple | None:
    if not isinstance(reply, list) or not reply:
        return None
    if reply[0] == Headers.CONNECT_ACK.value:
        return (Headers.CONNECT_ACK.value,)
    return tuple(reply[:3]) if len(reply) >= 3 else None


class HandoffChannel:
    """Multiplex CLAIM_KV and RELEASE_KV exchanges over one DEALER socket.

    Requests do not hold a lock across their round trip, so one slow exchange
    no longer delays every other handoff from the same source. Each request
    waits on a future keyed by the acknowledgement it expects, and a single
    reader task resolves them. The coordinator answers one connection in
    arrival order, so requests that share a key are resolved FIFO. A request
    that stops waiting stays queued, so a late reply resolves it rather than a
    newer request with the same key.
    """

    def __init__(self, context: zmq.asyncio.Context, address: str) -> None:
        self.address = address
        self.socket = context.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.SNDHWM, 0)
        self.socket.setsockopt(zmq.RCVHWM, 0)
        self.socket.connect(address)
        self._pending: dict[tuple, deque[asyncio.Future]] = {}
        self._instance_id: str | None = None
        self._handshake: asyncio.Task | None = None
        self._active = 0
        self._retired = False
        self._closed = False
        self._reader = asyncio.create_task(self._read())

    @property
    def closed(self) -> bool:
        """Whether the socket has been closed."""
        return self._closed

    async def instance_id(self, *, refresh: bool = False) -> str:
        """Register with the coordinator and return its instance ID.

        A replacement coordinator does not know this socket until CONNECT is
        resent, so callers refresh when they expect a different instance.
        Concurrent callers share one handshake.
        """
        if self._instance_id is not None and not refresh:
            return self._instance_id
        if self._handshake is None or self._handshake.done():
            self._handshake = asyncio.create_task(self._connect())
            self._handshake.add_done_callback(lambda task: task.cancelled() or task.exception())
        self._active += 1
        try:
            return await asyncio.shield(self._handshake)
        finally:
            self._active -= 1
            self._close_if_drained()

    async def _connect(self) -> str:
        reply = await self._exchange([Headers.CONNECT.value], (Headers.CONNECT_ACK.value,))
        if len(reply) != 2 or not isinstance(reply[1], str) or not reply[1]:
            raise HandoffChannelClosed("Handoff control requires a coordinator instance ID")
        self._instance_id = reply[1]
        return reply[1]

    async def request(self, message: list, reply_key: tuple) -> list:
        """Send one message and wait, without a deadline, for its reply."""
        if self._retired:
            raise HandoffChannelClosed(f"Handoff control channel to {self.address} is retired")
        self._active += 1
        try:
            return await self._exchange(message, reply_key)
        finally:
            self._active -= 1
            self._close_if_drained()

    async def _exchange(self, message: list, reply_key: tuple) -> list:
        future = asyncio.get_running_loop().create_future()
        waiters = self._pending.setdefault(reply_key, deque())
        waiters.append(future)
        try:
            await self.socket.send(msgpack.packb(message, use_bin_type=True))
        except BaseException:
            # Nothing was sent, so no reply will arrive to consume this entry.
            waiters.remove(future)
            if not waiters and self._pending.get(reply_key) is waiters:
                del self._pending[reply_key]
            raise
        try:
            return await future
        finally:
            # Leave an abandoned future queued for its late reply.
            future.cancel()

    def retire(self) -> None:
        """Accept no new requests; close once every waiting caller has finished."""
        self._retired = True
        self._close_if_drained()

    def close(self) -> None:
        """Close the socket now, failing every request still waiting."""
        self._retired = True
        if self._closed:
            return
        self._closed = True
        # The reader's cleanup fails every waiting request, including a handshake.
        self._reader.cancel()
        self.socket.close(linger=0)

    async def aclose(self) -> None:
        """Close and wait for the reader and any handshake to finish."""
        self.close()
        tasks = [self._reader] + ([self._handshake] if self._handshake is not None else [])
        await asyncio.gather(*tasks, return_exceptions=True)

    def _close_if_drained(self) -> None:
        # An unanswered handshake has no caller of its own, so it does not
        # keep a retired channel open.
        if self._retired and self._active == 0:
            self.close()

    async def _read(self) -> None:
        error = HandoffChannelClosed(f"Handoff control channel to {self.address} closed")
        try:
            while True:
                reply = msgpack.unpackb(await self.socket.recv(), raw=False)
                key = _reply_key(reply)
                waiters = self._pending.get(key) if key is not None else None
                if not waiters:
                    continue  # Unsolicited reply: nothing is waiting for this key.
                future = waiters.popleft()
                if not waiters:
                    del self._pending[key]
                if not future.done():
                    future.set_result(reply)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error.__cause__ = exc
            self._retired = True
        finally:
            for waiters in self._pending.values():
                for future in waiters:
                    if not future.done():
                        future.set_exception(error)
            self._pending.clear()
