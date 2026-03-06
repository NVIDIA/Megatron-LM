# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional, Tuple

from megatron.core.inference.headers import Headers
from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

try:
    import msgpack

    HAVE_MSGPACK = True
except Exception:
    HAVE_MSGPACK = False


class AsyncZmqSendRecv:
    """Shared async ZMQ send/receive helper used by composition.

    Encapsulates the send queue pattern (asyncio Queue + background drain task)
    and multipart frame encoding/decoding with msgpack serialization.
    """

    def __init__(self):
        assert HAVE_MSGPACK, (
            "please install the messagepack library to use AsyncZmqSendRecv\n"
            "pip install msgpack"
        )
        self._send_awaitables = asyncio_Queue()

    def isend(self, socket, header, data=None, *, identity=None, serialize=True):
        """Enqueue a non-blocking multipart send.

        Args:
            socket: The zmq.asyncio socket to send on.
            header (Headers): The signal header to send.
            data: The data payload to send.
            identity (Optional[bytes]): The ZMQ identity of the recipient (ROUTER sockets).
            serialize (bool): Whether to serialize data with msgpack.
        """
        frames = []
        if identity is not None:
            frames.append(identity)
        frames.append(header.value.to_bytes())
        if data is not None:
            if serialize:
                data = msgpack.packb(data, use_bin_type=True)
            frames.append(data)
        awaitable = socket.send_multipart(frames)
        self._send_awaitables.put_nowait((awaitable, identity))

    async def irecv(
        self, socket, *, socket_uses_identity=False, deserialize=True
    ) -> Tuple[Optional[bytes], Headers, List | bytes | None]:
        """Await a multipart receive and parse frames.

        Args:
            socket: The zmq.asyncio socket to receive from.
            socket_uses_identity (bool): Whether the first frame is a ZMQ identity.
            deserialize (bool): Whether to deserialize data with msgpack.

        Returns:
            identity (Optional[bytes]): The source identity, or None.
            header (Headers): The signal header received.
            data: The data payload received.
        """
        raw = await socket.recv_multipart()
        if socket_uses_identity:
            identity, header, *rest = raw
        else:
            header, *rest = raw
            identity = None

        header = Headers(int.from_bytes(header))
        data = rest[0] if rest else None

        if deserialize:
            message = msgpack.unpackb(data, raw=False) if data is not None else None
        else:
            message = data

        return identity, header, message

    @trace_async_exceptions
    async def send_task(self, on_send_error=None):
        """Background task: drain the send queue and await each send.

        Args:
            on_send_error: Optional callback ``on_send_error(e, identity) -> bool``.
                Called with the exception and the identity of the recipient (or
                None for non-ROUTER sockets).  If it returns True the error is
                considered handled.  If it returns False or raises, the exception
                propagates.
        """
        while True:
            try:
                awaitable, identity = await self._send_awaitables.get()
                await awaitable
                self._send_awaitables.task_done()
            except asyncio_QueueShutDown:
                break
            except Exception as e:
                if on_send_error is not None and on_send_error(e, identity):
                    self._send_awaitables.task_done()
                else:
                    raise

    def shutdown(self):
        """Shut down the send queue."""
        self._send_awaitables.shutdown()
