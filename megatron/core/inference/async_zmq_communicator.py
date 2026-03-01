# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import errno
import logging
import socket as _socket_mod
from typing import Optional

import torch.distributed as dist

from megatron.core.inference.headers import Headers
from megatron.core.inference.utils import asyncio_Queue, asyncio_QueueShutDown
from megatron.core.utils import trace_async_exceptions

try:
    import msgpack

    HAVE_MSGPACK = True
except Exception:
    HAVE_MSGPACK = False

try:
    import zmq
    import zmq.asyncio

    HAVE_ZMQ = True
except ImportError:
    HAVE_ZMQ = False

_SOCKET_TYPES = {
    'DEALER': zmq.DEALER,
    'ROUTER': zmq.ROUTER,
    'PUB': zmq.PUB,
    'SUB': zmq.SUB,
    'PUSH': zmq.PUSH,
    'PULL': zmq.PULL,
}

# Sentinel to distinguish "no process group" from process_group=None
_NO_PROCESS_GROUP = object()


class AsyncZmqSendRecv:
    """Shared async ZMQ send/receive helper used by composition.

    Encapsulates the send queue pattern (asyncio Queue + background drain task)
    and multipart frame encoding/decoding with msgpack serialization.
    """

    def __init__(self):
        assert HAVE_ZMQ, "please install the pyzmq library \n pip install pyzmq"
        assert HAVE_MSGPACK, "please install the msgpack library \n pip install msgpack"
        self._send_awaitables = asyncio_Queue()

    def isend(
        self,
        socket,
        header: Headers,
        data=None,
        *,
        identity: Optional[bytes] = None,
        serialize: bool = True,
    ):
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
        self, socket, *, socket_uses_identity: bool = False, deserialize: bool = True
    ) -> tuple[Optional[bytes], Headers, list | bytes | None]:
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
    async def send_task(self):
        """Background task: drain the send queue and await each send."""
        while True:
            try:
                awaitable, identity = await self._send_awaitables.get()
                await awaitable
                self._send_awaitables.task_done()
            except asyncio_QueueShutDown:
                break
            except zmq.error.ZMQError as e:
                if e.errno == zmq.EHOSTUNREACH:
                    logging.warning(
                        "ZMQ send failed, recipient unreachable (identity=%s)", identity
                    )
                    self._send_awaitables.task_done()
                else:
                    raise

    def shutdown(self):
        """Shut down the send queue."""
        self._send_awaitables.shutdown()


class AsyncZmqEndpoint:
    """Base class for async ZMQ endpoints with send queue and startup buffering.

    Subclasses should implement `_recv_task`. Subclasses that need a handshake before `is_running`
    should call `start(loop, set_running=False)` and set `is_running` in their `_recv_task`.

    Multiple sockets can be created by passing in a list to the appropriate parameters.
    If `process_group` is provided, the bind address will be broadcasted from the leader.

    Args:
        socket_type: Socket type name(s) (e.g. "DEALER").
        connect: Address(es) to connect to.
        bind: Whether to bind (per socket).
        bind_port: Preferred port(s); falls back to random on failure.
        identity: ZMQ IDENTITY option(s).
        process_group: process group for address broadcasting.
        is_leader: Whether this rank is the leader of the process group.
    """

    def __init__(
        self,
        socket_type: str | list[str],
        *,
        connect: str | list[str] | None = None,
        bind: bool | list[bool] = False,
        bind_port: int | list[int] | None = None,
        identity: str | list[str] | None = None,
        process_group=_NO_PROCESS_GROUP,
        is_leader: bool = True,
    ):
        ctx = zmq.asyncio.Context.instance()

        has_pg = process_group is not _NO_PROCESS_GROUP

        # Normalize parameters to lists.
        socket_type = [socket_type] if not isinstance(socket_type, list) else socket_type
        n = len(socket_type)
        connect = [connect] * n if not isinstance(connect, list) else connect
        bind = [bind] * n if not isinstance(bind, list) else list(bind)
        bind_port = [bind_port] * n if not isinstance(bind_port, list) else bind_port
        identity = [identity] * n if not isinstance(identity, list) else identity

        # Create sockets.
        self._sockets = []
        self._socket_uses_identity = []

        for i in range(n):
            st = _SOCKET_TYPES[socket_type[i]]
            sock = ctx.socket(st)
            conn = connect[i]

            # Socket options.
            if identity[i] is not None:
                sock.setsockopt(zmq.IDENTITY, identity[i].encode('utf-8'))
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.setsockopt(zmq.RCVHWM, 0)
            if st == zmq.SUB:
                sock.setsockopt_string(zmq.SUBSCRIBE, "")
            if st == zmq.ROUTER:
                sock.setsockopt(zmq.ROUTER_MANDATORY, 1)

            # Bind (leader only when broadcasting).
            broadcast_bind = has_pg and bind[i]
            if bind[i] or bind_port[i] is not None:
                if not broadcast_bind or is_leader:
                    local_ip = _socket_mod.gethostname()
                    port = bind_port[i]
                    if port is not None:
                        try:
                            sock.bind(f"tcp://{local_ip}:{port}")
                        except zmq.error.ZMQError as e:
                            logging.warning(f"Port {port} is in use. Binding to available port.")
                    if sock.getsockopt_string(zmq.LAST_ENDPOINT) == '':
                        sock.bind_to_random_port(f"tcp://{local_ip}")
                    self.address = sock.getsockopt_string(zmq.LAST_ENDPOINT)
                    if broadcast_bind:
                        conn = self.address  # leader broadcasts bound address

            # Broadcast address to all ranks in the process group.
            if has_pg:
                bcast = [conn]
                dist.broadcast_object_list(bcast, group_src=0, group=process_group)
                conn = bcast[0]

            # Connect (skip for leader in bind+broadcast scenario).
            if conn is not None and not (broadcast_bind and is_leader):
                sock.connect(conn)

            self._sockets.append(sock)
            self._socket_uses_identity.append(st == zmq.ROUTER)

        self._ctx = ctx
        self._zmq = AsyncZmqSendRecv()
        self.is_running = asyncio.Event()
        self.is_shutdown = False
        self._startup_sends: list | None = []

    def _isend(
        self,
        header: Headers,
        data=None,
        *,
        identity: Optional[bytes] = None,
        serialize: bool = True,
        sock: int = 0,
    ):
        """Send a message, buffering it if the endpoint is not yet running.

        Args:
            header: The signal header to send.
            data: The data payload to send.
            identity: The ZMQ identity of the recipient (ROUTER sockets only).
            serialize: Whether to serialize data with msgpack.
            sock: Index into ``self._sockets`` (default 0).
        """
        if not self.is_running.is_set():
            self._startup_sends.append((header, data, identity, sock))
            return
        self._zmq.isend(self._sockets[sock], header, data, identity=identity, serialize=serialize)

    async def _irecv(
        self, deserialize: bool = True, *, sock: int = 0
    ) -> tuple[Optional[bytes], Headers, list | bytes | None]:
        """Receive a message from a socket.

        Args:
            deserialize: Whether to deserialize data with msgpack.
            sock: Index into ``self._sockets`` (default 0).

        Returns:
            identity: The source identity, or None for non-ROUTER sockets.
            header: The signal header received.
            data: The data payload received.
        """
        return await self._zmq.irecv(
            self._sockets[sock],
            socket_uses_identity=self._socket_uses_identity[sock],
            deserialize=deserialize,
        )

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None, *, set_running: bool = True):
        """Start background tasks.

        Resolves the event loop, creates the send-queue and startup-buffer
        tasks, and — if the subclass defines ``_recv_task`` — creates the
        receive task as well.

        Args:
            loop: Event loop to schedule on; defaults to the running loop.
            set_running: If True (default), mark the endpoint as running
                immediately.  Subclasses that need a handshake before
                ``is_running`` (e.g. waiting for ACK) should pass
                ``set_running=False`` and set it themselves.
        """
        from megatron.core.utils import get_asyncio_loop

        loop = get_asyncio_loop(loop)

        @trace_async_exceptions
        async def startup_sends_task():
            """Drain the startup send buffer once ``is_running`` is set."""
            await self.is_running.wait()
            for header, data, identity, sock in self._startup_sends:
                self._isend(header, data, identity=identity, sock=sock)
            self._startup_sends = None

        self._startup_sends_task = loop.create_task(startup_sends_task())
        self._send_task = loop.create_task(self._zmq.send_task())
        if hasattr(self, '_recv_task'):
            self.recv_task = loop.create_task(self._recv_task())
        if set_running:
            self.is_running.set()

    async def shutdown(self):
        """Stop background tasks and close all sockets."""
        self.is_shutdown = True
        self._zmq.shutdown()
        for s in self._sockets:
            if not s.closed:
                s.close(linger=0)
        tasks = [getattr(self, a, None) for a in ('recv_task', '_startup_sends_task', '_send_task')]
        tasks = [t for t in tasks if t is not None and not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
