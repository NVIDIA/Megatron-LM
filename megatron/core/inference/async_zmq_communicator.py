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

# Sentinel to distinguish "no process group" from process_group=None
_NO_PROCESS_GROUP = object()


class AsyncZmqEndpoint:
    """Base class for async ZMQ endpoints with send queue and startup buffering.

    Handles socket creation, multipart frame encoding/decoding with msgpack,
    a background send queue (one send at a time per zmq.asyncio socket), and
    startup buffering (messages sent before `is_running` are queued until the event fires).

    Subclasses should implement `_recv_task`. Subclasses that need a handshake before `is_running`
    should call `start(loop, set_running=False)` and set `is_running` in their `_recv_task`.

    Multiple sockets can be created by passing a list to `socket_type` (and optionally to the other
    parameters). If `process_group` is provided, bind addresses are broadcast from the leader.

    Context ownership: pass `context` to share a context with other endpoints (the caller owns
    lifecycle and is responsible for `term()`). If `context` is None, a private context is
    created and terminated during `shutdown()`. Never use `zmq.asyncio.Context.instance()` here —
    the process-wide singleton is unsafe because terminating it bombs every unrelated socket.
    """

    if HAVE_ZMQ:
        _SOCKET_TYPES = {
            'DEALER': zmq.DEALER,
            'ROUTER': zmq.ROUTER,
            'PUB': zmq.PUB,
            'SUB': zmq.SUB,
            'PUSH': zmq.PUSH,
            'PULL': zmq.PULL,
        }

    def __init__(
        self,
        socket_type: "str | list[str]",
        *,
        context: "zmq.asyncio.Context | None" = None,
        connect: "str | list[str] | None" = None,
        bind: "bool | list[bool]" = False,
        bind_port: "int | list[int] | None" = None,
        identity: "str | list[str] | None" = None,
        process_group=_NO_PROCESS_GROUP,
        is_leader: bool = True,
    ):
        assert HAVE_ZMQ, "please install the pyzmq library \n pip install pyzmq"
        assert HAVE_MSGPACK, "please install the msgpack library \n pip install msgpack"

        if context is None:
            self._ctx = zmq.asyncio.Context()
            self._owns_ctx = True
        else:
            self._ctx = context
            self._owns_ctx = False
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
            st = self._SOCKET_TYPES[socket_type[i]]
            sock = self._ctx.socket(st)
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
                            if e.errno == errno.EADDRINUSE:
                                logging.warning(
                                    f"Port {port} is already in use. "
                                    "Binding to a random available port instead."
                                )
                            else:
                                logging.warning(
                                    f"Unknown error when binding to port {port}: {e}. "
                                    "Attempting to bind to a random available port instead."
                                )
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

        self._send_awaitables = asyncio_Queue()
        self.is_running = asyncio.Event()
        self.is_shutdown = False
        self._startup_sends: "list | None" = []

    def _isend(
        self,
        header: Headers,
        data=None,
        *,
        identity: Optional[bytes] = None,
        serialize: bool = True,
        sock: int = 0,
    ):
        """Send a message, buffering it if the endpoint is not yet running."""
        if not self.is_running.is_set():
            self._startup_sends.append((header, data, identity, serialize, sock))
            return
        frames = []
        if identity is not None:
            frames.append(identity)
        frames.append(header.value.to_bytes())
        if data is not None:
            if serialize:
                data = msgpack.packb(data, use_bin_type=True)
            frames.append(data)
        awaitable = self._sockets[sock].send_multipart(frames)
        self._send_awaitables.put_nowait((awaitable, identity))

    async def _irecv(
        self, deserialize: bool = True, *, sock: int = 0
    ) -> tuple[Optional[bytes], Headers, list | bytes | None]:
        """Receive and decode a multipart message from a socket."""
        raw = await self._sockets[sock].recv_multipart()
        if self._socket_uses_identity[sock]:
            identity, header, *rest = raw
        else:
            header, *rest = raw
            identity = None
        header = Headers(int.from_bytes(header))
        data = rest[0] if rest else None
        if deserialize:
            data = msgpack.unpackb(data, raw=False) if data is not None else None
        return identity, header, data

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None, *, set_running: bool = True):
        """Start background tasks (send queue, startup buffer, recv task)."""
        from megatron.core.utils import get_asyncio_loop

        loop = get_asyncio_loop(loop)

        @trace_async_exceptions
        async def startup_sends_task():
            await self.is_running.wait()
            for header, data, identity, serialize, sock in self._startup_sends:
                self._isend(header, data, identity=identity, serialize=serialize, sock=sock)
            self._startup_sends = None

        self._startup_sends_task = loop.create_task(startup_sends_task())
        self.send_task = loop.create_task(self._send_task())
        if hasattr(self, '_recv_task'):
            self.recv_task = loop.create_task(self._recv_task())
        if set_running:
            self.is_running.set()

    async def shutdown(self):
        """Stop background tasks, close sockets, and terminate the context if owned.

        Call order matters: tasks must exit before sockets close, sockets must close
        before `ctx.term()` runs. `ctx.term()` blocks until every socket in the context
        is closed, so any remaining open socket would deadlock the terming coroutine.
        """
        self.is_shutdown = True

        # Cancel recv and startup tasks.
        cancel_tasks = []
        for attr in ('recv_task', '_startup_sends_task'):
            task = getattr(self, attr, None)
            if task is not None and not task.done():
                task.cancel()
                cancel_tasks.append(task)
        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        # Drain the send queue, then stop the send task.
        self._send_awaitables.shutdown()
        send_task = getattr(self, 'send_task', None)
        if send_task is not None and not send_task.done():
            await asyncio.gather(send_task, return_exceptions=True)

        # Close all sockets after tasks have exited.
        for s in self._sockets:
            if not s.closed:
                s.close(linger=0)

        # Terminate the context only if we own it. Shared contexts are owned by the
        # caller and terminated there after all sibling endpoints have shut down.
        if self._owns_ctx and not self._ctx.closed:
            self._ctx.term()

    @trace_async_exceptions
    async def _send_task(self):
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
                elif self.is_shutdown:
                    logging.debug("ZMQ send error during shutdown: %s", e)
                    self._send_awaitables.task_done()
                else:
                    raise
