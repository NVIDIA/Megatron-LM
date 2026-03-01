# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""ZMQ layer between DynamicInferenceEngine and the DP coordinator.

Three classes:

- `_DPCoordinator` (AsyncZmqEndpoint subclass): DEALER connected to the DP coordinator.
    Receives messages and accumulates them. Only exists on MP coordinator ranks.

- `_MPChannel` (AsyncZmqEndpoint subclass): PUB on the MP coordinator, SUB on followers.
    Leader sends batched messages via PUB; followers unpack into `pending_messages`.

- `_CollectiveChannel` (AsyncZmqEndpoint subclass): Multiplexed PUSH/PULL + PUB/SUB channel
    supporting an all-reduce-max collective. Used for both EP consensus and world barriers.
    The EP instance also carries a lightweight peer-has-work wakeup signal via
    `notify_has_signal` / `has_signal_event`; the world instance does not use that path.

- `EngineCoordinatorClient`: Owns the above endpoints. Provides API methods to the engine:
    `schedule_requests`, `send_engine_reply`, `ep_establish_consensus`, and `world_barrier`.
"""

import asyncio
import struct
from collections import deque

import msgpack
import torch
import zmq
import zmq.asyncio
from torch.cuda.nvtx import range_pop, range_push

from megatron.core.inference.async_zmq_communicator import AsyncZmqEndpoint
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import asyncio_Queue
from megatron.core.utils import get_asyncio_loop, get_pg_rank, get_pg_size, trace_async_exceptions


class _DPCoordinator(AsyncZmqEndpoint):
    """DEALER socket connected to the DP coordinator (MP coordinator only).

    Receives messages from the coordinator and accumulates them in `pending_messages`.
    `schedule_requests` later sends the accumulated batch to all MP ranks via the `_MPChannel`.

    Provides `send_engine_reply` to send messages back to the coordinator.
    """

    _CONTROL_HEADERS = {
        Headers.PAUSE,
        Headers.UNPAUSE,
        Headers.SUSPEND,
        Headers.RESUME,
        Headers.STOP,
    }

    def __init__(
        self,
        dp_addr: str | None,
        dp_rank: int,
        pending_messages: deque,
        cond: asyncio.Condition,
        *,
        context: zmq.asyncio.Context,
        process_group,
    ):
        identity = f'mp-coord-{dp_rank}'
        super().__init__(
            "DEALER",
            context=context,
            identity=identity,
            connect=dp_addr,
            process_group=process_group,
        )

        self._pending_messages = pending_messages
        self._cond = cond

        # Register with the coordinator (buffered until start()).
        self._isend(Headers.ENGINE_CONNECT)

    def send_engine_reply(self, data: list):
        """Send an ENGINE_REPLY to the coordinator."""
        self._isend(Headers.ENGINE_REPLY, data)

    @trace_async_exceptions
    async def _recv_task(self):
        """Receive from the coordinator and accumulate in pending_messages.

        Control signals (PAUSE, UNPAUSE, etc.) are placed directly into
        pending_messages rather than routed through the world channel PUB/SUB.
        schedule_requests then broadcasts them to TP followers via _MPChannel,
        which is a reliable DEALER/PUB path. The world channel is reserved for
        world_barrier collectives only.
        """
        while True:
            try:
                _, header, data = await self._irecv(deserialize=False)
                self._pending_messages.append(
                    (header, None) if header in self._CONTROL_HEADERS else (header, data)
                )
                async with self._cond:
                    self._cond.notify_all()
            except asyncio.CancelledError:
                break

    async def shutdown(self):
        """Send DISCONNECT, then stop background tasks and close."""
        if not self.is_shutdown:
            try:
                self._isend(Headers.DISCONNECT)
            except Exception:
                pass
        await super().shutdown()


class _MPChannel(AsyncZmqEndpoint):
    """PUB/SUB channel for broadcasting batched messages to all MP ranks.

    Leader (MP coordinator) sends batches via PUB from `schedule_requests`. Followers listen
    via SUB, unpack the batch into `pending_messages`, and signal `messages_processing_event`.
    """

    def __init__(
        self,
        pending_messages: deque,
        messages_processing_event: asyncio.Event,
        cond: asyncio.Condition,
        *,
        context: zmq.asyncio.Context,
        process_group,
        is_leader: bool,
    ):
        socket_type = "PUB" if is_leader else "SUB"
        super().__init__(
            socket_type,
            context=context,
            bind=True,
            process_group=process_group,
            is_leader=is_leader,
        )

        self._is_leader = is_leader
        self._pending_messages = pending_messages
        self._messages_processing_event = messages_processing_event
        self._cond = cond

    @trace_async_exceptions
    async def _recv_task(self):
        """Receive batched messages from the leader and unpack."""
        if self._is_leader:
            return  # PUB socket is send-only.
        while True:
            try:
                _, header, batch = await self._irecv()
                for header_int, data in batch:
                    self._pending_messages.append((Headers(header_int), data))
                self._messages_processing_event.set()
                async with self._cond:
                    self._cond.notify_all()
            except asyncio.CancelledError:
                break


class _CollectiveChannel(AsyncZmqEndpoint):
    """Multiplexed gather+bcast channel supporting collectives and signals.

    Owns a PULL/PUSH + PUB/SUB socket pair and runs a `_recv_task` that demuxes by header,
    feeding `all_reduce_max` data into `_collective_inbox` and `COLLECTIVE_SIGNAL`
    notifications into `has_signal_event`.

    Socket creation is handled by the base class via `bind=True` + `process_group`: the leader
    binds PULL+PUB, addresses are broadcast, and followers auto-complement to PUSH+SUB and connect.
    """

    GATHER, BCAST = 0, 1

    def __init__(
        self,
        process_group,
        *,
        context: zmq.asyncio.Context,
        cond: asyncio.Condition | None = None,
    ):
        if process_group is not None:
            self._rank = get_pg_rank(process_group)
            self._world_size = get_pg_size(process_group)
        else:
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        self._is_leader = self._rank == 0

        if self._is_leader:
            socket_types = ["PULL", "PUB"]
        else:
            socket_types = ["PUSH", "SUB"]

        super().__init__(
            socket_types,
            context=context,
            bind=True,
            process_group=process_group,
            is_leader=self._is_leader,
        )

        self._cond = cond
        self._collective_inbox = asyncio_Queue()
        self.has_signal_event = asyncio.Event()

    @trace_async_exceptions
    async def _recv_task(self):
        """Sole consumer of the receive socket; demuxes by header."""
        recv_sock = self.GATHER if self._is_leader else self.BCAST
        while True:
            try:
                _, header, data = await self._irecv(deserialize=False, sock=recv_sock)

                if header == Headers.COLLECTIVE_DATA or header == Headers.COLLECTIVE_RESULT:
                    await self._collective_inbox.put(data)
                elif header == Headers.COLLECTIVE_SIGNAL:
                    if self._is_leader:
                        self._isend(Headers.COLLECTIVE_SIGNAL, sock=self.BCAST)
                    self.has_signal_event.set()
                    if self._cond is not None:
                        async with self._cond:
                            self._cond.notify_all()
            except asyncio.CancelledError:
                break

    async def all_reduce_max(self, *local_vals: int) -> int | tuple[int, ...]:
        """Element-wise all-reduce max, routed through the recv_task."""
        n = len(local_vals)
        if n == 0:
            raise ValueError("all_reduce_max requires at least one value")

        fmt = f'!{n}i'
        payload = struct.pack(fmt, *local_vals)

        if self._is_leader:
            rows = [local_vals]
            for _ in range(self._world_size - 1):
                msg = await self._collective_inbox.get()
                rows.append(struct.unpack(fmt, msg))
            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self._isend(
                Headers.COLLECTIVE_RESULT,
                struct.pack(fmt, *maxes),
                sock=self.BCAST,
                serialize=False,
            )
            return maxes[0] if n == 1 else maxes
        else:
            self._isend(Headers.COLLECTIVE_DATA, payload, sock=self.GATHER, serialize=False)
            msg = await self._collective_inbox.get()
            result = struct.unpack(fmt, msg)
            return result[0] if n == 1 else result

    def notify_has_signal(self):
        """Inject a signal into the channel. Any rank can call this.

        The signal is just a wakeup — the receiving side only cares that some peer has work,
        not what the specific signal was, so no payload is sent.
        """
        if self._is_leader:
            self._isend(Headers.COLLECTIVE_SIGNAL, sock=self.BCAST)
            # The leader will not receive its own broadcast, so handle the signal locally.
            self.has_signal_event.set()
            # Wake the engine's wait_for loop so it re-evaluates ep_peer_has_work.
            if self._cond is not None:
                get_asyncio_loop().create_task(self._notify_cond())
        else:
            self._isend(Headers.COLLECTIVE_SIGNAL, sock=self.GATHER)

    async def _notify_cond(self):
        async with self._cond:
            self._cond.notify_all()


class EngineCoordinatorClient:
    """ZMQ layer between DynamicInferenceEngine and the DP coordinator.

    On all ranks:
      - `_mp_channel` (_MPChannel) — PUB on MP coordinator, SUB on followers

    On MP coordinator ranks only:
      - `_coord` (_DPCoordinator) — DEALER connected to the DP coordinator

    Also owns `_CollectiveChannel` instances for EP consensus and world barriers.
    """

    def __init__(
        self,
        *,
        is_mp_coordinator: bool,
        dp_addr: str | None,
        dp_rank: int,
        dp_group,
        mp_group,
        ep_group,
        cond: asyncio.Condition,
        listening_timeout: float = 0,
        steps_before_listen: int = 1,
    ):
        self.is_mp_coordinator = is_mp_coordinator
        self.listening_timeout = listening_timeout
        self.steps_before_listen = steps_before_listen

        # Private ZMQ context shared by all endpoints this client owns. Never use
        # `zmq.asyncio.Context.instance()` here — a process-wide singleton means a
        # `term()` in one consumer (InferenceClient, tests, etc.) would tear down
        # sockets owned by this client too.
        self._ctx = zmq.asyncio.Context()

        # Shared mutable state consumed by endpoints and schedule_requests.
        self.pending_messages = deque()
        self.messages_processing_event = asyncio.Event()

        # Combined PUB/SUB channel for all MP ranks.
        # Leader (MP coordinator) binds PUB; followers connect SUB.
        self._mp_channel = _MPChannel(
            self.pending_messages,
            self.messages_processing_event,
            cond,
            context=self._ctx,
            process_group=mp_group,
            is_leader=is_mp_coordinator,
        )

        # Track all AsyncZmqEndpoint instances for lifecycle management.
        self._endpoints = [self._mp_channel]

        if is_mp_coordinator:
            # dp_addr is broadcast from DP rank 0 via process_group.
            self._coord = _DPCoordinator(
                dp_addr,
                dp_rank,
                self.pending_messages,
                cond,
                context=self._ctx,
                process_group=dp_group,
            )
            self._endpoints.append(self._coord)
        else:
            self._coord = None

        # EP and world collective channels.
        self.ep_rank = get_pg_rank(ep_group)
        self.ep_world_size = get_pg_size(ep_group)

        self._ep_channel = (
            _CollectiveChannel(ep_group, context=self._ctx, cond=cond)
            if self.ep_world_size > 1
            else None
        )

        total_world_size = torch.distributed.get_world_size()
        self._world_channel = (
            _CollectiveChannel(None, context=self._ctx, cond=cond)
            if total_world_size > 1
            else None
        )

        if self._ep_channel is not None:
            self._endpoints.append(self._ep_channel)
        if self._world_channel is not None:
            self._endpoints.append(self._world_channel)

    def start(self, loop: asyncio.AbstractEventLoop | None = None):
        """Start all owned endpoints and collective channels."""
        loop = get_asyncio_loop(loop)
        for ep in self._endpoints:
            ep.start(loop)

    async def shutdown(self):
        """Stop all endpoints and collective channels, then terminate the shared context.

        Order matters: each endpoint's shutdown closes its sockets (linger=0). Only after
        every child has closed its sockets can we safely term the shared context, otherwise
        term() blocks waiting for stragglers.
        """
        for ep in self._endpoints:
            await ep.shutdown()
        if not self._ctx.closed:
            self._ctx.term()

    def send_engine_reply(self, data: list):
        """Send an ENGINE_REPLY to the coordinator."""
        self._coord.send_engine_reply(data)

    @property
    def ep_peer_has_work(self):
        """True if an EP peer has notified that it has work."""
        return self._ep_channel is not None and self._ep_channel.has_signal_event.is_set()

    async def ep_establish_consensus(
        self, local_work: int, signal_consensus: bool
    ) -> tuple[int, bool]:
        """EP all-reduce to share work counts and pause consensus.

        All-reduces two integers at once:
        - local_work: actual pending request count (always >= 0).
        - consensus flag: -1 if this rank wants to pause, 0 otherwise.

        Using max for both:
        - max(work) > 0 means at least one EP peer has real work.
        - max(consensus) == -1 means ALL peers signaled -1 (PAUSING). A RUNNING peer contributes 0.

        Args:
            local_work: Pending request count for this rank.
            signal_consensus: True if this rank is ready to pause.
        Returns:
            (global_work, all_pausing): max work across EP; whether all peers signaled consensus.
        """
        range_push("ep_establish_consensus")

        consensus_val = -1 if signal_consensus else 0

        if self._ep_channel is not None:
            self._ep_channel.has_signal_event.clear()
            global_work, global_consensus = await self._ep_channel.all_reduce_max(
                local_work, consensus_val
            )
        else:
            global_work, global_consensus = local_work, consensus_val

        range_pop()
        return global_work, global_consensus == -1

    async def world_barrier(self):
        """World-wide ZMQ all-reduce barrier for global rank consensus.

        Used for all state transitions that require global synchronization:
        PAUSING -> PAUSED, UNPAUSING -> RUNNING, SUSPENDING -> SUSPENDED,
        RESUMING -> PAUSED, and STOPPING -> STOPPED.

        No-op when world_size == 1 (communicator is not created).
        """
        range_push("world_barrier")
        if self._world_channel is not None:
            await self._world_channel.all_reduce_max(1)
        range_pop()

    async def schedule_requests(self, engine):
        """Drains pending ZMQ messages and adds requests to the engine.

        This method is a collective operation that must be called by all ranks in an MP group.
        It does not block; idle-blocking is handled by the engine's `_cond`.

        .. note:: EngineState is imported here to avoid a circular import with dynamic_engine.

        The synchronization uses a batched message pattern:
        1.  Background `_DPCoordinator._recv_task` continuously receives messages from the
            coordinator and stores them in `pending_messages` without deserialization.
        2.  The MP coordinator sends the accumulated messages as a single `MESSAGES` to TP ranks.
        3.  Background `_MPChannel._recv_task` on follower ranks unpacks
            the batch into `pending_messages` and sets `messages_processing_event`.
        """
        from megatron.core.inference.engines.dynamic_engine import EngineState

        # If the engine has active requests and is not at (step % N) == 0, return early.
        engine_idle = engine.state in (EngineState.PAUSED, EngineState.SUSPENDED)
        count_trigger = (engine.context.step_count % self.steps_before_listen) == 0
        if not engine_idle and engine.has_unfinished_requests() and (not count_trigger):
            return

        # Yield to the event loop so ZMQ recv tasks can process pending I/O.
        await asyncio.sleep(self.listening_timeout)

        # MP coordinator: snapshot pending_messages and send as a single batch.
        # We always send, even when the snapshot is empty, because followers block on
        # messages_processing_event each iteration — skipping the send would leave them
        # hanging. The empty batch is an intentional heartbeat that keeps followers in
        # lockstep with the leader.
        if self.is_mp_coordinator:
            messages = list(self.pending_messages)
            self.pending_messages.clear()
            self._mp_channel._isend(
                Headers.MESSAGES, [(h.value, d) for h, d in messages]
            )
        else:
            await self.messages_processing_event.wait()
            self.messages_processing_event.clear()
            messages = list(self.pending_messages)
            self.pending_messages.clear()

        # Process batch. Data is raw msgpack bytes — deserialize at consumption.
        # Control signals arrive here via _DPCoordinator → pending_messages → _MPChannel batch.
        new_generation_epoch = None
        pending_signals = []
        has_new_requests = False
        for header, raw_data in messages:
            if header == Headers.SUBMIT_REQUEST:
                request_id, prompt, sampling_params = msgpack.unpackb(raw_data, raw=False)
                sampling_params = SamplingParams.deserialize(sampling_params)
                range_push("add_request")
                engine.add_request(request_id, prompt, sampling_params)
                range_pop()
                has_new_requests = True
            elif header == Headers.SET_GENERATION_EPOCH:
                new_generation_epoch = msgpack.unpackb(raw_data, raw=False)[0]
            else:
                pending_signals.append(header)

        # Notify EP peers that new work arrived.
        if has_new_requests and self._ep_channel is not None:
            self._ep_channel.notify_has_signal()

        if new_generation_epoch is not None:
            engine._generation_epoch = new_generation_epoch
            # Stamp all active requests with the new epoch.
            for entry in engine.requests.values():
                request = entry.record[-1]
                total = len(request.prompt_tokens) + len(request.generated_tokens)
                if total > 0:
                    boundary = (total - 1, new_generation_epoch)
                    if request.policy_epoch is None:
                        request.policy_epoch = [(0, new_generation_epoch)]
                    else:
                        request.policy_epoch.append(boundary)
                    if request.kv_cache_epoch is None:
                        request.kv_cache_epoch = [(0, new_generation_epoch)]
                    else:
                        request.kv_cache_epoch.append(boundary)

        # Apply at most one control signal per iteration. Only the MP coordinator puts
        # unprocessed signals back into its own pending_messages — they'll be rebroadcast
        # to followers on the next iteration as part of the normal batch. Followers must
        # NOT put back, or they accumulate duplicates of every signal the leader rebroadcasts.
        if self.is_mp_coordinator:
            for sig in reversed(pending_signals[1:]):
                self.pending_messages.appendleft((sig, None))
        if pending_signals:
            header = pending_signals[0]

            if header == Headers.PAUSE:
                if engine.state == EngineState.RUNNING:
                    engine.state = EngineState.PAUSING
                    engine._state_events[EngineState.RUNNING].clear()

            elif header == Headers.UNPAUSE:
                assert (
                    engine.state == EngineState.PAUSED
                ), f"Received UNPAUSE in state {engine.state}"
                engine.state = EngineState.UNPAUSING

            elif header == Headers.SUSPEND:
                assert (
                    engine.state == EngineState.PAUSED
                ), f"Received SUSPEND in state {engine.state}"
                engine._state_events[EngineState.RESUMED].clear()
                engine.suspend()
                engine.state = EngineState.SUSPENDING

            elif header == Headers.RESUME:
                assert (
                    engine.state == EngineState.SUSPENDED
                ), f"Received RESUME in state {engine.state}"
                engine._state_events[EngineState.SUSPENDED].clear()
                engine.resume()
                engine.state = EngineState.RESUMING

            elif header == Headers.STOP:
                assert engine.state in (
                    EngineState.PAUSED,
                    EngineState.SUSPENDED,
                ), f"Received STOP in state {engine.state}"
                if engine.state == EngineState.SUSPENDED:
                    engine._state_events[EngineState.SUSPENDED].clear()
                engine.state = EngineState.STOPPING

            else:
                raise UnknownHeaderError(header)
