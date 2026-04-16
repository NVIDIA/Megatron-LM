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
        bind_host: str | None = None,
    ):
        socket_type = "PUB" if is_leader else "SUB"
        super().__init__(
            socket_type,
            context=context,
            bind=True,
            bind_host=bind_host,
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
                if batch is not None:
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
        bind_host: str | None = None,
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
            bind_host=bind_host,
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
        """Broadcast a wakeup signal to EP peers.

        Called when the local engine has new work. The signal carries no payload — peers
        only need to know *some* peer has work so they participate in EP consensus.

        The originator does NOT set its own ``has_signal_event`` or notify ``_cond``:
        it already knows it has work (``has_unfinished_requests`` is true), so both would
        be no-ops. The ``_recv_task`` on remote peers handles the event + cond wake.
        """
        if self._is_leader:
            self._isend(Headers.COLLECTIVE_SIGNAL, sock=self.BCAST)
        else:
            self._isend(Headers.COLLECTIVE_SIGNAL, sock=self.GATHER)


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
        bind_host: str | None = None,
    ):
        self.is_mp_coordinator = is_mp_coordinator
        self.listening_timeout = listening_timeout
        self.steps_before_listen = steps_before_listen

        self._deferred_requests: list[tuple] = []
        self._deferred_epoch: int | None = None
        self._deferred_signals: list = []

        self._ctx = zmq.asyncio.Context()

        self.pending_messages = deque()
        self.messages_processing_event = asyncio.Event()

        self._mp_channel = _MPChannel(
            self.pending_messages,
            self.messages_processing_event,
            cond,
            context=self._ctx,
            process_group=mp_group,
            is_leader=is_mp_coordinator,
            bind_host=bind_host,
        )

        self._endpoints = [self._mp_channel]

        # Ensure all MP ranks have completed their PUB/SUB socket setup.
        torch.distributed.barrier(group=mp_group)

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
            _CollectiveChannel(ep_group, context=self._ctx, cond=cond, bind_host=bind_host)
            if self.ep_world_size > 1
            else None
        )

        total_world_size = torch.distributed.get_world_size()
        self._world_channel = (
            _CollectiveChannel(None, context=self._ctx, cond=cond, bind_host=bind_host)
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
        self, local_work: int, local_step_count: int
    ) -> tuple[int, int]:
        """EP all-reduce for pause consensus and step synchronization.

        Called only by ranks in PAUSING state; the all-reduce blocks for every EP peer.
        Intended to be fired as a background task so the main loop can keep stepping.

        All-reduces two integers at once:
        - local_work: pending request count on this rank.
        - local_step_count: current `context.step_count` on this rank.

        The max of `local_step_count` across ranks is used to compute a common `stop_at_step`.

        Args:
            local_work: Pending request count for this rank.
            local_step_count: This rank's current step count at call time.
        Returns:
            (global_work, max_step_count): max across EP of each field.
        """
        range_push("ep_establish_consensus")

        if self._ep_channel is not None:
            self._ep_channel.has_signal_event.clear()
            global_work, max_step_count = await self._ep_channel.all_reduce_max(
                local_work, local_step_count
            )
        else:
            global_work, max_step_count = local_work, local_step_count

        range_pop()
        return global_work, max_step_count

    def projected_pending_count(self, engine) -> int:
        """Pending request count including drained-but-not-yet-applied SUBMIT_REQUESTs."""
        return (
            engine.context.get_active_request_count()
            + len(engine.waiting_request_ids)
            + len(self._deferred_requests)
        )

    def _apply_signals(self, signals, engine):
        """Apply at most one control signal. Extra signals are put back for re-broadcast.

        Only the MP coordinator puts unprocessed signals back into ``pending_messages`` —
        they'll be rebroadcast to followers on the next iteration as part of the normal
        batch. Followers must NOT put back, or they accumulate duplicates of every signal
        the leader rebroadcasts.
        """
        from megatron.core.inference.engines.dynamic_engine import EngineState

        if self.is_mp_coordinator:
            for sig in reversed(signals[1:]):
                self.pending_messages.appendleft((sig, None))
        if signals:
            header = signals[0]

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

    def apply_deferred(self, engine):
        """Apply state mutations drained by a pipelined `schedule_requests` call.

        Must be called from the main loop AFTER `async_step` has finished;
        never while a step is in flight!
        """
        if self._deferred_requests:
            for request_id, prompt, sampling_params in self._deferred_requests:
                range_push("add_request")
                engine.add_request(request_id, prompt, sampling_params)
                range_pop()
            self._deferred_requests = []

        if self._deferred_epoch is not None:
            engine._generation_epoch = self._deferred_epoch
            # Stamp all active requests with the new epoch.
            for entry in engine.requests.values():
                request = entry.record[-1]
                total = len(request.prompt_tokens) + len(request.generated_tokens)
                if total > 0:
                    boundary = (total - 1, self._deferred_epoch)
                    if request.policy_epoch is None:
                        request.policy_epoch = [(0, self._deferred_epoch)]
                    else:
                        request.policy_epoch.append(boundary)
                    if request.kv_cache_epoch is None:
                        request.kv_cache_epoch = [(0, self._deferred_epoch)]
                    else:
                        request.kv_cache_epoch.append(boundary)
            self._deferred_epoch = None

        if self._deferred_signals:
            signals = self._deferred_signals
            self._deferred_signals = []
            self._apply_signals(signals, engine)

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

    async def schedule_requests(self, engine, *, defer: bool = False):
        """Drains pending ZMQ messages.

        This method is a collective operation that must be called by all ranks in an MP group.
        It does not block; idle-blocking is handled by the engine's `_cond`.

        The synchronization uses a batched message pattern:
        1.  Background `_DPCoordinator._recv_task` continuously receives messages from the
            coordinator and stores them in `pending_messages` without deserialization.
        2.  The MP coordinator sends the accumulated messages as a single `MESSAGES` to TP ranks.
        3.  Background `_MPChannel._recv_task` on follower ranks unpacks
            the batch into `pending_messages` and sets `messages_processing_event`.

        Args:
            engine: The DynamicInferenceEngine instance.
            defer: If True, ALL engine mutations are deferred: SUBMIT_REQUEST and
                SET_GENERATION_EPOCH are stashed in `_deferred_requests` / `_deferred_epoch`,
                and control signals (PAUSE, UNPAUSE, etc.) are stashed in `_deferred_signals`.
                The caller must invoke `apply_deferred(engine)` after `async_step` finishes to
                commit everything atomically.
                When False (default), requests, epochs, and signals are applied
                immediately since no step is in flight.
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
        # hanging. Empty batches send a header-only frame (no msgpack payload) to avoid
        # the serialization round-trip on the hot path.
        if self.is_mp_coordinator:
            messages = list(self.pending_messages)
            self.pending_messages.clear()
            if messages:
                self._mp_channel._isend(Headers.MESSAGES, [(h.value, d) for h, d in messages])
            else:
                self._mp_channel._isend(Headers.MESSAGES)  # header-only heartbeat
        else:
            await self.messages_processing_event.wait()
            self.messages_processing_event.clear()
            messages = list(self.pending_messages)
            self.pending_messages.clear()

        # Process batch. Data is raw msgpack bytes — deserialize at consumption.
        # Control signals arrive here via _DPCoordinator → pending_messages → _MPChannel batch.
        pending_signals = []
        has_new_requests = False
        for header, raw_data in messages:
            if header == Headers.SUBMIT_REQUEST:
                request_id, prompt, sampling_params = msgpack.unpackb(raw_data, raw=False)
                sampling_params = SamplingParams.deserialize(sampling_params)
                self._deferred_requests.append((request_id, prompt, sampling_params))
                has_new_requests = True
            elif header == Headers.SET_GENERATION_EPOCH:
                self._deferred_epoch = msgpack.unpackb(raw_data, raw=False)[0]
            else:
                pending_signals.append(header)

        # Notify EP peers that new work arrived.
        if has_new_requests and self._ep_channel is not None:
            self._ep_channel.notify_has_signal()

        if defer:
            # Pipelined path: stash signals for apply_deferred. No engine mutation
            # occurs during the step — all state changes are deferred until the main
            # loop calls apply_deferred after the step completes.
            self._deferred_signals.extend(pending_signals)
        else:
            # Foreground path: apply requests, epoch, and signals immediately.
            self.apply_deferred(engine)
            self._apply_signals(pending_signals, engine)
