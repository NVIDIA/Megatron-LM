# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket
import struct
from collections.abc import Iterable

import torch.distributed as dist

try:
    import zmq

    HAVE_ZMQ = True
except ImportError:
    from unittest.mock import MagicMock

    zmq = MagicMock()
    HAVE_ZMQ = False


class RankedPubSub:
    """Create PUB/SUB sockets with rank-identified subscription readiness."""

    def __init__(self, readiness_topic_prefix: bytes):
        self.readiness_topic_prefix = readiness_topic_prefix

    def _readiness_topic(self, rank: int) -> bytes:
        return self.readiness_topic_prefix + struct.pack("!I", rank)

    @staticmethod
    def create_publisher(zmq_context: zmq.Context) -> zmq.Socket:
        """Create an XPUB socket that exposes every subscription notification."""
        publisher_socket = zmq_context.socket(zmq.XPUB)
        publisher_socket.setsockopt(zmq.XPUB_VERBOSE, 1)
        return publisher_socket

    def create_subscriber(self, zmq_context: zmq.Context, address: str, rank: int) -> zmq.Socket:
        """Create a connected SUB socket with a rank-specific readiness topic."""
        subscriber_socket = zmq_context.socket(zmq.SUB)
        subscriber_socket.connect(address)
        # Subscribe to readiness before the empty collective topic so XPUB
        # receives an explicit identity command for this rank.
        subscriber_socket.setsockopt(zmq.SUBSCRIBE, self._readiness_topic(rank))
        subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        return subscriber_socket

    def wait_for_subscribers(
        self, publisher_socket: zmq.Socket, ranks: Iterable[int], *, timeout_ms: int = 60_000
    ) -> None:
        """Wait until the XPUB socket reports every distinct rank as subscribed."""
        topics_by_rank = {rank: self._readiness_topic(rank) for rank in ranks}
        expected_topics = set(topics_by_rank.values())
        subscribed_topics = set()
        publisher_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            while subscribed_topics != expected_topics:
                notification = publisher_socket.recv()
                event, topic = notification[:1], notification[1:]
                if topic not in expected_topics:
                    continue
                if event == b"\x01":
                    subscribed_topics.add(topic)
                elif event == b"\x00":
                    subscribed_topics.discard(topic)
        except zmq.Again as exc:
            missing_ranks = [
                rank for rank, topic in topics_by_rank.items() if topic not in subscribed_topics
            ]
            raise RuntimeError(
                f"Timed out waiting for ZMQ subscribers for prefix "
                f"{self.readiness_topic_prefix!r}; missing ranks: {missing_ranks}"
            ) from exc
        finally:
            publisher_socket.setsockopt(zmq.RCVTIMEO, -1)


class AsyncZMQCommunicator:
    """
    An asyncio-friendly communicator abstraction using ZMQ.
    Can be used to implement collective operations like all-reduce,
    and bcast which are asyncio friendly on top of ZMQ sockets.
    Only to be used with small amounts of data (e.g., 1 integer)
    on the CPU.
    """

    def __init__(
        self,
        zmq_context: zmq.Context,
        process_group: dist.ProcessGroup,
        hostname: str | None = None,
    ):
        """
        Constructor for AsyncZMQCommunicator. Sets up ZMQ sockets
        for communication among ranks in the given process group.
        Args:
            zmq_context (zmq.Context): ZMQ context to create sockets.
            process_group (dist.ProcessGroup): Process group for communication.
            hostname (str | None): Hostname or IP address to use for ZMQ socket binding.
                If None, defaults to socket.gethostname().
        """
        # Normalize None to the default (world) group. get_rank/get_world_size
        # already treat None this way, but get_process_group_ranks below does
        # not accept None, so resolve it once here for all three calls.
        if process_group is None:
            process_group = dist.group.WORLD

        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.is_leader = self.rank == 0
        broadcast_pub_sub = RankedPubSub(b"AsyncZMQCommunicator.broadcast:")
        # Get the global rank of the leader (first rank in the process group).
        src_rank = dist.get_process_group_ranks(process_group)[0]

        if self.is_leader:
            local_ip = hostname or socket.gethostname()
            self.gather_sock = zmq_context.socket(zmq.PULL)
            self.gather_sock.bind_to_random_port(f"tcp://{local_ip}")
            gather_socket_addr = self.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            self.bcast_sock = broadcast_pub_sub.create_publisher(zmq_context)
            self.bcast_sock.bind_to_random_port(f"tcp://{local_ip}")
            bcast_socket_addr = self.bcast_sock.getsockopt_string(zmq.LAST_ENDPOINT)

            # Share the socket addresses with all peers
            dist.broadcast_object_list(
                [gather_socket_addr, bcast_socket_addr], src=src_rank, group=process_group
            )

        else:
            bcast_output = [None, None]
            dist.broadcast_object_list(bcast_output, src=src_rank, group=process_group)
            gather_socket_addr, bcast_socket_addr = bcast_output
            self.gather_sock = zmq_context.socket(zmq.PUSH)
            self.gather_sock.connect(gather_socket_addr)
            self.bcast_sock = broadcast_pub_sub.create_subscriber(
                zmq_context, bcast_socket_addr, self.rank
            )

        # Wait until every ProcessGroup peer has subscribed to the leader.
        # Rank-specific topics make duplicate reconnect notifications idempotent.
        if self.is_leader:
            broadcast_pub_sub.wait_for_subscribers(self.bcast_sock, range(1, self.world_size))

        # Wait until all ProcessGroup peers have subscribed to the ZMQ leader.
        # Otherwise, ranks that fail to subscribe in time will deadlock.
        if self.is_leader:
            # XPUB subscription messages are one byte for an empty-topic
            # subscription: b"\x01". XPUB_VERBOSE is required so identical
            # subs from every peer are reported rather than coalesced.
            subscribed_peers = 0
            self.bcast_sock.setsockopt(zmq.RCVTIMEO, 60_000)
            try:
                while subscribed_peers < self.world_size - 1:
                    subscription = self.bcast_sock.recv()
                    if subscription == b"\x01":
                        subscribed_peers += 1
            except zmq.Again as exc:
                raise RuntimeError(
                    "[AsyncZMQCommunicator] Timed out waiting for ZMQ subscribers: "
                    f"{subscribed_peers}/{self.world_size - 1} connected"
                ) from exc
            finally:
                self.bcast_sock.setsockopt(zmq.RCVTIMEO, -1)

    async def all_reduce_max(self, *local_vals: int, async_op=True) -> int | tuple[int, ...]:
        """Element-wise all-reduce max of one or more integers.

        Packs all values into a single message so the communication cost
        is independent of the number of values.

        Returns a single int when called with one argument, otherwise a tuple.
        """
        n = len(local_vals)
        if n == 0:
            raise ValueError("all_reduce_max requires at least one value")

        if self.world_size <= 1:
            return local_vals[0] if n == 1 else local_vals

        fmt = f'!{n}i'
        payload = struct.pack(fmt, *local_vals)

        if self.is_leader:
            rows = [local_vals]

            while len(rows) < self.world_size:
                try:
                    if async_op:
                        msg = self.gather_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.gather_sock.recv()
                    rows.append(struct.unpack(fmt, msg))
                except zmq.Again:
                    await asyncio.sleep(0.001)

            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self.bcast_sock.send(struct.pack(fmt, *maxes))
            if not async_op:
                await asyncio.sleep(
                    0
                )  # Yield control once to ensure that other coroutines can run.
                # This might be needed for colocated RL.
            return maxes[0] if n == 1 else maxes

        else:
            self.gather_sock.send(payload)

            while True:
                try:
                    if async_op:
                        msg = self.bcast_sock.recv(flags=zmq.NOBLOCK)
                    else:
                        msg = self.bcast_sock.recv()
                    result = struct.unpack(fmt, msg)
                    if not async_op:
                        await asyncio.sleep(
                            0
                        )  # Yield control once to ensure that other coroutines can run.
                        # This might be needed for colocated RL.
                    return result[0] if n == 1 else result
                except zmq.Again:
                    await asyncio.sleep(0.001)

    def sync_all_reduce_max(self, *local_vals: int) -> int | tuple[int, ...]:
        """Synchronous (non-asyncio) variant of all_reduce_max.

        Uses blocking ZMQ sends/recvs so it can be called from synchronous
        call sites that need a CPU-only MAX reduction across the process
        group. Intended for tiny payloads (e.g. a few integers) that would
        otherwise force a NCCL AllReduce kernel on the compute stream.

        Note: when called from inside a running asyncio event loop, the
        blocking recv will pause other coroutines on this rank until all
        peers respond. This is acceptable here because every rank reaches
        the call simultaneously and the message size is trivial.

        Returns a single int when called with one argument, otherwise a tuple.
        """
        n = len(local_vals)
        if n == 0:
            raise ValueError("sync_all_reduce_max requires at least one value")

        if self.world_size <= 1:
            return local_vals[0] if n == 1 else local_vals

        fmt = f'!{n}i'
        payload = struct.pack(fmt, *local_vals)

        if self.is_leader:
            rows = [local_vals]
            while len(rows) < self.world_size:
                msg = self.gather_sock.recv()
                rows.append(struct.unpack(fmt, msg))
            maxes = tuple(max(row[i] for row in rows) for i in range(n))
            self.bcast_sock.send(struct.pack(fmt, *maxes))
            return maxes[0] if n == 1 else maxes
        else:
            self.gather_sock.send(payload)
            msg = self.bcast_sock.recv()
            result = struct.unpack(fmt, msg)
            return result[0] if n == 1 else result

    def close(self):
        """
        Close the ZMQ sockets.
        """
        # linger=0: discard unsent messages immediately on close rather than blocking until sent.
        # The ZMQ default is to not allow `close` until all messages have been successfully sent.
        self.gather_sock.close(linger=0)
        self.bcast_sock.close(linger=0)
