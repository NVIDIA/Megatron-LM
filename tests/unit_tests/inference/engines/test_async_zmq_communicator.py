# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import struct
from unittest.mock import MagicMock, patch

import pytest

from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_comm(rank=0, world_size=1):
    """Construct an AsyncZMQCommunicator instance via __new__ + manual attribute injection.

    The real __init__ rendezvouses against a ProcessGroup and binds ZMQ sockets;
    we bypass that to exercise the public collective methods on plain Mocks.
    """
    comm = AsyncZMQCommunicator.__new__(AsyncZMQCommunicator)
    comm.rank = rank
    comm.world_size = world_size
    comm.is_leader = rank == 0
    comm.gather_sock = MagicMock(name="gather_sock")
    comm.bcast_sock = MagicMock(name="bcast_sock")
    return comm


class TestAsyncZMQCommunicator:

    def test_all_reduce_max_requires_at_least_one_value(self):
        """all_reduce_max raises ValueError if called with zero values."""
        comm = _make_comm(world_size=1)

        async def scenario():
            with pytest.raises(ValueError):
                await comm.all_reduce_max()

        _run(scenario())

    def test_all_reduce_max_single_rank_single_value(self):
        """In a 1-rank world, a single int round-trips unchanged (no socket I/O)."""
        comm = _make_comm(world_size=1)

        async def scenario():
            return await comm.all_reduce_max(42)

        out = _run(scenario())
        assert out == 42
        comm.gather_sock.send.assert_not_called()
        comm.bcast_sock.send.assert_not_called()

    def test_all_reduce_max_single_rank_multiple_values(self):
        """In a 1-rank world, multiple ints come back as a tuple."""
        comm = _make_comm(world_size=1)

        async def scenario():
            return await comm.all_reduce_max(3, 4, 5)

        assert _run(scenario()) == (3, 4, 5)

    def test_all_reduce_max_leader_two_ranks(self):
        """Leader gathers one row from a follower, computes max, and broadcasts."""
        comm = _make_comm(rank=0, world_size=2)
        # Simulate the follower's payload (struct-packed `!1i`).
        follower_payload = struct.pack("!1i", 9)
        comm.gather_sock.recv.return_value = follower_payload

        async def scenario():
            return await comm.all_reduce_max(7)

        result = _run(scenario())
        assert result == 9
        # Leader should have broadcast struct.pack('!1i', 9).
        sent = comm.bcast_sock.send.call_args.args[0]
        assert struct.unpack("!1i", sent) == (9,)

    def test_all_reduce_max_leader_async_again_then_recv(self):
        """When async_op=True, the leader's gather loop tolerates zmq.Again."""
        import zmq

        comm = _make_comm(rank=0, world_size=2)
        first_call = [True]

        def fake_recv(*args, **kwargs):
            if first_call[0]:
                first_call[0] = False
                raise zmq.Again()
            return struct.pack("!1i", 5)

        comm.gather_sock.recv.side_effect = fake_recv

        async def scenario():
            return await comm.all_reduce_max(3, async_op=True)

        assert _run(scenario()) == 5

    def test_all_reduce_max_follower(self):
        """Follower sends its payload and waits for a broadcast reply."""
        comm = _make_comm(rank=1, world_size=2)
        comm.bcast_sock.recv.return_value = struct.pack("!1i", 11)

        async def scenario():
            return await comm.all_reduce_max(2)

        result = _run(scenario())
        assert result == 11
        # Follower should have sent its payload to the leader.
        comm.gather_sock.send.assert_called_once()
        sent_payload = comm.gather_sock.send.call_args.args[0]
        assert struct.unpack("!1i", sent_payload) == (2,)

    def test_all_reduce_max_follower_async_again_then_recv(self):
        """Follower's poll loop swallows zmq.Again and retries until reply arrives."""
        import zmq

        comm = _make_comm(rank=1, world_size=2)
        first = [True]

        def fake_recv(*args, **kwargs):
            if first[0]:
                first[0] = False
                raise zmq.Again()
            return struct.pack("!2i", 8, 9)

        comm.bcast_sock.recv.side_effect = fake_recv

        async def scenario():
            return await comm.all_reduce_max(1, 2, async_op=True)

        assert _run(scenario()) == (8, 9)

    def test_sync_all_reduce_max_requires_at_least_one_value(self):
        """sync_all_reduce_max raises ValueError on zero arguments."""
        comm = _make_comm(world_size=1)
        with pytest.raises(ValueError):
            comm.sync_all_reduce_max()

    def test_sync_all_reduce_max_single_rank(self):
        """In a 1-rank world, sync_all_reduce_max short-circuits without socket I/O."""
        comm = _make_comm(world_size=1)
        assert comm.sync_all_reduce_max(7) == 7
        assert comm.sync_all_reduce_max(1, 2) == (1, 2)

    def test_sync_all_reduce_max_leader(self):
        """sync_all_reduce_max leader path collects and broadcasts the element-wise max."""
        comm = _make_comm(rank=0, world_size=3)
        # Two followers send two rows.
        comm.gather_sock.recv.side_effect = [
            struct.pack("!2i", 4, 8),
            struct.pack("!2i", 5, 7),
        ]
        result = comm.sync_all_reduce_max(3, 6)
        assert result == (5, 8)
        # Broadcast was called with the element-wise max.
        sent = comm.bcast_sock.send.call_args.args[0]
        assert struct.unpack("!2i", sent) == (5, 8)

    def test_sync_all_reduce_max_follower(self):
        """sync_all_reduce_max follower sends and reads back the broadcast."""
        comm = _make_comm(rank=1, world_size=2)
        comm.bcast_sock.recv.return_value = struct.pack("!1i", 99)
        assert comm.sync_all_reduce_max(5) == 99

    def test_close_closes_both_sockets_with_linger_zero(self):
        """close() closes gather and bcast sockets with linger=0."""
        comm = _make_comm(world_size=1)
        comm.close()
        comm.gather_sock.close.assert_called_once_with(linger=0)
        comm.bcast_sock.close.assert_called_once_with(linger=0)
