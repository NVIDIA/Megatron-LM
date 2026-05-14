# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import struct
from unittest.mock import MagicMock

import pytest

from megatron.core.inference.engines.async_zmq_communicator import AsyncZMQCommunicator

pytestmark = pytest.mark.asyncio


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


# (rank, world_size, local_vals, peer_payloads, expected_result, expected_broadcast)
# `peer_payloads` is what gather_sock.recv (leader) or bcast_sock.recv (follower) returns;
# the leader case feeds its own row + N-1 follower payloads then broadcasts the max,
# the follower case sends its own payload and reads back the broadcast.
@pytest.mark.parametrize(
    "rank,world_size,local_vals,peer_payloads,expected_result,expected_broadcast",
    [
        # Single-rank short-circuit: scalar input → scalar return.
        (0, 1, (42,), [], 42, None),
        # Single-rank short-circuit: vector input → tuple return.
        (0, 1, (3, 4, 5), [], (3, 4, 5), None),
        # Leader rank with one follower; element-wise max + broadcast.
        (0, 2, (7,), [struct.pack("!1i", 9)], 9, struct.pack("!1i", 9)),
        # Leader with two followers and two values; element-wise max.
        (0, 3, (3, 6), [struct.pack("!2i", 4, 8), struct.pack("!2i", 5, 7)], (5, 8), struct.pack("!2i", 5, 8)),
        # Follower path: reads broadcast, returns the value.
        (1, 2, (2,), struct.pack("!1i", 11), 11, None),
    ],
)
async def test_sync_all_reduce_max(rank, world_size, local_vals, peer_payloads, expected_result, expected_broadcast):
    """sync_all_reduce_max implements element-wise max across ranks via ZMQ:
    the leader collects N-1 follower payloads, computes the max, and broadcasts;
    each follower sends its payload and reads back the broadcast.

    Single-rank short-circuits avoid socket I/O entirely.
    """
    comm = _make_comm(rank=rank, world_size=world_size)
    if rank == 0 and world_size > 1:
        comm.gather_sock.recv.side_effect = peer_payloads
    elif rank > 0:
        comm.bcast_sock.recv.return_value = peer_payloads

    result = comm.sync_all_reduce_max(*local_vals)
    assert result == expected_result

    if expected_broadcast is not None:
        assert comm.bcast_sock.send.call_args.args[0] == expected_broadcast
    elif world_size == 1:
        # No socket I/O on single-rank.
        comm.gather_sock.send.assert_not_called()
        comm.bcast_sock.send.assert_not_called()


@pytest.mark.parametrize(
    "rank,world_size,with_again",
    [
        # Leader async path: handles zmq.Again before the real payload.
        (0, 2, True),
        # Follower async path: also tolerates zmq.Again before the broadcast.
        (1, 2, True),
        # Async path without retries also works.
        (0, 2, False),
    ],
)
async def test_all_reduce_max_async_handles_zmq_again_retries(rank, world_size, with_again):
    """The async variant must tolerate `zmq.Again` from non-blocking recvs by
    awaiting and retrying. This is the only behaviour that distinguishes it
    from `sync_all_reduce_max`."""
    import zmq

    comm = _make_comm(rank=rank, world_size=world_size)
    real_payload = struct.pack("!1i", 9)

    if with_again:
        calls = [zmq.Again(), real_payload]

        def fake_recv(*args, **kwargs):
            v = calls.pop(0)
            if isinstance(v, BaseException):
                raise v
            return v

    else:

        def fake_recv(*args, **kwargs):
            return real_payload

    target_sock = comm.gather_sock if rank == 0 else comm.bcast_sock
    target_sock.recv.side_effect = fake_recv

    result = await comm.all_reduce_max(7, async_op=True)
    assert result == 9


@pytest.mark.parametrize("method_name", ["all_reduce_max", "sync_all_reduce_max"])
async def test_reject_zero_values(method_name):
    """Both async and sync variants reject empty arg lists (no element-wise max
    is defined over zero rows)."""
    comm = _make_comm(world_size=1)
    fn = getattr(comm, method_name)
    if method_name == "all_reduce_max":
        with pytest.raises(ValueError):
            await fn()
    else:
        with pytest.raises(ValueError):
            fn()
