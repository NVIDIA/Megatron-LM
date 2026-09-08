# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The coordinator orders data-parallel ranks by identity, whatever order they connect in.

The least-loaded tie-break and the prefix-cache rank indices are derived from that order, so it
decides which engine a request lands on. It must depend neither on connection timing nor on
``deterministic_mode``: sorting a handful of identities once at start-up costs nothing, so the
coordinator always does it, and engines that register after start-up slot into the same order.
"""

import unittest.mock as mock

import pytest

from megatron.core.inference.data_parallel_inference_coordinator import (
    coordinator as coordinator_module,
)

RANKS = [f"mp-coord-{i}".encode() for i in range(6)]


def _start_coordinator(connection_order, **kwargs):
    """Construct a coordinator whose ROUTER socket sees ``connection_order`` registrations."""
    zmq_mock = mock.MagicMock()
    router = zmq_mock.Context.return_value.socket.return_value
    router.recv_multipart.side_effect = [(identity, b"") for identity in connection_order]
    router.getsockopt_string.return_value = "tcp://127.0.0.1:1"
    with (
        mock.patch.object(coordinator_module, "zmq", zmq_mock),
        mock.patch.object(coordinator_module, "HAVE_ZMQ", True),
        mock.patch.object(coordinator_module, "HAVE_MSGPACK", True),
    ):
        return coordinator_module.DataParallelInferenceCoordinator(
            pipe_connection=mock.MagicMock(),
            data_parallel_size=len(connection_order),
            tokenizer=None,
            max_requests=4,
            **kwargs,
        )


@pytest.mark.parametrize("deterministic_mode", [False, True])
def test_rank_order_is_independent_of_connection_order(deterministic_mode):
    orders = [RANKS, list(reversed(RANKS)), [RANKS[i] for i in (3, 0, 5, 1, 4, 2)]]
    for order in orders:
        coordinator = _start_coordinator(order, deterministic_mode=deterministic_mode)
        assert list(coordinator.identities_of_data_parallel_ranks) == sorted(RANKS)
        assert coordinator._identities_list == sorted(RANKS)
        assert coordinator.identity_to_rank_index == {
            identity: idx for idx, identity in enumerate(sorted(RANKS))
        }
        # Nothing is loaded yet, so the tie-break picks the lowest rank index every time.
        assert coordinator.get_least_loaded_data_parallel_rank() == sorted(RANKS)[0]


def test_dynamic_registration_keeps_identity_order():
    """Engines that connect after start-up (``data_parallel_size=0``, reconnects) are inserted at
    their sorted position, and the index-keyed state (pending counts, prefix-cache hash table)
    follows them, so a late-joining lower identity still wins an equal-load tie."""
    coordinator = _start_coordinator([])
    coordinator._handle_rank_registration(b"mp-coord-1")
    first = coordinator.identity_to_rank_index[b"mp-coord-1"]
    coordinator._pending_counts[first] = 5
    coordinator._hash_table = {"block": {first: 1.0}}

    coordinator._handle_rank_registration(b"mp-coord-0")
    coordinator._handle_rank_registration(b"mp-coord-2")
    coordinator._handle_rank_registration(b"mp-coord-0")  # a reconnect changes nothing

    assert coordinator._identities_list == [b"mp-coord-0", b"mp-coord-1", b"mp-coord-2"]
    assert list(coordinator.identities_of_data_parallel_ranks) == coordinator._identities_list
    assert coordinator.identity_to_rank_index == {
        b"mp-coord-0": 0,
        b"mp-coord-1": 1,
        b"mp-coord-2": 2,
    }
    assert coordinator._pending_counts.tolist() == [0, 5, 0]
    assert coordinator._hash_table == {"block": {1: 1.0}}
    assert coordinator.get_least_loaded_data_parallel_rank() == b"mp-coord-0"
