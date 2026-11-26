# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Authentication of the sockets that engine ranks open among themselves.

These sockets are reachable over TCP next to the (authenticated) coordinator, so a peer that is
not a rank of the group must not be able to join, inject messages, or eavesdrop.
"""

import asyncio
import os
import socket
import struct
import time

import pytest
import torch.multiprocessing as mp

zmq = pytest.importorskip("zmq")

from megatron.core.inference.data_parallel_inference_coordinator.auth import (
    SocketAuth,
    engine_socket_auth,
    generate_auth_token,
    secure_client,
    secure_server,
    stop_context_authentication,
)
from megatron.core.inference.disaggregation.handoff_completion_tracker import (
    HandoffCompletionTracker,
)
from megatron.core.inference.engines.async_zmq_communicator import (
    AsyncZMQCommunicator,
    RankedPubSub,
)

HOST = "127.0.0.1"


@pytest.fixture
def context():
    ctx = zmq.Context()
    yield ctx
    stop_context_authentication(ctx)
    ctx.destroy(linger=0)  # Closes leftover sockets, so a failing test cannot hang term().


def _delivered(server, client, payload=b"x", timeout_ms=1000) -> bool:
    client.send(payload)
    if not server.poll(timeout_ms):
        return False
    server.recv()  # Consume it, so a later call cannot mistake it for its own.
    return True


def _pull_server(context, auth):
    server = context.socket(zmq.PULL)
    secure_server(server, auth)
    port = server.bind_to_random_port(f"tcp://{HOST}")
    return server, f"tcp://{HOST}:{port}"


def _push_client(context, endpoint, auth):
    client = context.socket(zmq.PUSH)
    client.setsockopt(zmq.LINGER, 0)
    secure_client(client, auth)
    client.connect(endpoint)
    return client


def test_only_the_engine_secret_is_accepted(context):
    token = generate_auth_token()
    auth = engine_socket_auth(context, token)
    server, endpoint = _pull_server(context, auth)

    # A credential-less peer, a wrong secret in the right domain, and the right secret in
    # another domain (as another engine of the process would hold) are all refused.
    assert not _delivered(server, _push_client(context, endpoint, None))
    wrong_secret = SocketAuth(auth.domain, generate_auth_token())
    assert not _delivered(server, _push_client(context, endpoint, wrong_secret))
    other_engine = engine_socket_auth(context, generate_auth_token())
    assert not _delivered(server, _push_client(context, endpoint, other_engine))
    assert _delivered(server, _push_client(context, endpoint, auth))


def test_every_rank_can_install_the_same_secret(context):
    token = generate_auth_token()
    first, second = engine_socket_auth(context, token), engine_socket_auth(context, token)
    assert first == second
    server, endpoint = _pull_server(context, first)

    assert _delivered(server, _push_client(context, endpoint, second))


def test_engines_sharing_a_context_do_not_share_secrets(context):
    first = engine_socket_auth(context, generate_auth_token())
    second = engine_socket_auth(context, generate_auth_token())
    server_a, endpoint_a = _pull_server(context, first)
    server_b, endpoint_b = _pull_server(context, second)

    assert _delivered(server_a, _push_client(context, endpoint_a, first))
    assert _delivered(server_b, _push_client(context, endpoint_b, second))
    assert not _delivered(server_a, _push_client(context, endpoint_a, second))
    assert not _delivered(server_b, _push_client(context, endpoint_b, first))


def test_unauthenticated_sockets_are_unaffected_by_a_handler_on_the_context(context):
    engine_socket_auth(context, generate_auth_token())  # Installs the context's ZAP handler.
    server, endpoint = _pull_server(context, None)

    assert _delivered(server, _push_client(context, endpoint, None))


def test_publisher_refuses_subscribers_without_the_secret(context):
    auth = engine_socket_auth(context, generate_auth_token())
    pubsub = RankedPubSub(b"test:")
    publisher = pubsub.create_publisher(context, auth)
    port = publisher.bind_to_random_port(f"tcp://{HOST}")
    endpoint = f"tcp://{HOST}:{port}"

    intruder = pubsub.create_subscriber(context, endpoint, 7)
    member = pubsub.create_subscriber(context, endpoint, 1, auth)
    pubsub.wait_for_subscribers(publisher, [1], timeout_ms=5000)
    publisher.send(b"secret prompt")

    assert member.poll(2000)
    assert member.recv() == b"secret prompt"
    assert not intruder.poll(300)


def test_stopping_the_handler_lets_the_context_terminate():
    ctx = zmq.Context()
    engine_socket_auth(ctx, generate_auth_token())
    stop_context_authentication(ctx)
    ctx.term()  # Would block forever if the handler were still running.
    stop_context_authentication(ctx)  # Idempotent.


# ------------------------------------------------------------------ multi-rank, real collectives


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def _rank_main(rank, world_size, port, token, results):
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR=HOST, MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    ctx = zmq.Context()
    auth = engine_socket_auth(ctx, token) if token is not None else None

    async def run():
        comm = AsyncZMQCommunicator(ctx, dist.group.WORLD, hostname=HOST, auth=auth)
        tracker = HandoffCompletionTracker(ctx, dist.group.WORLD, hostname=HOST, auth=auth)
        dist.barrier()
        out = {}
        if rank == 0:
            attacker = zmq.Context()
            gather = attacker.socket(zmq.PUSH)
            gather.connect(comm.gather_sock.getsockopt_string(zmq.LAST_ENDPOINT))
            forger = attacker.socket(zmq.PUSH)
            forger.connect(tracker._socket.getsockopt_string(zmq.LAST_ENDPOINT))
            sniffer = attacker.socket(zmq.SUB)
            sniffer.connect(comm.bcast_sock.getsockopt_string(zmq.LAST_ENDPOINT))
            sniffer.setsockopt(zmq.SUBSCRIBE, b"")
            time.sleep(0.3)
            gather.send(struct.pack("!1i", 999))  # A forged row for the max-reduction.
            for forged_rank in range(world_size):
                forger.send(
                    struct.pack(
                        HandoffCompletionTracker._REPORT_FORMAT, 5, forged_rank, False, True
                    )
                )
            time.sleep(0.5)
            out["forged_reports_completed"] = tracker.drain_completed()
        dist.barrier()
        out["max"] = await comm.all_reduce_max(rank + 1)  # Honest inputs: 1 and 2.
        dist.barrier()
        if rank == 0:
            out["sniffed"] = bool(sniffer.poll(300))
        results.put((rank, out))
        dist.barrier()

    asyncio.run(run())
    results.put((rank, "done"))
    os._exit(0)  # Skip teardown: only the assertions above matter here.


def _run_two_ranks(authenticate):
    spawn = mp.get_context("spawn")
    results = spawn.Queue()
    port = _free_port()
    token = generate_auth_token() if authenticate else None  # One secret for every rank.
    procs = [
        spawn.Process(target=_rank_main, args=(rank, 2, port, token, results)) for rank in range(2)
    ]
    for proc in procs:
        proc.start()
    outputs = {}
    deadline = time.time() + 120
    while len(outputs) < 2 and time.time() < deadline:
        try:
            rank, out = results.get(timeout=1)
        except Exception:
            continue
        if out != "done":
            outputs[rank] = out
    for proc in procs:
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
    assert len(outputs) == 2, "ranks did not finish"
    return outputs


def test_forged_messages_do_not_reach_authenticated_rank_sockets():
    outputs = _run_two_ranks(authenticate=True)

    assert outputs[0]["max"] == outputs[1]["max"] == 2  # The forged 999 was refused.
    assert outputs[0]["forged_reports_completed"] == []
    assert outputs[0]["sniffed"] is False


def test_the_same_forgeries_work_without_authentication():
    # Guards the test above: its attacks do land when the sockets are not authenticated.
    outputs = _run_two_ranks(authenticate=False)

    assert outputs[0]["max"] == outputs[1]["max"] == 999
    assert outputs[0]["forged_reports_completed"] == [(5, False, True)]
    assert outputs[0]["sniffed"] is True
