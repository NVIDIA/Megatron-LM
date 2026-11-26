# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio

import msgpack
import pytest
import zmq
import zmq.asyncio

from megatron.core.inference.data_parallel_inference_coordinator.auth import (
    authenticated_address,
    generate_auth_token,
    require_authentication,
    split_authenticated_address,
    start_server_authentication,
)
from megatron.core.inference.headers import Headers
from megatron.inference.integrations.dynamo.args import parse_args
from megatron.inference.integrations.dynamo.handoff_channel import HandoffChannel
from megatron.inference.integrations.dynamo.source_credentials import (
    SourceCredentials,
    UnknownSourceCoordinator,
)

PREFILL_0 = "tcp://prefill-0:5000"
PREFILL_1 = "tcp://prefill-1:5000"


def _write(path, address):
    path.write_text(address + "\n")
    return str(path)


def test_without_files_any_endpoint_is_used_unauthenticated():
    credentials = SourceCredentials()

    assert not credentials.restricted
    assert credentials.connect_address(PREFILL_0) == PREFILL_0


def test_secret_comes_from_the_files_not_from_the_metadata(tmp_path):
    token = generate_auth_token()
    path = _write(tmp_path / "p0", authenticated_address(PREFILL_0, token))
    credentials = SourceCredentials([path])

    assert credentials.restricted
    resolved = credentials.connect_address(PREFILL_0)
    assert split_authenticated_address(resolved) == (PREFILL_0, token)

    # A secret smuggled into the metadata is discarded, never used.
    forged = authenticated_address(PREFILL_0, generate_auth_token())
    assert credentials.connect_address(forged) == resolved


def test_only_configured_sources_are_reachable(tmp_path):
    credentials = SourceCredentials([_write(tmp_path / "p0", PREFILL_0)])

    assert credentials.connect_address(PREFILL_0) == PREFILL_0  # A source without a secret.
    with pytest.raises(UnknownSourceCoordinator):
        credentials.connect_address("tcp://169.254.169.254:80")
    with pytest.raises(UnknownSourceCoordinator):
        credentials.connect_address(PREFILL_1)


def test_files_are_reread_so_a_restarted_source_is_picked_up(tmp_path):
    path = tmp_path / "p0"
    first, second = generate_auth_token(), generate_auth_token()
    credentials = SourceCredentials([_write(path, authenticated_address(PREFILL_0, first))])
    assert split_authenticated_address(credentials.connect_address(PREFILL_0))[1] == first

    _write(path, authenticated_address(PREFILL_0, second))

    assert split_authenticated_address(credentials.connect_address(PREFILL_0))[1] == second


def test_several_sources(tmp_path):
    t0, t1 = generate_auth_token(), generate_auth_token()
    credentials = SourceCredentials(
        [
            _write(tmp_path / "p0", authenticated_address(PREFILL_0, t0)),
            _write(tmp_path / "p1", authenticated_address(PREFILL_1, t1)),
        ]
    )

    assert split_authenticated_address(credentials.connect_address(PREFILL_0))[1] == t0
    assert split_authenticated_address(credentials.connect_address(PREFILL_1))[1] == t1


@pytest.mark.asyncio
async def test_channel_reaches_an_authenticated_source_only_through_the_files(tmp_path):
    token = generate_auth_token()
    context = zmq.asyncio.Context()
    authenticator = start_server_authentication(context, token)
    router = context.socket(zmq.ROUTER)
    require_authentication(router)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    endpoint = f"tcp://127.0.0.1:{port}"

    async def serve():
        while True:
            identity, payload = await router.recv_multipart()
            message = msgpack.unpackb(payload, raw=False)
            if message[0] == Headers.CONNECT.value:
                reply = [Headers.CONNECT_ACK.value, "instance-1"]
                await router.send_multipart([identity, msgpack.packb(reply, use_bin_type=True)])

    server = asyncio.create_task(serve())
    channel_context = zmq.asyncio.Context()
    try:
        # The metadata names only the endpoint. With the file's secret the handshake works...
        good = SourceCredentials(
            [_write(tmp_path / "good", authenticated_address(endpoint, token))]
        )
        channel = HandoffChannel(channel_context, good.connect_address(endpoint))
        with channel.admit():
            assert await asyncio.wait_for(channel.instance_id(), 5) == "instance-1"
        await channel.aclose()

        # ...and without it (a source that is listed with no secret) the source never answers.
        bare = SourceCredentials([_write(tmp_path / "bare", endpoint)])
        channel = HandoffChannel(channel_context, bare.connect_address(endpoint))
        with channel.admit():
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(channel.instance_id(), 1.5)
        await channel.aclose()
    finally:
        server.cancel()
        router.close(linger=0)
        authenticator.stop()
        channel_context.destroy(linger=0)
        context.destroy(linger=0)


def _argv(*extra, role="decode"):
    return [
        "--model",
        "m",
        "--role",
        role,
        "--nproc-per-node",
        "1",
        "--coordinator-host",
        "decode-0",
        *extra,
        "--",
        "--load",
        "/checkpoint",
    ]


def test_cli_collects_repeated_source_files():
    config = parse_args(
        _argv(
            "--source-coordinator-address-file",
            "/run/p0",
            "--source-coordinator-address-file",
            "/run/p1",
        )
    )

    assert config.source_coordinator_address_files == ["/run/p0", "/run/p1"]
    assert parse_args(_argv()).source_coordinator_address_files == []


def test_cli_rejects_source_files_for_other_roles():
    with pytest.raises(SystemExit):
        parse_args(_argv("--source-coordinator-address-file", "/run/p0", role="prefill"))
