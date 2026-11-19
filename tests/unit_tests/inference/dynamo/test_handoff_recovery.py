# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-owned recovery uses a real control socket, without a model or database."""

import asyncio
import logging
import os
import signal
import stat
import sys
import threading
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
import zmq
import zmq.asyncio

pytest.importorskip("dynamo")

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.data_parallel_inference_coordinator.auth import (
    authenticated_address,
    generate_auth_token,
    require_authentication,
    start_server_authentication,
)
from megatron.core.inference.data_parallel_inference_coordinator.handlers import HANDLERS
from megatron.core.inference.disaggregation.handoff_ownership import (
    HandoffClaimStatus,
    HandoffOwnership,
)
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceRequestError
from megatron.inference.integrations.dynamo import handoff_recovery
from megatron.inference.integrations.dynamo.handoff_recovery import (
    HandoffRelease,
    confirm_terminated,
)
from megatron.inference.integrations.dynamo.llm_engine import (
    MegatronLLMEngine,
    _write_private_file,
)
from tests.unit_tests.inference.dynamo.test_llm_engine import _config, _Context, _endpoint
from tests.unit_tests.inference.test_disagg_handoff_lifecycle import _HandoffHarness


@pytest.mark.asyncio
async def test_killed_owner_recovery_restores_prefill_capacity_and_fences_replacement():
    context = zmq.asyncio.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    address = f"tcp://127.0.0.1:{port}"
    sources = [_HandoffHarness(asyncio.get_running_loop(), hybrid=True) for _ in range(2)]
    for source in sources:
        source._pinned_handoff_blocks[7] = source.context.kv_block_allocator.allocate_memory_blocks(
            1
        ).tolist()
        source._pinned_handoff_ssm_slots[7] = 3

    def send(identity, frames, *, remove_unreachable):
        assert not remove_unreachable
        assert identity == b"prefill-engine"
        assert msgpack.unpackb(frames[0]) == [Headers.RELEASE_KV.value, 7]
        for source in sources:
            source.release_handoff_blocks(7)
        return True

    ownership = HandoffOwnership()
    coordinator = SimpleNamespace(
        instance_id="prefill",
        known_clients=set(),
        router_socket=router,
        _send_to_client=router.send_multipart,
        identities_of_data_parallel_ranks={b"prefill-engine"},
        handoff_ownership=ownership,
        is_disaggregated_inference=lambda: False,
        _send_to_engine=send,
    )
    for request_id in (7, 8, 9):
        HANDLERS[Headers.REGISTER_KV](
            coordinator, b"prefill-engine", [Headers.REGISTER_KV.value, request_id, True], []
        )

    async def serve():
        while True:
            identity, payload = await router.recv_multipart()
            metadata = msgpack.unpackb(payload, raw=False)
            HANDLERS[Headers(metadata[0])](coordinator, identity, metadata, [])

    server = asyncio.create_task(serve())
    child = None
    replacement = MegatronLLMEngine(_config("decode"))
    try:
        # No Python finally block can rescue this process after its claim.
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            """
import msgpack, signal, sys, zmq
socket = zmq.Context.instance().socket(zmq.DEALER)
socket.connect(sys.argv[1])
socket.send(msgpack.packb([int(sys.argv[2])]))
instance = msgpack.unpackb(socket.recv())[1]
socket.send(msgpack.packb([int(sys.argv[3]), 7, instance, "old"]))
assert msgpack.unpackb(socket.recv())[-1] == "accepted"
print("claimed", flush=True)
signal.pause()
""",
            address,
            str(Headers.CONNECT.value),
            str(Headers.CLAIM_KV.value),
            stdout=asyncio.subprocess.PIPE,
        )
        assert await asyncio.wait_for(child.stdout.readline(), 30) == b"claimed\n"
        child.kill()
        assert await asyncio.wait_for(child.wait(), 5) == -signal.SIGKILL

        live = asdict(HandoffRelease(address, "prefill", 8))
        await replacement._claim_remote_handoff(live)
        # Even the same owner cannot start two reads with one release lifetime.
        with pytest.raises(RuntimeError, match="acknowledgement"):
            await replacement._claim_remote_handoff(live)
        assert all(source.pinned_handoff_block_count == 1 for source in sources)
        # Adapter death alone does not prove every remote GPU rank is gone.
        with pytest.raises(RuntimeError, match="acknowledgement"):
            await replacement._claim_remote_handoff(asdict(HandoffRelease(address, "prefill", 7)))
        await confirm_terminated(address, "old")
        await confirm_terminated(address, "old")
        assert ownership.claim(9, "old") is HandoffClaimStatus.REJECTED
        assert ownership.claim(7, "replacement") is HandoffClaimStatus.REJECTED
        assert ownership.claim(8, "other") is HandoffClaimStatus.REJECTED
        assert ownership.confirm_terminated(replacement._handoff_owner) == [(8, b"prefill-engine")]
        for source in sources:
            assert source.pinned_handoff_block_count == 0
            assert source.context.kv_block_allocator.pc_state.block_ref_counts.sum() == 0
            assert source.context.mamba_metadata.mamba_state_free_slot_count == 1
            assert source.context.mamba_metadata.allocate_slot() is not None
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        await replacement.cleanup()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
        router.close(linger=0)
        context.term()


@pytest.mark.asyncio
async def test_abort_deadline_does_not_cancel_eventual_safe_release():
    engine = MegatronLLMEngine(_config("decode"))
    engine.config.drain_timeout = 0.001
    ack = asyncio.get_running_loop().create_future()
    engine.client = SimpleNamespace(abort_request_and_wait=MagicMock(return_value=ack))
    engine._release_completed_handoff = AsyncMock()
    engine._request_ids["dynamo-request"] = 4
    release = asdict(HandoffRelease("tcp://prefill:5000", "prefill", 7))
    engine._schedule_cleanup(engine._release_aborted_handoff(4, release))
    with pytest.raises(TimeoutError):
        await engine.abort(_Context())
    assert not ack.done()
    engine._release_completed_handoff.assert_not_awaited()
    ack.set_result(True)
    await asyncio.gather(*engine._cleanup_tasks)
    engine._release_completed_handoff.assert_awaited_once_with(release)


@pytest.mark.asyncio
@pytest.mark.parametrize("source_safe", [False, True])
async def test_decode_claims_before_submission_and_retries_only_safe_releases(source_safe):
    engine = MegatronLLMEngine(_config("decode"))
    release = HandoffRelease("tcp://prefill:5000", "prefill", 7)
    stream = AsyncStream(4, MagicMock())
    stream.finish(exception=InferenceRequestError("decode failed", source_safe=source_safe))
    engine._claim_remote_handoff = AsyncMock()

    def submit(*args):
        engine._claim_remote_handoff.assert_awaited_once_with(asdict(release))
        return stream

    engine.client = MagicMock(
        add_request_with_kv_handoff_streaming=submit,
        abort_request_and_wait=AsyncMock(return_value=False),
    )
    engine._release_remote_handoff = AsyncMock(side_effect=TimeoutError("source down"))
    try:
        with patch(
            "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
            return_value={"disaggregated_params": {"release": asdict(release)}},
        ):
            with pytest.raises(InferenceRequestError, match="decode failed"):
                _ = [chunk async for chunk in engine.generate({"token_ids": [1]}, _Context())]
        await asyncio.gather(*engine._cleanup_tasks, return_exceptions=True)
        assert engine._pending_releases == ({release} if source_safe else set())
        engine._release_remote_handoff = AsyncMock()
        await engine._replay_handoffs()
        assert engine._release_remote_handoff.await_count == int(source_safe)
        assert not engine._pending_releases
    finally:
        await engine.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["consumer", "abort"])
async def test_cancellation_during_claim_cleans_up_without_submitting(cancellation):
    engine = MegatronLLMEngine(_config("decode"))
    engine.client = MagicMock()
    engine._release_remote_handoff = AsyncMock()
    release = HandoffRelease("tcp://prefill:5000", "prefill", 7)
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def claim(_metadata):
        started.set()
        await proceed.wait()

    engine._claim_remote_handoff = claim
    try:
        with patch(
            "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
            return_value={"disaggregated_params": {"release": asdict(release)}},
        ):
            consumer = asyncio.create_task(anext(engine.generate({"token_ids": [1]}, _Context())))
            await asyncio.wait_for(started.wait(), 2)
            if cancellation == "abort":
                await engine.abort(_Context())
            else:
                consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            engine.client.add_request_with_kv_handoff_streaming.assert_not_called()
            engine._release_remote_handoff.assert_not_awaited()
            proceed.set()
            await asyncio.wait_for(asyncio.gather(*engine._cleanup_tasks), 5)
        engine._release_remote_handoff.assert_awaited_once_with(
            release.coordinator_addr, release.request_id, release.coordinator_instance_id
        )
    finally:
        proceed.set()
        await engine.cleanup()


@pytest.mark.asyncio
async def test_recovery_hook_authenticates_to_an_authenticated_source_coordinator():
    """The documented invocation passes the full launch address; a bare address is refused."""
    token = generate_auth_token()
    context = zmq.Context()
    authenticator = start_server_authentication(context, token)
    router = context.socket(zmq.ROUTER)
    require_authentication(router)
    endpoint = f"tcp://127.0.0.1:{router.bind_to_random_port('tcp://127.0.0.1')}"

    ownership = HandoffOwnership()
    for request_id in (7, 8):
        ownership.offer(request_id, b"engine", require_claim=True)
    assert ownership.claim(7, "old") == HandoffClaimStatus.ACCEPTED
    assert ownership.claim(8, "replacement") == HandoffClaimStatus.ACCEPTED
    released = []

    def send(identity, frames, *, remove_unreachable):
        released.append(msgpack.unpackb(frames[0])[1])
        return True

    coordinator = SimpleNamespace(
        instance_id="prefill",
        known_clients=set(),
        router_socket=router,
        _send_to_client=router.send_multipart,
        identities_of_data_parallel_ranks={b"engine"},
        handoff_ownership=ownership,
        is_disaggregated_inference=lambda: False,
        _send_to_engine=send,
    )
    stop = threading.Event()

    def serve():
        while not stop.is_set():
            if router.poll(50):
                identity, payload = router.recv_multipart()
                metadata = msgpack.unpackb(payload, raw=False)
                HANDLERS[Headers(metadata[0])](coordinator, identity, metadata, [])

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    try:
        # Without the launch secret the coordinator refuses the connection: the
        # supervisor's attestation never arrives and nothing is released.
        with pytest.raises(TimeoutError):
            await confirm_terminated(endpoint, "old", timeout=1.0)
        assert released == []

        await confirm_terminated(authenticated_address(endpoint, token), "old", timeout=15.0)
        assert released == [7], "the terminated owner's source allocation was not released"
        assert ownership.source_engine(7) is None
        assert ownership.source_engine(8) == b"engine", "a replacement's handoff must be intact"
    finally:
        stop.set()
        server.join(5)
        router.close(linger=0)
        authenticator.stop()
        context.term()


def test_recovery_cli_reads_authenticated_addresses_from_files(tmp_path, monkeypatch):
    """Files keep the launch secret out of the supervisor's command line."""
    address = authenticated_address("tcp://prefill-0:5000", generate_auth_token())
    address_file = tmp_path / "prefill-0.coordinator"
    address_file.write_text(address + "\n")
    confirm = AsyncMock()
    monkeypatch.setattr(handoff_recovery, "confirm_terminated", confirm)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "handoff_recovery",
            "--coordinator-address-file",
            str(address_file),
            "--confirm-terminated-owner",
            "old",
        ],
    )

    handoff_recovery.main()

    confirm.assert_awaited_once_with(address, "old")
    assert "#auth=" not in " ".join(sys.argv)


def test_recovery_cli_requires_a_source_address(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["handoff_recovery", "--confirm-terminated-owner", "old"]
    )
    with pytest.raises(SystemExit):
        handoff_recovery.main()


def test_private_file_is_owner_only_and_replaced_atomically(tmp_path):
    path = tmp_path / "nested" / "prefill.coordinator"
    _write_private_file(path, "first\n")
    assert path.read_text() == "first\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    _write_private_file(path, "second\n")
    assert path.read_text() == "second\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert os.listdir(path.parent) == [path.name], "no temporary file may be left behind"


@pytest.mark.asyncio
async def test_failed_release_replay_does_not_log_the_launch_secret(caplog):
    """A retained release is logged on failure; its address is a credential."""
    token = generate_auth_token()
    address = authenticated_address("tcp://prefill:5000", token)
    release = HandoffRelease(address, "prefill-instance", 7)
    assert token.hex() not in repr(release) and "#auth=" not in repr(release)
    assert token.hex() not in repr(_endpoint(address))

    engine = MegatronLLMEngine(_config("decode"))
    engine._pending_releases.add(release)
    engine._release_remote_handoff = AsyncMock(side_effect=TimeoutError("source down"))
    try:
        with caplog.at_level(logging.ERROR):
            await engine._replay_handoffs()

        assert "Retaining handoff" in caplog.text
        assert "prefill-instance" in caplog.text and "request_id=7" in caplog.text
        assert token.hex() not in caplog.text and "#auth=" not in caplog.text
        # The release is still retried with the complete address, and serializes in full.
        assert engine._pending_releases == {release}
        assert asdict(release)["coordinator_addr"] == address

        engine._release_remote_handoff = AsyncMock()
        await engine._replay_handoffs()
        engine._release_remote_handoff.assert_awaited_once_with(address, 7, "prefill-instance")
        assert not engine._pending_releases
    finally:
        await engine.cleanup()
