# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Source-owned recovery uses a real control socket, without a model or database."""

import asyncio
import signal
import sys
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgpack
import pytest
import zmq
import zmq.asyncio

pytest.importorskip("dynamo")

from megatron.core.inference.async_stream import AsyncStream
from megatron.core.inference.data_parallel_inference_coordinator.handlers import HANDLERS
from megatron.core.inference.disaggregation.handoff_ownership import HandoffOwnership
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceRequestError
from megatron.inference.integrations.dynamo.handoff_recovery import (
    HandoffRelease,
    confirm_terminated,
)
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine
from tests.unit_tests.inference.dynamo.test_llm_engine import _config, _Context
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

    def send(identity, frames):
        assert identity == b"prefill-engine"
        assert msgpack.unpackb(frames[0]) == [Headers.RELEASE_KV.value, 7]
        for source in sources:
            source.release_handoff_blocks(7)

    ownership = HandoffOwnership()
    coordinator = SimpleNamespace(
        instance_id="prefill",
        known_clients=set(),
        router_socket=router,
        identities_of_data_parallel_ranks={b"prefill-engine"},
        handoff_ownership=ownership,
        _send_to_engine=send,
    )
    for request_id in (7, 8, 9):
        HANDLERS[Headers.REGISTER_KV](
            coordinator, b"prefill-engine", [Headers.REGISTER_KV.value, request_id], []
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
assert msgpack.unpackb(socket.recv())[-1] is True
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
        assert not ownership.claim(9, "old")
        assert not ownership.claim(7, "replacement")
        assert not ownership.claim(8, "other")
        assert ownership.confirm_terminated(replacement._handoff_owner) == [(8, b"prefill-engine")]
        for source in sources:
            assert source.pinned_handoff_block_count == 0
            assert source.context.kv_block_allocator.block_ref_counts.sum() == 0
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
async def test_rejected_claim_neither_submits_nor_releases_another_owners_handoff():
    engine = MegatronLLMEngine(_config("decode"))
    engine.client = MagicMock()
    engine._claim_remote_handoff = AsyncMock(side_effect=RuntimeError("claim rejected"))
    engine._release_remote_handoff = AsyncMock()
    try:
        with patch(
            "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
            return_value={},
        ):
            with pytest.raises(RuntimeError, match="claim rejected"):
                await anext(engine.generate({"token_ids": [1]}, _Context()))
        await asyncio.gather(*engine._cleanup_tasks)
        engine.client.add_request_with_kv_handoff_streaming.assert_not_called()
        engine._release_remote_handoff.assert_not_awaited()
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
