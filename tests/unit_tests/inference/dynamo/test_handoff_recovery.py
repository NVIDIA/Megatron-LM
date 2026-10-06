# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Crash recovery tests use a real journal and release socket, without a model."""

import asyncio
import signal
import sys
import threading
from contextlib import closing
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
from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceRequestError
from megatron.inference.integrations.dynamo.handoff_journal import HandoffJournal, HandoffRelease
from megatron.inference.integrations.dynamo.llm_engine import MegatronLLMEngine
from tests.unit_tests.inference.dynamo.test_llm_engine import _config, _Context
from tests.unit_tests.inference.test_disagg_handoff_lifecycle import _HandoffHarness


def _journal_engine(path, owner):
    config = _config("decode")
    config.engine_launch_mode = "external"
    config.handoff_journal = str(path)
    config.handoff_owner = owner
    return MegatronLLMEngine(config)


@pytest.mark.asyncio
async def test_killed_owner_recovery_restores_prefill_capacity_and_fences_replacement(tmp_path):
    path = tmp_path / "handoffs.db"
    context = zmq.asyncio.Context()
    router = context.socket(zmq.ROUTER)
    port = router.bind_to_random_port("tcp://127.0.0.1")
    address = f"tcp://127.0.0.1:{port}"
    sources = [_HandoffHarness(asyncio.get_running_loop(), hybrid=True) for _ in range(2)]
    for source in sources:
        blocks = source.context.kv_block_allocator.allocate_memory_blocks(1).tolist()
        source._pinned_handoff_blocks[7] = blocks
        source._pinned_handoff_ssm_slots[7] = 3

    def broadcast(metadata):
        assert metadata == [Headers.RELEASE_KV.value, 7]
        for source in sources:
            source.release_handoff_blocks(metadata[1])

    coordinator = SimpleNamespace(
        instance_id="prefill",
        known_clients=set(),
        router_socket=router,
        _broadcast_to_engines=broadcast,
    )

    async def serve():
        while True:
            identity, payload = await router.recv_multipart()
            metadata = msgpack.unpackb(payload, raw=False)
            HANDLERS[Headers(metadata[0])](coordinator, identity, metadata, [])

    server = asyncio.create_task(serve())
    child = None
    replacement = None
    try:
        # Kill a separate owner process after its durable record but before any
        # completion callback. No Python finally block can rescue this request.
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            """
import signal
import sys
from megatron.inference.integrations.dynamo.handoff_journal import HandoffJournal, HandoffRelease
journal = HandoffJournal(sys.argv[1])
journal.start_owner("old")
journal.record("old", HandoffRelease(sys.argv[2], "prefill", 7))
print("recorded", flush=True)
signal.pause()
""",
            str(path),
            address,
            stdout=asyncio.subprocess.PIPE,
        )
        assert await asyncio.wait_for(child.stdout.readline(), timeout=30) == b"recorded\n"
        child.kill()
        assert await asyncio.wait_for(child.wait(), timeout=5) == -signal.SIGKILL

        replacement = _journal_engine(path, "replacement")
        journal = HandoffJournal(str(path))
        # Even a running replacement is not sufficient proof of termination of
        # the old GPU ranks: none of the old source memory may be released yet.
        await replacement._replay_handoffs()
        assert not journal.releasable()
        assert all(source.pinned_handoff_block_count == 1 for source in sources)

        new_receipt = journal.record("replacement", HandoffRelease(address, "prefill", 8))
        with pytest.raises(ValueError, match="Unknown"):
            journal.confirm_terminated("wrong-owner")
        journal.confirm_terminated("old")
        journal.confirm_terminated("old")  # Controller retries are harmless.
        with pytest.raises(RuntimeError, match="terminated"):
            journal.record("old", HandoffRelease(address, "prefill", 9))
        assert [item.request_id for _, item in journal.releasable()] == [7]
        await replacement._replay_handoffs()
        assert not journal.releasable()
        for source in sources:
            assert source.pinned_handoff_block_count == 0
            assert source.context.kv_block_allocator.block_ref_counts.sum() == 0
            assert source.context.mamba_metadata.mamba_state_free_slot_count == 1
            # Capacity is actually usable, not just absent from the pinned map.
            assert source.context.mamba_metadata.allocate_slot() is not None
        journal.mark_source_safe(new_receipt)
        assert [item.request_id for _, item in journal.releasable()] == [8]
    finally:
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        if replacement is not None:
            await replacement.cleanup()
        server.cancel()
        await asyncio.gather(server, return_exceptions=True)
        router.close(linger=0)
        context.term()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_safe", [False, True])
async def test_decode_journals_before_submission_and_preserves_failed_release(
    tmp_path, source_safe
):
    path = tmp_path / "handoffs.db"
    engine = _journal_engine(path, "old")
    journal = engine._handoff_journal
    release = HandoffRelease("tcp://prefill:5000", "prefill", 7)
    stream = AsyncStream(4, MagicMock())
    stream.finish(exception=InferenceRequestError("decode failed", source_safe=source_safe))

    def submit(*args):
        # A second connection must see the committed record before submission.
        with closing(journal._connect()) as db:
            assert db.execute("SELECT request_id FROM handoffs").fetchall() == [(7,)]
        assert not journal.releasable()
        return stream

    engine.client = MagicMock(
        add_request_with_kv_handoff_streaming=submit,
        abort_request_and_wait=AsyncMock(return_value=False),
    )
    engine._release_handoff_from_meta_async = AsyncMock(side_effect=TimeoutError("source down"))
    try:
        with patch(
            "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
            return_value={"disaggregated_params": {"release": asdict(release)}},
        ):
            with pytest.raises(InferenceRequestError, match="decode failed"):
                _ = [chunk async for chunk in engine.generate({"token_ids": [1]}, _Context())]
        await asyncio.gather(*engine._cleanup_tasks, return_exceptions=True)
        assert bool(journal.releasable()) == source_safe
        assert engine._release_handoff_from_meta_async.await_count == int(source_safe)
        # Parent shutdown must not attest termination of externally managed ranks.
        await engine.cleanup()
        replacement = _journal_engine(path, "new")
        replacement._release_remote_handoff = AsyncMock(side_effect=TimeoutError("source down"))
        await replacement._replay_handoffs()
        assert bool(journal.releasable()) == source_safe
        replacement._release_remote_handoff = AsyncMock()
        await replacement._replay_handoffs()
        assert replacement._release_remote_handoff.await_count == int(source_safe)
        assert not journal.releasable()
        await replacement.cleanup()
    finally:
        await engine.cleanup()


@pytest.mark.asyncio
async def test_journal_write_failure_prevents_decode_submission(tmp_path):
    engine = _journal_engine(tmp_path / "handoffs.db", "owner")
    engine.client = MagicMock()
    engine._release_remote_handoff = AsyncMock()
    release = HandoffRelease("tcp://prefill:5000", "prefill", 7)
    try:
        with (
            patch.object(engine._handoff_journal, "record", side_effect=OSError("disk full")),
            patch(
                "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
                return_value={"disaggregated_params": {"release": asdict(release)}},
            ),
        ):
            with pytest.raises(OSError, match="disk full"):
                await anext(engine.generate({"token_ids": [1]}, _Context()))
        engine.client.add_request_with_kv_handoff_streaming.assert_not_called()
        await asyncio.gather(*engine._cleanup_tasks)
        engine._release_remote_handoff.assert_awaited_once_with(
            release.coordinator_addr, release.request_id, release.coordinator_instance_id
        )
    finally:
        await engine.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation", ["consumer", "abort"])
async def test_cancellation_during_journal_write_cleans_up_without_submitting(
    tmp_path, cancellation
):
    engine = _journal_engine(tmp_path / "handoffs.db", "owner")
    engine.client = MagicMock()
    engine._release_remote_handoff = AsyncMock()
    release = HandoffRelease("tcp://prefill:5000", "prefill", 7)
    started = threading.Event()
    proceed = threading.Event()
    record = engine._handoff_journal.record

    def delayed_record(*args):
        started.set()
        assert proceed.wait(timeout=5)
        return record(*args)

    try:
        with (
            patch.object(engine._handoff_journal, "record", side_effect=delayed_record),
            patch(
                "megatron.inference.integrations.dynamo.llm_engine.require_prefill_result",
                return_value={"disaggregated_params": {"release": asdict(release)}},
            ),
        ):
            generation = engine.generate({"token_ids": [1]}, _Context())
            consumer = asyncio.create_task(anext(generation))
            assert await asyncio.to_thread(started.wait, 2)
            if cancellation == "abort":
                await engine.abort(_Context())
            else:
                consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await consumer
            engine.client.add_request_with_kv_handoff_streaming.assert_not_called()
            engine._release_remote_handoff.assert_not_awaited()
            proceed.set()
            await asyncio.wait_for(asyncio.gather(*engine._cleanup_tasks), timeout=5)
        engine._release_remote_handoff.assert_awaited_once_with(
            release.coordinator_addr, release.request_id, release.coordinator_instance_id
        )
        with closing(engine._handoff_journal._connect()) as db:
            assert db.execute("SELECT COUNT(*) FROM handoffs").fetchone()[0] == 0
    finally:
        proceed.set()
        await engine.cleanup()
