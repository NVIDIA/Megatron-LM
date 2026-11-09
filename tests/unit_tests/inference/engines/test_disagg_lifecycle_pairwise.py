# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real allocator/backpressure and message-level disaggregation regressions."""

import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from threading import Event
from unittest import mock

import msgpack
import pytest
import torch
import torch.distributed as dist
import zmq

from megatron.core.inference.config import KVCacheManagementMode, PrefixCachingEvictionPolicy
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.engines.disagg_test_utils import (
    ForwardWitness,
    _enqueue_decode_handoff,
    assert_import_equal,
    assert_released,
    collocated_reference,
    decode_peer_meta,
    deferred_pulls,
    disagg_config,
    exchange,
    prompt,
    push_handoff,
    real_engine,
    run_to_completion,
    sampling,
    snapshot_source,
)
from tests.unit_tests.inference.engines.test_disagg_pairwise import transport_world  # noqa: F401
from tests.unit_tests.test_utilities import Utils


@contextmanager
def abort_channel(engine):
    """Keep the real coordinator socket alive through delayed cancellation."""
    context = zmq.Context()
    receiver, sender, publisher = (
        context.socket(zmq.PAIR),
        context.socket(zmq.PAIR),
        context.socket(zmq.PUB),
    )
    address = f"inproc://abort-handoff-{uuid.uuid4().hex}"
    receiver.bind(address)
    sender.connect(address)
    engine.is_mp_coordinator = True
    engine.use_coordinator = engine.local_metadata_ledger_enabled = True
    engine.socket_for_receiving_requests = receiver
    engine.model_parallel_publisher_socket = publisher
    try:
        yield sender
    finally:
        for socket in (receiver, sender, publisher):
            socket.close(linger=0)
        context.term()


def send_abort(engine, sender, request_id):
    sender.send_multipart([msgpack.packb([Headers.ABORT_REQUEST.value, request_id])])
    assert engine.socket_for_receiving_requests.poll(1000)
    assert engine.schedule_requests() == 1
    return sender.recv_multipart() if sender.poll(1000) else None


def deliver_abort(engine, request_id):
    """Deliver one ABORT through the production ZMQ receive handler."""
    with abort_channel(engine) as sender:
        return send_abort(engine, sender, request_id)


@pytest.mark.parametrize("policy", list(PrefixCachingEvictionPolicy))
@torch.inference_mode()
def test_real_handoff_capacity_fifo(transport_world, policy):
    config = disagg_config(prefix_caching_eviction_policy=policy)
    tokens = prompt(33)
    weights, expected = collocated_reference(config, tokens, sampling())
    source = dist.get_rank() % 2 == 0
    with real_engine(config, role="prefill" if source else "decode", weights=weights) as engine:
        metadata = state = None
        if source:
            result = run_to_completion(
                engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
            )
            metadata, state = snapshot_source(engine, result)
        peer = exchange((metadata, state) if source else None, transport_world)
        allocator = engine.context.kv_block_allocator
        held = None
        if not source:
            metadata, state = peer
            # Consume actual capacity. No allocator predicate or backend is mocked.
            held = allocator.allocate_memory_blocks(allocator.pool_avail).clone()
            futures = [
                engine.add_request_with_kv_handoff(
                    rid, tokens, sampling(), metadata["kv_meta"], metadata["block_ids"]
                )
                for rid in (101, 102)
            ]
            assert [item.request_id for item in engine._deferred_kv_handoffs] == [101, 102]
            assert not engine._pending_kv_imports
            allocator.release_memory_blocks(held[:2])
            assert engine._drain_deferred_kv_handoffs() == 0
            allocator.release_memory_blocks(held[2:3])
            with deferred_pulls(engine):
                assert engine._drain_deferred_kv_handoffs() == 1
            assert [item.request_id for item in engine._deferred_kv_handoffs] == [102]
        for index, rid in enumerate((101, 102)):
            pending = None
            if not source:
                if index:
                    with deferred_pulls(engine):
                        assert engine._drain_deferred_kv_handoffs() == 1
                assert len(engine._pending_kv_imports) == 1
                pending = engine._pending_kv_imports[0]
                assert pending.request_id == rid
                cached = 2 if index and policy == PrefixCachingEvictionPolicy.LRU else 0
                assert pending.cached_prefix_block_count == cached
            peer = exchange(None if source else decode_peer_meta(engine, pending), transport_world)
            if source:
                push_handoff(engine, 101, [peer])
            handles = (
                [handle for _, group in engine._pending_kv_pushes for handle in group]
                if source
                else engine._pending_transfer_handles(pending)
            )
            for handle in handles:
                handle.wait()
                assert handle.poll()
            if not source:
                assert_import_equal(engine, pending, state, len(tokens))
                assert engine._poll_pending_kv_imports() == 1
                assert engine._admit_pending_kv_imports() == 1
                with ForwardWitness(engine, rid) as witness:
                    result = run_to_completion(engine, futures[index])
                assert result.generated_tokens == expected
                assert witness.steps and all(not step[2] for step in witness.steps)
            else:
                assert engine._poll_pending_kv_pushes() == 1
            dist.barrier(group=transport_world)
        if source:
            engine.release_handoff_blocks(101)
        else:
            allocator.release_memory_blocks(held[3:])
        assert_released(engine)


@torch.inference_mode()
def test_abort_capacity_queued_handoff_resolves_without_transfer():
    """A delivered ABORT must reach handoffs not yet present in engine.requests.

    This is a focused negative/lifecycle regression, not positive transfer credit.
    """
    Utils.initialize_model_parallel()
    try:
        config = disagg_config()
        with real_engine(config, role="decode") as engine:
            allocator = engine.context.kv_block_allocator
            held = allocator.allocate_memory_blocks(allocator.pool_avail).clone()
            params = sampling(detokenize_generations=False)
            future = engine.add_request_with_kv_handoff(
                101, prompt(33), params, {"resume_tokens": [9]}, [1, 2, 3]
            )
            assert not engine.requests and not engine._pending_kv_imports
            assert [item.request_id for item in engine._deferred_kv_handoffs] == [101]
            try:
                assert not future.done()
                with ForwardWitness(engine) as witness:
                    reply = deliver_abort(engine, 101)
                # Assert while capacity is still exhausted, before teardown can
                # cancel anything or release the independently held blocks.
                resolved = future.cancelled()
                retired = not engine._deferred_kv_handoffs and not engine._pending_kv_imports
                assert not witness.steps
                assert torch.equal(allocator.block_ref_counts[held], torch.ones_like(held))
                assert resolved, "Delivered ABORT left a capacity-queued handoff future unresolved"
                assert retired, "Aborted handoff remained eligible for a later real transfer"
                assert not (
                    engine.requests or engine.failed_request_ids or engine.local_metadata_ledger
                )
                assert reply is not None, "Delivered ABORT did not publish REQUEST_ABORTED"
                assert len(reply) == 1
                assert msgpack.unpackb(reply[0], raw=False) == [
                    Headers.REQUEST_ABORTED.value,
                    101,
                    True,
                ]
                assert deliver_abort(engine, 101) is None
            finally:
                allocator.release_memory_blocks(held)
                engine._reset_pending_kv_imports()
    finally:
        Utils.destroy_model_parallel()


@torch.inference_mode()
def test_abort_inflight_handoff_retains_storage_until_real_nccl_settles(transport_world):
    tokens = prompt(33)
    source = dist.get_rank() % 2 == 0
    config = disagg_config()
    with (
        real_engine(config, role="prefill" if source else "decode") as engine,
        ExitStack() as stack,
    ):
        metadata = state = None
        if source:
            result = run_to_completion(
                engine, engine.add_request(101, tokens, sampling(1, do_kv_handoff=True))
            )
            metadata, state = snapshot_source(engine, result)
        peer = exchange((metadata, state) if source else None, transport_world)
        if not source:
            metadata, state = peer

        pending = future = owned_blocks = None
        if not source:
            future = _enqueue_decode_handoff(engine, metadata, tokens, sampling())
            pending = engine._pending_kv_imports[0]
            owned_blocks = torch.tensor(
                pending.local_blocks + pending.continuation_blocks, dtype=torch.int64
            )
            peer_meta = decode_peer_meta(engine, pending)
            receive_started = Event()
            batch = dist.batch_isend_irecv

            def receive(ops):
                assert ops and all(op.op is dist.irecv for op in ops)
                receive_started.set()
                return batch(ops)

            def post_receive():
                torch.cuda.set_device(dist.get_rank())
                with mock.patch.object(dist, "batch_isend_irecv", side_effect=receive):
                    return pending.handle._start()

            receive_poster = ThreadPoolExecutor(max_workers=1)
            receive_post = receive_poster.submit(post_receive)
            assert receive_started.wait(timeout=5) and not receive_post.done()
        else:
            peer_meta = None
        peer_meta = exchange(peer_meta, transport_world)

        witness = None
        if not source:
            assert not future.done()
            sender = stack.enter_context(abort_channel(engine))
            with ForwardWitness(engine) as witness:
                reply = send_abort(engine, sender, 101)
            assert not future.done() and pending.cancel_requested
            assert list(engine._pending_kv_imports) == [pending]
            assert not engine._quarantined_kv_imports
            assert torch.equal(
                engine.context.kv_block_allocator.block_ref_counts[owned_blocks],
                torch.ones_like(owned_blocks, dtype=torch.int32),
            )
            assert reply is None
        dist.barrier(group=transport_world)

        if source:
            engine.push_handoff_kv(101, [peer_meta])
            handles = engine._pending_kv_pushes[0][1]
            blocks = engine._pinned_handoff_blocks[101]
            assert (engine.context.kv_block_allocator.block_ref_counts[blocks] > 0).all()
        else:
            assert receive_post.result(timeout=30) is pending.handle.real_handle
            receive_poster.shutdown()
            handles = engine._pending_transfer_handles(pending)
        for handle in handles:
            handle.wait()
            assert handle.poll()

        if source:
            assert engine._poll_pending_kv_pushes() == 1
            engine.release_handoff_blocks(101)
            engine.release_handoff_blocks(101)
        else:
            assert_import_equal(engine, pending, state, len(tokens))
            assert engine._poll_pending_kv_imports() == 1
            assert engine._admit_pending_kv_imports() == 0
            assert future.cancelled() and not engine._pending_kv_imports
            assert not engine._quarantined_kv_imports
            assert sender.poll(1000)
            assert msgpack.unpackb(sender.recv(), raw=False) == [
                Headers.REQUEST_ABORTED.value,
                101,
                True,
            ]
            assert send_abort(engine, sender, 101) is None
            assert (engine.context.kv_block_allocator.block_ref_counts[owned_blocks] == 0).all()
            assert engine._poll_pending_kv_imports() == 0
            assert not witness.steps and not engine._handoff_completion_notifications
        assert_released(engine)


@torch.inference_mode()
def test_prefill_handoff_pin_survives_persist_suspend_until_release(transport_world):
    source = dist.get_rank() % 2 == 0
    with real_engine(disagg_config(), role="prefill" if source else "decode") as engine:
        if source:
            result = run_to_completion(
                engine, engine.add_request(101, prompt(33), sampling(1, do_kv_handoff=True))
            )
            metadata, state = snapshot_source(engine, result)
            state["kv"] = state["kv"].view(torch.uint8)
            blocks = metadata["block_ids"]
            allocator = engine.context.kv_block_allocator
            refs = allocator.block_ref_counts[blocks].clone()
            assert engine.context.kv_cache_management_mode == KVCacheManagementMode.PERSIST

            engine.suspend()
            assert engine.state == EngineState.SUSPENDED
            assert engine._pinned_handoff_blocks[101] == blocks
            assert torch.equal(allocator.block_ref_counts[blocks], refs)
            assert torch.equal(
                engine.context.memory_buffer[:, :, blocks].cpu().view(torch.uint8), state["kv"]
            )

            engine.resume()
            assert engine.state == EngineState.RUNNING
            assert engine._pinned_handoff_blocks[101] == blocks
            assert torch.equal(allocator.block_ref_counts[blocks], refs)
            assert torch.equal(
                engine.context.memory_buffer[:, :, blocks].cpu().view(torch.uint8), state["kv"]
            )
            engine.release_handoff_blocks(101)
            engine.release_handoff_blocks(101)
            assert (allocator.block_ref_counts[blocks] == 0).all()
        dist.barrier(group=transport_world)
        assert_released(engine)
