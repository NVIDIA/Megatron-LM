# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real allocator/backpressure and message-level disaggregation regressions."""

import uuid

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


def deliver_abort(engine, request_id):
    """Deliver one ABORT through the production ZMQ receive handler."""

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
        sender.send_multipart(
            [msgpack.packb([Headers.ABORT_REQUEST.value, request_id], use_bin_type=True)]
        )
        assert receiver.poll(1000)
        assert engine.schedule_requests() == 1
        reply = sender.recv_multipart() if sender.poll(1000) else None
    finally:
        for socket in (receiver, sender, publisher):
            socket.close(linger=0)
        context.term()
    return reply


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
