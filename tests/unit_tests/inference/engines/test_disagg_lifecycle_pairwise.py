# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real allocator/backpressure and message-level disaggregation regressions."""

import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from threading import Event
from types import SimpleNamespace
from unittest import mock

import msgpack
import pytest
import torch
import torch.distributed as dist
import zmq

from megatron.core.inference.config import KVCacheManagementMode, PrefixCachingEvictionPolicy
from megatron.core.inference.data_parallel_inference_coordinator.handlers import HANDLERS
from megatron.core.inference.disaggregation.coordinator_runtime import DisaggCoordinatorRuntime
from megatron.core.inference.disaggregation.coordinator_setup import NativeDisaggregationConfig
from megatron.core.inference.disaggregation.engine import DisaggDynamicInferenceEngine
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.headers import Headers
from tests.unit_tests.inference.coordinator_test_utils import make_coordinator_direct
from tests.unit_tests.inference.engines import disagg_test_utils
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
                assert torch.equal(allocator.pc_state.block_ref_counts[held], torch.ones_like(held))
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
                engine.context.kv_block_allocator.pc_state.block_ref_counts[owned_blocks],
                torch.ones_like(owned_blocks, dtype=torch.int32),
            )
            assert reply is None
        dist.barrier(group=transport_world)

        if source:
            engine.push_handoff_kv(101, [peer_meta])
            handles = engine._pending_kv_pushes[0][1]
            blocks = engine._pinned_handoff_blocks[101]
            assert (engine.context.kv_block_allocator.pc_state.block_ref_counts[blocks] > 0).all()
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
            assert (
                engine.context.kv_block_allocator.pc_state.block_ref_counts[owned_blocks] == 0
            ).all()
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
            refs = allocator.pc_state.block_ref_counts[blocks].clone()
            assert engine.context.kv_cache_management_mode == KVCacheManagementMode.PERSIST

            engine.suspend()
            assert engine.state == EngineState.SUSPENDED
            assert engine._pinned_handoff_blocks[101] == blocks
            assert torch.equal(allocator.pc_state.block_ref_counts[blocks], refs)
            assert torch.equal(
                engine.context.memory_buffer[:, :, blocks].cpu().view(torch.uint8), state["kv"]
            )

            engine.resume()
            assert engine.state == EngineState.RUNNING
            assert engine._pinned_handoff_blocks[101] == blocks
            assert torch.equal(allocator.pc_state.block_ref_counts[blocks], refs)
            assert torch.equal(
                engine.context.memory_buffer[:, :, blocks].cpu().view(torch.uint8), state["kv"]
            )
            engine.release_handoff_blocks(101)
            engine.release_handoff_blocks(101)
            assert (allocator.pc_state.block_ref_counts[blocks] == 0).all()
        dist.barrier(group=transport_world)
        assert_released(engine)


class _EngineSocket:
    """An engine's coordinator socket: frames the coordinator routed in, replies out."""

    def __init__(self):
        self.inbox, self.outbox = deque(), deque()

    def recv_multipart(self, flags=0):
        if not self.inbox:
            raise zmq.Again()
        return self.inbox.popleft()

    def send(self, frame):
        self.outbox.append([frame])

    def send_multipart(self, frames):
        self.outbox.append(list(frames))


class _LoopbackP2P:
    """Match the NCCL backend's planned sends and receives in post order, in one process."""

    class _Work:
        done = False

        def is_completed(self):
            return self.done

        def wait(self):
            assert self.done, "loopback: a point-to-point operation has no peer"

    def __init__(self):
        self.sends, self.receives = deque(), deque()

    def batch_isend_irecv(self, ops):
        works = []
        for op in ops:
            work = self._Work()
            (self.sends if op.op is dist.isend else self.receives).append((op.tensor, work))
            works.append(work)
        while self.sends and self.receives:
            (source, sent), (destination, received) = self.sends.popleft(), self.receives.popleft()
            assert source.shape == destination.shape
            destination.copy_(source)
            sent.done = received.done = True
        return works


class _NativeDisaggregation:
    """The native coordinator's handlers and runtime, routing between in-process engines."""

    def __init__(self, engines):
        self.engines = engines
        self.client = []
        coordinator = make_coordinator_direct(data_parallel_size=0)
        coordinator.disaggregated_runtime = DisaggCoordinatorRuntime(coordinator)
        coordinator.known_clients = {b"client"}
        coordinator.request_id_to_client_id = {}
        coordinator.request_id_to_client_request_id = {}
        coordinator.client_request_to_request_id = {}
        coordinator.next_request_id = 101
        coordinator.router_socket = SimpleNamespace(send_multipart=self._route)
        self.coordinator = coordinator
        for identity, engine in engines.items():
            role = engine._kv_transfer_role
            engine._disagg_config = NativeDisaggregationConfig(
                role=role, identity=identity, spawn_coordinator=False, coordinator_group=None
            )
            engine.use_coordinator = engine.is_mp_coordinator = True
            engine.socket_for_receiving_requests = _EngineSocket()
            engine.model_parallel_publisher_socket = mock.Mock()
            metadata = [Headers.REGISTER_ROLE.value, role, "nccl"]
            metadata.append(engine._build_instance_transfer_meta())
            HANDLERS[Headers.REGISTER_ROLE](coordinator, identity, metadata, [])
            engine.socket_for_receiving_requests.inbox.clear()  # The registration ACK.

    def _route(self, frames):
        identity, *body = frames
        if identity in self.engines:
            self.engines[identity].socket_for_receiving_requests.inbox.append(body)
        else:
            self.client.append(frames)  # [client identity, metadata, bodies...]

    def submit(self, client_request_id, tokens, params):
        HANDLERS[Headers.SUBMIT_REQUEST](
            self.coordinator,
            b"client",
            [Headers.SUBMIT_REQUEST.value, client_request_id, params.serialize(), None],
            [msgpack.packb(tokens), msgpack.packb([]), msgpack.packb(None), msgpack.packb(None)],
        )

    def run(self, done, *, limit=256):
        """Step each engine as its coordinator loop does and deliver every message."""
        for _ in range(limit):
            if done():
                return
            for identity, engine in self.engines.items():
                engine.schedule_requests()
                if (
                    engine.context.get_active_request_count()
                    or engine.waiting_request_ids
                    or engine.has_admittable_kv_import
                ):
                    engine.step_modern()
                socket = engine.socket_for_receiving_requests
                while socket.outbox:
                    frames = socket.outbox.popleft()
                    metadata = msgpack.unpackb(frames[0], raw=False)
                    HANDLERS[Headers(metadata[0])](self.coordinator, identity, metadata, frames[1:])
        raise AssertionError("The native disaggregation flow did not settle")

    def replies(self, client_request_id):
        return [
            (Headers(header[0]), header, frames[2:])
            for frames in self.client
            for header in [msgpack.unpackb(frames[1], raw=False)]
            if header[1] == client_request_id
        ]


@torch.inference_mode()
def test_impossible_handoff_fails_on_arrival_and_releases_its_source():
    """A handoff that can never fit decode's pool fails once; later handoffs still run.

    Both real engines run in one process behind the native coordinator's real handlers.
    A loopback carries the NCCL backend's own planned operations between them.
    """
    Utils.initialize_model_parallel()
    try:
        config = disagg_config()
        short_tokens = prompt(9)

        def params():
            return sampling(7, detokenize_generations=False)

        weights, expected = collocated_reference(config, short_tokens, params())
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    disagg_test_utils,
                    "StateHandoffDynamicInferenceEngine",
                    DisaggDynamicInferenceEngine,
                )
            )
            prefill = stack.enter_context(real_engine(config, role="prefill", weights=weights))
            buffer = prefill.context.memory_buffer
            block_bytes = buffer[:, :, 0].numel() * buffer.element_size()
            # A decode pool of four usable blocks, smaller than the prefill pool.
            decode_config = disagg_config(context_buffer_size_gb=5.5 * block_bytes / 1024**3)
            decode = stack.enter_context(real_engine(decode_config, role="decode", weights=weights))
            usable = decode.context.kv_block_allocator.pool_size - 1
            assert usable == 4 and prefill.context.kv_block_allocator.pool_size > 2 * usable
            loopback = _LoopbackP2P()
            stack.enter_context(
                mock.patch.object(dist, "batch_isend_irecv", loopback.batch_isend_irecv)
            )
            native = _NativeDisaggregation({b"prefill": prefill, b"decode": decode})
            runtime = native.coordinator.disaggregated_runtime
            block = config.context_block_size_tokens

            # The prompt alone needs five blocks: the import can never start on decode.
            native.submit(1, prompt(5 * block), params())
            native.run(lambda: native.replies(1))
            [(header, reply, _)] = native.replies(1)
            assert header == Headers.REQUEST_ERROR, reply
            assert "the import needs 6 KV blocks, but the pool has 4" in reply[2]
            assert reply[3] is True  # Source-safe: the coordinator released the prefill pins.
            native.run(lambda: not prefill._pinned_handoff_blocks)
            assert not decode._deferred_kv_handoffs and not decode._pending_kv_imports
            assert not runtime.requests and not runtime.terminating_request_ids

            # A later handoff that fits runs at once and matches a collocated engine.
            native.submit(2, short_tokens, params())
            native.run(lambda: native.replies(2))
            [(header, _, bodies)] = native.replies(2)
            assert header == Headers.ENGINE_REPLY
            assert msgpack.unpackb(bodies[0], raw=False)["generated_tokens"] == expected
            assert native.replies(1) == [(Headers.REQUEST_ERROR, reply, [])]
            native.run(lambda: not prefill._pinned_handoff_blocks)
            assert not runtime.requests and not runtime.terminating_request_ids
            assert not runtime.hop1_request_ids
            assert_released(prefill)
            assert_released(decode)
    finally:
        Utils.destroy_model_parallel()
