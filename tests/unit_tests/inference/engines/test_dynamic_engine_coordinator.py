# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mock-based unit tests for the ZMQ coordinator paths in dynamic_engine.py.

The coordinator-mode branches of `DynamicInferenceEngine` (the message-drain
loop, the `Headers` dispatch, the state-machine transitions on SUSPEND/RESUME/
STOP, plus `shutdown()`) only fire when `use_coordinator=True` and there's a
real `DataParallelInferenceCoordinator` driving the sockets. The existing
functional tests in `test_dynamic_engine.py` run engines standalone and never
hit any of this code.

These tests stand up a `DynamicInferenceEngine` skeleton via `__new__`, inject
mock ZMQ sockets and pre-packed msgpack payloads, and exercise the dispatch
and shutdown logic without any real coordinator daemon.
"""

import asyncio
from collections import deque
from unittest.mock import MagicMock

import msgpack
import pytest

from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, EngineState
from megatron.core.inference.headers import Headers, UnknownHeaderError
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
    Status,
)
from megatron.core.inference.sampling_params import SamplingParams


def _make_state_events():
    """Build the {EngineState: asyncio.Event} dict the engine expects."""
    return {
        EngineState.RUNNING: MagicMock(),
        EngineState.PAUSED: MagicMock(),
        EngineState.SUSPENDED: MagicMock(),
        EngineState.RESUMED: MagicMock(),
        EngineState.STOPPED: MagicMock(),
    }


def _make_engine_skeleton(
    *,
    is_mp_coordinator=True,
    state=EngineState.RUNNING,
    pending_signals=None,
    requests=None,
    generation_epoch=None,
):
    """Construct a DynamicInferenceEngine via __new__ with the attributes schedule_requests reads."""
    eng = DynamicInferenceEngine.__new__(DynamicInferenceEngine)
    eng.is_mp_coordinator = is_mp_coordinator
    eng.state = state
    eng._pending_signals = deque(pending_signals or [])
    eng.requests = requests if requests is not None else {}
    eng._generation_epoch = generation_epoch
    eng._state_events = _make_state_events()

    # Sockets (all paths need these even when only one is touched).
    eng.socket_for_receiving_requests = MagicMock(name="recv_sock")
    eng.model_parallel_publisher_socket = MagicMock(name="pub_sock")
    eng.model_parallel_subscriber_socket = MagicMock(name="sub_sock")

    # Mock the helpers schedule_requests calls so we don't need the full engine.
    eng.add_request = MagicMock(name="add_request")
    eng.suspend = MagicMock(name="suspend")
    eng.resume = MagicMock(name="resume")
    return eng


def _pack(*payload):
    return msgpack.packb(list(payload), use_bin_type=True)


def _make_request_entry(request_id=1, prompt_tokens=None, generated_tokens=None):
    """Build a RequestEntry-like object for epoch-stamping tests."""
    sp = SamplingParams(num_tokens_to_generate=5, termination_id=0)
    req = DynamicInferenceRequest(
        request_id=request_id,
        prompt_tokens=(
            prompt_tokens if prompt_tokens is not None else __import__('torch').tensor([1, 2, 3])
        ),
        sampling_params=sp,
        generated_tokens=list(generated_tokens or []),
    )
    record = DynamicInferenceRequestRecord.from_request(req)
    entry = MagicMock()
    entry.record = record
    entry.future = MagicMock()
    return entry, req


class TestScheduleRequestsCoordinatorDrain:

    def test_coordinator_drains_then_broadcasts(self):
        """Rank 0 drains the inbound socket and broadcasts the batch via TP_BROADCAST."""
        import zmq

        eng = _make_engine_skeleton(is_mp_coordinator=True)
        # Two messages then zmq.Again to break the drain loop.
        msg1 = _pack(Headers.SET_GENERATION_EPOCH.value, 5)
        msg2 = _pack(Headers.SET_GENERATION_EPOCH.value, 7)

        calls = [msg1, msg2]

        def fake_recv(*args, **kwargs):
            if calls:
                return calls.pop(0)
            raise zmq.Again()

        eng.socket_for_receiving_requests.recv.side_effect = fake_recv

        n = eng.schedule_requests()

        assert n == 2
        # Multipart broadcast was sent with the TP_BROADCAST header byte + two messages.
        eng.model_parallel_publisher_socket.send_multipart.assert_called_once()
        broadcast_args = eng.model_parallel_publisher_socket.send_multipart.call_args.args[0]
        assert broadcast_args[0] == bytes([Headers.TP_BROADCAST.value])
        assert broadcast_args[1:] == [msg1, msg2]
        # _generation_epoch should be the last SET_GENERATION_EPOCH we saw (7).
        assert eng._generation_epoch == 7

    def test_coordinator_empty_drain_still_broadcasts(self):
        """Even with zero pending messages, rank 0 still sends a TP_BROADCAST (so peers don't block)."""
        import zmq

        eng = _make_engine_skeleton(is_mp_coordinator=True)
        eng.socket_for_receiving_requests.recv.side_effect = zmq.Again()

        n = eng.schedule_requests()

        assert n == 0
        eng.model_parallel_publisher_socket.send_multipart.assert_called_once()
        # Only the TP_BROADCAST header byte, no payloads.
        broadcast_args = eng.model_parallel_publisher_socket.send_multipart.call_args.args[0]
        assert broadcast_args == [bytes([Headers.TP_BROADCAST.value])]


class TestScheduleRequestsSubscriber:

    def test_non_coordinator_consumes_broadcast(self):
        """Non-coordinator ranks read from the subscriber and skip the leading header byte."""
        eng = _make_engine_skeleton(is_mp_coordinator=False)
        msg1 = _pack(Headers.SET_GENERATION_EPOCH.value, 99)
        eng.model_parallel_subscriber_socket.recv_multipart.return_value = [
            bytes([Headers.TP_BROADCAST.value]),
            msg1,
        ]

        n = eng.schedule_requests()

        assert n == 1
        assert eng._generation_epoch == 99
        # Subscriber path does NOT send to the publisher.
        eng.model_parallel_publisher_socket.send_multipart.assert_not_called()


class TestScheduleRequestsHeaderDispatch:

    def _setup_with_message(self, *payload, **engine_kwargs):
        """Build an engine that will see exactly one message containing the given payload."""
        eng = _make_engine_skeleton(is_mp_coordinator=False, **engine_kwargs)
        eng.model_parallel_subscriber_socket.recv_multipart.return_value = [
            bytes([Headers.TP_BROADCAST.value]),
            _pack(*payload),
        ]
        return eng

    def test_submit_request_calls_add_request(self):
        """SUBMIT_REQUEST decodes its payload and invokes engine.add_request."""
        sp = SamplingParams(num_tokens_to_generate=5)
        eng = self._setup_with_message(
            Headers.SUBMIT_REQUEST.value, 1, "hello prompt", sp.serialize()
        )
        eng.schedule_requests()
        eng.add_request.assert_called_once()
        kwargs = eng.add_request.call_args
        assert kwargs.args[0] == 1
        assert kwargs.args[1] == "hello prompt"
        # The third arg should be a SamplingParams (deserialized).
        assert kwargs.args[2].num_tokens_to_generate == 5

    def test_pause_in_running_transitions_to_pausing(self):
        """PAUSE while RUNNING moves the engine to PAUSING and clears the RUNNING event."""
        eng = self._setup_with_message(Headers.PAUSE.value, state=EngineState.RUNNING)
        eng.schedule_requests()
        assert eng.state == EngineState.PAUSING
        eng._state_events[EngineState.RUNNING].clear.assert_called_once()

    def test_pause_in_paused_is_idempotent(self):
        """A PAUSE received while already PAUSED is silently ignored — no state change."""
        eng = self._setup_with_message(Headers.PAUSE.value, state=EngineState.PAUSED)
        eng.schedule_requests()
        assert eng.state == EngineState.PAUSED

    def test_unpause_in_paused_transitions_to_unpausing(self):
        """UNPAUSE while PAUSED moves the engine to UNPAUSING."""
        eng = self._setup_with_message(Headers.UNPAUSE.value, state=EngineState.PAUSED)
        eng.schedule_requests()
        assert eng.state == EngineState.UNPAUSING

    def test_unpause_in_running_asserts(self):
        """UNPAUSE outside PAUSED state triggers an AssertionError."""
        eng = self._setup_with_message(Headers.UNPAUSE.value, state=EngineState.RUNNING)
        with pytest.raises(AssertionError):
            eng.schedule_requests()

    def test_suspend_in_paused_calls_suspend_and_transitions(self):
        """SUSPEND in PAUSED calls engine.suspend() and moves to SUSPENDING."""
        eng = self._setup_with_message(Headers.SUSPEND.value, state=EngineState.PAUSED)
        eng.schedule_requests()
        eng.suspend.assert_called_once()
        assert eng.state == EngineState.SUSPENDING
        eng._state_events[EngineState.RESUMED].clear.assert_called_once()

    def test_suspend_in_running_asserts(self):
        """SUSPEND outside PAUSED state triggers an AssertionError."""
        eng = self._setup_with_message(Headers.SUSPEND.value, state=EngineState.RUNNING)
        with pytest.raises(AssertionError):
            eng.schedule_requests()

    def test_resume_in_suspended_calls_resume_and_transitions(self):
        """RESUME in SUSPENDED calls engine.resume() and moves to RESUMING."""
        eng = self._setup_with_message(Headers.RESUME.value, state=EngineState.SUSPENDED)
        eng.schedule_requests()
        eng.resume.assert_called_once()
        assert eng.state == EngineState.RESUMING
        eng._state_events[EngineState.SUSPENDED].clear.assert_called_once()

    def test_resume_in_running_asserts(self):
        """RESUME outside SUSPENDED state triggers an AssertionError."""
        eng = self._setup_with_message(Headers.RESUME.value, state=EngineState.RUNNING)
        with pytest.raises(AssertionError):
            eng.schedule_requests()

    def test_stop_in_paused_transitions_to_stopping(self):
        """STOP in PAUSED moves the engine to STOPPING."""
        eng = self._setup_with_message(Headers.STOP.value, state=EngineState.PAUSED)
        eng.schedule_requests()
        assert eng.state == EngineState.STOPPING

    def test_stop_in_suspended_clears_suspended_event(self):
        """STOP in SUSPENDED clears the SUSPENDED event and moves to STOPPING."""
        eng = self._setup_with_message(Headers.STOP.value, state=EngineState.SUSPENDED)
        eng.schedule_requests()
        assert eng.state == EngineState.STOPPING
        eng._state_events[EngineState.SUSPENDED].clear.assert_called_once()

    def test_stop_in_running_asserts(self):
        """STOP outside PAUSED/SUSPENDED triggers an AssertionError."""
        eng = self._setup_with_message(Headers.STOP.value, state=EngineState.RUNNING)
        with pytest.raises(AssertionError):
            eng.schedule_requests()

    def test_unknown_header_raises(self):
        """An unrecognised Headers value triggers UnknownHeaderError."""
        # CONNECT is a Headers value but not handled by schedule_requests.
        eng = self._setup_with_message(Headers.CONNECT.value, state=EngineState.PAUSED)
        with pytest.raises(UnknownHeaderError):
            eng.schedule_requests()


class TestScheduleRequestsGenerationEpochStamping:

    def test_epoch_stamps_active_requests(self):
        """SET_GENERATION_EPOCH stamps every active request with new policy/kv-cache epoch boundaries."""
        eng = _make_engine_skeleton(is_mp_coordinator=False)
        entry, req = _make_request_entry(request_id=1)
        eng.requests = {1: entry}
        # Pre-condition: epoch fields are None.
        assert req.policy_epoch is None
        assert req.kv_cache_epoch is None
        eng.model_parallel_subscriber_socket.recv_multipart.return_value = [
            bytes([Headers.TP_BROADCAST.value]),
            _pack(Headers.SET_GENERATION_EPOCH.value, 42),
        ]
        eng.schedule_requests()
        assert eng._generation_epoch == 42
        # The request was stamped with a (0, epoch) boundary.
        assert req.policy_epoch == [(0, 42)]
        assert req.kv_cache_epoch == [(0, 42)]

    def test_epoch_appends_boundary_when_request_already_has_one(self):
        """When a request already has policy_epoch entries, the new boundary is appended."""
        import torch

        eng = _make_engine_skeleton(is_mp_coordinator=False)
        entry, req = _make_request_entry(
            request_id=1, prompt_tokens=torch.tensor([1, 2, 3]), generated_tokens=[10, 11]
        )
        # Pre-set existing epoch history.
        req.policy_epoch = [(0, 1)]
        req.kv_cache_epoch = [(0, 1)]
        eng.requests = {1: entry}
        eng.model_parallel_subscriber_socket.recv_multipart.return_value = [
            bytes([Headers.TP_BROADCAST.value]),
            _pack(Headers.SET_GENERATION_EPOCH.value, 9),
        ]
        eng.schedule_requests()
        assert eng._generation_epoch == 9
        # 5 total tokens (3 prompt + 2 generated), boundary at total - 1 = 4.
        assert req.policy_epoch == [(0, 1), (4, 9)]
        assert req.kv_cache_epoch == [(0, 1), (4, 9)]

    def test_epoch_skips_request_with_no_tokens(self):
        """Requests with zero total tokens are not stamped (no boundary written)."""
        import torch

        eng = _make_engine_skeleton(is_mp_coordinator=False)
        entry, req = _make_request_entry(
            request_id=1, prompt_tokens=torch.tensor([], dtype=torch.long)
        )
        eng.requests = {1: entry}
        eng.model_parallel_subscriber_socket.recv_multipart.return_value = [
            bytes([Headers.TP_BROADCAST.value]),
            _pack(Headers.SET_GENERATION_EPOCH.value, 3),
        ]
        eng.schedule_requests()
        assert eng._generation_epoch == 3
        assert req.policy_epoch is None
        assert req.kv_cache_epoch is None


class TestScheduleRequestsControlSignalQueueing:

    def test_control_signals_are_processed_one_per_call(self):
        """Multiple control signals in a single batch are processed one at a time across calls."""
        eng = _make_engine_skeleton(is_mp_coordinator=False, state=EngineState.RUNNING)
        # Two PAUSE signals; only the first should cause a transition this call.
        eng.model_parallel_subscriber_socket.recv_multipart.return_value = [
            bytes([Headers.TP_BROADCAST.value]),
            _pack(Headers.PAUSE.value),
            _pack(Headers.PAUSE.value),
        ]
        eng.schedule_requests()
        # First PAUSE handled, second queued for next call.
        assert eng.state == EngineState.PAUSING
        assert len(eng._pending_signals) == 1


class TestShutdown:

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_shutdown_cancels_pending_futures_and_closes_sockets(self):
        """shutdown sets state STOPPED, cancels open futures, sends DISCONNECT, and closes sockets."""
        eng = _make_engine_skeleton(state=EngineState.RUNNING)

        # Two requests: one already-done future (skip cancel), one open (cancel).
        loop = asyncio.new_event_loop()
        try:
            done_fut = loop.create_future()
            done_fut.set_result(None)
            open_fut = loop.create_future()
        finally:
            pass
        e1 = MagicMock()
        e1.future = done_fut
        e2 = MagicMock()
        e2.future = open_fut
        eng.requests = {1: e1, 2: e2}

        # Sockets to be closed by shutdown().
        eng.socket_for_receiving_requests.closed = False
        sock_a = MagicMock()
        sock_b = MagicMock()
        eng.zmq_sockets = [sock_a, sock_b]
        eng.zmq_context = MagicMock()
        eng.zmq_context.closed = False

        self._run(eng.shutdown())

        # State and event signaled.
        assert eng.state == EngineState.STOPPED
        eng._state_events[EngineState.STOPPED].set.assert_called_once()
        # Open future was cancelled; done future was not.
        assert open_fut.cancelled()
        assert not done_fut.cancelled()
        # DISCONNECT was sent on the inbound socket.
        eng.socket_for_receiving_requests.send.assert_called_once()
        # All zmq_sockets closed with linger=0; list is cleared.
        sock_a.close.assert_called_once_with(linger=0)
        sock_b.close.assert_called_once_with(linger=0)
        assert eng.zmq_sockets == []
        # Context terminated.
        eng.zmq_context.term.assert_called_once()
        loop.close()

    def test_shutdown_idempotent_when_socket_already_closed(self):
        """shutdown does not send DISCONNECT when the inbound socket is already closed."""
        eng = _make_engine_skeleton(state=EngineState.PAUSED)
        eng.requests = {}
        eng.socket_for_receiving_requests.closed = True
        eng.zmq_sockets = []
        eng.zmq_context = MagicMock()
        eng.zmq_context.closed = True  # already closed → don't term again

        self._run(eng.shutdown())

        eng.socket_for_receiving_requests.send.assert_not_called()
        eng.zmq_context.term.assert_not_called()
        assert eng.state == EngineState.STOPPED

    def test_shutdown_swallows_send_exception(self):
        """If DISCONNECT send raises (socket closed mid-flight), shutdown still completes."""
        eng = _make_engine_skeleton(state=EngineState.PAUSED)
        eng.requests = {}
        eng.socket_for_receiving_requests.closed = False
        eng.socket_for_receiving_requests.send.side_effect = RuntimeError("socket gone")
        eng.zmq_sockets = []
        eng.zmq_context = MagicMock()
        eng.zmq_context.closed = False

        self._run(eng.shutdown())

        # Still transitions to STOPPED despite the send error.
        assert eng.state == EngineState.STOPPED
        eng.zmq_context.term.assert_called_once()

    def test_shutdown_closes_optional_communicators(self):
        """When expert_parallel / world ZMQ communicators are present, shutdown closes them."""
        eng = _make_engine_skeleton(state=EngineState.PAUSED)
        eng.requests = {}
        eng.socket_for_receiving_requests.closed = True
        eng.zmq_sockets = []
        eng.zmq_context = MagicMock()
        eng.zmq_context.closed = True
        eng.expert_parallel_zmq_communicator = MagicMock()
        eng.world_zmq_communicator = MagicMock()

        self._run(eng.shutdown())

        eng.expert_parallel_zmq_communicator.close.assert_called_once()
        eng.world_zmq_communicator.close.assert_called_once()
