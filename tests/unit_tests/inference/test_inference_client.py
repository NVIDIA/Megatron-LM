# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import logging
from unittest.mock import MagicMock, patch

import msgpack
import pytest

from megatron.core.inference.headers import Headers
from megatron.core.inference.inference_client import InferenceClient
from megatron.core.inference.sampling_params import SamplingParams


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_client(deserialize: bool = False):
    """Build an InferenceClient with a mocked zmq Context/Socket."""
    fake_socket = MagicMock(name="zmq_socket")
    fake_context = MagicMock(name="zmq_context")
    fake_context.socket.return_value = fake_socket
    with patch("megatron.core.inference.inference_client.zmq.Context", return_value=fake_context):
        client = InferenceClient("tcp://127.0.0.1:5555", deserialize=deserialize)
    return client, fake_context, fake_socket


class TestInferenceClient:

    def test_init_connects_socket(self):
        """__init__ creates a DEALER socket and connects to the supplied address."""
        client, fake_context, fake_socket = _make_client()
        fake_socket.connect.assert_called_once_with("tcp://127.0.0.1:5555")
        # SNDHWM and RCVHWM should be set to 0.
        setsockopt_calls = fake_socket.setsockopt.call_args_list
        opts = {call.args[0]: call.args[1] for call in setsockopt_calls}
        import zmq

        assert opts[zmq.SNDHWM] == 0
        assert opts[zmq.RCVHWM] == 0
        assert client.next_request_id == 0
        assert client.completion_futures == {}
        assert client.deserialize is False

    def test_init_with_deserialize_flag(self):
        """deserialize=True is preserved on the client."""
        client, _, _ = _make_client(deserialize=True)
        assert client.deserialize is True

    def test_add_request_increments_id_and_sends(self):
        """Each add_request sends a SUBMIT_REQUEST payload and increments the id counter."""

        async def scenario():
            client, _, fake_socket = _make_client()
            sp = SamplingParams(temperature=0.5)
            f1 = client.add_request("hello", sp)
            f2 = client.add_request("world", sp)
            return client, fake_socket, f1, f2

        client, fake_socket, f1, f2 = _run(scenario())
        assert client.next_request_id == 2
        assert isinstance(f1, asyncio.Future)
        assert isinstance(f2, asyncio.Future)
        # Each call sent exactly one message.
        assert fake_socket.send.call_count == 2
        # Decode the first sent payload and check its structure.
        sent_bytes = fake_socket.send.call_args_list[0].args[0]
        payload = msgpack.unpackb(sent_bytes, raw=False)
        assert payload[0] == Headers.SUBMIT_REQUEST.value
        assert payload[1] == 0
        assert payload[2] == "hello"
        assert isinstance(payload[3], dict)
        assert payload[3]["temperature"] == 0.5

    def test_add_request_records_submission_time(self):
        """add_request stores a perf_counter timestamp keyed by request id."""

        async def scenario():
            client, _, _ = _make_client()
            client.add_request("p", SamplingParams())
            return client

        client = _run(scenario())
        assert 0 in client.request_submission_times

    def test_send_signal_helpers_send_correct_headers(self):
        """Each control helper sends a payload starting with the matching Headers value."""
        client, _, fake_socket = _make_client()

        helpers = [
            (client.pause_engines, Headers.PAUSE),
            (client.unpause_engines, Headers.UNPAUSE),
            (client.suspend_engines, Headers.SUSPEND),
            (client.resume_engines, Headers.RESUME),
            (client.stop_engines, Headers.STOP),
            (client.shutdown_coordinator, Headers.SHUTDOWN),
        ]

        for fn, expected_header in helpers:
            fake_socket.send.reset_mock()
            fn()
            sent = fake_socket.send.call_args.args[0]
            payload = msgpack.unpackb(sent, raw=False)
            assert payload[0] == expected_header.value

    def test_set_generation_epoch_carries_value(self):
        """set_generation_epoch sends SET_GENERATION_EPOCH along with the epoch number."""
        client, _, fake_socket = _make_client()
        client.set_generation_epoch(42)
        sent = fake_socket.send.call_args.args[0]
        payload = msgpack.unpackb(sent, raw=False)
        assert payload[0] == Headers.SET_GENERATION_EPOCH.value
        assert payload[1] == 42

    def test_connect_handshake_asserts_on_ack(self):
        """_connect_with_inference_coordinator validates that a CONNECT_ACK reply arrives."""
        client, _, fake_socket = _make_client()
        fake_socket.recv.return_value = msgpack.packb(
            [Headers.CONNECT_ACK.value], use_bin_type=True
        )
        client._connect_with_inference_coordinator()
        # Ensure a CONNECT was sent.
        sent = fake_socket.send.call_args.args[0]
        assert msgpack.unpackb(sent, raw=False)[0] == Headers.CONNECT.value

    def test_connect_handshake_raises_on_unexpected_reply(self):
        """A non-CONNECT_ACK reply during handshake raises AssertionError."""
        client, _, fake_socket = _make_client()
        fake_socket.recv.return_value = msgpack.packb([Headers.STOP.value], use_bin_type=True)
        with pytest.raises(AssertionError):
            client._connect_with_inference_coordinator()

    def test_stop_cleans_up_resources(self):
        """stop() cancels the listener task, the futures, and closes the socket/context."""

        async def scenario():
            client, fake_context, fake_socket = _make_client()
            sp = SamplingParams()
            future = client.add_request("p", sp)
            # Simulate a listener task.
            client.listener_task = asyncio.create_task(asyncio.sleep(60))
            client.stop()
            await asyncio.sleep(0)  # let cancellation propagate
            return client, fake_context, fake_socket, future

        client, fake_context, fake_socket, future = _run(scenario())
        # listener_task should be cancelled.
        assert client.listener_task.cancelled() or client.listener_task.done()
        # The pending future must be cancelled.
        assert future.cancelled()
        # completion_futures dict has been cleared.
        assert client.completion_futures == {}
        # Socket and context are torn down.
        fake_socket.close.assert_called_once_with(linger=0)
        fake_context.term.assert_called_once_with()

    def test_stop_without_listener_task_is_safe(self):
        """stop() works even if start() was never called (no listener_task attribute)."""
        client, fake_context, fake_socket = _make_client()
        client.stop()  # must not raise
        fake_socket.close.assert_called_once_with(linger=0)
        fake_context.term.assert_called_once_with()

    def test_recv_task_handles_engine_reply(self):
        """The receive loop pops the future for ENGINE_REPLY and sets its result."""
        import zmq as _zmq

        async def scenario():
            client, _, fake_socket = _make_client()
            # Pre-create a pending future as add_request would have.
            client.next_request_id = 1
            future = asyncio.get_running_loop().create_future()
            client.completion_futures[0] = future
            client.request_submission_times[0] = 0.0
            reply_payload = msgpack.packb(
                [Headers.ENGINE_REPLY.value, 0, {"foo": "bar"}], use_bin_type=True
            )
            # First call returns reply, second call raises zmq.Again to break the loop.
            fake_socket.recv.side_effect = [reply_payload, _zmq.Again()]
            # Run the listener until the future resolves.
            task = asyncio.create_task(client._recv_task())
            try:
                await asyncio.wait_for(future, timeout=2.0)
            finally:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            return future, client

        future, client = _run(scenario())
        assert future.done()
        result = future.result()
        assert result["foo"] == "bar"
        # latency was injected.
        assert "latency" in result
        # The future and submission entry are cleaned up.
        assert 0 not in client.completion_futures
        assert 0 not in client.request_submission_times

    def test_start_handshakes_and_creates_listener(self):
        """start() runs the CONNECT handshake and spawns the listener task."""

        async def scenario():
            client, _, fake_socket = _make_client()
            fake_socket.recv.return_value = msgpack.packb(
                [Headers.CONNECT_ACK.value], use_bin_type=True
            )
            client.start()
            # listener_task must have been created on the running loop.
            task = client.listener_task
            assert isinstance(task, asyncio.Task)
            # Cleanly tear it down.
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return client

        client = _run(scenario())
        assert client._loop is not None

    def test_recv_task_drops_already_done_future(self):
        """If a future was cancelled before the reply arrived, the listener logs and continues."""
        import zmq as _zmq

        async def scenario():
            client, _, fake_socket = _make_client()
            future = asyncio.get_running_loop().create_future()
            future.cancel()  # already done
            client.completion_futures[0] = future
            client.request_submission_times[0] = 0.0
            reply_payload = msgpack.packb([Headers.ENGINE_REPLY.value, 0, {}], use_bin_type=True)
            # Use an iterator-like callable so subsequent calls keep returning
            # zmq.Again() rather than exhausting and raising StopIteration.
            calls = [reply_payload]

            def fake_recv(*args, **kwargs):
                if calls:
                    return calls.pop(0)
                raise _zmq.Again()

            fake_socket.recv.side_effect = fake_recv
            task = asyncio.create_task(client._recv_task())
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except (Exception, asyncio.CancelledError):
                pass
            return client

        client = _run(scenario())
        # The submission time was popped before the warning short-circuit.
        assert 0 not in client.request_submission_times
