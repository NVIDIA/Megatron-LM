# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for coordinator observability (metrics instrumentation).

Tests verify that the correct metric counters and observations are emitted by
the DataParallelInferenceCoordinator for each instrumented scenario.

The TestMetrics class is used as an in-memory backend so tests can inspect
recorded values without any real metrics infrastructure.
"""

from collections import defaultdict, deque
from unittest.mock import MagicMock, patch

import pytest

from megatron.core.inference.config import PrefixCachingCoordinatorPolicy
from megatron.core.inference.coordinator_metrics import CoordinatorMetrics, NoOpMetrics


# ---------------------------------------------------------------------------
# In-memory metrics implementation used by tests
# ---------------------------------------------------------------------------


class TestMetrics(CoordinatorMetrics):
    """In-memory metrics backend for unit testing.

    Captures all metric operations so test cases can assert on their effects
    without depending on a real metrics backend.
    """

    def __init__(self):
        self.counters: dict[str, int] = defaultdict(int)
        self.observations: dict[str, list[float]] = defaultdict(list)
        self.gauges: dict[str, float] = {}

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def observe(self, name: str, value: float) -> None:
        self.observations[name].append(value)

    def gauge(self, name: str, value: float) -> None:
        self.gauges[name] = value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_coordinator(
    *,
    data_parallel_size: int = 1,
    enable_prefix_caching: bool = False,
    block_size_tokens: int = 4,
    policy: PrefixCachingCoordinatorPolicy = PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK,
    metrics: CoordinatorMetrics | None = None,
):
    """Build a DataParallelInferenceCoordinator without ZMQ or tokenizer.

    Bypasses __init__ to avoid ZMQ socket setup.  Only the fields required by
    the methods under test are initialised.
    """
    # Import here so that missing optional deps don't break collection.
    from megatron.core.inference.data_parallel_inference_coordinator import (
        DataParallelInferenceCoordinator,
    )

    coordinator = object.__new__(DataParallelInferenceCoordinator)

    # Mimic the state set by __init__ that the tested methods rely on.
    coordinator.data_parallel_size = data_parallel_size
    coordinator.enable_prefix_caching = enable_prefix_caching
    coordinator.block_size_tokens = block_size_tokens
    coordinator.prefix_caching_coordinator_policy = policy
    coordinator.hash_to_rank_info = {}
    coordinator._assignment_counter = 0
    coordinator._round_robin_idx = 0
    coordinator.next_request_id = 0
    coordinator.request_id_to_client_id = {}
    coordinator.request_id_to_client_request_id = {}
    coordinator.schedule_records = None
    coordinator.identity_to_rank_index = {}

    # Use two fake engine identities.
    coordinator.identities_of_data_parallel_ranks = deque([b"engine-0", b"engine-1"])
    coordinator.metrics = metrics if metrics is not None else NoOpMetrics()

    return coordinator


def _make_start_loop_coordinator(
    metrics: CoordinatorMetrics,
    num_engines: int = 1,
):
    """Build a coordinator ready to run start() with a mock ZMQ socket.

    Bypasses __init__ and populates only the state required by the start() loop.
    The caller must set mock_socket.recv_multipart.side_effect with the desired
    message sequence before calling coordinator.start().
    """
    from megatron.core.inference.data_parallel_inference_coordinator import (
        DataParallelInferenceCoordinator,
    )

    coordinator = object.__new__(DataParallelInferenceCoordinator)
    coordinator.data_parallel_size = num_engines
    coordinator.enable_prefix_caching = False
    coordinator.block_size_tokens = 4
    coordinator.prefix_caching_coordinator_policy = PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK
    coordinator.hash_to_rank_info = {}
    coordinator._assignment_counter = 0
    coordinator._round_robin_idx = 0
    coordinator.next_request_id = 0
    coordinator.request_id_to_client_id = {}
    coordinator.request_id_to_client_request_id = {}
    coordinator.schedule_records = None
    coordinator.state = DataParallelInferenceCoordinator.CoordinatorState.RUNNING

    engines = [f"engine-{i}".encode() for i in range(num_engines)]
    coordinator.identities_of_data_parallel_ranks = deque(engines)
    coordinator.identity_to_rank_index = {e: i for i, e in enumerate(engines)}
    coordinator.metrics = metrics
    coordinator.tokenizer = MagicMock()

    mock_socket = MagicMock()
    coordinator.router_socket = mock_socket

    return coordinator, mock_socket


# ---------------------------------------------------------------------------
# Tests: CoordinatorMetrics abstraction
# ---------------------------------------------------------------------------


class TestCoordinatorMetricsAbstraction:
    def test_metrics_noop_calls_no_exception(self):
        m = NoOpMetrics()
        m.inc("any_counter")
        m.observe("any_observation", 0.1)
        m.gauge("any_gauge", 42.0)

    def test_metrics_counter_increments_accumulated_value(self):
        m = TestMetrics()
        m.inc("foo")
        m.inc("foo")
        m.inc("bar", 3)
        assert m.counters["foo"] == 2
        assert m.counters["bar"] == 3

    def test_metrics_observe_appends_samples_in_order(self):
        m = TestMetrics()
        m.observe("latency", 0.5)
        m.observe("latency", 1.2)
        assert m.observations["latency"] == [0.5, 1.2]

    def test_metrics_gauge_overwrite_keeps_latest_value(self):
        m = TestMetrics()
        m.gauge("engines", 4.0)
        m.gauge("engines", 3.0)  # overwrite
        assert m.gauges["engines"] == 3.0


# ---------------------------------------------------------------------------
# Tests: _log_protocol_error
# ---------------------------------------------------------------------------


class TestLogProtocolError:
    def test_protocol_error_client_error_increments_invalid_message_counter(self):
        m = TestMetrics()
        coordinator = _make_minimal_coordinator(metrics=m)
        coordinator._log_protocol_error("client_error", "bad header")
        assert m.counters["coordinator_invalid_message_total"] == 1

    def test_protocol_error_internal_error_increments_internal_error_counter(self):
        m = TestMetrics()
        coordinator = _make_minimal_coordinator(metrics=m)
        coordinator._log_protocol_error("internal_error", "unexpected state")
        assert m.counters["coordinator_internal_error_total"] == 1

    def test_protocol_error_unknown_type_does_not_increment_known_counters(self):
        m = TestMetrics()
        coordinator = _make_minimal_coordinator(metrics=m)
        coordinator._log_protocol_error("bogus_type", "some message")
        assert m.counters["coordinator_invalid_message_total"] == 0
        assert m.counters["coordinator_internal_error_total"] == 0

    def test_protocol_error_multiple_calls_accumulate_counters(self):
        m = TestMetrics()
        coordinator = _make_minimal_coordinator(metrics=m)
        coordinator._log_protocol_error("client_error", "err1")
        coordinator._log_protocol_error("client_error", "err2")
        coordinator._log_protocol_error("internal_error", "err3")
        assert m.counters["coordinator_invalid_message_total"] == 2
        assert m.counters["coordinator_internal_error_total"] == 1


# ---------------------------------------------------------------------------
# Tests: routing quality metrics
# ---------------------------------------------------------------------------


class TestRoutingMetrics:
    """Tests for cache hit / miss / stale routing metrics."""

    def _make_cache_coordinator(self, policy=PrefixCachingCoordinatorPolicy.FIRST_PREFIX_BLOCK):
        m = TestMetrics()
        c = _make_minimal_coordinator(
            enable_prefix_caching=True,
            block_size_tokens=4,
            policy=policy,
            metrics=m,
        )
        return c, m

    def test_routing_cache_hit_increments_hit_counter(self):
        c, m = self._make_cache_coordinator()
        # Pre-populate hash table: hash 99 → engine-0 alive in pool.
        c.hash_to_rank_info[99] = {b"engine-0": 1}

        result = c.get_best_data_parallel_rank([99], record_metrics=True)

        assert result == b"engine-0"
        assert m.counters["routing_cache_hit_total"] == 1
        assert m.counters["routing_cache_miss_total"] == 0
        assert m.counters["routing_stale_detected_total"] == 0

    def test_routing_cache_miss_increments_miss_counter(self):
        c, m = self._make_cache_coordinator()
        # hash_to_rank_info is empty → no match.

        c.get_best_data_parallel_rank([42, 43], record_metrics=True)

        assert m.counters["routing_cache_miss_total"] == 1
        assert m.counters["routing_cache_hit_total"] == 0

    def test_routing_stale_entry_increments_stale_counter(self):
        c, m = self._make_cache_coordinator()
        # Hash points to a rank that has been removed from the pool.
        c.hash_to_rank_info[99] = {b"dead-engine": 1}
        # b"dead-engine" is NOT in identities_of_data_parallel_ranks.

        c.get_best_data_parallel_rank([99], record_metrics=True)

        assert m.counters["routing_stale_detected_total"] == 1
        assert m.counters["routing_cache_hit_total"] == 0

    def test_routing_prefix_caching_disabled_records_no_quality_metrics(self):
        m = TestMetrics()
        c = _make_minimal_coordinator(enable_prefix_caching=False, metrics=m)
        c.hash_to_rank_info[99] = {b"engine-0": 1}

        c.get_best_data_parallel_rank([99], record_metrics=True)

        # Routing-quality counters are meaningless when prefix caching is off.
        assert m.counters["routing_cache_hit_total"] == 0
        assert m.counters["routing_cache_miss_total"] == 0
        assert m.counters["routing_stale_detected_total"] == 0

    def test_routing_round_robin_policy_records_no_quality_metrics(self):
        c, m = self._make_cache_coordinator(
            policy=PrefixCachingCoordinatorPolicy.ROUND_ROBIN
        )
        c.hash_to_rank_info[99] = {b"engine-0": 1}

        c.get_best_data_parallel_rank([99], record_metrics=True)

        assert m.counters["routing_cache_hit_total"] == 0
        assert m.counters["routing_cache_miss_total"] == 0

    def test_routing_empty_hashes_records_no_quality_metrics(self):
        c, m = self._make_cache_coordinator()
        # No hashes → can't do prefix match.
        c.get_best_data_parallel_rank([], record_metrics=True)

        assert m.counters["routing_cache_hit_total"] == 0
        assert m.counters["routing_cache_miss_total"] == 0

    def test_routing_longest_prefix_policy_returns_cache_hit_metric(self):
        c, m = self._make_cache_coordinator(
            policy=PrefixCachingCoordinatorPolicy.LONGEST_PREFIX
        )
        c.hash_to_rank_info[10] = {b"engine-0": 1}
        c.hash_to_rank_info[20] = {b"engine-0": 2}

        # Reversed scan finds hash 20 first (longest match).
        result = c.get_best_data_parallel_rank([10, 20], record_metrics=True)

        assert result == b"engine-0"
        assert m.counters["routing_cache_hit_total"] == 1

    def test_routing_record_metrics_false_prevents_double_counting(self):
        c, m = self._make_cache_coordinator()
        c.hash_to_rank_info[99] = {b"engine-0": 1}

        c.get_best_data_parallel_rank([99], record_metrics=True)
        c.get_best_data_parallel_rank([99], record_metrics=False)

        assert m.counters["routing_cache_hit_total"] == 1


# ---------------------------------------------------------------------------
# Tests: engine-unreachable metrics
# ---------------------------------------------------------------------------


class TestEngineUnreachableMetric:
    def test_send_ehostunreach_failure_increments_unreachable_counter(self):
        try:
            import zmq
        except ImportError:
            pytest.skip("pyzmq not installed")

        m = TestMetrics()
        c = _make_minimal_coordinator(metrics=m)

        # Fake a router socket that raises EHOSTUNREACH.
        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = zmq.error.ZMQError(zmq.EHOSTUNREACH)
        c.router_socket = mock_socket
        c.identities_of_data_parallel_ranks = deque([b"engine-0", b"engine-1"])

        result = c._send_to_engine(b"engine-0", b"payload")

        assert result is False
        assert m.counters["coordinator_engine_unreachable_total"] == 1
        # Active engines gauge should reflect removal.
        assert m.gauges["coordinator_active_engines"] == 1.0

    def test_send_success_does_not_increment_unreachable_counter(self):
        m = TestMetrics()
        c = _make_minimal_coordinator(metrics=m)

        mock_socket = MagicMock()
        c.router_socket = mock_socket

        result = c._send_to_engine(b"engine-0", b"payload")

        assert result is True
        assert m.counters["coordinator_engine_unreachable_total"] == 0

    def test_send_non_ehostunreach_failure_still_increments_unreachable_counter(self):
        try:
            import zmq
        except ImportError:
            pytest.skip("pyzmq not installed")

        m = TestMetrics()
        c = _make_minimal_coordinator(metrics=m)

        mock_socket = MagicMock()
        mock_socket.send_multipart.side_effect = zmq.error.ZMQError(zmq.EINVAL)
        c.router_socket = mock_socket

        with pytest.raises(zmq.error.ZMQError):
            c._send_to_engine(b"engine-0", b"payload")

        assert m.counters["coordinator_engine_unreachable_total"] == 1


# ---------------------------------------------------------------------------
# Tests: routing latency is recorded
# ---------------------------------------------------------------------------


class TestRoutingLatencyRecorded:
    """Verify that coordinator_routing_latency_seconds is recorded by start()."""

    def test_routing_latency_recorded_by_start_loop(self):
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m)
        mock_socket.recv_multipart.side_effect = [
            (b"client-1", msgpack.packb([Headers.CONNECT.value], use_bin_type=True)),
            (
                b"client-1",
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 0, [1, 2, 3], {}], use_bin_type=True
                ),
            ),
            (b"client-1", msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True)),
        ]

        coordinator.start()

        assert len(m.observations["coordinator_routing_latency_seconds"]) == 1
        assert m.observations["coordinator_routing_latency_seconds"][0] >= 0.0


# ---------------------------------------------------------------------------
# Tests: active engines gauge
# ---------------------------------------------------------------------------


class TestActiveEnginesGauge:
    def test_active_engines_remove_engine_updates_gauge_value(self):
        m = TestMetrics()
        c = _make_minimal_coordinator(metrics=m)
        # Pool starts with 2 engines.
        assert len(c.identities_of_data_parallel_ranks) == 2

        c._remove_engine(b"engine-0")

        assert m.gauges["coordinator_active_engines"] == 1.0

    def test_active_engines_multiple_removals_decrement_gauge_value(self):
        m = TestMetrics()
        c = _make_minimal_coordinator(metrics=m)

        c._remove_engine(b"engine-0")
        c._remove_engine(b"engine-1")

        assert m.gauges["coordinator_active_engines"] == 0.0


# ---------------------------------------------------------------------------
# Tests: default metrics (NoOp) does not crash
# ---------------------------------------------------------------------------


class TestDefaultNoOpMetrics:
    def test_default_metrics_none_uses_noop_and_no_exception(self):
        """When no metrics arg is passed, coordinator uses NoOpMetrics silently."""
        c = _make_minimal_coordinator(metrics=None)
        assert isinstance(c.metrics, NoOpMetrics)

        # Execute a small basic flow with default NoOpMetrics.
        c.router_socket = MagicMock()
        assert c._send_to_engine(b"engine-0", b"payload") is True

        # Exercise instrumented paths; none should raise.
        c._log_protocol_error("client_error", "test")
        c.get_best_data_parallel_rank([], record_metrics=True)
        c._remove_engine(b"engine-0")

    def test_default_metrics_parameter_optional_defaults_to_noop(self):
        c = _make_minimal_coordinator(metrics=None)
        assert isinstance(c.metrics, NoOpMetrics)


# ---------------------------------------------------------------------------
# Tests: message processing latency
# ---------------------------------------------------------------------------


class TestMessageProcessingLatencyRecorded:
    """Verify coordinator_message_processing_latency_seconds is recorded per message."""

    def test_message_processing_latency_recorded_per_message(self):
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m)
        mock_socket.recv_multipart.side_effect = [
            (b"client-1", msgpack.packb([Headers.CONNECT.value], use_bin_type=True)),
            (
                b"client-1",
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 0, [1, 2, 3], {}], use_bin_type=True
                ),
            ),
            (b"client-1", msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True)),
        ]

        coordinator.start()

        # One observation per message: CONNECT + SUBMIT_REQUEST + SHUTDOWN.
        assert len(m.observations["coordinator_message_processing_latency_seconds"]) == 3
        assert all(
            v >= 0.0 for v in m.observations["coordinator_message_processing_latency_seconds"]
        )


# ---------------------------------------------------------------------------
# Tests: unknown sender metric
# ---------------------------------------------------------------------------


class TestUnknownSenderMetric:
    """Verify coordinator_unknown_sender_total increments for all unknown-sender paths."""

    def test_submit_request_from_unknown_sender_increments_counter(self):
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m)
        mock_socket.recv_multipart.side_effect = [
            (
                b"unknown-client",
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 0, [1, 2, 3], {}], use_bin_type=True
                ),
            ),
        ]

        with pytest.raises(StopIteration):
            coordinator.start()

        assert m.counters["coordinator_unknown_sender_total"] == 1

    def test_control_signal_from_unknown_sender_increments_counter(self):
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m)
        mock_socket.recv_multipart.side_effect = [
            (b"unknown-client", msgpack.packb([Headers.PAUSE.value], use_bin_type=True)),
        ]

        with pytest.raises(StopIteration):
            coordinator.start()

        assert m.counters["coordinator_unknown_sender_total"] == 1

    def test_shutdown_from_unknown_sender_increments_counter(self):
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m)
        mock_socket.recv_multipart.side_effect = [
            (b"unknown-client", msgpack.packb([Headers.SHUTDOWN.value], use_bin_type=True)),
        ]

        with pytest.raises(StopIteration):
            coordinator.start()

        assert m.counters["coordinator_unknown_sender_total"] == 1


# ---------------------------------------------------------------------------
# Tests: all-engines-exhausted metric
# ---------------------------------------------------------------------------


class TestAllEnginesExhaustedMetric:
    """Verify coordinator_all_engines_exhausted_total fires when every engine is unreachable."""

    def test_all_engines_exhausted_metric_increments_when_no_engine_reachable(self):
        try:
            import zmq
            import msgpack
        except ImportError:
            pytest.skip("pyzmq and msgpack required")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m, num_engines=1)

        # CONNECT ACK is sent to b"client-1"; engine sends go to b"engine-0".
        def send_side_effect(parts):
            if parts[0] == b"engine-0":
                raise zmq.error.ZMQError(zmq.EHOSTUNREACH)

        mock_socket.send_multipart.side_effect = send_side_effect
        mock_socket.recv_multipart.side_effect = [
            (b"client-1", msgpack.packb([Headers.CONNECT.value], use_bin_type=True)),
            (
                b"client-1",
                msgpack.packb(
                    [Headers.SUBMIT_REQUEST.value, 0, [1, 2, 3], {}], use_bin_type=True
                ),
            ),
        ]

        coordinator.start()  # Returns via return after all-engines-dead path.

        assert m.counters["coordinator_all_engines_exhausted_total"] == 1


# ---------------------------------------------------------------------------
# Tests: ENGINE_REPLY internal error metric
# ---------------------------------------------------------------------------


class TestEngineReplyInternalErrorMetric:
    """Verify coordinator_internal_error_total fires for ENGINE_REPLY from unregistered engine."""

    def test_engine_reply_from_unregistered_sender_emits_internal_error_metric(self):
        try:
            import msgpack
        except ImportError:
            pytest.skip("msgpack not installed")

        from megatron.core.inference.headers import Headers

        m = TestMetrics()
        coordinator, mock_socket = _make_start_loop_coordinator(m)

        # ENGINE_REPLY from a sender not in identities_of_data_parallel_ranks.
        mock_socket.recv_multipart.side_effect = [
            (
                b"unknown-engine",
                msgpack.packb([Headers.ENGINE_REPLY.value, []], use_bin_type=True),
            ),
        ]

        with pytest.raises(AssertionError):
            coordinator.start()

        assert m.counters["coordinator_internal_error_total"] == 1


# ---------------------------------------------------------------------------
# Tests: initial active engines gauge (via __init__)
# ---------------------------------------------------------------------------


class TestInitActiveEnginesGauge:
    """Verify coordinator_active_engines gauge is written during __init__."""

    def test_active_engines_gauge_set_at_init_time(self):
        try:
            import zmq
            import msgpack
        except ImportError:
            pytest.skip("pyzmq and msgpack required")

        from megatron.core.inference.data_parallel_inference_coordinator import (
            DataParallelInferenceCoordinator,
        )

        m = TestMetrics()
        mock_pipe = MagicMock()
        mock_socket = MagicMock()
        mock_socket.recv_multipart.side_effect = [(b"eng-0", b""), (b"eng-1", b"")]
        mock_socket.getsockopt_string.return_value = "tcp://localhost:1234"

        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        with patch(
            "megatron.core.inference.data_parallel_inference_coordinator.zmq.Context",
            return_value=mock_context,
        ), patch("socket.gethostname", return_value="localhost"):
            coordinator = DataParallelInferenceCoordinator(
                pipe_connection=mock_pipe,
                data_parallel_size=2,
                tokenizer=MagicMock(),
                metrics=m,
            )

        assert m.gauges.get("coordinator_active_engines") == 2.0


# ---------------------------------------------------------------------------
# Tests: entrypoint metrics forwarding
# ---------------------------------------------------------------------------


class TestEntrypointMetricsForwarding:
    """Verify metrics passed to entrypoint() reach the constructed coordinator."""

    def test_entrypoint_forwards_metrics_to_coordinator(self):
        try:
            import zmq
            import msgpack
        except ImportError:
            pytest.skip("pyzmq and msgpack required")

        from megatron.core.inference.data_parallel_inference_coordinator import (
            DataParallelInferenceCoordinator,
        )

        m = TestMetrics()
        mock_pipe = MagicMock()
        mock_ready_event = MagicMock()
        mock_socket = MagicMock()
        mock_socket.recv_multipart.side_effect = [(b"eng-0", b"")]
        mock_socket.getsockopt_string.return_value = "tcp://localhost:1234"

        mock_context = MagicMock()
        mock_context.socket.return_value = mock_socket

        with patch(
            "megatron.core.inference.data_parallel_inference_coordinator.zmq.Context",
            return_value=mock_context,
        ), patch("socket.gethostname", return_value="localhost"), patch.object(
            DataParallelInferenceCoordinator, "start"
        ):
            DataParallelInferenceCoordinator.entrypoint(
                pipe_connection=mock_pipe,
                ready_event=mock_ready_event,
                data_parallel_size=1,
                tokenizer=MagicMock(),
                metrics=m,
            )

        mock_ready_event.set.assert_called_once()
        # If metrics were forwarded correctly, the init gauge was written.
        assert m.gauges.get("coordinator_active_engines") == 1.0
