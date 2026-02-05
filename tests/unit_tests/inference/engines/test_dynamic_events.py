# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Comprehensive tests for DynamicInferenceEvent and request event lifecycle.

This test suite covers all 9 event types, payload validation, serialization,
request methods, and lifecycle sequences.
"""

import time

import pytest
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
)
from megatron.core.inference.contexts.dynamic_context import (
    BlockOverflowError,
    MaxSequenceLengthOverflowError,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.sampling_params import SamplingParams
from tests.unit_tests.inference.test_utils import TestPriority

# Set this to control which tests run:
# - TestPriority.CRITICAL: Run only critical tests
# - TestPriority.IMPORTANT: Run critical + important tests
# - TestPriority.MEDIUM: Run critical + important + medium tests
# - TestPriority.LOW: Run all tests (default)
TEST_PRIORITY = TestPriority.LOW


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_request():
    """Create a basic request for testing."""
    return DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=10),
    )


@pytest.fixture
def request_with_lifecycle():
    """Create a request with a full lifecycle of events."""
    req = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=10),
    )
    req.add_event_add_engine()
    req.add_event_add_context()
    req.add_event_generated_token(100)
    req.add_event_generated_token(200)
    req.add_event_finish()
    return req


# ============================================================================
# 1. TestDynamicInferenceEventType [CRITICAL]
# ============================================================================


class TestDynamicInferenceEventType:
    """Tests for the DynamicInferenceEventType enum."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_all_nine_event_types_exist(self):
        """Verify all 9 event types are defined."""
        expected_types = [
            'ADD_ENGINE',
            'ADD_CONTEXT',
            'GENERATED_TOKEN',
            'PAUSE',
            'EVICT',
            'FINISH',
            'FAIL',
            'ERROR_TRANSIENT',
            'ERROR_NONTRANSIENT',
        ]
        for event_type in expected_types:
            assert hasattr(
                DynamicInferenceEventType, event_type
            ), f"Missing event type: {event_type}"
        # Verify exactly 9 types
        assert len(DynamicInferenceEventType) == 9

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_old_event_types_removed(self):
        """Verify old ADD and FIRST_TOKEN event types no longer exist."""
        assert not hasattr(DynamicInferenceEventType, 'ADD')
        assert not hasattr(DynamicInferenceEventType, 'FIRST_TOKEN')

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_event_types_are_unique(self):
        """Verify no duplicate auto() values."""
        values = [e.value for e in DynamicInferenceEventType]
        assert len(values) == len(set(values)), "Duplicate event type values detected"


# ============================================================================
# 2. TestEventCreation [CRITICAL]
# ============================================================================


class TestEventCreation:
    """Tests for creating DynamicInferenceEvent instances."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    @pytest.mark.parametrize(
        "event_type",
        [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.PAUSE,
            DynamicInferenceEventType.EVICT,
            DynamicInferenceEventType.FINISH,
            DynamicInferenceEventType.FAIL,
        ],
    )
    def test_create_event_without_payload(self, event_type):
        """Test creating events that should have no payload."""
        event = DynamicInferenceEvent(type=event_type)
        assert event.type == event_type
        assert event.payload is None
        assert event.timestamp is not None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_create_generated_token_event(self):
        """Test creating GENERATED_TOKEN events with int payload."""
        token_id = 42
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.GENERATED_TOKEN, payload=token_id
        )
        assert event.type == DynamicInferenceEventType.GENERATED_TOKEN
        assert event.payload == token_id
        assert event.timestamp is not None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_create_error_transient_event(self):
        """Test creating ERROR_TRANSIENT events with Exception payload."""
        error = RequestOverflowError(request_id=1, message="Too many requests")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=error
        )
        assert event.type == DynamicInferenceEventType.ERROR_TRANSIENT
        assert event.payload is error
        assert event.timestamp is not None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_create_error_nontransient_event(self):
        """Test creating ERROR_NONTRANSIENT events with Exception payload."""
        error = MaxSequenceLengthOverflowError(request_id=2, message="Sequence too long")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_NONTRANSIENT, payload=error
        )
        assert event.type == DynamicInferenceEventType.ERROR_NONTRANSIENT
        assert event.payload is error
        assert event.timestamp is not None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_auto_timestamp_assigned(self):
        """Test that timestamp is auto-generated when not provided."""
        before = time.time()
        event = DynamicInferenceEvent(type=DynamicInferenceEventType.ADD_ENGINE)
        after = time.time()
        assert before <= event.timestamp <= after

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_explicit_timestamp_preserved(self):
        """Test that explicit timestamp is preserved."""
        explicit_time = 1234567890.123
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ADD_ENGINE, timestamp=explicit_time
        )
        assert event.timestamp == explicit_time


# ============================================================================
# 3. TestPayloadValidation [CRITICAL]
# ============================================================================


class TestPayloadValidation:
    """Tests for payload validation in DynamicInferenceEvent."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    @pytest.mark.parametrize(
        "event_type",
        [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.PAUSE,
            DynamicInferenceEventType.EVICT,
            DynamicInferenceEventType.FINISH,
            DynamicInferenceEventType.FAIL,
        ],
    )
    def test_no_payload_events_reject_payload(self, event_type):
        """Test that events that must have None payload reject non-None payload."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=event_type, payload="not allowed")

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_token_requires_int_rejects_none(self):
        """Test GENERATED_TOKEN rejects None payload."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN)

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_token_requires_int_rejects_string(self):
        """Test GENERATED_TOKEN rejects string payload."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(
                type=DynamicInferenceEventType.GENERATED_TOKEN, payload="not an int"
            )

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_token_requires_int_rejects_float(self):
        """Test GENERATED_TOKEN rejects float payload."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(
                type=DynamicInferenceEventType.GENERATED_TOKEN, payload=3.14
            )

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_error_transient_requires_payload(self):
        """Test ERROR_TRANSIENT rejects None payload."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_TRANSIENT)

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_error_nontransient_requires_payload(self):
        """Test ERROR_NONTRANSIENT rejects None payload."""
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_NONTRANSIENT)


# ============================================================================
# 4. TestEventSerialization [CRITICAL]
# ============================================================================


class TestEventSerialization:
    """Tests for event serialization and deserialization."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    @pytest.mark.parametrize(
        "event_type",
        [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.PAUSE,
            DynamicInferenceEventType.EVICT,
            DynamicInferenceEventType.FINISH,
            DynamicInferenceEventType.FAIL,
        ],
    )
    def test_simple_event_roundtrip(self, event_type):
        """Test serialize/deserialize roundtrip for simple events."""
        original = DynamicInferenceEvent(type=event_type)
        serialized = original.serialize()

        assert serialized['type'] == event_type.name
        assert 'timestamp' in serialized

        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.type == original.type
        assert deserialized.timestamp == original.timestamp
        assert deserialized.payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_token_roundtrip(self):
        """Test serialize/deserialize preserves token ID payload."""
        token_id = 12345
        original = DynamicInferenceEvent(
            type=DynamicInferenceEventType.GENERATED_TOKEN, payload=token_id
        )
        serialized = original.serialize()

        assert serialized['type'] == 'GENERATED_TOKEN'
        assert serialized['payload'] == token_id

        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.type == DynamicInferenceEventType.GENERATED_TOKEN
        assert deserialized.payload == token_id
        assert deserialized.timestamp == original.timestamp

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_error_transient_roundtrip(self):
        """Test serialize/deserialize preserves ERROR_TRANSIENT payload via ContextErrorFactory."""
        error = TokenOverflowError(request_id=5, message="Token overflow")
        original = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=error
        )
        serialized = original.serialize()

        assert serialized['type'] == 'ERROR_TRANSIENT'
        assert serialized['payload'] is not None

        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.type == DynamicInferenceEventType.ERROR_TRANSIENT
        assert deserialized.payload.request_id == error.request_id
        assert deserialized.payload.message == error.message

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_error_nontransient_roundtrip(self):
        """Test serialize/deserialize preserves ERROR_NONTRANSIENT payload."""
        error = MaxSequenceLengthOverflowError(request_id=10, message="Max sequence exceeded")
        original = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_NONTRANSIENT, payload=error
        )
        serialized = original.serialize()

        assert serialized['type'] == 'ERROR_NONTRANSIENT'

        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.type == DynamicInferenceEventType.ERROR_NONTRANSIENT
        assert deserialized.payload.request_id == error.request_id
        assert deserialized.payload.message == error.message

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_timestamp_preserved(self):
        """Test that exact timestamp is preserved after roundtrip."""
        explicit_time = 1700000000.123456
        original = DynamicInferenceEvent(
            type=DynamicInferenceEventType.FINISH, timestamp=explicit_time
        )
        serialized = original.serialize()
        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.timestamp == explicit_time


# ============================================================================
# 5. TestErrorPayloads [IMPORTANT]
# ============================================================================


class TestErrorPayloads:
    """Tests for error payload handling in events."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_request_overflow_error_payload(self):
        """Test RequestOverflowError can be used as event payload."""
        error = RequestOverflowError(request_id=1, message="Max requests exceeded")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=error
        )
        assert isinstance(event.payload, RequestOverflowError)

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_token_overflow_error_payload(self):
        """Test TokenOverflowError can be used as event payload."""
        error = TokenOverflowError(request_id=2, message="Token limit exceeded")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=error
        )
        assert isinstance(event.payload, TokenOverflowError)

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_max_seq_overflow_error_payload(self):
        """Test MaxSequenceLengthOverflowError can be used as event payload."""
        error = MaxSequenceLengthOverflowError(request_id=3, message="Sequence too long")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_NONTRANSIENT, payload=error
        )
        assert isinstance(event.payload, MaxSequenceLengthOverflowError)

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_error_preserves_request_id(self):
        """Test that request_id is accessible from error payload."""
        error = BlockOverflowError(request_id=42, message="Block overflow")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=error
        )
        assert event.payload.request_id == 42

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_error_preserves_is_transient(self):
        """Test that is_transient flag is accessible from error payload."""
        # RequestOverflowError should be transient by default
        transient_error = RequestOverflowError(request_id=1, message="Transient error")
        assert transient_error.is_transient is True

        # MaxSequenceLengthOverflowError should be non-transient by default
        nontransient_error = MaxSequenceLengthOverflowError(
            request_id=2, message="Non-transient error"
        )
        assert nontransient_error.is_transient is False


# ============================================================================
# 6. TestRequestEventMethods [CRITICAL]
# ============================================================================


class TestRequestEventMethods:
    """Tests for DynamicInferenceRequest event helper methods."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_add_engine(self, basic_request):
        """Test add_event_add_engine() method."""
        assert len(basic_request.events) == 0
        basic_request.add_event_add_engine()
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.ADD_ENGINE
        assert basic_request.events[0].payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_add_context(self, basic_request):
        """Test add_event_add_context() method."""
        basic_request.add_event_add_context()
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.ADD_CONTEXT
        assert basic_request.events[0].payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_generated_token(self, basic_request):
        """Test add_event_generated_token() method with token payload."""
        token_id = 42
        basic_request.add_event_generated_token(token_id)
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.GENERATED_TOKEN
        assert basic_request.events[0].payload == token_id

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_pause(self, basic_request):
        """Test add_event_pause() method."""
        basic_request.add_event_pause()
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.PAUSE
        assert basic_request.events[0].payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_evict(self, basic_request):
        """Test add_event_evict() method."""
        basic_request.add_event_evict()
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.EVICT
        assert basic_request.events[0].payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_finish(self, basic_request):
        """Test add_event_finish() method."""
        basic_request.add_event_finish()
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.FINISH
        assert basic_request.events[0].payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_fail(self, basic_request):
        """Test add_event_fail() method."""
        basic_request.add_event_fail()
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.FAIL
        assert basic_request.events[0].payload is None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_error_transient(self, basic_request):
        """Test add_event_error_transient() method with exception."""
        error = TokenOverflowError(request_id=1, message="Token overflow")
        basic_request.add_event_error_transient(error)
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.ERROR_TRANSIENT
        assert basic_request.events[0].payload is error

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_error_nontransient(self, basic_request):
        """Test add_event_error_nontransient() method with exception."""
        error = MaxSequenceLengthOverflowError(request_id=1, message="Sequence too long")
        basic_request.add_event_error_nontransient(error)
        assert len(basic_request.events) == 1
        assert basic_request.events[0].type == DynamicInferenceEventType.ERROR_NONTRANSIENT
        assert basic_request.events[0].payload is error


# ============================================================================
# 7. TestGeneratedTokensProperty [CRITICAL]
# ============================================================================


class TestGeneratedTokensProperty:
    """Tests for the generated_tokens property on DynamicInferenceRequest."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_empty_when_no_events(self, basic_request):
        """Test generated_tokens returns [] when no events."""
        assert basic_request.generated_tokens == []

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_empty_when_no_token_events(self, basic_request):
        """Test generated_tokens returns [] when only non-token events."""
        basic_request.add_event_add_engine()
        basic_request.add_event_add_context()
        basic_request.add_event_finish()
        assert basic_request.generated_tokens == []

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_single_token(self, basic_request):
        """Test generated_tokens returns [token_id] for single token."""
        basic_request.add_event_generated_token(999)
        assert basic_request.generated_tokens == [999]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_multiple_tokens_in_order(self, basic_request):
        """Test generated_tokens preserves generation order."""
        basic_request.add_event_generated_token(10)
        basic_request.add_event_generated_token(20)
        basic_request.add_event_generated_token(30)
        assert basic_request.generated_tokens == [10, 20, 30]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_filters_only_token_events(self, basic_request):
        """Test generated_tokens ignores other event types."""
        basic_request.add_event_add_engine()
        basic_request.add_event_add_context()
        basic_request.add_event_generated_token(100)
        basic_request.add_event_pause()
        basic_request.add_event_generated_token(200)
        basic_request.add_event_finish()

        assert basic_request.generated_tokens == [100, 200]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_zero_token_id_valid(self, basic_request):
        """Test that token ID 0 works correctly."""
        basic_request.add_event_generated_token(0)
        assert basic_request.generated_tokens == [0]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_large_token_id(self, basic_request):
        """Test near INT64_MAX token ID works."""
        large_id = 2**62  # Large but valid int
        basic_request.add_event_generated_token(large_id)
        assert basic_request.generated_tokens == [large_id]


# ============================================================================
# 8. TestRequestSerialization [IMPORTANT]
# ============================================================================


class TestRequestSerialization:
    """Tests for DynamicInferenceRequest serialization with events."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_request_with_no_events(self, basic_request):
        """Test serialize produces correct structure with empty events list."""
        serialized = basic_request.serialize()
        assert 'events' in serialized
        assert serialized['events'] == []
        assert 'generated_tokens' in serialized
        assert serialized['generated_tokens'] == []

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_request_with_full_lifecycle(self, request_with_lifecycle):
        """Test serialize produces correct event structure."""
        serialized = request_with_lifecycle.serialize()
        assert len(serialized['events']) == 5

        # Verify event types in serialized form
        event_type_names = [e['type'] for e in serialized['events']]
        expected = ['ADD_ENGINE', 'ADD_CONTEXT', 'GENERATED_TOKEN', 'GENERATED_TOKEN', 'FINISH']
        assert event_type_names == expected

        # Verify generated_tokens is included
        assert serialized['generated_tokens'] == [100, 200]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_preserves_event_order(self, basic_request):
        """Test that event order is maintained in serialization."""
        basic_request.add_event_finish()
        basic_request.add_event_pause()
        basic_request.add_event_add_engine()

        serialized = basic_request.serialize()
        event_type_names = [e['type'] for e in serialized['events']]
        expected = ['FINISH', 'PAUSE', 'ADD_ENGINE']
        assert event_type_names == expected

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_generated_tokens_in_serialized_output(self, request_with_lifecycle):
        """Test generated_tokens is correctly included in serialized output."""
        serialized = request_with_lifecycle.serialize()
        assert 'generated_tokens' in serialized
        assert serialized['generated_tokens'] == [100, 200]


# ============================================================================
# 9. TestRequestRecordMerge [IMPORTANT]
# ============================================================================


class TestRequestRecordMerge:
    """Tests for DynamicInferenceRequestRecord merge functionality."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_single_request_merge(self, request_with_lifecycle):
        """Test merge with single request returns same events."""
        record = DynamicInferenceRequestRecord.from_request(request_with_lifecycle)
        merged = record.merge()

        assert len(merged.events) == len(request_with_lifecycle.events)
        assert merged.generated_tokens == request_with_lifecycle.generated_tokens

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_two_checkpoints_merge(self):
        """Test merge concatenates events from multiple checkpoints."""
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=10),
        )
        req1.add_event_add_engine()
        req1.add_event_add_context()
        req1.add_event_generated_token(100)

        req2 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 3, 100], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=9),
        )
        req2.add_event_add_engine()
        req2.add_event_generated_token(200)
        req2.add_event_finish()

        record = DynamicInferenceRequestRecord()
        record.requests.append(req1)
        record.requests.append(req2)

        merged = record.merge()
        assert len(merged.events) == 6  # 3 + 3 events

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_preserves_generated_tokens(self):
        """Test that merge preserves generated_tokens from all checkpoints."""
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=5),
        )
        req1.add_event_generated_token(10)
        req1.add_event_generated_token(20)

        req2 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 10, 20], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=3),
        )
        req2.add_event_generated_token(30)

        record = DynamicInferenceRequestRecord()
        record.requests.append(req1)
        record.requests.append(req2)

        merged = record.merge()
        assert merged.generated_tokens == [10, 20, 30]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_eviction_events_preserved(self):
        """Test that EVICT events are preserved in merged list."""
        req1 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=5),
        )
        req1.add_event_add_engine()
        req1.add_event_evict()

        req2 = DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
            sampling_params=SamplingParams(num_tokens_to_generate=5),
        )
        req2.add_event_add_engine()
        req2.add_event_finish()

        record = DynamicInferenceRequestRecord()
        record.requests.append(req1)
        record.requests.append(req2)

        merged = record.merge()
        event_types = [e.type for e in merged.events]
        assert DynamicInferenceEventType.EVICT in event_types


# ============================================================================
# 10. TestEventLifecycleSequences [IMPORTANT]
# ============================================================================


class TestEventLifecycleSequences:
    """Tests for common event lifecycle sequences."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_successful_lifecycle(self, basic_request):
        """Test ADD_ENGINE -> ADD_CONTEXT -> GENERATED_TOKEN(s) -> FINISH lifecycle."""
        basic_request.add_event_add_engine()
        basic_request.add_event_add_context()
        basic_request.add_event_generated_token(100)
        basic_request.add_event_generated_token(200)
        basic_request.add_event_generated_token(300)
        basic_request.add_event_finish()

        event_types = [e.type for e in basic_request.events]
        assert event_types == [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.GENERATED_TOKEN,
            DynamicInferenceEventType.GENERATED_TOKEN,
            DynamicInferenceEventType.GENERATED_TOKEN,
            DynamicInferenceEventType.FINISH,
        ]
        assert basic_request.generated_tokens == [100, 200, 300]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_failed_lifecycle(self, basic_request):
        """Test ADD_ENGINE -> ERROR_NONTRANSIENT -> FAIL lifecycle."""
        error = MaxSequenceLengthOverflowError(request_id=1, message="Sequence too long")
        basic_request.add_event_add_engine()
        basic_request.add_event_error_nontransient(error)
        basic_request.add_event_fail()

        event_types = [e.type for e in basic_request.events]
        assert event_types == [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ERROR_NONTRANSIENT,
            DynamicInferenceEventType.FAIL,
        ]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_eviction_lifecycle(self, basic_request):
        """Test full evict/recover sequence."""
        # First attempt - evicted
        basic_request.add_event_add_engine()
        basic_request.add_event_add_context()
        basic_request.add_event_generated_token(100)
        basic_request.add_event_evict()

        # Second attempt after recovery (re-added to engine)
        basic_request.add_event_add_engine()
        basic_request.add_event_add_context()
        basic_request.add_event_generated_token(200)
        basic_request.add_event_finish()

        event_types = [e.type for e in basic_request.events]
        assert DynamicInferenceEventType.EVICT in event_types
        assert basic_request.generated_tokens == [100, 200]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_pause_lifecycle(self, basic_request):
        """Test lifecycle with PAUSE events."""
        basic_request.add_event_add_engine()
        basic_request.add_event_add_context()
        basic_request.add_event_generated_token(100)
        basic_request.add_event_pause()
        # Resumed
        basic_request.add_event_generated_token(200)
        basic_request.add_event_finish()

        event_types = [e.type for e in basic_request.events]
        assert DynamicInferenceEventType.PAUSE in event_types
        assert basic_request.generated_tokens == [100, 200]


# ============================================================================
# 11. TestEventTimestamps [MEDIUM]
# ============================================================================


class TestEventTimestamps:
    """Tests for event timestamp behavior."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_auto_timestamp_recent(self):
        """Test that auto-generated timestamp is within 1 second of now."""
        before = time.time()
        event = DynamicInferenceEvent(type=DynamicInferenceEventType.ADD_ENGINE)
        after = time.time()

        assert before <= event.timestamp <= after
        assert (after - event.timestamp) < 1.0

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_monotonically_increasing(self, basic_request):
        """Test that sequential events have ordered timestamps."""
        basic_request.add_event_add_engine()
        time.sleep(0.001)
        basic_request.add_event_add_context()
        time.sleep(0.001)
        basic_request.add_event_generated_token(42)

        timestamps = [e.timestamp for e in basic_request.events]
        assert timestamps == sorted(timestamps), "Timestamps should be increasing"

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_ttft_calculable(self, basic_request):
        """Test that TTFT can be computed from GENERATED_TOKEN[0] - ADD_ENGINE."""
        basic_request.add_event_add_engine()
        time.sleep(0.01)  # Small delay to make TTFT measurable
        basic_request.add_event_add_context()
        time.sleep(0.01)
        basic_request.add_event_generated_token(100)

        add_engine_time = None
        first_token_time = None

        for event in basic_request.events:
            if event.type == DynamicInferenceEventType.ADD_ENGINE:
                add_engine_time = event.timestamp
            elif event.type == DynamicInferenceEventType.GENERATED_TOKEN and first_token_time is None:
                first_token_time = event.timestamp

        assert add_engine_time is not None
        assert first_token_time is not None

        ttft = first_token_time - add_engine_time
        assert ttft > 0, "TTFT should be positive"
        assert ttft < 1.0, "TTFT should be less than 1 second in test"


# ============================================================================
# 12. TestEventEdgeCases [MEDIUM]
# ============================================================================


class TestEventEdgeCases:
    """Tests for edge cases in event handling."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    @pytest.mark.parametrize(
        "event_type",
        [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
            DynamicInferenceEventType.PAUSE,
            DynamicInferenceEventType.EVICT,
            DynamicInferenceEventType.FINISH,
            DynamicInferenceEventType.FAIL,
        ],
    )
    def test_event_str_representation_simple(self, event_type):
        """Test __str__ for simple events (no payload)."""
        event = DynamicInferenceEvent(type=event_type)
        str_repr = str(event)
        assert event_type.name in str_repr
        assert str(event.timestamp)[:5] in str_repr  # Timestamp should appear

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_event_str_representation_generated_token(self):
        """Test __str__ for GENERATED_TOKEN event shows token ID."""
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.GENERATED_TOKEN, payload=42
        )
        str_repr = str(event)
        assert 'GENERATED_TOKEN' in str_repr
        assert 'token=42' in str_repr

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_event_str_representation_error(self):
        """Test __str__ for error events shows error type."""
        error = RequestOverflowError(request_id=1, message="Error")
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=error
        )
        str_repr = str(event)
        assert 'ERROR_TRANSIENT' in str_repr
        assert 'RequestOverflowError' in str_repr

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_rapid_events_distinct_timestamps(self, basic_request):
        """Test that rapidly added events get distinct or at least ordered timestamps."""
        for i in range(100):
            basic_request.add_event_generated_token(i)

        timestamps = [e.timestamp for e in basic_request.events]
        # Timestamps should be non-decreasing (may be equal if added very fast)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1], "Timestamps should not decrease"

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_many_generated_tokens(self, basic_request):
        """Test performance with 100+ generated tokens."""
        num_tokens = 150
        for i in range(num_tokens):
            basic_request.add_event_generated_token(i)

        assert len(basic_request.events) == num_tokens
        assert len(basic_request.generated_tokens) == num_tokens
        assert basic_request.generated_tokens == list(range(num_tokens))


# ============================================================================
# 13. TestCompactGeneratedTokenSerialization [CRITICAL]
# ============================================================================


class TestCompactGeneratedTokenSerialization:
    """Tests for compact GENERATED_TOKEN serialization (track_generated_token_events=False)."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_token_serializes_to_int(self):
        """GENERATED_TOKEN events serialize to just the token ID when compact."""
        event = DynamicInferenceEvent(
            type=DynamicInferenceEventType.GENERATED_TOKEN, payload=42
        )
        serialized = event.serialize(track_generated_token_events=False)
        assert serialized == 42

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_other_events_serialize_normally(self):
        """Non-GENERATED_TOKEN events serialize normally regardless of flag."""
        event = DynamicInferenceEvent(type=DynamicInferenceEventType.ADD_ENGINE)
        serialized = event.serialize(track_generated_token_events=False)
        assert isinstance(serialized, dict)
        assert serialized["type"] == "ADD_ENGINE"

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_deserialize_int_to_generated_token(self):
        """Integer deserializes to GENERATED_TOKEN event with timestamp=-1."""
        event = DynamicInferenceEvent.deserialize(42)
        assert event.type == DynamicInferenceEventType.GENERATED_TOKEN
        assert event.payload == 42
        assert event.timestamp == -1

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_compact_roundtrip(self):
        """Compact serialize -> deserialize preserves token ID."""
        original = DynamicInferenceEvent(
            type=DynamicInferenceEventType.GENERATED_TOKEN, payload=12345
        )
        serialized = original.serialize(track_generated_token_events=False)
        restored = DynamicInferenceEvent.deserialize(serialized)
        assert restored.type == original.type
        assert restored.payload == original.payload
        # Timestamp is lost in compact mode
        assert restored.timestamp == -1

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_request_compact_serialization(self, basic_request):
        """Request serializes events in compact mode when flag is False."""
        basic_request.add_event_add_engine()
        basic_request.add_event_generated_token(100)
        basic_request.add_event_generated_token(200)
        basic_request.add_event_finish()

        serialized = basic_request.serialize(track_generated_token_events=False)
        events = serialized["events"]

        # ADD_ENGINE and FINISH are dicts, GENERATED_TOKEN are ints
        assert isinstance(events[0], dict)  # ADD_ENGINE
        assert events[1] == 100  # GENERATED_TOKEN compact
        assert events[2] == 200  # GENERATED_TOKEN compact
        assert isinstance(events[3], dict)  # FINISH

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_request_compact_roundtrip(self, basic_request):
        """Request roundtrip works in compact mode."""
        basic_request.add_event_add_engine()
        basic_request.add_event_generated_token(100)
        basic_request.add_event_finish()

        serialized = basic_request.serialize(track_generated_token_events=False)
        restored = DynamicInferenceRequest.deserialize(serialized)

        assert restored.generated_tokens == [100]
        assert len(restored.events) == 3

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_record_compact_serialization(self, basic_request):
        """Record serializes nested requests in compact mode."""
        basic_request.add_event_generated_token(42)
        record = DynamicInferenceRequestRecord.from_request(basic_request)

        serialized = record.serialize(track_generated_token_events=False)
        assert serialized["requests"][0]["events"][0] == 42

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_generated_tokens_property_with_compact_events(self, basic_request):
        """generated_tokens property works after deserializing compact events."""
        basic_request.add_event_generated_token(10)
        basic_request.add_event_generated_token(20)
        basic_request.add_event_generated_token(30)

        serialized = basic_request.serialize(track_generated_token_events=False)
        restored = DynamicInferenceRequest.deserialize(serialized)

        assert restored.generated_tokens == [10, 20, 30]
