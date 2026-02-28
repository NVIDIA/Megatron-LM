# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Comprehensive tests for DynamicInferenceEvent and request event lifecycle.

This test suite covers all 9 event types, payload validation, serialization,
request methods, and lifecycle sequences with exactly 9 substantial tests.
"""

import time

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import (
    BlockOverflowError,
    MaxSequenceLengthOverflowError,
    RequestOverflowError,
    TokenOverflowError,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
)
from megatron.core.inference.sampling_params import SamplingParams

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


# ============================================================================
# Test 1: All Event Types Creation and Payload Validation
# ============================================================================


def test_all_event_types_creation_and_payload_validation():
    """Test all 9 event types exist, are unique, and enforce payload validation rules.

    Coverage:
    - All 9 event types (ADD_ENGINE, ADD_CONTEXT, GENERATED_TOKEN, PAUSE, EVICT,
      FINISH, FAIL, ERROR_TRANSIENT, ERROR_NONTRANSIENT)
    - Payload requirements (int for GENERATED_TOKEN, exception for errors, None for others)
    - Auto-timestamp generation
    - Invalid payload rejection via AssertionError
    """
    # Verify exactly 9 event types exist
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
        assert hasattr(DynamicInferenceEventType, event_type), f"Missing: {event_type}"
    assert len(DynamicInferenceEventType) == 9

    # Verify all values are unique
    values = [e.value for e in DynamicInferenceEventType]
    assert len(values) == len(set(values)), "Duplicate event type values"

    # Create events that require no payload
    no_payload_types = [
        DynamicInferenceEventType.ADD_ENGINE,
        DynamicInferenceEventType.ADD_CONTEXT,
        DynamicInferenceEventType.PAUSE,
        DynamicInferenceEventType.EVICT,
        DynamicInferenceEventType.FINISH,
        DynamicInferenceEventType.FAIL,
    ]
    for event_type in no_payload_types:
        event = DynamicInferenceEvent(type=event_type)
        assert event.type == event_type
        assert event.payload is None
        assert event.timestamp is not None
        # Verify these types reject non-None payloads
        with pytest.raises(AssertionError):
            DynamicInferenceEvent(type=event_type, payload="not allowed")

    # Create GENERATED_TOKEN event with dict payload
    token_event = DynamicInferenceEvent(
        type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"token_id": 42}
    )
    assert token_event.type == DynamicInferenceEventType.GENERATED_TOKEN
    assert token_event.payload == {"token_id": 42}
    assert token_event.timestamp is not None

    # GENERATED_TOKEN rejects None, string, float, and int payloads (must be dict with "token_id")
    with pytest.raises(AssertionError):
        DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN)
    with pytest.raises(AssertionError):
        DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN, payload="bad")
    with pytest.raises(AssertionError):
        DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN, payload=3.14)
    with pytest.raises(AssertionError):
        DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN, payload=42)

    # Create error events with exception payloads
    transient_error = RequestOverflowError(request_id=1, message="Transient")
    event_transient = DynamicInferenceEvent(
        type=DynamicInferenceEventType.ERROR_TRANSIENT, payload=transient_error
    )
    assert event_transient.type == DynamicInferenceEventType.ERROR_TRANSIENT
    assert event_transient.payload is transient_error

    nontransient_error = MaxSequenceLengthOverflowError(request_id=2, message="Fatal")
    event_nontransient = DynamicInferenceEvent(
        type=DynamicInferenceEventType.ERROR_NONTRANSIENT, payload=nontransient_error
    )
    assert event_nontransient.type == DynamicInferenceEventType.ERROR_NONTRANSIENT
    assert event_nontransient.payload is nontransient_error

    # Error events reject None payloads
    with pytest.raises(AssertionError):
        DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_TRANSIENT)
    with pytest.raises(AssertionError):
        DynamicInferenceEvent(type=DynamicInferenceEventType.ERROR_NONTRANSIENT)

    # Verify auto-timestamp is reasonable
    before = time.time()
    event = DynamicInferenceEvent(type=DynamicInferenceEventType.ADD_ENGINE)
    after = time.time()
    assert before <= event.timestamp <= after


# ============================================================================
# Test 2: Successful Request Lifecycle with Serialization Roundtrip
# ============================================================================


def test_successful_request_lifecycle_with_serialization_roundtrip(basic_request):
    """Test full request lifecycle: ADD_ENGINE -> ADD_CONTEXT -> GENERATED_TOKEN(s) -> FINISH.

    Coverage:
    - Full successful lifecycle event sequence
    - Event order preservation
    - Request serialization/deserialization roundtrip
    - generated_tokens as direct List[int] field
    """
    # Build complete successful lifecycle
    basic_request.add_event_add_engine()
    basic_request.add_event_add_context()
    basic_request.generated_tokens.extend([100, 200, 300])
    basic_request.add_event_generated_token(100)
    basic_request.add_event_generated_token(200)
    basic_request.add_event_generated_token(300)
    basic_request.add_event_finish()

    # Verify event sequence
    event_types = [e.type for e in basic_request.events]
    assert event_types == [
        DynamicInferenceEventType.ADD_ENGINE,
        DynamicInferenceEventType.ADD_CONTEXT,
        DynamicInferenceEventType.GENERATED_TOKEN,
        DynamicInferenceEventType.GENERATED_TOKEN,
        DynamicInferenceEventType.GENERATED_TOKEN,
        DynamicInferenceEventType.FINISH,
    ]

    # Verify generated_tokens is a direct list field
    assert basic_request.generated_tokens == [100, 200, 300]
    assert isinstance(basic_request.generated_tokens, list)

    # Serialize request
    serialized = basic_request.serialize()
    assert 'events' in serialized
    assert len(serialized['events']) == 6
    assert serialized['generated_tokens'] == [100, 200, 300]

    # Verify GENERATED_TOKEN events serialize with {"token_id": ...} payload
    token_events = [e for e in serialized['events'] if e['type'] == 'GENERATED_TOKEN']
    assert len(token_events) == 3
    assert token_events[0]['payload'] == {"token_id": 100}
    assert token_events[1]['payload'] == {"token_id": 200}
    assert token_events[2]['payload'] == {"token_id": 300}

    # Deserialize and verify restoration
    restored = DynamicInferenceRequest.deserialize(serialized)
    assert len(restored.events) == 6
    assert restored.generated_tokens == [100, 200, 300]

    # Verify event order preserved after roundtrip
    restored_types = [e.type for e in restored.events]
    assert restored_types == event_types

    # Verify timestamps preserved
    for orig, rest in zip(basic_request.events, restored.events):
        assert rest.timestamp == orig.timestamp


# ============================================================================
# Test 3: Error Event Serialization with ContextErrorFactory
# ============================================================================


def test_error_event_serialization_with_context_error_factory():
    """Test error event serialization using ContextErrorFactory for all error types.

    Coverage:
    - ERROR_TRANSIENT and ERROR_NONTRANSIENT events
    - All 4 error types: RequestOverflowError, TokenOverflowError,
      MaxSequenceLengthOverflowError, BlockOverflowError
    - Error attribute preservation (request_id, message, is_transient) through roundtrip
    """
    error_cases = [
        # (event_type, error_class, request_id, message, is_transient)
        (
            DynamicInferenceEventType.ERROR_TRANSIENT,
            RequestOverflowError,
            1,
            "Max requests exceeded",
            True,
        ),
        (
            DynamicInferenceEventType.ERROR_TRANSIENT,
            TokenOverflowError,
            2,
            "Token limit exceeded",
            True,
        ),
        (DynamicInferenceEventType.ERROR_TRANSIENT, BlockOverflowError, 3, "Block overflow", True),
        (
            DynamicInferenceEventType.ERROR_NONTRANSIENT,
            MaxSequenceLengthOverflowError,
            4,
            "Sequence too long",
            False,
        ),
    ]

    for event_type, error_class, req_id, message, is_transient in error_cases:
        # Create error and event
        error = error_class(request_id=req_id, message=message)
        event = DynamicInferenceEvent(type=event_type, payload=error)

        # Verify error attributes before serialization
        assert event.payload.request_id == req_id
        assert event.payload.message == message
        assert event.payload.is_transient == is_transient

        # Serialize and deserialize
        serialized = event.serialize()
        assert serialized['type'] == event_type.name
        assert serialized['payload'] is not None

        deserialized = DynamicInferenceEvent.deserialize(serialized)

        # Verify restoration
        assert deserialized.type == event_type
        assert deserialized.payload.request_id == req_id
        assert deserialized.payload.message == message
        assert deserialized.payload.is_transient == is_transient
        assert deserialized.timestamp == event.timestamp


# ============================================================================
# Test 4: Generated Tokens Property Filters and Preserves Order
# ============================================================================


def test_generated_tokens_is_direct_list_field(basic_request):
    """Test generated_tokens is a direct List[int] field, independent of events.

    Coverage:
    - generated_tokens starts as empty list
    - Direct append works independently of events
    - Token ID edge cases: 0, large values (2^62)
    - Field is serialized/deserialized correctly
    """
    # Start with empty list
    assert basic_request.generated_tokens == []
    assert isinstance(basic_request.generated_tokens, list)

    # Direct append works
    basic_request.generated_tokens.append(0)
    assert basic_request.generated_tokens == [0]

    basic_request.generated_tokens.append(100)
    assert basic_request.generated_tokens == [0, 100]

    # Large token ID
    large_id = 2**62
    basic_request.generated_tokens.append(large_id)
    assert basic_request.generated_tokens == [0, 100, large_id]

    basic_request.generated_tokens.append(200)
    assert basic_request.generated_tokens == [0, 100, large_id, 200]

    # Events are independent - add some events
    basic_request.add_event_add_engine()
    basic_request.add_event_add_context()
    basic_request.add_event_finish()
    assert len(basic_request.events) == 3
    assert len(basic_request.generated_tokens) == 4

    # Serialization roundtrip preserves generated_tokens
    serialized = basic_request.serialize()
    assert serialized['generated_tokens'] == [0, 100, large_id, 200]
    restored = DynamicInferenceRequest.deserialize(serialized)
    assert restored.generated_tokens == [0, 100, large_id, 200]


# ============================================================================
# Test 5: Request Record Merge Across Eviction Recovery
# ============================================================================


def test_request_record_merge_across_eviction_recovery():
    """Test RequestRecord.merge() combines events from multiple checkpoints across eviction/recovery.

    Coverage:
    - RequestRecord creation and merge()
    - Eviction lifecycle simulation (first attempt evicted, second attempt succeeds)
    - Multi-checkpoint event combination
    - Token aggregation from all checkpoints
    """
    # First request: starts, generates some tokens, then gets evicted
    req1 = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=10),
    )
    req1.add_event_add_engine()
    req1.add_event_add_context()
    req1.generated_tokens.extend([100, 101])
    req1.add_event_generated_token(100)
    req1.add_event_generated_token(101)
    req1.add_event_evict()

    # Second request: recovered, generates more tokens, completes
    req2 = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3, 100, 101], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=8),
    )
    req2.add_event_add_engine()
    req2.add_event_add_context()
    req2.generated_tokens.extend([200, 201])
    req2.add_event_generated_token(200)
    req2.add_event_generated_token(201)
    req2.add_event_finish()

    # Create record with both checkpoints
    record = DynamicInferenceRequestRecord()
    record.requests.append(req1)
    record.requests.append(req2)

    # Merge
    merged = record.merge()

    # Verify all events present (5 from req1 + 5 from req2)
    assert len(merged.events) == 10

    # Verify event types include eviction
    event_types = [e.type for e in merged.events]
    assert DynamicInferenceEventType.EVICT in event_types
    assert event_types.count(DynamicInferenceEventType.ADD_ENGINE) == 2
    assert event_types.count(DynamicInferenceEventType.ADD_CONTEXT) == 2

    # Verify all tokens combined correctly
    assert merged.generated_tokens == [100, 101, 200, 201]

    # Test from_request convenience method
    single_record = DynamicInferenceRequestRecord.from_request(req1)
    single_merged = single_record.merge()
    assert len(single_merged.events) == len(req1.events)
    assert single_merged.generated_tokens == req1.generated_tokens


# ============================================================================
# Test 6: TTFT Calculation from Event Timestamps
# ============================================================================


def test_ttft_calculation_from_event_timestamps(basic_request):
    """Test TTFT (time-to-first-token) calculation using ADD_ENGINE and GENERATED_TOKEN timestamps.

    Coverage:
    - Timestamp accuracy
    - ADD_ENGINE/ADD_CONTEXT split for TTFT calculation
    - Timestamps are monotonically increasing
    - Explicit timestamp preservation through serialization
    """
    # Add events with deliberate delays
    basic_request.add_event_add_engine()
    time.sleep(0.01)
    basic_request.add_event_add_context()
    time.sleep(0.01)
    basic_request.add_event_generated_token(100)

    # Find timestamps
    add_engine_time = None
    first_token_time = None
    for event in basic_request.events:
        if event.type == DynamicInferenceEventType.ADD_ENGINE:
            add_engine_time = event.timestamp
        elif event.type == DynamicInferenceEventType.GENERATED_TOKEN and first_token_time is None:
            first_token_time = event.timestamp

    assert add_engine_time is not None
    assert first_token_time is not None

    # Calculate TTFT
    ttft = first_token_time - add_engine_time
    assert ttft > 0, "TTFT should be positive"
    assert ttft > 0.01, "TTFT should include deliberate delay"
    assert ttft < 1.0, "TTFT should be reasonable in test"

    # Verify timestamps are monotonically increasing
    timestamps = [e.timestamp for e in basic_request.events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], "Timestamps should not decrease"

    # Test explicit timestamp preservation through serialization
    explicit_time = 1700000000.123456
    event_with_explicit = DynamicInferenceEvent(
        type=DynamicInferenceEventType.FINISH, timestamp=explicit_time
    )
    serialized = event_with_explicit.serialize()
    deserialized = DynamicInferenceEvent.deserialize(serialized)
    assert deserialized.timestamp == explicit_time


# ============================================================================
# Test 7: Failed Request Lifecycle with Error Propagation
# ============================================================================


def test_failed_request_lifecycle_with_error_propagation(basic_request):
    """Test failed request lifecycle: transient errors followed by nontransient error and FAIL.

    Coverage:
    - FAIL event
    - Error event -> failure path
    - Multiple transient errors followed by nontransient
    - Error serialization preserved in full request roundtrip
    """
    # Build failure lifecycle
    basic_request.add_event_add_engine()

    # Transient errors (retryable)
    transient1 = TokenOverflowError(request_id=1, message="Token overflow 1")
    basic_request.add_event_error_transient(transient1)

    transient2 = BlockOverflowError(request_id=1, message="Block overflow")
    basic_request.add_event_error_transient(transient2)

    # Fatal error
    fatal = MaxSequenceLengthOverflowError(request_id=1, message="Sequence too long")
    basic_request.add_event_error_nontransient(fatal)

    # Request fails
    basic_request.add_event_fail()

    # Verify event sequence
    event_types = [e.type for e in basic_request.events]
    assert event_types == [
        DynamicInferenceEventType.ADD_ENGINE,
        DynamicInferenceEventType.ERROR_TRANSIENT,
        DynamicInferenceEventType.ERROR_TRANSIENT,
        DynamicInferenceEventType.ERROR_NONTRANSIENT,
        DynamicInferenceEventType.FAIL,
    ]

    # Serialize entire request
    serialized = basic_request.serialize()
    assert len(serialized['events']) == 5

    # Deserialize
    restored = DynamicInferenceRequest.deserialize(serialized)
    assert len(restored.events) == 5

    # Verify error payloads preserved
    error_events = [e for e in restored.events if 'ERROR' in e.type.name]
    assert len(error_events) == 3

    # Check first transient error
    assert error_events[0].payload.message == "Token overflow 1"
    assert error_events[0].payload.is_transient is True

    # Check second transient error
    assert error_events[1].payload.message == "Block overflow"
    assert error_events[1].payload.is_transient is True

    # Check nontransient error
    assert error_events[2].payload.message == "Sequence too long"
    assert error_events[2].payload.is_transient is False


# ============================================================================
# Test 8: Event String Representation for All Types
# ============================================================================


def test_event_str_representation_for_all_types():
    """Test __str__ method produces correct format for all event types.

    Coverage:
    - Simple events show type and timestamp
    - GENERATED_TOKEN shows token=<id>
    - Error events show error class name
    - Format consistency across types
    """
    # Simple events (no payload)
    simple_types = [
        DynamicInferenceEventType.ADD_ENGINE,
        DynamicInferenceEventType.ADD_CONTEXT,
        DynamicInferenceEventType.PAUSE,
        DynamicInferenceEventType.EVICT,
        DynamicInferenceEventType.FINISH,
        DynamicInferenceEventType.FAIL,
    ]
    for event_type in simple_types:
        event = DynamicInferenceEvent(type=event_type)
        str_repr = str(event)
        assert event_type.name in str_repr, f"{event_type.name} not in {str_repr}"
        # Timestamp should appear (at least first few digits)
        assert str(event.timestamp)[:5] in str_repr

    # GENERATED_TOKEN shows token=<id>
    token_event = DynamicInferenceEvent(
        type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"token_id": 42}
    )
    str_repr = str(token_event)
    assert 'GENERATED_TOKEN' in str_repr
    assert 'token=42' in str_repr

    # Large token ID
    large_token_event = DynamicInferenceEvent(
        type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"token_id": 9999999}
    )
    assert 'token=9999999' in str(large_token_event)

    # Error events show error class name
    error_cases = [
        (DynamicInferenceEventType.ERROR_TRANSIENT, RequestOverflowError, "RequestOverflowError"),
        (DynamicInferenceEventType.ERROR_TRANSIENT, TokenOverflowError, "TokenOverflowError"),
        (DynamicInferenceEventType.ERROR_TRANSIENT, BlockOverflowError, "BlockOverflowError"),
        (
            DynamicInferenceEventType.ERROR_NONTRANSIENT,
            MaxSequenceLengthOverflowError,
            "MaxSequenceLengthOverflowError",
        ),
    ]
    for event_type, error_class, expected_name in error_cases:
        error = error_class(request_id=1, message="Test error")
        event = DynamicInferenceEvent(type=event_type, payload=error)
        str_repr = str(event)
        assert event_type.name in str_repr
        assert expected_name in str_repr


# ============================================================================
# Test 9: Complex Multi-Pause-Evict Lifecycle with Record
# ============================================================================


def test_complex_multi_pause_evict_lifecycle_with_record():
    """Test complex real-world scenario with multiple pauses, evictions, and recoveries.

    Coverage:
    - Complex lifecycle with multiple PAUSE events
    - Multiple evictions and recoveries
    - Full roundtrip with RequestRecord
    - Complete event history and all tokens present
    """
    # First attempt: starts, generates token, pauses, generates more, gets evicted
    req1 = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=20),
    )
    req1.add_event_add_engine()
    req1.add_event_add_context()
    req1.generated_tokens.append(10)
    req1.add_event_generated_token(10)
    req1.add_event_pause()  # Paused for higher priority request
    req1.generated_tokens.append(11)
    req1.add_event_generated_token(11)
    req1.generated_tokens.append(12)
    req1.add_event_generated_token(12)
    req1.add_event_evict()  # Memory pressure

    # Second attempt: recovered, generates tokens, evicted again
    req2 = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3, 10, 11, 12], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=17),
    )
    req2.add_event_add_engine()
    req2.add_event_add_context()
    req2.generated_tokens.append(20)
    req2.add_event_generated_token(20)
    req2.add_event_pause()
    req2.add_event_pause()  # Paused twice
    req2.generated_tokens.append(21)
    req2.add_event_generated_token(21)
    req2.add_event_evict()

    # Third attempt: finally completes
    req3 = DynamicInferenceRequest(
        request_id=1,
        prompt_tokens=torch.tensor([1, 2, 3, 10, 11, 12, 20, 21], dtype=torch.int64),
        sampling_params=SamplingParams(num_tokens_to_generate=15),
    )
    req3.add_event_add_engine()
    req3.add_event_add_context()
    req3.generated_tokens.extend([30, 31, 32])
    req3.add_event_generated_token(30)
    req3.add_event_generated_token(31)
    req3.add_event_generated_token(32)
    req3.add_event_finish()

    # Create record with all checkpoints
    record = DynamicInferenceRequestRecord()
    record.requests.append(req1)
    record.requests.append(req2)
    record.requests.append(req3)

    # Serialize and deserialize record
    serialized = record.serialize()
    assert len(serialized['requests']) == 3

    # Merge all checkpoints
    merged = record.merge()

    # Verify all tokens collected
    assert merged.generated_tokens == [10, 11, 12, 20, 21, 30, 31, 32]

    # Count event types in merged history
    event_types = [e.type for e in merged.events]
    assert event_types.count(DynamicInferenceEventType.ADD_ENGINE) == 3
    assert event_types.count(DynamicInferenceEventType.ADD_CONTEXT) == 3
    assert event_types.count(DynamicInferenceEventType.EVICT) == 2
    assert event_types.count(DynamicInferenceEventType.PAUSE) == 3
    assert event_types.count(DynamicInferenceEventType.FINISH) == 1
    assert event_types.count(DynamicInferenceEventType.GENERATED_TOKEN) == 8

    # Total events: 7 from req1 + 7 from req2 + 6 from req3 = 20
    assert len(merged.events) == 20

    # Test serialization roundtrip of entire record
    serialized = record.serialize()
    assert len(serialized['requests']) == 3

    # Verify all events are dicts (no compact mode)
    first_request_events = serialized['requests'][0]['events']
    assert all(isinstance(e, dict) for e in first_request_events)

    # Verify GENERATED_TOKEN events use {"token_id": ...} payload format
    token_events = [e for e in first_request_events if e['type'] == 'GENERATED_TOKEN']
    assert token_events[0]['payload'] == {"token_id": 10}
    assert token_events[1]['payload'] == {"token_id": 11}
    assert token_events[2]['payload'] == {"token_id": 12}
