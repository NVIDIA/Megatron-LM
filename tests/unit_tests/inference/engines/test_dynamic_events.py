# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for DynamicInferenceEvent and request event lifecycle."""

import pytest
import torch

from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    DynamicInferenceRequest,
)
from tests.unit_tests.inference.test_utils import TestPriority

# Set this to control which tests run:
# - TestPriority.CRITICAL: Run only critical tests
# - TestPriority.IMPORTANT: Run critical + important tests
# - TestPriority.MEDIUM: Run critical + important + medium tests
# - TestPriority.LOW: Run all tests (default)
TEST_PRIORITY = TestPriority.LOW


class TestDynamicInferenceEventType:
    """Tests for the DynamicInferenceEventType enum."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_new_event_types_exist(self):
        """Verify ADD_ENGINE, ADD_CONTEXT, GENERATED_TOKEN event types exist."""
        assert hasattr(DynamicInferenceEventType, 'ADD_ENGINE')
        assert hasattr(DynamicInferenceEventType, 'ADD_CONTEXT')
        assert hasattr(DynamicInferenceEventType, 'GENERATED_TOKEN')

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_old_event_types_removed(self):
        """Verify old ADD and FIRST_TOKEN event types no longer exist."""
        assert not hasattr(DynamicInferenceEventType, 'ADD')
        assert not hasattr(DynamicInferenceEventType, 'FIRST_TOKEN')

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_all_event_types_exist(self):
        """Verify all expected event types are defined."""
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


class TestDynamicInferenceEvent:
    """Tests for DynamicInferenceEvent creation and serialization."""

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    @pytest.mark.parametrize(
        "event_type",
        [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
        ],
    )
    def test_event_creation(self, event_type):
        """Test creating events with new event types."""
        event = DynamicInferenceEvent(type=event_type)
        assert event.type == event_type
        assert event.payload is None
        assert event.timestamp is not None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_token_event_creation(self):
        """Test creating GENERATED_TOKEN events with token payload."""
        token_id = 42
        event = DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN, payload=token_id)
        assert event.type == DynamicInferenceEventType.GENERATED_TOKEN
        assert event.payload == token_id
        assert event.timestamp is not None

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    @pytest.mark.parametrize(
        "event_type",
        [
            DynamicInferenceEventType.ADD_ENGINE,
            DynamicInferenceEventType.ADD_CONTEXT,
        ],
    )
    def test_event_serialization_roundtrip(self, event_type):
        """Test serialize/deserialize preserves event data."""
        original = DynamicInferenceEvent(type=event_type)
        serialized = original.serialize()

        assert serialized['type'] == event_type.name
        assert 'timestamp' in serialized

        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.type == original.type
        assert deserialized.timestamp == original.timestamp

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_generated_token_serialization_roundtrip(self):
        """Test serialize/deserialize preserves GENERATED_TOKEN event data including payload."""
        token_id = 123
        original = DynamicInferenceEvent(type=DynamicInferenceEventType.GENERATED_TOKEN, payload=token_id)
        serialized = original.serialize()

        assert serialized['type'] == 'GENERATED_TOKEN'
        assert serialized['payload'] == token_id
        assert 'timestamp' in serialized

        deserialized = DynamicInferenceEvent.deserialize(serialized)
        assert deserialized.type == original.type
        assert deserialized.payload == token_id
        assert deserialized.timestamp == original.timestamp


class TestRequestEventMethods:
    """Tests for DynamicInferenceRequest event helper methods."""

    @pytest.fixture
    def req(self):
        """Create a basic request for testing."""
        return DynamicInferenceRequest(
            request_id=1,
            prompt_tokens=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        )

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_add_engine(self, req):
        """Test add_event_add_engine() method."""
        assert len(req.events) == 0
        req.add_event_add_engine()
        assert len(req.events) == 1
        assert req.events[0].type == DynamicInferenceEventType.ADD_ENGINE

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_add_context(self, req):
        """Test add_event_add_context() method."""
        req.add_event_add_context()
        assert len(req.events) == 1
        assert req.events[0].type == DynamicInferenceEventType.ADD_CONTEXT

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_add_event_generated_token(self, req):
        """Test add_event_generated_token() method."""
        token_id = 42
        req.add_event_generated_token(token_id)
        assert len(req.events) == 1
        assert req.events[0].type == DynamicInferenceEventType.GENERATED_TOKEN
        assert req.events[0].payload == token_id

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.CRITICAL, reason="Test priority not met"
    )
    def test_generated_tokens_property(self, req):
        """Test that generated_tokens property returns correct token list."""
        assert req.generated_tokens == []

        req.add_event_generated_token(10)
        req.add_event_generated_token(20)
        req.add_event_generated_token(30)

        assert req.generated_tokens == [10, 20, 30]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_generated_tokens_filters_only_token_events(self, req):
        """Test that generated_tokens only includes GENERATED_TOKEN events."""
        req.add_event_add_engine()
        req.add_event_add_context()
        req.add_event_generated_token(100)
        req.add_event_generated_token(200)
        req.add_event_finish()

        # Should only include the token payloads, not other events
        assert req.generated_tokens == [100, 200]

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.IMPORTANT, reason="Test priority not met"
    )
    def test_event_sequence(self, req):
        """Test adding multiple events creates correct sequence."""
        req.add_event_add_engine()
        req.add_event_add_context()
        req.add_event_generated_token(42)
        req.add_event_finish()

        event_types = [e.type.name for e in req.events]
        assert event_types == ['ADD_ENGINE', 'ADD_CONTEXT', 'GENERATED_TOKEN', 'FINISH']

    @pytest.mark.skipif(
        TEST_PRIORITY < TestPriority.MEDIUM, reason="Test priority not met"
    )
    def test_event_timestamps_increasing(self, req):
        """Test that event timestamps are monotonically increasing."""
        import time

        req.add_event_add_engine()
        time.sleep(0.001)
        req.add_event_add_context()
        time.sleep(0.001)
        req.add_event_generated_token(42)

        timestamps = [e.timestamp for e in req.events]
        assert timestamps == sorted(timestamps), "Timestamps should be increasing"
