# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared test utilities for inference tests."""

from enum import IntEnum


class TestPriority(IntEnum):
    """Test priority levels for selective test execution.

    Priority ordering (lower number = higher priority):
    - CRITICAL (1): Fundamental correctness (hash collisions, safety, TTFT)
    - IMPORTANT (2): Robustness and common edge cases
    - MEDIUM (3): Lifecycle integration and additional edge cases
    - LOW (4): Observability, metrics, and advanced scenarios

    Usage with pytest:
        @pytest.mark.skipif(TEST_PRIORITY < TestPriority.MEDIUM,
                           reason="Test priority not met")
        def test_something():
            ...

    Run only critical tests:
        TEST_PRIORITY = TestPriority.CRITICAL

    Run all tests:
        TEST_PRIORITY = TestPriority.LOW
    """

    CRITICAL = 1  # Must always pass - core functionality
    IMPORTANT = 2  # Should pass - significant features
    MEDIUM = 3  # Nice to have - edge cases
    LOW = 4  # Optional - performance, stress tests
