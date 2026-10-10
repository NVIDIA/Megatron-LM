# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import pytest

from tests.functional_tests.python_test_utils.test_grpo_training_loop import validate_with_tolerance


@pytest.mark.parametrize(
    ('golden', 'actual'),
    [
        (1.0, float('nan')),
        (0.0, float('nan')),
        (float('inf'), 1.0),
        (float('-inf'), float('-inf')),
        (float('nan'), float('nan')),
    ],
)
def test_tolerance_comparison_rejects_nonfinite_samples(golden, actual):
    passing, mismatches = validate_with_tolerance({1: golden}, {1: actual}, relative_tolerance=0.01)

    assert passing is False
    assert len(mismatches) == 1
    assert 'non-finite' in mismatches[0]


def test_finite_tolerance_behavior_is_preserved():
    assert validate_with_tolerance({1: 1.0}, {1: 1.005}, relative_tolerance=0.01) == (True, [])
    assert validate_with_tolerance({1: 1.0}, {1: 2.0}, relative_tolerance=0.01)[0] is False
