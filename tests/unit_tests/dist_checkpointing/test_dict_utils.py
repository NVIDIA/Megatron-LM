# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
import pytest

from megatron.core.dist_checkpointing.dict_utils import diff


@pytest.mark.parametrize("sequence_type", [list, tuple, np.array])
@pytest.mark.parametrize("extra_on_left", [True, False])
def test_diff_sequence_length_reports_nested_paths(sequence_type, extra_on_left):
    """Missing sequence elements carry the same path prefix as mismatched values."""
    shorter = {"layers": [sequence_type([1])]}
    longer = {"layers": [sequence_type([1, 2, 3])]}
    left, right = (longer, shorter) if extra_on_left else (shorter, longer)

    only_left, only_right, mismatch = diff(left, right, prefix=("checkpoint",))

    expected = [("checkpoint", "layers", 0, 2), ("checkpoint", "layers", 0, 1)]
    assert only_left == (expected if extra_on_left else [])
    assert only_right == ([] if extra_on_left else expected)
    assert not mismatch


def test_diff_empty_sequence_reports_index_paths():
    """Root-level missing elements use tuple paths too, preserving index order."""
    assert diff([1, 2], []) == ([(1,), (0,)], [], [])
    assert diff([], [1, 2]) == ([], [(1,), (0,)], [])
    assert diff([], []) == ([], [], [])


def test_diff_sequence_mismatch_preserves_paths():
    """Shared elements still report their recursive path rather than a missing key."""
    only_left, only_right, mismatch = diff({"layers": [1, 2]}, {"layers": [1, 3]})

    assert not only_left
    assert not only_right
    assert mismatch == [(("layers", 1), int, int)]
