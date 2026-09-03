# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.training.training import _should_reset_logging_interval


@pytest.mark.parametrize(
    ("is_first_iteration", "log_interval", "expected"),
    [
        (True, 1, True),
        (True, 2, False),
        (False, 1, True),
        (False, 2, True),
    ],
    ids=[
        "first-iteration-single-iteration-window",
        "first-iteration-multi-iteration-window",
        "later-iteration-single-iteration-window",
        "later-iteration-multi-iteration-window",
    ],
)
def test_should_reset_logging_interval(is_first_iteration, log_interval, expected):
    """Reset each completed logging window without truncating a multi-iteration first window."""
    assert _should_reset_logging_interval(is_first_iteration, log_interval) is expected