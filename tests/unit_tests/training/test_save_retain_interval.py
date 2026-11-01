# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""--save-retain-interval deletes only checkpoints older than the one just saved."""

import pytest

from megatron.training.checkpointing import _should_delete_previous_checkpoint


@pytest.mark.parametrize(
    "prev_iteration, iteration, expected",
    [
        (100, 200, True),  # The usual rotation: the previous save is replaced.
        (2000, 2100, False),  # Kept: a multiple of the retain interval.
        (0, 100, False),  # No previous checkpoint.
        (200, 200, False),  # Saving the same iteration again.
        (2700, 100, False),  # Restarted from scratch: the newer checkpoint survives.
        (2700, 2600, False),  # Rolled back: the newer checkpoint survives.
    ],
)
def test_only_older_checkpoints_are_deleted(prev_iteration, iteration, expected):
    assert _should_delete_previous_checkpoint(prev_iteration, iteration, 1000) is expected
