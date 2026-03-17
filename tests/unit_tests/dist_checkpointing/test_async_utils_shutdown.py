# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Tests for shutdown-safety of async checkpoint callers.

Verifies that ``PersistentAsyncCaller.__del__`` / ``close()`` does not raise
when the distributed process group has already been destroyed (the scenario
described in issue #3775).
"""

from unittest import mock

import pytest
import torch

from megatron.core.dist_checkpointing.strategies.async_utils import (
    PersistentAsyncCaller,
    TemporalAsyncCaller,
)


class TestPersistentAsyncCallerShutdown:
    """Verify ``PersistentAsyncCaller`` does not crash during GC shutdown."""

    def test_close_without_process_group(self):
        """Calling close() after process group destruction must not raise."""
        caller = PersistentAsyncCaller()
        # Simulate the state where no process was ever spawned (process is None)
        # but close() still logs with the rank.
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            # Must not raise
            caller.close()

    def test_del_without_process_group(self):
        """``__del__`` must not raise when dist is uninitialised."""
        caller = PersistentAsyncCaller()
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            # Must not raise
            caller.__del__()


class TestTemporalAsyncCallerShutdown:
    """Verify ``TemporalAsyncCaller`` does not crash during GC shutdown."""

    def test_close_without_process_group(self):
        """Calling close() after process group destruction must not raise."""
        caller = TemporalAsyncCaller()
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            caller.close()
