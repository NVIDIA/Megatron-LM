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
    _get_rank_or_unknown,
)


class TestGetRankOrUnknown:
    """Unit tests for the ``_get_rank_or_unknown`` helper."""

    def test_returns_rank_when_initialised(self):
        """When distributed is initialised the rank should be returned."""
        with mock.patch("torch.distributed.is_initialized", return_value=True), \
             mock.patch("torch.distributed.get_rank", return_value=7):
            assert _get_rank_or_unknown() == "7"

    def test_returns_unknown_when_not_initialised(self):
        """When the process group is already torn down, return '?'."""
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            assert _get_rank_or_unknown() == "?"

    def test_returns_unknown_on_exception(self):
        """If ``is_initialized`` itself raises (e.g. interpreter teardown),
        the helper should still return '?' without propagating."""
        with mock.patch(
            "torch.distributed.is_initialized", side_effect=RuntimeError("shutdown")
        ):
            assert _get_rank_or_unknown() == "?"


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
