# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Tests for slurm_utils module."""

import os
from unittest.mock import patch

from megatron.core._slurm_utils import (
    is_slurm_job,
    resolve_slurm_local_rank,
    resolve_slurm_rank,
    resolve_slurm_world_size,
)


class TestIsSLURMJob:
    """Test is_slurm_job function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"})
    def test_is_slurm_job_true(self):
        """Test detection returns True when SLURM_NTASKS is set."""
        assert is_slurm_job() is True

    @patch.dict(os.environ, {}, clear=True)
    def test_is_slurm_job_false(self):
        """Test detection returns False when SLURM_NTASKS is not set."""
        assert is_slurm_job() is False


class TestResolveSLURMRank:
    """Test resolve_slurm_rank function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_PROCID": "5"}, clear=True)
    def test_resolve_slurm_rank(self):
        """Test resolving rank from SLURM_PROCID."""
        assert resolve_slurm_rank() == 5

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_slurm_rank_not_slurm(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_rank() is None

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"}, clear=True)
    def test_resolve_slurm_rank_missing_procid(self):
        """Test returns None when SLURM_PROCID not set."""
        assert resolve_slurm_rank() is None


class TestResolveSLURMWorldSize:
    """Test resolve_slurm_world_size function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "16"}, clear=True)
    def test_resolve_slurm_world_size(self):
        """Test resolving world size from SLURM_NTASKS."""
        assert resolve_slurm_world_size() == 16

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_slurm_world_size_not_slurm(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_world_size() is None


class TestResolveSLURMLocalRank:
    """Test resolve_slurm_local_rank function."""

    @patch.dict(os.environ, {"SLURM_NTASKS": "8", "SLURM_LOCALID": "3"}, clear=True)
    def test_resolve_slurm_local_rank(self):
        """Test resolving local rank from SLURM_LOCALID."""
        assert resolve_slurm_local_rank() == 3

    @patch.dict(os.environ, {}, clear=True)
    def test_resolve_slurm_local_rank_not_slurm(self):
        """Test returns None when not in SLURM environment."""
        assert resolve_slurm_local_rank() is None

    @patch.dict(os.environ, {"SLURM_NTASKS": "8"}, clear=True)
    def test_resolve_slurm_local_rank_missing_localid(self):
        """Test returns None when SLURM_LOCALID not set."""
        assert resolve_slurm_local_rank() is None
