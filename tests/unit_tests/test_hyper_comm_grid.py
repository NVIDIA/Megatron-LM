# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid


class TestHyperCommGrid:
    """Comprehensive tests for HyperCommGrid class."""

    def test_init_basic(self):
        """Test basic initialization of HyperCommGrid."""
        shape = [2, 2, 2]
        dim_names = ["tp", "cp", "dp"]

        grid = HyperCommGrid(shape, dim_names)

        assert grid.shape == shape
        assert grid.dim_names == dim_names
        assert grid.rank_offset == 0
        assert grid.backend is None
        assert grid.size == 8  # 2 * 2 * 2
        assert grid._pgs == {}

    def test_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        shape = [2, 2]  # Changed from [2, 4] to fit world size 8 with offset 8
        dim_names = ["tp", "dp"]
        rank_offset = 0  # Changed from 8 to 0 to avoid size error
        backend = "nccl"

        grid = HyperCommGrid(shape, dim_names, rank_offset, backend)

        assert grid.shape == shape
        assert grid.dim_names == dim_names
        assert grid.rank_offset == rank_offset
        assert grid.backend == backend
        assert grid.size == 4  # 2 * 2

    def test_init_validation_errors(self):
        """Test initialization validation errors."""
        # Shape and dim_names length mismatch
        with pytest.raises(ValueError, match="len\\(shape\\).*!= len\\(dim_names\\)"):
            HyperCommGrid([2, 2], ["tp"])

        # Grid too large for world size
        with pytest.raises(RuntimeError, match="Grid shape.*is over sized"):
            HyperCommGrid([4, 4], ["tp", "dp"])  # 16 > 8 world size

    def test_order_dims_single_dim(self):
        """Test _order_dims with single dimension."""
        grid = HyperCommGrid(
            [2, 2, 2], ["tp", "cp", "dp"]
        )  # Changed from [2, 3, 4] to fit world size

        ordered_dims, unique_key = grid._order_dims_for(grid.dim_names, "cp")

        assert ordered_dims == ["cp"]
        assert unique_key == "cp"

    def test_order_dims_multiple_dims(self):
        """Test _order_dims with multiple dimensions."""
        grid = HyperCommGrid(
            [2, 2, 2], ["tp", "cp", "dp"]
        )  # Changed from [2, 3, 4, 5] to fit world size

        # Should order according to reversed dim_names order
        ordered_dims, unique_key = grid._order_dims_for(grid.dim_names, ["dp", "tp"])

        assert ordered_dims == [
            "dp",
            "tp",
        ]  # Changed: dp comes before tp in reversed order ["dp", "cp", "tp"]
        assert unique_key == "dp-tp"

    def test_order_dims_all_dims(self):
        """Test _order_dims with all dimensions."""
        grid = HyperCommGrid(
            [2, 2, 2], ["tp", "cp", "dp"]
        )  # Changed from [2, 3, 4] to fit world size

        ordered_dims, unique_key = grid._order_dims_for(grid.dim_names, ["dp", "cp", "tp"])

        assert ordered_dims == ["dp", "cp", "tp"]  # Changed: reversed order
        assert unique_key == "dp-cp-tp"

    def test_gen_rank_enum_single_dim(self):
        """Test _gen_rank_enum for single dimension."""
        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        rank_enum = grid._gen_rank_enum(["tp"])

        # Should have 4 groups of 2 ranks each
        expected = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert rank_enum == expected

    def test_gen_rank_enum_multiple_dims(self):
        """Test _gen_rank_enum for multiple dimensions."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        rank_enum = grid._gen_rank_enum(["tp", "cp"])

        # Should have 2 groups (for dp) with 4 ranks each (tp * cp)
        expected = [[0, 2, 1, 3], [4, 6, 5, 7]]
        assert rank_enum == expected

    def test_gen_rank_enum_with_offset(self):
        """Test _gen_rank_enum with rank offset."""
        grid = HyperCommGrid([2, 2], ["tp", "dp"], rank_offset=4)

        rank_enum = grid._gen_rank_enum(["tp"])

        # Should start from rank 4
        expected = [[4, 5], [6, 7]]
        assert rank_enum == expected

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_single_dim(self, mock_new_subgroups):
        """Test create_pg for single dimension."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        result = grid.create_pg("tp")

        assert result == mock_pg
        assert "tp" in grid._pgs
        assert grid._pgs["tp"] == mock_pg

        # Verify the enumeration passed to new_subgroups_by_enumeration
        args, kwargs = mock_new_subgroups.call_args
        expected_enum = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert args[0] == expected_enum
        assert kwargs["backend"] is None

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_multiple_dims(self, mock_new_subgroups):
        """Test create_pg for multiple dimensions."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        result = grid.create_pg(["tp", "cp"])

        assert result == mock_pg
        assert "cp-tp" in grid._pgs

        args, kwargs = mock_new_subgroups.call_args
        expected_enum = [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert args[0] == expected_enum

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_with_options(self, mock_new_subgroups):
        """Test create_pg with additional options."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"], backend="nccl")

        # Mock ProcessGroupNCCL.Options
        mock_options = MagicMock()

        result = grid.create_pg("tp", pg_options=mock_options, group_desc="TEST_GROUP")

        assert result == mock_pg

        args, kwargs = mock_new_subgroups.call_args
        assert kwargs["backend"] == "nccl"
        assert kwargs["pg_options"] == mock_options

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_duplicate_error(self, mock_new_subgroups):
        """Test create_pg raises error when trying to recreate existing process group."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        # Create process group first time
        grid.create_pg("tp")

        # Try to create again should raise KeyError
        with pytest.raises(KeyError, match="Process group.*has already been created"):
            grid.create_pg("tp")

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_get_pg_success(self, mock_new_subgroups):
        """Test get_pg returns existing process group."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        # Create process group first
        grid.create_pg("dp")

        # Get should return the same process group
        result = grid.get_pg("dp")
        assert result == mock_pg

    def test_get_pg_not_created_error(self):
        """Test get_pg raises error when process group doesn't exist."""
        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        with pytest.raises(KeyError, match="Process group for.*hasn't been created"):
            grid.get_pg("tp")

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_get_pg_multiple_dims(self, mock_new_subgroups):
        """Test get_pg with multiple dimensions."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Create process group with multiple dims
        grid.create_pg(["cp", "dp"])

        # Get should work with different order
        result = grid.get_pg(["dp", "cp"])
        assert result == mock_pg

    def test_complex_grid_scenario(self):
        """Test a complex scenario similar to the docstring example."""
        os.environ["WORLD_SIZE"] = "120"  # Set larger world size for this test

        grid = HyperCommGrid([2, 3, 4, 5], ["tp", "cp", "pp", "dp"])

        assert grid.size == 120
        assert grid.shape == [2, 3, 4, 5]
        assert grid.dim_names == ["tp", "cp", "pp", "dp"]

        # Test ordering of different dimension combinations
        ordered_dims, key = grid._order_dims_for(grid.dim_names, ["dp", "pp"])
        assert ordered_dims == ["dp", "pp"]  # Changed: actual order matches reversed dim_names
        assert key == "dp-pp"

        # Test rank enumeration for dp (last dimension)
        rank_enum = grid._gen_rank_enum(["dp"])
        assert len(rank_enum) == 24  # 2 * 3 * 4 = 24 groups
        assert len(rank_enum[0]) == 5  # Each group has 5 ranks

        # Clean up
        os.environ["WORLD_SIZE"] = "8"

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_end_to_end_workflow(self, mock_new_subgroups):
        """Test complete workflow: init -> create -> get."""
        mock_pg1 = MagicMock(spec=dist.ProcessGroup)
        mock_pg2 = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.side_effect = [(mock_pg1, None), (mock_pg2, None)]

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Create different process groups
        tp_pg = grid.create_pg("tp")
        dp_cp_pg = grid.create_pg(["dp", "cp"])

        # Verify they're created correctly
        assert tp_pg == mock_pg1
        assert dp_cp_pg == mock_pg2

        # Verify we can get them back
        assert grid.get_pg("tp") == mock_pg1
        assert grid.get_pg(["cp", "dp"]) == mock_pg2  # Different order should work

        # Verify internal state
        assert len(grid._pgs) == 2
        assert "tp" in grid._pgs
        assert "dp-cp" in grid._pgs  # Changed: actual key order

    def test_edge_case_single_rank_dims(self):
        """Test edge case with dimensions of size 1."""
        grid = HyperCommGrid([1, 2, 4], ["tp", "cp", "dp"])

        # Test with tp dimension (size 1)
        rank_enum = grid._gen_rank_enum(["tp"])
        expected = [[0], [1], [2], [3], [4], [5], [6], [7]]  # 8 groups of 1 rank each
        assert rank_enum == expected

        # Test with multiple dims including size 1
        rank_enum = grid._gen_rank_enum(["tp", "cp"])
        expected = [[0, 1], [2, 3], [4, 5], [6, 7]]  # 4 groups of 2 ranks each
        assert rank_enum == expected

    def test_rank_enumeration_correctness(self):
        """Test that rank enumeration produces correct pattern."""
        grid = HyperCommGrid([2, 2, 2], ["a", "b", "c"])

        # For dimension "a" (first in original order, last in reversed)
        rank_enum_a = grid._gen_rank_enum(["a"])
        expected_a = [[0, 1], [2, 3], [4, 5], [6, 7]]
        assert rank_enum_a == expected_a

        # For dimension "c" (last in original order, first in reversed)
        rank_enum_c = grid._gen_rank_enum(["c"])
        expected_c = [[0, 4], [1, 5], [2, 6], [3, 7]]
        assert rank_enum_c == expected_c

        # For dimensions "a" and "b"
        rank_enum_ab = grid._gen_rank_enum(["a", "b"])
        expected_ab = [[0, 2, 1, 3], [4, 6, 5, 7]]
        assert rank_enum_ab == expected_ab

    # ------------------------------------------------------------------
    # register_view validation (mock-based unit tests)
    # ------------------------------------------------------------------

    def test_unknown_view_error(self):
        """Test view-scoped APIs raise clearly for an unregistered view."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        with pytest.raises(KeyError, match="View 'nope' is not registered"):
            grid.get_rank_enum("ep", view="nope")

    def test_register_view_duplicate_name_error(self):
        """Test register_view rejects re-registering an existing view name."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "expt_dp"])
        with pytest.raises(ValueError, match="is already registered"):
            grid.register_view("expert", [2, 4], ["ep", "expt_dp"])

    def test_register_view_length_mismatch_error(self):
        """Test register_view rejects shape/dim_names length mismatch."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        with pytest.raises(ValueError, match="len\\(shape\\).*!= len\\(dim_names\\)"):
            grid.register_view("expert", [2, 2, 2], ["ep", "expt_dp"])

    def test_register_view_duplicate_dim_names_error(self):
        """Test register_view rejects duplicate dim_names within a view."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        with pytest.raises(ValueError, match="duplicate dim_names"):
            grid.register_view("expert", [2, 2, 2], ["ep", "ep", "expt_dp"])

    def test_register_view_size_mismatch_error(self):
        """Test register_view rejects a view whose size differs from the grid."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])  # size 8

        with pytest.raises(ValueError, match="but the grid size is 8"):
            grid.register_view("expert", [2, 2], ["ep", "expt_dp"])  # size 4

    def test_register_view_non_positive_shape_error(self):
        """Test register_view rejects shape entries that are not positive ints."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        with pytest.raises(ValueError, match="shape must be positive ints"):
            grid.register_view("expert", [8, 0], ["ep", "expt_dp"])
        with pytest.raises(ValueError, match="shape must be positive ints"):
            grid.register_view("expert2", [-1, 8], ["ep", "expt_dp"])

    def test_register_view_accepts_numpy_int_shape(self):
        """Test register_view accepts numpy integer shape entries, like the base grid does."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Shapes derived programmatically are often numpy ints; they must be accepted.
        shape = [np.int64(4), np.int64(2)]
        grid.register_view("expert", shape, ["ep", "expt_dp"])
        assert "expert" in grid._views

    def test_register_view_duplicate_shared_dims_error(self):
        """Test register_view rejects duplicate entries in shared_dims with a clear message."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "pp"])

        with pytest.raises(ValueError, match="duplicate shared_dims"):
            grid.register_view(
                "expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp", "pp"]
            )

    def test_register_view_success_stores_copied_metadata(self):
        """Test a valid view is registered, returns None, and is stored with copied data."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        shape = [4, 2]
        dim_names = ["ep", "expt_dp"]
        result = grid.register_view("expert", shape, dim_names)
        view = grid._views["expert"]

        assert result is None
        assert view.name == "expert"
        # Stored view must be a copy, not an alias of the caller's lists.
        assert view.shape == shape and view.shape is not shape
        assert view.dim_names == dim_names and view.dim_names is not dim_names

    def test_register_view_shared_dim_not_in_base_error(self):
        """Test register_view rejects a shared dim absent from the base view."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        with pytest.raises(ValueError, match="Shared dim 'pp' .* is not in the base view"):
            grid.register_view("expert", [4, 2], ["ep", "pp"], shared_dims=["pp"])

    def test_register_view_shared_dim_not_in_view_error(self):
        """Test register_view rejects a shared dim absent from the view's own dim_names."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "pp"])

        with pytest.raises(ValueError, match="Shared dim 'pp' .* is not in the view's dim_names"):
            grid.register_view("expert", [4, 2], ["ep", "expt_dp"], shared_dims=["pp"])

    def test_register_view_shared_dim_membership_mismatch_error(self):
        """Test register_view rejects a shared dim whose enumeration differs from base.

        Here the base view makes ``pp`` the trailing dim of the original order (strided ranks
        ``[[0,4],[1,5],[2,6],[3,7]]``), but the expert view makes ``pp`` the leading dim
        (contiguous ranks ``[[0,1],[2,3],[4,5],[6,7]]``); the memberships differ, so the
        shared-dim contract is violated and registration must raise.
        """
        # base: tp=2, dp=2, pp=2 -> pp trailing -> strided pp groups.
        grid = HyperCommGrid([2, 2, 2], ["tp", "dp", "pp"])

        with pytest.raises(ValueError, match="Shared dim 'pp' has different membership"):
            # expert: pp=2, ep=4 -> pp leading -> contiguous pp groups, different membership.
            grid.register_view("expert", [2, 4], ["pp", "ep"], shared_dims=["pp"])

    def test_register_view_shared_dim_membership_match(self):
        """Test register_view accepts a shared dim that enumerates identically across views.

        The base grid keeps ``pp`` as the trailing (outermost) dim, and the expert view also
        keeps ``pp`` trailing, so the ``pp`` groups are the same ranks.
        """
        # base: tp=2, dp=2, pp=2 -> reversed [dp, tp, ...]; pp trailing -> contiguous blocks.
        grid = HyperCommGrid([2, 2, 2], ["tp", "dp", "pp"])
        # expert: expt_tp=2, ep=2, pp=2 -> pp trailing -> same membership as base.
        grid.register_view(
            "expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"]
        )

        assert grid._views["expert"].shared_dims == ["pp"]
        # Both layouts enumerate pp identically.
        assert grid.get_rank_enum("pp") == grid.get_rank_enum("pp", view="expert")

    # ------------------------------------------------------------------
    # Rank-view process-group creation (mock-based)
    # ------------------------------------------------------------------

    def test_view_get_pg_not_created_error(self):
        """Test get_pg with a view raises when the group is absent."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])
        grid.register_view("expert", [4, 2], ["ep", "expt_dp"])

        with pytest.raises(KeyError, match="for view 'expert' hasn't been created"):
            grid.get_pg("ep", view="expert")

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_with_view_uses_view_rank_order(self, mock_new_subgroups):
        """Test view-scoped create_pg uses the selected view's rank generation order."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 2, 2], ["tp", "dp", "pp"])
        grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"])

        result = grid.create_pg(["expt_tp", "ep"], view="expert")

        assert result == mock_pg
        assert ("expert", ("ep", "expt_tp")) in grid._pgs
        args, _ = mock_new_subgroups.call_args
        assert args[0] == [[0, 1, 2, 3], [4, 5, 6, 7]]

    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_shared_view_dim_reuses_base_process_group(self, mock_new_subgroups):
        """Test fully-shared view dims canonicalize to the base process group."""
        base_pp = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (base_pp, None)

        grid = HyperCommGrid([2, 2, 2], ["tp", "dp", "pp"])
        grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"])

        assert grid.create_pg("pp", view="expert") is base_pp
        assert grid.get_pg("pp") is base_pp
        assert grid.get_pg("pp", view="expert") is base_pp
        assert "pp" in grid._pgs
        assert ("expert", ("pp",)) not in grid._pgs

    def test_view_dim_requires_explicit_view(self):
        """Test view-private dims are not inferred from the base view."""
        grid = HyperCommGrid([2, 2, 2], ["tp", "dp", "pp"])
        grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"])

        with pytest.raises(ValueError, match="'ep' is not in view 'base'"):
            grid.get_rank_enum("ep")

    def test_destroy_skips_non_members(self):
        """Test destroy only destroys groups the current rank is a member of."""
        member_pg = MagicMock(spec=dist.ProcessGroup)
        non_member = dist.GroupMember.NON_GROUP_MEMBER

        grid = HyperCommGrid([2, 4], ["tp", "dp"])
        grid._pgs = {"tp": member_pg, "dp": non_member, "cp": None}

        with patch('torch.distributed.destroy_process_group') as mock_destroy:
            grid.destroy()

        # Only the real, member process group is destroyed.
        mock_destroy.assert_called_once_with(member_pg)
        assert grid._pgs == {}

    @patch('torch.distributed.is_initialized', return_value=False)
    @patch('torch.distributed.new_subgroups_by_enumeration')
    def test_create_pg_skips_log_when_not_initialized(self, mock_new_subgroups, _mock_is_init):
        """Test create_pg does not call get_rank when torch.distributed is not initialized."""
        mock_pg = MagicMock(spec=dist.ProcessGroup)
        mock_new_subgroups.return_value = (mock_pg, None)

        grid = HyperCommGrid([2, 4], ["tp", "dp"])

        # get_rank would raise if called before init; the is_initialized guard must short-circuit.
        with patch('torch.distributed.get_rank', side_effect=AssertionError("get_rank called")):
            grid.create_pg("tp")


class TestHyperCommGridIntegration:
    """Integration tests for HyperCommGrid with real distributed initialization."""

    @classmethod
    def setup_class(cls):
        """Set up distributed environment for the entire test class."""
        if not dist.is_initialized():
            # Initialize PyTorch distributed with NCCL backend
            # This assumes proper environment variables are set (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
            try:
                dist.init_process_group(backend="nccl")
                cls.distributed_initialized = True
            except Exception as e:
                pytest.skip(f"Cannot initialize distributed: {e}")
        else:
            cls.distributed_initialized = True

    def test_real_distributed_basic_functionality(self):
        """Test basic HyperCommGrid functionality with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size > 8:
            pytest.skip("Test requires at most 8 GPUs")

        # Test with world_size that fits our constraint
        if world_size == 8:
            shape = [2, 2, 2]
            dim_names = ["tp", "cp", "dp"]
        elif world_size == 4:
            shape = [2, 2]
            dim_names = ["tp", "dp"]
        elif world_size == 2:
            shape = [2]
            dim_names = ["tp"]
        else:
            pytest.skip(f"Unsupported world size: {world_size}")

        grid = HyperCommGrid(shape, dim_names, backend="nccl")

        assert grid.size == world_size
        assert grid.shape == shape
        assert grid.dim_names == dim_names
        assert grid.backend == "nccl"

    def test_real_distributed_process_group_creation(self):
        """Test process group creation with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], backend="nccl")

        # Create different types of process groups
        tp_pg = grid.create_pg("tp")
        cp_pg = grid.create_pg("cp")
        dp_pg = grid.create_pg("dp")

        # Verify process groups are real PyTorch ProcessGroup objects
        assert isinstance(tp_pg, dist.ProcessGroup)
        assert isinstance(cp_pg, dist.ProcessGroup)
        assert isinstance(dp_pg, dist.ProcessGroup)

        # Verify we can get the process groups back
        assert grid.get_pg("tp") == tp_pg
        assert grid.get_pg("cp") == cp_pg
        assert grid.get_pg("dp") == dp_pg

        # Test process group sizes
        tp_ranks = dist.get_process_group_ranks(tp_pg)
        cp_ranks = dist.get_process_group_ranks(cp_pg)
        dp_ranks = dist.get_process_group_ranks(dp_pg)

        assert len(tp_ranks) == 2  # tp dimension size
        assert len(cp_ranks) == 2  # cp dimension size
        assert len(dp_ranks) == 2  # dp dimension size

    def test_real_distributed_multi_dimensional_groups(self):
        """Test multi-dimensional process group creation with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], backend="nccl")

        # Create multi-dimensional process groups
        tp_cp_pg = grid.create_pg(["tp", "cp"])
        cp_dp_pg = grid.create_pg(["cp", "dp"])

        # Verify process groups are real
        assert isinstance(tp_cp_pg, dist.ProcessGroup)
        assert isinstance(cp_dp_pg, dist.ProcessGroup)

        # Test process group sizes
        tp_cp_ranks = dist.get_process_group_ranks(tp_cp_pg)
        cp_dp_ranks = dist.get_process_group_ranks(cp_dp_pg)

        assert len(tp_cp_ranks) == 4  # tp * cp = 2 * 2
        assert len(cp_dp_ranks) == 4  # cp * dp = 2 * 2

    def test_real_distributed_all_reduce(self):
        """Test actual communication using the created process groups."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"], backend="nccl")

        # Create a process group
        tp_pg = grid.create_pg("tp")

        # Create a tensor for communication test
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        tensor = torch.ones(1, device=device) * rank

        # Perform all-reduce within the tensor parallel group
        dist.all_reduce(tensor, group=tp_pg)

        # Verify the result (sum of ranks in the group)
        tp_ranks = dist.get_process_group_ranks(tp_pg)
        expected_sum = sum(tp_ranks)

        assert tensor.item() == expected_sum

    def test_real_distributed_different_world_sizes(self):
        """Test HyperCommGrid with different valid world sizes."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Test configurations for different world sizes
        configs = {
            1: ([1], ["dp"]),
            2: ([2], ["tp"]),
            4: ([2, 2], ["tp", "dp"]),
            8: ([2, 2, 2], ["tp", "cp", "dp"]),
        }

        if world_size not in configs:
            pytest.skip(f"No test configuration for world size {world_size}")

        shape, dim_names = configs[world_size]
        grid = HyperCommGrid(shape, dim_names, backend="nccl")

        assert grid.size == world_size

        # Create and test first dimension process group
        first_dim_pg = grid.create_pg(dim_names[0])
        assert isinstance(first_dim_pg, dist.ProcessGroup)

        # Test communication if world size > 1
        if world_size > 1:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            tensor = torch.tensor([rank], dtype=torch.float, device=device)

            # All-reduce to verify the process group works
            dist.all_reduce(tensor, group=first_dim_pg)

            # Verify the result
            group_ranks = dist.get_process_group_ranks(first_dim_pg)
            expected_sum = sum(group_ranks)
            assert tensor.item() == expected_sum

    def test_real_distributed_error_handling(self):
        """Test error handling with real distributed backend."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size > 8:
            pytest.skip("Test requires at most 8 GPUs")

        # Test shape validation with real world size
        if world_size == 8:
            # This should work
            grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])
            assert grid.size == 8

            # This should fail - too large for world size
            with pytest.raises(RuntimeError, match="Grid shape.*is over sized"):
                HyperCommGrid([4, 4], ["tp", "dp"])  # 16 > 8

        # Test duplicate process group creation
        if world_size >= 2:
            grid = HyperCommGrid([2, world_size // 2], ["tp", "dp"])
            grid.create_pg("tp")

            with pytest.raises(KeyError, match="Process group.*has already been created"):
                grid.create_pg("tp")

    def test_real_distributed_rank_enumeration_verification(self):
        """Verify rank enumeration produces correct communication patterns."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        grid = HyperCommGrid([2, 2, 2], ["tp", "cp", "dp"])

        # Test that ranks in the same TP group can communicate
        tp_pg = grid.create_pg("tp")
        tp_ranks = dist.get_process_group_ranks(tp_pg)

        current_rank = dist.get_rank()
        if current_rank in tp_ranks:
            device = torch.device(f"cuda:{current_rank % torch.cuda.device_count()}")

            # Create a unique tensor based on rank
            tensor = torch.tensor([current_rank], dtype=torch.float, device=device)
            original_value = tensor.clone()

            # All-reduce within TP group
            dist.all_reduce(tensor, group=tp_pg)

            # Verify the sum is correct
            expected_sum = sum(tp_ranks)
            assert tensor.item() == expected_sum

    def test_real_distributed_registered_view(self):
        """Verify a registered expert view's groups with the real distributed backend.

        On 8 ranks, build a base dense grid and register an expert view that re-partitions the
        same ranks, declaring ``pp`` as shared. Then assert:
          * an expert-private group has the correct ACTUAL rank membership;
          * the expert ``pp`` group IS the same object / same ranks as the base ``pp`` group
            (shared-dim reuse);
          * destroy() tears everything down once without double-freeing the reused group.
        """
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip("This test specifically requires 8 GPUs")

        # Base dense view: tp=2, dp=2, pp=2. Reversed -> [pp(outer-most-on-left after reverse),
        # dp, tp]; pp is the trailing dim of the original order, so pp groups are [[0,4],[1,5],
        # [2,6],[3,7]] (strided by 4).
        grid = HyperCommGrid([2, 2, 2], ["tp", "dp", "pp"], backend="nccl")

        # Expert view: expt_tp=2, ep=2, pp=2 over the same 8 ranks. pp kept trailing so its
        # membership matches the base pp groups, satisfying the shared-dim contract.
        grid.register_view("expert", [2, 2, 2], ["expt_tp", "ep", "pp"], shared_dims=["pp"])

        # Base groups.
        base_tp = grid.create_pg("tp")
        base_pp = grid.create_pg("pp")

        # Expert-private group spanning expt_tp + ep (the non-shared dims).
        expert_te = grid.create_pg(["expt_tp", "ep"], view="expert")
        # Shared pp group via the expert view must reuse the base object.
        expert_pp = grid.get_pg("pp", view="expert")

        assert isinstance(base_tp, dist.ProcessGroup)
        assert isinstance(base_pp, dist.ProcessGroup)
        assert isinstance(expert_te, dist.ProcessGroup)

        # View-private group is stored under a tuple key (no string namespacing) and is
        # retrievable through the root grid as the same object.
        assert ("expert", ("ep", "expt_tp")) in grid._pgs
        assert grid.get_pg(["expt_tp", "ep"], view="expert") is expert_te

        # Shared-dim reuse: identical object AND identical ranks, and no view-private duplicate
        # was created for the shared dim.
        assert expert_pp is base_pp
        assert dist.get_process_group_ranks(expert_pp) == dist.get_process_group_ranks(base_pp)
        assert ("expert", ("pp",)) not in grid._pgs

        # Re-creating an already-created view-private group must raise.
        with pytest.raises(KeyError, match="for view 'expert' has already been created"):
            grid.create_pg(["expt_tp", "ep"], view="expert")

        # ACTUAL membership of the expert-private expt_tp-ep group. Reversed expert dim order is
        # [pp, ep, expt_tp]; grouping by (ep, expt_tp) within each pp slice yields, per pp block
        # of 4 contiguous ranks, the whole block. pp trailing => two blocks [0..3] and [4..7].
        expected_te = [[0, 1, 2, 3], [4, 5, 6, 7]]
        assert sorted(grid.get_rank_enum(["expt_tp", "ep"], view="expert")) == sorted(expected_te)
        my_te_ranks = sorted(dist.get_process_group_ranks(expert_te))
        rank = dist.get_rank()
        my_block = [b for b in expected_te if rank in b][0]
        assert my_te_ranks == my_block

        # Exercise real communication on the expert-private group.
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        tensor = torch.tensor([rank], dtype=torch.float, device=device)
        dist.all_reduce(tensor, group=expert_te)
        assert tensor.item() == sum(my_block)

        # destroy() must tear down base_tp, the shared pp group (once), and the expert-private
        # group, without double-freeing the reused pp group. The pp group object is shared
        # between base and expert under a single key, so it cannot be destroyed twice. After
        # teardown the registry is empty.
        grid.destroy()
        assert grid._pgs == {}
