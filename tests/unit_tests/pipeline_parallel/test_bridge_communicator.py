import logging
import os
import sys

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_group, get_tensor_model_parallel_rank
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2."""
    # Set up environment for world size 8 if not already set
    if not dist.is_initialized():
        raise RuntimeError("Distributed process group is not initialized")

    #  tests below assume a world size of 8
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"

    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl",
    )
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    return grid


class TestBridgeCommunicator:

    @classmethod
    def setup_class(cls):
        """Set up distributed environment for the entire test class."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(
                f"These tests require 8 GPUs, but only {world_size} are available.",
                allow_module_level=True,
            )

    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def test_bridge_communicator_init(self):

        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=1, pp=1, dp=2)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=1, pp=1, dp=2)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        assert bridge_communicator.src_grid is grid1
        assert bridge_communicator.dest_grid is grid2
        assert bridge_communicator.current_rank == dist.get_rank()
        assert bridge_communicator.comm_map is not None

    @pytest.mark.parametrize(
        "tp, cp, pp, dp, expected_src_ranks, expected_dest_ranks",
        [
            # Test Case 1: tp=2, cp=1, pp=2, dp=2
            (2, 1, 2, 2, [[2, 3], [6, 7]], [[0, 1], [4, 5]]),
            # Test Case 2: tp=4, cp=1, pp=2, dp=1
            (4, 1, 2, 1, [[4, 5, 6, 7]], [[0, 1, 2, 3]]),
            # Test Case 3: tp=1, cp=1, pp=2, dp=4
            (1, 1, 2, 4, [[1], [3], [5], [7]], [[0], [2], [4], [6]]),
            # Test Case 4: tp=2, cp=1, pp=4, dp=1
            (2, 1, 4, 1, [[6, 7]], [[0, 1]]),
        ],
    )
    def test_get_boundary_pp_stage_ranks(
        self, tp, cp, pp, dp, expected_src_ranks, expected_dest_ranks
    ):
        """Test get_boundary_pp_stage_ranks function with different parallelism configurations."""

        # Create grid with specified parallelism dimensions
        grid = create_hypercomm_grid(offset=0, tp=tp, cp=cp, pp=pp, dp=dp)
        bridge_communicator = BridgeCommunicator(grid, grid)  # Using same grid for simplicity

        # For source grid (is_src=True), should return ranks from last pp stage
        src_boundary_ranks = bridge_communicator.get_boundary_pp_stage_ranks(grid, is_src=True)
        assert (
            src_boundary_ranks == expected_src_ranks
        ), f"Source: Expected {expected_src_ranks}, got {src_boundary_ranks}"

        # For destination grid (is_src=False), should return ranks from first pp stage
        dest_boundary_ranks = bridge_communicator.get_boundary_pp_stage_ranks(grid, is_src=False)
        assert (
            dest_boundary_ranks == expected_dest_ranks
        ), f"Dest: Expected {expected_dest_ranks}, got {dest_boundary_ranks}"

    @pytest.mark.parametrize(
        "tp, cp, pp, dp, expected_src_leaders, expected_dest_leaders",
        [
            # Test Case 1: tp=2, cp=1, pp=2, dp=2
            (2, 1, 2, 2, [3, 7], [0, 4]),
            # Test Case 2: tp=4, cp=1, pp=2, dp=1
            (4, 1, 2, 1, [7], [0]),
            # Test Case 3: tp=1, cp=1, pp=2, dp=4
            (1, 1, 2, 4, [1, 3, 5, 7], [0, 2, 4, 6]),
            # Test Case 4: tp=2, cp=1, pp=4, dp=1
            (2, 1, 4, 1, [7], [0]),
        ],
    )
    def test_get_leader_rank(self, tp, cp, pp, dp, expected_src_leaders, expected_dest_leaders):
        """Test get_leader_rank function with different parallelism configurations."""

        # Create grid with specified parallelism dimensions
        grid = create_hypercomm_grid(offset=0, tp=tp, cp=cp, pp=pp, dp=dp)
        bridge_communicator = BridgeCommunicator(grid, grid)  # Using same grid for simplicity

        # For source grid (is_src=True), should return leader ranks from last pp stage of each dp replica
        src_leaders, _ = bridge_communicator.get_leader_rank(grid, is_src=True)
        assert (
            src_leaders == expected_src_leaders
        ), f"Source leaders: Expected {expected_src_leaders}, got {src_leaders}"

        # For destination grid (is_src=False), should return leader ranks from first pp stage of each dp replica
        dest_leaders, _ = bridge_communicator.get_leader_rank(grid, is_src=False)
        assert (
            dest_leaders == expected_dest_leaders
        ), f"Dest leaders: Expected {expected_dest_leaders}, got {dest_leaders}"
