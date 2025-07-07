import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    """
    Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2.
    
    Returns:
        HyperCommGrid: A grid configured with tp=2, cp=2, dp=2 (total size = 8).
    """
    # Set up environment for world size 8 if not already set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"

    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl"
    )
    print(grid)
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    return grid


class TestBridgeCommunicator:
    """Test suite for BridgeCommunicator usage."""

    @classmethod
    def setup_class(cls):
        """Set up distributed environment for the entire test class."""
        if not dist.is_initialized():
            try:
                # Initialize PyTorch distributed with NCCL backend
                dist.init_process_group(backend="nccl")
                cls.distributed_initialized = True
            except Exception as e:
                pytest.skip(f"Cannot initialize distributed: {e}")
        else:
            cls.distributed_initialized = True

        grid = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=2)
        if dist.get_rank() == 0: 
            print(f"tp rank by enum {grid._get_rank_enum(['tp'])}")
            print(f"cp rank by enum {grid._get_rank_enum(['cp'])}")
            print(f"dp rank by enum {grid._get_rank_enum(['dp'])}")
            print(f"pp rank by enum {grid._get_rank_enum(['pp'])}")
            print(f"tp-cp rank by enum {grid._get_rank_enum(['tp', 'cp'])}")
            print(f"tp-cp-pp rank by enum {grid._get_rank_enum(['tp', 'cp', 'pp'])}")

    def test_bridge_communicator_init(self):
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")
        
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()
        assert bridge_communicator.comm_map is not None


    def test_tensor_reconstruction(self):
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        grid1 = create_hypercomm_grid(offset=0, dp=2, tp=2, cp=1)
        grid2 = create_hypercomm_grid(offset=4, dp=2, tp=2, cp=1)
        if dist.get_rank() < 4:
            source_grid = grid1
        else:
            source_grid = grid2
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        tp_cp_group_ranks = dist.get_process_group_ranks(bridge_communicator.dp_pg)
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        gathered_tensors = [torch.randn(3, 128, 256, device=device) for _ in range(len(tp_cp_group_ranks))]
        aggregated_tensor = bridge_communicator._reconstruct_tensor_from_gathered(gathered_tensors, source_grid)
        assert aggregated_tensor.shape == (3, 128, 512), f"Expected aggregated tensor shape (3, 128, 512), got {aggregated_tensor.shape}"