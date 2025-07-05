import torch.distributed as dist
import os
import pytest
import torch
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator


def create_hypercomm_grid(offset=0, dp=1, tp=1, cp=1):
    """
    Create a HyperCommGrid with tensor parallelism=2, context parallelism=2, and data parallelism=2.
    
    Returns:
        HyperCommGrid: A grid configured with tp=2, cp=2, dp=2 (total size = 8).
    """
    # Set up environment for world size 8 if not already set
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "8"
    
    # Create HyperCommGrid with shape [2, 2, 2] and dimension names ["tp", "cp", "dp"]
    grid = HyperCommGrid(
        shape=[tp, cp, dp],
        dim_names=["tp", "cp", "dp"],
        rank_offset=offset,
        backend="nccl"
    )
    print(grid)
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

    def test_tensor_reconstruction(self):
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        grid1 = create_hypercomm_grid(offset=0, dp=2, tp=2, cp=1)
        grid2 = create_hypercomm_grid(offset=4, dp=2, tp=2, cp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        tp_cp_group_ranks = dist.get_process_group_ranks(bridge_communicator.dp_pg)
        device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        gathered_tensors = [torch.randn(3, 128, 256, device=device) for _ in range(len(tp_cp_group_ranks))]
        aggregated_tensor = bridge_communicator._reconstruct_tensor_from_gathered(gathered_tensors, grid1)
        assert aggregated_tensor.shape == (3, 128, 512), f"Expected aggregated tensor shape (3, 128, 512), got {aggregated_tensor.shape}"