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
        backend="nccl",
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
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
       

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

    # def test_tensor_reconstruction(self):
    #     if not dist.is_initialized():
    #         pytest.skip("Distributed not initialized")

    #     world_size = dist.get_world_size()
    #     if world_size != 8:
    #         pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

    #     grid1 = create_hypercomm_grid(offset=0, dp=2, tp=2, cp=1)
    #     grid2 = create_hypercomm_grid(offset=4, dp=2, tp=2, cp=1)
    #     if dist.get_rank() < 4:
    #         source_grid = grid1
    #     else:
    #         source_grid = grid2
    #     bridge_communicator = BridgeCommunicator(grid1, grid2)
    #     tp_cp_group_ranks = dist.get_process_group_ranks(bridge_communicator.dp_pg)
    #     device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
    #     gathered_tensors = [
    #         torch.randn(3, 128, 256, device=device) for _ in range(len(tp_cp_group_ranks))
    #     ]
    #     aggregated_tensor = bridge_communicator._reconstruct_tensor_from_gathered(
    #         gathered_tensors, source_grid
    #     )
    #     assert aggregated_tensor.shape == (
    #         3,
    #         128,
    #         512,
    #     ), f"Expected aggregated tensor shape (3, 128, 512), got {aggregated_tensor.shape}"

    def test_send_forward_recv_backward_send_backward_recv_forward(self):
        """Test combined send_forward_recv_backward and send_backward_recv_forward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        
        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = torch.randn(16, 128, 512).cuda()  # (batch_size, seq_len, hidden_size)
            received_grad = bridge_communicator.send_forward_recv_backward(
                random_hidden_state, grad_shape=(16, 128, 512), dtype=random_hidden_state.dtype
            )
            
            # Assert that the returned gradient tensor is valid
            assert received_grad is not None, "send_forward_recv_backward should return a gradient tensor"
            assert isinstance(received_grad, torch.Tensor), f"Expected torch.Tensor, got {type(received_grad)}"
            assert received_grad.shape == random_hidden_state.shape, f"Expected gradient shape {random_hidden_state.shape}, got {received_grad.shape}"
            assert received_grad.device == random_hidden_state.device, f"Expected device {random_hidden_state.device}, got {received_grad.device}"
            
        else:
            random_grad_state = torch.randn(16, 128, 512).cuda()  # (batch_size, seq_len, hidden_size)
            received_activation = bridge_communicator.send_backward_recv_forward(
                random_grad_state, forward_shape=(16, 128, 512), dtype=random_grad_state.dtype
            )
            
            # Assert that the returned activation tensor is valid
            assert received_activation is not None, "send_backward_recv_forward should return an activation tensor"
            assert isinstance(received_activation, torch.Tensor), f"Expected torch.Tensor, got {type(received_activation)}"
            assert received_activation.shape == random_grad_state.shape, f"Expected activation shape {random_grad_state.shape}, got {received_activation.shape}"
            assert received_activation.device == random_grad_state.device, f"Expected device {random_grad_state.device}, got {received_activation.device}"

    def test_send_forward_recv_forward(self):
        """Test send_forward and recv_forward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        
        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        random_hidden_state = torch.randn(16, 128, 512)
        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = random_hidden_state.cuda()
            bridge_communicator.send_forward(random_hidden_state)
                 
        else:
            received_activation = bridge_communicator.receive_forward(tensor_shape=(16, 128, 512), dtype=random_hidden_state.dtype)      
            # Assert that the returned activation tensor is valid
            assert received_activation is not None, "recv_forward should return an activation tensor"
            assert isinstance(received_activation, torch.Tensor), f"Expected torch.Tensor, got {type(received_activation)}"
            assert received_activation.shape == (16, 128, 512), f"Expected activation shape {(16, 128, 512)}, got {received_activation.shape}"


    def test_send_backward_recv_backward(self):
        """Test send_backward and recv_backward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        
        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        random_grad_state = torch.randn(16, 128, 512)
        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.dest_grid):
            # In backward pass, gradients flow from destination grid back to source grid
            random_grad_state = random_grad_state.cuda()
            bridge_communicator.send_backward(random_grad_state)
                 
        else:
            received_gradient = bridge_communicator.receive_backward(tensor_shape=(16, 128, 512), dtype=random_grad_state.dtype)      
            # Assert that the returned gradient tensor is valid
            assert received_gradient is not None, "recv_backward should return a gradient tensor"
            assert isinstance(received_gradient, torch.Tensor), f"Expected torch.Tensor, got {type(received_gradient)}"
            assert received_gradient.shape == (16, 128, 512), f"Expected gradient shape {(16, 128, 512)}, got {received_gradient.shape}"
