import os

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core import parallel_state
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from tests.unit_tests.test_utilities import Utils


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
    _ = grid.create_pg(["tp"])
    _ = grid.create_pg(["cp"])
    _ = grid.create_pg(["pp"])
    _ = grid.create_pg(["dp"])
    return grid


def create_transformer_block(tp_size, cp_size, pp_size, dp_size, grid_offset: int = 0, use_global_parallel_state: bool = False, hidden_size: int = 4096):
    """Utility to build a ``TransformerBlock`` for tests."""

    torch.manual_seed(12345)
    model_parallel_cuda_manual_seed(123)
    transformer_config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        num_attention_heads=16,
        use_cpu_initialization=True,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        bf16=True,
        context_parallel_size=cp_size,
    )

    if use_global_parallel_state:
        block = TransformerBlock(transformer_config, get_gpt_layer_with_transformer_engine_spec()).cuda().bfloat16()
        return block, None

    grid = HyperCommGrid(
        shape=[tp_size, cp_size, pp_size, dp_size],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=grid_offset,
        backend="nccl",
    )

    tp_group = grid.create_pg("tp")
    cp_group = grid.create_pg("cp")
    pp_group = grid.create_pg("pp")

    model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group, pp=pp_group)

    # Only instantiate the block on the ranks that belong to the grid.
    current_rank = dist.get_rank()
    in_grid = grid.rank_offset <= current_rank < grid.rank_offset + grid.size
    if not in_grid:
        return None, grid

    block = TransformerBlock(transformer_config, get_gpt_layer_with_transformer_engine_spec(), model_comm_pgs=model_comm_pgs).cuda().bfloat16()
    return block, grid


class TestBridgeCommunicator:
    """Test suite for BridgeCommunicator usage."""

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

    def test_bridge_communicator_init(self):

        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()
        assert bridge_communicator.comm_map is not None

    def test_send_forward_recv_backward_send_backward_recv_forward(self):
        """Test combined send_forward_recv_backward and send_backward_recv_forward operations."""

        # Create source and destination grids
        grid1 = create_hypercomm_grid(offset=0, tp=2, cp=2, pp=1, dp=1)
        grid2 = create_hypercomm_grid(offset=4, tp=2, cp=2, pp=1, dp=1)
        bridge_communicator = BridgeCommunicator(grid1, grid2)

        # Verify basic properties
        assert bridge_communicator.src_grid == grid1
        assert bridge_communicator.dest_grid == grid2
        assert bridge_communicator.current_rank == dist.get_rank()

        if bridge_communicator.is_current_rank_in_grid(bridge_communicator.src_grid):
            random_hidden_state = torch.randn(16, 128, 512).cuda()
            received_grad = bridge_communicator.send_forward_recv_backward(
                random_hidden_state, dtype=random_hidden_state.dtype
            )

            # Assert that the returned gradient tensor is valid
            assert received_grad is not None, "send_forward_recv_backward should return a gradient tensor"
            assert isinstance(received_grad, torch.Tensor), f"Expected torch.Tensor, got {type(received_grad)}"
            assert received_grad.shape == random_hidden_state.shape, f"Expected gradient shape {random_hidden_state.shape}, got {received_grad.shape}"
            assert received_grad.device == random_hidden_state.device, f"Expected device {random_hidden_state.device}, got {received_grad.device}"

        else:
            random_grad_state = torch.randn(16, 128, 512).cuda()
            received_activation = bridge_communicator.send_backward_recv_forward(
                random_grad_state, dtype=random_grad_state.dtype
            )

            # Assert that the returned activation tensor is valid
            assert received_activation is not None, "send_backward_recv_forward should return an activation tensor"
            assert isinstance(received_activation, torch.Tensor), f"Expected torch.Tensor, got {type(received_activation)}"
            assert received_activation.shape == random_grad_state.shape, f"Expected activation shape {random_grad_state.shape}, got {received_activation.shape}"
            assert received_activation.device == random_grad_state.device, f"Expected device {random_grad_state.device}, got {received_activation.device}"

    def test_send_forward_recv_forward(self):
        """Test send_forward and recv_forward operations."""
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
            received_activation = bridge_communicator.receive_forward(
                dtype=random_hidden_state.dtype
            )
            assert received_activation.shape == (16, 128, 512), f"Expected activation shape {(16, 128, 512)}, got {received_activation.shape}"

    def test_send_backward_recv_backward(self):
        """Test send_backward and recv_backward operations."""
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

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
            received_gradient = bridge_communicator.receive_backward(dtype=random_grad_state.dtype)
            # Assert that the returned gradient tensor is valid
            assert received_gradient is not None, "recv_backward should return a gradient tensor"
            assert isinstance(received_gradient, torch.Tensor), f"Expected torch.Tensor, got {type(received_gradient)}"
            assert received_gradient.shape == (16, 128, 512), f"Expected gradient shape {(16, 128, 512)}, got {received_gradient.shape}"


    @pytest.mark.parametrize(
        "grid1_tp, grid1_cp, grid1_pp, grid1_dp, grid2_tp, grid2_cp, grid2_pp, grid2_dp, mbs",
        [
            (1, 4, 1, 1, 4, 1, 1, 1, 2),  # Current setup: Grid1 cp=4, Grid2 tp=4,
            (1, 4, 1, 1, 1, 1, 1, 4, 8),  # Fan-out test
            (1, 1, 1, 4, 4, 1, 1, 1, 8),  # Fan-in test
            (2, 1, 1, 2, 2, 1, 1, 2, 8),  # Multiple dp groups test
            (1, 1, 1, 4, 2, 1, 1, 2, 8),  # Multiple dp groups test different dp sizes
        ],
    )
    def test_bridge_communicator_with_transformer_blocks(
        self, grid1_tp, grid1_cp, grid1_pp, grid1_dp, grid2_tp, grid2_cp, grid2_pp, grid2_dp, mbs
    ):
        """Test bridge communicator with two transformer blocks having different process group configurations."""

        hidden_size = 4096
        sequence_length = 2048
        micro_batch_size = mbs

        block1, grid1 = create_transformer_block(
            grid1_tp, grid1_cp, grid1_pp, grid1_dp, grid_offset=0, hidden_size=hidden_size
        )
        block2, grid2 = create_transformer_block(
            grid2_tp, grid2_cp, grid2_pp, grid2_dp, grid_offset=4, hidden_size=hidden_size
        )

        # Create bridge communicator linking the two grids.
        bridge_communicator = BridgeCommunicator(
            grid1, grid2, dim_mapping={'s': 0, 'h': 2, 'b': 1}, requires_scatter_gather=False
        )

        # Grid 1 Forward send
        hidden_states, output1 = None, None
        if if bridge_communicator.is_current_rank_in_grid(grid1):
            hidden_states = torch.randn((sequence_length, micro_batch_size, hidden_size), device="cuda").bfloat16()
            # Send forward activation to grid2
            output1 = block1(hidden_states=hidden_states, attention_mask=None)
            logging.info(f" Grid 1 rank {dist.get_rank()}: Sending activation shape {output1.shape}")
            bridge_communicator.send_forward(output1)

        # Grid 2 Forward receive
        if bridge_communicator.is_current_rank_in_grid(grid2):
            received_activation = bridge_communicator.receive_forward(dtype=torch.bfloat16)
            assert received_activation is not None, "Should receive activation from grid1"
            logging.info(f" Grid 2 rank {dist.get_rank()}: Received activation shape {received_activation.shape}")

            output2 = block2(hidden_states=received_activation, attention_mask=None)
            factor = max(grid1_dp, grid2_dp) // min(grid1_dp, grid2_dp)
            expected_output_shape = (
                sequence_length,
                micro_batch_size * factor if grid1_dp > grid2_dp else micro_batch_size // factor,
                hidden_size,
            )
           
            assert output2.shape == expected_output_shape, f"Output2 shape mismatch: {output2.shape}"

            logging.info(f" Grid 2 rank {dist.get_rank()}: forward pass with output shape {output2.shape}")

        complete_block_1, _ = create_transformer_block(2,2,2, use_global_parallel_state=True)
        complete_block_2, _ = create_transformer_block(2,2,2, use_global_parallel_state=True)
        full_block_output = complete_block_2(complete_block_1(hidden_states=hidden_states))

        if bridge_communicator.is_current_rank_in_grid(grid2):s
            torch.testing.assert_close(full_block_output, output2, rtol=1e-3, atol=1e-3)
