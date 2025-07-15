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
            random_hidden_state = torch.randn(
                16, 128, 512
            ).cuda()  # (batch_size, seq_len, hidden_size)
            received_grad = bridge_communicator.send_forward_recv_backward(
                random_hidden_state, grad_shape=(16, 128, 512), dtype=random_hidden_state.dtype
            )

            # Assert that the returned gradient tensor is valid
            assert (
                received_grad is not None
            ), "send_forward_recv_backward should return a gradient tensor"
            assert isinstance(
                received_grad, torch.Tensor
            ), f"Expected torch.Tensor, got {type(received_grad)}"
            assert (
                received_grad.shape == random_hidden_state.shape
            ), f"Expected gradient shape {random_hidden_state.shape}, got {received_grad.shape}"
            assert (
                received_grad.device == random_hidden_state.device
            ), f"Expected device {random_hidden_state.device}, got {received_grad.device}"

        else:
            random_grad_state = torch.randn(
                16, 128, 512
            ).cuda()  # (batch_size, seq_len, hidden_size)
            received_activation = bridge_communicator.send_backward_recv_forward(
                random_grad_state, forward_shape=(16, 128, 512), dtype=random_grad_state.dtype
            )

            # Assert that the returned activation tensor is valid
            assert (
                received_activation is not None
            ), "send_backward_recv_forward should return an activation tensor"
            assert isinstance(
                received_activation, torch.Tensor
            ), f"Expected torch.Tensor, got {type(received_activation)}"
            assert (
                received_activation.shape == random_grad_state.shape
            ), f"Expected activation shape {random_grad_state.shape}, got {received_activation.shape}"
            assert (
                received_activation.device == random_grad_state.device
            ), f"Expected device {random_grad_state.device}, got {received_activation.device}"

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
            received_activation = bridge_communicator.receive_forward(
                tensor_shape=(16, 128, 512), dtype=random_hidden_state.dtype
            )
            # Assert that the returned activation tensor is valid
            assert (
                received_activation is not None
            ), "recv_forward should return an activation tensor"
            assert isinstance(
                received_activation, torch.Tensor
            ), f"Expected torch.Tensor, got {type(received_activation)}"
            assert received_activation.shape == (
                16,
                128,
                512,
            ), f"Expected activation shape {(16, 128, 512)}, got {received_activation.shape}"

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
            received_gradient = bridge_communicator.receive_backward(
                tensor_shape=(16, 128, 512), dtype=random_grad_state.dtype
            )
            # Assert that the returned gradient tensor is valid
            assert received_gradient is not None, "recv_backward should return a gradient tensor"
            assert isinstance(
                received_gradient, torch.Tensor
            ), f"Expected torch.Tensor, got {type(received_gradient)}"
            assert received_gradient.shape == (
                16,
                128,
                512,
            ), f"Expected gradient shape {(16, 128, 512)}, got {received_gradient.shape}"

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    def test_bridge_communicator_with_transformer_blocks(self):
        """
        Test bridge communicator with two transformer blocks having different process group configurations.
        First block: tp=4, cp=1, dp=1, pp=1
        Second block: tp=1, cp=4, dp=1, pp=1
        """
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Initialize model parallel
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel()

        # Set random seeds for reproducibility
        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)

        # Create transformer configuration
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=2048,
            num_attention_heads=16,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            context_parallel_size=4,  # Will be overridden per block
        )

        # Create first grid: tp=4, cp=1, dp=1, pp=1 (offset 0-3)
        grid1 = HyperCommGrid(
            shape=[1, 4, 1, 1], dim_names=["tp", "cp", "pp", "dp"], rank_offset=0, backend="nccl"
        )

        tp_group1 = grid1.create_pg("tp")
        cp_group1 = grid1.create_pg("cp")
        pp_group1 = grid1.create_pg("pp")
        model_comm_pgs1 = ModelCommProcessGroups(tp=tp_group1, cp=cp_group1, pp=pp_group1)

        # Create second grid: tp=1, cp=4, dp=1, pp=1 (offset 4-7)
        transformer_config_2 = TransformerConfig(
            num_layers=2,
            hidden_size=2048,
            num_attention_heads=16,
            use_cpu_initialization=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            context_parallel_size=1,  # Second block uses cp=4
        )

        grid2 = HyperCommGrid(
            shape=[4, 1, 1, 1], dim_names=["tp", "cp", "pp", "dp"], rank_offset=4, backend="nccl"
        )

        tp_group2 = grid2.create_pg("tp")
        cp_group2 = grid2.create_pg("cp")
        pp_group2 = grid2.create_pg("pp")
        model_comm_pgs2 = ModelCommProcessGroups(tp=tp_group2, cp=cp_group2, pp=pp_group2)

        # Create bridge communicator
        bridge_communicator = BridgeCommunicator(grid1, grid2, dim_mapping={'s': 0, 'h': 2, 'b': 1}, requires_scatter_gather=False)

        # Create transformer blocks
        block1 = None
        block2 = None

        if bridge_communicator.is_current_rank_in_grid(grid1):
            block1 = (
                TransformerBlock(
                    transformer_config,
                    get_gpt_layer_with_transformer_engine_spec(),
                    model_comm_pgs=model_comm_pgs1,
                )
                .cuda()
                .bfloat16()
            )

        if bridge_communicator.is_current_rank_in_grid(grid2):
            block2 = (
                TransformerBlock(
                    transformer_config_2,
                    get_gpt_layer_with_transformer_engine_spec(),
                    model_comm_pgs=model_comm_pgs2,
                )
                .cuda()
                .bfloat16()
            )

        # Test forward pass with bridge communicator
        sequence_length = 2048
        micro_batch_size = 2

        # Create input tensor (only on grid1)
        hidden_states = None
        if block1 is not None:
            hidden_states = torch.randn(
                (sequence_length, micro_batch_size, transformer_config.hidden_size), device="cuda"
            ).bfloat16()

        # Forward pass through first block
        output1 = None
            
        # Bridge communication: send forward activation from grid1 to grid2
        if bridge_communicator.is_current_rank_in_grid(grid1):
            # Send forward activation to grid2
            output1 = block1(hidden_states=hidden_states, attention_mask=None)
            print(f"Grid1 rank {dist.get_rank()}: Sending initial activation with shape {output1.shape}")
            bridge_communicator.send_forward(output1)

        if bridge_communicator.is_current_rank_in_grid(grid2):
            # Receive forward activation from grid1
            received_activation = bridge_communicator.receive_forward(
                tensor_shape=(sequence_length, micro_batch_size, transformer_config.hidden_size),
                dtype=torch.bfloat16,
            )

            # Verify received activation
            assert received_activation is not None, "Should receive activation from grid1"
            assert received_activation.shape == (
                sequence_length,
                micro_batch_size,
                transformer_config.hidden_size,
            ), f"Activation shape mismatch: {received_activation.shape}"
            assert (
                received_activation.device.type == "cuda"
            ), f"Activation should be on CUDA device: {received_activation.device}"

            print(
                f"Grid2 rank {dist.get_rank()}: Successfully received activation with shape {received_activation.shape}"
            )

            # Forward pass through second block
            output2 = block2(hidden_states=received_activation, attention_mask=None)

            # Verify output shape
            assert output2.shape == (
                sequence_length,
                micro_batch_size,
                transformer_config.hidden_size,
            ), f"Output2 shape mismatch: {output2.shape}"

            print(
                f"Grid2 rank {dist.get_rank()}: Successfully completed forward pass with output shape {output2.shape}"
            )

        # Clean up
        Utils.destroy_model_parallel()

    def test_bridge_communicator_with_linear_layers(self):
        """
        Test bridge communicator with two linear layers having different process group configurations.
        First layer: tp=4, cp=1, dp=1, pp=1 (ColumnParallelLinear)
        Second layer: tp=1, cp=4, dp=1, pp=1 (RowParallelLinear)
        """
        if not dist.is_initialized():
            pytest.skip("Distributed not initialized")

        world_size = dist.get_world_size()
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Initialize model parallel
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel()

        # Set random seeds for reproducibility
        torch.manual_seed(12345)
        model_parallel_cuda_manual_seed(123)

        # Import the linear layers
        from megatron.core.model_parallel_config import ModelParallelConfig
        from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

        # Create first grid: tp=4, cp=1, dp=1, pp=1 (offset 0-3)
        grid1 = HyperCommGrid(
            shape=[4, 1, 1, 1], dim_names=["tp", "cp", "pp", "dp"], rank_offset=0, backend="nccl"
        )

        tp_group1 = grid1.create_pg("tp")
        cp_group1 = grid1.create_pg("cp")
        pp_group1 = grid1.create_pg("pp")
        model_comm_pgs1 = ModelCommProcessGroups(tp=tp_group1, cp=cp_group1, pp=pp_group1)

        # Create second grid: tp=1, cp=4, dp=1, pp=1 (offset 4-7)
        grid2 = HyperCommGrid(
            shape=[1, 4, 1, 1], dim_names=["tp", "cp", "pp", "dp"], rank_offset=4, backend="nccl"
        )

        tp_group2 = grid2.create_pg("tp")
        cp_group2 = grid2.create_pg("cp")
        pp_group2 = grid2.create_pg("pp")
        model_comm_pgs2 = ModelCommProcessGroups(tp=tp_group2, cp=cp_group2, pp=pp_group2)

        # Create model parallel configs
        config1 = ModelParallelConfig(
            tensor_model_parallel_size=4,
            context_parallel_size=1,
            use_cpu_initialization=True,
            perform_initialization=True,
            bf16=True,
        )

        config2 = ModelParallelConfig(
            tensor_model_parallel_size=1,
            context_parallel_size=4,
            use_cpu_initialization=True,
            perform_initialization=True,
            bf16=True,
        )

        # Layer dimensions
        input_size = 2048
        hidden_size = 4096

        # Create init method
        def init_method(tensor):
            torch.nn.init.normal_(tensor, mean=0.0, std=0.02)

        # Create 2 normal linear layers with no parallelization
        normal_layer1 = torch.nn.Linear(input_size, hidden_size, bias=True).cuda().bfloat16()
        normal_layer2 = torch.nn.Linear(hidden_size, input_size, bias=True).cuda().bfloat16()

        # Initialize the normal layers
        init_method(normal_layer1.weight)
        init_method(normal_layer2.weight)
        torch.nn.init.zeros_(normal_layer1.bias)
        torch.nn.init.zeros_(normal_layer2.bias)

        # Create bridge communicator
        bridge_communicator = BridgeCommunicator(grid1, grid2, dim_mapping={'s': 0, 'h': 2, 'b': 1})

        # Create linear layers
        layer1 = None
        layer2 = None

        if bridge_communicator.is_current_rank_in_grid(grid1):
            # First layer: ColumnParallelLinear with tp=4
            layer1 = (
                ColumnParallelLinear(
                    input_size=input_size,
                    output_size=hidden_size,
                    config=config1,
                    init_method=init_method,
                    bias=True,
                    gather_output=False,
                    skip_bias_add=False,
                    tp_group=tp_group1,
                )
                .cuda()
                .bfloat16()
            )

            # Initialize ColumnParallelLinear weights using normal_layer1 weights
            # For column parallel, we need to split the normal layer weights across tensor parallel ranks
            tp_rank = dist.get_rank(tp_group1)
            tp_size = dist.get_world_size(tp_group1)

            # Calculate the portion of weights for this TP rank
            output_per_rank = hidden_size // tp_size
            start_idx = tp_rank * output_per_rank
            end_idx = (tp_rank + 1) * output_per_rank

            # Copy the corresponding slice of weights and bias
            with torch.no_grad():
                layer1.weight.copy_(normal_layer1.weight[start_idx:end_idx, :].bfloat16())
                if layer1.bias is not None:
                    layer1.bias.copy_(normal_layer1.bias[start_idx:end_idx].bfloat16())

        if bridge_communicator.is_current_rank_in_grid(grid2):
            # Second layer: RowParallelLinear with tp=1, cp=4
            layer2 = (
                RowParallelLinear(
                    input_size=hidden_size,
                    output_size=input_size,
                    config=config2,
                    init_method=init_method,
                    bias=True,
                    input_is_parallel=True,
                    skip_bias_add=False,
                    tp_group=tp_group2,
                )
                .cuda()
                .bfloat16()
            )

            # Initialize RowParallelLinear weights using normal_layer2 weights
            # For row parallel, we need to split the normal layer weights across tensor parallel ranks
            tp_rank = dist.get_rank(tp_group2)
            tp_size = dist.get_world_size(tp_group2)

            # Calculate the portion of weights for this TP rank
            input_per_rank = hidden_size // tp_size
            start_idx = tp_rank * input_per_rank
            end_idx = (tp_rank + 1) * input_per_rank

            # Copy the corresponding slice of weights
            with torch.no_grad():
                layer2.weight.copy_(normal_layer2.weight[:, start_idx:end_idx].bfloat16())
                # For row parallel, only rank 0 has bias
                if layer2.bias is not None and tp_rank == 0:
                    layer2.bias.copy_(normal_layer2.bias.bfloat16())

        # Test forward pass with bridge communicator
        batch_size = 2
        sequence_length = 512

        # Create input tensor (only on grid1) - ENABLE GRADIENTS
        input_tensor = torch.randn(
                (sequence_length, batch_size, input_size),
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,  # Enable gradients for backward pass testing
        )
        # FORWARD PASS
        if bridge_communicator.is_current_rank_in_grid(grid1):
            # Forward pass through first layer
            output1, _ = layer1(input_tensor)
            print(f"Grid1 rank {dist.get_rank()}: Sending activation with shape {output1.shape}")
            bridge_communicator.send_forward(output1)

        if bridge_communicator.is_current_rank_in_grid(grid2):
            # Receive forward activation from grid1
            received_activation = bridge_communicator.receive_forward(
                tensor_shape=(sequence_length // 4, batch_size, hidden_size), dtype=torch.bfloat16
            )

            # Verify received activation
            assert received_activation is not None, "Should receive activation from grid1"
            assert received_activation.shape == (
                sequence_length // 4,
                batch_size,
                hidden_size,
            ), f"Activation shape mismatch: {received_activation.shape}"
            assert (
                received_activation.device.type == "cuda"
            ), f"Activation should be on CUDA device: {received_activation.device}"

            print(
                f"Grid2 rank {dist.get_rank()}: Successfully received activation with shape {received_activation.shape}"
            )

            # Forward pass through second layer
            output2, _ = layer2(received_activation)

            # Verify output shape
            assert output2.shape == (
                sequence_length // 4,
                batch_size,
                input_size,
            ), f"Output2 shape mismatch: {output2.shape}"

            print(
                f"Grid2 rank {dist.get_rank()}: Successfully completed forward pass with output shape {output2.shape}"
            )
            print("Generating output using normal linear layers with no parallelization")
            output = normal_layer2(normal_layer1(input_tensor))
            print(
                f"Generated output shape using linear layers with no parallelization: {output.shape}"
            )

            cp_group = grid2.get_pg(["cp"])
            print(f"Running all-gather across CP group on grid2: {cp_group}")
            gathered_outputs = [torch.zeros_like(output2) for _ in range(cp_group.size())]
            dist.all_gather(gathered_outputs, output2, group=cp_group)
            full_output = torch.cat(gathered_outputs, dim=0)
            print(f"Grid2 rank {dist.get_rank()}: All-gathered output shape: {full_output.shape}")

            # Check if full_output and output are the same tensors
            print(f"Comparing full_output {full_output.shape} with expected output {output.shape}")
            assert torch.allclose(
                full_output, output, rtol=1e-3, atol=1e-3
            ), "full_output and output tensors should be approximately equal"
            print("SUCCESS: full_output and output tensors match!")

        Utils.destroy_model_parallel()
