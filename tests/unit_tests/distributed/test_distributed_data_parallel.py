# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
from packaging import version
from torch import testing

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# Test model for testing DDP
class TestModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim * 4)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(input_dim * 4, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class TestDistributedDataParallel:
    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse('2.3.0'),
        reason="Device mesh feature requires PyTorch 2.3 or later",
    )
    @pytest.mark.parametrize("dp_size", [2, 8])  # Test with 2 or 8 GPUs
    def test_ddp_with_dp_process_groups(self, dp_size):
        """Test that DDP works correctly with dp pgs from parallel state and user defined pgs."""

        # Skip test if we don't have enough GPUs
        world_size = torch.distributed.get_world_size()
        if world_size != dp_size:
            pytest.skip(f"This test requires {dp_size} GPUs, but only {world_size} are available")

        # Simple model config
        input_dim = 13
        output_dim = 17

        # Setup DDP config
        ddp_config = DistributedDataParallelConfig(overlap_grad_reduce=True, bucket_size=10000)

        # Create two identical models
        model1 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()
        model2 = TestModel(input_dim=input_dim, output_dim=output_dim).cuda()

        # Ensure identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)

        # Wrap first model with default process groups
        transformer_config = TransformerConfig(
            num_attention_heads=1, num_layers=1, context_parallel_size=1
        )

        ddp_model1 = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=model1
        )

        # Initialize torch.distributed if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

        # Create HyperCommGrid with dimension ep, pp, dp (reversed from device mesh order)
        grid = HyperCommGrid([1, 1, 1, 1, dp_size], ["tp", "cp", "ep", "pp", "dp"])

        # Create process groups config with ONLY dp group
        pg_collection = ProcessGroupCollection()

        pg_collection.dp = grid.create_pg("dp")
        pg_collection.dp_cp = grid.create_pg(["dp", "cp"])
        pg_collection.pp = grid.create_pg("pp")
        pg_collection.tp = grid.create_pg("tp")
        pg_collection.ep = grid.create_pg("ep")

        # Wrap second model with minimal process groups (only dp)
        ddp_model2 = DistributedDataParallel(
            transformer_config, ddp_config=ddp_config, module=model2, pg_collection=pg_collection
        )

        # Create identical inputs with integer values
        batch_size = 2
        input_data = torch.randint(0, 10, (batch_size, input_dim), device='cuda', dtype=torch.long)
        input_data = input_data.float()  # Convert to float for model compatibility

        # Forward pass
        out1 = ddp_model1(input_data)
        out2 = ddp_model2(input_data)

        testing.assert_close(out1, out2, rtol=0, atol=0)

        # Loss and backward
        loss1 = out1.sum()
        loss2 = out2.sum()

        loss1.backward()
        loss2.backward()

        # Check gradients are identical using torch.testing
        for p1, p2 in zip(ddp_model1.parameters(), ddp_model2.parameters()):
            if hasattr(p1, 'main_grad') and hasattr(p2, 'main_grad'):
                testing.assert_close(p1.main_grad, p2.main_grad, rtol=0, atol=0)
