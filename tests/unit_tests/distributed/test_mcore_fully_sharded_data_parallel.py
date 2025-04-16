# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch
from packaging import version
from torch import testing

from megatron.core.wrapped_process_group import WrappedProcessGroup
from megatron.core.device_utils import get_current_device, get_current_device_type
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.custom_fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import GradCommProcessGroups
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


# Test model for testing FSDP
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


class TestFullyShardedDataParallel:
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
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
    def test_fsdp_with_process_groups(self, dp_size):
        """Test that FSDP works correctly with different process group configurations."""
        from torch.distributed.device_mesh import init_device_mesh

        # Skip test if we don't have enough GPUs
        world_size = torch.distributed.get_world_size()
        if world_size != dp_size:
            pytest.skip(f"This test requires {dp_size} GPUs, but only {world_size} are available")

        # Simple model config
        input_dim = 13
        output_dim = 17

        # Setup FSDP config - using optim_grads_params for full sharding test
        fsdp_config = DistributedDataParallelConfig(
            data_parallel_sharding_strategy="optim_grads_params",
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            bucket_size=10000,
            use_custom_fsdp=True,
        )

        # Create two identical models
        model1 = TestModel(input_dim=input_dim, output_dim=output_dim).to(get_current_device())
        model2 = TestModel(input_dim=input_dim, output_dim=output_dim).to(get_current_device())

        # Ensure identical weights
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p2.data.copy_(p1.data)

        transformer_config = TransformerConfig(
            num_attention_heads=1, num_layers=1, context_parallel_size=1  # Explicitly set CP=1
        )
        fsdp_model1 = FullyShardedDataParallel(
            config=transformer_config,
            ddp_config=fsdp_config,
            module=model1,
            fsdp_unit_modules=[torch.nn.Linear],
        )

        # Create a 1D mesh with dimension [dp_size]
        device_mesh = init_device_mesh(get_current_device_type(), (dp_size,), mesh_dim_names=("dp",))
        grad_comm_pgs = GradCommProcessGroups()

        # Get dp process group from device mesh
        dp_group = device_mesh.get_group(mesh_dim="dp")
        grad_comm_pgs.dp = WrappedProcessGroup(dp_group)

        # Wrap second model with explicit process groups
        fsdp_model2 = FullyShardedDataParallel(
            config=transformer_config,
            ddp_config=fsdp_config,
            module=model2,
            grad_comm_pgs=grad_comm_pgs,
            fsdp_unit_modules=[torch.nn.Linear],
        )

        # Create optimizer config
        lr = 3
        optimizer_config = OptimizerConfig(optimizer="adam", lr=lr)
        grad_scaler = None

        optimizer1 = DistributedOptimizer(
            optimizer=None,
            config=optimizer_config,
            grad_scaler=grad_scaler,
            init_state_fn=None,
            model_chunks=[fsdp_model1],
            per_model_buffers={0: [fsdp_model1.param_and_grad_buffer]},
            data_parallel_group=fsdp_model1.dp_cp_group,
            data_parallel_group_gloo=None,
            data_parallel_group_idx=0,
            distributed_optimizer_instance_id=0,
        )

        optimizer2 = DistributedOptimizer(
            optimizer=None,
            config=optimizer_config,
            grad_scaler=grad_scaler,
            init_state_fn=None,
            model_chunks=[fsdp_model2],
            per_model_buffers={0: [fsdp_model2.param_and_grad_buffer]},
            data_parallel_group=fsdp_model2.dp_cp_group,
            data_parallel_group_gloo=None,
            data_parallel_group_idx=0,
            distributed_optimizer_instance_id=1,
        )

        # Create identical inputs
        batch_size = 2
        input_data = torch.randint(0, 10, (batch_size, input_dim), device=get_current_device(), dtype=torch.long)
        input_data = input_data.float()
        input_data.requires_grad = True

        def loss_fn(output, _):
            return output.sum()

        def train_step(model, optimizer, inputs):
            inputs_clone = inputs.clone().detach().requires_grad_(True)
            optimizer.zero_grad()
            outputs = model(inputs_clone)
            loss = loss_fn(outputs, None)
            loss.backward()
            optimizer.step()
            return outputs, loss

        out1, loss1 = train_step(fsdp_model1, optimizer1, input_data)
        out2, loss2 = train_step(fsdp_model2, optimizer2, input_data)

        testing.assert_close(out1, out2, rtol=0, atol=0)
        testing.assert_close(loss1, loss2, rtol=0, atol=0)

        # Check parameters after optimization step
        for (name1, param1), (_, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            if hasattr(param1, 'fully_shard_param_local_shard') and hasattr(
                param2, 'fully_shard_param_local_shard'
            ):
                testing.assert_close(
                    param1.fully_shard_param_local_shard,
                    param2.fully_shard_param_local_shard,
                    rtol=0,
                    atol=0,
                    msg=f"Parameters for {name1} don't match",
                )

        if hasattr(torch.nn.parameter.Parameter, "main_grad"):
            # Custom fsdp adds the `main_grad` attribute function to the
            # torch Parameter, remove this attribute function so that
            # it doesn't conflict with the code in the non-custom fsdp
            # test branch.
            delattr(torch.nn.parameter.Parameter, "main_grad")
