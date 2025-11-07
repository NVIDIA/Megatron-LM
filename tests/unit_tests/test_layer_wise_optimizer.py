# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_size
from tests.unit_tests.test_utilities import Utils

# Skip all tests in this file for LTS versions
pytestmark = pytest.mark.skipif(
    Version(os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")) <= Version("25.05"),
    reason="Skip layer-wise optimizer for LTS test",
)


class SimpleModel(nn.Module):
    """Simple model for testing LayerWiseDistributedOptimizer.

    Model with 5 layers to ensure more than 8 parameters (10 total: 5 weights + 5 biases).
    """

    def __init__(self, input_size=80, hidden_size=48, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 24)
        self.fc4 = nn.Linear(24, 16)
        self.fc5 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class TinyModel(nn.Module):
    """Tiny model with only 1 layer (2 parameters: weight and bias)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc1(x)


@pytest.mark.skipif(
    int(os.getenv('WORLD_SIZE', '1')) == 1, reason="Multi-rank test requires WORLD_SIZE > 1"
)
class TestLayerWiseOptimizer:
    """Test class for LayerWiseDistributedOptimizer with common setup code."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and teardown for each test."""
        world = int(os.getenv('WORLD_SIZE', '1'))
        rank = int(os.getenv('RANK', '0'))
        Utils.initialize_model_parallel()
        yield
        Utils.destroy_model_parallel()

    def create_model_and_optimizer(
        self,
        model_class=SimpleModel,
        clip_grad=1.0,
        model_kwargs=None,
        use_layer_wise=True,
        copy_from=None,
    ):
        """Create model, DDP wrapper, and optimizer.

        Args:
            model_class: Model class to instantiate
            clip_grad: Optional gradient clipping value
            model_kwargs: Optional kwargs for model initialization
            use_layer_wise: If True, wrap optimizer in LayerWiseDistributedOptimizer;
                          if False, use get_megatron_optimizer instead (for reference)

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        if model_kwargs is None:
            model_kwargs = {}

        model = model_class(**model_kwargs).bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )
        if copy_from:
            model.module.load_state_dict(copy_from.module.state_dict())
        else:
            model.broadcast_params()

        optimizer_config = OptimizerConfig(
            optimizer='adam',
            lr=0.01,
            weight_decay=0.01,
            bf16=not use_layer_wise,
            use_distributed_optimizer=False,
            clip_grad=clip_grad,
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = get_megatron_optimizer(optimizer_config, [model])
        if use_layer_wise:
            optimizer_config.bf16 = True
            optimizer = LayerWiseDistributedOptimizer(
                optimizer.chained_optimizers, optimizer_config, pg_collection
            )
        return model, optimizer, pg_collection

    def create_reference_model(self, model):
        """Create a reference model by cloning the current model."""
        reference_model = type(model.module)().bfloat16().cuda()
        reference_model.load_state_dict(model.module.state_dict())
        return reference_model

    def test_basic(self):
        """Test basic LayerWiseDistributedOptimizer initialization and step with bf16."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Verify basic properties
        assert optimizer is not None, "Optimizer should not be None"
        assert hasattr(optimizer, 'chained_optimizers'), "Should be a ChainedOptimizer"

        reference_model = self.create_reference_model(model)

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

        # Verify parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # Verify all ranks have the same updated parameters (test allgather)
        dp_size = get_pg_size(pg_collection.dp_cp)

        if dp_size > 1:
            for name, param in model.named_parameters():
                # Gather parameters from all ranks
                param_list = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(param_list, param.data, group=pg_collection.dp_cp)

                # Verify all ranks have the same parameter values
                for i in range(1, dp_size):
                    try:
                        torch.testing.assert_close(param_list[0], param_list[i])
                    except AssertionError as e:
                        # Append additional context without overwriting the default message
                        raise AssertionError(
                            f"Parameter {name} differs between rank 0 and rank {i}. {str(e)}"
                        ) from None

    def test_get_grad_norm(self):
        """Test LayerWiseDistributedOptimizer gradient norm computation."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            use_layer_wise=False
        )

        # Set same gradients on both models
        # note that model is different at this point but we're only testing grad norm here
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.float().detach()
            ref_param.main_grad = grad_value.float().detach()

        # Test get_grad_norm on both optimizers
        optimizer.prepare_grads()
        grad_norm = optimizer.get_grad_norm()

        reference_optimizer.prepare_grads()
        reference_grad_norm = reference_optimizer.get_grad_norm()

        assert grad_norm is not None, "Grad norm should not be None"
        assert grad_norm >= 0, "Grad norm should be non-negative"

        # Compare with reference optimizer grad norm
        torch.testing.assert_close(grad_norm, reference_grad_norm, rtol=1e-5, atol=1e-5)

    def test_state_dict(self):
        """Test LayerWiseDistributedOptimizer state dict save and load."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Test state_dict
        state_dict = optimizer.state_dict()

        # Test load_state_dict
        # TODO(deyuf): fix this. not going through get() will cause missing keys like wd_mult
        # optimizer.load_state_dict(state_dict)

    def test_sharded_state_dict(self):
        """Test LayerWiseDistributedOptimizer sharded_state_dict method."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        for param in model.parameters():
            param.grad = torch.randn_like(param)
        optimizer.step()

        # Get model sharded state dict
        model_sharded_state_dict = model.sharded_state_dict()

        # Test sharded_state_dict
        sharded_state_dict = optimizer.sharded_state_dict(model_sharded_state_dict)

        # Verify the sharded_state_dict is not None and has expected structure
        assert sharded_state_dict is not None, "Sharded state dict should not be None"
        assert (
            'optimizer' in sharded_state_dict
        ), "Sharded state dict should contain 'optimizer' key"

        # Verify that replica_id is set correctly (should be 0 for DP dimension)
        from megatron.core.dist_checkpointing import ShardedTensor
        from megatron.core.dist_checkpointing.dict_utils import nested_values

        for sh_base in nested_values(sharded_state_dict):
            if isinstance(sh_base, ShardedTensor):
                assert (
                    len(sh_base.replica_id) == 3
                ), f'Expected replica_id format (PP, TP, DP), got: {sh_base.replica_id}'
                assert (
                    sh_base.replica_id[2] == 0
                ), f'Expected DP replica_id to be 0 for layer-wise optimizer, got: {sh_base.replica_id[2]}'

    def test_multiple_optimizers(self):
        """Test LayerWiseDistributedOptimizer with multiple chained optimizers.

        This test properly tests allgather functionality with multiple ranks.
        """
        model = SimpleModel().bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        # Split parameters into two groups for testing multiple optimizers
        params = list(model.parameters())
        mid_point = len(params) // 2
        param_groups_1 = [{'params': params[:mid_point]}]
        param_groups_2 = [{'params': params[mid_point:]}]

        # Create two separate base optimizers
        base_optimizer_1 = torch.optim.Adam(param_groups_1, lr=optimizer_config.lr)
        base_optimizer_2 = torch.optim.Adam(param_groups_2, lr=optimizer_config.lr)

        wrapped_optimizer_1 = FP32Optimizer(base_optimizer_1, optimizer_config, None)
        wrapped_optimizer_2 = FP32Optimizer(base_optimizer_2, optimizer_config, None)

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = LayerWiseDistributedOptimizer(
            [wrapped_optimizer_1, wrapped_optimizer_2], optimizer_config, pg_collection
        )

        assert len(optimizer.chained_optimizers) == 2, "Should have two chained optimizers"

        # Set gradients and test optimizer step - this will trigger allgather
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

    def test_bf16_wrapping(self):
        """Test LayerWiseDistributedOptimizer automatically wraps optimizer with bf16."""
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # Verify bf16 wrapping happened
        assert isinstance(
            optimizer.chained_optimizers[0], Float16OptimizerWithFloat16Params
        ), "Optimizer should be wrapped in Float16OptimizerWithFloat16Params"

        for param in model.parameters():
            param.grad = torch.randn_like(param)

        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

    def test_bf16_error(self):
        """Test LayerWiseDistributedOptimizer raises error when receiving pre-wrapped Float16 optimizer."""
        model = SimpleModel().bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=False)
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )

        optimizer_config = OptimizerConfig(
            optimizer='adam', lr=0.01, bf16=True, use_distributed_optimizer=False
        )

        # Create base optimizer and manually wrap in Float16 optimizer
        param_groups = [{'params': list(model.parameters())}]
        base_optimizer = torch.optim.Adam(param_groups, lr=optimizer_config.lr)
        wrapped_optimizer = Float16OptimizerWithFloat16Params(
            base_optimizer, optimizer_config, None, None
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        # Should raise TypeError when receiving already-wrapped Float16 optimizer
        with pytest.raises(
            TypeError, match='LayerWiseDistributedOptimizer received Float16 optimizer already'
        ):
            LayerWiseDistributedOptimizer([wrapped_optimizer], optimizer_config, pg_collection)

    def _run_parameter_update_test(self, model_class=SimpleModel):
        """Helper method to test parameter updates with a given model class.

        Args:
            model_class: Model class to use for testing
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer(model_class=model_class)

        # Create reference model and optimizer using the same function
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=model_class, use_layer_wise=False, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)

    def test_parameter_updates(self):
        """Test LayerWiseDistributedOptimizer actually updates model parameters."""
        self._run_parameter_update_test()

    def test_parameter_updates_insufficient_parameters(self):
        """Test LayerWiseDistributedOptimizer when there are insufficient parameters for all ranks.

        Uses a tiny model with only 1 layer (2 parameters: weight and bias).
        This will be insufficient when world size > 2.
        """
        self._run_parameter_update_test(model_class=TinyModel)

    def test_broadcast_vs_allgather(self):
        """Test LayerWiseDistributedOptimizer allgather code agains broadcast code."""
        model, optimizer, pg_collection = self.create_model_and_optimizer(model_class=SimpleModel)

        # Create reference model and optimizer using the same function
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=SimpleModel, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Verify at least some parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # step() internal call allgather_params. replace reference object with bcast
        reference_optimizer.allgather_params = reference_optimizer.broadcast_params
        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)
