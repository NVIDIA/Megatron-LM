# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import Version

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import get_pg_rank, get_pg_size
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
            use_layer_wise: If True, use LayerWiseDistributedOptimizer via dist_muon;
                          if False, use standard muon ChainedOptimizer (for reference)

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
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            clip_grad=clip_grad,
            muon_tp_mode="duplicated",
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=use_layer_wise,
            pg_collection=pg_collection,
        )
        return model, optimizer, pg_collection

    def create_model_and_optimizer_with_overlap_param_gather(
        self,
        model_class=SimpleModel,
        clip_grad=1.0,
        model_kwargs=None,
        copy_from=None,
        async_allgather=True,
        grad_reduce_in_fp32=False,
        bucket_size=None,
    ):
        """Create model, DDP wrapper, and optimizer with overlap-param-gather enabled.

        This variant sets overlap_param_gather=True in DDP config and uses
        get_megatron_muon_optimizer with layer_wise_distributed_optimizer=True,
        enabling the bucket-based async param gather path.

        Args:
            model_class: Model class to instantiate
            clip_grad: Optional gradient clipping value
            model_kwargs: Optional kwargs for model initialization
            copy_from: Optional DDP model to copy weights from
            async_allgather: If True, defer param all-gather to bucket infrastructure
            grad_reduce_in_fp32: If True, reduce grads in fp32 (regression test for dtype fix)
            bucket_size: Maximum number of parameters per bucket (None = single bucket)

        Returns:
            tuple: (model, optimizer, pg_collection)
        """
        if model_kwargs is None:
            model_kwargs = {}

        model = model_class(**model_kwargs).bfloat16().cuda()
        model.requires_grad_(True)

        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_param_gather=True,
            grad_reduce_in_fp32=grad_reduce_in_fp32,
            bucket_size=bucket_size,
        )
        model = DistributedDataParallel(
            TransformerConfig(num_attention_heads=1, num_layers=1), ddp_config, model
        )
        if copy_from:
            model.module.load_state_dict(copy_from.module.state_dict())
        else:
            model.broadcast_params()

        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            weight_decay=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            clip_grad=clip_grad,
            overlap_param_gather=async_allgather,
            muon_tp_mode="duplicated",
        )

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=True,
            pg_collection=pg_collection,
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

        # Verify the sharded_state_dict is not None and has expected structure.
        # With multiple chained optimizers (muon + adam), the top-level keys are
        # integer indices; each sub-dict should contain an 'optimizer' key.
        assert sharded_state_dict is not None, "Sharded state dict should not be None"
        if isinstance(sharded_state_dict, dict) and all(
            isinstance(k, int) for k in sharded_state_dict.keys()
        ):
            for idx, sub_dict in sharded_state_dict.items():
                assert (
                    'optimizer' in sub_dict
                ), f"Sub-dict {idx} should contain 'optimizer' key"
        else:
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

        Uses get_megatron_muon_optimizer which produces multiple chained optimizers
        (muon for 2D weights + adam for 1D biases). Tests allgather with multiple ranks.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer()

        # get_megatron_muon_optimizer produces muon + adam chained optimizers
        assert len(optimizer.chained_optimizers) >= 2, "Should have multiple chained optimizers"

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

        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(with_context_parallel=True)
        pg_collection.expt_dp = parallel_state.get_expert_data_parallel_group()

        # Create muon optimizer (non-layer-wise) — produces Float16-wrapped chained optimizers
        optimizer_config = OptimizerConfig(
            optimizer='muon',
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=False,
            muon_tp_mode="duplicated",
        )
        muon_optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=False,
            pg_collection=pg_collection,
        )

        # Extract a Float16-wrapped chained optimizer
        wrapped_optimizer = muon_optimizer.chained_optimizers[0]
        assert isinstance(wrapped_optimizer, Float16OptimizerWithFloat16Params)

        # Should raise TypeError when receiving already-wrapped Float16 optimizer
        # Use a fresh config since get_megatron_muon_optimizer mutates config.optimizer
        lw_config = OptimizerConfig(
            optimizer='muon', lr=0.01, bf16=True, use_distributed_optimizer=False
        )
        with pytest.raises(
            TypeError, match='LayerWiseDistributedOptimizer received Float16 optimizer already'
        ):
            LayerWiseDistributedOptimizer([wrapped_optimizer], lw_config, pg_collection)

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

    # ---- Overlap-param-gather tests ----

    def test_overlap_param_gather_basic(self):
        """Test overlap-param-gather path: init, forward/backward/step, bucket-based param sync."""
        model, optimizer, pg_collection = (
            self.create_model_and_optimizer_with_overlap_param_gather()
        )

        assert optimizer is not None, "Optimizer should not be None"
        assert optimizer.async_allgather, "async_allgather should be True"

        reference_model = self.create_reference_model(model)

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # step() updates local params but skips allgather (async_allgather=True)
        update_successful, grad_norm, num_zeros = optimizer.step()

        assert update_successful, "Optimizer step should be successful"

        # Manually sync params through the bucket-based param sync path
        # force_sync=True does synchronous allgather via bucket infrastructure
        model.start_param_sync(force_sync=True)

        # Verify parameters were updated
        params_updated = 0
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            if not torch.equal(param.data, ref_param.data):
                params_updated += 1

        assert params_updated > 0, "At least some parameters should be updated"

        # Verify all ranks have the same updated parameters
        dp_size = get_pg_size(pg_collection.dp_cp)

        if dp_size > 1:
            for name, param in model.named_parameters():
                param_list = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(param_list, param.data, group=pg_collection.dp_cp)

                for i in range(1, dp_size):
                    torch.testing.assert_close(
                        param_list[0],
                        param_list[i],
                        msg=f"Parameter {name} differs between rank 0 and rank {i}",
                    )

    def test_overlap_param_gather_parameter_updates(self):
        """Test overlap-param-gather produces same parameter updates as standard optimizer."""
        model, optimizer, pg_collection = (
            self.create_model_and_optimizer_with_overlap_param_gather()
        )

        # Create reference model with standard (non-layer-wise) optimizer
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            use_layer_wise=False, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        # step() with async_allgather=True: updates but no allgather
        optimizer.step()
        # Manually sync params via bucket infrastructure
        model.start_param_sync(force_sync=True)

        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)

    def test_overlap_param_gather_vs_sync_allgather(self):
        """Key correctness test: overlap path and sync allgather produce identical updates.

        Compares:
        - Overlap path: async_allgather=True, bucket-based param sync
        - Sync path: async_allgather=False, optimizer.allgather_params() in step()
        """
        # Create overlap model
        overlap_model, overlap_optimizer, pg_collection = (
            self.create_model_and_optimizer_with_overlap_param_gather(async_allgather=True)
        )

        # Create sync model with same weights (overlap_param_gather=True but sync allgather)
        sync_model, sync_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=overlap_model
        )

        # Verify initial parameters match
        for op, sp in zip(overlap_model.parameters(), sync_model.parameters()):
            assert torch.equal(op.data, sp.data)

        # Set identical gradients on both
        for op, sp in zip(overlap_model.parameters(), sync_model.parameters()):
            grad_value = torch.randn_like(op)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            op.main_grad = grad_value.clone().detach()
            sp.main_grad = grad_value.clone().detach()

        # Overlap path: step + manual sync
        overlap_optimizer.step()
        overlap_model.start_param_sync(force_sync=True)

        # Sync path: step (includes allgather_params)
        sync_optimizer.step()

        # Both paths should produce identical parameter values
        for op, sp in zip(overlap_model.parameters(), sync_model.parameters()):
            torch.testing.assert_close(
                op.data,
                sp.data,
                rtol=0,
                atol=0,
                msg="Overlap and sync allgather paths produced different parameter updates",
            )

    def test_overlap_param_gather_bucket_lw_params(self):
        """Verify bucket.lw_params_list is populated when async_allgather is enabled."""
        model, optimizer, pg_collection = (
            self.create_model_and_optimizer_with_overlap_param_gather()
        )

        dp_size = get_pg_size(pg_collection.dp_cp)

        for bucket_group in model.bucket_groups:
            for bucket in bucket_group.buckets:
                # lw_params_list should be populated by set_bucket_lw_params_list
                assert (
                    bucket.lw_params_list is not None
                ), "bucket.lw_params_list should be populated"
                assert (
                    len(bucket.lw_params_list) == dp_size
                ), f"Expected {dp_size} per-rank lists, got {len(bucket.lw_params_list)}"

                # The union of all per-rank param lists should cover all bucket params
                all_lw_params = set()
                for rank_params in bucket.lw_params_list:
                    for p in rank_params:
                        all_lw_params.add(p)
                assert (
                    all_lw_params == bucket.params
                ), "Union of per-rank lw_params should equal bucket params"

                # lw_param_flat_sizes should be populated and have correct length
                assert bucket.lw_param_flat_sizes is not None
                assert len(bucket.lw_param_flat_sizes) == dp_size

                # Each flat size should equal the sum of param numels for that rank
                for rank_idx in range(dp_size):
                    expected_size = sum(p.numel() for p in bucket.lw_params_list[rank_idx])
                    assert bucket.lw_param_flat_sizes[rank_idx] == expected_size, (
                        f"Rank {rank_idx}: expected flat_size {expected_size}, "
                        f"got {bucket.lw_param_flat_sizes[rank_idx]}"
                    )

    def test_overlap_param_gather_vs_standard_ddp(self):
        """Verify DDP with overlap_param_gather=True produces same results as standard DDP.

        Both use LayerWiseDistributedOptimizer but with different DDP configs:
        - Overlap path: overlap_param_gather=True (padded buffers)
        - Standard path: overlap_param_gather=False (unpadded buffers)
        """
        # Create overlap-param-gather model (sync allgather for simpler comparison)
        opg_model, opg_optimizer, pg_collection = (
            self.create_model_and_optimizer_with_overlap_param_gather(async_allgather=False)
        )

        # Create standard model with same weights
        std_model, std_optimizer, _ = self.create_model_and_optimizer(copy_from=opg_model)

        # Set identical gradients
        for op, sp in zip(opg_model.parameters(), std_model.parameters()):
            assert torch.equal(op.data, sp.data)
            grad_value = torch.randn_like(op)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            op.main_grad = grad_value.clone().detach()
            sp.main_grad = grad_value.clone().detach()

        opg_optimizer.step()
        std_optimizer.step()

        # Both should produce identical parameter values
        for op, sp in zip(opg_model.parameters(), std_model.parameters()):
            torch.testing.assert_close(
                op.data,
                sp.data,
                rtol=1e-5,
                atol=1e-5,
                msg="Overlap-param-gather and standard paths produced different updates",
            )

    def test_overlap_param_gather_insufficient_parameters(self):
        """Test overlap-param-gather with TinyModel (only 2 params).

        Many ranks will have no assigned params when world_size > 2.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            model_class=TinyModel
        )

        # Create reference model with standard (non-layer-wise) optimizer
        reference_model, reference_optimizer, _ = self.create_model_and_optimizer(
            model_class=TinyModel, use_layer_wise=False, copy_from=model
        )

        # Set same gradients on both models
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            assert torch.equal(param.data, ref_param.data)
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()
        model.start_param_sync(force_sync=True)

        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=1e-5, atol=1e-5)

    def test_overlap_param_gather_broadcast_vs_allgather(self):
        """Test overlap-param-gather: allgather vs broadcast produce same results."""
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            model_class=SimpleModel, async_allgather=False
        )

        # Create reference model with overlap-param-gather path too
        reference_model, reference_optimizer, _ = (
            self.create_model_and_optimizer_with_overlap_param_gather(
                model_class=SimpleModel, async_allgather=False, copy_from=model
            )
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

        # step() internally calls allgather_params. Replace reference with broadcast.
        reference_optimizer.allgather_params = reference_optimizer.broadcast_params
        reference_optimizer.step()

        # Verify updated values match reference optimizer
        for param, ref_param in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(param.data, ref_param.data, rtol=0, atol=0)

    def test_overlap_param_gather_multi_iteration(self):
        """Test overlap-param-gather correctness over multiple training iterations.

        Runs multiple forward/backward/step iterations using the async allgather path.
        After each iteration, manually syncs params and verifies they match a reference
        model using the sync path.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True
        )

        # Create reference model with sync allgather for comparison
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model
        )

        for iteration in range(3):
            # Set identical gradients on both models
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                grad_value = torch.randn_like(param)
                torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
                param.main_grad = grad_value.clone().detach()
                ref_param.main_grad = grad_value.clone().detach()

            # Async path: step (no allgather) + manual sync
            optimizer.step()
            model.start_param_sync(force_sync=True)

            # Sync path: step (includes allgather)
            ref_optimizer.step()

            # Verify parameters match after each iteration
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                torch.testing.assert_close(
                    param.data,
                    ref_param.data,
                    rtol=0,
                    atol=0,
                    msg=f"Parameters diverged at iteration {iteration}",
                )

    def test_overlap_param_gather_async_dispatch_and_finish(self):
        """Test async dispatch + finish_param_sync cycle (the actual runtime path).

        start_param_sync() (no force_sync) dispatches async all-gathers, then
        finish_param_sync() waits on the handle and unflattens gathered params.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True
        )
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model
        )

        # Set identical gradients on both models
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        # Async path: step (no allgather) + async dispatch + explicit finish
        optimizer.step()
        model.start_param_sync()  # async dispatch to all bucket groups
        for bucket_group in model.bucket_groups:
            bucket_group.finish_param_sync(skip_next_bucket_dispatch=True)

        # Sync path: step (includes allgather)
        ref_optimizer.step()

        # Verify params match sync path
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=0,
                atol=0,
                msg="Async dispatch + finish path produced different params than sync path",
            )

        # Verify all ranks have identical parameters
        dp_size = get_pg_size(pg_collection.dp_cp)
        if dp_size > 1:
            for name, param in model.named_parameters():
                param_list = [torch.zeros_like(param.data) for _ in range(dp_size)]
                torch.distributed.all_gather(param_list, param.data, group=pg_collection.dp_cp)
                for i in range(1, dp_size):
                    torch.testing.assert_close(
                        param_list[0],
                        param_list[i],
                        msg=f"Parameter {name} differs between rank 0 and rank {i}",
                    )

    def test_overlap_param_gather_finish_chains_next_bucket(self):
        """Test that finish_param_sync() dispatches next_param_gather_bucket_group.

        Uses a small bucket_size to force multiple bucket groups, then dispatches
        only the last bucket group and verifies that finishing it chains to the next.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True, bucket_size=2000
        )

        bucket_groups = model.bucket_groups
        if len(bucket_groups) <= 1:
            pytest.skip("Need multiple bucket groups to test chaining")

        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model, bucket_size=2000
        )

        # Set identical gradients on both models
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()

        # Dispatch ONLY the last bucket group (which has next_param_gather_bucket_group set)
        last_bg = bucket_groups[-1]
        last_bg.start_param_sync()

        # Verify: next bucket group has NOT been dispatched yet
        next_bg = last_bg.next_param_gather_bucket_group
        assert next_bg is not None, "Last bucket group should have a next"
        assert not next_bg.param_gather_dispatched, "Next bucket should not be dispatched yet"

        # Finish the last bucket group — should chain-dispatch the next one
        last_bg.finish_param_sync()

        # Verify: next bucket group IS now dispatched via chaining
        assert (
            next_bg.param_gather_dispatched
        ), "finish_param_sync should have dispatched next bucket group"

        # Finish remaining bucket groups through the chain
        for bg in reversed(bucket_groups[:-1]):
            bg.finish_param_sync()

        # Reference: sync step
        ref_optimizer.step()

        # Verify params match
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=0,
                atol=0,
                msg="Chained bucket finish produced different params than sync path",
            )

    def test_overlap_param_gather_forward_pre_hook(self):
        """Test forward pre-hooks trigger finish_param_sync during model(input).

        After async dispatch, running model(input) fires forward pre-hooks that
        call finish_param_sync() on each bucket group, completing the param sync.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True
        )
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model
        )

        # Set identical gradients on both models
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        # Async path: step (no allgather) + async dispatch
        optimizer.step()
        model.start_param_sync()  # dispatch async broadcasts

        # Forward pass triggers hooks that call finish_param_sync()
        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')
        output = model(input_tensor)

        # Sync path: step (includes allgather)
        ref_optimizer.step()

        # Verify params match
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=0,
                atol=0,
                msg="Forward pre-hook path produced different params than sync path",
            )

    def test_overlap_param_gather_grad_reduce_in_fp32(self):
        """Regression test: grad_reduce_in_fp32 must not cause dtype mismatch in broadcasts.

        When grad_reduce_in_fp32=True, the grad buffer dtype is fp32 but broadcast
        buffers must use param dtype (bf16). Without the fix (commit cbed167fc), this
        would cause a dtype mismatch error in the per-rank broadcast calls.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True, grad_reduce_in_fp32=True
        )
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model, grad_reduce_in_fp32=True
        )

        # Set identical gradients on both models
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        # Async path: step + force_sync
        optimizer.step()
        model.start_param_sync(force_sync=True)

        # Sync path: step (includes allgather)
        ref_optimizer.step()

        # Verify params match
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=1e-5,
                atol=1e-5,
                msg="grad_reduce_in_fp32 path produced different params than reference",
            )

    def test_overlap_param_gather_hook_enable_disable_cycle(self):
        """Test the training loop's hook lifecycle: disable → manual sync → enable → forward.

        The training loop disables hooks before iteration 1 (for initialization),
        then enables them for subsequent iterations. This test exercises that cycle.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True
        )
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model
        )

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')

        # Iteration 1: hooks disabled, manual sync
        model.disable_forward_pre_hook(param_sync=False)

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()
        model.start_param_sync(force_sync=True)  # manual sync

        ref_optimizer.step()

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=0,
                atol=0,
                msg="Params diverged after iteration 1 (hooks disabled)",
            )

        # Iteration 2: hooks re-enabled, forward pass triggers sync
        model.enable_forward_pre_hook()

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        optimizer.step()
        model.start_param_sync()  # async dispatch
        output = model(input_tensor)  # hooks finish sync

        ref_optimizer.step()

        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=0,
                atol=0,
                msg="Params diverged after iteration 2 (hooks re-enabled)",
            )

    def test_overlap_param_gather_multi_iteration_with_hooks(self):
        """Test multiple iterations using forward pre-hooks (not manual force_sync).

        Runs 3 iterations where each iteration uses: set grads → step → async dispatch →
        forward pass (hooks wait+unflatten). Compares against reference model using sync
        allgather after each iteration.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True
        )
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model
        )

        input_tensor = torch.randn(16, 80, dtype=torch.bfloat16, device='cuda')

        for iteration in range(3):
            # Set identical gradients on both models
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                grad_value = torch.randn_like(param)
                torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
                param.main_grad = grad_value.clone().detach()
                ref_param.main_grad = grad_value.clone().detach()

            # Async path: step + dispatch + forward (hooks wait+unflatten)
            optimizer.step()
            model.start_param_sync()  # async dispatch
            output = model(input_tensor)  # hooks trigger finish_param_sync

            # Sync path: step (includes allgather)
            ref_optimizer.step()

            # Verify parameters match after each iteration
            for param, ref_param in zip(model.parameters(), ref_model.parameters()):
                torch.testing.assert_close(
                    param.data,
                    ref_param.data,
                    rtol=0,
                    atol=0,
                    msg=f"Parameters diverged at iteration {iteration}",
                )

    def test_overlap_param_gather_start_sync_with_autograd(self):
        """Regression test: start_param_sync must work when autograd is active.

        _flatten_dense_tensors on params with requires_grad=True produces a tensor
        that also requires grad.  Since all_gather writes into gather_list entries
        in-place and the local rank's slot reuses src, this triggers:
            RuntimeError: a view of a leaf Variable that requires grad is being
            used in an in-place operation.
        The fix is to .detach() the flattened tensor before using it as src.

        This test calls start_param_sync (synchronous via force_sync) WITHOUT
        torch.no_grad() to reproduce the exact scenario that occurs during the
        forward pass when finish_param_sync chains to start_param_sync for the
        next bucket group.
        """
        model, optimizer, pg_collection = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=True
        )
        ref_model, ref_optimizer, _ = self.create_model_and_optimizer_with_overlap_param_gather(
            async_allgather=False, copy_from=model
        )

        # Confirm params require grad (the precondition for this bug).
        for param in model.parameters():
            assert param.requires_grad, "Test requires params with requires_grad=True"

        # Set identical gradients on both models.
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            grad_value = torch.randn_like(param)
            torch.distributed.broadcast(grad_value, src=0, group=pg_collection.dp_cp)
            param.main_grad = grad_value.clone().detach()
            ref_param.main_grad = grad_value.clone().detach()

        # Step both optimizers (async path skips allgather, ref path includes it).
        optimizer.step()
        ref_optimizer.step()

        # Call start_param_sync with autograd ENABLED (no torch.no_grad()).
        # Before the .detach() fix, this would raise RuntimeError.
        model.start_param_sync(force_sync=True)

        # Verify gathered params match the reference.
        for param, ref_param in zip(model.parameters(), ref_model.parameters()):
            torch.testing.assert_close(
                param.data,
                ref_param.data,
                rtol=0,
                atol=0,
                msg="Params incorrect after start_param_sync with autograd enabled",
            )
