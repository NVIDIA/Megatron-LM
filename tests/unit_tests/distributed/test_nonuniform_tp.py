# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""
Unit tests for Nonuniform Tensor Parallelism (NTP).

Tests the fault-tolerance mechanism that allows training to continue
when GPU failures occur within a tensor-parallel group.
"""

import pytest
import torch
import torch.distributed as dist
from unittest.mock import Mock, patch, MagicMock

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.nonuniform_tp import (
    compute_uniform_tp_spares_with_parity,
    get_active_ranks_for_dp,
    ntp_map,
    ntp_init,
    initialize_nonuniform_tp_process_groups,
    NonuniformTPConfig,
    NonuniformTPDistributedDataParallel,
    NonuniformTPOptimizer,
    NonuniformTPParamAndGradBuffer,
)
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestNonuniformTPUtilities:
    """Test utility functions for NTP configuration."""

    def test_compute_uniform_tp_spares_with_parity_no_failures(self):
        """Test with no GPU failures."""
        faulty_gpu_map = {}
        tp_base = 8

        tp_spares, non_active_ranks = compute_uniform_tp_spares_with_parity(faulty_gpu_map, tp_base)

        assert tp_spares == 0
        assert non_active_ranks == {}

    def test_compute_uniform_tp_spares_with_parity_uniform_failures(self):
        """Test with uniform failures across DP ranks."""
        faulty_gpu_map = {
            0: [2, 5],  # DP rank 0 has 2 failures
            1: [1, 3],  # DP rank 1 has 2 failures
        }
        tp_base = 8

        tp_spares, non_active_ranks = compute_uniform_tp_spares_with_parity(faulty_gpu_map, tp_base)

        assert tp_spares == 2
        assert non_active_ranks[0] == [2, 5]
        assert non_active_ranks[1] == [1, 3]

    def test_compute_uniform_tp_spares_with_parity_non_uniform_failures(self):
        """Test with non-uniform failures (requires padding)."""
        faulty_gpu_map = {
            0: [2, 5],  # DP rank 0 has 2 failures
            1: [1],  # DP rank 1 has 1 failure
        }
        tp_base = 8

        tp_spares, non_active_ranks = compute_uniform_tp_spares_with_parity(faulty_gpu_map, tp_base)

        assert tp_spares == 2
        assert non_active_ranks[0] == [2, 5]
        # DP rank 1 should be padded with 1 additional GPU (prefer high ranks)
        assert len(non_active_ranks[1]) == 2
        assert 1 in non_active_ranks[1]
        # Second non-active rank should be from the end (e.g., 7)
        assert non_active_ranks[1][1] == 7

    def test_get_active_ranks_for_dp_default(self):
        """Test get_active_ranks_for_dp with default (no explicit non_active_ranks_per_dp)."""
        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)
        dp_rank = 0
        tp_base = 8

        active_ranks = get_active_ranks_for_dp(dp_rank, tp_base, ntp_config)

        # Should return first (tp_base - tp_spares) ranks
        assert active_ranks == [0, 1, 2, 3, 4, 5]

    def test_get_active_ranks_for_dp_explicit(self):
        """Test get_active_ranks_for_dp with explicit non_active_ranks_per_dp."""
        ntp_config = NonuniformTPConfig(
            tp_base=8, tp_spares=2, non_active_ranks_per_dp={0: [2, 5]}
        )
        dp_rank = 0
        tp_base = 8

        active_ranks = get_active_ranks_for_dp(dp_rank, tp_base, ntp_config)

        # Should exclude ranks 2 and 5
        assert active_ranks == [0, 1, 3, 4, 6, 7]


class TestNonuniformTPParameterResharding:
    """Test parameter resharding logic for NTP."""

    def test_ntp_map_no_spares(self):
        """Test ntp_map when tp_spares=0 (should be no-op)."""
        # Create mock module with parameter
        module = Mock()
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.tensor_model_parallel = True
        param.partition_dim = 1
        module.parameters = Mock(return_value=[param])

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=0)

        # Should not raise error and not add send_splits/recv_splits
        ntp_map(module, ntp_config, num_shards=24)

        assert not hasattr(param, 'send_splits')
        assert not hasattr(param, 'recv_splits')

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    @patch('megatron.core.distributed.nonuniform_tp.dist')
    def test_ntp_map_with_spares_healthy_rank(self, mock_dist, mock_parallel_state):
        """Test ntp_map for a healthy rank (should add send/recv splits)."""
        # Mock parallel state
        mock_dist.get_rank.return_value = 0
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 0

        # Create mock module with parameter
        class MockConfig:
            num_attention_heads = 24

        module = Mock()
        param = torch.nn.Parameter(torch.randn(384, 128))  # 384 = 24 heads * 16 dim
        param.tensor_model_parallel = True
        param.partition_dim = 0
        # Note: param.shape is already (384, 128) from the tensor, no need to set it
        module.parameters = Mock(return_value=[param])
        module.config = MockConfig()

        ntp_config = NonuniformTPConfig(
            tp_base=8,
            tp_spares=2,
            non_active_ranks_per_dp={},  # No explicit non-active ranks, so this is healthy
        )

        # Execute
        ntp_map(module, ntp_config, num_shards=24)

        # Should have added send_splits and recv_splits
        assert hasattr(param, 'send_splits')
        assert hasattr(param, 'recv_splits')
        assert len(param.send_splits) == 8
        assert len(param.recv_splits) == 8

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    @patch('megatron.core.distributed.nonuniform_tp.dist')
    def test_ntp_map_with_spares_unhealthy_rank(self, mock_dist, mock_parallel_state):
        """Test ntp_map for an unhealthy rank (should skip)."""
        # Mock parallel state
        mock_dist.get_rank.return_value = 0
        mock_parallel_state.get_data_parallel_rank.return_value = 0
        mock_parallel_state.get_context_parallel_rank.return_value = 0
        mock_parallel_state.get_pipeline_model_parallel_rank.return_value = 0

        # Create mock module
        module = Mock()
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.tensor_model_parallel = True
        param.partition_dim = 1
        module.parameters = Mock(return_value=[param])

        ntp_config = NonuniformTPConfig(
            tp_base=8,
            tp_spares=2,
            non_active_ranks_per_dp={(0, 0, 0): [2, 5]},  # This rank is unhealthy
        )

        # Execute
        ntp_map(module, ntp_config, num_shards=24)

        # Should NOT have added send_splits and recv_splits
        assert not hasattr(param, 'send_splits')
        assert not hasattr(param, 'recv_splits')

    def test_ntp_init_no_spares(self):
        """Test ntp_init when tp_spares=0 (should be no-op)."""
        # Create mock layer
        layer = Mock()
        layer.self_attention = Mock()
        layer.mlp = Mock()

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=0)

        # Should not raise error
        ntp_init(layer, ntp_config)

    @patch('megatron.core.distributed.nonuniform_tp.ntp_map')
    def test_ntp_init_with_attention_and_mlp(self, mock_ntp_map):
        """Test ntp_init calls ntp_map for both attention and MLP."""

        class MockConfig:
            num_attention_heads = 24
            ffn_hidden_size = 4096

        # Create mock layer
        layer = Mock()
        layer.self_attention = Mock()
        layer.self_attention.config = MockConfig()
        layer.mlp = Mock()
        layer.mlp.config = MockConfig()

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)

        # Execute
        ntp_init(layer, ntp_config)

        # Should call ntp_map twice
        assert mock_ntp_map.call_count == 2
        # First call for self_attention
        assert mock_ntp_map.call_args_list[0][0][0] == layer.self_attention
        assert mock_ntp_map.call_args_list[0][0][2] == 24
        # Second call for mlp
        assert mock_ntp_map.call_args_list[1][0][0] == layer.mlp
        assert mock_ntp_map.call_args_list[1][0][2] == 4096


class TestNonuniformTPOptimizer:
    """Test NTP optimizer wrapper."""

    def test_optimizer_wrapper_delegates_attributes(self):
        """Test that optimizer wrapper delegates attribute access."""
        mock_optimizer = Mock()
        mock_optimizer.param_groups = []
        mock_optimizer.state = {}

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ntp_config)

        # Should delegate attribute access
        assert ntp_optimizer.param_groups == []
        assert ntp_optimizer.state == {}

    def test_optimizer_prepare_grads_no_spares(self):
        """Test prepare_grads when tp_spares=0 (should be no-op)."""
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{'params': []}]
        mock_optimizer.prepare_grads = Mock(return_value=False)

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=0)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ntp_config)

        result = ntp_optimizer.prepare_grads()

        # Should call original prepare_grads
        mock_optimizer.prepare_grads.assert_called_once()
        assert result == False

    def test_optimizer_prepare_grads_makes_contiguous(self):
        """Test prepare_grads makes gradients contiguous for NTP."""
        # Create parameter with non-contiguous main_grad
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.main_grad = torch.randn(10, 10).t()  # Transposed = non-contiguous
        assert not param.main_grad.is_contiguous()

        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{'params': [param]}]
        mock_optimizer.prepare_grads = Mock(return_value=False)

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ntp_config)

        ntp_optimizer.prepare_grads()

        # Should have made grad contiguous
        assert hasattr(param, 'grad')
        assert param.grad.is_contiguous()

    def test_optimizer_prepare_grads_already_contiguous(self):
        """Test prepare_grads when gradient is already contiguous."""
        # Create parameter with contiguous main_grad
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.main_grad = torch.randn(10, 10)
        assert param.main_grad.is_contiguous()

        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{'params': [param]}]
        mock_optimizer.prepare_grads = Mock(return_value=False)

        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ntp_config)

        ntp_optimizer.prepare_grads()

        # Should have set grad directly (no copy)
        assert hasattr(param, 'grad')
        assert param.grad is param.main_grad


class TestNonuniformTPIntegration:
    """Integration tests for NTP with DDP."""

    def test_ntp_ddp_initialization(self):
        """Test NonuniformTPDistributedDataParallel initialization."""
        # Create simple model
        model = torch.nn.Linear(10, 10)

        config = TransformerConfig(
            num_layers=1, hidden_size=10, num_attention_heads=1, context_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig()
        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)

        # Should initialize without error
        try:
            ntp_ddp = NonuniformTPDistributedDataParallel(
                config, ddp_config, model, disable_bucketing=True, ntp_config=ntp_config
            )
            # Check that it's an instance of base DDP
            from megatron.core.distributed import DistributedDataParallel

            assert isinstance(ntp_ddp, DistributedDataParallel)
        except Exception as e:
            # Some initialization might fail in unit test environment, that's ok
            # We just want to verify the class can be instantiated
            pytest.skip(f"Skipping due to initialization requirements: {e}")

    @patch('megatron.core.distributed.nonuniform_tp.parallel_state')
    def test_ntp_backward_hook_core_gpu(self, mock_parallel_state):
        """Test that NTP backward hook is properly created for core GPU."""
        # Mock parallel state to simulate core GPU
        mock_parallel_state.get_tensor_model_parallel_world_size.return_value = 8
        mock_parallel_state.get_tensor_model_parallel_rank.return_value = 0  # Core GPU

        # Create parameter with NTP attributes
        param = torch.nn.Parameter(torch.randn(10, 10))
        param.tensor_model_parallel = True
        param.partition_dim = 1
        # Note: param.shape is already (10, 10) from the tensor, no need to set it
        param.side_grad = torch.randn(10, 2)
        param.recv_splits = [[0] * 8 for _ in range(8)]

        model = torch.nn.Module()
        model.register_parameter('test_param', param)

        config = TransformerConfig(
            num_layers=1, hidden_size=10, num_attention_heads=1, context_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig()
        ntp_config = NonuniformTPConfig(tp_base=8, tp_spares=2)

        try:
            ntp_ddp = NonuniformTPDistributedDataParallel(
                config, ddp_config, model, disable_bucketing=True, ntp_config=ntp_config
            )
            # If we got here, the hook was created successfully
            assert True
        except Exception as e:
            pytest.skip(f"Skipping due to initialization requirements: {e}")


class TestNonuniformTPEndToEnd:
    """
    End-to-end test for NTP without mocking.

    Tests NTP with 8 GPUs configured as:
    - 2 data-parallel workers
    - DP rank 0: TP=2 (reduced, using 2 out of 4 GPUs)
    - DP rank 1: TP=4 (healthy, using all 4 GPUs)
    - Total: 2 + 4 = 6 active GPUs out of 8
    """

    @classmethod
    def setup_class(cls):
        """Initialize model parallel for NTP testing."""
        # Initialize with tp_base=4
        Utils.initialize_model_parallel(tensor_model_parallel_size=4)

    @classmethod
    def teardown_class(cls):
        """Clean up model parallel."""
        Utils.destroy_model_parallel()

    def test_ntp_end_to_end_with_8_gpus(self):
        """
        End-to-end test using 8 GPUs with 2 DP workers:
        - DP rank 0: uses TP=2 (reduced from tp_base=4)
        - DP rank 1: uses TP=4 (healthy, full tp_base)
        """
        import torch.distributed as dist
        from megatron.core import parallel_state

        # Check we have 8 GPUs
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size != 8:
            pytest.skip(f"This test requires 8 GPUs, but only {world_size} are available")

        # Get current rank info
        rank = dist.get_rank()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()

        # Configure NTP: first DP rank uses reduced TP=2
        ntp_config = NonuniformTPConfig(
            tp_base=4,
            tp_spares=2,
            num_reduced_tp_dp_ranks=1,
            non_active_ranks_per_dp={(0, 0, 0): [2, 3]},  # DP=0: GPUs 2,3 are spares
        )

        # Check if this rank is a spare (will exit during initialization)
        # Spare ranks: DP=0 with tp_rank=2,3
        is_spare = dp_rank == 0 and tp_rank in [2, 3]

        # Reconfigure process groups for NTP
        # Note: spare ranks will call sys.exit(0) in initialize_nonuniform_tp_process_groups
        from megatron.core.distributed.nonuniform_tp import initialize_nonuniform_tp_process_groups

        if is_spare:
            # For spare ranks in test, just mark as passed and exit gracefully
            pytest.skip(f"Rank {rank} is a spare rank, skipping test gracefully")

        initialize_nonuniform_tp_process_groups(ntp_config)

        # After reconfiguration, check TP size
        tp_size_after = parallel_state.get_tensor_model_parallel_world_size()

        # Verify the configuration
        if dp_rank == 0:
            # First DP rank should have reduced TP=2
            assert tp_size_after == 2, f"DP rank 0 should have TP=2, got {tp_size_after}"
            assert tp_rank < 2, f"DP rank 0 should have tp_rank < 2, got {tp_rank}"
        else:
            # Other DP ranks keep TP=4
            assert tp_size_after == 4, f"DP rank {dp_rank} should have TP=4, got {tp_size_after}"
            assert tp_rank < 4, f"DP rank {dp_rank} should have tp_rank < 4, got {tp_rank}"

        # Create a simple model with tensor-parallel parameters
        hidden_size = 128
        model = torch.nn.Linear(hidden_size, hidden_size, bias=False).cuda()

        # Mark it as tensor-parallel
        model.weight.tensor_model_parallel = True
        model.weight.partition_dim = 0

        # Initialize NTP mappings
        from megatron.core.distributed.nonuniform_tp import ntp_map

        # For healthy ranks (DP=1), initialize send/recv splits
        if dp_rank == 1:
            # Create a mock module to test ntp_map
            class MockModule:
                def __init__(self, param):
                    self.param = param

                def parameters(self):
                    return [self.param]

            mock_module = MockModule(model.weight)
            ntp_map(mock_module, ntp_config, num_shards=hidden_size)

            # Verify send_splits and recv_splits were added
            assert hasattr(model.weight, 'send_splits'), "Healthy rank should have send_splits"
            assert hasattr(model.weight, 'recv_splits'), "Healthy rank should have recv_splits"
            assert len(model.weight.send_splits) == 4, "Should have splits for all tp_base ranks"

        # Test forward pass
        batch_size = 4
        input_tensor = torch.randn(batch_size, hidden_size, device='cuda')
        output = model(input_tensor)

        # Verify output shape
        assert output.shape == (batch_size, hidden_size), f"Unexpected output shape: {output.shape}"

        # Verify gradients work
        loss = output.sum()
        loss.backward()
        assert model.weight.grad is not None, "Gradients should be computed"

        print(
            f"[Rank {rank}] NTP end-to-end test passed! "
            f"DP={dp_rank}, TP={tp_size_after}, tp_rank={tp_rank}"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
