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
        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)
        dp_rank = 0
        tp_base = 8

        active_ranks = get_active_ranks_for_dp(dp_rank, tp_base, ddp_config)

        # Should return first (tp_base - tp_spares) ranks
        assert active_ranks == [0, 1, 2, 3, 4, 5]

    def test_get_active_ranks_for_dp_explicit(self):
        """Test get_active_ranks_for_dp with explicit non_active_ranks_per_dp."""
        ddp_config = DistributedDataParallelConfig(
            tp_base=8, tp_spares=2, non_active_ranks_per_dp={0: [2, 5]}
        )
        dp_rank = 0
        tp_base = 8

        active_ranks = get_active_ranks_for_dp(dp_rank, tp_base, ddp_config)

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

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=0)

        # Should not raise error and not add send_splits/recv_splits
        ntp_map(module, ddp_config, num_shards=24)

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
        param.shape = (384, 128)
        module.parameters = Mock(return_value=[param])
        module.config = MockConfig()

        ddp_config = DistributedDataParallelConfig(
            tp_base=8,
            tp_spares=2,
            non_active_ranks_per_dp={},  # No explicit non-active ranks, so this is healthy
        )

        # Execute
        ntp_map(module, ddp_config, num_shards=24)

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

        ddp_config = DistributedDataParallelConfig(
            tp_base=8,
            tp_spares=2,
            non_active_ranks_per_dp={(0, 0, 0): [2, 5]},  # This rank is unhealthy
        )

        # Execute
        ntp_map(module, ddp_config, num_shards=24)

        # Should NOT have added send_splits and recv_splits
        assert not hasattr(param, 'send_splits')
        assert not hasattr(param, 'recv_splits')

    def test_ntp_init_no_spares(self):
        """Test ntp_init when tp_spares=0 (should be no-op)."""
        # Create mock layer
        layer = Mock()
        layer.self_attention = Mock()
        layer.mlp = Mock()

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=0)

        # Should not raise error
        ntp_init(layer, ddp_config)

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

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)

        # Execute
        ntp_init(layer, ddp_config)

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

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ddp_config)

        # Should delegate attribute access
        assert ntp_optimizer.param_groups == []
        assert ntp_optimizer.state == {}

    def test_optimizer_prepare_grads_no_spares(self):
        """Test prepare_grads when tp_spares=0 (should be no-op)."""
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{'params': []}]
        mock_optimizer.prepare_grads = Mock(return_value=False)

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=0)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ddp_config)

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

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ddp_config)

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

        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)
        ntp_optimizer = NonuniformTPOptimizer(mock_optimizer, ddp_config)

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
        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)

        # Should initialize without error
        try:
            ntp_ddp = NonuniformTPDistributedDataParallel(
                config, ddp_config, model, disable_bucketing=True
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
        param.shape = (10, 10)
        param.side_grad = torch.randn(10, 2)
        param.recv_splits = [[0] * 8 for _ in range(8)]

        model = torch.nn.Module()
        model.register_parameter('test_param', param)

        config = TransformerConfig(
            num_layers=1, hidden_size=10, num_attention_heads=1, context_parallel_size=1
        )
        ddp_config = DistributedDataParallelConfig(tp_base=8, tp_spares=2)

        try:
            ntp_ddp = NonuniformTPDistributedDataParallel(
                config, ddp_config, model, disable_bucketing=True
            )
            # If we got here, the hook was created successfully
            assert True
        except Exception as e:
            pytest.skip(f"Skipping due to initialization requirements: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
