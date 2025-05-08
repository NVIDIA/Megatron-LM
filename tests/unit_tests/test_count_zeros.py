import unittest.mock as mock

from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig


class MockChainedOptimizer(ChainedOptimizer):
    """Mock ChainedOptimizer that bypasses config equality check."""
    def __init__(self, optimizers):
        # Skip the parent's __init__ to avoid the config equality assertion
        self.chained_optimizers = optimizers
        self.config = optimizers[0].config if optimizers else None
        self.model_chunks = []
        for optimizer in optimizers:
            for model_chunk in getattr(optimizer, "model_chunks", []):
                if model_chunk not in self.model_chunks:
                    self.model_chunks.append(model_chunk)


class TestCountZerosMock:
    """Test the count_zeros function with mocks instead of actual models or tensors."""

    def test_chained_count_zeros_single_optimizer(self):
        """Test basic functionality of count_zeros with mocked optimizer."""
        opt = mock.MagicMock()
        opt.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt.count_zeros.return_value = 42.0
        opt.model_chunks = []

        chained_opt = MockChainedOptimizer([opt])
        chained_opt.grads_states_parallel_group_is_shared = mock.MagicMock(return_value=False)

        result = chained_opt.count_zeros()

        opt.count_zeros.assert_called_once()
        assert result == opt.count_zeros.return_value

    def test_chained_count_zeros_multiple_optimizers(self):
        """Test count_zeros with multiple optimizers in the chain."""
        opt1 = mock.MagicMock()
        opt1.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt1.count_zeros.return_value = 30.0
        opt1.model_chunks = []

        opt2 = mock.MagicMock()
        opt2.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt2.count_zeros.return_value = 12.0
        opt2.model_chunks = []

        chained_opt = MockChainedOptimizer([opt1, opt2])
        chained_opt.grads_states_parallel_group_is_shared = mock.MagicMock(return_value=False)

        result = chained_opt.count_zeros()

        opt1.count_zeros.assert_called_once()
        opt2.count_zeros.assert_called_once()
        assert result == opt1.count_zeros.return_value + opt2.count_zeros.return_value

    def test_chained_count_zeros_with_shared_parallel_group(self):
        """Test count_zeros when parallel groups are shared."""
        param1 = mock.MagicMock()
        param2 = mock.MagicMock()

        opt1 = mock.MagicMock()
        opt1.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt1.get_parameters.return_value = [param1]
        opt1.model_chunks = []

        opt2 = mock.MagicMock()
        opt2.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt2.get_parameters.return_value = [param2]
        opt2.model_chunks = []

        pg = mock.MagicMock()
        opt1.get_grad_stats_parallel_group.return_value = pg
        opt2.get_grad_stats_parallel_group.return_value = pg

        chained_opt = MockChainedOptimizer([opt1, opt2])

        with mock.patch('megatron.core.optimizer.optimizer.count_zeros_fp32') as mock_count_zeros:
            mock_count_zeros.return_value = 3.0
            chained_opt.grads_states_parallel_group_is_shared = mock.MagicMock(return_value=True)

            result = chained_opt.count_zeros()

            mock_count_zeros.assert_called_once()
            args, kwargs = mock_count_zeros.call_args
            assert len(args[0]) == 2
            assert kwargs["grad_stats_parallel_group"] == pg
            assert result == mock_count_zeros.return_value

    def test_chained_with_count_zeros_one(self):
        """Test the fix that ensures count_zeros is only called when log_num_zeros_in_grad is True."""
        opt1 = mock.MagicMock()
        opt1.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt1.count_zeros.return_value = 10.0
        opt1.model_chunks = []

        opt2 = mock.MagicMock()
        opt2.config = OptimizerConfig(log_num_zeros_in_grad=False)
        opt2.count_zeros.return_value = 20.0  # Should never be called
        opt2.model_chunks = []

        chained_opt = MockChainedOptimizer([opt1, opt2])
        chained_opt.grads_states_parallel_group_is_shared = mock.MagicMock(return_value=False)

        result = chained_opt.count_zeros()

        opt1.count_zeros.assert_called_once()
        opt2.count_zeros.assert_not_called()
        assert result == opt1.count_zeros.return_value

    def test_chained_with_count_zeros_all(self):
        """Test when all optimizers have log_num_zeros_in_grad=True."""
        opt1 = mock.MagicMock()
        opt1.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt1.count_zeros.return_value = 15.0
        opt1.model_chunks = []

        opt2 = mock.MagicMock()
        opt2.config = OptimizerConfig(log_num_zeros_in_grad=True)
        opt2.count_zeros.return_value = 25.0
        opt2.model_chunks = []

        chained_opt = MockChainedOptimizer([opt1, opt2])
        chained_opt.grads_states_parallel_group_is_shared = mock.MagicMock(return_value=False)

        result = chained_opt.count_zeros()

        opt1.count_zeros.assert_called_once()
        opt2.count_zeros.assert_called_once()
        assert result == opt1.count_zeros.return_value + opt2.count_zeros.return_value

    def test_chained_with_count_zeros_none(self):
        """Test when all optimizers have log_num_zeros_in_grad=False."""
        opt1 = mock.MagicMock()
        opt1.config = OptimizerConfig(log_num_zeros_in_grad=False)
        opt1.count_zeros.return_value = 15.0  # Should never be called
        opt1.model_chunks = []

        opt2 = mock.MagicMock()
        opt2.config = OptimizerConfig(log_num_zeros_in_grad=False)
        opt2.count_zeros.return_value = 25.0  # Should never be called
        opt2.model_chunks = []

        chained_opt = MockChainedOptimizer([opt1, opt2])
        chained_opt.grads_states_parallel_group_is_shared = mock.MagicMock(return_value=False)

        result = chained_opt.count_zeros()

        opt1.count_zeros.assert_not_called()
        opt2.count_zeros.assert_not_called()
        assert result == 0
