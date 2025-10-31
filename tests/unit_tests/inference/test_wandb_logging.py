# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for wandb logging functionality in inference."""

from unittest.mock import MagicMock, Mock, create_autospec, patch

import pytest
import torch

from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


def set_rounder(value):
    """Utility function to set the DynamicInferenceContext rounder."""
    DynamicInferenceContext.ROUNDER = value  # For backwards compatibility
    DynamicInferenceContext.TOKEN_ROUNDER = value
    DynamicInferenceContext.REQUEST_ROUNDER = value


class TestInferenceWandbLogging:
    """Test suite for wandb logging in inference."""

    def setup_method(self):
        """Set up test fixtures."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        set_rounder(64)

    def teardown_method(self):
        """Clean up test fixtures."""
        set_rounder(64)
        Utils.destroy_model_parallel()

    def _get_dynamic_context(
        self,
        params_dtype=torch.float32,
        num_layers=4,
        kv_channels=8,
        num_attention_heads=2,
        max_sequence_length=512,
        buffer_size_gb=0.03,
        block_size_tokens=128,
        buffer_guaranteed_fraction=0.1,
        metrics_writer=None,
    ):
        """Helper to create a DynamicInferenceContext."""
        return DynamicInferenceContext(
            params_dtype=params_dtype,
            num_layers=num_layers,
            kv_channels=kv_channels,
            num_attention_heads=num_attention_heads,
            max_sequence_length=max_sequence_length,
            num_cuda_graphs=None,
            buffer_size_gb=buffer_size_gb,
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            block_size_tokens=block_size_tokens,
            metrics_writer=metrics_writer,
        )

    @pytest.mark.internal
    def test_get_kvcache_utilization_stats_with_requests(self):
        """Test get_kvcache_utilization_stats() with empty context and then with active requests."""
        dynamic_context = self._get_dynamic_context()

        # First, test with empty context
        stats = dynamic_context.get_kvcache_utilization_stats()

        # Verify all required fields are present
        assert 'total_blocks' in stats
        assert 'allocated_blocks' in stats
        assert 'active_unique_blocks' in stats
        assert 'allocated_utilization' in stats
        assert 'active_utilization' in stats
        assert 'active_request_count' in stats
        assert 'paused_request_count' in stats
        assert 'gtd_block_count' in stats
        assert 'block_count_avail' in stats
        assert 'num_non_gtd_blocks' in stats
        assert 'active_token_count' in stats
        assert 'total_request_count' in stats
        assert 'max_requests' in stats

        # Verify values for empty context
        assert stats['allocated_blocks'] == 0
        assert stats['active_unique_blocks'] == 0
        assert stats['allocated_utilization'] == 0.0
        assert stats['active_utilization'] == 0.0
        assert stats['active_request_count'] == 0
        assert stats['paused_request_count'] == 0
        assert stats['active_token_count'] == 0
        assert stats['total_request_count'] == 0

        # Now add a request and verify stats update correctly
        context_length = 144
        dynamic_context.add_request(
            DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=torch.arange(0, context_length, dtype=torch.long, device='cuda'),
                sampling_params=SamplingParams(
                    num_tokens_to_generate=dynamic_context.max_tokens - context_length
                ),
            )
        )

        # Initialize attention state to populate block table
        dynamic_context.initialize_attention_state()

        # Get stats after adding request
        stats_after = dynamic_context.get_kvcache_utilization_stats()

        # Verify that we have allocated blocks
        assert stats_after['allocated_blocks'] > 0
        assert stats_after['active_unique_blocks'] > 0
        assert stats_after['allocated_utilization'] > 0.0
        assert stats_after['active_utilization'] > 0.0

        # Verify request counts
        assert stats_after['active_request_count'] == 1
        assert stats_after['total_request_count'] == 1
        assert stats_after['active_token_count'] == context_length
        assert stats_after['paused_request_count'] == 0

        # Verify that total_blocks remains constant
        assert stats_after['total_blocks'] == stats['total_blocks']
        assert stats_after['total_blocks'] > 0

        # Verify that gtd_block_count remains constant
        assert stats_after['gtd_block_count'] == stats['gtd_block_count']

        # Verify that max_requests remains constant
        assert stats_after['max_requests'] == stats['max_requests']
        assert stats_after['max_requests'] > 0

        # Verify block availability decreased after allocation
        assert stats_after['block_count_avail'] < stats['block_count_avail']

        # Verify relationship: allocated_blocks + block_count_avail + 1 (dummy) = total
        assert (
            stats_after['allocated_blocks'] + stats_after['block_count_avail'] + 1
            == dynamic_context.block_allocator.block_count_total
        )

        # Verify utilization bounds [0, 1]
        assert 0.0 <= stats_after['allocated_utilization'] <= 1.0
        assert 0.0 <= stats_after['active_utilization'] <= 1.0

        # Verify relationship: active_utilization <= allocated_utilization
        # (active blocks are a subset of allocated blocks)
        assert stats_after['active_utilization'] <= stats_after['allocated_utilization']

        # Verify relationship: active_unique_blocks <= allocated_blocks
        assert stats_after['active_unique_blocks'] <= stats_after['allocated_blocks']

        # Calculate expected number of blocks needed for this request
        expected_blocks_needed = (
            context_length + dynamic_context.block_size_tokens - 1
        ) // dynamic_context.block_size_tokens
        assert stats_after['allocated_blocks'] == expected_blocks_needed

    @pytest.mark.internal
    def test_kvcache_utilization_stats_types(self):
        """Test that get_kvcache_utilization_stats() returns correct types."""
        dynamic_context = self._get_dynamic_context()
        stats = dynamic_context.get_kvcache_utilization_stats()

        # All integer fields
        int_fields = [
            'total_blocks',
            'allocated_blocks',
            'active_unique_blocks',
            'active_request_count',
            'paused_request_count',
            'gtd_block_count',
            'block_count_avail',
            'num_non_gtd_blocks',
            'active_token_count',
            'total_request_count',
            'max_requests',
        ]

        for field in int_fields:
            assert isinstance(
                stats[field], int
            ), f"{field} should be int but is {type(stats[field])}"

        # All float fields
        float_fields = ['allocated_utilization', 'active_utilization']
        for field in float_fields:
            assert isinstance(
                stats[field], float
            ), f"{field} should be float but is {type(stats[field])}"

    @pytest.mark.internal
    @patch('megatron.core.inference.engines.dynamic_engine.HAVE_WANDB', True)
    def test_engine_logging_step_interval_zero(self):
        """Test that no logging occurs when inference_logging_step_interval is 0."""
        mock_wandb = Mock()
        mock_wandb.__name__ = "wandb"
        mock_wandb.log = Mock()

        dynamic_context = self._get_dynamic_context(metrics_writer=mock_wandb)

        # Create mock controller with proper spec to pass isinstance checks
        mock_controller = create_autospec(TextGenerationController, instance=True)
        # Set up nested mock structure
        mock_controller.inference_wrapped_model = Mock()
        mock_controller.inference_wrapped_model.model = Mock()
        mock_controller.inference_wrapped_model.model.config = Mock()
        mock_controller.inference_wrapped_model.model.config.cuda_graph_impl = "none"

        engine = DynamicInferenceEngine(
            controller=mock_controller,
            context=dynamic_context,
            random_seed=123,
            inference_logging_step_interval=0,  # Disabled
        )

        # Verify log was never called
        mock_wandb.log.assert_not_called()

    @pytest.mark.internal
    def test_paused_requests_in_stats(self):
        """Test that paused requests are correctly reflected in stats."""
        set_rounder(1)
        dynamic_context = DynamicInferenceContext(
            params_dtype=torch.float32,
            num_layers=2,
            kv_channels=64,
            num_attention_heads=8,
            max_sequence_length=128,
            num_cuda_graphs=None,
            buffer_size_gb=0.01,  # Small buffer to force pausing
            buffer_guaranteed_fraction=0.1,
            block_size_tokens=32,
        )

        # Add multiple requests to potentially trigger pausing
        for i in range(5):
            dynamic_context.add_request(
                DynamicInferenceRequest(
                    request_id=i,
                    prompt_tokens=torch.zeros(10, device='cuda'),
                    sampling_params=SamplingParams(num_tokens_to_generate=10),
                )
            )

        if dynamic_context.total_request_count > 0:
            dynamic_context.initialize_attention_state()
            stats = dynamic_context.get_kvcache_utilization_stats()

            # Verify paused request count is included
            assert 'paused_request_count' in stats
            assert stats['paused_request_count'] >= 0

    @pytest.mark.internal
    def test_metrics_writer_none_handling(self):
        """Test that engine handles None metrics_writer gracefully."""
        dynamic_context = self._get_dynamic_context(metrics_writer=None)

        # Create mock controller with proper spec to pass isinstance checks
        mock_controller = create_autospec(TextGenerationController, instance=True)
        # Set up nested mock structure
        mock_controller.inference_wrapped_model = Mock()
        mock_controller.inference_wrapped_model.model = Mock()
        mock_controller.inference_wrapped_model.model.config = Mock()
        mock_controller.inference_wrapped_model.model.config.cuda_graph_impl = "none"

        # Should not raise error even with logging interval set
        engine = DynamicInferenceEngine(
            controller=mock_controller,
            context=dynamic_context,
            random_seed=123,
            inference_logging_step_interval=10,
        )

        # Verify engine was created successfully
        assert engine.inference_logging_step_interval == 10
        assert engine.context.metrics_writer is None
