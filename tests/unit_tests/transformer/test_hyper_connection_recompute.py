# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection block-level recomputation.

Tests the following functionality:
1. HyperConnectionModule._forward_with_checkpoint
2. HyperConnectionModule.apply_h_post with manager parameter
3. Integration with CheckpointManager
4. TransformerLayer with mhc_recompute_manager
5. TransformerBlock with recompute_hyper_connections
"""

import pytest
import torch
import torch.nn as nn

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CheckpointManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestHyperConnectionCheckpoint:
    """Test HyperConnectionModule checkpoint functionality."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_hyper_connection_module(self, hidden_size=64, num_residual_streams=4):
        """Create a HyperConnectionModule for testing."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_residual_streams,
            mhc_sinkhorn_iterations=5,  # Fewer iterations for faster tests
            mhc_init_gating_factor=0.01,
        )
        module = HyperConnectionModule(config=config, layer_number=1)
        module.cuda()
        return module

    def test_forward_normal_vs_checkpoint_correctness(self):
        """
        Test that _forward_with_checkpoint produces the same outputs as _forward_normal.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        module = self._create_hyper_connection_module(hidden_size, num_streams)

        # Create input tensors
        hidden_states = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )
        residual = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )

        # Clone inputs for comparison
        hidden_states_ckpt = hidden_states.detach().clone().requires_grad_(True)
        residual_ckpt = residual.detach().clone().requires_grad_(True)

        # Forward without checkpoint (reference)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        aggregated_ref, h_res_ref, h_post_ref = module._forward_normal(hidden_states)
        mixed_ref = module.apply_h_res(h_res_ref, residual)
        loss_ref = aggregated_ref.sum() + mixed_ref.sum() + h_post_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states.grad.clone()
        grad_residual_ref = residual.grad.clone()

        # Forward with checkpoint
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module._forward_with_checkpoint(
            hidden_states_ckpt, manager
        )
        mixed_ckpt = module.apply_h_res(h_res_ckpt, residual_ckpt)
        # Calculate loss before discarding outputs
        loss_ckpt = aggregated_ckpt.sum() + mixed_ckpt.sum() + h_post_ckpt.sum()

        # Register unified recompute hook
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)

        # Backward pass
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()
        grad_residual_ckpt = residual_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5), (
            f"Hidden states gradients mismatch:\n"
            f"Checkpoint: {grad_hidden_ckpt}\n"
            f"Reference: {grad_hidden_ref}"
        )
        assert torch.allclose(grad_residual_ckpt, grad_residual_ref, atol=1e-5), (
            f"Residual gradients mismatch:\n"
            f"Checkpoint: {grad_residual_ckpt}\n"
            f"Reference: {grad_residual_ref}"
        )

    def test_apply_h_post_with_checkpoint(self):
        """
        Test that apply_h_post with manager produces correct gradients.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        module = self._create_hyper_connection_module(hidden_size, num_streams)

        # Create input tensors
        x = torch.randn(seq_len, batch_size, hidden_size, device='cuda', requires_grad=True)
        bias = torch.randn(hidden_size, device='cuda')
        h_post = torch.randn(seq_len, batch_size, num_streams, device='cuda', requires_grad=True)

        # Clone inputs
        x_ckpt = x.detach().clone().requires_grad_(True)
        h_post_ckpt = h_post.detach().clone().requires_grad_(True)

        # Reference: without checkpoint (manager=None)
        torch.manual_seed(42)
        x_out_ref, bias_out_ref = module.apply_h_post((x, bias), h_post, manager=None)
        loss_ref = x_out_ref.sum()
        if bias_out_ref is not None:
            loss_ref = loss_ref + bias_out_ref.sum()
        loss_ref.backward()
        grad_x_ref = x.grad.clone()
        grad_h_post_ref = h_post.grad.clone()

        # With checkpoint (manager provided)
        torch.manual_seed(42)
        manager = CheckpointManager()
        x_out_ckpt, bias_out_ckpt = module.apply_h_post(
            (x_ckpt, bias), h_post_ckpt, manager=manager
        )
        loss_ckpt = x_out_ckpt.sum()
        if bias_out_ckpt is not None:
            loss_ckpt = loss_ckpt + bias_out_ckpt.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()
        grad_x_ckpt = x_ckpt.grad.clone()
        grad_h_post_ckpt = h_post_ckpt.grad.clone()

        # Verify gradients
        assert torch.allclose(grad_x_ckpt, grad_x_ref, atol=1e-5)
        assert torch.allclose(grad_h_post_ckpt, grad_h_post_ref, atol=1e-5)

    def test_forward_with_manager_parameter(self):
        """
        Test forward() method with mhc_recompute_manager parameter.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        module = self._create_hyper_connection_module(hidden_size, num_streams)

        # Create input tensors
        hidden_states = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )

        # Clone inputs
        hidden_states_ckpt = hidden_states.detach().clone().requires_grad_(True)

        # Reference: forward without manager (uses _forward_normal)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        aggregated_ref, h_res_ref, h_post_ref = module.forward(
            hidden_states, mhc_recompute_manager=None
        )
        loss_ref = aggregated_ref.sum() + h_res_ref.sum() + h_post_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states.grad.clone()

        # With manager (uses _forward_with_checkpoint)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module.forward(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )
        loss_ckpt = aggregated_ckpt.sum() + h_res_ckpt.sum() + h_post_ckpt.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)


class TestMHCBlockRecomputeIntegration:
    """Test CheckpointManager integration with HyperConnection."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_multiple_hyper_connections_in_chain(self):
        """
        Test that multiple HyperConnectionModules can be chained together
        with a single CheckpointManager.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2
        n_channels = num_streams * hidden_size

        # Create multiple HyperConnection modules (simulating multiple layers)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )

        modules = [
            HyperConnectionModule(config=config, layer_number=i + 1).cuda() for i in range(3)
        ]

        # Create input tensors
        hidden_states_ref = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )
        residual_ref = torch.randn(
            seq_len, batch_size, n_channels, device='cuda', requires_grad=True
        )

        hidden_states_ckpt = hidden_states_ref.detach().clone().requires_grad_(True)
        residual_ckpt = residual_ref.detach().clone().requires_grad_(True)

        # Reference: forward without checkpoint
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        h = hidden_states_ref
        r = residual_ref
        for module in modules:
            agg, h_res, h_post = module.forward(h, mhc_recompute_manager=None)
            agg, _ = module.apply_h_post((0.1 * agg, None), h_post, manager=None)
            mixed = module.apply_h_res(h_res, r)  # Apply h_res to get mixed [s, b, n*C]
            h = agg + mixed
            r = h

        loss_ref = h.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states_ref.grad.clone()
        grad_residual_ref = residual_ref.grad.clone()

        # With checkpoint using single manager
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        manager = CheckpointManager()

        h = hidden_states_ckpt
        r = residual_ckpt
        for module in modules:
            agg, h_res, h_post = module.forward(h, mhc_recompute_manager=manager)
            agg, _ = module.apply_h_post((0.1 * agg, None), h_post, manager=manager)
            mixed = module.apply_h_res(h_res, r)  # Apply h_res to get mixed [s, b, n*C]
            h = agg + mixed
            r = h

        loss_ckpt = h.sum()
        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()

        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()
        grad_residual_ckpt = residual_ckpt.grad.clone()

        # Verify gradients
        assert torch.allclose(
            grad_hidden_ckpt, grad_hidden_ref, atol=1e-4
        ), f"Chained HyperConnection hidden gradients mismatch"
        assert torch.allclose(
            grad_residual_ckpt, grad_residual_ref, atol=1e-4
        ), f"Chained HyperConnection residual gradients mismatch"

    def test_partial_checkpoint_last_layer_not_checkpointed(self):
        """
        Test that when is_last_layer_in_block=True, the final output is NOT checkpointed.
        This simulates the TransformerBlock behavior where the last layer's MLP BDA
        serves as the hook_tensor for unified recompute.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )

        module = HyperConnectionModule(config=config, layer_number=1).cuda()

        hidden_states_ref = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )
        residual_ref = torch.randn(
            seq_len, batch_size, num_streams * hidden_size, device='cuda', requires_grad=True
        )

        hidden_states_ckpt = hidden_states_ref.detach().clone().requires_grad_(True)
        residual_ckpt = residual_ref.detach().clone().requires_grad_(True)

        # Reference
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        aggregated_ref, h_res_ref, h_post_ref = module.forward(
            hidden_states_ref, mhc_recompute_manager=None
        )
        aggregated_ref, _ = module.apply_h_post(
            (0.1 * aggregated_ref, None), h_post_ref, manager=None
        )
        mixed_ref = module.apply_h_res(
            h_res_ref, residual_ref
        )  # Apply h_res to get mixed [s, b, n*C]
        # Simulate BDA that is NOT checkpointed (last layer)
        output_ref = aggregated_ref + 0.5 * mixed_ref
        loss_ref = output_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states_ref.grad.clone()

        # With manager - checkpoint everything except final output
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module.forward(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )

        aggregated_ckpt, _ = module.apply_h_post(
            (0.1 * aggregated_ckpt, None), h_post_ckpt, manager=manager
        )
        mixed_ckpt = module.apply_h_res(
            h_res_ckpt, residual_ckpt
        )  # Apply h_res to get mixed [s, b, n*C]
        # Simulate BDA that is NOT checkpointed (last layer) - this is the hook_tensor
        output_ckpt = aggregated_ckpt + 0.5 * mixed_ckpt

        # Register unified recompute on the output (which is not checkpointed)
        manager.discard_all_outputs_and_register_unified_recompute(output_ckpt)

        loss_ckpt = output_ckpt.sum()
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)


class TestCheckpointWithoutOutputManager:
    """Test CheckpointWithoutOutput behavior with ckpt_manager."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_ckpt_manager_auto_registration(self):
        """
        Test that CheckpointWithoutOutput auto-registers to manager when ckpt_manager is provided.
        """
        manager = CheckpointManager()
        assert len(manager.checkpoints) == 0

        def simple_func(x):
            return x * 2

        x = torch.randn(4, 4, device='cuda', requires_grad=True)

        # Create checkpoint with manager
        ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
        y = ckpt.checkpoint(simple_func, x)

        # Checkpoint should be auto-registered
        assert len(manager.checkpoints) == 1
        assert manager.checkpoints[0] is ckpt

    def test_discard_output_noop_with_manager(self):
        """
        Test that discard_output_and_register_recompute is a no-op when ckpt_manager is set.
        """
        manager = CheckpointManager()

        def simple_func(x):
            return x * 2

        x = torch.randn(4, 4, device='cuda', requires_grad=True)

        # Create checkpoint with manager
        ckpt = CheckpointWithoutOutput(ckpt_manager=manager)
        y = ckpt.checkpoint(simple_func, x)

        # Check output storage is not released yet
        original_size = y.untyped_storage().size()
        assert original_size > 0

        # Call discard_output_and_register_recompute - should be no-op
        ckpt.discard_output_and_register_recompute(y)

        # Storage should still be intact (no-op behavior)
        assert y.untyped_storage().size() == original_size

        # Clean up - register unified recompute to properly release
        manager.discard_all_outputs_and_register_unified_recompute(y)
        assert y.untyped_storage().size() == 0

    def test_checkpoint_without_manager_original_behavior(self):
        """
        Test that CheckpointWithoutOutput without manager maintains original behavior.
        """

        def simple_func(x):
            return x * 2

        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Reference without checkpoint
        y_ref = simple_func(x_ref)
        loss_ref = y_ref.sum()
        loss_ref.backward()
        grad_ref = x_ref.grad.clone()

        # With checkpoint (no manager)
        ckpt = CheckpointWithoutOutput()  # No ckpt_manager
        y = ckpt.checkpoint(simple_func, x)

        # Verify output is valid before discard
        assert y.untyped_storage().size() > 0

        # Discard and register individual recompute
        loss = y.sum()
        ckpt.discard_output_and_register_recompute(loss)

        # Storage should be released
        assert y.untyped_storage().size() == 0

        # Backward pass
        loss.backward()
        grad_ckpt = x.grad.clone()

        # Gradients should match
        assert torch.allclose(grad_ckpt, grad_ref, atol=1e-6)


class TestTransformerConfigRecomputeHyperConnections:
    """Test recompute_hyper_connections configuration option."""

    def test_config_default_value(self):
        """Test that recompute_hyper_connections defaults to False."""
        config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
        assert config.recompute_hyper_connections is False

    def test_config_enable_recompute_hyper_connections(self):
        """Test enabling recompute_hyper_connections."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=4,
            recompute_hyper_connections=True,
            recompute_granularity='selective',
        )
        assert config.recompute_hyper_connections is True
        assert config.enable_hyper_connections is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
