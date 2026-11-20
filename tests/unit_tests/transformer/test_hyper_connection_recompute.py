# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection block-level recomputation.

Tests the following functionality:
1. HyperConnectionModule._forward_with_checkpoint correctness
2. HyperConnectionModule.apply_h_post with CheckpointWithoutOutputManager
3. Multiple HyperConnectionModules chained with a single CheckpointWithoutOutputManager
4. Partial checkpoint (last layer not checkpointed)
5. TransformerConfig 'mhc' in recompute_modules option
"""

from collections.abc import Callable, Iterable

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutputManager,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _parameter_gradients(module):
    """Snapshot every original parameter, rejecting missing gradient paths."""
    parameters = dict(module.named_parameters())
    assert parameters
    gradients = {}
    for name, parameter in parameters.items():
        assert parameter.grad is not None, f"Missing gradient for {name}"
        gradients[name] = parameter.grad.detach().clone()
    return gradients


def _assert_parameter_gradients_close(module, expected, **kwargs):
    torch.testing.assert_close(_parameter_gradients(module), expected, **kwargs)


def _weighted_loss(outputs, weights):
    # Uniform sums can hide changes to the doubly stochastic residual mixing map.
    return sum((output * weight).sum() for output, weight in zip(outputs, weights, strict=True))


def _discard_and_check_outputs(
    manager: CheckpointWithoutOutputManager,
    boundary: torch.Tensor,
    *,
    live_tensors: Iterable[torch.Tensor] = (),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Verify the discard and return a hook that checks the rebuilt values during backward.

    The manager installs its rebuild hook first, so the caller's check sees the restored outputs.
    """
    outputs = [
        output.detach()
        for checkpoint in manager.checkpoints
        for output in checkpoint.outputs
        if isinstance(output, torch.Tensor)
    ]
    assert outputs, "The real mHC path must register replay work"
    assert all(output.untyped_storage().size() > 0 for output in outputs)
    expected = [output.clone() for output in outputs]
    live = [tensor.detach() for tensor in (boundary, *live_tensors)]

    manager.discard_all_outputs_and_register_unified_recompute(boundary)

    assert all(output.untyped_storage().size() == 0 for output in outputs)
    assert all(tensor.untyped_storage().size() > 0 for tensor in live)

    def check_reconstruction(grad: torch.Tensor) -> torch.Tensor:
        assert all(output.untyped_storage().size() > 0 for output in outputs)
        assert all(tensor.untyped_storage().size() > 0 for tensor in live)
        for output, original in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output, original)
        return grad

    return check_reconstruction


class TestHyperConnectionModuleCheckpoint:
    """Test HyperConnectionModule checkpoint functionality."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _create_hyper_connection_module(self, hidden_size=64, mhc_num_residual_streams=4):
        """Create a HyperConnectionModule for testing."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=mhc_num_residual_streams,
            mhc_sinkhorn_iterations=5,  # Fewer iterations for faster tests
            mhc_init_gating_factor=0.01,
        )
        module = HyperConnectionModule(config=config, layer_number=1)
        module.cuda()
        return module

    def test_apply_h_res_uses_h_res_transpose(self):
        """apply_h_res should compute H_res.T @ residual."""
        module = self._create_hyper_connection_module(hidden_size=4, mhc_num_residual_streams=2)
        h_res = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cuda')
        residual = torch.tensor([[[10.0, 100.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0]]], device='cuda')
        expected = torch.tensor(
            [[[13.0, 106.0, 18.0, 22.0, 24.0, 208.0, 26.0, 32.0]]], device='cuda'
        )

        mixed = module.apply_h_res(h_res, residual)

        torch.testing.assert_close(mixed, expected, atol=0.0, rtol=0.0)

    def test_forward_preserves_three_tuple_api_and_hybrid_can_request_residual(self):
        module = self._create_hyper_connection_module(hidden_size=8, mhc_num_residual_streams=2)
        hidden_states = torch.randn(4, 1, 16, device='cuda', requires_grad=True)

        compatible_output = module(hidden_states)
        hybrid_output = module(hidden_states, return_residual=True)

        assert len(compatible_output) == 3
        assert len(hybrid_output) == 4
        for compatible, hybrid in zip(compatible_output, hybrid_output[:3]):
            torch.testing.assert_close(compatible, hybrid)
        assert hybrid_output[3].shape == hidden_states.shape

    def test_forward_normal_vs_checkpoint_correctness(self):
        """Real aggregate replay preserves values and every input/parameter gradient."""
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
        aggregated_ref, h_res_ref, h_post_ref, residual_ref = module._forward_normal(hidden_states)
        mixed_ref = module.apply_h_res(h_res_ref, residual)
        weights = [torch.randn_like(output) for output in (aggregated_ref, mixed_ref, h_post_ref)]
        _weighted_loss((aggregated_ref, mixed_ref, h_post_ref), weights).backward()
        grad_hidden_ref = hidden_states.grad.clone()
        grad_residual_ref = residual.grad.clone()
        parameter_grads_ref = _parameter_gradients(module)
        module.zero_grad(set_to_none=True)

        # Forward with checkpoint
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt, residual_ckpt_out = (
            module._forward_with_checkpoint(hidden_states_ckpt, manager)
        )
        mixed_ckpt = module.apply_h_res(h_res_ckpt, residual_ckpt)
        torch.testing.assert_close(
            (aggregated_ckpt, h_res_ckpt, h_post_ckpt, residual_ckpt_out, mixed_ckpt),
            (aggregated_ref, h_res_ref, h_post_ref, residual_ref, mixed_ref),
        )
        loss_ckpt = _weighted_loss((aggregated_ckpt, mixed_ckpt, h_post_ckpt), weights)
        check_reconstruction = _discard_and_check_outputs(
            manager,
            loss_ckpt,
            live_tensors=(
                hidden_states_ckpt,
                residual_ckpt,
                residual_ckpt_out,
                h_res_ckpt,
                h_post_ckpt,
            ),
        )
        loss_ckpt.register_hook(check_reconstruction)
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
        _assert_parameter_gradients_close(module, parameter_grads_ref, atol=1e-5, rtol=1e-5)

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
        manager = CheckpointWithoutOutputManager()
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
        """The public module call must enter the real checkpoint path when given a manager."""
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
        aggregated_ref, h_res_ref, h_post_ref = module(hidden_states, mhc_recompute_manager=None)
        weights = [torch.randn_like(output) for output in (aggregated_ref, h_res_ref, h_post_ref)]
        _weighted_loss((aggregated_ref, h_res_ref, h_post_ref), weights).backward()
        grad_hidden_ref = hidden_states.grad.clone()
        parameter_grads_ref = _parameter_gradients(module)
        module.zero_grad(set_to_none=True)

        # With manager (uses _forward_with_checkpoint)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )
        torch.testing.assert_close(
            (aggregated_ckpt, h_res_ckpt, h_post_ckpt), (aggregated_ref, h_res_ref, h_post_ref)
        )
        loss_ckpt = _weighted_loss((aggregated_ckpt, h_res_ckpt, h_post_ckpt), weights)

        check_reconstruction = _discard_and_check_outputs(
            manager, loss_ckpt, live_tensors=(hidden_states_ckpt, h_res_ckpt, h_post_ckpt)
        )
        loss_ckpt.register_hook(check_reconstruction)
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)
        _assert_parameter_gradients_close(module, parameter_grads_ref, atol=1e-5, rtol=1e-5)


class TestMHCBlockRecomputeIntegration:
    """Test CheckpointWithoutOutputManager integration with HyperConnection."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_multiple_hyper_connections_in_chain(self):
        """
        Test that multiple HyperConnectionModules can be chained together
        with a single CheckpointWithoutOutputManager.
        """
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2
        n_channels = num_streams * hidden_size

        # Compose real mHC modules, including checkpointed nonterminal BDA operations.
        config = TransformerConfig(
            num_layers=4,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=num_streams,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
        )

        modules = torch.nn.ModuleList(
            HyperConnectionModule(config=config, layer_number=i + 1).cuda() for i in range(3)
        )

        def forward(hidden_states, residual, manager=None):
            mappings = []
            for index, module in enumerate(modules):
                aggregated, h_res, h_post = module(hidden_states, mhc_recompute_manager=manager)
                mappings.extend((h_res, h_post))
                hidden_states = module.fused_h_res_h_post_bda(
                    h_res,
                    residual,
                    h_post,
                    (0.1 * aggregated, None),
                    dropout_prob=0.0,
                    training=True,
                    fused=False,
                    manager=manager if index < len(modules) - 1 else None,
                )
                residual = hidden_states
            return hidden_states, mappings

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

        output_ref, mappings_ref = forward(hidden_states_ref, residual_ref)
        weight = torch.randn_like(output_ref)
        _weighted_loss((output_ref,), (weight,)).backward()
        grad_hidden_ref = hidden_states_ref.grad.clone()
        grad_residual_ref = residual_ref.grad.clone()
        parameter_grads_ref = _parameter_gradients(modules)
        modules.zero_grad(set_to_none=True)

        # With checkpoint using single manager
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        manager = CheckpointWithoutOutputManager()

        output_ckpt, mappings_ckpt = forward(hidden_states_ckpt, residual_ckpt, manager)
        torch.testing.assert_close(output_ckpt, output_ref)
        torch.testing.assert_close(mappings_ckpt, mappings_ref)
        check_reconstruction = _discard_and_check_outputs(
            manager, output_ckpt, live_tensors=(hidden_states_ckpt, residual_ckpt, *mappings_ckpt)
        )
        output_ckpt.register_hook(check_reconstruction)
        # The terminal BDA output must remain usable after earlier outputs are discarded.
        _weighted_loss((output_ckpt,), (weight,)).backward()

        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()
        grad_residual_ckpt = residual_ckpt.grad.clone()

        # Verify gradients
        assert torch.allclose(
            grad_hidden_ckpt, grad_hidden_ref, atol=1e-4
        ), f"Chained HyperConnection hidden gradients mismatch"
        assert torch.allclose(
            grad_residual_ckpt, grad_residual_ref, atol=1e-4
        ), f"Chained HyperConnection residual gradients mismatch"
        _assert_parameter_gradients_close(modules, parameter_grads_ref, atol=1e-4, rtol=1e-5)

    def test_partial_checkpoint_last_layer_not_checkpointed(self):
        """The real terminal BDA stays live while its aggregate input is replayed."""
        hidden_size = 64
        num_streams = 4
        seq_len = 8
        batch_size = 2

        config = TransformerConfig(
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=num_streams,
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
        aggregated_ref, h_res_ref, h_post_ref = module(
            hidden_states_ref, mhc_recompute_manager=None
        )
        output_ref = module.fused_h_res_h_post_bda(
            h_res_ref,
            residual_ref,
            h_post_ref,
            (0.1 * aggregated_ref, None),
            dropout_prob=0.0,
            training=True,
            fused=False,
        )
        weight = torch.randn_like(output_ref)
        _weighted_loss((output_ref,), (weight,)).backward()
        grad_hidden_ref = hidden_states_ref.grad.clone()
        grad_residual_ref = residual_ref.grad.clone()
        parameter_grads_ref = _parameter_gradients(module)
        module.zero_grad(set_to_none=True)

        # With manager - checkpoint everything except final output
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointWithoutOutputManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt = module(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )
        output_ckpt = module.fused_h_res_h_post_bda(
            h_res_ckpt,
            residual_ckpt,
            h_post_ckpt,
            (0.1 * aggregated_ckpt, None),
            dropout_prob=0.0,
            training=True,
            fused=False,
        )
        torch.testing.assert_close(
            (aggregated_ckpt, h_res_ckpt, h_post_ckpt, output_ckpt),
            (aggregated_ref, h_res_ref, h_post_ref, output_ref),
        )
        check_reconstruction = _discard_and_check_outputs(
            manager,
            output_ckpt,
            live_tensors=(hidden_states_ckpt, residual_ckpt, h_res_ckpt, h_post_ckpt),
        )
        output_ckpt.register_hook(check_reconstruction)
        _weighted_loss((output_ckpt,), (weight,)).backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)
        torch.testing.assert_close(residual_ckpt.grad, grad_residual_ref, atol=1e-5, rtol=1e-5)
        _assert_parameter_gradients_close(module, parameter_grads_ref, atol=1e-5, rtol=1e-5)


class TestTransformerConfigRecomputeMhc:
    """Test 'mhc' in recompute_modules configuration."""

    def test_config_default_value(self):
        """Test that 'mhc' is not in recompute_modules by default."""
        config = TransformerConfig(num_layers=2, hidden_size=64, num_attention_heads=4)
        assert "mhc" not in config.recompute_modules

    def test_config_enable_mhc_recompute(self):
        """Test enabling 'mhc' in recompute_modules."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_mhc_connections=True,
            mhc_num_residual_streams=4,
            recompute_modules=["core_attn", "mhc"],
            recompute_granularity='selective',
        )
        assert "mhc" in config.recompute_modules
        assert config.enable_mhc_connections is True

    @pytest.mark.parametrize("vp_size", [None, 2])
    def test_config_accepts_pipeline_parallel(self, vp_size):
        """Both schedules size their P2P buffers for the multi-stream residual."""
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            enable_mhc_connections=True,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=vp_size,
            pipeline_dtype=torch.bfloat16,
        )
        assert config.pipeline_model_parallel_size == 2

    def test_config_rejects_fp32_residual_connection(self):
        """The mHC residual is the n-stream tensor fed to the H_res bmm."""
        with pytest.raises(NotImplementedError, match="fp32_residual_connection"):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                enable_mhc_connections=True,
                fp32_residual_connection=True,
            )

    @pytest.mark.parametrize(
        "extra_kwargs, error_type, match",
        [
            ({"inference_fuse_tp_communication": True}, NotImplementedError, "single-stream"),
            ({"mhc_sinkhorn_iterations": 0}, ValueError, "mhc_sinkhorn_iterations"),
            ({"mhc_init_gating_factor": -0.1}, ValueError, "mhc_init_gating_factor"),
        ],
    )
    def test_config_rejects_unsupported_combinations(self, extra_kwargs, error_type, match):
        """Unsupported mHC combinations must fail at config time, not mid-forward."""
        with pytest.raises(error_type, match=match):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                enable_mhc_connections=True,
                **extra_kwargs,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
