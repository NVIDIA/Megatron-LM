# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection block-level recomputation.

Tests the following functionality:
1. HyperConnectionModule._forward_with_checkpoint correctness
2. HyperConnectionModule.apply_h_post with CheckpointManager
3. Multiple HyperConnectionModules chained with a single CheckpointManager
4. Partial checkpoint (last layer not checkpointed)
5. TransformerConfig 'mhc' in recompute_modules option
"""

import types

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import (
    CheckpointManager,
    CheckpointWithoutOutput,
    CudaGraphCheckpointBridge,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import CudaGraphModule
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

    def test_apply_h_res_uses_h_res_transpose(self):
        """apply_h_res should compute H_res.T @ residual."""
        module = self._create_hyper_connection_module(hidden_size=4, num_residual_streams=2)
        h_res = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], device='cuda')
        residual = torch.tensor([[[10.0, 100.0, 3.0, 4.0, 1.0, 2.0, 5.0, 6.0]]], device='cuda')
        expected = torch.tensor(
            [[[13.0, 106.0, 18.0, 22.0, 24.0, 208.0, 26.0, 32.0]]], device='cuda'
        )

        mixed = module.apply_h_res(h_res, residual)

        torch.testing.assert_close(mixed, expected, atol=0.0, rtol=0.0)

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
        aggregated_ref, h_res_ref, h_post_ref, residual_ref = module._forward_normal(hidden_states)
        mixed_ref = module.apply_h_res(h_res_ref, residual)
        loss_ref = aggregated_ref.sum() + mixed_ref.sum() + h_post_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states.grad.clone()
        grad_residual_ref = residual.grad.clone()

        # Forward with checkpoint
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt, residual_ckpt_out = (
            module._forward_with_checkpoint(hidden_states_ckpt, manager)
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
        aggregated_ref, h_res_ref, h_post_ref, _ = module.forward(
            hidden_states, mhc_recompute_manager=None
        )
        loss_ref = aggregated_ref.sum() + h_res_ref.sum() + h_post_ref.sum()
        loss_ref.backward()
        grad_hidden_ref = hidden_states.grad.clone()

        # With manager (uses _forward_with_checkpoint)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        manager = CheckpointManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt, _ = module.forward(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )
        loss_ckpt = aggregated_ckpt.sum() + h_res_ckpt.sum() + h_post_ckpt.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_graph_bridge_rejects_distinct_views_of_shared_storage(self):
        """Discarding a logical view must never resize bridge-owned storage."""
        base = torch.empty(32, device="cuda")
        logical = base[:16]
        bridge_tensor = base[16:]
        assert logical.data_ptr() != bridge_tensor.data_ptr()

        bridge = CudaGraphCheckpointBridge(bridge_tensor)
        with pytest.raises(ValueError, match="different storage"):
            bridge.validate_logical_outputs((logical,))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_eager_recompute_materializes_captured_consumer_bridge(self):
        """Eager mHC recompute feeds one captured consumer backward.

        The logical checkpoint output and graph bridge have distinct storage.
        The consumer forward stages the original value into the bridge. Before
        captured backward replay, the checkpoint hook eagerly rebuilds the mHC
        aggregate and restores the bridge at the same address. One
        ``loss.backward()`` must propagate the consumer gradient through R.
        """
        from transformer_engine.pytorch.graph import make_graphed_callables

        hidden_size = 16
        num_streams = 4
        seq_len = 8
        batch_size = 2
        module = self._create_hyper_connection_module(hidden_size, num_streams)

        x_data = torch.randn(seq_len, batch_size, num_streams * hidden_size, device="cuda")
        h_pre_data = torch.randn(seq_len, batch_size, num_streams, device="cuda")

        # Negative control: logical recompute alone cannot repair the raw input
        # address captured by TE. Poisoning that surface must corrupt the
        # captured consumer's weight gradient when no bridge is attached.
        negative_surface = torch.empty(
            seq_len, batch_size, hidden_size, device="cuda", requires_grad=True
        )
        negative_consumer = torch.nn.Linear(hidden_size, hidden_size, bias=False, device="cuda")
        negative_consumer = make_graphed_callables(
            negative_consumer, (negative_surface,), num_warmup_iters=3
        )
        negative_consumer.zero_grad(set_to_none=True)
        negative_x = x_data.detach().clone().requires_grad_(True)
        negative_h_pre = h_pre_data.detach().clone().requires_grad_(True)
        negative_manager = CheckpointManager()
        negative_activation = CheckpointWithoutOutput(ckpt_manager=negative_manager).checkpoint(
            module.aggregate, negative_x, negative_h_pre
        )
        negative_output = negative_consumer(negative_activation)
        negative_manager.discard_all_outputs_and_register_unified_recompute(negative_output)
        with torch.no_grad():
            negative_surface.fill_(float("nan"))
        negative_output.square().mean().backward()
        assert not torch.isfinite(negative_consumer.weight.grad).all()

        # This sample argument becomes the exact static input surface captured
        # by TE. It is graph-owned and must retain its address across replays.
        bridge_tensor = torch.empty(
            seq_len, batch_size, hidden_size, device="cuda", requires_grad=True
        )
        bridge = CudaGraphCheckpointBridge(bridge_tensor)
        bridge_ptr = bridge_tensor.data_ptr()

        consumer = torch.nn.Linear(hidden_size, hidden_size, bias=False, device="cuda")
        consumer.train()
        consumer = make_graphed_callables(consumer, (bridge_tensor,), num_warmup_iters=3)
        consumer.zero_grad(set_to_none=True)

        reference_weight = consumer.weight.detach().clone().requires_grad_(True)
        reference_x = x_data.detach().clone().requires_grad_(True)
        reference_h_pre = h_pre_data.detach().clone().requires_grad_(True)
        reference_activation = module.aggregate(reference_x, reference_h_pre)
        reference_output = F.linear(reference_activation, reference_weight)
        reference_loss = reference_output.square().mean()
        reference_d_x, reference_d_h_pre, reference_d_weight = torch.autograd.grad(
            reference_loss, (reference_x, reference_h_pre, reference_weight)
        )

        runtime_x = x_data.detach().clone().requires_grad_(True)
        runtime_h_pre = h_pre_data.detach().clone().requires_grad_(True)
        manager = CheckpointManager()
        checkpoint = CheckpointWithoutOutput(ckpt_manager=manager, output_bridge=bridge)

        logical_activation = checkpoint.checkpoint(module.aggregate, runtime_x, runtime_h_pre)
        assert logical_activation.data_ptr() != bridge_ptr

        # TE stages the logical value into its static input surface before the
        # captured consumer forward replay.
        output = consumer(logical_activation)
        torch.testing.assert_close(bridge_tensor, logical_activation)

        manager.discard_all_outputs_and_register_unified_recompute(output)
        assert logical_activation.untyped_storage().nbytes() == 0

        # Make stale bridge use obvious. The checkpoint hook must restore B
        # from eager recompute before the captured consumer backward reads it.
        with torch.no_grad():
            bridge_tensor.fill_(float("nan"))

        loss = output.square().mean()
        loss.backward()

        assert bridge_tensor.data_ptr() == bridge_ptr
        assert torch.isfinite(bridge_tensor).all()
        torch.testing.assert_close(bridge_tensor, reference_activation.detach())
        torch.testing.assert_close(runtime_x.grad, reference_d_x, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(runtime_h_pre.grad, reference_d_h_pre, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(consumer.weight.grad, reference_d_weight, atol=1e-5, rtol=1e-5)


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
            agg, h_res, h_post, _ = module.forward(h, mhc_recompute_manager=None)
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
            agg, h_res, h_post, _ = module.forward(h, mhc_recompute_manager=manager)
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
        aggregated_ref, h_res_ref, h_post_ref, _ = module.forward(
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
        aggregated_ckpt, h_res_ckpt, h_post_ckpt, _ = module.forward(
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
            enable_hyper_connections=True,
            num_residual_streams=4,
            recompute_modules=["core_attn", "mhc"],
            recompute_granularity='selective',
        )
        assert "mhc" in config.recompute_modules
        assert config.enable_hyper_connections is True

    def test_config_accepts_initial_attention_only_te_graph_split(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=4,
            recompute_modules=["mhc"],
            recompute_granularity="selective",
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn],
        )
        assert config.cuda_graph_modules == [CudaGraphModule.attn]

    @pytest.mark.parametrize(
        ("cuda_graph_modules", "recompute_modules"),
        [
            ([], ["mhc"]),
            ([CudaGraphModule.mlp], ["mhc"]),
            ([CudaGraphModule.attn, CudaGraphModule.mlp], ["mhc"]),
            ([CudaGraphModule.attn], ["core_attn", "mhc"]),
        ],
    )
    def test_config_rejects_unimplemented_te_graph_splits(
        self, cuda_graph_modules, recompute_modules
    ):
        with pytest.raises(ValueError, match="initial attention-only split"):
            TransformerConfig(
                num_layers=2,
                hidden_size=64,
                num_attention_heads=4,
                enable_hyper_connections=True,
                num_residual_streams=4,
                recompute_modules=recompute_modules,
                recompute_granularity="selective",
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=cuda_graph_modules,
            )

    @staticmethod
    def _mhc_recompute_config_kwargs(**extra):
        base = dict(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=4,
            recompute_modules=["mhc"],
            recompute_granularity="selective",
        )
        base.update(extra)
        return base

    @pytest.mark.parametrize("modules", ["attn", ["attn"]])
    def test_config_accepts_string_module_forms_for_attention_split(self, modules):
        """The gate must compare cuda_graph_modules after string->enum normalization."""
        config = TransformerConfig(
            **self._mhc_recompute_config_kwargs(
                cuda_graph_impl="transformer_engine", cuda_graph_modules=modules
            )
        )
        assert config.cuda_graph_modules == [CudaGraphModule.attn]

    def test_config_rejects_deprecated_external_cuda_graph_with_mhc_recompute(self):
        """The legacy flag migrates to the TE impl and must reach the same gate."""
        with pytest.raises(ValueError, match="initial attention-only split"):
            TransformerConfig(**self._mhc_recompute_config_kwargs(external_cuda_graph=True))

    def test_config_rejects_deprecated_enable_cuda_graph_with_mhc_recompute(self):
        """The legacy flag migrates to the local impl, which has no split support."""
        with pytest.raises(ValueError, match="cuda_graph_impl='local'"):
            TransformerConfig(**self._mhc_recompute_config_kwargs(enable_cuda_graph=True))

    def test_config_rejects_local_impl_with_mhc_recompute(self):
        with pytest.raises(ValueError, match="cuda_graph_impl='local'"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    cuda_graph_impl="local", cuda_graph_modules=[CudaGraphModule.attn]
                )
            )

    def test_config_rejects_full_iteration_impl_with_mhc_recompute(self):
        with pytest.raises(ValueError, match="cuda_graph_impl='full_iteration'"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    cuda_graph_impl="full_iteration", cuda_graph_modules=[]
                )
            )

    def test_config_rejects_interleaved_pipeline_with_attention_split(self):
        with pytest.raises(ValueError, match="interleaved pipeline"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    num_layers=4,
                    cuda_graph_impl="transformer_engine",
                    cuda_graph_modules=[CudaGraphModule.attn],
                    pipeline_model_parallel_size=2,
                    virtual_pipeline_model_parallel_size=2,
                    pipeline_dtype=torch.bfloat16,
                )
            )

    def test_config_allows_vpp_with_mhc_recompute_without_cuda_graphs(self):
        """The VPP rejection is scoped to CUDA graphs; eager recompute + VPP stays legal."""
        config = TransformerConfig(
            **self._mhc_recompute_config_kwargs(
                num_layers=4,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                pipeline_dtype=torch.bfloat16,
            )
        )
        assert config.virtual_pipeline_model_parallel_size == 2

    def test_config_accepts_te_attention_graphs_without_mhc_recompute(self):
        """The gate is scoped to mHC recompute; plain TE attention graphs stay legal."""
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=4,
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=[CudaGraphModule.attn],
        )
        assert config.cuda_graph_modules == [CudaGraphModule.attn]

    def test_hybrid_mhc_layer_rejects_attention_split_at_construction(self):
        """HybridStack mHC layers capture the mHC producer and must fail closed."""
        from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer

        config = TransformerConfig(
            **self._mhc_recompute_config_kwargs(
                cuda_graph_impl="transformer_engine", cuda_graph_modules=[CudaGraphModule.attn]
            )
        )
        with pytest.raises(ValueError, match="HybridStack"):
            HyperConnectionHybridLayer(config, types.SimpleNamespace(layer_number=1))


class TestCudaGraphBridgeContracts:
    """Failure-path coverage for the fixed-address bridge safety rails."""

    def _cuda(self, *shape, dtype=torch.float32):
        return torch.randn(*shape, device="cuda").to(dtype)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bridge_constructor_rejects_invalid_inputs(self):
        with pytest.raises(ValueError, match="at least one tensor"):
            CudaGraphCheckpointBridge(())
        with pytest.raises(TypeError, match="Tensor or tuple"):
            CudaGraphCheckpointBridge([torch.empty(2, device="cuda")])
        with pytest.raises(TypeError, match="only supports Tensor"):
            CudaGraphCheckpointBridge((torch.empty(2, device="cuda"), "not-a-tensor"))
        with pytest.raises(ValueError, match="CUDA tensors"):
            CudaGraphCheckpointBridge(torch.empty(2))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bridge_validate_rejects_metadata_mismatches(self):
        bridge = CudaGraphCheckpointBridge(self._cuda(4, 8))
        with pytest.raises(ValueError, match="arity mismatch"):
            bridge.validate_logical_outputs((self._cuda(4, 8), self._cuda(4, 8)))
        with pytest.raises(ValueError, match="shape"):
            bridge.validate_logical_outputs((self._cuda(8, 4),))
        with pytest.raises(ValueError, match="dtype"):
            bridge.validate_logical_outputs((self._cuda(4, 8, dtype=torch.bfloat16),))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bridge_materialize_rejects_address_change(self):
        """The core safety rail: a bridge whose storage moved must refuse to copy."""
        bridge_tensor = self._cuda(4, 8)
        bridge = CudaGraphCheckpointBridge(bridge_tensor)
        bridge_tensor.data = torch.empty_like(bridge_tensor)
        with pytest.raises(RuntimeError, match="changed address"):
            bridge.materialize((self._cuda(4, 8),))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bridge_materialize_rejects_metadata_mismatches(self):
        bridge = CudaGraphCheckpointBridge(self._cuda(4, 8))
        with pytest.raises(ValueError, match="arity mismatch"):
            bridge.materialize((self._cuda(4, 8), self._cuda(4, 8)))
        with pytest.raises(ValueError, match="shape"):
            bridge.materialize((self._cuda(8, 4),))
        with pytest.raises(ValueError, match="metadata"):
            bridge.materialize((self._cuda(4, 8, dtype=torch.bfloat16),))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_tensor_bridge_materializes_every_output(self):
        first, second = self._cuda(4, 8), self._cuda(2, 3)
        bridge = CudaGraphCheckpointBridge((first, second))
        bridge.validate_logical_outputs((self._cuda(4, 8), self._cuda(2, 3)))

        new_first, new_second = self._cuda(4, 8), self._cuda(2, 3)
        bridge.materialize((new_first, new_second))
        assert torch.equal(first, new_first)
        assert torch.equal(second, new_second)


class TestCheckpointRngReplay:
    """Recompute must replay forward-time RNG (dropout masks) for every tracker kind.

    With a graph-safe tracker, generator handles share the live state, so the
    snapshot taken by ``CheckpointWithoutOutput`` must clone state contents;
    otherwise the recompute draws fresh offsets and reproduces a different
    dropout mask than the forward pass, silently corrupting gradients.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _seed(self, tracker_kind):
        if tracker_kind == "te":
            pytest.importorskip("transformer_engine")
            model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        elif tracker_kind == "graphsafe":
            model_parallel_cuda_manual_seed(123, use_cudagraphable_rng=True, force_reset_rng=True)
        else:
            model_parallel_cuda_manual_seed(123, force_reset_rng=True)

    def _roundtrip(self, run_function):
        x = torch.randn(4096, device="cuda", requires_grad=True)
        manager = CheckpointManager()
        checkpoint = CheckpointWithoutOutput(ckpt_manager=manager)
        output = checkpoint.checkpoint(run_function, x)
        forward_values = output.detach().clone()
        manager.discard_all_outputs()

        # Simulate other microbatches advancing the ambient RNG stream between
        # the forward pass and the backward-time recompute.
        torch.rand(8192, device="cuda")
        ambient_before = torch.cuda.get_rng_state()

        manager.recompute_now()

        assert torch.equal(
            output, forward_values
        ), "recompute produced a different dropout mask than the forward pass"
        assert torch.equal(
            ambient_before, torch.cuda.get_rng_state()
        ), "recompute leaked RNG stream advancement into the ambient state"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tracker_kind", ["plain", "graphsafe", "te"])
    def test_dropout_in_checkpoint_replays_forward_mask(self, tracker_kind):
        self._seed(tracker_kind)

        def run_function(value):
            return F.dropout(value * 3.0, p=0.5, training=True)

        self._roundtrip(run_function)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tracker_kind", ["plain", "graphsafe", "te"])
    def test_tracker_fork_in_checkpoint_replays_forward_mask(self, tracker_kind):
        self._seed(tracker_kind)

        def run_function(value):
            with get_cuda_rng_tracker().fork():
                return F.dropout(value * 3.0, p=0.5, training=True)

        self._roundtrip(run_function)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
