# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection block-level recomputation.

Tests the following functionality:
1. HyperConnectionModule._forward_with_checkpoint correctness
2. HyperConnectionModule.apply_h_post with MHCCheckpointManager
3. Multiple HyperConnectionModules chained with a single MHCCheckpointManager
4. Partial checkpoint (last layer not checkpointed)
5. TransformerConfig 'mhc' in recompute_modules option
"""

import types

import pytest
import torch
import torch.nn.functional as F

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    MHCCheckpointManager,
    get_all_rng_states,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.mhc_recompute import uses_mhc_recompute_attn_cuda_graph_split
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
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

    def test_forward_supports_empty_sequence(self):
        """Forward and backward should support an empty sequence."""
        module = self._create_hyper_connection_module(hidden_size=4, num_residual_streams=2)
        hidden_states = torch.empty(0, 1, 8, device='cuda', requires_grad=True)

        aggregated, h_res, h_post, residual = module(hidden_states)

        assert aggregated.shape == (0, 1, 4)
        assert h_res.shape == (0, 1, 2, 2)
        assert h_post.shape == (0, 1, 2)
        assert residual.shape == (0, 1, 8)

        (aggregated.sum() + h_res.sum() + h_post.sum() + residual.sum()).backward()
        assert hidden_states.grad is not None
        assert module.mapping_proj.weight.grad is not None

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
        manager = MHCCheckpointManager()
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
        manager = MHCCheckpointManager()
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
        manager = MHCCheckpointManager()
        aggregated_ckpt, h_res_ckpt, h_post_ckpt, _ = module.forward(
            hidden_states_ckpt, mhc_recompute_manager=manager
        )
        loss_ckpt = aggregated_ckpt.sum() + h_res_ckpt.sum() + h_post_ckpt.sum()

        manager.discard_all_outputs_and_register_unified_recompute(loss_ckpt)
        loss_ckpt.backward()
        grad_hidden_ckpt = hidden_states_ckpt.grad.clone()

        # Verify gradients match
        assert torch.allclose(grad_hidden_ckpt, grad_hidden_ref, atol=1e-5)


class TestMHCBlockRecomputeIntegration:
    """Test MHCCheckpointManager integration with HyperConnection."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_multiple_hyper_connections_in_chain(self):
        """
        Test that multiple HyperConnectionModules can be chained together
        with a single MHCCheckpointManager.
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

        manager = MHCCheckpointManager()

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
        manager = MHCCheckpointManager()
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
    def test_config_accepts_other_shapes_without_the_split_switch(
        self, cuda_graph_modules, recompute_modules
    ):
        """Without the opt-in switch, no shape is rejected on the split's behalf.

        Extra graph scopes and extra recompute modules are orthogonal to the split
        and were running before it existed, so the default path must leave them
        alone. Only opting in narrows the configuration. The exact [attn]+[mhc]
        shape additionally warns that the captured producer's checkpoint no longer
        pays; the shapes here are broader than that, so this asserts the predicate
        rather than anything about warnings.
        """
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=4,
            recompute_modules=recompute_modules,
            recompute_granularity="selective",
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=cuda_graph_modules,
            # core_attn recompute under a graphed attention asserts on nonzero
            # dropout, which would fail these cases before they reach the gate.
            hidden_dropout=0.0,
            attention_dropout=0.0,
        )
        assert not uses_mhc_recompute_attn_cuda_graph_split(config)

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

    @staticmethod
    def _mhc_overlap_config_kwargs(**extra):
        base = dict(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            enable_hyper_connections=True,
            num_residual_streams=4,
            recompute_modules=["mhc"],
            recompute_granularity="selective",
            num_moe_experts=8,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=8,
            overlap_moe_expert_parallel_comm=True,
            add_bias_linear=False,
            bf16=True,
            pipeline_dtype=torch.bfloat16,
        )
        base.update(extra)
        return base

    def test_config_accepts_attention_split_with_ep_overlap(self):
        """mHC recompute + attn TE CUDA graph composes with EP a2a overlap."""
        if not is_torch_min_version("2.6.0"):
            pytest.skip("EP a2a overlap requires torch >= 2.6.0")
        with pytest.warns(UserWarning, match="capturing the whole attention range"):
            config = TransformerConfig(
                **self._mhc_overlap_config_kwargs(
                    cuda_graph_impl="transformer_engine", cuda_graph_modules=[CudaGraphModule.attn]
                )
            )
        assert config.cuda_graph_modules == [CudaGraphModule.attn]
        assert config.overlap_moe_expert_parallel_comm is True

    @pytest.mark.parametrize("modules", ["attn", ["attn"]])
    def test_config_accepts_string_module_forms_for_attention_split(self, modules):
        """The gate must compare cuda_graph_modules after string->enum normalization."""
        with pytest.warns(UserWarning, match="capturing the whole attention range"):
            config = TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    cuda_graph_impl="transformer_engine", cuda_graph_modules=modules
                )
            )
        assert config.cuda_graph_modules == [CudaGraphModule.attn]

    def test_config_deprecated_external_cuda_graph_reaches_the_gate(self):
        """The legacy flag migrates to the TE impl, and the gate sees the result.

        It carries no cuda_graph_modules, so with the split switched on it lands
        outside the split's required shape -- which is only observable if the gate
        reads the migrated impl rather than the raw legacy flag.
        """
        with pytest.raises(ValueError, match="requires cuda_graph_modules"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    external_cuda_graph=True, mhc_recompute_attn_cuda_graph_split=True
                )
            )
        # Without the switch the same config is simply accepted.
        config = TransformerConfig(**self._mhc_recompute_config_kwargs(external_cuda_graph=True))
        assert config.cuda_graph_impl == "transformer_engine"

    def test_config_rejects_deprecated_enable_cuda_graph_with_mhc_recompute(self):
        """The legacy flag migrates to the local impl, which the gate rejects."""
        with pytest.raises(ValueError, match="cuda_graph_impl='local'"):
            TransformerConfig(**self._mhc_recompute_config_kwargs(enable_cuda_graph=True))

    def test_config_rejects_local_impl_with_mhc_recompute(self):
        """Local capture records the mHC checkpoints and their recompute hooks into
        the layer graphs, where the backward-time RNG rewind cannot run -- the same
        wrong-result mechanism full_iteration+dropout fails closed on. Rejecting
        forfeits nothing: a captured checkpoint recovers no memory either way."""
        with pytest.raises(ValueError, match="cuda_graph_impl='local'"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    cuda_graph_impl="local", cuda_graph_modules=[CudaGraphModule.attn]
                )
            )

    def test_config_accepts_full_iteration_with_mhc_recompute_and_no_dropout(self):
        """Full-iteration capture swallows the eager recompute; dropout=0 is the gate."""
        config = TransformerConfig(
            **self._mhc_recompute_config_kwargs(
                cuda_graph_impl="full_iteration",
                cuda_graph_modules=[],
                hidden_dropout=0.0,
                attention_dropout=0.0,
            )
        )
        assert config.cuda_graph_impl == "full_iteration"

    @pytest.mark.parametrize(
        "dropout_kwargs",
        [
            {"hidden_dropout": 0.1, "attention_dropout": 0.0},
            {"hidden_dropout": 0.0, "attention_dropout": 0.1},
        ],
    )
    def test_config_rejects_full_iteration_mhc_recompute_with_dropout(self, dropout_kwargs):
        """RNG cannot be rewound inside capture, so dropout>0 must fail closed."""
        with pytest.raises(ValueError, match="requires hidden_dropout=0"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    cuda_graph_impl="full_iteration", cuda_graph_modules=[], **dropout_kwargs
                )
            )

    def test_config_accepts_vpp_whole_attention_capture_with_ep_overlap(self):
        """VPP + attn-scope graph (switch off: whole-attention capture) + EP
        overlap is admitted. The PP4/VPP2
        divergence (grad norm ~1e8, reproduced on pure upstream dev) was a
        caching-allocator use-after-free: mHC post-processing ran inside the
        communication-stream combine node, so the recompute subgraph was
        allocated on one stream and read from another. It is fixed by giving
        the post-processing its own compute-stream schedule node."""
        with pytest.warns(UserWarning, match="capturing the whole attention range"):
            config = TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    num_layers=4,
                    cuda_graph_impl="transformer_engine",
                    cuda_graph_modules=[CudaGraphModule.attn],
                    pipeline_model_parallel_size=2,
                    virtual_pipeline_model_parallel_size=2,
                    pipeline_dtype=torch.bfloat16,
                    overlap_moe_expert_parallel_comm=True,
                    expert_model_parallel_size=2,
                    num_moe_experts=4,
                    moe_token_dispatcher_type="alltoall",
                    bf16=True,
                )
            )
        assert config.virtual_pipeline_model_parallel_size == 2

    def test_config_rejects_te_whole_layer_capture_with_ep_overlap(self):
        """Empty cuda_graph_modules means whole-layer TE capture, which covers
        the MoE/MLP part; the generic overlap gate must reject it at config
        time exactly like an explicit moe/mlp scope, mirroring the runtime
        assert ("EP overlap must be disabled when CUDA graph captures the
        whole MLP/MoE part"). This is a generic (non-mHC) gate; it lives here
        with the rest of the overlap-config matrix."""
        with pytest.raises(AssertionError, match="whole-layer"):
            TransformerConfig(
                **self._mhc_recompute_config_kwargs(
                    num_layers=4,
                    cuda_graph_impl="transformer_engine",
                    cuda_graph_modules=[],
                    pipeline_model_parallel_size=2,
                    virtual_pipeline_model_parallel_size=2,
                    pipeline_dtype=torch.bfloat16,
                    overlap_moe_expert_parallel_comm=True,
                    expert_model_parallel_size=2,
                    num_moe_experts=4,
                    moe_token_dispatcher_type="alltoall",
                    bf16=True,
                )
            )

    def test_config_accepts_full_iteration_vpp_with_ep_overlap(self):
        """full_iteration + VPP + EP overlap is admitted (overlap is exercised
        here but no longer required): the divergence that used to gate this came
        from mHC post-processing running on the communication stream, and
        StaticBufferLoader is VPP-safe."""
        config = TransformerConfig(
            **self._mhc_recompute_config_kwargs(
                num_layers=4,
                cuda_graph_impl="full_iteration",
                cuda_graph_modules=[],
                hidden_dropout=0.0,
                attention_dropout=0.0,
                pipeline_model_parallel_size=2,
                virtual_pipeline_model_parallel_size=2,
                pipeline_dtype=torch.bfloat16,
                overlap_moe_expert_parallel_comm=True,
                expert_model_parallel_size=2,
                num_moe_experts=4,
                moe_token_dispatcher_type="alltoall",
                bf16=True,
            )
        )
        assert config.virtual_pipeline_model_parallel_size == 2

    def test_config_allows_vpp_with_mhc_recompute_without_cuda_graphs(self):
        """Eager recompute + VPP is legal, as it is with graphs. Kept as the
        no-graph corner of the VPP matrix."""
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

    @pytest.mark.parametrize(
        "graph_kwargs",
        (
            {"cuda_graph_impl": "transformer_engine", "cuda_graph_modules": [CudaGraphModule.attn]},
            # Full-iteration capture reaches this layer too: the config-level gate
            # that admits it is not model-family aware, so without a matching
            # exemption here the hybrid path would construct silently.
            {"cuda_graph_impl": "full_iteration", "hidden_dropout": 0.0, "attention_dropout": 0.0},
        ),
        ids=("te-attn-scope", "full-iteration"),
    )
    def test_hybrid_mhc_layer_warns_on_cuda_graphs_at_construction(self, graph_kwargs):
        """HybridStack mHC layers capture the mHC producer, so that one checkpoint
        does not pay -- but the combination was constructible before the split
        existed and nothing about it is known to be wrong, so it warns."""
        from megatron.core.models.hybrid.hybrid_block import HyperConnectionHybridLayer

        config = TransformerConfig(**self._mhc_recompute_config_kwargs(**graph_kwargs))
        with pytest.warns(UserWarning, match="HybridStack"):
            HyperConnectionHybridLayer(config, types.SimpleNamespace(layer_number=1))


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
        manager = MHCCheckpointManager()
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


class TestCheckpointRecomputeUnderFullGraphCapture:
    """Full-graph capture must swallow eager mHC recompute wholesale.

    With ``cuda_graph_impl="full_iteration"`` the whole iteration — including
    mHC checkpoint registration, the storage discard, the backward-time eager
    recompute, and the storage rebind — is recorded into one CUDA graph, so
    replays re-execute the recompute at fixed addresses by construction (no
    partial-graph bridge involved). These tests mirror FullCudaGraphWrapper
    mechanics (side-stream warmup, registered graph-safe RNG states, static
    input buffers) around a checkpointed mHC forward+backward and compare
    replayed gradients against eager references on fresh data.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123, use_cudagraphable_rng=True, force_reset_rng=True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_mhc_checkpoint_recompute_captured_forward_backward_matches_eager(self):
        hidden_size, num_streams, seq_len, batch = 32, 4, 8, 2
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

        static_x = torch.randn(
            seq_len, batch, num_streams * hidden_size, device="cuda", requires_grad=True
        )

        def run_step(x):
            manager = MHCCheckpointManager()
            aggregated, _h_res, h_post, _residual = module.forward(x, mhc_recompute_manager=manager)
            loss = aggregated.square().mean() + h_post.square().mean()
            manager.discard_all_outputs_and_register_unified_recompute(loss)
            loss.backward()
            return loss

        def zero_grads(x):
            with torch.no_grad():
                if x.grad is not None:
                    x.grad.zero_()
                for p in module.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

        # Warmup on a side stream per the torch CUDA-graphs contract; this also
        # materializes .grad tensors at addresses that stay fixed for capture.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                zero_grads(static_x)
                run_step(static_x)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        for state in get_all_rng_states().values():
            if isinstance(state, torch.Generator):
                graph.register_generator_state(state)
        zero_grads(static_x)
        with torch.cuda.graph(graph):
            static_loss = run_step(static_x)

        for _trial in range(3):
            fresh = torch.randn_like(static_x)

            # Eager reference with the same weights, same recompute machinery.
            zero_grads(static_x)
            ref_x = fresh.detach().clone().requires_grad_(True)
            ref_loss_t = run_step(ref_x)
            ref_loss = ref_loss_t.detach().clone()
            ref_x_grad = ref_x.grad.detach().clone()
            ref_param_grads = [
                p.grad.detach().clone() for p in module.parameters() if p.grad is not None
            ]

            # Captured replay on the same fresh data.
            zero_grads(static_x)
            with torch.no_grad():
                static_x.copy_(fresh)
            graph.replay()
            torch.cuda.synchronize()

            torch.testing.assert_close(static_loss, ref_loss)
            torch.testing.assert_close(static_x.grad, ref_x_grad)
            replay_param_grads = [p.grad for p in module.parameters() if p.grad is not None]
            assert len(replay_param_grads) == len(ref_param_grads)
            for got, want in zip(replay_param_grads, ref_param_grads):
                torch.testing.assert_close(got, want)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
