# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Unit tests for HyperConnection recomputation and single-pass mHC.

Tests the following functionality:
1. HyperConnectionModule._forward_with_checkpoint correctness
2. HyperConnectionModule.apply_h_post with MHCCheckpointManager
3. Multiple HyperConnectionModules chained with a single MHCCheckpointManager
4. Partial checkpoint (last layer not checkpointed)
5. TransformerConfig 'mhc' in recompute_modules option
6. Single-pass mHC math, shifted gradients, and stack integration
"""

import types
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    MHCCheckpointManager,
    get_all_rng_states,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.hyper_connection import HyperConnectionModule, SinglePassMHCState
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


class TestSinglePassMHC:
    """Single-pass mHC math, independent switch, and stack integration.

    The oracle follows Block.hc_mixes/hc_pre/hc_post in DeepSeek-V4.1-Flash,
    revision dba1be0a40aa45a94ad051997016db3960a90277, using ordinary Torch
    operations rather than production mapping, Sinkhorn, or mixing helpers.
    """

    @pytest.fixture
    def device(self, monkeypatch):
        """Run native math on CPU where available; only profiling ranges are replaced."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: nullcontext())
        return torch.device("cpu")

    @pytest.fixture
    def pg_collection(self):
        """Create the real process groups required by the production stacks."""
        Utils.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        yield ProcessGroupCollection.use_mpu_process_groups()
        Utils.destroy_model_parallel()

    def _module(self, device, *, version="v4.1", **overrides):
        if version == "v4.1":
            from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import (
                _make_config,
            )

            config = _make_config(**overrides)
        else:
            config = TransformerConfig(
                num_layers=2,
                hidden_size=32,
                num_attention_heads=4,
                use_cpu_initialization=True,
                enable_hyper_connections=True,
                num_residual_streams=4,
                mhc_sinkhorn_iterations=20,
                **overrides,
            )
        module = HyperConnectionModule(config=config, layer_number=1).to(device)
        # Different streams and substantial dynamic terms expose transposes, shifted
        # coefficients, and early BF16 rounding that identity-like initialization hides.
        with torch.no_grad():
            module.mapping_proj.weight.normal_(std=0.13)
            module.bias.uniform_(-0.7, 0.8)
            module.alpha_pre.fill_(0.3)
            module.alpha_post.fill_(0.2)
            module.alpha_res.fill_(0.4)
        return module

    def _reference_mappings(self, module, hidden, *, legacy=False):
        x = hidden.float()
        if legacy:
            inverse_rms = (x.norm(dim=-1, keepdim=True) / x.shape[-1] ** 0.5 + 1e-6).reciprocal()
            eps = 1e-6
        else:
            inverse_rms = torch.rsqrt(
                x.square().mean(dim=-1, keepdim=True) + module.config.layernorm_epsilon
            )
            eps = module.config.mhc_epsilon
        projection = F.linear(x, module.mapping_proj.weight.float()) * inverse_rms
        n = module.n
        pre = (projection[..., :n] * module.alpha_pre + module.bias[:n]).sigmoid() + eps
        post = (
            2 * (projection[..., n : 2 * n] * module.alpha_post + module.bias[n : 2 * n]).sigmoid()
        )
        comb = projection[..., 2 * n :] * module.alpha_res + module.bias[2 * n :]
        comb = comb.reshape(*hidden.shape[:-1], n, n).softmax(dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(module.config.mhc_sinkhorn_iterations - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return tuple(value.to(hidden.dtype) if legacy else value for value in (pre, post, comb))

    def _reference_contract(self, hidden, pre, n):
        streams = hidden.reshape(*hidden.shape[:-1], n, hidden.shape[-1] // n).float()
        return (streams * pre.unsqueeze(-1)).sum(dim=-2).to(hidden.dtype)

    def _reference_merge(self, hidden, output, post, comb, bias=None):
        streams = hidden.reshape(
            *hidden.shape[:-1], post.shape[-1], hidden.shape[-1] // post.shape[-1]
        ).float()
        # comb[i, j] routes residual stream i into output stream j.
        mixed = (comb.unsqueeze(-1) * streams.unsqueeze(-2)).sum(dim=-3)
        expanded = output.float() if bias is None else output.float() + bias.float()
        return (mixed + post.unsqueeze(-1) * expanded.unsqueeze(-2)).flatten(-2).to(output.dtype)

    def test_single_pass_requires_local_state_and_rejects_recompute(self, device):
        """The API must not silently reinitialize the shift or use V4 recomputation."""
        module = self._module(device)
        hidden = torch.randn(3, 2, 128, device=device)
        with pytest.raises(ValueError, match="forward-local SinglePassMHCState"):
            module(hidden)
        for kwargs in ({"mhc_recompute_manager": object()}, {"output_slot": object()}):
            with pytest.raises(ValueError, match="activation recomputation"):
                module(hidden, mhc_state=SinglePassMHCState(), **kwargs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("amplitude", [1.0, 1e-12])
    @pytest.mark.parametrize("version", ["v4", "v4.1"])
    def test_single_pass_mapping_values_and_gradients(self, device, dtype, amplitude, version):
        """Keep epsilon inside RMS and all coefficient arithmetic in FP32."""
        torch.manual_seed(2201)
        module = self._module(
            device, version=version, mhc_single_pass=True, layernorm_epsilon=1e-20, mhc_epsilon=1e-6
        )
        assert module.single_pass
        hidden = (torch.randn(3, 2, 4 * module.hidden_size, device=device) * amplitude).to(dtype)
        hidden.requires_grad_()
        reference_input = hidden.detach().clone().requires_grad_()
        actual = module.compute_mappings(hidden)
        expected = self._reference_mappings(module, reference_input)
        probes = [torch.randn_like(value) for value in expected]
        for value, reference in zip(actual, expected):
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, reference, atol=2e-6, rtol=2e-6)
        parameters = tuple(module.parameters())
        actual_grads = torch.autograd.grad(
            sum((value * probe).sum() for value, probe in zip(actual, probes)),
            (hidden, *parameters),
        )
        expected_grads = torch.autograd.grad(
            sum((value * probe).sum() for value, probe in zip(expected, probes)),
            (reference_input, *parameters),
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            assert torch.isfinite(actual_grad).all()
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_initial_mix_selects_first_stream_and_receives_previous_mix_gradient(
        self, device, dtype
    ):
        """The initial input uses one-hot pre, while supplied pre remains differentiable."""
        torch.manual_seed(2202)
        module = self._module(device)
        hidden = torch.randn(
            3, 2, 4 * module.hidden_size, device=device, dtype=dtype, requires_grad=True
        )
        state = SinglePassMHCState()
        aggregated, _, _, _ = module(hidden, mhc_state=state)
        torch.testing.assert_close(aggregated, hidden[..., : module.hidden_size], atol=0, rtol=0)
        torch.testing.assert_close(state.pre_mix, self._reference_mappings(module, hidden)[0])

        previous = torch.randn(3, 2, 4, device=device, dtype=torch.float32, requires_grad=True)
        state = SinglePassMHCState(pre_mix=previous)
        aggregated, _, _, _ = module(hidden, mhc_state=state)
        expected = self._reference_contract(hidden, previous, module.n)
        torch.testing.assert_close(aggregated, expected, atol=0, rtol=0)
        probe = torch.randn_like(aggregated)
        actual_grad = torch.autograd.grad((aggregated * probe).sum(), previous)[0]
        expected_grad = (hidden.float().view(3, 2, 4, -1) * probe.float().unsqueeze(-2)).sum(-1)
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
        assert actual_grad.abs().sum() > 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_single_pass_post_merge_matches_reference(self, device, dtype, with_bias):
        """Residual transpose, FP32 accumulation, and final activation cast match reference."""
        torch.manual_seed(2203)
        module = self._module(device)
        hidden = torch.randn(
            3, 2, 4 * module.hidden_size, device=device, dtype=dtype, requires_grad=True
        )
        output = torch.randn(
            3, 2, module.hidden_size, device=device, dtype=dtype, requires_grad=True
        )
        bias = torch.randn(module.hidden_size, device=device, dtype=dtype) if with_bias else None
        _, post, comb = self._reference_mappings(module, hidden)
        actual = module.fused_h_res_h_post_bda(comb, hidden, post, (output, bias), 0.0, True, False)
        expected = self._reference_merge(hidden, output, post, comb, bias)
        assert actual.dtype == dtype
        torch.testing.assert_close(
            actual, expected, atol=2e-6 if dtype == torch.float32 else 0, rtol=2e-6
        )
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(
            (actual * probe).sum(), (hidden, output), retain_graph=True
        )
        expected_grads = torch.autograd.grad((expected * probe).sum(), (hidden, output))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=2e-5, rtol=2e-5)

    @staticmethod
    def _reference_dsv4_contract(hidden, pre, n):
        streams = hidden.unflatten(-1, (n, hidden.shape[-1] // n))
        return (streams * pre.to(hidden.dtype).unsqueeze(-1)).sum(-2)

    @staticmethod
    def _reference_dsv4_merge(hidden, output, post, comb, bias=None):
        s, b, C = output.shape
        n = post.shape[-1]
        streams = hidden.reshape(s * b, n, C)
        mixed = torch.bmm(comb.reshape(s * b, n, n).transpose(1, 2), streams).reshape(s, b, n, C)
        expanded = post.unsqueeze(-1) * output.unsqueeze(-2)
        if bias is not None:
            expanded = expanded + post.unsqueeze(-1) * bias
        return (expanded + mixed).flatten(-2)

    def _run_chain(self, modules, hidden, *, reference=False, dsv4_mixing=False):
        state = SinglePassMHCState()
        contract = self._reference_dsv4_contract if dsv4_mixing else self._reference_contract
        merge = self._reference_dsv4_merge if dsv4_mixing else self._reference_merge
        previous = torch.zeros(
            *hidden.shape[:-1], modules[0].n, device=hidden.device, dtype=torch.float32
        )
        previous[..., 0] = 1
        pre_mixes = []
        for index, module in enumerate(modules):
            if reference:
                pre, post, comb = self._reference_mappings(module, hidden)
                if dsv4_mixing:
                    pre, post, comb = (value.to(hidden.dtype) for value in (pre, post, comb))
                aggregated = contract(hidden, previous, module.n)
                residual = hidden
            else:
                aggregated, comb, post, residual = module(hidden, mhc_state=state)
                pre = state.pre_mix
            # A differentiable stand-in for each attention/FFN isolates the mHC
            # recurrence; production attention and FFN are exercised separately below.
            output = torch.tanh(aggregated.float() * (0.2 + 0.07 * index)).to(hidden.dtype)
            if reference:
                hidden = merge(residual, output, post, comb)
            else:
                hidden = module.fused_h_res_h_post_bda(
                    comb, residual, post, (output, None), 0.0, True, False
                )
            pre.retain_grad()
            pre_mixes.append(pre)
            previous = pre
        contracted = (
            contract(hidden, previous, modules[-1].n)
            if reference
            else state.contract(hidden, modules[-1].n, use_fused=modules[-1].config.use_fused_mhc)
        )
        return contracted, pre_mixes

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("version", ["v4", "v4.1"])
    def test_attention_ffn_shift_and_final_contraction_gradients(self, device, dtype, version):
        """Two layers use prior FFN/attention pre; the final FFN pre trains through the head."""
        torch.manual_seed(2204)
        modules = [self._module(device, version=version, mhc_single_pass=True) for _ in range(4)]
        hidden = torch.randn(3, 2, 128, device=device, dtype=dtype, requires_grad=True)
        expected_input = hidden.detach().clone().requires_grad_()
        actual, actual_pre = self._run_chain(modules, hidden)
        expected, expected_pre = self._run_chain(modules, expected_input, reference=True)
        torch.testing.assert_close(
            actual, expected, atol=2e-6 if dtype == torch.float32 else 0, rtol=3e-6
        )
        probe = torch.randn_like(actual)
        parameters = tuple(parameter for module in modules for parameter in module.parameters())
        actual_grads = torch.autograd.grad((actual * probe).sum(), (hidden, *parameters))
        expected_grads = torch.autograd.grad(
            (expected * probe).sum(), (expected_input, *parameters)
        )
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=4e-5, rtol=4e-5)
        for actual_mix, expected_mix in zip(actual_pre, expected_pre):
            assert actual_mix.grad is not None and actual_mix.grad.abs().sum() > 0
            torch.testing.assert_close(actual_mix.grad, expected_mix.grad, atol=3e-5, rtol=3e-5)
        # These rows only predict pre. The last sublayer's rows would be unused if
        # final contraction averaged streams or retained the legacy learned head.
        last_mapping_index = next(
            index
            for index, parameter in enumerate(parameters)
            if parameter is modules[-1].mapping_proj.weight
        )
        last_pre_gradient = actual_grads[1 + last_mapping_index][:4]
        assert last_pre_gradient.abs().sum() > 0

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("version", ["v4", "v4.1"])
    def test_disabled_single_pass_keeps_current_sublayer_mix_and_dtype(
        self, device, dtype, version
    ):
        """Disabling the switch preserves legacy math for either attention version."""
        torch.manual_seed(2205)
        module = self._module(
            device, version=version, mhc_single_pass=False, layernorm_epsilon=1e-20
        )
        assert not module.single_pass
        hidden = torch.randn(3, 2, 128, device=device, dtype=dtype, requires_grad=True)
        expected_pre, expected_post, expected_comb = self._reference_mappings(
            module, hidden, legacy=True
        )
        aggregated, comb, post, residual = module(hidden)
        expected = (hidden.view(3, 2, 4, -1) * expected_pre.unsqueeze(-1)).sum(dim=2)
        torch.testing.assert_close(aggregated, expected)
        torch.testing.assert_close(post, expected_post)
        torch.testing.assert_close(comb, expected_comb)
        assert post.dtype == comb.dtype == dtype
        torch.testing.assert_close(residual, hidden, atol=0, rtol=0)

    def _build_single_pass_stack(self, stack_kind, pg_collection, version, use_fused=False):
        from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
            get_transformer_block_with_experimental_attention_variant_spec,
        )
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec
        from megatron.core.transformer.transformer_block import TransformerBlock
        from tests.unit_tests.transformer.experimental_attention_variant.test_dsv41 import (
            _make_config,
        )

        # Hybrid represents attention and FFN as separate layers. Include both so
        # the test covers attention -> FFN -> next attention and the final FFN pre.
        stride = 2 if stack_kind == "hybrid" else 1
        layers = 6 * stride
        sharing = dict(
            csa_compress_ratios=[ratio for ratio in [0, 2, 2, 1, 1, 1] for _ in range(stride)],
            csa2_kv_source_layers=[stride, 3 * stride],
            csa2_index_source_layers=[stride, 3 * stride, 4 * stride],
            csa2_candidate_source_layer=3 * stride,
        )
        if version == "v4":
            sharing = dict(
                csa_compress_ratios=[0] * layers,
                csa2_kv_source_layers=None,
                csa2_index_source_layers=None,
                csa2_candidate_source_layer=None,
                csa2_candidate_topk_blocks=0,
                csa2_candidate_block_size=0,
            )
        config = _make_config(
            dsv4_version=version,
            mhc_single_pass=True,
            use_fused_mhc=use_fused,
            num_layers=layers,
            num_moe_experts=None,
            moe_ffn_hidden_size=None,
            moe_shared_expert_intermediate_size=None,
            moe_router_enable_expert_bias=False,
            ffn_hidden_size=48,
            activation_func_clamp_value=None,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            **sharing,
        )
        if stack_kind == "transformer":
            return (
                TransformerBlock(
                    config,
                    get_transformer_block_with_experimental_attention_variant_spec(config),
                    pg_collection=pg_collection,
                )
                .cuda()
                .train()
            )
        return (
            HybridStack(
                config,
                hybrid_dsv4_stack_spec(config).submodules,
                layer_type_list=[Symbols.DS_ATTENTION, Symbols.MLP] * 6,
                pg_collection=pg_collection,
            )
            .cuda()
            .train()
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Production attention requires CUDA")
    @pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is not installed")
    @pytest.mark.parametrize("stack_kind", ["transformer", "hybrid"])
    @pytest.mark.parametrize("version", ["v4", "v4.1"])
    @pytest.mark.parametrize("use_fused", [False, True])
    def test_production_stack_shift_final_contract_and_live_forward_graphs(
        self, pg_collection, monkeypatch, stack_kind, version, use_fused
    ):
        """Both existing DSv4 stacks carry one local state and contract the final live pre."""
        stack = self._build_single_pass_stack(stack_kind, pg_collection, version, use_fused)
        assert not any("hc_head_" in name for name, _ in stack.named_parameters())
        assert not any("hc_head_" in name for name in stack.state_dict())
        records = []
        final_contracts = []
        original_forward = HyperConnectionModule.forward
        original_contract = SinglePassMHCState.contract

        def observe_forward(module, hidden_states, *args, **kwargs):
            state = kwargs.get("mhc_state")
            assert isinstance(state, SinglePassMHCState)
            previous = state.pre_mix
            result = original_forward(module, hidden_states, *args, **kwargs)
            expected = (
                hidden_states[..., : module.hidden_size]
                if previous is None
                else self._reference_contract(hidden_states, previous, module.n)
            )
            torch.testing.assert_close(result[0], expected, atol=2e-6, rtol=2e-6)
            state.pre_mix.retain_grad()
            records.append((module, state, previous, state.pre_mix))
            return result

        def observe_contract(state, hidden_states, n, **kwargs):
            assert kwargs.get("use_fused", False) is use_fused
            result = original_contract(state, hidden_states, n, **kwargs)
            final_contracts.append((state, hidden_states, state.pre_mix, result))
            return result

        monkeypatch.setattr(HyperConnectionModule, "forward", observe_forward)
        monkeypatch.setattr(SinglePassMHCState, "contract", observe_contract)
        # Observe the final norm input without bypassing any real layer or operator.
        norm_inputs = []
        final_norm = stack.final_layernorm if stack_kind == "transformer" else stack.final_norm
        hook = final_norm.register_forward_pre_hook(
            lambda module, args: norm_inputs.append(args[0])
        )
        inputs, outputs, forward_records = [], [], []
        for length in (5, 7):
            start = len(records)
            hidden = torch.randn(length, 2, 32, device="cuda", requires_grad=True)
            output = stack(hidden_states=hidden, attention_mask=None)
            current_records = records[start:]
            assert len(current_records) == 12  # attention + FFN for each of six logical layers
            assert current_records[0][2] is None
            state = current_records[0][1]
            for index, (_, current_state, previous, pre) in enumerate(current_records):
                assert current_state is state
                assert pre.dtype == torch.float32
                if index:
                    assert previous is current_records[index - 1][3]
            final_state, streams, last_pre, contracted = final_contracts[-1]
            assert final_state is state
            assert last_pre is current_records[-1][3]
            torch.testing.assert_close(contracted, self._reference_contract(streams, last_pre, 4))
            torch.testing.assert_close(norm_inputs[-1], contracted, atol=0, rtol=0)
            inputs.append(hidden)
            outputs.append(output)
            forward_records.append(current_records)
        hook.remove()
        assert forward_records[0][0][1] is not forward_records[1][0][1]
        for records_for_forward in reversed(forward_records):
            for _, _, _, pre in records_for_forward:
                assert pre.grad is None
        # Backpropagate the later forward first while the first forward is still live.
        probes = [torch.randn_like(output) for output in outputs]
        (outputs[1] * probes[1]).sum().backward()
        assert inputs[0].grad is None
        assert all(record[3].grad is None for record in forward_records[0])
        (outputs[0] * probes[0]).sum().backward()
        for hidden, records_for_forward in zip(inputs, forward_records):
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
            for _, _, _, pre in records_for_forward:
                assert pre.grad is not None and torch.isfinite(pre.grad).all()
                assert pre.grad.abs().sum() > 0
        for module in stack.modules():
            assert not any(isinstance(value, SinglePassMHCState) for value in vars(module).values())

    @staticmethod
    def _assert_fused_close(actual, expected, tolerance):
        """Use a scale-relative RMS bound, including tiny inputs and BF16 gradients."""
        assert actual.dtype == expected.dtype
        assert torch.isfinite(actual).all()
        if actual.numel() == 0:
            assert actual.shape == expected.shape
            return
        difference = (actual.double() - expected.double()).square().mean().sqrt()
        scale = expected.double().abs().max().clamp_min(1e-30)
        assert difference / scale < tolerance, (difference.item(), scale.item(), tolerance)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("amplitude,eps", [(1.0, 1e-4), (1e-12, 1e-20), (0.0, 1e-20)])
    @pytest.mark.parametrize("outputs_used", ["all", "pre", "r"])
    @pytest.mark.parametrize("hidden_size", [32, 2048])
    def test_fused_regularized_rms_values_and_gradients(
        self, device, dtype, amplitude, eps, outputs_used, hidden_size
    ):
        """Check regularized RMS, its external gradient, and unused mapping outputs."""
        from megatron.core.fusions.fused_mhc_kernels import fused_proj_rms_compute_h

        torch.manual_seed(2210)
        module = self._module(
            device, use_fused_mhc=True, layernorm_epsilon=eps, hidden_size=hidden_size
        )
        with torch.no_grad():
            module.mapping_proj.weight.mul_((32 / hidden_size) ** 0.5)
        n, K = module.n, module.n * module.hidden_size
        # Six rows exercise padded scalar reductions; K=8192 also selects the
        # split-K projection and the large-K input/weight gradient kernel.
        x = (torch.randn(6, K, device=device) * amplitude).to(dtype).requires_grad_()
        ref_x = x.detach().clone().requires_grad_()
        parameters = (
            module.mapping_proj.weight,
            module.alpha_pre,
            module.alpha_post,
            module.alpha_res,
            module.bias,
        )
        actual = fused_proj_rms_compute_h(
            x, *parameters, n, eps, module.compute_h_eps, eps_inside_sqrt=True
        )
        weight, ap, apo, ar, bias = parameters
        r = (ref_x.float().square().mean(-1, keepdim=True) + eps).sqrt()
        projection = F.linear(ref_x.float(), weight) / r
        expected = (
            (projection[:, :n] * ap + bias[:n]).sigmoid() + module.compute_h_eps,
            (projection[:, n : 2 * n] * apo + bias[n : 2 * n]).sigmoid() * 2,
            projection[:, 2 * n :] * ar + bias[2 * n :],
            r,
        )
        for index, (value, reference) in enumerate(zip(actual, expected)):
            self._assert_fused_close(value, reference, 1e-5 if index == 3 else 1e-4)
        indices = range(4) if outputs_used == "all" else [0 if outputs_used == "pre" else 3]
        probes = [torch.randn_like(expected[index]) for index in indices]
        actual_grads = torch.autograd.grad(
            sum((actual[index] * probe).sum() for index, probe in zip(indices, probes)),
            (x, *parameters),
            allow_unused=True,
        )
        expected_grads = torch.autograd.grad(
            sum((expected[index] * probe).sum() for index, probe in zip(indices, probes)),
            (ref_x, *parameters),
            allow_unused=True,
        )
        for value, reference, parameter in zip(actual_grads, expected_grads, (x, *parameters)):
            value = torch.zeros_like(parameter) if value is None else value
            reference = torch.zeros_like(parameter) if reference is None else reference
            self._assert_fused_close(
                value, reference, 2e-3 if value.dtype == torch.bfloat16 else 5e-4
            )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_fused_mixing_uses_dsv4_dtypes(self, device, dtype, with_bias, monkeypatch):
        """Use activation-dtype coefficients with the existing DSv4 mixing operators."""
        from megatron.core.fusions import fused_mhc_kernels

        torch.manual_seed(2211)
        module = self._module(device, use_fused_mhc=True)
        hidden = torch.randn(3, 2, 128, device=device, dtype=dtype, requires_grad=True)
        x = torch.randn(3, 2, 32, device=device, dtype=dtype, requires_grad=True)
        previous = torch.randn(3, 2, 4, device=device, dtype=dtype, requires_grad=True)
        bias = (
            torch.randn(32, device=device, dtype=dtype, requires_grad=True) if with_bias else None
        )
        state = SinglePassMHCState(previous)
        aggregated, comb, post, residual = module(hidden, mhc_state=state)
        assert state.pre_mix.dtype == comb.dtype == post.dtype == dtype
        assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
        post_calls = []
        original_post = fused_mhc_kernels.fused_h_post_bda

        def observe_post(comb, streams, post, x, bias):
            assert comb.dtype == streams.dtype == post.dtype == x.dtype == dtype
            if bias is not None:
                assert bias.dtype == dtype
            post_calls.append(x.shape)
            return original_post(comb, streams, post, x, bias)

        monkeypatch.setattr(module, "_h_post_bda_op", observe_post)
        output = module.fused_h_res_h_post_bda(comb, residual, post, (x, bias), 0.0, True, False)
        assert len(post_calls) == 1
        assert aggregated.dtype == output.dtype == dtype
        # Use the DSv4 native expression, including activation-dtype rounding.
        expected_aggregate = self._reference_dsv4_contract(hidden, previous, 4)
        expected_output = self._reference_dsv4_merge(hidden, x, post, comb, bias)
        tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
        torch.testing.assert_close(aggregated, expected_aggregate, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(output, expected_output, atol=tolerance, rtol=tolerance)
        final = state.contract(output, 4, use_fused=True)
        inputs = (hidden, x, previous, *module.parameters()) + ((bias,) if with_bias else ())
        grads = torch.autograd.grad(
            (final, aggregated), inputs, (torch.randn_like(final), torch.randn_like(aggregated))
        )
        for value, parameter in zip(grads, inputs):
            assert value.dtype == parameter.dtype and torch.isfinite(value).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_fused_single_pass_chain_and_final_mix(self, device, dtype, batch_size, monkeypatch):
        """Check SBHD/THD-shaped recurrence and gradients through all shifted pre maps."""
        from megatron.core.fusions import fused_mhc_kernels

        torch.manual_seed(2212)
        modules = [self._module(device, use_fused_mhc=True) for _ in range(4)]
        calls = []
        grad_sum_calls = []
        aggregate_calls = []
        original_aggregate = fused_mhc_kernels.fused_h_aggregate

        def observe_aggregate(x, pre):
            assert pre.dtype == x.dtype == dtype
            aggregate_calls.append(pre)
            return original_aggregate(x, pre)

        monkeypatch.setattr(fused_mhc_kernels, "fused_h_aggregate", observe_aggregate)
        for module in modules:
            original = module._proj_rms_compute_h_op
            original_grad_sum = module._fused_add_3_op

            def observe_projection(x, *args, _original=original, **kwargs):
                assert x.dtype == dtype
                assert kwargs["eps_inside_sqrt"]
                calls.append(x.shape)
                return _original(x, *args, **kwargs)

            monkeypatch.setattr(module, "_proj_rms_compute_h_op", observe_projection)

            def observe_grad_sum(*grads, _original=original_grad_sum):
                grad_sum_calls.append(tuple(grad.shape for grad in grads))
                return _original(*grads)

            monkeypatch.setattr(module, "_fused_add_3_op", observe_grad_sum)
        hidden = torch.randn(7, batch_size, 128, device=device, dtype=dtype, requires_grad=True)
        ref_hidden = hidden.detach().clone().requires_grad_()
        actual, actual_pre = self._run_chain(modules, hidden)
        expected, expected_pre = self._run_chain(
            modules, ref_hidden, reference=True, dsv4_mixing=True
        )
        assert len(calls) == len(modules)
        # Three shifted contractions plus the last FFN's contraction at stack exit.
        assert len(aggregate_calls) == len(modules)
        assert all(used is predicted for used, predicted in zip(aggregate_calls, actual_pre))
        self._assert_fused_close(actual, expected, 2e-2 if dtype == torch.bfloat16 else 3e-4)
        parameters = tuple(parameter for module in modules for parameter in module.parameters())
        probe = torch.randn_like(actual)
        actual_grads = torch.autograd.grad((actual * probe).sum(), (hidden, *parameters))
        assert len(grad_sum_calls) == len(modules)
        expected_grads = torch.autograd.grad((expected * probe).sum(), (ref_hidden, *parameters))
        for value, reference in zip(actual_grads, expected_grads):
            if dtype == torch.bfloat16:
                # Existing DSv4 kernels round intermediates differently from eager Torch.
                torch.testing.assert_close(value, reference, atol=5e-2, rtol=5e-2)
            else:
                self._assert_fused_close(value, reference, 8e-4)
        for value, reference in zip(actual_pre, expected_pre):
            assert value.grad is not None and value.grad.abs().sum() > 0
            if dtype == torch.bfloat16:
                torch.testing.assert_close(value.grad, reference.grad, atol=5e-2, rtol=5e-2)
            else:
                self._assert_fused_close(value.grad, reference.grad, 8e-4)

    def test_fused_post_dropout_keeps_native_branch_mask(self, device, monkeypatch):
        """Training dropout masks only the expanded branch, leaving residual mixing intact."""
        torch.manual_seed(2213)
        module = self._module(device, use_fused_mhc=True)
        hidden = torch.randn(3, 2, 128, device=device, requires_grad=True)
        x = torch.randn(3, 2, 32, device=device, requires_grad=True)
        _, post, comb = self._reference_mappings(module, hidden)

        def unexpected_fused_post(*args):
            raise AssertionError("The no-dropout kernel cannot apply training dropout")

        monkeypatch.setattr(module, "_h_post_bda_op", unexpected_fused_post)
        torch.manual_seed(2214)
        actual = module.fused_h_res_h_post_bda(comb, hidden, post, (x, None), 0.3, True, False)
        streams = hidden.unflatten(-1, (4, 32))
        mixed = (comb.unsqueeze(-1) * streams.unsqueeze(-2)).sum(-3)
        torch.manual_seed(2214)
        expected = (mixed + F.dropout(post.unsqueeze(-1) * x.unsqueeze(-2), p=0.3)).flatten(-2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        probe = torch.randn_like(actual)
        inputs = (hidden, x, post, comb)
        actual_grads = torch.autograd.grad(actual, inputs, probe, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, inputs, probe)
        for value, reference in zip(actual_grads, expected_grads):
            torch.testing.assert_close(value, reference, rtol=0, atol=0)

    @pytest.mark.parametrize("enabled", [False, True])
    def test_single_pass_cli_roundtrip_without_attention_version(self, enabled):
        """The CLI switch works independently of any experimental attention variant."""
        from argparse import ArgumentParser

        from megatron.training.arguments import _add_network_size_args

        parser = ArgumentParser()
        _add_network_size_args(parser)
        args = parser.parse_args(["--mhc-single-pass"] if enabled else [])
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            enable_hyper_connections=True,
            mhc_single_pass=args.mhc_single_pass,
        )
        assert config.mhc_single_pass is enabled
        assert config.dsv4_version == "v4"
        assert config.experimental_attention_variant is None

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"enable_hyper_connections": False}, "requires enable_hyper_connections=True"),
            (
                {
                    "recompute_granularity": "full",
                    "recompute_method": "uniform",
                    "recompute_num_layers": 1,
                },
                "activation recomputation",
            ),
        ],
    )
    def test_single_pass_validates_its_own_prerequisites(self, overrides, message):
        values = dict(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            enable_hyper_connections=True,
            mhc_single_pass=True,
        )
        values.update(overrides)
        with pytest.raises(ValueError, match=message):
            TransformerConfig(**values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
