# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


class TestSharedExperts:
    def setup_method(self, method):
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=16,
            moe_shared_expert_intermediate_size=32,
            moe_shared_expert_overlap=False,
            moe_token_dispatcher_type="alltoall",
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=4,
            add_bias_linear=False,
        )

    def get_moe_layer(self, **kargs) -> MoELayer:
        new_config = dataclasses.replace(self.config, **kargs)
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=new_config.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        if get_tensor_model_parallel_world_size() > 1:
            new_config.sequence_parallel = True
        moe_layer = MoELayer(new_config, submodules)
        moe_layer.cuda()
        return moe_layer

    def assert_shared_expert_overlap_matches_no_overlap(self, hidden_shape, **config_kwargs):
        """Check that shared-expert overlap preserves forward and backward numerics."""
        # Create MoE layer with shared expert overlap enabled.
        model_parallel_cuda_manual_seed(123)
        moe_layer_overlap = self.get_moe_layer(moe_shared_expert_overlap=True, **config_kwargs).to(
            dtype=torch.bfloat16
        )

        # Create MoE layer with shared expert overlap disabled.
        model_parallel_cuda_manual_seed(123)
        moe_layer_no_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=False, **config_kwargs
        ).to(dtype=torch.bfloat16)
        moe_layer_no_overlap.load_state_dict(moe_layer_overlap.state_dict())

        # Sanity check that the weights are identical.
        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.equal(p_overlap, p_no_overlap)

        # Verify attributes of the MoE layers.
        num_weights_overlap = sum([p.numel() for p in moe_layer_overlap.parameters()])
        num_weights_no_overlap = sum([p.numel() for p in moe_layer_no_overlap.parameters()])
        assert num_weights_overlap == num_weights_no_overlap

        assert moe_layer_overlap.shared_experts is not None
        assert moe_layer_overlap.shared_experts.stream is not None
        assert moe_layer_overlap.token_dispatcher.shared_experts is not None

        assert moe_layer_no_overlap.shared_experts is not None
        assert moe_layer_no_overlap.token_dispatcher.shared_experts is None

        if moe_layer_overlap.config.moe_latent_size is None:
            assert moe_layer_overlap.token_dispatcher._shared_expert_prepost_in_dispatcher()
        else:
            assert moe_layer_overlap._uses_latent_shared_expert_dispatch_overlap()
            assert not moe_layer_overlap.token_dispatcher._shared_expert_prepost_in_dispatcher()

        # Create a dummy input tensor.
        hidden_states = torch.randn(
            hidden_shape, requires_grad=True, device="cuda", dtype=torch.bfloat16
        )
        hidden_states_no_overlap = hidden_states.clone().detach().requires_grad_(True)

        # Forward pass.
        output_overlap, _ = moe_layer_overlap(hidden_states)
        output_no_overlap, _ = moe_layer_no_overlap(hidden_states_no_overlap)
        torch.testing.assert_close(output_overlap, output_no_overlap)

        # Backward pass.
        output_overlap.mean().backward()
        output_no_overlap.mean().backward()

        # Check gradients.
        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.allclose(
                p_overlap.grad, p_no_overlap.grad
            ), f"max diff: {torch.max(torch.abs(p_overlap.grad - p_no_overlap.grad))}"

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dispatcher_type", ["alltoall"])
    @pytest.mark.parametrize("moe_latent_size", [None, 8])
    @pytest.mark.parametrize("tp_size, ep_size", [[1, 1], [4, 1], [1, 4], [2, 4]])
    def test_shared_expert_forward_backward(
        self, dispatcher_type: str, moe_latent_size, tp_size, ep_size
    ):
        """
        Tests that the MoELayer with and without shared expert overlap produce
        identical outputs and gradients.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        self.assert_shared_expert_overlap_matches_no_overlap(
            (32, 2, self.config.hidden_size),
            moe_token_dispatcher_type=dispatcher_type,
            moe_latent_size=moe_latent_size,
        )

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.skipif(not is_hybrid_ep_available(), reason="HybridEP is not available")
    def test_latent_shared_expert_hybridep_forward_backward(self):
        """
        Tests that latent MoE shared expert overlap with HybridEP produces identical
        outputs and gradients to the non-overlapped path.
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=8)
        self.assert_shared_expert_overlap_matches_no_overlap(
            (32, 2, 1024),
            hidden_size=1024,
            num_attention_heads=8,
            num_moe_experts=8,
            moe_router_topk=2,
            ffn_hidden_size=1024,
            moe_ffn_hidden_size=1024,
            moe_shared_expert_intermediate_size=1024,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_router_dtype="fp32",
            moe_latent_size=256,
        )
