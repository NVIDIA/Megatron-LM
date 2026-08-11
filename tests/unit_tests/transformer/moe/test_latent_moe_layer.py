# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestLatentMoELayer:
    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [4])
    @pytest.mark.parametrize("use_te,grouped_gemm", [(True, True), (True, False), (False, False)])
    @pytest.mark.parametrize("moe_latent_size", [8, 16])
    @pytest.mark.parametrize("moe_latent_output_norm", [False, True])
    def test_latent_moe_layer(
        self,
        num_moe_experts,
        moe_token_dispatcher_type,
        use_te,
        grouped_gemm,
        moe_latent_size,
        moe_latent_output_norm,
    ):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            moe_shared_expert_intermediate_size=128,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            moe_latent_size=moe_latent_size,
            moe_latent_output_norm=moe_latent_output_norm,
            normalization="RMSNorm",
        )
        if use_te:
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            )
        else:
            transformer_layer_submodules = get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
            )
        submodules = get_submodules(transformer_layer_submodules.mlp)
        assert isinstance(submodules, MoESubmodules)
        moe_layer = MoELayer(self.transformer_config, submodules)
        moe_layer.cuda()
        config = moe_layer.config

        if moe_latent_output_norm:
            assert moe_layer.routed_expert_norm.weight.shape[0] == config.moe_latent_size
        else:
            assert not hasattr(moe_layer, "routed_expert_norm")

        assert (
            moe_layer.shared_experts.linear_fc1.weight.shape[1] == config.hidden_size
        ), "Shared expert computation has to happen in hidden dimension."
        assert (
            moe_layer.shared_experts.linear_fc2.weight.shape[0] == config.hidden_size
        ), "Shared expert computation has to happen in hidden dimension."
        if grouped_gemm:
            for i in range(num_moe_experts):
                fc1_weight = getattr(moe_layer.experts.linear_fc1, f"weight{i}")
                fc2_weight = getattr(moe_layer.experts.linear_fc2, f"weight{i}")
                assert (
                    fc1_weight.shape[1] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc1_weight.shape=}"
                assert (
                    fc2_weight.shape[0] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc2_weight.shape=}"
        else:
            for i in range(num_moe_experts):
                expert = moe_layer.experts.local_experts[i]
                assert (
                    expert.linear_fc1.weight.shape[1] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc1_weight.shape=}"
                assert (
                    expert.linear_fc2.weight.shape[0] == config.moe_latent_size
                ), f"Shape mismatch for expert {i} {fc2_weight.shape=}"
        assert (
            moe_layer.router.weight.shape[1] == config.hidden_size
        ), "MoE routing has to happen in hidden dimension."

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((32, 2, config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, _ = moe_layer(hidden_states)
        assert output.shape[2] == config.hidden_size

        Utils.destroy_model_parallel()

    def test_latent_output_norm_requires_latent_size(self):
        with pytest.raises(ValueError, match="requires moe_latent_size"):
            TransformerConfig(
                num_layers=1, hidden_size=32, num_attention_heads=4, moe_latent_output_norm=True
            )

    def test_latent_output_norm_is_applied_after_combine_and_before_up_projection(self):
        class FakeDispatcher:
            @staticmethod
            def combine_postprocess(output):
                return output + 1

        class FakeNorm(torch.nn.Module):
            def forward(self, output):
                return output * 2

        class FakeProjection(torch.nn.Module):
            def forward(self, output):
                return output + 3, None

        moe_layer = object.__new__(MoELayer)
        torch.nn.Module.__init__(moe_layer)
        moe_layer.config = SimpleNamespace(moe_latent_size=2, moe_latent_output_norm=True)
        moe_layer.token_dispatcher = FakeDispatcher()
        moe_layer.routed_expert_norm = FakeNorm()
        moe_layer.fc2_latent_proj = FakeProjection()
        moe_layer._latent_shared_expert_output = None

        output = moe_layer.postprocess(torch.zeros(1, 2), shared_expert_output=None)

        # combine (+1), normalize (*2), then project (+3)
        torch.testing.assert_close(output, torch.full((1, 2), 5.0))
