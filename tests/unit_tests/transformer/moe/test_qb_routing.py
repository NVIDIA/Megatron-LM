# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import cast

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import (
    compute_quantile_balancing_histogram,
    get_updated_expert_bias_quantile_balancing,
)
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestQBHistogram:
    """Pure-tensor tests for whole-step histogram Quantile Balancing."""

    @pytest.mark.internal
    @pytest.mark.parametrize("m,n,k", [(64, 8, 2), (40, 8, 1), (12, 4, 1)])
    def test_histograms_are_additive_across_microbatches(self, m, n, k):
        torch.manual_seed(123)
        scores = torch.sigmoid(torch.randn(m, n))
        beta = torch.zeros(n)
        whole = compute_quantile_balancing_histogram(scores, beta, k, num_bins=1000)
        split = compute_quantile_balancing_histogram(scores[: m // 2], beta, k, num_bins=1000)
        split += compute_quantile_balancing_histogram(scores[m // 2 :], beta, k, num_bins=1000)
        torch.testing.assert_close(split, whole)


class TestQuantileBalancingRouter:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.num_moe_experts = 8
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_score_function="sigmoid",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            moe_router_quantile_balancing_histogram=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        self.submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=self.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(self.submodules, MoESubmodules)
        self.moe_layer = MoELayer(self.transformer_config, self.submodules)
        self.router = cast(Router, self.moe_layer.router)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_non_qb_router_has_no_qb_buffers(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        router = MoELayer(config, self.submodules).router
        assert router.qb_beta is None
        assert router.local_quantile_balancing_histogram is None

    @pytest.mark.internal
    def test_original_qb_remains_the_default_variant(self):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=self.num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
        )
        router = MoELayer(config, self.submodules).router
        assert router.local_quantile_balancing_histogram is None
        assert router.qb_beta_accum is not None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_router_forward(self):
        self.router = self.router.cuda()

        num_tokens = 32 * 2
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        with torch.no_grad():
            probs, routing_map = self.router(hidden_states)

        assert probs.shape == (num_tokens, self.num_moe_experts)
        assert routing_map.shape == (num_tokens, self.num_moe_experts)
        # Each token selects exactly topk distinct experts.
        assert routing_map.sum().item() == num_tokens * self.router.topk

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_histogram_accumulates_in_training(self):
        self.router = self.router.cuda()
        self.router.train()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()

        assert self.router.local_quantile_balancing_histogram.sum().item() == 0
        self.router(hidden_states)
        first_histogram = self.router.local_quantile_balancing_histogram.clone()
        assert first_histogram.sum().item() > 0
        self.router(hidden_states)
        assert self.router.local_quantile_balancing_histogram.sum().item() == 2 * first_histogram.sum().item()

        # No accumulation outside the training path (eval / recompute).
        histogram_before = self.router.local_quantile_balancing_histogram.clone()
        with torch.no_grad():
            self.router(hidden_states)
        torch.testing.assert_close(self.router.local_quantile_balancing_histogram, histogram_before)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_router_excludes_padding_from_histogram(self):
        self.router = self.router.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        padding_mask = torch.zeros((32, 2), dtype=torch.bool, device=hidden_states.device)
        padding_mask[-2:] = True

        self.router.train()
        self.router(hidden_states, padding_mask=padding_mask)
        valid_tokens = padding_mask.numel() - padding_mask.sum().item()
        assert self.router.local_quantile_balancing_histogram.sum().item() == valid_tokens * self.num_moe_experts
