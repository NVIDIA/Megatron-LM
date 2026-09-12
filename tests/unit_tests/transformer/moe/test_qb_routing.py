# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import replace
from typing import cast

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import qb_dual_update
from megatron.core.transformer.moe.router import Router, TopKRouter
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_transformer_config import _virtual_expert_hybridep_config

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.internal
@pytest.mark.parametrize("quantile", [False, True], ids=["seq_aux_loss", "quantile"])
def test_nt4_compact_fused_router(monkeypatch, quantile):
    """Recipe-sized expert IDs and scores agree with a dense unfused router and CPU math."""
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    config = _virtual_expert_hybridep_config(
        moe_virtual_expert_load_balance=False,
        num_moe_experts=512,
        moe_router_topk=10,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=2.5,
        moe_router_load_balancing_type="quantile_balancing" if quantile else "seq_aux_loss",
        moe_aux_loss_coeff=0 if quantile else 1e-4,
        moe_router_enable_expert_bias=not quantile,
        moe_router_fusion=True,
    )
    torch.manual_seed(321)
    logits_cpu = torch.randn(256, 512)
    logits_cpu[0] = -100  # Selected zero-probability routes must retain their expert ids.
    bias = torch.randn(512) * 0.3
    scores = logits_cpu.sigmoid()
    selection = logits_cpu - bias if quantile else scores + bias
    expected_ids = selection.argsort(dim=1, descending=True)[:, :10].sort(dim=1).values
    selected = scores.gather(1, expected_ids)
    expected_probs = 2.5 * selected / (selected.sum(dim=1, keepdim=True) + 1e-20)
    dy = torch.randn(256, 512, device="cuda")
    calls = []
    fused = moe_utils.fused_topk_with_score_function

    def record_fused(**kwargs):
        calls.append((virtual, kwargs["logits"].shape, kwargs.get("topk_indices")))
        if not virtual:
            assert "topk_indices" not in kwargs, "ordinary routing must work with older TE APIs"
        return fused(**kwargs)

    monkeypatch.setattr(moe_utils, "fused_topk_with_score_function", record_fused)
    monkeypatch.setattr(moe_utils.MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    moe_utils.MoEAuxLossAutoScaler.set_loss_scale(torch.tensor(0.37, device="cuda"))
    pg = ProcessGroupCollection.use_mpu_process_groups()
    try:
        reference = TopKRouter(
            replace(config, moe_router_fusion=False, moe_token_dispatcher_type="alltoall"), pg
        ).cuda()
        reference.set_layer_number(1)
        (reference.qb_beta if quantile else reference.expert_bias).copy_(bias)
        ref_logits = logits_cpu.cuda().requires_grad_()
        ref_probs, ref_map = reference.routing(ref_logits.view(256, 1, 512))
        (ref_probs * dy).sum().backward()
        torch.testing.assert_close(
            ref_map.cpu(),
            torch.zeros_like(logits_cpu, dtype=torch.bool).scatter(1, expected_ids, True),
        )
        for virtual in (False, True):
            router = TopKRouter(replace(config, moe_virtual_expert_load_balance=virtual), pg).cuda()
            router.set_layer_number(1)
            (router.qb_beta if quantile else router.expert_bias).copy_(bias)
            logits = logits_cpu.cuda().requires_grad_()
            probs, ids = router.routing(logits.view(256, 1, 512))
            assert (ids.dtype != torch.bool) == virtual
            if not virtual:
                assert ids.shape == probs.shape == (256, 512)
                ids = ids.to(torch.int8).topk(10, dim=1).indices
                probs = probs.gather(1, ids)
            assert probs.shape == ids.shape == (256, 10) and ids.dtype == torch.int64
            sorted_ids, order = ids.sort(dim=1)
            torch.testing.assert_close(sorted_ids.cpu(), expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(
                probs.gather(1, order).cpu(), expected_probs, rtol=1e-5, atol=1e-7
            )
            assert ids.max() > 64 and not probs[0].any()
            (probs * dy.gather(1, ids)).sum().backward()
            torch.testing.assert_close(logits.grad, ref_logits.grad, rtol=2e-4, atol=1e-6)
            assert logits.grad[1:].norm() > 0
            if quantile:
                alpha = selection.sort(dim=1, descending=True).values[:, 10:11]
                expected_beta = (logits_cpu - alpha).sort(dim=0, descending=True).values[5]
                torch.testing.assert_close(router.qb_beta_accum.cpu(), expected_beta)
                assert router.qb_beta_count.item() == 1
                # No duplicate accumulation during no-grad recomputation or eval.
                with torch.no_grad():
                    router.routing(logits.view(256, 1, 512))
                router.eval()
                router.routing(logits.view(256, 1, 512))
                torch.testing.assert_close(router.qb_beta_accum.cpu(), expected_beta)
                assert router.qb_beta_count.item() == 1
                assert "qb_beta" in router.state_dict()
                assert "qb_beta_accum" not in router.state_dict()
            else:
                torch.testing.assert_close(
                    router.local_tokens_per_expert, ref_map.sum(dim=0).float(), rtol=0, atol=0
                )
        for virtual, shape, index_buffer in calls:
            if quantile:
                assert shape == (256, 10) and index_buffer is None
            elif not virtual:
                assert shape == (256, 512) and index_buffer is None
            else:
                assert shape == (256, 512) and index_buffer.shape == (256, 10)
        assert calls, "the fused scorer must actually run"
    finally:
        from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker

        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()


class TestQBDualUpdate:
    """Pure-tensor tests for the quantile-balancing dual update (CPU, no distributed)."""

    @pytest.mark.internal
    @pytest.mark.parametrize("m,n,k", [(64, 8, 2), (40, 8, 1), (12, 4, 1)])
    def test_column_quantile_contract(self, m, n, k):
        """qb_beta_local is the (col_target+1)-th largest score minus alpha per expert."""
        torch.manual_seed(123)
        scores = torch.randn(m, n)
        beta = torch.zeros(n)

        _, beta_local = qb_dual_update(scores, k, beta, update_beta=True)

        alpha = (scores - beta).topk(k + 1, dim=1).values[:, -1:]
        adjusted = scores - alpha
        col_target = m * k // n
        expected = adjusted.sort(dim=0, descending=True).values[col_target]
        torch.testing.assert_close(beta_local, expected)


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
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
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
        assert router.qb_beta_accum is None
        assert router.qb_beta_count is None

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("moe_router_pre_softmax", [True, False])
    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_qb_router_forward(self, score_function, moe_router_pre_softmax):
        self.router = self.router.cuda()
        self.router.config.moe_router_score_function = score_function
        self.router.score_function = score_function
        self.router.config.moe_router_pre_softmax = moe_router_pre_softmax

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
    def test_qb_beta_accumulates_in_training(self):
        self.router = self.router.cuda()
        self.router.train()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()

        assert self.router.qb_beta_count.item() == 0
        self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 1
        assert self.router.qb_beta_accum.abs().sum().item() > 0
        self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 2

        # No accumulation outside the training path (eval / recompute).
        accum_before = self.router.qb_beta_accum.clone()
        with torch.no_grad():
            self.router(hidden_states)
        assert self.router.qb_beta_count.item() == 2
        torch.testing.assert_close(self.router.qb_beta_accum, accum_before)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_qb_router_rejects_padding_mask(self):
        self.router = self.router.cuda()
        hidden_states = torch.randn((32, 2, self.router.config.hidden_size)).cuda().bfloat16()
        padding_mask = torch.zeros((32, 2), dtype=torch.bool, device=hidden_states.device)
        padding_mask[-2:] = True

        with pytest.raises(AssertionError, match="does not support padding masks"):
            self.router(hidden_states, padding_mask=padding_mask)
