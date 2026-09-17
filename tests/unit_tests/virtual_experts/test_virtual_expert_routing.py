# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Configuration boundaries and NT4 compact routing semantics."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.internal


def _virtual_expert_hybridep_config(**overrides):
    """Build a minimal virtual-expert HybridEP config, then apply one override."""
    kwargs = dict(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_virtual_expert_load_balance=True,
        moe_grouped_gemm=True,
        moe_router_dtype="fp32",
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        add_bias_linear=False,
        activation_func=F.silu,
        gated_linear_unit=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_virtual_expert_hybridep_defaults_a_dropless_rank_capacity():
    """The backend is dropless by construction and allows the whole-layer moe graph."""
    config = _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=["moe"])

    assert config.moe_expert_rank_capacity_factor == 1.0
    assert config.moe_single_grouped_weight is False


def test_virtual_expert_hybridep_accepts_native_mxfp8_with_router_padding():
    """Native MXFP8 parameters are the only quantized storage the push understands."""
    config = _virtual_expert_hybridep_config(
        fp8="e4m3", fp8_recipe="mxfp8", fp8_param=True, moe_router_padding_for_quantization=True
    )

    assert (config.fp8, config.fp8_recipe, config.fp8_param) == ("e4m3", "mxfp8", True)
    assert config.moe_router_padding_for_quantization


@pytest.mark.parametrize(
    ("fp8", "fp8_recipe", "fp8_param"),
    [("e4m3", "mxfp8", False), ("e4m3", "tensorwise", True), ("hybrid", "mxfp8", True)],
)
def test_virtual_expert_hybridep_rejects_unsupported_fp8_parameter_storage(
    fp8, fp8_recipe, fp8_param
):
    with pytest.raises(ValueError, match="MXFP8 E4M3 with native FP8 parameters"):
        _virtual_expert_hybridep_config(fp8=fp8, fp8_recipe=fp8_recipe, fp8_param=fp8_param)


@pytest.mark.parametrize("scope", ["moe_router", "moe_preprocess"])
def test_virtual_expert_hybridep_rejects_partial_moe_cuda_graph_scopes(scope):
    """Only the whole-layer moe scope preserves the planner's per-forward metadata."""
    with pytest.raises(AssertionError, match="moe CUDA graph scope only"):
        _virtual_expert_hybridep_config(cuda_graph_impl="local", cuda_graph_modules=[scope])


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"moe_token_dispatcher_type": "alltoall"}, "moe_token_dispatcher_type='flex'"),
        ({"expert_model_parallel_size": 1}, "2<=expert_model_parallel_size<=64"),
        ({"num_moe_experts": 3}, "num_moe_experts divisible"),
        ({"moe_router_topk": 3}, "1<=moe_router_topk"),
        ({"moe_expert_capacity_factor": 1.0}, "moe_expert_capacity_factor=None"),
        (
            {"recompute_granularity": "selective", "recompute_modules": ["moe"]},
            "no MoE layer recompute",
        ),
    ],
)
def test_virtual_expert_rejects_unsupported_layout(overrides, match):
    """Reject combinations that would drop routes or invalidate runtime storage."""
    with pytest.raises(ValueError, match=match):
        _virtual_expert_hybridep_config(**overrides)


@pytest.mark.internal
@pytest.mark.parametrize(
    "quantile,expert_bias,compact_supported",
    [(True, False, True), (True, False, False), (False, False, True)],
    ids=["nt4-compact", "nt4-dense-fallback", "fused-seq-aux"],
)
def test_nt4_compact_router(monkeypatch, quantile, expert_bias, compact_supported):
    """Recipe-sized expert IDs and scores agree with a dense unfused router and CPU math."""
    monkeypatch.setattr(moe_utils, "hybrid_ep_dense_topk_routing", lambda *_: compact_supported)
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    config = _virtual_expert_hybridep_config(
        moe_virtual_expert_load_balance=False,
        num_moe_experts=512,
        moe_router_topk=10,
        moe_router_score_function="sigmoid",
        moe_router_dtype="fp32",
        moe_router_topk_scaling_factor=3.16,
        moe_router_load_balancing_type="quantile_balancing" if quantile else "seq_aux_loss",
        moe_aux_loss_coeff=0 if quantile else 1e-4,
        moe_router_enable_expert_bias=expert_bias,
        moe_router_fusion=not quantile and not expert_bias,
    )
    torch.manual_seed(321)
    logits_cpu = torch.randn(256, 512)
    if quantile or expert_bias:
        logits_cpu[0] = -100  # Bias resolves ties between selected zero-probability routes.
    bias = torch.randn(512) * 0.3 if quantile or expert_bias else torch.zeros(512)
    scores = logits_cpu.sigmoid()
    selection = logits_cpu - bias if quantile else scores + bias
    expected_ids = selection.argsort(dim=1, descending=True)[:, :10].sort(dim=1).values
    selected = scores.gather(1, expected_ids)
    expected_probs = 3.16 * selected / (selected.sum(dim=1, keepdim=True) + 1e-20)
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
        if quantile or expert_bias:
            (reference.qb_beta if quantile else reference.expert_bias).copy_(bias)
        ref_logits = logits_cpu.cuda().requires_grad_()
        ref_probs, ref_map = reference.routing(ref_logits.view(256, 1, 512))
        (ref_probs * dy).sum().backward()
        torch.testing.assert_close(
            ref_map.cpu(),
            torch.zeros_like(logits_cpu, dtype=torch.bool).scatter(1, expected_ids, True),
        )
        for virtual in ((False, True) if compact_supported and not expert_bias else (False,)):
            router = TopKRouter(replace(config, moe_virtual_expert_load_balance=virtual), pg).cuda()
            router.set_layer_number(1)
            if quantile or expert_bias:
                (router.qb_beta if quantile else router.expert_bias).copy_(bias)
            logits = logits_cpu.cuda().requires_grad_()
            probs, ids = router.routing(logits.view(256, 1, 512))
            compact = virtual or (quantile and compact_supported)
            assert (ids.dtype != torch.bool) == compact
            if not compact:
                assert ids.shape == probs.shape == (256, 512)
                ids = ids.to(torch.int8).topk(10, dim=1).indices
                probs = probs.gather(1, ids)
            assert probs.shape == ids.shape == (256, 10) and ids.dtype == torch.int64
            sorted_ids, order = ids.sort(dim=1)
            torch.testing.assert_close(sorted_ids.cpu(), expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(
                probs.gather(1, order).cpu(), expected_probs, rtol=1e-5, atol=1e-7
            )
            assert ids.max() > 64
            if quantile or expert_bias:
                assert not probs[0].any()
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
                router.config.moe_router_fusion = True
                with pytest.raises(AssertionError, match="does not support moe_router_fusion"):
                    router.routing(logits.view(256, 1, 512))
            elif expert_bias:
                torch.testing.assert_close(
                    router.local_tokens_per_expert, ref_map.sum(dim=0).float(), rtol=0, atol=0
                )
        for virtual, shape, index_buffer in calls:
            if not virtual:
                assert shape == (256, 512) and index_buffer is None
            else:
                assert shape == (256, 512) and index_buffer.shape == (256, 10)
        assert bool(calls) == config.moe_router_fusion
    finally:
        from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker

        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
