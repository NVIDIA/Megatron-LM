# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
from types import SimpleNamespace

import pytest
import torch

from megatron.core.extensions.transformer_engine import fused_topk_with_score_function_supports_qb
from megatron.core.distributed.finalize_model_grads import (
    _update_router_qb_histogram,
    reset_model_temporary_tensors,
)
from megatron.core.transformer.moe.moe_utils import (
    get_updated_expert_bias_with_quantile,
    topk_routing_with_score_function,
)
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_quantile_balancing_estimation_scope="global_batch",
        moe_aux_loss_coeff=0.0,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_qb_global_batch_config():
    config = _config()
    assert config.moe_router_quantile_balancing_estimation_scope == "global_batch"


def test_qb_cli_exposes_global_batch_scope_and_histogram_bins():
    from megatron.training.arguments import _add_moe_args

    parser = argparse.ArgumentParser()
    _add_moe_args(parser)
    args = parser.parse_args(
        [
            "--moe-router-load-balancing-type",
            "quantile_balancing",
            "--moe-router-quantile-balancing-estimation-scope",
            "global_batch",
            "--moe-router-qb-num-bins",
            "257",
        ]
    )

    assert args.moe_router_load_balancing_type == ["quantile_balancing"]
    assert args.moe_router_quantile_balancing_estimation_scope == "global_batch"
    assert args.moe_router_qb_num_bins == 257


def test_qb_main_accepts_micro_batch_scope():
    config = _config(moe_router_quantile_balancing_estimation_scope="micro_batch")
    assert config.moe_router_quantile_balancing_estimation_scope == "micro_batch"


def test_qb_main_defaults_to_micro_batch_scope():
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="quantile_balancing",
        moe_aux_loss_coeff=0.0,
    )
    assert config.moe_router_quantile_balancing_estimation_scope == "micro_batch"


def test_qb_rejects_nonzero_aux_loss():
    with pytest.raises(ValueError, match="moe_aux_loss_coeff=0"):
        _config(moe_aux_loss_coeff=0.01)


def test_qb_histogram_recovery():
    histogram = torch.tensor([[0, 2, 2, 0], [2, 2, 0, 0]], dtype=torch.int32)
    bin_bounds = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    expert_bias = torch.zeros(2, dtype=torch.float32)

    updated_bias, updated_bounds = get_updated_expert_bias_with_quantile(
        histogram, bin_bounds, expert_bias, topk=1
    )

    torch.testing.assert_close(updated_bias, torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(updated_bounds, torch.tensor([-1.25, 1.25]))


def test_qb_uses_pooled_global_batch_quantile_not_mean_microbatch_quantile():
    bounds = torch.tensor([0.0, 10.0], dtype=torch.float32)
    bias = torch.zeros(2, dtype=torch.float32)
    mb1 = torch.zeros(2, 10, dtype=torch.int32)
    mb2 = torch.zeros_like(mb1)
    mb1[0, 0], mb1[0, 9], mb1[1, 0] = 3, 1, 4
    mb2[0, 9], mb2[1, 0] = 4, 4

    pooled_bias, _ = get_updated_expert_bias_with_quantile(mb1 + mb2, bounds, bias, topk=1)
    mb1_bias, _ = get_updated_expert_bias_with_quantile(mb1, bounds, bias, topk=1)
    mb2_bias, _ = get_updated_expert_bias_with_quantile(mb2, bounds, bias, topk=1)

    assert not torch.allclose(pooled_bias, (mb1_bias + mb2_bias) / 2)


def test_qb_unfused_histogram_accumulates_microbatches():
    logits = torch.tensor([[-1.0, 0.5, 1.5], [2.0, -0.5, 0.25]], dtype=torch.float32)
    bias = torch.tensor([-0.2, 0.1, 0.0], dtype=torch.float32)
    bounds = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    histogram = torch.zeros(3, 8, dtype=torch.int32)

    scores = torch.sigmoid(logits)
    topk_result = torch.topk(scores + bias, 2, dim=1, sorted=True)
    cutoff = topk_result.values[:, -1:]
    expected_bins = torch.floor((cutoff - scores - bounds[0]) * (8 / (bounds[1] - bounds[0])))
    expected_bins = expected_bins.to(torch.int64).clamp_(0, 7)
    expected = torch.zeros_like(histogram)
    offsets = torch.arange(3, dtype=torch.int64) * 8
    flat_indices = (expected_bins + offsets).reshape(-1)
    expected.view(-1).scatter_add_(
        0, flat_indices, torch.ones_like(flat_indices, dtype=torch.int32)
    )

    outputs = []
    for _ in range(2):
        outputs.append(
            topk_routing_with_score_function(
                logits,
                topk=1,
                score_function="sigmoid",
                expert_bias=bias,
                fused=False,
                qb_histogram=histogram,
                qb_bin_bounds=bounds,
            )
        )

    torch.testing.assert_close(histogram, expected * 2)
    selected = topk_result.indices[:, :1]
    expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, selected, True)
    torch.testing.assert_close(outputs[0][1], expected_map)
    torch.testing.assert_close(outputs[1][1], expected_map)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_topk_with_score_function_supports_qb,
    reason="requires the Transformer Engine QB fused-router API",
)
def test_qb_fused_atomic_matches_unfused_histogram():
    torch.manual_seed(1234)
    logits = torch.randn(37, 64, device="cuda", dtype=torch.float32)
    bias = torch.linspace(-0.1, 0.1, 64, device="cuda", dtype=torch.float32)
    bounds = torch.tensor([-1.1, 1.1], device="cuda", dtype=torch.float32)
    unfused_histogram = torch.zeros(64, 128, device="cuda", dtype=torch.int32)
    fused_histogram = torch.zeros_like(unfused_histogram)

    unfused_probs, unfused_map = topk_routing_with_score_function(
        logits,
        topk=8,
        score_function="sigmoid",
        expert_bias=bias,
        fused=False,
        qb_histogram=unfused_histogram,
        qb_bin_bounds=bounds,
    )
    fused_probs, fused_map = topk_routing_with_score_function(
        logits,
        topk=8,
        score_function="sigmoid",
        expert_bias=bias,
        fused=True,
        qb_histogram=fused_histogram,
        qb_bin_bounds=bounds,
    )

    torch.testing.assert_close(fused_probs, unfused_probs)
    torch.testing.assert_close(fused_map, unfused_map)
    torch.testing.assert_close(fused_histogram, unfused_histogram)


def test_qb_finalize_updates_once_and_reset_preserves_buffers(monkeypatch):
    router = torch.nn.Module()
    router.register_buffer("qb_bias", torch.zeros(2, dtype=torch.float32))
    router.register_buffer(
        "qb_histogram",
        torch.tensor([[0, 2, 2, 0], [2, 2, 0, 0]], dtype=torch.int32),
        persistent=False,
    )
    router.register_buffer("qb_bin_bounds", torch.tensor([-1.0, 1.0], dtype=torch.float32))
    router.frozen_expert_bias = False
    model = torch.nn.Module()
    model.router = router
    config = SimpleNamespace(
        moe_router_topk=1,
        moe_router_enable_expert_bias=False,
        moe_router_load_balancing_type="quantile_balancing",
    )
    histogram_ptr = router.qb_histogram.data_ptr()
    bounds_ptr = router.qb_bin_bounds.data_ptr()

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None: tensor)
    _update_router_qb_histogram([model], config, reduction_groups=())

    torch.testing.assert_close(router.qb_bias, torch.tensor([0.25, -0.25]))
    reset_model_temporary_tensors(config, [model])
    assert torch.count_nonzero(router.qb_histogram) == 0
    assert router.qb_histogram.data_ptr() == histogram_ptr
    assert router.qb_bin_bounds.data_ptr() == bounds_ptr


def test_qb_paged_stash_retry_discards_failed_attempt_histogram():
    from megatron.core.transformer.moe.paged_stash import PagedStashRunner

    histogram = torch.ones(3, 7, dtype=torch.int32)
    histogram_ptr = histogram.data_ptr()
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.moe_layers = [SimpleNamespace(router=SimpleNamespace(qb_histogram=histogram))]

    runner._reset_qb_histograms()

    assert torch.count_nonzero(histogram) == 0
    assert histogram.data_ptr() == histogram_ptr


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_topk_with_score_function_supports_qb,
    reason="requires CUDA and the Transformer Engine QB fused-router API",
)
@pytest.mark.internal
def test_qb_mcore_router_accumulates_microbatches_and_finalizes():
    """Exercise fused QB through a real MoE router and its finalization helper."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.training.initialize import _set_random_seed
    from tests.unit_tests.test_utilities import Utils

    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(1, 1)
    _set_random_seed(seed_=123, data_parallel_random_init=False)
    config = _config(
        hidden_size=16,
        ffn_hidden_size=32,
        moe_router_topk=1,
        moe_router_fusion=True,
        moe_token_dispatcher_type="alltoall",
        moe_router_qb_num_bins=128,
        params_dtype=torch.float32,
        add_bias_linear=False,
    )
    submodules = get_submodules(
        get_gpt_layer_local_submodules(config.num_moe_experts, moe_grouped_gemm=False).mlp
    )
    assert isinstance(submodules, MoESubmodules)
    layer = MoELayer(config, submodules).cuda()
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[:, 0].copy_(torch.tensor([3.0, 1.0, -0.5, -2.0], device="cuda"))

    for direction in (1.0, -1.0):
        hidden_states = torch.zeros(64, 1, config.hidden_size, device="cuda")
        hidden_states[:, 0, 0] = direction * torch.linspace(-2.0, 2.0, 64, device="cuda")
        output, _ = layer(hidden_states)
        output.float().square().mean().backward()

    router = layer.router
    torch.testing.assert_close(
        router.qb_histogram.sum(dim=1),
        torch.full((config.num_moe_experts,), 128, dtype=torch.int64, device="cuda"),
    )
    histogram_ptr = router.qb_histogram.data_ptr()
    bounds_ptr = router.qb_bin_bounds.data_ptr()
    old_bias = router.qb_bias.clone()
    _update_router_qb_histogram([layer], config, reduction_groups=(torch.distributed.group.WORLD,))

    assert not torch.equal(router.qb_bias, old_bias)
    torch.testing.assert_close(
        router.qb_bias.mean(), torch.zeros((), device="cuda"), atol=1e-7, rtol=0
    )
    reset_model_temporary_tensors(config, [layer])
    assert torch.count_nonzero(router.qb_histogram) == 0
    assert router.qb_histogram.data_ptr() == histogram_ptr
    assert router.qb_bin_bounds.data_ptr() == bounds_ptr
    Utils.destroy_model_parallel()
