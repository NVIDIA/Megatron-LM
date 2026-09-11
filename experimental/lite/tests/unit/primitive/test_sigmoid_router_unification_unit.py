# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise contracts for the shared DeepSeek-family sigmoid router."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


def _config(*, grouped: bool, aux_loss_alpha: float | None):
    values = dict(
        hidden_size=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        scoring_func="sigmoid",
    )
    if grouped:
        values.update(n_group=2, topk_group=1)
    if aux_loss_alpha is not None:
        values["aux_loss_alpha"] = aux_loss_alpha
    return SimpleNamespace(**values)


def _ps():
    return SimpleNamespace(tp_size=1, tp_group=None)


def _legacy_custom_router_output(config, weight, expert_bias, x):
    from megatron.lite.primitive.utils.moe import topk_routing_with_score_function

    logits = F.linear(x.to(torch.float32), weight.to(torch.float32))
    kwargs = {}
    num_groups = getattr(config, "n_group", None)
    group_topk = getattr(config, "topk_group", None)
    if num_groups is not None and group_topk is not None:
        kwargs.update(num_groups=num_groups, group_topk=group_topk)
    probs_dense, routing_map = topk_routing_with_score_function(
        logits,
        config.num_experts_per_tok,
        score_function="sigmoid",
        expert_bias=expert_bias.to(logits.dtype),
        scaling_factor=config.routed_scaling_factor,
        **kwargs,
    )
    expert_ids = torch.arange(8).expand_as(routing_map)
    masked_ids = torch.where(routing_map, expert_ids, torch.full_like(expert_ids, 8))
    indices = torch.sort(masked_ids, dim=-1).values[:, :2]
    return torch.gather(probs_dense, 1, indices), indices, routing_map


@pytest.mark.parametrize(
    ("model_name", "aux_loss_alpha"),
    [("glm5", None), ("kimi_k2", 0.0)],
)
def test_shared_router_is_bitwise_equal_to_legacy_custom_router(
    model_name, aux_loss_alpha, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    del model_name
    config = _config(grouped=True, aux_loss_alpha=aux_loss_alpha)
    router = SigmoidTopKRouter(
        config,
        _ps(),
        compute_aux_loss=False,
        router_dtype=torch.float32,
        expert_bias_persistent=True,
    ).to(torch.bfloat16)
    weight = (torch.arange(64, dtype=torch.float32).view(8, 8) / 17).to(torch.bfloat16)
    bias = torch.linspace(-0.25, 0.25, 8)
    x = (torch.arange(24, dtype=torch.float32).view(3, 8) / 13).to(torch.bfloat16)
    with torch.no_grad():
        router.gate.weight.copy_(weight)
        router.expert_bias.copy_(bias)

    expected_scores, expected_indices, _ = _legacy_custom_router_output(
        config, weight, bias, x
    )
    actual_scores, actual_indices = router(x)

    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_indices, expected_indices)
    assert "expert_bias" in router.state_dict()


@pytest.mark.parametrize("model_name", ["deepseek_v4"])
def test_ungrouped_router_output_is_bitwise_unchanged(
    model_name, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    del model_name
    config = _config(grouped=False, aux_loss_alpha=0.0)
    router = SigmoidTopKRouter(config, _ps(), compute_aux_loss=False)
    weight = torch.arange(64, dtype=torch.float32).view(8, 8) / 19
    bias = torch.linspace(-0.125, 0.125, 8)
    x = torch.arange(32, dtype=torch.float32).view(4, 8) / 11
    with torch.no_grad():
        router.gate.weight.copy_(weight)
        router.expert_bias.copy_(bias)

    expected_scores, expected_indices, _ = _legacy_custom_router_output(
        config, weight, bias, x
    )
    actual_scores, actual_indices = router(x)

    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_indices, expected_indices)


@pytest.mark.parametrize("model_name", ["qwen3_moe", "qwen3_5"])
def test_qwen_topk_router_output_is_bitwise_unchanged(
    model_name, transformer_engine_import_stub
):
    del model_name
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import TopKRouter
    from megatron.lite.primitive.utils.moe import topk_routing_with_score_function

    config = SimpleNamespace(
        hidden_size=8,
        num_experts=8,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.0,
    )
    router = TopKRouter(config, _ps(), compute_aux_loss=False)
    weight = torch.arange(64, dtype=torch.float32).view(8, 8) / 23
    x = torch.arange(32, dtype=torch.float32).view(4, 8) / 7
    with torch.no_grad():
        router.gate.weight.copy_(weight)
    logits = F.linear(x, weight)
    probs_dense, routing_map = topk_routing_with_score_function(
        logits,
        2,
        score_function="softmax",
    )
    expert_ids = torch.arange(8).expand_as(routing_map)
    masked_ids = torch.where(routing_map, expert_ids, torch.full_like(expert_ids, 8))
    expected_indices = torch.sort(masked_ids, dim=-1).values[:, :2]
    expected_scores = torch.gather(probs_dense, 1, expected_indices)

    actual_scores, actual_indices = router(x)

    assert torch.equal(actual_scores, expected_scores)
    assert torch.equal(actual_indices, expected_indices)


@pytest.mark.parametrize("model_name", ["glm5", "kimi_k2"])
def test_group_limited_routing_changes_selected_experts(
    model_name, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    del model_name
    grouped = _config(grouped=True, aux_loss_alpha=0.0)
    ungrouped = _config(grouped=False, aux_loss_alpha=0.0)
    grouped_router = SigmoidTopKRouter(grouped, _ps(), compute_aux_loss=False)
    ungrouped_router = SigmoidTopKRouter(ungrouped, _ps(), compute_aux_loss=False)
    weight = torch.eye(8)
    x = torch.tensor([[10.0, 0.0, 0.0, 0.0, 9.0, 8.0, 0.0, 0.0]])
    with torch.no_grad():
        grouped_router.gate.weight.copy_(weight)
        ungrouped_router.gate.weight.copy_(weight)

    _, grouped_indices = grouped_router(x)
    _, ungrouped_indices = ungrouped_router(x)

    assert torch.equal(grouped_indices, torch.tensor([[4, 5]]))
    assert torch.equal(ungrouped_indices, torch.tensor([[0, 4]]))


def test_router_buffers_remain_float32_after_dtype_apply(
    transformer_engine_import_stub,
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter

    router = SigmoidTopKRouter(
        _config(grouped=True, aux_loss_alpha=0.0),
        _ps(),
        compute_aux_loss=False,
    ).to(torch.bfloat16)

    assert router.expert_bias.dtype == torch.float32
