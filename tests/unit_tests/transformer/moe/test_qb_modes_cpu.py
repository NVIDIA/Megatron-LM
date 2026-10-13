# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU numerical and lifecycle coverage for the two inherited QB estimators."""

import argparse
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import (
    finalize_model_grads,
    reset_model_temporary_tensors,
)
from megatron.core.transformer.moe.paged_stash import PagedStashRunner
from megatron.core.transformer.moe.router import InferenceTopKRouter, TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import add_megatron_arguments
from megatron.training.checkpointing import CheckpointType, load_args_from_checkpoint


def _config(scope, **kwargs):
    return TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_router_load_balancing_type="quantile_balancing",
        moe_router_score_function="sigmoid",
        moe_aux_loss_coeff=0.0,
        moe_router_quantile_balancing_estimation_scope=scope,
        **kwargs,
    )


def _groups():
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    return SimpleNamespace(tp=group, cp=group, expt_tp=group, tp_cp=group, tp_dp_cp=group)


@pytest.mark.parametrize("scope", ["global_batch", "micro_batch"])
def test_qb_mode_state_dict_round_trip(scope, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    config = _config(scope)
    router = TopKRouter(config, pg_collection=_groups())
    key = "expert_bias" if scope == "global_batch" else "qb_beta"
    getattr(router, key).copy_(torch.tensor([0.2, -0.1, 0.4, -0.5]))
    state = router.state_dict()
    expected = (
        {"weight", "bias", "expert_bias", "qb_bin_bounds"}
        if scope == "global_batch"
        else {"weight", "bias", "qb_beta"}
    )
    assert set(state) == expected
    restored = TopKRouter(config, pg_collection=_groups())
    restored.load_state_dict(state, strict=True)
    torch.testing.assert_close(getattr(restored, key), getattr(router, key))


def test_micro_batch_routing_ema_and_retry_lifecycle(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    config = _config("micro_batch", moe_router_quantile_balancing_ema=0.25)
    router = TopKRouter(config, pg_collection=_groups())
    beta = torch.tensor([0.2, -0.1, 0.4, -0.5])
    router.qb_beta.copy_(beta)
    logits = torch.tensor(
        [[0.1, 0.8, 0.3, -0.2], [0.7, 0.5, -0.9, 0.6], [-0.3, 0.9, 0.4, 0.2], [0.6, -0.5, 0.3, 0.1]]
    )
    expected_indices = (logits - beta).topk(2, dim=1).indices
    expected_map = torch.zeros_like(logits, dtype=torch.bool).scatter(1, expected_indices, True)
    corrections = (logits - beta).sort(dim=1, descending=True).values[:, 2:3]
    expected_quantile = (logits - corrections).sort(dim=0, descending=True).values[2]
    _, routing_map = router.routing(logits.reshape(4, 1, 4))
    assert torch.equal(routing_map, expected_map)
    torch.testing.assert_close(router.qb_beta_accum, expected_quantile)
    assert router.qb_beta_count.item() == 1
    torch.testing.assert_close(router.qb_beta, beta)
    model = torch.nn.Module()
    model.router = router
    model.config = config
    synced = []
    model.finish_grad_sync = lambda **kwargs: synced.append(kwargs)
    calls = []
    group = _groups().tp
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda value, op, group: calls.append(group)
    )
    # The exact estimator must work with main's DP/CP group contract; it does not
    # require the histogram estimator's extra tp_dp_cp group.
    pg_collection = SimpleNamespace(tp=group, pp=group, dp_cp=group, embd=None, pos_embd=None)
    # Process-group geometry and transport are explicit CPU fixtures here.
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    finalize_model_grads([model], pg_collection=pg_collection)
    assert synced == [{"force_all_reduce": False}]
    expected_beta = 0.25 * beta + 0.75 * expected_quantile
    expected_beta -= expected_beta.mean()
    assert calls == [group]
    torch.testing.assert_close(router.qb_beta, expected_beta)
    reset_model_temporary_tensors(config, [model])
    assert router.qb_beta_count.item() == 0
    assert not router.qb_beta_accum.any()
    router.qb_beta_accum.fill_(4)
    router.qb_beta_count.fill_(3)
    runner = PagedStashRunner.__new__(PagedStashRunner)
    runner.moe_layers = [SimpleNamespace(router=router)]
    ptr = router.qb_beta_accum.data_ptr()
    runner._reset_qb_histograms()
    assert router.qb_beta_accum.data_ptr() == ptr
    assert not router.qb_beta_accum.any()
    assert router.qb_beta_count.item() == 0
    torch.testing.assert_close(router.qb_beta, expected_beta)


def test_micro_batch_inference_uses_frozen_beta(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    router = InferenceTopKRouter(_config("micro_batch"), pg_collection=_groups())
    beta = torch.tensor([0.2, -0.1, 0.4, -0.5])
    router.qb_beta.copy_(beta)
    logits = torch.arange(16, dtype=torch.float32).reshape(4, 1, 4) / 10
    monkeypatch.setattr(router, "gating", lambda value: logits)
    probs, indices = router._forward(torch.zeros(4, 1, 8))
    expected = (logits.squeeze(1) - beta).topk(2, dim=1).indices
    assert torch.equal(indices, expected)
    scores = logits.squeeze(1).sigmoid().gather(1, expected)
    torch.testing.assert_close(probs, scores / scores.sum(dim=1, keepdim=True))
    assert router.qb_beta_count.item() == 0
    torch.testing.assert_close(router.qb_beta, beta)


def test_cli_parses_micro_batch_ema():
    parser = argparse.ArgumentParser()
    add_megatron_arguments(parser)
    args = parser.parse_args(
        [
            "--moe-router-quantile-balancing-estimation-scope",
            "micro_batch",
            "--moe-router-quantile-balancing-ema",
            "0.25",
        ]
    )
    assert args.moe_router_quantile_balancing_estimation_scope == "micro_batch"
    assert args.moe_router_quantile_balancing_ema == 0.25


def test_main_checkpoint_without_scope_restores_exact_estimator():
    checkpoint_args = SimpleNamespace(
        moe_router_load_balancing_type="quantile_balancing",
        moe_aux_loss_coeff=0.0,
        moe_router_quantile_balancing_ema=0.25,
    )
    args = SimpleNamespace(
        load="checkpoint",
        iteration=0,
        moe_router_quantile_balancing_estimation_scope="global_batch",
        use_tokenizer_model_from_checkpoint_args=False,
        use_mp_args_from_checkpoint_args=False,
    )
    with mock.patch(
        "megatron.training.checkpointing._load_base_checkpoint",
        return_value=(
            {"args": checkpoint_args, "iteration": 12},
            "checkpoint",
            False,
            CheckpointType.LEGACY,
        ),
    ):
        restored, _ = load_args_from_checkpoint(args)
    assert restored.moe_router_quantile_balancing_estimation_scope == "micro_batch"
    assert restored.moe_router_quantile_balancing_ema == 0.25
