# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Mock routing controls pre-balance rank skew without changing top-k cardinality."""

import argparse
import dataclasses

import pytest
import torch

from megatron.core.transformer.moe.mock_router import (
    MockRouteCache,
    mock_route_scores,
    validate_mock_route_parameters,
)
from megatron.core.transformer.moe.moonep_moe_scheduler import extract_semantic_routes
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import ArgumentGroupFactory

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize('ep,experts,topk', [(1, 8, 3), (2, 8, 1), (3, 12, 7), (16, 64, 6)])
@pytest.mark.parametrize('fraction', [0.0, 0.37, 1.0])
def test_rank_skew_and_distinct_experts_for_every_prefix(ep, experts, topk, fraction):
    upper = ep * min(topk, experts // ep) / topk
    target = 1 + fraction * (upper - 1)
    ids = MockRouteCache(experts, topk, ep, target).get(113, 'cpu')
    assert ids.dtype == torch.int64
    assert ((ids >= 0) & (ids < experts)).all()
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
    for n in range(1, 114):
        counts = torch.bincount((ids[:n] // (experts // ep)).flatten(), minlength=ep)
        assert counts.sum() == n * topk
        actual = counts.max().item() * ep / (n * topk)
        assert abs(actual - target) <= ep / (n * topk) + 1e-12


def test_current_recipe_exact_skew_and_cache_reuse():
    cache = MockRouteCache(64, 6, 16, 2.0)
    rng_before = torch.get_rng_state().clone()
    ids = cache.get(4096, 'cpu')
    assert cache.get(4096, torch.device('cpu')) is ids
    assert cache.get(17, 'cpu').shape == (17, 6)
    torch.testing.assert_close(cache.get(17, 'cpu'), ids[:17])
    counts = torch.bincount((ids // 4).flatten(), minlength=16)
    assert counts.max().item() / counts.float().mean().item() == 2.0
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    with torch.inference_mode():
        inference_warmed = cache.get(18, 'cpu')
    assert not inference_warmed.is_inference()


def test_cli_exposes_optional_maxvio():
    parser = argparse.ArgumentParser()
    exclude = [
        f.name for f in dataclasses.fields(TransformerConfig) if f.name != 'moe_router_mock_maxvio'
    ]
    ArgumentGroupFactory(TransformerConfig, exclude=exclude).build_group(parser)
    assert parser.parse_args([]).moe_router_mock_maxvio is None
    assert parser.parse_args(['--moe-router-mock-maxvio', '2.5']).moe_router_mock_maxvio == 2.5


@pytest.mark.parametrize('value', [0.0, 0.999, float('nan'), float('inf'), 11.0])
def test_invalid_maxvio(value):
    with pytest.raises(ValueError, match='maxvio'):
        validate_mock_route_parameters(64, 6, 16, value)


@pytest.mark.parametrize('experts,topk,ep', [(7, 2, 2), (8, 0, 2), (8, 9, 2), (8, 2, 0)])
def test_invalid_layout(experts, topk, ep):
    with pytest.raises(ValueError):
        validate_mock_route_parameters(experts, topk, ep, 1.0)


def _config(**kwargs):
    args = dict(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_moe_experts=8,
        moe_ffn_hidden_size=32,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        moe_router_topk=2,
        moe_router_mock_maxvio=1.5,
        moe_router_score_function='sigmoid',
        moe_aux_loss_coeff=0.1,
        moe_router_load_balancing_type='seq_aux_loss',
        cuda_graph_scope=None,
    )
    args.update(kwargs)
    return TransformerConfig(**args)


@pytest.mark.parametrize(
    'conflict',
    [
        {'moe_router_force_load_balancing': True},
        {'moe_router_force_biased': 0.1},
        {'moe_enable_routing_replay': True},
        {'moe_router_enable_expert_bias': True},
        {'moe_router_num_groups': 2, 'moe_router_group_topk': 1},
        {'moe_expert_capacity_factor': 1.0},
        {'expert_tensor_parallel_size': 2},
        {'moe_router_load_balancing_type': 'sinkhorn'},
    ],
)
def test_config_rejects_conflicting_routing_modes(conflict):
    with pytest.raises(ValueError, match='moe_router_mock_maxvio'):
        _config(**conflict)


def _router(scheduler=None, hash_layer=False, **config_overrides):
    # Exercise real routing() without requiring a gate projection or process groups.
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    config_args = dict(
        moe_token_dispatcher_type='flex' if scheduler else 'alltoall',
        moe_flex_dispatcher_backend='hybridep',
        moe_enable_scheduler=scheduler is not None,
        moe_scheduler_planner_type=scheduler or 'eplb',
        moe_scheduler_num_idle_experts=config_overrides.get('num_moe_experts', 8),
        moe_scheduler_expert_dispatcher_type='replica_nccl',
        hidden_size=128,
        moe_ffn_hidden_size=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        use_transformer_engine_op_fuser=True,
        gradient_accumulation_fusion=True,
        moe_router_dtype='fp32',
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
    )
    config_args.update(config_overrides)
    router.config = _config(**config_args)
    router.topk = router.config.moe_router_topk
    router.routing_type = 'seq_aux_loss'
    router.score_function = 'sigmoid'
    router.is_hash_layer = hash_layer
    router.enable_expert_bias = False
    router.use_micro_batch_quantile_balancing = False
    router.use_quantile_balancing = False
    router.mock_route_cache = MockRouteCache(
        router.config.num_moe_experts,
        router.topk,
        router.config.expert_model_parallel_size,
        router.config.moe_router_mock_maxvio,
    )
    return router


@pytest.mark.parametrize('scheduler', [None, 'echo', 'eplb', 'moon_ep'])
@pytest.mark.parametrize('hash_layer', [False, True])
@pytest.mark.parametrize('padding', [None, [False] * 12, [True, False, False] * 4, [True] * 12])
def test_router_padding_cache_and_gradients(scheduler, hash_layer, padding):
    router = _router(scheduler, hash_layer)
    mask = None if padding is None else torch.tensor(padding, dtype=torch.bool)
    logits = torch.randn(12, 1, 8, requires_grad=True)
    probs, route = router.routing(logits, padding_mask=mask)  # No input_ids even on Hash-MoE.
    valid = torch.ones(12, dtype=torch.bool) if mask is None else ~mask
    n = valid.sum().item()
    cached = router.mock_route_cache.get(12, 'cpu').clone()
    expected = torch.zeros((n, 8), dtype=torch.bool).scatter(1, cached[:n], True)
    torch.testing.assert_close(route[valid], expected)
    if scheduler:
        assert (route.sum(dim=1) == router.topk).all()
        _, ids = extract_semantic_routes(route, probs, router.topk)
        assert ((ids >= 0) & (ids < 8)).all()
    else:
        assert not route[~valid].any()
    assert not probs[~valid].any()
    (probs * torch.arange(probs.shape[-1])).sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[~valid].any()
    if n:
        assert logits.grad[valid].abs().sum() > 0
    route.fill_(0)  # A consumer cannot corrupt the shared cache.
    torch.testing.assert_close(router.mock_route_cache.get(12, 'cpu'), cached)


@pytest.mark.parametrize('target', [1.0, 5.0, 10.0])
@pytest.mark.parametrize('padding_count', [0, 1, 19, 4096])
def test_moonep_recipe_padding_preserves_planner_cardinality_and_valid_skew(target, padding_count):
    """THD/CP can pad just the last source rank; do not send -1 into the planner."""
    router = _router(
        scheduler='moon_ep',
        num_moe_experts=64,
        expert_model_parallel_size=16,
        moe_router_topk=6,
        moe_router_mock_maxvio=target,
    )
    mask = torch.arange(4096) >= 4096 - padding_count
    probs, route = router.routing(torch.randn(4096, 1, 64), padding_mask=mask)
    _, ids = extract_semantic_routes(route, probs, router.topk)
    assert ((ids >= 0) & (ids < 64)).all()
    assert (ids.sort(dim=1).values.diff(dim=1) > 0).all()
    counts = torch.bincount(ids.flatten(), minlength=64)
    assert counts.sum().item() == 4096 * 6  # planner's fixed physical route budget
    assert not probs[mask].any()
    if padding_count < 4096:
        valid_counts = torch.bincount((ids[~mask] // 4).flatten(), minlength=16)
        actual = valid_counts.max().item() * 16 / ((4096 - padding_count) * 6)
        assert abs(actual - target) <= 16 / ((4096 - padding_count) * 6) + 1e-12


@pytest.mark.parametrize('scheduler', [None, 'moon_ep'])
def test_empty_tokens_and_invalid_padding(scheduler):
    router = _router(scheduler)
    probs, route = router.routing(
        torch.empty(0, 1, 8), padding_mask=torch.empty(0, dtype=torch.bool)
    )
    assert probs.shape == route.shape == (0, 8)
    with pytest.raises(ValueError, match='boolean padding entry'):
        router.routing(torch.zeros(12, 1, 8), padding_mask=torch.zeros(11, dtype=torch.bool))
    with pytest.raises(ValueError, match='boolean padding entry'):
        router.routing(torch.zeros(12, 1, 8), padding_mask=torch.zeros(12, dtype=torch.int32))


def test_disabled_mock_preserves_learned_selection():
    router = _router()
    router.config.moe_router_mock_maxvio = None
    router.mock_route_cache = None
    router.routing_type = 'none'
    router.expert_bias = None
    router.router_replay = None
    logits = torch.arange(96).view(12, 1, 8).float() / 100
    probs, route = router.routing(logits)
    expected = torch.zeros(12, 8, dtype=torch.bool)
    expected[:, -2:] = True
    torch.testing.assert_close(route, expected)
    assert probs.shape == (12, 8)


@pytest.mark.parametrize('score', ['softmax', 'sigmoid', 'sqrtsoftplus'])
@pytest.mark.parametrize('hash_layer', [False, True])
@pytest.mark.parametrize('topk', [1, 3])
@pytest.mark.parametrize('pre_softmax', [False, True])
def test_scores_are_fresh_and_match_selected_expert_weights(score, hash_layer, topk, pre_softmax):
    ids = MockRouteCache(8, topk, 2, 1.25).get(7, 'cpu')
    for offset in (0.0, 2.0):
        logits = (torch.arange(56).view(7, 8).float() / 10 + offset).requires_grad_()
        probs = mock_route_scores(logits, ids, score, pre_softmax, 1.5, hash_layer)
        if score == 'softmax':
            expected = (
                logits.softmax(-1).gather(1, ids)
                if pre_softmax or hash_layer
                else logits.gather(1, ids).softmax(-1)
            )
        else:
            scores = (
                logits.sigmoid()
                if score == 'sigmoid'
                else torch.nn.functional.softplus(logits).sqrt()
            )
            expected = scores.gather(1, ids)
            if topk > 1 or hash_layer:
                expected = expected / expected.sum(-1, keepdim=True)
        torch.testing.assert_close(probs, expected * 1.5)
        (probs * torch.arange(topk)).sum().backward()
        assert torch.isfinite(logits.grad).all()


def test_cache_miss_during_capture_fails_before_cpu_build(monkeypatch):
    cache = MockRouteCache(8, 2, 2, 1.5)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    with pytest.raises(RuntimeError, match='warm this token shape'):
        cache.get(12, 'cuda:0')


@pytest.mark.parametrize('scheduler', [None, 'echo', 'eplb', 'moon_ep'])
def test_cuda_graph_replay_refreshes_padding_and_scores(scheduler):
    router = _router(scheduler)
    logits = torch.randn(12, 1, 8, device='cuda')
    mask = torch.zeros(12, dtype=torch.bool, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            router.routing(logits, padding_mask=mask)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        probs, route = router.routing(logits, padding_mask=mask)
    for padding in ([True, False, False] * 4, [True] * 12, [False] * 12):
        mask.copy_(torch.tensor(padding, device='cuda'))
        logits.normal_()
        expected_probs, expected_route = router.routing(logits, padding_mask=mask)
        graph.replay()
        torch.testing.assert_close(probs, expected_probs)
        torch.testing.assert_close(route, expected_route)
        if scheduler:
            assert (route.sum(dim=1) == router.topk).all()
            assert not probs[mask].any()
