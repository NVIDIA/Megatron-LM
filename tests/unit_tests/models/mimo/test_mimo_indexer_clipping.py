# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Separate indexer clipping through MIMO's cross-grid optimizer step."""

import math

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.mimo.optimizer import MimoOptimizer, ModuleOptimizerInfo
from megatron.core.optimizer.optimizer import ChainedOptimizer, FP32Optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.tensor_parallel import layers
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def module_groups(monkeypatch):
    """Split the world into an encoder grid and a language grid with explicit groups."""
    if not torch.cuda.is_available() or Utils.world_size < 4 or Utils.world_size % 2:
        pytest.skip("Requires an even distributed world size >= 4")
    Utils.initialize_distributed()
    assert not parallel_state.model_parallel_is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    size = world_size // 2
    groups = [
        torch.distributed.new_group(list(range(0, size))),
        torch.distributed.new_group(list(range(size, world_size))),
    ]
    singletons = [torch.distributed.new_group([r]) for r in range(world_size)]

    def no_global_groups(*_args, **_kwargs):
        pytest.fail("MIMO clipping must use explicit process groups")

    for name in (
        "get_model_parallel_group",
        "get_tensor_model_parallel_group",
        "get_context_parallel_group",
    ):
        monkeypatch.setattr(parallel_state, name, no_global_groups)
    for name in (
        "get_tensor_model_parallel_rank",
        "get_gtp_weight_remat_rank",
        "get_expert_gtp_weight_remat_rank",
    ):
        monkeypatch.setattr(layers, name, no_global_groups)
    try:
        yield rank >= size, groups[rank // size], singletons[rank]
    finally:
        for group in reversed(groups + singletons):
            if group != torch.distributed.GroupMember.NON_GROUP_MEMBER:
                torch.distributed.destroy_process_group(group)
        Utils.destroy_model_parallel()


def _parameter(gradient):
    param = torch.nn.Parameter(torch.zeros(1, device="cuda"))
    param.grad = torch.full_like(param, gradient)
    return param


def _build_optimizer(module_groups, config, frozen_encoder):
    """Main gradients have global norm 5 and the indexer gradient has norm 12 (total 13)."""
    language_rank, stats_group, singleton = module_groups

    def leaf(params, is_indexer):
        groups = [{"params": params, "is_dsa_indexer": is_indexer}] if params else []
        inner = torch.optim.SGD(groups, lr=1.0) if groups else None
        opt = FP32Optimizer(inner, config, init_state_fn=lambda _: None)
        opt.grad_stats_parallel_group = stats_group
        opt.tp_group = opt.expert_tp_group = singleton
        opt.gtp_group = opt.expert_gtp_group = singleton
        return opt

    opt = None
    params, indexer_params = [], []
    active = language_rank or not frozen_encoder
    if active:
        # Main gradients are spread over the module grid; the encoder contributes only when
        # it trains.
        main_norm = 5.0 if frozen_encoder else (3.0 if language_rank else 4.0)
        params.append(_parameter(main_norm / math.sqrt(stats_group.size())))
        if language_rank and stats_group.rank() == stats_group.size() - 1:
            # Other language ranks hold an empty local indexer shard.
            indexer_params.append(_parameter(12.0))
        opt = (
            ChainedOptimizer([leaf(params, False), leaf(indexer_params, True)])
            if language_rank
            else leaf(params, False)
        )

    infos = {
        "encoder": ModuleOptimizerInfo(
            opt if not language_rank else None, None, None, active and not language_rank
        ),
        "language": ModuleOptimizerInfo(opt if language_rank else None, None, None, language_rank),
    }
    all_params = params + indexer_params
    gradients = [param.grad.detach().clone() for param in all_params]
    return MimoOptimizer(infos, config), all_params, gradients, set(map(id, indexer_params))


def _scale(limit, norm):
    return min(1.0, limit / (norm + 1e-6)) if limit > 0 else 1.0


@pytest.mark.parametrize("frozen_encoder", [True, False])
@pytest.mark.parametrize(
    "main_clip, indexer_clip", [(1.0, 1.0), (1.0, 0.25), (0.0, 0.25), (1.0, 0.0)]
)
def test_separate_clipping_across_disjoint_grids(
    module_groups, frozen_encoder, main_clip, indexer_clip
):
    config = OptimizerConfig(
        optimizer="sgd", lr=1.0, clip_grad=main_clip, dsa_indexer_clip_grad=indexer_clip
    )
    optimizer, params, gradients, indexer_ids = _build_optimizer(
        module_groups, config, frozen_encoder
    )
    for _ in range(2):
        before = [param.detach().clone() for param in params]
        for param, gradient in zip(params, gradients):
            param.grad = gradient.clone()
        successful, norm, _ = optimizer.step()
        assert successful
        assert norm == pytest.approx(13.0, rel=1e-6)
        assert optimizer._last_dsa_split_grad_norms == pytest.approx((12.0, 5.0), rel=1e-6)
        for param, initial, gradient in zip(params, before, gradients):
            if id(param) in indexer_ids:
                scale = _scale(indexer_clip, 12.0)
            else:
                scale = _scale(main_clip, 5.0)
            torch.testing.assert_close(param, initial - gradient * scale)


@pytest.mark.parametrize("frozen_encoder", [True, False])
def test_shared_clipping_is_unchanged(module_groups, monkeypatch, frozen_encoder):
    config = OptimizerConfig(optimizer="sgd", lr=1.0, clip_grad=1.0)
    optimizer, params, gradients, _ = _build_optimizer(module_groups, config, frozen_encoder)

    def no_split_norms():
        pytest.fail("Joint clipping must not add split-norm collectives")

    monkeypatch.setattr(optimizer, "get_dsa_split_grad_norms", no_split_norms)
    successful, norm, _ = optimizer.step()
    assert successful
    assert norm == pytest.approx(13.0, rel=1e-6)
    assert optimizer._last_dsa_split_grad_norms is None
    for param, gradient in zip(params, gradients):
        torch.testing.assert_close(param, -gradient * _scale(1.0, 13.0))
