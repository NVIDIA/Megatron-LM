# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Distributed Muon requires layer-wise optimizer state."""

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import (
    HAVE_EMERGING_OPTIMIZERS,
    OptimizerConfig,
    get_megatron_optimizer,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def model_parallel():
    """Provide real DDP groups, including DP>1 under a multi-rank test launch."""
    Utils.initialize_model_parallel()
    yield
    Utils.destroy_model_parallel()


def _model(distributed, bf16=True):
    config = TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=8)
    dtype = torch.bfloat16 if bf16 else torch.float32
    torch.manual_seed(123)
    module = torch.nn.Linear(8, 8, device='cuda', dtype=dtype)
    return DistributedDataParallel(
        config, DistributedDataParallelConfig(use_distributed_optimizer=distributed), module
    )


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='emerging-optimizers is required')
@pytest.mark.parametrize('bf16', [False, True])
def test_distributed_muon_requires_layer_wise(bf16):
    """Muon cannot use the standard distributed optimizer without layer-wise mode."""
    model = _model(True, bf16)
    config = OptimizerConfig(optimizer='muon', lr=0.01, bf16=bf16, use_distributed_optimizer=True)
    with pytest.raises(
        AssertionError,
        match='Muon with use_distributed_optimizer=True requires '
        'use_layer_wise_distributed_optimizer=True',
    ):
        get_megatron_optimizer(config, [model])


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='emerging-optimizers is required')
@pytest.mark.parametrize('bf16', [False, True])
def test_non_layer_wise_muon_all_reduce_step(bf16):
    """Replicated Muon and Adam state produces identical updates across DP ranks."""
    model = _model(False, bf16)
    config = OptimizerConfig(optimizer='muon', lr=0.01, bf16=bf16, clip_grad=0.0)
    optimizer = get_megatron_optimizer(config, [model])
    assert not isinstance(optimizer, LayerWiseDistributedOptimizer)
    assert len(optimizer.chained_optimizers) == 2  # Muon matrix + Adam bias.
    before = model.module.weight.detach().clone()
    torch.manual_seed(456 + Utils.rank)
    x = torch.randn(4, 8, device='cuda', dtype=before.dtype)
    model(x).float().square().mean().backward()
    model.finish_grad_sync()
    assert optimizer.step()[0]
    for param in model.parameters():
        replicas = [torch.empty_like(param) for _ in range(Utils.world_size)]
        torch.distributed.all_gather(replicas, param)
        assert torch.isfinite(param).all()
        for replica in replicas:
            torch.testing.assert_close(param, replica, rtol=0, atol=0)
    assert not torch.equal(before, model.module.weight)


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='emerging-optimizers is required')
@pytest.mark.parametrize('legacy_alias', [False, True])
@pytest.mark.parametrize('layout', [False, True])
@pytest.mark.parametrize('optimizer_dist', [False, True])
def test_layer_wise_muon_remains_supported(legacy_alias, layout, optimizer_dist):
    """Explicit layer-wise mode and its deprecated alias accept either supported layout."""
    if layout:
        from megatron.training.training import wrap_model_chunks_with_ddp

        model_config = TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=8)
        module = torch.nn.Linear(8, 8, device='cuda', dtype=torch.bfloat16)
        models = wrap_model_chunks_with_ddp(
            [module],
            model_config,
            DistributedDataParallelConfig(),
            use_layer_wise_distributed_optimizer=True,
        )
    else:
        models = [_model(False)]
    config = OptimizerConfig(
        optimizer='dist_muon' if legacy_alias else 'muon',
        lr=0.01,
        bf16=True,
        use_layer_wise_distributed_optimizer=not legacy_alias,
        use_distributed_optimizer=optimizer_dist,
    )
    optimizer = get_megatron_optimizer(config, models)
    layer_wise = optimizer.chained_optimizers[0] if layout else optimizer
    assert isinstance(layer_wise, LayerWiseDistributedOptimizer)


@pytest.mark.parametrize(
    'optimizer_name,distributed', [('adam', False), ('adam', True), ('sgd', False)]
)
def test_standard_optimizers_remain_supported(optimizer_name, distributed):
    """Adam retains both DDP layouts and SGD retains its all-reduce path."""
    model = _model(distributed)
    config = OptimizerConfig(
        optimizer=optimizer_name, lr=0.01, bf16=True, use_distributed_optimizer=distributed
    )
    optimizer = get_megatron_optimizer(config, [model])
    assert isinstance(optimizer.chained_optimizers[0], DistributedOptimizer) == distributed
