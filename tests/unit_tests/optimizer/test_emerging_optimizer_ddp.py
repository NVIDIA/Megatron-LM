# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizers need all-reduced gradients unless their state is layer-wise."""

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


def _model(distributed, bf16=True, instances=1, expert=False):
    config = TransformerConfig(num_layers=1, num_attention_heads=1, hidden_size=8)
    dtype = torch.bfloat16 if bf16 else torch.float32
    torch.manual_seed(123)
    module = torch.nn.Linear(8, 8, device='cuda', dtype=dtype)
    if expert:
        for param in module.parameters():
            param.allreduce = False
    return DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=distributed, num_distributed_optimizer_instances=instances
        ),
        module,
    )


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='emerging-optimizers is required')
@pytest.mark.parametrize('optimizer', ['muon', 'adaptive_muon', 'soap', 'lion'])
@pytest.mark.parametrize('optimizer_dist', [False, True])
@pytest.mark.parametrize('sharded_chunk', [0, 1])
def test_reject_reduce_scatter_without_layer_wise(optimizer, optimizer_dist, sharded_chunk):
    """The actual DDP layout matters, including mismatched flags and later chunks."""
    if Utils.world_size < 2:
        pytest.skip('Partial gradients require more than one DP rank')
    models = [_model(False) for _ in range(sharded_chunk)] + [_model(True)]
    config = OptimizerConfig(
        optimizer=optimizer, lr=0.01, bf16=True, use_distributed_optimizer=optimizer_dist
    )
    with pytest.raises(ValueError, match=f"model chunk {sharded_chunk}.*reduce-scatter"):
        get_megatron_optimizer(config, models)


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='emerging-optimizers is required')
def test_reject_expert_reduce_scatter_without_layer_wise():
    """Expert buffers must be checked even when no dense parameters are present."""
    if Utils.world_size < 2:
        pytest.skip('Partial gradients require more than one expert-DP rank')
    model = _model(True, expert=True)
    assert not model.bucket_groups
    assert model.expert_parallel_bucket_groups
    with pytest.raises(ValueError, match='model chunk 0.*reduce-scatter'):
        get_megatron_optimizer(OptimizerConfig(optimizer='muon', lr=0.01, bf16=True), [model])


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
def test_layer_wise_muon_remains_supported(legacy_alias, layout):
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


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='emerging-optimizers is required')
@pytest.mark.parametrize('singleton_instances', [False, True])
def test_singleton_reduce_scatter_remains_supported(singleton_instances):
    """A one-rank reduce-scatter shard contains the entire gradient buffer."""
    instances = Utils.world_size if singleton_instances else 1
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1 if singleton_instances else Utils.world_size,
        num_distributed_optimizer_instances=instances,
    )
    model = _model(True, instances=instances)
    config = OptimizerConfig(optimizer='muon', lr=0.01, bf16=True, clip_grad=0.0)
    optimizer = get_megatron_optimizer(config, [model])
    torch.manual_seed(789 + Utils.rank)
    model(torch.randn(4, 8, device='cuda', dtype=torch.bfloat16)).float().sum().backward()
    expected = [param.main_grad.clone() for param in model.parameters()]
    if singleton_instances:
        for grad in expected:
            torch.distributed.all_reduce(grad)
    # DDP applies 1/DP scaling before its collective.
    for grad in expected:
        grad.div_(instances)
    model.finish_grad_sync()
    for param, grad in zip(model.parameters(), expected):
        torch.testing.assert_close(param.main_grad, grad, rtol=0, atol=0)
    assert optimizer.step()[0]
