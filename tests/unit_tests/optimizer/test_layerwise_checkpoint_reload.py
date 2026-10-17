# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest import mock

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, Float16OptimizerWithFloat16Params


def make_optimizer(chunks, owned_params):
    # Exercise reload without constructing CUDA buffers or process groups.
    child = object.__new__(Float16OptimizerWithFloat16Params)
    child.float16_groups = [owned_params]
    child.fp32_from_float16_groups = [
        [torch.nn.Parameter(torch.zeros_like(param, dtype=torch.float32)) for param in owned_params]
    ]
    child.is_stub_optimizer = False
    child._dummy_overflow_buf = None
    child.optimizer = mock.Mock(param_groups=[{'params': child.fp32_from_float16_groups[0]}])
    optimizer = object.__new__(LayerWiseDistributedOptimizer)
    optimizer.chained_optimizers = [child]
    optimizer.model_chunks = chunks
    return optimizer, child


@pytest.mark.parametrize('format', ['flat', 'model', 'model0', 'model_0'])
def test_reload_preserves_checkpoint_precision_and_only_updates_owned_masters(format):
    model = torch.nn.Linear(2, 2).bfloat16()
    wrapped = torch.nn.Module()
    wrapped.add_module('module', model)
    checkpoint = {
        'language_model.weight': torch.full((2, 2), 1.001),
        'language_model.bias': torch.full((2,), 2.002),
    }
    state_dict = checkpoint if format == 'flat' else {format: checkpoint}
    original_model = {name: param.detach().clone() for name, param in model.named_parameters()}
    optimizer, child = make_optimizer([wrapped], [model.weight])

    optimizer.reload_model_params(state_dict)

    torch.testing.assert_close(
        child.fp32_from_float16_groups[0][0], checkpoint['language_model.weight'], rtol=0, atol=0
    )
    assert (
        child.fp32_from_float16_groups[0][0][0, 0].item()
        != checkpoint['language_model.weight'].bfloat16()[0, 0].item()
    )
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, original_model[name], rtol=0, atol=0)


@pytest.mark.parametrize('prefix', ['model', 'model_'])
def test_reload_nested_layerwise_optimizers_splits_model_chunks(prefix):
    models = [torch.nn.Linear(2, 2, bias=False).bfloat16() for _ in range(2)]
    children = [make_optimizer([model], [model.weight]) for model in models]
    outer = object.__new__(ChainedOptimizer)
    outer.chained_optimizers = [optimizer for optimizer, _ in children]
    outer.model_chunks = models
    state_dict = {f'{prefix}{i}': {'weight': torch.full((2, 2), i + 1.001)} for i in range(2)}

    outer.reload_model_params(state_dict)

    for i, (_, child) in enumerate(children):
        torch.testing.assert_close(
            child.fp32_from_float16_groups[0][0],
            state_dict[f'{prefix}{i}']['weight'],
            rtol=0,
            atol=0,
        )


def test_reload_without_checkpoint_preserves_model_copy_behavior():
    model = torch.nn.Linear(2, 2, bias=False).bfloat16()
    optimizer, child = make_optimizer([model], [model.weight])
    optimizer.reload_model_params()
    torch.testing.assert_close(
        child.fp32_from_float16_groups[0][0], model.weight.float(), rtol=0, atol=0
    )


def test_reload_rank_with_no_owned_parameters():
    model = torch.nn.Linear(2, 2, bias=False).bfloat16()
    optimizer, child = make_optimizer([model], [])
    optimizer.reload_model_params({'weight': torch.full((2, 2), 1.001)})
    assert child.fp32_from_float16_groups == [[]]


def test_reload_rejects_missing_checkpoint_parameter_without_mutating_masters():
    model = torch.nn.Linear(2, 2, bias=False).bfloat16()
    optimizer, child = make_optimizer([model], [model.weight])
    with pytest.raises(AssertionError, match='0 matches'):
        optimizer.reload_model_params({'unrelated': torch.ones(2)})
    torch.testing.assert_close(
        child.fp32_from_float16_groups[0][0], torch.zeros(2, 2), rtol=0, atol=0
    )


def test_distributed_optimizer_retains_shared_parameter_matcher():
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.model_chunks = [model]
    checkpoint = {'model': {'weight': torch.full((2, 2), 1.001)}}
    mapping = optimizer._build_model_param_to_state_dict_param_map(checkpoint)
    assert mapping[model.weight] is checkpoint['model']['weight']


@pytest.mark.parametrize('optimizer_class', [DistributedOptimizer, LayerWiseDistributedOptimizer])
def test_shared_matcher_preserves_canonical_checkpoint_names(optimizer_class):
    model = torch.nn.Module()
    model.add_module('runtime', torch.nn.Linear(2, 2, bias=False))
    model.submodules_config = mock.Mock(sharded_state_dict_keys_map={'runtime.': 'canonical.'})
    optimizer = object.__new__(optimizer_class)
    optimizer.model_chunks = [model]
    checkpoint = {'canonical.weight': torch.full((2, 2), 1.001)}
    mapping = optimizer._build_model_param_to_state_dict_param_map(checkpoint)
    assert mapping[model.runtime.weight] is checkpoint['canonical.weight']


@pytest.mark.parametrize('optimizer_class', [DistributedOptimizer, LayerWiseDistributedOptimizer])
def test_shared_matcher_splits_grouped_checkpoint_weights(optimizer_class):
    model = torch.nn.Module()
    model.num_gemms = 2
    model.single_grouped_weight = False
    model.register_parameter('weight0', torch.nn.Parameter(torch.zeros(2, 2)))
    model.register_parameter('weight1', torch.nn.Parameter(torch.zeros(2, 2)))
    model._split_grouped_checkpoint_tensor = lambda tensor, key: tensor.unbind(0)
    optimizer = object.__new__(optimizer_class)
    optimizer.model_chunks = [model]
    grouped = torch.stack([torch.full((2, 2), 1.001), torch.full((2, 2), 2.002)])
    mapping = optimizer._build_model_param_to_state_dict_param_map({'weight': grouped})
    torch.testing.assert_close(mapping[model.weight0], grouped[0], rtol=0, atol=0)
    torch.testing.assert_close(mapping[model.weight1], grouped[1], rtol=0, atol=0)
