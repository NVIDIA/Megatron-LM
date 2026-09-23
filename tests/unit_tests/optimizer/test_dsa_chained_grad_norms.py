# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DSA split norms retain each optimizer child's gradient and replica semantics."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import optimizer as optimizer_module
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


class _LeafOptimizer:
    _filter_grads_for_norm = MegatronOptimizer._filter_grads_for_norm
    get_dsa_split_parameters = MegatronOptimizer.get_dsa_split_parameters
    get_dsa_split_grad_norms = MegatronOptimizer.get_dsa_split_grad_norms

    def __init__(self, config, params, stats_group):
        self.config = config
        self.optimizer = SimpleNamespace(param_groups=params)
        self.stats_group = stats_group

    def get_grad_stats_parallel_group(self):
        return self.stats_group


@pytest.mark.parametrize('decoupled', [False, True])
@pytest.mark.parametrize('shared_stats_group', [False, True])
def test_chained_dsa_split_norms_preserve_child_filtering(
    monkeypatch, decoupled, shared_stats_group
):
    config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=decoupled,
        use_precision_aware_optimizer=False,
    )

    def param(value, **tags):
        result = torch.nn.Parameter(torch.zeros(1))
        grad = torch.tensor([float(value)])
        if decoupled:
            result.grad = torch.full_like(grad, 1000)
            result.decoupled_grad = grad
        else:
            result.grad = grad
        for name, tag in tags.items():
            setattr(result, name, tag)
        return result

    group = object()
    children = [
        _LeafOptimizer(
            config,
            [
                {'is_dsa_indexer': True, 'params': [param(3), param(100, shared=True)]},
                {
                    'is_dsa_indexer': False,
                    'params': [param(5), param(100, test_tp_duplicate=True)],
                },
            ],
            group,
        ),
        _LeafOptimizer(
            config,
            [
                {
                    'is_dsa_indexer': True,
                    'params': [param(4), param(100, test_gtp_duplicate=True)],
                },
                {'is_dsa_indexer': False, 'params': [param(12)]},
            ],
            group if shared_stats_group else object(),
        ),
    ]

    # Keep real subset/gradient/shared filtering; isolate replica ownership and reduction
    # from the process-group API used by each repository.
    monkeypatch.setattr(
        optimizer_module.tensor_parallel,
        'param_is_not_tensor_parallel_duplicate',
        lambda param, **kwargs: not getattr(param, 'test_tp_duplicate', False),
    )
    monkeypatch.setattr(
        optimizer_module.tensor_parallel,
        'param_is_not_gtp_duplicate',
        lambda param, **kwargs: not getattr(param, 'test_gtp_duplicate', False),
    )
    monkeypatch.setattr(
        optimizer_module,
        'get_grad_norm_fp32',
        lambda grads, **kwargs: sum(grad.double().square().sum().item() for grad in grads) ** 0.5,
    )
    assert ChainedOptimizer(children).get_dsa_split_grad_norms() == pytest.approx((5.0, 13.0))
