# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""A grouped runtime layout must not change DSv4's Muon update matrix."""

import os

import pytest
import torch

from megatron.core.optimizer.emerging_optimizers import (
    HAVE_EMERGING_OPTIMIZERS,
    TensorParallelMuon,
    _is_muon_excluded,
)
from megatron.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer
from megatron.core.optimizer.optimizer import copy_optimizer_param_metadata


@pytest.mark.skipif(not HAVE_EMERGING_OPTIMIZERS, reason='Emerging Optimizers is required')
@pytest.mark.parametrize('layer_sharded', [False, True])
def test_grouped_projection_preserves_flat_muon_steps_and_resume(layer_sharded):
    """Exercise real momentum, NS, scaling, weight decay and state restoration."""
    generator = torch.Generator().manual_seed(42)
    device = (
        torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0)))
        if torch.cuda.is_available()
        else torch.device('cpu')
    )
    initial = torch.randn(2, 8, 16, generator=generator).to(device)
    grouped = torch.nn.Parameter(initial.clone())
    grouped.muon_matrix_shape = (16, 16)
    flat = torch.nn.Parameter(initial.reshape(16, 16).clone())
    master = torch.nn.Parameter(grouped.detach().clone())
    copy_optimizer_param_metadata(master, grouped)
    assert master.muon_matrix_shape == (16, 16)
    assert not _is_muon_excluded(master)
    assert is_managed_by_layer_wise_optimizer(master)

    def make_optimizer(parameter):
        optimizer_type = TensorParallelMuon
        kwargs = {}
        if layer_sharded:
            from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

            optimizer_type = LayerShardedMuon
            kwargs['gtp_remat_group'] = None
        optimizer = optimizer_type(
            [parameter],
            lr=0.01,
            momentum=0.95,
            coefficient_type='deepseekv4',
            num_ns_steps=10,
            fp32_matmul_prec='highest',
            **kwargs,
        )
        if layer_sharded:
            optimizer.set_param_ns_homes({})
        return optimizer

    grouped_optimizer = make_optimizer(master)
    flat_optimizer = make_optimizer(flat)
    for step in range(3):
        gradient = torch.randn(initial.shape, generator=generator).to(device)
        master.grad = gradient.clone()
        flat.grad = gradient.reshape_as(flat).clone()
        grouped_optimizer.step()
        flat_optimizer.step()
        torch.testing.assert_close(master.reshape_as(flat), flat, rtol=0, atol=0)
        if step == 1:
            state = grouped_optimizer.state_dict()
            grouped_optimizer = make_optimizer(master)
            grouped_optimizer.load_state_dict(state)


@pytest.mark.parametrize('exclusion', ['use_muon', 'is_embedding_or_output_parameter'])
def test_grouped_projection_honors_scalar_optimizer_exclusions(exclusion):
    parameter = torch.nn.Parameter(torch.zeros(2, 8, 16))
    assert _is_muon_excluded(parameter)
    assert not is_managed_by_layer_wise_optimizer(parameter)
    parameter.muon_matrix_shape = (16, 16)
    setattr(parameter, exclusion, exclusion != 'use_muon')
    assert _is_muon_excluded(parameter)
    assert not is_managed_by_layer_wise_optimizer(parameter)
