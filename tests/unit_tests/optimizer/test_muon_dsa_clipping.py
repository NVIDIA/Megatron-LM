# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Muon exemptions also apply to DSA's independent indexer clipping path."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer import optimizer as optimizer_module
from megatron.core.optimizer.optimizer import ChainedOptimizer


@pytest.mark.parametrize('dsa_clip', [None, 0.5])
@pytest.mark.parametrize('exemptions', [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize('shared_group', [True, False])
def test_muon_exemption_preserves_dsa_clipping(monkeypatch, dsa_clip, exemptions, shared_group):
    def norm(grads, **kwargs):
        return sum(grad.float().square().sum().item() for grad in grads) ** 0.5

    def clip(params, max_norm, total_norm, **kwargs):
        for param in params:
            param.grad.mul_(min(1.0, max_norm / (total_norm + 1e-6)))

    monkeypatch.setattr(optimizer_module, 'get_grad_norm_fp32', norm)
    monkeypatch.setattr(optimizer_module, 'clip_grad_by_total_norm_fp32', clip)
    config = SimpleNamespace(
        clip_grad=1.0,
        dsa_indexer_clip_grad=dsa_clip,
        grad_norm_skip_threshold=float('inf'),
        use_precision_aware_optimizer=False,
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False,
        log_num_zeros_in_grad=False,
    )

    class Member:
        def __init__(self, scale, exempt, group):
            self.config = config
            self.skip_grad_norm_clip = exempt
            self.params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(2)]
            self.params[0].grad = torch.tensor([3.0 * scale])
            self.params[1].grad = torch.tensor([4.0 * scale])
            self.param_groups = [
                {'params': self.params[:1], 'is_dsa_indexer': True},
                {'params': self.params[1:], 'is_dsa_indexer': False},
            ]
            self.group = group

        def prepare_grads(self):
            return False

        def get_parameters(self):
            return self.params

        def get_grad_stats_parallel_group(self):
            return self.group

        def get_grads_for_grad_norm(self, grad_norm_group=None):
            return [param.grad for param in self.params]

        def get_grad_norm(self):
            return norm(self.get_grads_for_grad_norm())

        def get_dsa_split_parameters(self):
            return self.params[:1], self.params[1:]

        def get_dsa_split_grad_norms(self):
            return tuple(param.grad.norm().item() for param in self.params)

        def _filter_grads_for_norm(self, params):
            return [param.grad for param in params]

        def has_grad_norm_group(self, grad_norm_group):
            return False

    members = [Member(10, exemptions[0], 0), Member(1, exemptions[1], 0 if shared_group else 1)]
    before = [[param.grad.clone() for param in member.params] for member in members]
    chain = ChainedOptimizer(members)
    monkeypatch.setattr(chain, 'step_with_ready_grads', lambda: True)
    success, reported_norm, _ = chain.step()
    assert success
    assert reported_norm == pytest.approx(2525**0.5)
    assert chain._last_dsa_split_grad_norms == pytest.approx((909**0.5, 1616**0.5))
    for member, original, exempt in zip(members, before, exemptions):
        for index, (param, grad) in enumerate(zip(member.params, original)):
            if exempt:
                expected = grad
            else:
                selected = [values for values, skip in zip(before, exemptions) if not skip]
                if dsa_clip is None:
                    denominator = (
                        sum(value.square().item() for values in selected for value in values) ** 0.5
                    )
                    limit = 1.0
                else:
                    denominator = sum(values[index].square().item() for values in selected) ** 0.5
                    limit = dsa_clip if index == 0 else 1.0
                expected = grad * min(1.0, limit / (denominator + 1e-6))
            torch.testing.assert_close(param.grad, expected)
