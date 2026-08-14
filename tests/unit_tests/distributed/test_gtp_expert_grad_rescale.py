# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Cover the GTP/EGTP share of the DP normalization for expert gradients.

GTP carves its ranks out of the data-parallel axis, so DDP's 1/DP scaling shrinks as GTP
grows. Dense params get the missing factor back from the gtp_remat AVG in finalize_model_grads;
expert params only ever see compensation sized to the EGTP axis, leaving a factor GTP/EGTP by
which their gradients come out too large.

The failure is silent -- no error, no NaN, just expert gradients off by a constant -- so it is
invisible in a short run and only shows up as a convergence difference much later. These tests
pin the arithmetic directly, on CPU, without a process group.
"""

import pytest
import torch

from megatron.core.distributed.finalize_model_grads import _rescale_expert_grads_for_gtp_remat


class _Group:
    """Stands in for a process group: the function only ever asks for its size."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class _Model(torch.nn.Module):
    """One dense param and one expert param, each carrying a main_grad of ones.

    `allreduce=False` is what marks a param as expert-parallel in Megatron; the dense param
    keeps the default and must not be touched.
    """

    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Parameter(torch.zeros(4))
        self.expert = torch.nn.Parameter(torch.zeros(4))
        self.expert.allreduce = False
        self.dense.main_grad = torch.ones(4)
        self.expert.main_grad = torch.ones(4)


def _run(gtp, egtp, per_token_loss=False):
    model = _Model()
    _rescale_expert_grads_for_gtp_remat(
        [model],
        _Group(gtp) if gtp is not None else None,
        _Group(egtp) if egtp is not None else None,
        per_token_loss,
    )
    return model


@pytest.mark.parametrize(
    "gtp,egtp,expected", [(4, 1, 0.25), (4, 2, 0.5), (8, 2, 0.25), (2, 1, 0.5)]
)
def test_expert_grads_get_back_the_gtp_over_egtp_factor(gtp, egtp, expected):
    model = _run(gtp, egtp)
    assert torch.allclose(model.expert.main_grad, torch.full((4,), expected))


@pytest.mark.parametrize("gtp,egtp", [(4, 1), (4, 2), (8, 2)])
def test_dense_grads_are_left_alone(gtp, egtp):
    """Dense params are compensated by the gtp_remat AVG, so touching them here double-counts."""
    model = _run(gtp, egtp)
    assert torch.allclose(model.dense.main_grad, torch.ones(4))


@pytest.mark.parametrize(
    "gtp,egtp",
    [
        (1, 1),  # no GTP at all
        (4, 4),  # every gtp rank is also an egtp rank: nothing is left over
        (2, 4),  # EGTP wider than GTP: the reduce-scatter mean already covers it
        (None, None),  # groups absent entirely
    ],
)
def test_no_rescale_when_gtp_does_not_exceed_egtp(gtp, egtp):
    model = _run(gtp, egtp)
    assert torch.allclose(model.expert.main_grad, torch.ones(4))
    assert torch.allclose(model.dense.main_grad, torch.ones(4))


def test_per_token_loss_is_skipped():
    """Under calculate_per_token_loss the gtp axis is SUM-reduced and the per-token divisor
    already counts the gtp peers' tokens, so rescaling here would divide twice."""
    model = _run(4, 1, per_token_loss=True)
    assert torch.allclose(model.expert.main_grad, torch.ones(4))


def test_params_without_a_grad_are_skipped():
    model = _Model()
    del model.expert.main_grad
    _rescale_expert_grads_for_gtp_remat([model], _Group(4), _Group(1), False)
    assert not hasattr(model.expert, "main_grad")


def test_frozen_expert_params_are_skipped():
    model = _Model()
    model.expert.requires_grad = False
    _rescale_expert_grads_for_gtp_remat([model], _Group(4), _Group(1), False)
    assert torch.allclose(model.expert.main_grad, torch.ones(4))
