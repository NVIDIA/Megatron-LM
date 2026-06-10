# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.schedules import _get_mtp_loss_scale


def test_mtp_grad_scale_func_config():
    """Test that mtp_grad_scale_func config defaults to None and can be set."""
    config = ModelParallelConfig()
    assert config.mtp_grad_scale_func is None

    scale_fn = lambda: torch.tensor(0.5)
    config = ModelParallelConfig(mtp_grad_scale_func=scale_fn)
    assert config.mtp_grad_scale_func is scale_fn
    assert config.mtp_grad_scale_func().item() == 0.5


def test_mtp_loss_scale_selection():
    """Test MTP loss scale selection and device normalization."""

    device = (
        torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    # Case 1: mtp_grad_scale_func takes priority
    config = ModelParallelConfig(
        mtp_grad_scale_func=lambda: torch.tensor([0.25], device='cpu'),
        grad_scale_func=lambda x: x * 2.0,
    )
    loss_scale = _get_mtp_loss_scale(config, device)
    assert loss_scale.item() == 0.25
    assert loss_scale.device == device

    # Case 2: Falls back to grad_scale_func
    config = ModelParallelConfig(grad_scale_func=lambda x: x * 3.0)
    loss_scale = _get_mtp_loss_scale(config, device)
    assert loss_scale.item() == 3.0
    assert loss_scale.device == device

    # Case 3: Falls back to ones
    config = ModelParallelConfig()
    loss_scale = _get_mtp_loss_scale(config, device)
    assert loss_scale.item() == 1.0
    assert loss_scale.device == device


@pytest.mark.parametrize(
    "config,scale_func_name",
    [
        (
            ModelParallelConfig(mtp_grad_scale_func=lambda: torch.tensor([0.25, 0.5])),
            "mtp_grad_scale_func",
        ),
        (
            ModelParallelConfig(grad_scale_func=lambda _: torch.tensor([0.25, 0.5])),
            "grad_scale_func",
        ),
    ],
)
def test_mtp_loss_scale_rejects_non_scalar_scale(config, scale_func_name):
    """Test that MTP loss scaling rejects per-token or per-sample scale values."""

    device = (
        torch.device('cuda', torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device('cpu')
    )

    with pytest.raises(ValueError, match=f"{scale_func_name} must return a scalar"):
        _get_mtp_loss_scale(config, device)
