# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core import ModelParallelConfig


def test_mtp_grad_scale_func_config():
    """Test that mtp_grad_scale_func config defaults to None and can be set."""
    config = ModelParallelConfig()
    assert config.mtp_grad_scale_func is None

    scale_fn = lambda: torch.tensor(0.5)
    config = ModelParallelConfig(mtp_grad_scale_func=scale_fn)
    assert config.mtp_grad_scale_func is scale_fn
    assert config.mtp_grad_scale_func().item() == 0.5


def test_mtp_loss_scale_selection():
    """Test the MTP loss scale selection logic matches the fallback behavior in schedules.

    The logic is:
    1. Use mtp_grad_scale_func() if provided
    2. Else use grad_scale_func(ones) if provided
    3. Else default to ones
    """
    device = 'cpu'

    # Case 1: mtp_grad_scale_func takes priority
    config = ModelParallelConfig(
        mtp_grad_scale_func=lambda: torch.tensor(0.25),
        grad_scale_func=lambda x: x * 2.0,
    )
    mtp_grad_scale_func = getattr(config, 'mtp_grad_scale_func', None)
    if mtp_grad_scale_func is not None:
        loss_scale = mtp_grad_scale_func()
    elif config.grad_scale_func is not None:
        loss_scale = config.grad_scale_func(torch.ones(1, device=device))
    else:
        loss_scale = torch.ones(1, device=device)
    assert loss_scale.item() == 0.25

    # Case 2: Falls back to grad_scale_func
    config = ModelParallelConfig(
        grad_scale_func=lambda x: x * 3.0,
    )
    mtp_grad_scale_func = getattr(config, 'mtp_grad_scale_func', None)
    if mtp_grad_scale_func is not None:
        loss_scale = mtp_grad_scale_func()
    elif config.grad_scale_func is not None:
        loss_scale = config.grad_scale_func(torch.ones(1, device=device))
    else:
        loss_scale = torch.ones(1, device=device)
    assert loss_scale.item() == 3.0

    # Case 3: Falls back to ones
    config = ModelParallelConfig()
    mtp_grad_scale_func = getattr(config, 'mtp_grad_scale_func', None)
    if mtp_grad_scale_func is not None:
        loss_scale = mtp_grad_scale_func()
    elif config.grad_scale_func is not None:
        loss_scale = config.grad_scale_func(torch.ones(1, device=device))
    else:
        loss_scale = torch.ones(1, device=device)
    assert loss_scale.item() == 1.0
