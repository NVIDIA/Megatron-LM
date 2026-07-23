# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from megatron.core.optimizer.clip_grads import count_zeros_fp32
from megatron.core.optimizer.optimizer_config import OptimizerConfig


def test_grad_norm_skip_threshold_config():
    """Test that grad_norm_skip_threshold config has correct default."""
    config = OptimizerConfig()
    assert config.grad_norm_skip_threshold == float('inf')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_count_zeros_filters_shared_and_tp_duplicate_fsdp_params():
    """FSDP gradients should use the same duplicate filters as regular gradients."""

    def make_param(grad, *, shared=False, tp_duplicate=False):
        return SimpleNamespace(
            __fsdp_param__=True,
            grad=SimpleNamespace(_local_tensor=grad),
            shared=shared,
            tp_duplicate=tp_duplicate,
        )

    params = [
        make_param(torch.tensor([0.0, 1.0], device="cuda")),
        make_param(torch.zeros(2, device="cuda"), shared=True),
        make_param(torch.zeros(2, device="cuda"), tp_duplicate=True),
    ]

    with (
        mock.patch(
            "megatron.core.optimizer.clip_grads.param_is_not_shared",
            side_effect=lambda param: not param.shared,
        ),
        mock.patch(
            "megatron.core.optimizer.clip_grads.param_is_not_tensor_parallel_duplicate",
            side_effect=lambda param, tp_group=None: not param.tp_duplicate,
        ),
        mock.patch("megatron.core.optimizer.clip_grads.torch.distributed.all_reduce"),
    ):
        num_zeros = count_zeros_fp32(params, grad_stats_parallel_group=object())

    assert num_zeros == 1
