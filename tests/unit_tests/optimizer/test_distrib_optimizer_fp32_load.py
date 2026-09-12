# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU regressions for loading an optimizer shard that aliases an FP32 model parameter."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

pytestmark = pytest.mark.internal


def _optimizer_for_parameter(model_param):
    """Supply the parameter/state tables consumed by the production load method."""
    main_param = model_param.view(-1)[2:6]
    if model_param.dtype != torch.float32:
        main_param = main_param.detach().float().clone()
    state = {"exp_avg": torch.zeros_like(main_param), "exp_avg_sq": torch.zeros_like(main_param)}
    optimizer = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        model_param_group_index_map={model_param: (0, 0)},
        optimizer=SimpleNamespace(
            param_groups=[{"params": [main_param]}], state={main_param: state}
        ),
    )
    return optimizer, main_param, state


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float64])
def test_load_fp32_optimizer_shard_preserves_autograd_and_other_model_rows(source_dtype):
    """Legacy load has no warning counter and copies into a live leaf-parameter view."""
    model_param = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))
    optimizer, main_param, state = _optimizer_for_parameter(model_param)
    assert main_param.requires_grad and main_param._base is model_param
    tensors = {
        "param": torch.arange(4, dtype=source_dtype) + 100,
        "exp_avg": torch.arange(4, dtype=source_dtype) + 200,
        "exp_avg_sq": torch.arange(4, dtype=source_dtype) + 300,
    }
    DistributedOptimizer._set_main_param_and_optimizer_states(optimizer, model_param, tensors)
    torch.testing.assert_close(model_param[:2], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(model_param[6:], torch.tensor([6.0, 7.0]))
    torch.testing.assert_close(main_param, tensors["param"].float())
    for key in state:
        torch.testing.assert_close(state[key], tensors[key].float())
    model_param.sum().backward()
    torch.testing.assert_close(model_param.grad, torch.ones_like(model_param))
    assert not hasattr(optimizer, "_loaded_master_mismatch")
