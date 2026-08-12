# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_bias_dropout import (
    ResidualForgetGate,
    get_bias_dropout_add,
)
from megatron.core.transformer.transformer_config import TransformerConfig, compute_depth_scale


def _transformer_config(**kwargs) -> TransformerConfig:
    return TransformerConfig(
        num_layers=8,
        hidden_size=16,
        num_attention_heads=4,
        use_cpu_initialization=True,
        **kwargs,
    )


def test_compute_depth_scale_supports_tunable_exponent():
    assert compute_depth_scale(False, num_layers=16) is None
    assert compute_depth_scale(True, num_layers=16) == pytest.approx(1.0 / 16.0)
    assert compute_depth_scale(
        True, num_layers=16, residual_depth_scaling_exponent=0.5
    ) == pytest.approx(0.25)
    assert compute_depth_scale(True, num_layers=8, residual_depth_scaling_L_ref=16) == 1.0


def test_residual_scaling_is_applied_inside_bda():
    residual = torch.tensor([2.0, -4.0], dtype=torch.bfloat16)
    branch = torch.tensor([6.0, 2.0], dtype=torch.bfloat16)
    bias = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)

    output = get_bias_dropout_add(
        training=False, fused=False, branch_scale=0.25
    )((branch, bias), residual, 0.0)
    torch.testing.assert_close(output, residual + 0.25 * (branch + bias))


def test_residual_forget_gate_is_initialized_and_learnable():
    residual = torch.tensor([2.0, -4.0])
    branch = torch.tensor([6.0, 2.0])
    gate = ResidualForgetGate(gamma_init=0.99, max_forget=0.1)

    residual_scale = gate(residual.dtype)
    output = get_bias_dropout_add(
        training=False,
        fused=False,
        branch_scale=0.25,
        residual_scale=residual_scale,
    )((branch, None), residual, 0.0)

    assert gate.gamma().item() == pytest.approx(0.99)
    torch.testing.assert_close(output, 0.99 * residual + 0.25 * branch)
    output.sum().backward()
    assert gate.forget_logit.grad is not None
    assert gate.forget_logit.grad.abs().item() > 0.0

    with torch.no_grad():
        gate.forget_logit.zero_()
    gate.reset_parameters()
    assert gate.gamma().item() == pytest.approx(0.99)


def test_residual_forget_gate_config_validation():
    with pytest.raises(ValueError, match="requires residual_depth_scaling"):
        _transformer_config(use_residual_forget_gate=True)

    with pytest.raises(ValueError, match="incompatible with inference_fuse_tp_communication"):
        _transformer_config(
            residual_depth_scaling=True, inference_fuse_tp_communication=True
        )

    with pytest.raises(ValueError, match="CPU activation offloading"):
        _transformer_config(
            residual_depth_scaling=True,
            use_residual_forget_gate=True,
            cpu_offloading=True,
        )

    with pytest.raises(ValueError, match="CPU activation offloading"):
        _transformer_config(
            residual_depth_scaling=True,
            use_residual_forget_gate=True,
            fine_grained_activation_offloading=True,
        )

    config = _transformer_config(
        residual_depth_scaling=True,
        residual_depth_scaling_exponent=0.5,
        use_residual_forget_gate=True,
        residual_forget_gate_init=0.99,
        residual_forget_gate_max_forget=0.1,
    )
    assert config.residual_depth_scaling_exponent == 0.5
