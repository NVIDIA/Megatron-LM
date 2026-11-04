# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Output-weight detachment preserves MTP input gradients and main-head training."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, process_mtp_loss
from megatron.core.transformer.transformer_config import TransformerConfig


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(5, 4) / 20)

    def forward(self, hidden, weight=None, **kwargs):
        return F.linear(hidden, self.weight if weight is None else weight), None


def loss(labels, logits):
    return F.cross_entropy(
        logits.transpose(0, 1).reshape(-1, 5), labels.reshape(-1), reduction="none"
    ).reshape(labels.shape)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("detach_output", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("legacy_detach", [False, True])
@pytest.mark.parametrize("early_backward", [False, True])
def test_output_weight_detach_keeps_auxiliary_hidden_gradients(
    tied, detach_output, per_token, legacy_detach, early_backward
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        use_cpu_initialization=True,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=per_token,
        mtp_detach_heads=legacy_detach,
        mtp_loss_early_backward=early_backward,
        mtp_detach_output_weight=detach_output,
        attention_cp_layout="zigzag",
    )
    layer = OutputLayer()
    shared_weight = layer.weight if tied else None
    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4).div(24).requires_grad_()
    labels = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    output = process_mtp_loss(inputs, labels, mask, layer, shared_weight, False, loss, config)
    (output.sum() * 0).backward()
    actual_hidden_grad = inputs.grad.clone()
    assert torch.count_nonzero(actual_hidden_grad[3:]) > 0
    assert torch.count_nonzero(actual_hidden_grad[:3]) == 0
    assert (layer.weight.grad is None) == (detach_output or legacy_detach)
    # Independent reference: labels/mask shift by one additional token for MTP.
    reference = inputs.detach()[3:].clone().requires_grad_()
    shifted_labels = torch.tensor([[2, 3, 0]])
    shifted_mask = torch.tensor([[1.0, 1.0, 0.0]])
    reference_loss = (
        loss(shifted_labels, F.linear(reference, layer.weight.detach())) * shifted_mask
    ).sum()
    coefficient = 0.1 * (mask.sum() / shifted_mask.sum() if per_token else 1 / shifted_mask.sum())
    (reference_loss * coefficient).backward()
    torch.testing.assert_close(actual_hidden_grad[3:], reference.grad)


def test_main_output_weight_remains_trainable():
    """Auxiliary detachment must not disable the main projection's weight gradient."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=1,
        mtp_num_layers=1,
        mtp_detach_output_weight=True,
        use_cpu_initialization=True,
    )
    layer = OutputLayer()
    hidden = torch.randn(6, 1, 4, requires_grad=True)
    labels = torch.tensor([[1, 2, 3]])
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    output = process_mtp_loss(hidden, labels, None, layer, layer.weight, False, loss, config)
    logits, _ = layer(output)
    loss(labels, logits).sum().backward()
    assert layer.weight.requires_grad
    assert layer.weight.grad is not None
    assert torch.count_nonzero(hidden.grad[3:]) > 0
