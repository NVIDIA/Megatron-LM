# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import (
    MTPLossAutoScaler,
    MTPLossGradMask,
    process_mtp_loss,
)


@pytest.fixture(autouse=True)
def restore_mtp_autoscaler():
    """Do not leave the shared MTP backward scale or mask provider changed for other tests."""
    previous_scale = MTPLossAutoScaler.main_loss_backward_scale
    previous_provider = MTPLossAutoScaler.grad_mask_provider
    yield
    MTPLossAutoScaler.set_loss_scale(previous_scale)
    MTPLossAutoScaler.set_grad_mask_provider(previous_provider)


@pytest.fixture
def device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def test_filled_mask_scales_the_mtp_loss_gradient(device):
    MTPLossAutoScaler.set_loss_scale(torch.tensor(2.0, device=device))
    hidden = torch.randn(3, 4, device=device, requires_grad=True)
    loss = torch.randn(3, 4, device=device, requires_grad=True)
    grad_mask = MTPLossGradMask()
    output = MTPLossAutoScaler.apply(hidden, loss, grad_mask)
    grad_mask.mask = torch.tensor([[1.0], [0.0], [1.0]], device=device)
    output.sum().backward()
    torch.testing.assert_close(loss.grad, 2.0 * grad_mask.mask.expand(3, 4))
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))


@pytest.mark.parametrize("extra", [(), (None,)])
def test_unmasked_calls_keep_the_existing_gradient(device, extra):
    """Two-argument callers and an explicit None mask behave as before."""
    MTPLossAutoScaler.set_loss_scale(torch.tensor(3.0, device=device))
    hidden = torch.randn(2, 2, device=device, requires_grad=True)
    loss = torch.randn(2, 2, device=device, requires_grad=True)
    MTPLossAutoScaler.apply(hidden, loss, *extra).sum().backward()
    torch.testing.assert_close(loss.grad, torch.full_like(loss, 3.0))


def test_a_requested_but_unfilled_mask_fails_the_backward(device):
    hidden = torch.randn(2, 2, device=device, requires_grad=True)
    loss = torch.randn(2, 2, device=device, requires_grad=True)
    output = MTPLossAutoScaler.apply(hidden, loss, MTPLossGradMask())
    with pytest.raises(RuntimeError, match="never filled"):
        output.sum().backward()


def _config(*, depth=2, early_backward=False):
    return SimpleNamespace(
        mtp_num_layers=depth,
        mtp_detach_heads=False,
        mtp_loss_scaling_factor=0.3,
        calculate_per_token_loss=True,
        cross_entropy_loss_fusion=False,
        attention_cp_layout="zigzag",
        mtp_loss_early_backward=early_backward,
    )


def _mtp_hidden_grad(hidden, weight, labels, config, packed, grad_mask_value=None):
    """Return d(MTP loss)/d(MTP hidden states), optionally with a gradient mask."""
    # process_mtp_loss takes the main and MTP hidden states concatenated along the sequence.
    hidden = hidden.clone().reshape(-1, 1, 4).requires_grad_()
    grad_mask = MTPLossGradMask() if grad_mask_value is not None else None
    MTPLossAutoScaler.set_grad_mask_provider((lambda: grad_mask) if grad_mask is not None else None)
    MTPLossAutoScaler.set_loss_scale(torch.ones((), device=hidden.device, dtype=hidden.dtype))
    output = process_mtp_loss(
        hidden_states=hidden,
        labels=labels,
        loss_mask=torch.ones_like(labels, dtype=hidden.dtype),
        output_layer=lambda states, **_: (F.linear(states, weight), None),
        output_weight=None,
        is_training=False,
        compute_language_model_loss=lambda targets, values: F.cross_entropy(
            values.permute(1, 2, 0), targets, reduction="none"
        ),
        config=config,
        packed_seq_params=packed,
    )
    if grad_mask is not None:
        # Filled only after the forward, as a loss function deciding rejections would.
        grad_mask.mask = grad_mask_value
    # Isolate MTP's custom backward from the main-model gradient.
    (output.sum() * 0).backward()
    length = labels.size(1)
    return hidden.grad.reshape(config.mtp_num_layers + 1, length, 1, 4)[1:]


@pytest.mark.parametrize("packed", [False, True])
def test_masked_sequences_contribute_no_mtp_gradient(device, packed):
    """A per-sequence mask removes rejected sequences and leaves kept ones unchanged."""
    generator = torch.Generator(device=device).manual_seed(5124)
    config = _config()
    hidden = torch.randn(3, 16, 1, 4, device=device, dtype=torch.float64, generator=generator)
    weight = torch.randn(7, 4, device=device, dtype=torch.float64, generator=generator)
    labels = torch.randint(0, 7, (1, 16), device=device, generator=generator)
    params = (
        PackedSeqParams(qkv_format="thd", cu_seqlens_q=torch.tensor([0, 8, 16], device=device))
        if packed
        else None
    )
    # Keep the first packed sequence, reject the second (or, unpacked, the second half).
    keep = torch.cat([torch.ones(1, 8), torch.zeros(1, 8)], dim=1).to(device, torch.float64)

    unmasked = _mtp_hidden_grad(hidden, weight, labels, config, params)
    masked = _mtp_hidden_grad(hidden, weight, labels, config, params, grad_mask_value=keep)

    assert unmasked[:, 8:].abs().sum() > 0
    torch.testing.assert_close(masked[:, 8:], torch.zeros_like(masked[:, 8:]))
    if packed:
        # Rolling stays inside each packed sequence, so the kept one is unaffected.
        torch.testing.assert_close(masked[:, :8], unmasked[:, :8])


def test_early_backward_rejects_a_gradient_mask(device):
    config = _config(early_backward=True)
    hidden = torch.randn(3 * 8, 1, 4, device=device, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(7, 4, device=device, dtype=torch.float64)
    labels = torch.randint(0, 7, (1, 8), device=device)
    MTPLossAutoScaler.set_grad_mask_provider(MTPLossGradMask)
    with pytest.raises(ValueError, match="mtp_loss_early_backward"):
        process_mtp_loss(
            hidden_states=hidden,
            labels=labels,
            loss_mask=torch.ones(1, 8, device=device, dtype=torch.float64),
            output_layer=lambda states, **_: (F.linear(states, weight), None),
            output_weight=None,
            is_training=False,
            compute_language_model_loss=lambda targets, values: F.cross_entropy(
                values.permute(1, 2, 0), targets, reduction="none"
            ),
            config=config,
        )
