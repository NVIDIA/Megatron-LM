# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType


def _mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(attention_mask, -10000.0)


def _make_softmax(window_size, attn_mask_type=AttnMaskType.causal, fusion=False):
    return FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=fusion,
        attn_mask_type=attn_mask_type,
        scaled_masked_softmax_fusion=fusion,
        mask_func=_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=window_size,
    )


@pytest.mark.parametrize("entrypoint", ["forward", "forward_torch_softmax"])
@pytest.mark.parametrize("attn_mask_type", [AttnMaskType.causal, AttnMaskType.padding])
@pytest.mark.parametrize("fusion", [False, True])
@pytest.mark.parametrize("with_offset", [False, True])
def test_sliding_window_rejects_caller_mask(entrypoint, attn_mask_type, fusion, with_offset):
    """Reject ambiguous masks before fused dispatch, including all-False caller masks."""
    softmax = _make_softmax((4, 0), attn_mask_type, fusion)
    # These shapes and BF16 flags would otherwise reach fused-kernel selection.
    scores = torch.zeros(4, 1, 32, 32, dtype=torch.bfloat16 if fusion else torch.float32)
    mask = torch.zeros(4, 1, 32, 32, dtype=torch.bool)
    offset = torch.zeros(1) if with_offset else None

    with pytest.raises(ValueError, match="mask and window_size cannot be supplied together"):
        getattr(softmax, entrypoint)(scores, mask, offset)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sliding-window mask builder allocates on CUDA"
)
@pytest.mark.parametrize("sq, sk", [(8, 8), (4, 8)])
@pytest.mark.parametrize("with_offset", [False, True])
def test_sliding_window_without_caller_mask(sq, sk, with_offset):
    """Keep the existing window-only probabilities and gradients."""
    softmax = _make_softmax((4, 0))
    scores = torch.randn(2, 1, sq, sk, device="cuda", requires_grad=True)
    reference = scores.detach().clone().requires_grad_()
    query = torch.arange(sq, device="cuda")[:, None] + sk - sq
    key = torch.arange(sk, device="cuda")[None, :]
    mask = (key > query) | (key < query - 4)
    offset = torch.zeros(1, device="cuda") if with_offset else None
    expected_scores = _mask_func(reference, mask)
    if with_offset:
        expected_scores = torch.cat([expected_scores, torch.zeros_like(scores[..., :1])], -1)
    expected = torch.softmax(expected_scores, -1)[..., :sk]

    actual = softmax(scores, None, offset)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    torch.testing.assert_close(scores.grad, reference.grad)


@pytest.mark.parametrize("attn_mask_type", [AttnMaskType.causal, AttnMaskType.padding])
@pytest.mark.parametrize("with_offset", [False, True])
def test_caller_mask_without_window_size(attn_mask_type, with_offset):
    """Use the caller's complete mask when no internal window is requested."""
    softmax = _make_softmax(None, attn_mask_type)
    scores = torch.zeros(2, 1, 8, 8)
    # The caller can explicitly combine window and padding restrictions with |.
    query = torch.arange(8)[:, None]
    key = torch.arange(8)[None, :]
    window_mask = (key > query) | (key < query - 4)
    padding_mask = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    padding_mask[1, :, :, -3:] = True
    mask = padding_mask | window_mask
    original_mask = mask.clone()
    offset = torch.zeros(1) if with_offset else None
    expected_scores = _mask_func(scores, mask)
    if with_offset:
        expected_scores = torch.cat([expected_scores, torch.zeros_like(scores[..., :1])], -1)

    actual = softmax(scores, mask, offset)
    torch.testing.assert_close(actual, torch.softmax(expected_scores, -1)[..., :8])
    torch.testing.assert_close(mask, original_mask)
