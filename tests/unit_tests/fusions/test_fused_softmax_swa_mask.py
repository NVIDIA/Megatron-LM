# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util

import pytest
import torch

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType

HAVE_SCALED_SOFTMAX_EXT = importlib.util.find_spec("scaled_masked_softmax_cuda") is not None and (
    importlib.util.find_spec("scaled_upper_triang_masked_softmax_cuda") is not None
)


def _mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(attention_mask, -10000.0)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sliding-window mask builder allocates on CUDA"
)
def test_sliding_window_softmax_respects_caller_mask():
    """A caller-provided padding mask must compose with the sliding-window mask
    instead of being silently discarded (previously pad keys received attention)."""
    softmax = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=False,
        attn_mask_type=AttnMaskType.causal,
        scaled_masked_softmax_fusion=False,
        mask_func=_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=(4, 0),
    )
    b, np_, sq, sk = 2, 1, 8, 8
    scores = torch.zeros(b, np_, sq, sk, device="cuda")
    pad_mask = torch.zeros(b, 1, sq, sk, dtype=torch.bool, device="cuda")
    pad_mask[1, :, :, -3:] = True  # sample 1: last three keys are padding

    probs = softmax(scores, pad_mask)

    # padding keys must receive (numerically) zero attention on the SWA path
    assert probs[1, :, :, -3:].max().item() < 1e-6
    # rows keep valid in-window keys, so nothing degenerates to NaN
    assert torch.isfinite(probs).all()
    # the un-padded sample still follows the sliding-window causal pattern
    assert probs[0, 0, 5, 5].item() > 0.0
    assert probs[0, 0, 5, 0].item() < 1e-6


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="sliding-window mask builder allocates on CUDA"
)
@pytest.mark.skipif(not HAVE_SCALED_SOFTMAX_EXT, reason="apex fused softmax extensions missing")
def test_windowed_config_never_selects_fused_kernel():
    """A windowed config must not route to the apex fused kernels.

    The fused kernels have no window parameter, so selecting them for a
    windowed config silently computes full-causal attention (issue #7847:
    out-of-window prob 0.03125 instead of ~0 on the T4 repro below).
    """
    softmax = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=True,
        attn_mask_type=AttnMaskType.causal,
        scaled_masked_softmax_fusion=True,
        mask_func=_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=(4, 0),
    )
    b, np_, sq, sk = 8, 1, 32, 32
    scores = torch.zeros(b, np_, sq, sk, device="cuda", dtype=torch.bfloat16)

    # Windowed configs are ineligible for the fused kernel even when every
    # shape/dtype gate would otherwise pass (b*np % batch_per_block == 0).
    assert not softmax.is_kernel_available(None, b, np_, sq, sk)

    # ... and fall through to the torch path, which applies the window.
    probs = softmax(scores, None)
    assert probs[0, 0, 31, 0].item() < 1e-6  # out of window: ~0
    assert probs[0, 0, 31, 31].item() > 0.0  # in window: receives mass
