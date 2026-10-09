# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.enums import AttnMaskType


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


@pytest.mark.parametrize("window_size", [None, (4, 0)])
def test_sliding_window_skips_fused_kernel(monkeypatch, window_size):
    """The fused kernels have no window support, so SWA must use the torch path."""
    import megatron.core.fusions.fused_softmax as fused_softmax

    def _cpu_swa_mask(sq, skv, window_size):
        m = torch.ones(sq, skv, dtype=torch.bool)
        mu = torch.triu(m, diagonal=skv - sq - window_size[0])
        return ~torch.tril(mu, diagonal=skv - sq + window_size[1])

    monkeypatch.setattr(fused_softmax, "get_sliding_window_causal_mask", _cpu_swa_mask)
    softmax = FusedScaleMaskSoftmax(
        input_in_fp16=False,
        input_in_bf16=True,
        attn_mask_type=AttnMaskType.causal,
        scaled_masked_softmax_fusion=True,
        mask_func=_mask_func,
        softmax_in_fp32=True,
        scale=None,
        window_size=window_size,
    )
    # pretend the fused kernel constraints are satisfied
    monkeypatch.setattr(softmax, "is_kernel_available", lambda *args: True)
    fused_calls = []
    monkeypatch.setattr(
        softmax,
        "forward_fused_softmax",
        lambda x, mask: fused_calls.append(1) or torch.softmax(x.float(), dim=-1),
    )

    scores = torch.zeros(8, 1, 32, 32, dtype=torch.bfloat16)
    probs = softmax(scores, None)

    if window_size is None:
        assert fused_calls
    else:
        assert not fused_calls
        assert probs[0, 0, 31, 0].item() < 1e-6
        assert probs[0, 0, 31, 31].item() > 0.0
