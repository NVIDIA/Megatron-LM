# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real VERL padding can run without the optional FlashAttention CUDA package."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite import compat

pytestmark = pytest.mark.optional


def _missing(name):
    def functions():
        raise ModuleNotFoundError(name=name)

    return functions


def test_padding_fallback_real_verl_roundtrip_and_gradient(monkeypatch):
    from tensordict import TensorDict
    from verl.utils import attention_utils as attention
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    monkeypatch.setattr(attention, '_get_attention_functions', _missing('flash_attn'))
    assert compat._patch_verl_padding_helpers()
    assert compat._patch_verl_padding_helpers()  # idempotent
    mask = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    values = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2).requires_grad_()
    unpacked, indices, offsets, maximum, lengths = attention.unpad_input(values, mask)
    assert torch.equal(unpacked, values[mask.bool()])
    assert offsets.tolist() == [0, 4, 8] and maximum == 4 and lengths.tolist() == [4, 4]
    restored = attention.pad_input(unpacked, indices, 2, 5)
    assert torch.equal(restored[mask.bool()], values[mask.bool()])
    restored.sum().backward()
    assert torch.equal(values.grad, mask[..., None].expand_as(values).float())
    ids = torch.tensor([[0, 11, 12, 21, 22], [13, 14, 15, 23, 0]])
    batch = TensorDict(
        dict(
            input_ids=ids,
            attention_mask=mask,
            response_mask=torch.tensor([[1, 1], [1, 0]]),
            position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
            prompts=ids[:, :3],
            responses=ids[:, 3:],
        ),
        batch_size=[2],
    )
    packed = left_right_2_no_padding(batch)
    assert packed['input_ids'].values().tolist() == [11, 12, 21, 22, 13, 14, 15, 23]
    scores = no_padding_2_padding(torch.arange(8, dtype=torch.float32), packed)
    assert torch.equal(
        scores.view(torch.int32), torch.tensor([[1.0, 2.0], [6.0, 0.0]]).view(torch.int32)
    )


def test_padding_keeps_installed_native_provider(monkeypatch):
    from verl.utils import attention_utils as attention

    sentinel = (object(),) * 4
    monkeypatch.setattr(attention, '_get_attention_functions', lambda: sentinel)
    assert compat._patch_verl_padding_helpers()
    assert attention._get_attention_functions() is sentinel


def test_padding_does_not_hide_other_missing_dependency(monkeypatch):
    from verl.utils import attention_utils as attention

    monkeypatch.setattr(attention, '_get_attention_functions', _missing('flash_attn_2_cuda'))
    assert compat._patch_verl_padding_helpers()
    with pytest.raises(ModuleNotFoundError) as caught:
        attention._get_attention_functions()
    assert caught.value.name == 'flash_attn_2_cuda'
