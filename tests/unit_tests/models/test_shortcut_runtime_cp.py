# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for runtime CP metadata at the shortcut-MoE integration boundary."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.packed_seq_params import PackedSeqParams


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_shortcut_router_receives_runtime_cp_metadata(cp_size):
    hidden = torch.randn(4, 1, 2)
    mask = torch.tensor([[False, True, False, False]])
    packed = PackedSeqParams(local_cp_size=cp_size, cp_group=SimpleNamespace(size=lambda: cp_size))
    calls = []

    def unflatten(value, padding_mask, packed_seq_params):
        assert padding_mask is mask and packed_seq_params is packed
        return value, padding_mask, None

    def route(value, padding_mask, packed_seq_params=None):
        assert padding_mask is mask and packed_seq_params is packed
        calls.append(packed_seq_params.cp_group.size())
        return value, value

    block = SimpleNamespace(
        recompute_shortcut_pre_mlp_layernorm=False,
        shortcut_pre_mlp_layernorm=torch.nn.Identity(),
        moe_layer=SimpleNamespace(
            _maybe_unflatten_for_moe=unflatten,
            mlp=SimpleNamespace(route=route, preprocess=lambda value, *args: (value, value)),
        ),
    )
    result, _ = ShortcutMoEBlock._moe_router_preprocess(block, hidden, mask, packed)
    assert result is hidden
    assert calls == [cp_size]


@pytest.mark.parametrize("with_layout", [False, True])
def test_shortcut_uses_moe_layout_mask_for_routed_and_shared_paths(with_layout):
    input_mask = torch.tensor([[False, True]])
    moe_mask = torch.tensor([[True, False]])
    masks_by_layout = {"zigzag": moe_mask}
    input_packed, attn_packed, moe_packed = object(), object(), object()
    calls = []

    def prepare(layer_idx, hidden):
        calls.append(("prepare", layer_idx))
        return hidden, attn_packed if layer_idx == 4 else moe_packed

    def mask_for_layer(layer_idx, padding_mask, padding_mask_by_layout):
        assert layer_idx == 5
        assert padding_mask is input_mask and padding_mask_by_layout is masks_by_layout
        return moe_mask

    def route(shortcut_hidden, padding_mask, packed_seq_params):
        assert padding_mask is (moe_mask if with_layout else input_mask)
        assert packed_seq_params is (moe_packed if with_layout else input_packed)
        return shortcut_hidden, shortcut_hidden

    def shared(hidden, padding_mask, packed_seq_params):
        route(hidden, padding_mask, packed_seq_params)
        return torch.zeros_like(hidden), None, hidden, ()

    def attention(hidden_states, packed_seq_params, **kwargs):
        assert packed_seq_params is (attn_packed if with_layout else input_packed)
        return (hidden_states,)

    block = SimpleNamespace(
        attn_layer=SimpleNamespace(
            config=object(),
            forward_pre_attn_and_core_attn=attention,
            forward_post_core_attn=lambda value: value,
        ),
        moe_layer=SimpleNamespace(
            config=object(),
            mlp=SimpleNamespace(routed_experts_compute=lambda value, probs: (value, None)),
        ),
        attn_local_idx=4,
        moe_local_idx=5,
        attn_layer_idx=8,
        moe_layer_idx=9,
        overlap_mode=False,
        shortcut_pre_mlp_layernorm_checkpoint=None,
        _moe_router_preprocess=route,
        _moe_shared_experts=shared,
        _launch_dispatch=lambda value, probs, **kwargs: (value, probs),
        _launch_combine=lambda value, **kwargs: value,
        _postprocess=lambda residual, *args, **kwargs: residual,
    )
    layout = SimpleNamespace(
        prepare_layer=prepare,
        get_layer_padding_mask=mask_for_layer,
        finalize_layer=lambda layer_idx, value: value,
    )
    hidden = torch.randn(2, 1, 2)
    output = ShortcutMoEBlock.forward(
        block,
        hidden_states=hidden,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        sequence_len_offset=None,
        packed_seq_params=input_packed,
        padding_mask=input_mask,
        quant_context_factory=lambda *args: nullcontext(),
        cp_layout_state=layout if with_layout else None,
        padding_mask_by_layout=masks_by_layout,
    )
    torch.testing.assert_close(output, hidden)
    assert calls == ([("prepare", 4), ("prepare", 5), ("prepare", 5)] if with_layout else [])
