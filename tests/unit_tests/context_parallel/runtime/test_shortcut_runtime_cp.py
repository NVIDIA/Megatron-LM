# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU checks for runtime CP metadata at the shortcut-MoE integration boundary."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.shortcut_block import ShortcutMoEBlock
from megatron.core.packed_seq_params import PackedSeqParams

pytestmark = pytest.mark.launch_on_gb200


def test_shortcut_router_receives_runtime_cp_metadata():
    hidden = torch.randn(4, 1, 2)
    mask = torch.tensor([[False, True, False, False]])
    packed = PackedSeqParams(local_cp_size=2, cp_group=SimpleNamespace(size=lambda: 2))
    calls = []
    missing = object()

    def unflatten(value, padding_mask, input_ids, packed_seq_params):
        assert input_ids is None
        assert padding_mask is mask and packed_seq_params is packed
        return value, padding_mask, None, None

    def route(value, padding_mask, packed_seq_params=None):
        assert padding_mask is mask and packed_seq_params is packed
        calls.append(packed_seq_params)
        return value, value

    def preprocess(value, probs, routing_map, padding_mask, *, packed_seq_params=missing):
        assert value is hidden and probs is hidden and routing_map is hidden
        assert padding_mask is mask
        # Newer MoE interfaces also pass the packed metadata to preprocessing.
        assert packed_seq_params is missing or packed_seq_params is packed
        return value, value

    block = SimpleNamespace(
        recompute_shortcut_pre_mlp_layernorm=False,
        shortcut_pre_mlp_layernorm=torch.nn.Identity(),
        moe_layer=SimpleNamespace(
            _maybe_unflatten_for_moe=unflatten,
            mlp=SimpleNamespace(route=route, preprocess=preprocess),
        ),
    )
    result, _ = ShortcutMoEBlock._moe_router_preprocess(block, hidden, mask, packed)
    assert result is hidden
    assert calls == [packed]


def test_shortcut_preserves_boundary_mask_for_routed_and_shared_paths():
    input_mask = torch.tensor([[False, True]])
    input_packed = object()

    def route(shortcut_hidden, padding_mask, packed_seq_params, recompute_context=None):
        assert recompute_context is None
        assert padding_mask is input_mask
        assert packed_seq_params is input_packed
        return shortcut_hidden, shortcut_hidden

    def shared(hidden_states, padding_mask, packed_seq_params, recompute_context=None):
        route(hidden_states, padding_mask, packed_seq_params, recompute_context)
        return torch.zeros_like(hidden_states), None, hidden_states, ()

    def attention(hidden_states, packed_seq_params, **kwargs):
        assert packed_seq_params is input_packed
        return (hidden_states,)

    block = SimpleNamespace(
        compute_layer=SimpleNamespace(
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
        _read_shortcut_hidden=lambda value, recompute_context: value,
        _moe_router_preprocess=route,
        _moe_shared_experts=shared,
        _launch_dispatch=lambda value, probs, **kwargs: (value, probs),
        _dispatch_and_compute_routed_experts=lambda value, probs, overlap: (
            ShortcutMoEBlock._dispatch_and_compute_routed_experts(block, value, probs, overlap)
        ),
        _launch_combine=lambda value, **kwargs: value,
        _postprocess=lambda residual, *args, **kwargs: residual,
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
    )
    torch.testing.assert_close(output, hidden)
