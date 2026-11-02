# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Upstream absorbed-MLA interface regressions retained alongside dev numerical tests."""

from types import SimpleNamespace

import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import (
    absorbed_mla as absorbed_mla_module,
)
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
)


def test_checkpointed_attention_forward_captures_metadata(monkeypatch):
    """Optional metadata should stay in the closure instead of checkpoint tensor args."""

    packed_seq_params = PackedSeqParams(qkv_format='thd')
    checkpoint_args = None

    def fake_checkpoint(run_function, distribute_saved_activations, *args):
        nonlocal checkpoint_args
        del distribute_saved_activations
        checkpoint_args = args
        assert all(torch.is_tensor(arg) for arg in args)
        return run_function(*args)

    class CoreAttention(torch.nn.Module):
        def forward(self, query, key, *, value, attention_mask, **kwargs):
            del query, key, value, attention_mask
            assert kwargs["packed_seq_params"] is packed_seq_params
            assert kwargs["position_ids"] is None
            return kwargs["x"]

    dummy_attention = type(
        "DummyAttention",
        (),
        {
            "attn_mask_type": AttnMaskType.causal,
            "config": SimpleNamespace(),
            "core_attention": CoreAttention(),
            "pg_collection": SimpleNamespace(cp=object()),
        },
    )()

    monkeypatch.setattr(absorbed_mla_module.tensor_parallel, "checkpoint", fake_checkpoint)

    hidden_states = torch.randn(4, 1, 8)
    output = AbsorbedMLASelfAttention._checkpointed_attention_forward(
        dummy_attention,
        q_absorbed=torch.randn(4, 1, 2, 8),
        k_compressed=torch.randn(4, 1, 1, 8),
        hidden_states=hidden_states,
        q_compressed=torch.randn(4, 1, 8),
        attention_mask=torch.empty(1),
        up_v_weight=torch.randn(2, 4, 4),
        position_ids=None,
        packed_seq_params=packed_seq_params,
    )

    assert checkpoint_args is not None
    assert all(arg is not packed_seq_params for arg in checkpoint_args)
    assert all(arg is not None for arg in checkpoint_args)
    assert output is hidden_states


def test_restore_packed_thd_batch_dim_when_core_output_is_2d():
    """Packed-THD absorbed MLA should restore a missing singleton batch dim."""
    hidden_states = torch.empty(7, 1, 16)
    core_attn_out = torch.empty(7, 16)
    packed_seq_params = PackedSeqParams(qkv_format='thd')

    restored = absorbed_mla_module._restore_packed_thd_batch_dim(
        core_attn_out, hidden_states, packed_seq_params
    )

    assert restored.shape == (7, 1, 16)


def test_restore_packed_thd_batch_dim_keeps_already_normalized_output():
    """Packed-THD absorbed MLA should keep an already restored batch dim."""
    hidden_states = torch.empty(7, 1, 16)
    core_attn_out = torch.empty(7, 1, 16)
    packed_seq_params = PackedSeqParams(qkv_format='thd')

    restored = absorbed_mla_module._restore_packed_thd_batch_dim(
        core_attn_out, hidden_states, packed_seq_params
    )

    assert restored is core_attn_out
    assert restored.shape == hidden_states.shape


def test_absorbed_v_up_projection_applies_when_core_did_not_consume_weight():
    """Absorbed MLA should apply V-up when core attention returns latent channels."""
    torch.manual_seed(123)
    num_heads, kv_lora_rank, v_head_dim = 2, 3, 3
    core_attn_out = torch.randn(5, 1, num_heads * kv_lora_rank)
    v_up_weight = torch.randn(num_heads, v_head_dim, kv_lora_rank)

    projected = absorbed_mla_module._apply_absorbed_v_up_projection(
        core_attn_out,
        v_up_weight,
        num_attention_heads_per_partition=num_heads,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        core_consumed_v_up_projection=False,
    )
    expected = core_attn_out.view(5, 1, num_heads, kv_lora_rank)
    expected = torch.einsum("...nc,ndc->...nd", expected, v_up_weight)
    expected = expected.contiguous().view(5, 1, -1)

    torch.testing.assert_close(projected, expected, rtol=0, atol=0)


def test_absorbed_v_up_projection_skips_when_core_consumed_weight():
    """Absorbed MLA should not reapply V-up when core attention already consumed it."""
    num_heads, kv_lora_rank, v_head_dim = 2, 3, 3
    core_attn_out = torch.randn(5, 1, num_heads * v_head_dim)
    v_up_weight = torch.randn(num_heads, v_head_dim, kv_lora_rank)

    projected = absorbed_mla_module._apply_absorbed_v_up_projection(
        core_attn_out,
        v_up_weight,
        num_attention_heads_per_partition=num_heads,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
        core_consumed_v_up_projection=True,
    )

    assert projected is core_attn_out


def test_load_from_state_dict_backwards_compatible_with_split_kv_up_projection(monkeypatch):
    """Pre-refactor split K/V up-projection checkpoints load into the combined layout."""

    dummy_attention = object.__new__(AbsorbedMLASelfAttention)
    dummy_attention._uses_combined_kv_up_projection = True
    dummy_attention.num_attention_heads_per_partition = 2
    dummy_attention.config = SimpleNamespace(qk_head_dim=2, v_head_dim=3, kv_lora_rank=4)

    prefix = "self_attention."
    k_weight = torch.arange(2 * 2 * 4, dtype=torch.float32).view(2 * 2, 4)
    v_weight = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2 * 3, 4)
    state_dict = {
        f"{prefix}linear_k_up_proj.weight": k_weight.clone(),
        f"{prefix}linear_v_up_proj.weight": v_weight.clone(),
        f"{prefix}linear_k_up_proj._extra_state": torch.empty(0),
        f"{prefix}linear_v_up_proj._extra_state": torch.empty(0),
    }
    captured_state_dict = {}

    def fake_super_load(self, state_dict, *args, **kwargs):
        del self, args, kwargs
        captured_state_dict.update(state_dict)

    monkeypatch.setattr(absorbed_mla_module.Attention, "_load_from_state_dict", fake_super_load)

    AbsorbedMLASelfAttention._load_from_state_dict(
        dummy_attention, state_dict, prefix, {}, True, [], [], []
    )

    expected_weight = (
        torch.cat((k_weight.view(2, 2, 4), v_weight.view(2, 3, 4)), dim=1)
        .contiguous()
        .view(2 * (2 + 3), 4)
    )
    torch.testing.assert_close(
        captured_state_dict[f"{prefix}linear_kv_up_proj.weight"], expected_weight
    )
    assert f"{prefix}linear_k_up_proj.weight" not in captured_state_dict
    assert f"{prefix}linear_v_up_proj.weight" not in captured_state_dict
    assert f"{prefix}linear_kv_up_proj._extra_state" in captured_state_dict
    assert f"{prefix}linear_k_up_proj._extra_state" not in captured_state_dict
    assert f"{prefix}linear_v_up_proj._extra_state" not in captured_state_dict
