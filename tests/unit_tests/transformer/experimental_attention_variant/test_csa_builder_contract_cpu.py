# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pin main's core-output ownership and static packed-input contract."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as dsv4,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class _NoOffload:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_args):
        return False

    def group_offload(self, value, **_kwargs):
        return value


@pytest.mark.parametrize("fused", [False, True])
def test_sbhd_wrapper_owns_inverse_rope_for_custom_core(monkeypatch, fused):
    raw = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)
    original = raw.detach().clone()
    calls = []
    group = SimpleNamespace(rank=lambda: 0, size=lambda: 1)

    class Core(torch.nn.Module):
        def forward(
            self,
            query,
            key,
            value,
            mask,
            *,
            packed_seq_params,
            x,
            qr,
            boundary_hidden=None,
            boundary_kv=None,
        ):
            assert packed_seq_params is None
            assert boundary_hidden is None and boundary_kv is None
            return raw

    def inverse_rotation(value, *args, **kwargs):
        assert kwargs["inverse"] is True
        calls.append("inverse")
        return torch.stack((value[..., 1], -value[..., 0]), dim=-1)

    def fused_inverse(value, *args, **kwargs):
        content, rotary = value.split((2, 2), dim=-1)
        return torch.cat((content, inverse_rotation(rotary, **kwargs)), dim=-1)

    class Rotary:
        def __call__(self, *_args, **_kwargs):
            return torch.empty(1)

        def get_cached_cos_sin(self, *_args, **_kwargs):
            return torch.empty(1), torch.empty(1)

    layer = SimpleNamespace(
        core_attention=Core(),
        config=SimpleNamespace(
            qk_pos_emb_head_dim=2,
            v_head_dim=4,
            apply_rope_fusion=fused,
            output_projection_lora_rank=4,
        ),
        pg_collection=SimpleNamespace(cp=group),
        rotary_pos_emb=Rotary(),
        _dsv4_uses_yarn_rope=False,
        num_attention_heads_per_partition=1,
        output_projection_local_groups=1,
        linear_o_group_proj=torch.eye(4),
        linear_proj=lambda value: (value, None),
        get_query_key_value_tensors=lambda *args, **kwargs: (raw,) * 5,
        offload_core_attention=False,
        offload_attn_proj=False,
        recompute_up_proj=False,
        training=True,
    )
    monkeypatch.setattr(dsv4, "off_interface", lambda enabled, value, name: _NoOffload(value))
    monkeypatch.setattr(dsv4, "apply_rotary_pos_emb", inverse_rotation)
    monkeypatch.setattr(dsv4, "fused_mla_rope_inplace", object())
    monkeypatch.setattr(dsv4, "fused_mla_rope_out_of_place", fused_inverse)

    output, bias = dsv4.DSv4HybridAttention.forward(layer, raw, None)

    torch.testing.assert_close(output, torch.tensor([[[1.0, 2.0, 4.0, -3.0]]]))
    torch.testing.assert_close(raw, original)
    assert calls == ["inverse"]
    assert bias is None
    output.sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[[1.0, 1.0, -1.0, 1.0]]]))


def test_dsv4_rejects_packed_native_backend_before_calling_custom_core():
    with pytest.raises(ValueError, match="requires dsa_kernel_backend='cudnn'"):
        dsv4.DSv4HybridAttention.forward(
            SimpleNamespace(
                config=SimpleNamespace(dsa_kernel_backend="none"),
                pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 1)),
            ),
            torch.ones(2, 1, 4),
            None,
            packed_seq_params=SimpleNamespace(qkv_format="thd"),
        )


def _scope_config(**overrides):
    values = dict(
        experimental_attention_variant="dsv4_hybrid",
        context_parallel_size=1,
        dynamic_context_parallel=False,
        hybrid_context_parallel=False,
        sequence_packing_scheduler=None,
        dsa_cp_balance_indexer=False,
        dsa_indexer_precision="bf16",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides,error,message",
    [
        ({"dynamic_context_parallel": True}, ValueError, "dynamic context parallelism"),
        ({"hybrid_context_parallel": True}, ValueError, "dynamic context parallelism"),
        ({"dsa_cp_balance_indexer": True}, ValueError, "balanced CP indexer"),
        ({"dsa_indexer_precision": "mxfp8"}, ValueError, "MXFP8 indexer"),
    ],
)
def test_dsv4_scope_rejects_deferred_options(overrides, error, message):
    with pytest.raises(error, match=message):
        TransformerConfig._validate_dsv4_execution_scope(_scope_config(**overrides))


def test_main_sbhd_scope_remains_available():
    TransformerConfig._validate_dsv4_execution_scope(_scope_config())


def test_main_static_packed_cp_scope_is_available():
    TransformerConfig._validate_dsv4_execution_scope(
        _scope_config(context_parallel_size=2, sequence_packing_scheduler="dp_balanced")
    )


def test_dsv4_scope_does_not_disable_other_models_context_parallelism():
    TransformerConfig._validate_dsv4_execution_scope(
        _scope_config(experimental_attention_variant="gdn", context_parallel_size=2)
    )
