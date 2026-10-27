# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pin dev CSA output ownership behind the retained main construction interfaces."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import transformer_config as config_module
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
@pytest.mark.parametrize("packed", [False, True])
def test_wrapper_does_not_repeat_core_inverse_rope(monkeypatch, fused, packed):
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
            assert (packed_seq_params is not None) == packed
            assert boundary_hidden is None and boundary_kv is None
            if fused:
                return fused_inverse(raw, inverse=True)
            return torch.cat((raw[..., :2], inverse_rotation(raw[..., 2:], inverse=True)), dim=-1)

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
            o_lora_rank=4,
        ),
        pg_collection=SimpleNamespace(cp=group),
        rotary_pos_emb=Rotary(),
        _dsv4_uses_yarn_rope=False,
        num_attention_heads_per_partition=1,
        output_projection_local_groups=1,
        o_local_groups=1,
        linear_o_group_proj=torch.eye(4),
        linear_proj=lambda value: (value, None),
        get_query_key_value_tensors=lambda *args, **kwargs: (raw,) * 5,
        offload_core_attention=False,
        offload_attn_proj=False,
        recompute_up_proj=False,
        training=True,
    )
    monkeypatch.setattr(dsv4, "off_interface", lambda enabled, value, name: _NoOffload(value))
    monkeypatch.setattr(
        dsv4, "apply_rotary_pos_emb", lambda *args, **kwargs: pytest.fail("duplicate inverse RoPE")
    )

    packed_params = PackedSeqParams(qkv_format="thd") if packed else None
    output, bias = dsv4.DSv4HybridAttention.forward(
        layer, raw, None, packed_seq_params=packed_params
    )

    torch.testing.assert_close(output, torch.tensor([[[1.0, 2.0, 4.0, -3.0]]]))
    torch.testing.assert_close(raw, original)
    assert calls == ["inverse"]
    assert bias is None
    output.sum().backward()
    torch.testing.assert_close(raw.grad, torch.tensor([[[1.0, 1.0, -1.0, 1.0]]]))


def test_dsv4_rejects_non_thd_context_parallel_input():
    with pytest.raises(ValueError, match="requires qkv_format='thd'"):
        dsv4.DSv4HybridAttention.forward(
            SimpleNamespace(
                config=SimpleNamespace(dsa_kernel_backend="none"),
                pg_collection=SimpleNamespace(cp=SimpleNamespace(size=lambda: 2)),
            ),
            torch.ones(2, 1, 4),
            None,
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
    "overrides",
    [
        {"dynamic_context_parallel": True},
        {"hybrid_context_parallel": True},
        {"dsa_cp_balance_indexer": True},
        {"dsa_indexer_precision": "mxfp8"},
    ],
)
def test_dsv4_scope_keeps_dev_options_available(overrides):
    TransformerConfig._validate_dsv4_execution_scope(_scope_config(**overrides))


def test_dsv4_rejects_unknown_indexer_precision():
    with pytest.raises(ValueError, match="dsa_indexer_precision must be"):
        TransformerConfig._validate_dsv4_execution_scope(
            _scope_config(dsa_indexer_precision="invalid")
        )


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


@pytest.mark.parametrize("dynamic", [False, True])
def test_dsv4_retains_dev_runtime_cp_layout_ownership(monkeypatch, dynamic):
    """The real config keeps uniform per-pack CP instead of selecting main's static converter."""
    monkeypatch.setattr(config_module, "is_te_min_version", lambda *args, **kwargs: True)
    monkeypatch.setattr(config_module, "_validate_dsa_kernel_backend_dependencies", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    cudnn = ModuleType("cudnn")

    def indexer_forward_top_k_wrapper(*, deterministic=False):
        raise AssertionError("This CPU test only checks configuration, not kernel execution")

    def cp_wrapper(*, q_causal_offsets=None):
        raise AssertionError("This CPU test only checks configuration, not kernel execution")

    cudnn.DSA = SimpleNamespace(
        indexer_forward_top_k_wrapper=indexer_forward_top_k_wrapper,
        indexer_forward_wrapper=cp_wrapper,
        dense_indexer_score_recompute_wrapper=cp_wrapper,
        dense_attn_score_recompute_wrapper=cp_wrapper,
        dense_indexer_backward_wrapper=cp_wrapper,
    )
    monkeypatch.setitem(sys.modules, "cudnn", cudnn)
    config = config_module.MLATransformerConfig(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        multi_latent_attention=True,
        experimental_attention_variant="dsv4_hybrid",
        is_hybrid_model=True,
        add_bias_linear=False,
        csa_compress_ratios=[4, 0],
        dsa_kernel_backend="cudnn",
        context_parallel_size=1 if dynamic else 2,
        dynamic_context_parallel=dynamic,
        cp_partition_mode="contiguous",
        sequence_packing_scheduler="default_dynamic_cp" if dynamic else "dp_balanced",
        max_seqlen_per_dp_cp_rank=128,
        dsa_cp_balance_indexer=True,
        # Required by TransformerConfig for the dynamic=True case: loss scaling cannot
        # use the static TP x CP group once CP groups are picked per microbatch. This
        # test only asserts CP-layout ownership, so the flag is incidental to it, and
        # the dynamic=False case ignores it.
        calculate_per_token_loss=True,
    )
    assert config.dsa_cp_balance_indexer
    assert config.dynamic_context_parallel == dynamic
    assert config.cp_partition_mode == config.attention_cp_layout == "contiguous"
    assert config.linear_cp_mode == "chunkwise"
    assert not config._linear_cp_layout_explicit
