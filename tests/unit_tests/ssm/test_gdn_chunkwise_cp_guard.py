# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GDN/GDN2 must reject unsupported chunkwise CP at construction and runtime."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import common, gdn, gdn2
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.transformer import TransformerConfig


def _group(size):
    return SimpleNamespace(size=lambda: size, rank=lambda: 0)


def _config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        calculate_per_token_loss=True,
        linear_cp_mode="chunkwise",
        linear_cp_layout="contiguous",
        perform_initialization=False,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def _stub_class(mixer_cls):
    """A subclass whose variant attrs satisfy the post-_setup assertion, so
    construction proceeds past the guard to the projections (which we mock)."""
    attrs = {
        "dt_bias_dim": 1,
        "a_log_dim": 1,
        "in_proj_extra_dim": 1,
        "in_proj_split_names": [],
        "in_proj_split_sections": [],
    }

    class _Stub(mixer_cls):
        _setup_variant_attrs = lambda self: None
        gated_delta_rule = staticmethod(lambda *a, **k: None)

    for k, v in attrs.items():
        setattr(_Stub, k, v)
    return _Stub


@pytest.fixture
def mocked_projections(monkeypatch):
    monkeypatch.setattr(common, "HAVE_FLA", True)
    monkeypatch.setattr(common, "build_module", lambda *a, **k: torch.nn.Identity())


def _build(mixer_cls, config, cp_size, extra_submodule_attrs=(), dynamic_cp_sizes=()):
    submodules = SimpleNamespace(
        in_proj=None, out_proj=None, out_norm=None, **{name: None for name in extra_submodule_attrs}
    )
    groups = dict(tp=_group(1), cp=_group(cp_size))
    if dynamic_cp_sizes:
        groups.update(
            dp_cp=_group(max(dynamic_cp_sizes)),
            dynamic_cp={size: _group(size) for size in dynamic_cp_sizes},
        )
    return mixer_cls(
        config=config, submodules=submodules, pg_collection=ProcessGroupCollection(**groups)
    )


@pytest.mark.parametrize("mixer_cls", [gdn.GatedDeltaNet, gdn2.GatedDeltaNet2])
def test_chunkwise_cp_rejected_at_construction(mocked_projections, mixer_cls):
    stub = _stub_class(mixer_cls)
    with pytest.raises(ValueError, match=r"chunkwise' is not implemented.*headwise.*#9731"):
        _build(stub, _config(), cp_size=2)


@pytest.mark.parametrize("mixer_cls", [gdn.GatedDeltaNet, gdn2.GatedDeltaNet2])
def test_dynamic_cp_rejected_before_runtime_switch(mocked_projections, monkeypatch, mixer_cls):
    stub = _stub_class(mixer_cls)
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda _: True
    )
    config = _config(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=128)
    with pytest.raises(ValueError, match="sequence-sharded inputs") as error:
        _build(stub, config, cp_size=1, dynamic_cp_sizes=(1, 2))
    assert "every CP rank must hold the full sequence" not in str(error.value)


def test_dynamic_cp_singleton_still_accepted(mocked_projections, monkeypatch):
    monkeypatch.setattr(
        "megatron.core.transformer.transformer_config.is_te_min_version", lambda _: True
    )
    config = _config(dynamic_context_parallel=True, max_seqlen_per_dp_cp_rank=128)
    mixer = _build(_stub_class(gdn.GatedDeltaNet), config, cp_size=1, dynamic_cp_sizes=(1,))
    assert mixer.cp_size == 1


@pytest.mark.parametrize("mixer_cls", [gdn.GatedDeltaNet, gdn2.GatedDeltaNet2])
def test_runtime_cp_rejected_before_projection(mocked_projections, monkeypatch, mixer_cls):
    monkeypatch.setattr(gdn, "get_pg_size", lambda group: group.size())
    monkeypatch.setattr(gdn2, "get_pg_size", lambda group: group.size())
    mixer = _build(_stub_class(mixer_cls), _config(), cp_size=1)
    packed = PackedSeqParams(local_cp_size=2, cp_group=_group(2))
    with pytest.raises(ValueError, match="linear_cp_layout='zigzag'"):
        mixer.forward_pre_attn_and_core_attn(torch.empty(4, 1, 32), None, packed_seq_params=packed)


def test_headwise_cp_still_accepted(mocked_projections):
    stub = _stub_class(gdn.GatedDeltaNet)
    mixer = _build(stub, _config(linear_cp_mode="headwise", linear_cp_layout="zigzag"), cp_size=2)
    assert mixer.cp_size == 2


def test_chunkwise_cp_with_cp1_is_not_gated(mocked_projections):
    stub = _stub_class(gdn.GatedDeltaNet)
    mixer = _build(stub, _config(), cp_size=1)
    assert mixer.cp_size == 1


def test_kda_config_bypasses_guard(mocked_projections, monkeypatch):
    from megatron.core.ssm.gated_delta_net import kda

    monkeypatch.setattr(kda, "HAVE_FLA", True)
    monkeypatch.setattr(kda, "HAVE_FLA_KDA", True)
    stub = _stub_class(kda.KimiDeltaAttention)
    kda_config = KDALayerConfig(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        linear_cp_mode="chunkwise",
        linear_cp_layout="contiguous",
        perform_initialization=False,
    )
    mixer = _build(stub, kda_config, cp_size=2, extra_submodule_attrs=('beta_proj',))
    assert mixer.cp_size == 2
