# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.modules.lora import (
    GroupedLinearLoRA,
    LinearLoRA,
    SharedGroupedLinearLoRA,
    freeze_non_lora_params,
    normalize_lora_config,
    trainable_param_stats,
)


def test_lora_config_aliases_and_trainable_param_accounting():
    cfg = normalize_lora_config({"enabled": True, "rank": 2, "alpha": 6, "targets": ["qkv", "fc2"]})

    assert cfg.enabled
    assert cfg.scale == 3.0
    assert cfg.targets() == {"linear_qkv", "linear_fc2"}
    assert cfg.targets_module("qkv")
    assert cfg.targets_module("linear_fc2")
    assert not normalize_lora_config({"enabled": False, "rank": 8}).enabled
    with pytest.raises(TypeError, match="LoRA config"):
        normalize_lora_config(object())

    class TinyAdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(3, 2)
            self.lora_adapter = nn.Linear(3, 2)

    model = TinyAdapterModel()
    stats = freeze_non_lora_params(model)

    assert stats["lora_tensors"] == 2
    assert stats["frozen_tensors"] == 2
    assert not model.base.weight.requires_grad
    assert model.lora_adapter.weight.requires_grad
    assert trainable_param_stats(model) == {
        "trainable_tensors": 2,
        "trainable_numel": model.lora_adapter.weight.numel() + model.lora_adapter.bias.numel(),
    }


def test_linear_lora_forward_backward_matches_low_rank_delta():
    layer = LinearLoRA(3, 2, rank=2, alpha=4, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        layer.lora_b.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    output = layer(x)

    torch.testing.assert_close(output, torch.tensor([[10.0, 22.0]]))
    output.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([[8.0, 12.0, 0.0]]))


def test_grouped_lora_respects_per_expert_splits():
    layer = GroupedLinearLoRA(2, 2, 2, rank=1, alpha=1, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]))
        layer.lora_b.copy_(torch.tensor([[[2.0], [3.0]], [[5.0], [7.0]]]))

    x = torch.tensor([[2.0, 9.0], [4.0, 1.0], [6.0, 3.0]])
    output = layer(x, [1, 2])

    torch.testing.assert_close(output, torch.tensor([[4.0, 6.0], [5.0, 7.0], [15.0, 21.0]]))
    with pytest.raises(ValueError, match="expected 2 splits"):
        layer(x, [3])


def test_shared_grouped_lora_uses_one_adapter_for_all_experts():
    layer = SharedGroupedLinearLoRA(2, 2, 2, rank=1, alpha=2, dropout=0.0)
    with torch.no_grad():
        layer.lora_a.copy_(torch.tensor([[1.0, -1.0]]))
        layer.lora_b.copy_(torch.tensor([[2.0], [3.0]]))

    output = layer(torch.tensor([[3.0, 1.0], [4.0, 7.0]]), [1, 1])

    torch.testing.assert_close(output, torch.tensor([[8.0, 12.0], [-12.0, -18.0]]))


def test_mrope_interleaves_text_height_and_width_sections():
    from megatron.lite.primitive.modules.mrope import MultimodalRotaryEmbedding

    base = torch.arange(3 * 2 * 6, dtype=torch.float32).reshape(3, 2, 6)

    interleaved = MultimodalRotaryEmbedding._apply_interleaved_mrope(base, mrope_section=[1, 1, 1])

    expected = base[0].clone()
    expected[..., 1] = base[1, ..., 1]
    expected[..., 2] = base[2, ..., 2]
    torch.testing.assert_close(interleaved, expected)


def test_mtp_aux_loss_scaler_threads_independent_gradient(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.mtp import MTPLossAutoScaler

    MTPLossAutoScaler.set_loss_scale(torch.tensor(0.125))
    output = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    mtp_loss = torch.tensor(4.0, requires_grad=True)

    MTPLossAutoScaler.apply(output * 3.0, mtp_loss).sum().backward()

    torch.testing.assert_close(output.grad, torch.full_like(output, 3.0))
    torch.testing.assert_close(mtp_loss.grad, torch.tensor(0.125))
    MTPLossAutoScaler.main_loss_backward_scale = 1.0


def test_gated_delta_static_helpers_are_finite_and_shape_stable(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.gated_delta_net import GatedDeltaNet

    alpha = torch.tensor([[[0.0, 1.0], [-1.0, 2.0]]])
    beta = torch.tensor([[[0.0, 2.0], [-2.0, 4.0]]])

    g, beta_sigmoid = GatedDeltaNet._compute_g_and_beta(torch.zeros(2), torch.ones(2), alpha, beta)

    assert g.shape == alpha.shape
    assert beta_sigmoid.shape == beta.shape
    assert torch.isfinite(g).all()
    assert torch.isfinite(beta_sigmoid).all()
    assert torch.all(g < 0)


def test_gated_delta_tp4_replicates_two_head_state(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import gated_delta_net as gdn_module

    class _FakeColumnParallelLinear(nn.Module):
        def __init__(self, _in_features, out_features, ps, **_kwargs):
            super().__init__()
            self.local_out = out_features // ps.tp_size

    class _FakeRowParallelLinear(nn.Module):
        def __init__(self, in_features, _out_features, ps, **_kwargs):
            super().__init__()
            self.local_in = in_features // ps.tp_size

    class _FakeRMSNorm(nn.Module):
        def __init__(self, hidden_size, **_kwargs):
            super().__init__()
            self.hidden_size = hidden_size

    monkeypatch.setattr(gdn_module, "ColumnParallelLinear", _FakeColumnParallelLinear)
    monkeypatch.setattr(gdn_module, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(gdn_module.te, "RMSNorm", _FakeRMSNorm)

    ps = SimpleNamespace(
        tp_size=4,
        tp_rank=0,
        tp_group=object(),
        cp_size=1,
        cp_rank=0,
        cp_group=None,
    )
    gdn = gdn_module.GatedDeltaNet(
        hidden_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        rms_norm_eps=1e-6,
        ps=ps,
    )

    assert gdn._replicate_heads is True
    assert gdn.num_k_heads_local == 2
    assert gdn.num_v_heads_local == 2
    assert gdn.conv1d.in_channels == 3
    assert gdn.dt_bias.shape == (2,)
    assert gdn.A_log.shape == (2,)
    assert gdn.o_proj.local_in == 1


def test_gated_delta_replicated_state_reduces_tp_gradients(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import gated_delta_net as gdn_module

    group = object()
    calls = []

    def fake_all_reduce(grad, *, group):
        calls.append(group)
        grad.add_(torch.tensor([3.0, 5.0]))

    monkeypatch.setattr(gdn_module.dist, "all_reduce", fake_all_reduce)
    state = torch.tensor([1.0, 2.0], requires_grad=True)

    synced = gdn_module._ReplicatedParameterWithGradReduce.apply(state, group)
    (synced * torch.tensor([2.0, 4.0])).sum().backward()

    assert calls == [group]
    torch.testing.assert_close(state.grad, torch.tensor([5.0, 9.0]))


def test_gated_delta_syncs_replicated_state_from_tp_rank_zero(
    monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import gated_delta_net as gdn_module

    class _FakeColumnParallelLinear(nn.Module):
        def __init__(self, _in_features, out_features, ps, **_kwargs):
            super().__init__()
            self.local_out = out_features // ps.tp_size

    class _FakeRowParallelLinear(nn.Module):
        def __init__(self, in_features, _out_features, ps, **_kwargs):
            super().__init__()
            self.local_in = in_features // ps.tp_size

    class _FakeRMSNorm(nn.Module):
        def __init__(self, hidden_size, **_kwargs):
            super().__init__()
            self.hidden_size = hidden_size

    monkeypatch.setattr(gdn_module, "ColumnParallelLinear", _FakeColumnParallelLinear)
    monkeypatch.setattr(gdn_module, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(gdn_module.te, "RMSNorm", _FakeRMSNorm)
    monkeypatch.setattr(gdn_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gdn_module.dist, "get_world_size", lambda _group: 4)
    monkeypatch.setattr(gdn_module.dist, "get_process_group_ranks", lambda _group: (12, 13, 14, 15))
    broadcasts = []
    monkeypatch.setattr(
        gdn_module.dist,
        "broadcast",
        lambda tensor, *, src, group: broadcasts.append((tensor, src, group)),
    )

    group = object()
    ps = SimpleNamespace(
        tp_size=4, tp_rank=1, tp_group=group, cp_size=1, cp_rank=0, cp_group=None
    )
    gdn = gdn_module.GatedDeltaNet(
        hidden_size=8,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        rms_norm_eps=1e-6,
        ps=ps,
    )

    gdn.sync_tp_replicated_parameters()

    assert [(src, sync_group) for _tensor, src, sync_group in broadcasts] == [
        (12, group),
        (12, group),
    ]
    assert [tensor.data_ptr() for tensor, _src, _group in broadcasts] == [
        gdn.A_log.data_ptr(),
        gdn.dt_bias.data_ptr(),
    ]
