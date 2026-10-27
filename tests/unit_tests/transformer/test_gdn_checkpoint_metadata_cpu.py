# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU coverage of the actual GDN-family TP-local checkpoint split contract.

DTensor transport is replaced with Tensor views. This exercises key matching,
variant metadata, optimizer splitting and in-place load aliases, not DTensor collectives.
"""

import pytest
import torch

from megatron.core.transformer import fsdp_dtensor_checkpoint as checkpoint


@pytest.mark.parametrize("tp", [1, 2])
@pytest.mark.parametrize("variant", ["gdn", "kda", "gdn2", "legacy_gdn"])
def test_family_checkpoint_uses_variant_projection_sections(variant, tp, monkeypatch):
    qk, value, heads = 8 // tp, 16 // tp, 4 // tp
    if variant in ("gdn", "legacy_gdn"):
        names = ["query", "key", "value", "z", "beta", "alpha"]
        sizes = [qk, qk, value, value, heads, heads]
    elif variant == "kda":
        names = ["query", "key", "value", "g", "gate"]
        sizes = [qk, qk, value, qk, value]
    else:
        names = ["query", "key", "value", "z", "f", "b", "w"]
        sizes = [qk, qk, value, value, qk, qk, value]
    layer = torch.nn.Module()
    layer.qk_dim, layer.v_dim, layer.num_value_heads, layer.tp_size = 8, 16, 4, tp
    layer.in_proj_dim = sum(sizes) * tp
    if variant != "legacy_gdn":
        layer.in_proj_split_names = names
        layer.in_proj_split_sections = tuple(sizes)
    layer.in_proj = torch.nn.Linear(3, sum(sizes), bias=False)
    weight = layer.in_proj.weight
    with torch.no_grad():
        weight.copy_(torch.arange(weight.numel()).view_as(weight))
    model = torch.nn.Module()
    model.module = torch.nn.Module()
    model.module.attn = layer
    key = "attn.in_proj.weight"
    opt_key = "module.module." + key
    exp_avg = weight.detach().clone() + 100
    exp_avg_sq = weight.detach().clone() + 200
    model_state = {key: weight.detach()}
    optimizer_state = {
        "state": {opt_key: {"step": 7, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}}
    }
    calls = []

    def split(data, sections, dim, update_uneven_dtensor_chunk_meta):
        assert update_uneven_dtensor_chunk_meta is True
        calls.append(tuple(sections))
        return torch.split(data, sections, dim=dim)

    monkeypatch.setattr(checkpoint, "HAVE_MEGATRON_FSDP", True)
    monkeypatch.setattr(checkpoint, "DTensor", torch.Tensor)
    monkeypatch.setattr(checkpoint, "split_dtensor", split)
    result, optim = checkpoint.handle_gdn_in_state_dict(model, model_state, optimizer_state)
    assert set(result) == {key + "." + name for name in names}
    assert set(optim["state"]) == {opt_key + "." + name for name in names}
    assert calls == [tuple(sizes)] * 3
    offset = 0
    for name, width in zip(names, sizes):
        torch.testing.assert_close(result[key + "." + name], weight[offset : offset + width])
        state = optim["state"][opt_key + "." + name]
        assert state["step"] == 7
        torch.testing.assert_close(state["exp_avg"], exp_avg[offset : offset + width])
        torch.testing.assert_close(state["exp_avg_sq"], exp_avg_sq[offset : offset + width])
        offset += width
    # Loading through component views must still update the original fused tensor.
    result[key + "." + names[0]].fill_(-10)
    assert torch.all(weight[: sizes[0]] == -10)
    assert set(model_state) == {key}
    assert set(optimizer_state["state"]) == {opt_key}


@pytest.mark.parametrize("inside_mtp", [False, True])
def test_legacy_fused_checkpoint_names_remain_loadable(inside_mtp, monkeypatch):
    layer = torch.nn.Module()
    layer.qk_dim, layer.v_dim, layer.num_value_heads, layer.tp_size = 8, 16, 4, 1
    layer.in_proj_split_names = ["query", "key", "value", "z", "f", "b", "w"]
    layer.in_proj_split_sections = (8, 8, 16, 16, 8, 8, 16)
    layer.in_proj_dim = sum(layer.in_proj_split_sections)
    model = torch.nn.Module()
    model.module = torch.nn.Module()
    if inside_mtp:
        model.module.mtp = torch.nn.Module()
        model.module.mtp.mtp_model_layer = layer
        model.module.mtp.mtp_layer_pattern = None
        key, disk_key = "mtp.mtp_model_layer.in_proj.weight", "mtp.transformer_layer.in_proj.weight"
    else:
        model.module.attn = layer
        key = disk_key = "attn.in_proj.weight"
    value = torch.arange(layer.in_proj_dim * 3).view(layer.in_proj_dim, 3).float()
    opt_key = "module.module." + key
    disk_opt_key = "module.module." + disk_key
    model_state = {key: value}
    optimizer_state = {
        "state": {opt_key: {"step": 7, "exp_avg": value.clone(), "exp_avg_sq": value.clone()}}
    }
    keys = {
        "model." + disk_key,
        "optimizer.state." + disk_opt_key + ".exp_avg",
        "optimizer.state." + disk_opt_key + ".exp_avg_sq",
    }
    monkeypatch.setattr(checkpoint, "HAVE_MEGATRON_FSDP", True)
    monkeypatch.setattr(
        checkpoint,
        "split_dtensor",
        lambda *a, **k: pytest.fail("legacy fused tensor must not be split"),
    )
    result, optim = checkpoint.handle_gdn_in_state_dict(
        model, model_state, optimizer_state, checkpoint_keys=keys
    )
    assert set(result) == {key}
    assert result[key] is value
    assert set(optim["state"]) == {opt_key}
    assert optim["state"][opt_key] is optimizer_state["state"][opt_key]
    with pytest.raises(RuntimeError, match="original TP size"):
        checkpoint.handle_gdn_in_state_dict(
            model, model_state, optimizer_state, checkpoint_keys=keys, checkpoint_tp_size=2
        )
