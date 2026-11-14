# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Load-template compatibility without changing model construction or save keys."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.dist_checkpointing import layer_key_aliases
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

FUSED = "decoder.layers.self_attention.linear_qkv.layer_norm_weight"
LEGACY = "decoder.layers.input_layernorm.weight"
KEYS_MAP = {"input_layernorm.": "self_attention.linear_qkv.layer_norm_"}


def model_with_maps(*keys_maps):
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList()
    for keys_map in keys_maps:
        layer = torch.nn.Module()
        layer.submodules_config = SimpleNamespace(sharded_state_dict_keys_map=keys_map)
        model.layers.append(layer)
    return model


def tensor(key):
    return ShardedTensor.from_rank_offsets(key, torch.arange(4, dtype=torch.float32))


def remap(model, state_dict, checkpoint_keys):
    with patch.object(
        layer_key_aliases, "load_tensors_metadata", return_value=dict.fromkeys(checkpoint_keys)
    ) as read:
        layer_key_aliases.remap_sharded_state_dict_for_load([model], state_dict, "/selected/iter_5")
    return read


@pytest.mark.parametrize("checkpoint_keys", [{FUSED}, {FUSED, LEGACY}, set()])
def test_canonical_keys_are_preferred_and_missing_keys_are_not_hidden(checkpoint_keys):
    original = tensor(FUSED)
    state_dict = {"model": {"weight": original}}
    remap(model_with_maps(KEYS_MAP), state_dict, checkpoint_keys)
    assert state_dict["model"]["weight"] is original


def test_mixed_layer_names_and_stock_mla_mtp_mapping():
    mla_map = {"self_attention.linear_qkv_down_proj.layer_norm_": "input_layernorm."}
    model = model_with_maps(KEYS_MAP, mla_map)
    canonical_layer = "decoder.layers.1.self_attention.linear_qkv.layer_norm_weight"
    mtp = "mtp.layers.0.transformer_layer.input_layernorm.weight"
    state_dict = {"model": [tensor(FUSED), tensor(canonical_layer), tensor(mtp)]}
    remap(model, state_dict, {LEGACY, canonical_layer, mtp})
    assert [value.key for value in state_dict["model"]] == [LEGACY, canonical_layer, mtp]
    assert [layer.submodules_config.sharded_state_dict_keys_map for layer in model.layers] == [
        KEYS_MAP,
        mla_map,
    ]


def test_prefix_diagnostics_observe_resolved_storage_keys():
    from megatron.inference import utils

    def build(key, data, replica_id, flattened_range):
        return {"part": ShardedTensor.from_rank_offsets(key + ".part", data, replica_id=replica_id)}

    model_sd = {
        "weight": ShardedTensorFactory(FUSED, torch.ones(4), build, lambda state: state["part"])
    }
    requested = []
    modifier = utils._get_checkpoint_model_modifier(
        SimpleNamespace(checkpoint_model_prefix="teacher."), requested
    )
    modifier(model_sd)
    storage_key = "teacher." + LEGACY + ".part"
    remap(model_with_maps(KEYS_MAP), {"model": model_sd}, {storage_key})
    assert set(utils._checkpoint_tensor_keys(requested[0])) == {storage_key}


def test_factory_resolves_stored_keys_without_changing_build_or_merge_contract():
    def build(key, data, replica_id, flattened_range):
        return {"part": ShardedTensor.from_rank_offsets(key + ".part", data, replica_id=replica_id)}

    factory = ShardedTensorFactory(FUSED, torch.ones(4), build, lambda state: state["part"])
    state_dict = {"model": {"weight": factory}}
    remap(model_with_maps(KEYS_MAP), state_dict, {LEGACY + ".part"})
    loaded = state_dict["model"]["weight"]
    assert loaded.build()["part"].key == LEGACY + ".part"
    assert loaded.merge_fn is factory.merge_fn
    assert loaded.data is factory.data
    assert factory.build()["part"].key == FUSED + ".part"


def test_ambiguous_aliases_fail_instead_of_picking_a_mapping():
    model = model_with_maps(
        {"a.": "self_attention.linear_qkv.layer_norm_"},
        {"b.": "self_attention.linear_qkv.layer_norm_"},
    )
    with pytest.raises(ValueError, match="Ambiguous checkpoint aliases"):
        remap(
            model, {"weight": tensor(FUSED)}, {"decoder.layers.a.weight", "decoder.layers.b.weight"}
        )


def test_no_maps_do_not_read_checkpoint_metadata():
    read = remap(model_with_maps({}), {"weight": tensor(FUSED)}, {LEGACY})
    read.assert_not_called()
