# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 Engram, checkpoint and pipeline state transport contracts."""
import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.lite.model.deepseek_v41.lite.checkpoint import (
    attach_checkpoint_metadata,
    load_hf_weights,
)
from megatron.lite.model.deepseek_v41.lite.protocol import forward_step
from megatron.lite.runtime.contracts import PackedBatch


@pytest.mark.parametrize("rank", [0, 1])
def test_forward_step_rows_are_original_unpadded_rows(rank):
    class Rows(torch.nn.Module):
        ps = SimpleNamespace(cp_size=2, cp_rank=rank, cp_group=object())

        def forward(self, input_ids, loss_mask, **_):  # physical rows, as the model reports
            n = input_ids.numel()
            rows = torch.arange(n) + rank * n
            return {"log_probs": torch.zeros(n), "rows": rows, "valid_mask": loss_mask.bool()}

    ids = torch.arange(12)
    output = forward_step(Rows(), PackedBatch(ids, ids, torch.tensor([5, 1, 6])))
    expected = [0, 1, 2, 3, 4, -1, -1, -1, 5, -1] if rank == 0 else [-1, -1, *range(6, 12), -1, -1]
    assert output["rows"].tolist() == expected
    assert not output["valid_mask"][output["rows"] < 0].any()
    assert output["valid_mask"].sum() == (4 if rank == 0 else 5)


def test_checkpoint_metadata_keeps_engram_rows_at_global_offsets(tmp_path):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/store", rank=0, world_size=1)
    try:
        group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)  # rank 1 of a 2-way row shard
        table = torch.nn.Module()
        table.register_buffer("weight", torch.ones(3, 4, dtype=torch.bfloat16))
        table.lookup = SimpleNamespace(group=group, boundaries=(0, 3, 6))
        engram = torch.nn.Module()
        engram.embed, engram.wkv = table, torch.nn.Linear(4, 4, bias=False)
        engram.register_buffer("cache", torch.zeros(2), persistent=False)
        native = MegatronModule(
            TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
        )
        native.tp_group, native.bias = dist.group.WORLD, torch.nn.Parameter(torch.zeros(4))
        model = torch.nn.Module()
        model.engrams, model.native = torch.nn.ModuleDict({"1": engram}), native
        attach_checkpoint_metadata(
            [model], SimpleNamespace(dp_cp_rank=1, dp_cp_group=dist.group.WORLD)
        )
        state = model.sharded_state_dict()
        assert set(model.state_dict()) == set(state)
        assert model._mlite_dist_opt_sharded_state_dict
        assert set(state) == {"engrams.1.embed.weight", "engrams.1.wkv.weight", "native.bias"}
        rows = state["engrams.1.embed.weight"]
        assert rows.global_shape == (6, 4) and rows.global_offset == (3, 0)
        assert rows.replica_id == (0, 0, 0)  # the row group covers dp x cp: no replicas
        table.lookup.boundaries = (0, 2, 6)
        with pytest.raises(ValueError, match="equal Core row shards"):
            model.sharded_state_dict()
    finally:
        dist.destroy_process_group()


def test_release_load_fails_on_unread_stage_keys_or_unloaded_state(tmp_path):
    names = [
        "layers.0.attn_norm.weight",
        "layers.1.attn_norm.weight",
        "layers.0.ffn.gate.bias_vl",
        "mtp.x",
    ]
    tensors = {n: torch.ones(4, dtype=torch.bfloat16) for n in names}
    tensors["embed.weight"] = torch.ones(2, 4, dtype=torch.float8_e4m3fn)
    tensors["embed.weight_scale_inv"] = torch.full((1, 1), 2.0)
    save_file(tensors, tmp_path / "a.safetensors")
    names += ["embed.weight", "embed.weight_scale_inv"]
    index = {"weight_map": {n: "a.safetensors" for n in names}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    def stage():  # first stage holding layer 0; layer 1 and MTP belong elsewhere
        model, layer = torch.nn.Module(), torch.nn.Module()
        layer.attn_norm = torch.nn.Module()
        layer.attn_norm.weight = torch.nn.Parameter(torch.zeros(4))
        model.embed, model.embed.weight = torch.nn.Module(), torch.nn.Parameter(torch.zeros(2, 4))
        model.layers, model.local_layer_range = torch.nn.ModuleDict({"0": layer}), (0, 1)
        model.layout = SimpleNamespace(has_embed=True, has_head=False)
        return model

    cfg, ps = SimpleNamespace(n_routed_experts=1), SimpleNamespace(ep_size=1, ep_rank=0)
    model = stage()
    load_hf_weights(model, str(tmp_path), cfg, ps)
    assert torch.equal(model.layers["0"].attn_norm.weight, torch.ones(4))
    assert torch.equal(model.embed.weight, torch.full((2, 4), 2.0))  # dequantized
    unread = stage()
    del unread.embed  # a stage-local release key nobody reads
    with pytest.raises(ValueError, match=r"unread \['embed.weight'\]"):
        load_hf_weights(unread, str(tmp_path), cfg, ps)
    unloaded = stage()
    unloaded.layers["0"].register_buffer("state", torch.zeros(1))  # persistent, unmapped
    with pytest.raises(ValueError, match=r"unloaded \['layers.0.state'\]"):
        load_hf_weights(unloaded, str(tmp_path), cfg, ps)

    key = "layers.0.ffn.gate.bias_vl_extra"  # near-match must not be ignored
    tensors[key] = torch.ones(1)
    save_file(tensors, tmp_path / "a.safetensors")
    index["weight_map"][key] = "a.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="unread.*bias_vl_extra"):
        load_hf_weights(stage(), str(tmp_path), cfg, ps)


@pytest.mark.gpus(1)
def test_native_fp4_fp8_gpu_decode_matches_cpu_bits(tmp_path):
    from megatron.lite.primitive.ckpt.hf_weights import SafeTensorReader

    codes = torch.arange(256, dtype=torch.uint8).repeat(2)
    codes[(codes & 127) == 127] = 0  # every finite E4M3 byte, exclude the two NaNs
    tensors = {
        "expert.weight": torch.arange(256, dtype=torch.uint8).view(torch.int8).reshape(8, 32),
        "expert.scale": torch.arange(120, 136, dtype=torch.uint8)
        .view(torch.float8_e8m0fnu)
        .reshape(8, 2),
        "attention.weight": codes.view(torch.float8_e4m3fn).reshape(8, 64),
        "attention.scale": torch.tensor([[120, 125]], dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    with SafeTensorReader(tmp_path) as reader:
        for key in ("expert.weight", "attention.weight"):
            args = dict(target_shape=torch.Size((8, 64)), target_dtype=torch.bfloat16)
            cpu = reader.get_tensor(key, device="cpu", **args)
            gpu = reader.get_tensor(key, device="cuda", **args)
            for dtype in (torch.float32, torch.bfloat16):
                assert torch.equal(
                    cpu.to(dtype).view(torch.uint8), gpu.to(dtype).cpu().view(torch.uint8)
                )
