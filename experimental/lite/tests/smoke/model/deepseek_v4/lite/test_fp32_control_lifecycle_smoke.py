# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Gate DS4 Lite FP32 controls through load, FSDP2, and block-FP8 export."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
from megatron.lite.model.deepseek_v4.lite.checkpoint import (
    export_hf_weights,
    load_hf_weights,
    save_hf_weights,
)
from megatron.lite.model.deepseek_v4.lite.model import DeepseekV4Layer, DeepseekV4Model
from megatron.lite.model.deepseek_v4.lite.protocol import (
    MODULE_MAP,
    _cast_training_parameters,
    _is_native_fp32_control,
)
from megatron.lite.primitive.modules.router_replay import (
    RouterReplay,
    RouterReplayAction,
    attach_router_replay,
    detach_router_replay,
)
from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.primitive.recompute import apply_recompute


def _config() -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=128,
        hidden_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=128,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        o_groups=1,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        compress_ratios=[4, 4],
        num_hash_layers=1,
        index_head_dim=128,
        index_n_heads=4,
        index_topk=16,
        num_nextn_predict_layers=1,
        expert_dtype="fp8",
        quantization_config={"weight_block_size": [128, 128]},
    )


def _parallel_state() -> ParallelState:
    return ParallelState(
        dp_group=dist.group.WORLD,
        dp_cp_group=dist.group.WORLD,
        ep_dp_group=dist.group.WORLD,
        dp_size=1,
        dp_cp_size=1,
        expert_dp_size=1,
    )


def _model(config: DeepseekV4Config, ps: ParallelState) -> DeepseekV4Model:
    model = DeepseekV4Model(
        config,
        SimpleNamespace(vpp=None, fp8=False),
        ps,
        mtp_enable=True,
    )
    _cast_training_parameters(model)
    return model.cuda()


def _init_dist() -> None:
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")
        dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _optimizer(model: DeepseekV4Model, ps: ParallelState):
    return build_fsdp2_training_optimizer(
        [model],
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-6,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            clip_grad=1.0,
            offload_fraction=0.0,
        ),
        ps,
        unit_modules=(DeepseekV4Layer,),
        replicated_param_classifier=(
            lambda name, _parameter: _is_native_fp32_control(name)
        ),
        use_fp32_shards=False,
        cast_forward_inputs=True,
    )


def _fp32_controls(model: DeepseekV4Model) -> dict[str, torch.Tensor]:
    controls = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if _is_native_fp32_control(name)
    }
    controls.update(
        {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
            if name.endswith(".mlp.gate.expert_bias")
        }
    )
    return controls


def _release_fp32_controls(weights) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in weights
        if name.startswith("hc_head_")
        or ".hc_attn_" in name
        or ".hc_ffn_" in name
        or ".hc_head_" in name
        or name.endswith(".attn.attn_sink")
        or name.endswith(".ape")
        or name.endswith(".ffn.gate.bias")
    }


def test_fp32_controls_survive_real_lite_load_fsdp2_and_block_fp8_export(tmp_path) -> None:
    if not torch.cuda.is_available():
        return
    _init_dist()
    config = _config()
    ps = _parallel_state()
    source = _model(config, ps)
    source_controls = _fp32_controls(source)
    assert any(name.endswith(".sinks") for name in source_controls)
    assert any(".attn_hc.fn" in name for name in source_controls)
    assert any(name.startswith("mtp.0.hc_head.") for name in source_controls)
    assert any(name.endswith(".mlp.gate.expert_bias") for name in source_controls)

    save_hf_weights(source, str(tmp_path), config, ps)
    del source
    torch.cuda.empty_cache()

    loaded = _model(config, ps)
    load_hf_weights(loaded, str(tmp_path), config, ps)
    loaded_controls = _fp32_controls(loaded)
    assert loaded_controls.keys() == source_controls.keys()
    for name, expected in source_controls.items():
        assert loaded_controls[name].dtype == torch.float32, name
        assert torch.equal(loaded_controls[name], expected), name
    expected_release_controls = _release_fp32_controls(
        export_hf_weights(loaded, config, ps)
    )

    _optimizer(loaded, ps)

    release_controls = _release_fp32_controls(
        export_hf_weights(
            loaded,
            config,
            ps,
            target="block_fp8",
            resync_config={"expert_dtype": "fp8"},
        )
    )
    assert release_controls.keys() == expected_release_controls.keys()
    for name, tensor in release_controls.items():
        assert tensor.dtype == torch.float32, name
        assert torch.equal(tensor, expected_release_controls[name]), name


def test_r3_replay_survives_real_lite_full_recompute_and_fsdp2() -> None:
    if not torch.cuda.is_available():
        return
    _init_dist()
    config = _config()
    config.num_nextn_predict_layers = 0
    ps = _parallel_state()
    model = _model(config, ps)
    apply_recompute(list(model.layers.values()), ["full"], MODULE_MAP)
    _optimizer(model, ps)

    RouterReplay.clear_global_router_replay_instances()
    routers = sum(
        attach_router_replay(layer, reset=False) for layer in model.layers.values()
    )
    attached = [layer.mlp.gate.router_replay for layer in model.layers.values()]
    assert routers == config.num_hidden_layers
    assert [id(item) for item in attached] == [
        id(item) for item in RouterReplay.global_router_replay_instances
    ]

    input_ids = torch.arange(16, device="cuda").unsqueeze(0) % config.vocab_size
    position_ids = torch.arange(16, device="cuda").unsqueeze(0)
    try:
        RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
        with torch.no_grad():
            native = model(
                input_ids=input_ids, position_ids=position_ids, enable_mtp=False
            )["logits"]
        routes = [route.detach().clone() for route in RouterReplay.get_recorded_data()]
        assert all(route is not None for route in routes)

        RouterReplay.set_replay_data(routes, replay_mask=torch.ones(16, dtype=torch.bool))
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        RouterReplay.reset_replay_stats()
        with torch.no_grad():
            replayed = model(
                input_ids=input_ids, position_ids=position_ids, enable_mtp=False
            )["logits"]
        assert RouterReplay.replay_stats()["calls"] == config.num_hidden_layers
        assert torch.equal(replayed, native)
    finally:
        for layer in model.layers.values():
            detach_router_replay(layer)
        RouterReplay.clear_global_router_replay_instances()
