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
    _cast_training_parameters,
    _is_native_fp32_control,
)
from megatron.lite.primitive.optimizers.fsdp2 import build_fsdp2_training_optimizer
from megatron.lite.primitive.parallel.state import ParallelState


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
    if not dist.is_initialized():
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29519")
        dist.init_process_group("nccl", init_method="env://")

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
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

    build_fsdp2_training_optimizer(
        [loaded],
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
        use_fp32_shards=False,
        cast_forward_inputs=True,
    )

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
