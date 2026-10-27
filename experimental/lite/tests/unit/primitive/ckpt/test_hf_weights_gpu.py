# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real-device coverage for HF load/export collectives and CUDA state."""

from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import save_file

from megatron.lite.model.kimi_k2.config import KimiK2Config
from megatron.lite.model.kimi_k2.lite.checkpoint import KimiK2WeightSpec
from megatron.lite.model.qwen3_5.config import Qwen35Config
from megatron.lite.model.qwen3_5.lite.checkpoint import Qwen35WeightSpec
from megatron.lite.primitive.ckpt.hf_weights import (
    export_hf_weights,
    load_hf_weights,
)
from megatron.lite.primitive.quantization.qat import QATSpec, apply_qat_to_chunks


pytestmark = [pytest.mark.mlite, pytest.mark.gpus(1)]


@pytest.fixture(scope="module", autouse=True)
def _cuda_process_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for real-device HF weight tests.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    created_process_group = False
    if not dist.is_initialized():
        required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        missing = [name for name in required if name not in os.environ]
        assert not missing, (
            "Run real-device HF weight tests with torchrun; "
            f"missing environment variables: {missing}"
        )
        dist.init_process_group(backend="nccl", init_method="env://")
        created_process_group = True
    yield
    if created_process_group:
        dist.destroy_process_group()


def _qwen35_config(*, num_experts: int = 4) -> Qwen35Config:
    return Qwen35Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        vocab_size=16,
        num_experts=num_experts,
        num_experts_per_tok=2,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        linear_num_key_heads=2,
        linear_key_head_dim=2,
        linear_num_value_heads=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=2,
        layer_types=["full_attention"],
        partial_rotary_factor=1.0,
    )


def _parallel_state(*, tp_size: int = 1, ep_size: int = 1) -> SimpleNamespace:
    rank = dist.get_rank()
    return SimpleNamespace(
        pp_size=1,
        tp_size=tp_size,
        tp_rank=rank if tp_size > 1 else 0,
        tp_group=dist.group.WORLD if tp_size > 1 else None,
        ep_size=ep_size,
        ep_rank=rank if ep_size > 1 else 0,
        ep_group=dist.group.WORLD if ep_size > 1 else None,
        etp_size=1,
        etp_rank=0,
        etp_group=None,
    )


def _require_two_ranks() -> None:
    assert dist.get_world_size() == 2, (
        "EP2/TP2 real-device tests require exactly two torchrun ranks, "
        f"got {dist.get_world_size()}"
    )


def _shared_checkpoint(tensors: dict[str, torch.Tensor]) -> str:
    paths = [None]
    if dist.get_rank() == 0:
        path = tempfile.mkdtemp(prefix="mlite-hf-weights-gpu-")
        save_file(
            {name: tensor.detach().cpu() for name, tensor in tensors.items()},
            os.path.join(path, "model.safetensors"),
        )
        paths[0] = path
    dist.broadcast_object_list(
        paths,
        src=0,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    dist.barrier()
    assert isinstance(paths[0], str)
    return paths[0]


def _remove_shared_checkpoint(path: str) -> None:
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(path)
    dist.barrier()


def _expert_tensor(global_expert_id: int) -> torch.Tensor:
    base = torch.arange(8 * 8, device="cuda", dtype=torch.float32).reshape(8, 8)
    return base + global_expert_id * 1000


@pytest.mark.gpus(2)
def test_ep2_export_gather_matches_single_rank_reference_bitwise() -> None:
    _require_two_ranks()
    config = _qwen35_config()
    rank = dist.get_rank()

    class CudaMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()
            for local_idx in range(config.num_experts // 2):
                global_idx = rank * (config.num_experts // 2) + local_idx
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{local_idx}",
                    nn.Parameter(_expert_tensor(global_idx)),
                )

    model = CudaMoE()
    exported = dict(
        export_hf_weights(
            model,
            Qwen35WeightSpec(config),
            _parallel_state(ep_size=2),
            cpu=False,
        )
    )

    key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    expected = torch.stack(
        [_expert_tensor(global_idx) for global_idx in range(config.num_experts)]
    )
    assert exported.keys() == {key}
    assert exported[key].device.type == "cuda"
    assert torch.equal(exported[key], expected)


@pytest.mark.gpus(2)
def test_tp2_fused_gate_up_load_and_export_match_full_tensor_bitwise() -> None:
    _require_two_ranks()
    config = _qwen35_config()
    spec = Qwen35WeightSpec(config)
    native_name = "layers.0.moe.shared_expert.gate_up.linear.weight"
    gate_name, up_name = spec.weight_map()[native_name]
    gate = torch.arange(4 * 8, dtype=torch.float32).reshape(4, 8)
    up = torch.arange(1000, 1000 + 4 * 8, dtype=torch.float32).reshape(4, 8)
    checkpoint = _shared_checkpoint({gate_name: gate, up_name: up})

    class SharedExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.shared_expert = nn.Module()
            self.layers[0].moe.shared_expert.gate_up = nn.Module()
            self.layers[0].moe.shared_expert.gate_up.linear = nn.Linear(
                8, 4, bias=False, device="cuda"
            )

    try:
        model = SharedExpert()
        ps = _parallel_state(tp_size=2)
        load_hf_weights(model, checkpoint, spec, ps)

        expected_local = torch.cat(
            [
                gate.chunk(2, dim=0)[dist.get_rank()],
                up.chunk(2, dim=0)[dist.get_rank()],
            ],
            dim=0,
        ).cuda()
        local = model.layers[0].moe.shared_expert.gate_up.linear.weight
        assert local.device.type == "cuda"
        assert torch.equal(local, expected_local)

        exported = dict(export_hf_weights(model, spec, ps, cpu=False))
        assert exported.keys() == {gate_name, up_name}
        assert exported[gate_name].device.type == "cuda"
        assert exported[up_name].device.type == "cuda"
        assert torch.equal(exported[gate_name], gate.cuda())
        assert torch.equal(exported[up_name], up.cuda())
    finally:
        _remove_shared_checkpoint(checkpoint)


def test_persistent_router_buffer_roundtrips_on_cuda() -> None:
    config = KimiK2Config(
        num_hidden_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        intermediate_size=16,
        moe_intermediate_size=4,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=2,
    )
    spec = KimiK2WeightSpec(config)
    native_name = "layers.0.moe.router.expert_bias"
    hf_name = spec.weight_map()[native_name][0]
    expected = torch.tensor([1.25, -2.5], dtype=torch.float32)
    checkpoint = _shared_checkpoint({hf_name: expected})

    class RouterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.router = nn.Module()
            self.layers[0].moe.router.register_buffer(
                "expert_bias", torch.zeros(2, device="cuda")
            )

    try:
        model = RouterModel()
        ps = _parallel_state()
        load_hf_weights(model, checkpoint, spec, ps)
        buffer = model.layers[0].moe.router.expert_bias
        assert buffer.device.type == "cuda"
        assert native_name in model.state_dict()
        assert torch.equal(buffer, expected.cuda())

        exported = dict(export_hf_weights(model, spec, ps, cpu=False))
        assert exported.keys() == {hf_name}
        assert exported[hf_name].device.type == "cuda"
        assert torch.equal(exported[hf_name], expected.cuda())
    finally:
        _remove_shared_checkpoint(checkpoint)


def test_incomplete_packed_expert_group_fails_loud_on_cuda_moe() -> None:
    config = _qwen35_config()

    class IncompleteCudaMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.experts = nn.Module()
            self.layers[0].moe.experts.fc1 = nn.Module()
            for expert_idx in range(config.num_experts - 1):
                self.layers[0].moe.experts.fc1.register_parameter(
                    f"weight{expert_idx}",
                    nn.Parameter(_expert_tensor(expert_idx)),
                )

    model = IncompleteCudaMoE()
    assert {parameter.device.type for parameter in model.parameters()} == {"cuda"}
    with pytest.raises(
        RuntimeError,
        match=r"Qwen35WeightSpec.*layers\.0\.moe\.experts\.fc1\.packed.*3/4",
    ):
        list(
            export_hf_weights(
                model,
                Qwen35WeightSpec(config),
                _parallel_state(),
                cpu=False,
            )
        )


def test_qat_on_and_off_export_the_same_hf_keys_from_cuda_model() -> None:
    config = _qwen35_config()

    class SharedExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].moe = nn.Module()
            self.layers[0].moe.shared_expert = nn.Module()
            self.layers[0].moe.shared_expert.gate_up = nn.Module()
            self.layers[0].moe.shared_expert.gate_up.linear = nn.Linear(
                8, 8, bias=False, device="cuda"
            )

    torch.manual_seed(1234)
    qat_off = SharedExpert()
    qat_on = SharedExpert()
    qat_on.load_state_dict(qat_off.state_dict())
    stats = apply_qat_to_chunks(
        [qat_on],
        QATSpec(enabled=True, format="int8", group_size=-1),
    )
    assert stats["quantized_modules"] == 1
    master = qat_on.layers[
        0
    ].moe.shared_expert.gate_up.linear.parametrizations.weight.original
    assert master.device.type == "cuda"

    ps = _parallel_state()
    off_export = dict(export_hf_weights(qat_off, Qwen35WeightSpec(config), ps))
    on_export = dict(export_hf_weights(qat_on, Qwen35WeightSpec(config), ps))
    expected_keys = {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
    }
    assert off_export.keys() == expected_keys
    assert on_export.keys() == expected_keys
    assert all(tensor.device.type == "cuda" for tensor in off_export.values())
    assert all(tensor.device.type == "cuda" for tensor in on_export.values())
