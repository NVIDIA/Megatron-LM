# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from megatron.lite.primitive.optimizers.fsdp2 import (
    FSDP2Config,
    build_fsdp2_adamw,
    build_fsdp2_device_mesh,
    build_fsdp2_training_optimizer,
    fsdp2_available,
    wrap_fsdp2,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import iter_torch_optimizers, to_local_tensor
from megatron.lite.primitive.parallel import init_parallel
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = pytest.mark.gpus(1)


class TinyUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.linear(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.unit0 = TinyUnit()
        self.unit1 = TinyUnit()
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        return self.out(self.unit1(self.unit0(x)))


class TinyExpertBank(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.gelu(self.up(x)))


class TinyMoEUnit(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        self.experts = TinyExpertBank(hidden_size)

    def forward(self, x):
        return x + self.experts(self.dense(x))


class TinyPipelineMoEModel(nn.Module):
    def __init__(self, hidden_size: int = 64, num_units: int = 3):
        super().__init__()
        self.units = nn.ModuleList(TinyMoEUnit(hidden_size) for _ in range(num_units))

    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return x


class MemoryStressExperts(nn.Module):
    def __init__(self, numel: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(numel, device="cuda", dtype=torch.bfloat16)
        )

    def forward(self, x):
        return x + self.weight[0] * 0


class MemoryStressMoEUnit(nn.Module):
    def __init__(self, expert_numel: int):
        super().__init__()
        self.dense_scale = nn.Parameter(
            torch.ones(1, device="cuda", dtype=torch.bfloat16)
        )
        self.experts = MemoryStressExperts(expert_numel)

    def forward(self, x):
        return self.experts(x * self.dense_scale[0])


class MemoryStressMoEModel(nn.Module):
    def __init__(self, expert_numel: int, num_units: int):
        super().__init__()
        self.units = nn.ModuleList(
            MemoryStressMoEUnit(expert_numel) for _ in range(num_units)
        )

    def forward(self, x):
        for unit in self.units:
            x = unit(x)
        return x


@pytest.fixture(scope="module", autouse=True)
def _single_node_cuda_dist():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FSDP2 offload tests.")
    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    created_pg = False
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        created_pg = True
    yield
    if created_pg and dist.is_initialized():
        dist.destroy_process_group()


def _parallel_state() -> ParallelState:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return ParallelState(
        dp_group=dist.group.WORLD,
        dp_cp_group=dist.group.WORLD,
        dp_size=world_size,
        dp_cp_size=world_size,
        dp_rank=rank,
        dp_cp_rank=rank,
    )


def _build_fsdp2_model(dtype: torch.dtype = torch.bfloat16) -> tuple[nn.Module, ParallelState]:
    torch.manual_seed(1234)
    model = TinyModel().cuda().to(dtype=dtype)
    ps = _parallel_state()
    config = FSDP2Config(unit_modules=(TinyUnit,), reshard_after_forward=True)
    mesh = build_fsdp2_device_mesh(ps, config)
    return wrap_fsdp2(model, ps, config, mesh=mesh), ps


def _build_optimizer(model: nn.Module, ps: ParallelState, *, offload_fraction: float):
    return build_fsdp2_adamw(
        [model],
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            clip_grad=1.0,
            offload_fraction=offload_fraction,
        ),
        ps,
        use_fp32_master=True,
    )


def _local_param_devices(model: nn.Module) -> set[str]:
    return {to_local_tensor(param.detach()).device.type for param in model.parameters()}


def _optimizer_state_devices(optimizer) -> set[str]:
    devices: set[str] = set()
    for child in iter_torch_optimizers(optimizer.optimizer):
        for param_state in getattr(child, "state", {}).values():
            if not isinstance(param_state, dict):
                continue
            for value in param_state.values():
                if isinstance(value, torch.Tensor):
                    devices.add(to_local_tensor(value).device.type)
    return devices


def test_fsdp2_runtime_model_and_optimizer_offload_roundtrip_single_gpu():
    model, ps = _build_fsdp2_model()
    optimizer = _build_optimizer(model, ps, offload_fraction=0.0)
    handle = ModelHandle(
        model=model, optimizer=optimizer, parallel_state=ps, _extras={"model_chunks": [model]}
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    assert _local_param_devices(model) == {"cuda"}
    assert _optimizer_state_devices(optimizer) == {"cuda"}

    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    assert _local_param_devices(model) == {"cpu"}
    assert _optimizer_state_devices(optimizer) == {"cpu"}

    runtime.to(handle, "cuda", model=True, optimizer=True, grad=True)
    assert _local_param_devices(model) == {"cuda"}
    assert _optimizer_state_devices(optimizer) == {"cuda"}


def test_fsdp2_offload_fraction_keeps_optimizer_update_state_on_cpu_single_gpu():
    model, ps = _build_fsdp2_model()
    optimizer = _build_optimizer(model, ps, offload_fraction=1.0)

    assert _optimizer_state_devices(optimizer) == {"cpu"}

    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss.backward()
    success, grad_norm, _ = optimizer.step()

    assert success
    assert torch.isfinite(torch.tensor(grad_norm))
    assert _local_param_devices(model) == {"cuda"}
    assert _optimizer_state_devices(optimizer) == {"cpu"}


@pytest.mark.gpus(8)
def test_fsdp2_pp_edp_reshard_and_offload_roundtrip_eight_gpus():
    if dist.get_world_size() != 8:
        pytest.skip("PP2+CP2+EP2/EDP2 coverage requires torchrun with exactly 8 ranks.")

    ps = init_parallel(SimpleNamespace(tp=1, pp=2, cp=2, ep=2, etp=1, vpp=1))
    assert ps.dp_cp_size == 4
    assert ps.expert_dp_size == 2

    torch.manual_seed(1234)
    model = TinyPipelineMoEModel().cuda().to(dtype=torch.bfloat16)
    expert_global_numels = {
        name: param.numel()
        for name, param in model.named_parameters()
        if ".experts." in name
    }
    optimizer = build_fsdp2_training_optimizer(
        [model],
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            clip_grad=1.0,
            offload_fraction=1.0,
        ),
        ps,
        unit_modules=(TinyMoEUnit,),
        leaf_module_names=(),
        expert_classifier=lambda name: ".experts." in f".{name}",
        reshard_after_forward=True,
        forward_prefetch_depth=0,
        backward_prefetch_depth=0,
        use_fp32_shards=False,
        use_fp32_master=True,
    )

    def assert_experts_are_local_shards(device: str) -> None:
        named_params = dict(model.named_parameters())
        for name, global_numel in expert_global_numels.items():
            local = to_local_tensor(named_params[name].detach())
            assert local.device.type == device
            assert local.numel() == global_numel // ps.expert_dp_size

    def assert_grad_devices(device: str) -> None:
        grads = [param.grad for param in model.parameters()]
        assert all(grad is not None for grad in grads)
        assert {to_local_tensor(grad).device.type for grad in grads if grad is not None} == {
            device
        }

    def check_expert_shard_after_forward(_module, _inputs, _output) -> None:
        assert_experts_are_local_shards("cuda")

    hooks = [
        unit.experts.register_forward_hook(check_expert_shard_after_forward)
        for unit in model.units
    ]

    def train_backward(seed: int) -> None:
        torch.manual_seed(seed)
        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        optimizer.zero_grad()
        model(x).float().square().mean().backward()
        torch.cuda.synchronize()
        assert_experts_are_local_shards("cuda")
        assert_grad_devices("cuda")

    assert_experts_are_local_shards("cuda")
    train_backward(4321)

    handle = ModelHandle(
        model=model, optimizer=optimizer, parallel_state=ps, _extras={"model_chunks": [model]}
    )
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    assert_experts_are_local_shards("cpu")
    assert_grad_devices("cpu")

    runtime.to(handle, "cuda", model=True, optimizer=True, grad=True)
    assert_experts_are_local_shards("cuda")
    assert_grad_devices("cuda")
    train_backward(4322)
    for hook in hooks:
        hook.remove()

    runtime.to(handle, "cpu", model=True, optimizer=True, grad=True)
    torch.cuda.empty_cache()

    # Ten 512-MiB experts leave about 2.5 GiB resident with EDP2. Restrict
    # the allocator to the resident shards plus three full expert buffers.
    # Correct post-forward resharding stays below the limit; retaining expert
    # materializations across layers must OOM and fail this test.
    expert_bytes = 512 * 1024**2
    expert_numel = expert_bytes // torch.empty((), dtype=torch.bfloat16).element_size()
    stress_model = MemoryStressMoEModel(expert_numel, num_units=10)
    stress_optimizer = build_fsdp2_training_optimizer(
        [stress_model],
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            clip_grad=1.0,
            offload_fraction=0.0,
        ),
        ps,
        unit_modules=(MemoryStressMoEUnit,),
        leaf_module_names=(),
        expert_classifier=lambda name: ".experts." in f".{name}",
        reshard_after_forward=True,
        forward_prefetch_depth=0,
        backward_prefetch_depth=0,
        use_fp32_shards=False,
        use_fp32_master=False,
    )
    del stress_optimizer
    torch.cuda.empty_cache()
    resident_bytes = torch.cuda.memory_allocated()
    stress_shard_checks = []

    def check_stress_expert_shard(_module, _inputs, _output) -> None:
        local = to_local_tensor(_module.weight.detach())
        assert local.numel() == expert_numel // ps.expert_dp_size
        stress_shard_checks.append(True)

    stress_hooks = [
        unit.experts.register_forward_hook(check_stress_expert_shard)
        for unit in stress_model.units
    ]
    device_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    memory_limit_bytes = resident_bytes + 3 * expert_bytes
    memory_fraction = min(0.9, memory_limit_bytes / device_bytes)
    torch.cuda.set_per_process_memory_fraction(memory_fraction)
    try:
        with torch.no_grad():
            stress_output = stress_model(
                torch.ones(1, device="cuda", dtype=torch.bfloat16)
            )
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() <= resident_bytes + expert_bytes
        assert len(stress_shard_checks) == len(stress_model.units)
        del stress_output
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)
        for hook in stress_hooks:
            hook.remove()

    stress_handle = ModelHandle(
        model=stress_model,
        optimizer=None,
        parallel_state=ps,
        _extras={"model_chunks": [stress_model]},
    )
    runtime.to(stress_handle, "cpu", model=True, optimizer=False, grad=False)
    del stress_model, stress_handle
    torch.cuda.empty_cache()

    # Exercise the context-switch case that motivated the explicit reshard
    # barrier: parameters are intentionally left materialized after forward,
    # then runtime.to(..., "cpu") must reshard them before Module.to() sees
    # their DTensor state.
    materialized_model = TinyPipelineMoEModel(hidden_size=512, num_units=1).cuda().to(
        dtype=torch.bfloat16
    )
    materialized_expert_numels = {
        name: param.numel()
        for name, param in materialized_model.named_parameters()
        if ".experts." in name
    }
    build_fsdp2_training_optimizer(
        [materialized_model],
        SimpleNamespace(
            optimizer="adam",
            lr=1.0e-3,
            weight_decay=0.0,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1.0e-8,
            clip_grad=1.0,
            offload_fraction=0.0,
        ),
        ps,
        unit_modules=(TinyMoEUnit,),
        leaf_module_names=(),
        expert_classifier=lambda name: ".experts." in f".{name}",
        reshard_after_forward=False,
        forward_prefetch_depth=0,
        backward_prefetch_depth=0,
        use_fp32_shards=False,
        use_fp32_master=False,
    )
    x = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
    output = materialized_model(x)
    del output, x
    torch.cuda.synchronize()

    materialized_handle = ModelHandle(
        model=materialized_model,
        optimizer=None,
        parallel_state=ps,
        _extras={"model_chunks": [materialized_model]},
    )
    runtime.to(materialized_handle, "cpu", model=True, optimizer=False, grad=False)
    for name, param in materialized_model.named_parameters():
        local = to_local_tensor(param.detach())
        assert local.device.type == "cpu"
        if name in materialized_expert_numels:
            assert local.numel() == materialized_expert_numels[name] // ps.expert_dp_size

    runtime.to(materialized_handle, "cuda", model=True, optimizer=False, grad=False)
    assert _local_param_devices(materialized_model) == {"cuda"}
