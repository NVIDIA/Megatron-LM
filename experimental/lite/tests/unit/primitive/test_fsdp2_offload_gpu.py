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
    fsdp2_available,
    wrap_fsdp2,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import iter_torch_optimizers, to_local_tensor
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.handle import ModelHandle


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
