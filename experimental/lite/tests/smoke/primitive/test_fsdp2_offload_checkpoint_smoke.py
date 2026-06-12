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
    fsdp2_available,
    wrap_fsdp2,
)
from megatron.lite.primitive.optimizers.fsdp2.adamw import iter_torch_optimizers, to_local_tensor
from megatron.lite.primitive.parallel.state import ParallelState
from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
from megatron.lite.runtime.contracts.config import ParallelConfig
from megatron.lite.runtime.contracts.handle import ModelHandle

pytestmark = [pytest.mark.mlite, pytest.mark.smoke, pytest.mark.gpu, pytest.mark.distributed]


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
        pytest.skip("CUDA is required for FSDP2 smoke tests.")
    if not fsdp2_available():
        pytest.skip("Installed PyTorch does not expose FSDP2 fully_shard.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 8:
        pytest.skip("Megatron Lite smoke tests are capped at single-node 8 GPUs.")

    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29521")

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


def _shared_tmp_path(tmp_path) -> str:
    payload = [str(tmp_path) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _checkpoint_config():
    return SimpleNamespace(parallel=ParallelConfig())


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


def _local_named_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: to_local_tensor(param.detach()).cpu().clone()
        for name, param in model.named_parameters()
    }


def _train_step(model: nn.Module, optimizer, x: torch.Tensor, target: torch.Tensor):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x).float(), target.float())
    loss.backward()
    success, grad_norm, _ = optimizer.step()
    assert success
    assert torch.isfinite(torch.tensor(grad_norm))
    return loss.detach(), float(grad_norm)


def _assert_grad_norm_exact(lhs: float, rhs: float) -> None:
    assert lhs == rhs


def _assert_local_params_close(lhs: nn.Module, rhs: nn.Module):
    lhs_params = _local_named_params(lhs)
    rhs_params = _local_named_params(rhs)
    assert lhs_params.keys() == rhs_params.keys()
    for name in lhs_params:
        torch.testing.assert_close(lhs_params[name], rhs_params[name], atol=0.0, rtol=0.0)


def test_fsdp2_runtime_model_and_optimizer_offload_roundtrip_single_node():
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


def test_fsdp2_offload_fraction_matches_non_offloaded_grad_clip_single_node():
    if dist.get_world_size() < 2:
        pytest.skip(
            "multi-rank FSDP2 grad clipping equivalence requires WORLD_SIZE > 1."
        )

    baseline_model, baseline_ps = _build_fsdp2_model()
    baseline_optimizer = _build_optimizer(
        baseline_model, baseline_ps, offload_fraction=0.0
    )
    offload_model, offload_ps = _build_fsdp2_model()
    offload_optimizer = _build_optimizer(
        offload_model, offload_ps, offload_fraction=1.0
    )

    assert _optimizer_state_devices(baseline_optimizer) == {"cuda"}
    assert _optimizer_state_devices(offload_optimizer) == {"cpu"}

    torch.manual_seed(4321)
    x = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    _loss, baseline_grad_norm = _train_step(
        baseline_model, baseline_optimizer, x, target
    )
    _loss, offload_grad_norm = _train_step(offload_model, offload_optimizer, x, target)

    _assert_grad_norm_exact(offload_grad_norm, baseline_grad_norm)
    _assert_local_params_close(offload_model, baseline_model)
    assert _local_param_devices(offload_model) == {"cuda"}
    assert _optimizer_state_devices(offload_optimizer) == {"cpu"}


def test_fsdp2_checkpoint_load_matches_uninterrupted_training_single_node(tmp_path):
    model_for_ckpt, ps = _build_fsdp2_model()
    optimizer_for_ckpt = _build_optimizer(model_for_ckpt, ps, offload_fraction=0.0)
    direct_model, direct_ps = _build_fsdp2_model()
    direct_optimizer = _build_optimizer(direct_model, direct_ps, offload_fraction=0.0)
    loaded_model, loaded_ps = _build_fsdp2_model()
    loaded_optimizer = _build_optimizer(loaded_model, loaded_ps, offload_fraction=0.0)
    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)

    torch.manual_seed(4321)
    x0 = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    y0 = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)
    x1 = torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
    y1 = torch.randn(4, 4, device="cuda", dtype=torch.bfloat16)

    _train_step(model_for_ckpt, optimizer_for_ckpt, x0, y0)
    _train_step(direct_model, direct_optimizer, x0, y0)
    checkpoint_dir = _shared_tmp_path(tmp_path)

    runtime.save_checkpoint(
        ModelHandle(
            model=model_for_ckpt,
            optimizer=optimizer_for_ckpt,
            parallel_state=ps,
            config=_checkpoint_config(),
            _extras={"model_chunks": [model_for_ckpt]},
        ),
        checkpoint_dir,
        step=1,
    )
    assert (
        runtime.load_checkpoint(
            ModelHandle(
                model=loaded_model,
                optimizer=loaded_optimizer,
                parallel_state=loaded_ps,
                config=_checkpoint_config(),
                _extras={"model_chunks": [loaded_model]},
            ),
            checkpoint_dir,
        )
        == 1
    )

    _train_step(direct_model, direct_optimizer, x1, y1)
    _train_step(loaded_model, loaded_optimizer, x1, y1)
    _assert_local_params_close(direct_model, loaded_model)
