# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU/Gloo contracts for colocated actor/rollout memory admission."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

VERL_ROOT = Path(__file__).resolve().parents[3] / "examples" / "verl"
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

pytestmark = [pytest.mark.mlite, pytest.mark.optional]


@pytest.fixture
def memory_module():
    import verl_mlite.optimizer_memory as module

    return module


@pytest.fixture
def engine_module():
    return pytest.importorskip("verl_mlite.engine.mlite_engine")


def test_real_gloo_prewarm_deduplicates_groups_and_preserves_training_state(
    memory_module, tmp_path, monkeypatch
):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    assert not dist.is_initialized(), "This CPU unit owns its Gloo process group"
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        group = dist.new_group([0], backend="gloo")
        owners = [
            SimpleNamespace(get_grad_stats_parallel_group=lambda group=g: group)
            for g in (None, group, group)
        ]
        # A non-Core optimizer must not introduce an unrelated default-group SUM.
        owners.append(SimpleNamespace())
        parameter = torch.nn.Parameter(torch.tensor([1.25, -0.0]))
        parameter.grad = torch.tensor([2.0, -3.0])
        optimizer = SimpleNamespace(chained_optimizers=owners, state={"step": 7})
        before = (
            parameter.detach().view(torch.int32).clone(),
            parameter.grad.view(torch.int32).clone(),
            torch.get_rng_state(),
        )
        native_all_reduce, calls = dist.all_reduce, []

        def observe(value, **kwargs):
            calls.append((value.dtype, value.numel(), kwargs["group"], kwargs["op"]))
            return native_all_reduce(value, **kwargs)

        monkeypatch.setattr(dist, "all_reduce", observe)
        memory_module.prewarm_grad_stats_groups(optimizer, torch.device("cpu"))
        assert calls == [
            (torch.float32, 1, None, dist.ReduceOp.SUM),
            (torch.float32, 1, group, dist.ReduceOp.SUM),
        ]
        assert torch.equal(before[0], parameter.detach().view(torch.int32))
        assert torch.equal(before[1], parameter.grad.view(torch.int32))
        assert torch.equal(before[2], torch.get_rng_state())
        assert optimizer.state == {"step": 7}
    finally:
        dist.destroy_process_group()


def test_prewarm_rejects_corrupted_scratch_and_propagates_collective_error(
    memory_module, monkeypatch
):
    owner = SimpleNamespace(get_grad_stats_parallel_group=lambda: None)
    monkeypatch.setattr(dist, "all_reduce", lambda value, **kwargs: value.fill_(1))
    with pytest.raises(RuntimeError, match="changed zero scratch"):
        memory_module.prewarm_grad_stats_groups(owner, "cpu")

    def fail(*args, **kwargs):
        raise RuntimeError("original NCCL init error")

    monkeypatch.setattr(dist, "all_reduce", fail)
    with pytest.raises(RuntimeError, match="original NCCL init error"):
        memory_module.prewarm_grad_stats_groups(owner, "cpu")


def test_initialize_warms_before_offload_and_optimizer_releases_before_step(
    engine_module, monkeypatch
):
    engine = object.__new__(engine_module.MegatronLiteEngine)
    engine.engine_config = SimpleNamespace(
        full_determinism=False, param_offload=False, optimizer_offload=False, grad_offload=False
    )
    engine.device_name = "cuda"
    engine.model_config = SimpleNamespace(local_path="unused")
    engine._build_mlite_config = lambda: object()
    model, optimizer, calls = object(), object(), []
    handle = SimpleNamespace(_optimizer=optimizer, _lr_scheduler=object())
    runtime = SimpleNamespace(
        build_model=lambda: handle,
        optimizer_step=lambda actual: calls.append(("step", actual)) or (True, 3.25, 0),
    )
    monkeypatch.setattr(engine_module, "create_runtime", lambda *a, **kw: runtime)
    monkeypatch.setattr(engine_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        engine_module, "_release_inactive_cuda_cache", lambda: calls.append("release")
    )
    monkeypatch.setattr(
        engine_module,
        "_prewarm_grad_stats_groups",
        lambda actual, device: calls.append(("warm", actual, str(device))),
    )
    engine._extract_primary_module = lambda: model
    engine.to = lambda **kw: calls.append(("offload", kw))
    engine.initialize()
    assert calls == [
        "release",
        ("warm", optimizer, "cuda:0"),
        ("offload", dict(device="cpu", model=False, optimizer=False, grad=False)),
    ]
    calls.clear()
    assert engine.optimizer_step() == 3.25
    assert calls == ["release", ("step", handle)]
    assert engine.module is model and engine.handle is handle and handle._optimizer is optimizer


def test_release_only_synchronizes_and_returns_unused_allocator_blocks(memory_module, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    memory_module.release_inactive_cuda_cache()
    assert calls == ["synchronize", "empty_cache"]
