# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for local optimizer tensors with checkpoint-only DTensors."""

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    load_checkpoint,
    microbatch,
    save_checkpoint,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.mixed_precision import MixedPrecisionPolicy
from megatron.core.optimizer.fully_sharded_optimizer import _local_grad_and_replication
from tests.unit_tests.dist_checkpointing import TempNamedDir


def _build(device, world_size, dtype, placement, deferred=False):
    torch.manual_seed(1234)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 4)).to(
        device=device, dtype=dtype
    )
    mesh = init_device_mesh(device.type, (world_size,))
    with fully_shard_context(device=device):
        fully_shard(
            model,
            mesh,
            Placements(
                dp_axes=[0],
                parameter=[placement],
                gradient=[Partial("avg") if deferred else placement],
                optimizer=[placement],
            ),
            mixed_precision_policy=MixedPrecisionPolicy(main_params_dtype=torch.float32),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, foreach=True)
    fully_shard_optimizer(optimizer)
    return model, optimizer


def _step(model, optimizer, inputs, set_to_none=True):
    optimizer.zero_grad(set_to_none=set_to_none)
    losses = []
    for index, x in enumerate(inputs):
        with microbatch(model.context, is_last=index == len(inputs) - 1):
            loss = model(x).float().square().mean()
            (loss / len(inputs)).backward()
            losses.append(loss.detach())
    optimizer.step()
    return torch.stack(losses)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("placement", [Shard(0), Replicate()], ids=["sharded", "replicated"])
@pytest.mark.parametrize("deferred", [False, True], ids=["immediate", "deferred"])
def test_optimizer_tensors_remain_local(distributed_setup, dtype, placement, deferred):
    """Accumulation and zeroing never construct runtime DTensor wrappers."""
    device = distributed_setup.device
    model, optimizer = _build(device, distributed_setup.world_size, dtype, placement, deferred)
    torch.manual_seed(42 + distributed_setup.rank)
    inputs = torch.randn(2, 3, 8, device=device, dtype=dtype)
    for set_to_none in (True, False, True):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            actual = _step(model, optimizer, inputs, set_to_none)
        assert torch.isfinite(actual).all()
        assert not any("_FromTorchTensor" in event.name for event in prof.events())
        for parameter in model.parameters():
            assert not isinstance(parameter, DTensor)
            assert not isinstance(parameter.grad, DTensor)
            local_grad, replication = _local_grad_and_replication(parameter)
            assert local_grad is parameter.grad
            assert replication == (distributed_setup.world_size if placement.is_replicate() else 1)
            for state in optimizer.state[parameter].values():
                assert not isinstance(state, DTensor)


def test_local_tensor_checkpoint_roundtrip(distributed_setup, tmp_path_dist_ckpt):
    """Checkpoint-only wrappers preserve restart values and optimizer parameter identity."""
    device = distributed_setup.device
    source, source_optimizer = _build(
        device, distributed_setup.world_size, torch.bfloat16, Shard(0)
    )
    target, target_optimizer = _build(
        device, distributed_setup.world_size, torch.bfloat16, Shard(0)
    )
    inputs = torch.randn(2, 3, 8, device=device, dtype=torch.bfloat16)
    _step(source, source_optimizer, inputs)
    identities = tuple(id(p) for p in target.parameters())
    with TempNamedDir(tmp_path_dist_ckpt / "local", sync=True) as path:
        save_checkpoint(source, source_optimizer, path)
        assert all(
            not isinstance(v, DTensor) for s in source_optimizer.state.values() for v in s.values()
        )
        load_checkpoint(target, target_optimizer, path)
    assert tuple(id(p) for p in target.parameters()) == identities
    assert all(not isinstance(p, DTensor) for p in target.parameters())
    assert all(
        not isinstance(v, DTensor) for s in target_optimizer.state.values() for v in s.values()
    )
    torch.testing.assert_close(
        _step(target, target_optimizer, inputs),
        _step(source, source_optimizer, inputs),
        rtol=0,
        atol=0,
    )
