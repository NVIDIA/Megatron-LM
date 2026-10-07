# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Independent residual units must preserve values, prefetch, and checkpoint state."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
    fully_shard_optimizer,
    load_checkpoint,
    save_checkpoint,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig, WideResidualConfig
from megatron.core.transformer.wide_residual_layer import StreamwiseSigmoidWideResidualConnection


class _Block(nn.Module):
    def __init__(self, *, retention=True):
        super().__init__()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            wide_residual=WideResidualConfig(num_streams=3, learned_retention=retention),
        )
        self.residual = StreamwiseSigmoidWideResidualConnection(
            config, 1, "mlp", ProcessGroupCollection()
        )
        self.branch = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        branch, state = self.residual(x, operation="read")
        return self.residual(
            self.branch(branch),
            operation="write",
            state=state,
            dropout_probability=0.0,
            training=self.training,
        )


def _shard(model, mesh, device):
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(device=device) as context:
        for module in (model.residual.reader, model.residual.writer, model.branch, model):
            fully_shard(module, mesh=mesh, placements=placements)
        context.set_forward_order(
            model, (model, model.residual.reader, model.branch, model.residual.writer)
        )
    return model


def _full_values(model, attribute):
    result = {}
    for owner, module in model.named_modules():
        if not isinstance(module, FsdpModule):
            continue
        for group in module.parameter_groups:
            buffer = getattr(group, attribute).redistribute([Replicate()] * group.mesh.ndim)
            for index, parameter in enumerate(group.fsdp_parameters):
                result[owner + "." + parameter.fqns[0]] = buffer.get_tensor_view(index)
    return result


@pytest.mark.parametrize("retention", [False, True])
def test_independent_ownership_preserves_legacy_model_keys(distributed_setup, retention):
    model = _Block(retention=retention).to(distributed_setup.device)
    residual = model.residual
    assert not set(residual.reader.parameters()) & set(residual.writer.parameters())
    state = model.state_dict()
    expected = {"residual.read_map.logit", "residual.write_map.logit", "branch.weight"}
    if retention:
        expected.add("residual.retention.retention_logit")
    assert set(state) == expected
    restored = _Block(retention=retention).to(distributed_setup.device)
    restored.load_state_dict(state, strict=True)
    x = torch.randn(8, 96, device=distributed_setup.device)
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize("recompute", [False, True])
def test_split_prefetch_preserves_training(distributed_setup, recompute):
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    torch.manual_seed(123)
    model = _Block().to(device)
    reference = copy.deepcopy(model)
    _shard(model, mesh, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fully_shard_optimizer(optimizer)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    prefetched = []
    model.residual.writer.register_forward_pre_hook(
        lambda module, args: prefetched.append(module._unshard_event is not None), prepend=True
    )
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(900 + step)
        x = torch.randn(8, 96, device=device, requires_grad=True)
        expected_x = x.detach().clone().requires_grad_()
        output = checkpoint(model, x, use_reentrant=False) if recompute else model(x)
        expected = reference(expected_x)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        assert prefetched[-1]
        assert all(
            group._unsharded_model_weight.local_buffer.untyped_storage().nbytes() == 0
            for group in model.branch.parameter_groups
        )
        output.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=0, atol=0)
        gradients = _full_values(model, "main_grad")
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(gradients[name], parameter.grad, rtol=0, atol=0)
        assert all(
            module._unshard_event is None
            for module in model.modules()
            if isinstance(module, FsdpModule)
        )
        optimizer.step()
        reference_optimizer.step()
        weights = _full_values(model, "main_weight")
        for name, parameter in reference.named_parameters():
            torch.testing.assert_close(weights[name], parameter, rtol=0, atol=0)


def _cpu(value):
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return value


def _assert_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def test_split_distributed_checkpoint_resume(distributed_setup, tmp_path):
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))

    def build():
        model = _shard(_Block().to(device), mesh, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, foreach=False)
        fully_shard_optimizer(optimizer)
        return model, optimizer

    def step(model, optimizer, seed):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(seed + distributed_setup.rank)
        x = torch.randn(8, 96, device=device, requires_grad=True)
        output = model(x)
        output.square().mean().backward()
        optimizer.step()
        return _cpu((output, x.grad))

    def capture(model, optimizer):
        return _cpu((model.state_dict(), get_optimizer_state_dict(model, optimizer)))

    model, optimizer = build()
    for seed in (123, 456, 789):
        step(model, optimizer, seed)
    paths = [str(tmp_path / "checkpoint") if distributed_setup.rank == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    expected = capture(model, optimizer)
    save_checkpoint(model, optimizer, paths[0])
    expected_output = step(model, optimizer, 1000)
    expected_after = capture(model, optimizer)
    restored, restored_optimizer = build()
    load_checkpoint(restored, restored_optimizer, paths[0])
    _assert_equal(capture(restored, restored_optimizer), expected)
    _assert_equal(step(restored, restored_optimizer, 1000), expected_output)
    _assert_equal(capture(restored, restored_optimizer), expected_after)
