# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP annotations."""

import re
from typing import Literal, NamedTuple

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Placements,
    fully_shard,
    fully_shard_context,
)

_NVTX_LABEL_PATTERN = re.compile(r"MFSDP (.+) (forward|backward|unshard|reshard|gradient_reduce)")


class NvtxEvent(NamedTuple):
    kind: Literal["push", "pop"]
    name: str
    operation: str


class NestedLinearModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.bias
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class FrozenFirstLayerModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.ones(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layers[0](x))
        return torch.relu(self.layers[1](x + self.bias))


class TiedLM(nn.Module):
    """Tiny language model with shared input and output embedding weights."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4, dtype=torch.bfloat16)
        self.lm_head = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
        self.lm_head.weight = self.embed_tokens.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(token_ids)).float().sum()


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)])


def _setup_nvtx_recording(monkeypatch: pytest.MonkeyPatch, events: list[NvtxEvent]) -> None:
    label_stack: list[tuple[str, str]] = []

    def parse_nvtx_label(label: str) -> tuple[str, str]:
        match = _NVTX_LABEL_PATTERN.fullmatch(label)
        assert match is not None
        return match.groups()

    def record_push(label: str) -> None:
        name, operation = parse_nvtx_label(label)
        label_stack.append((name, operation))
        events.append(NvtxEvent("push", name, operation))

    def record_pop() -> None:
        name, operation = label_stack.pop()
        events.append(NvtxEvent("pop", name, operation))

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", record_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", record_pop)


def _lifecycle_events(events: list[NvtxEvent]) -> list[NvtxEvent]:
    """Return the forward/backward ranges asserted by the lifecycle tests."""
    return [event for event in events if event.operation in ("forward", "backward")]


def test_fsdp_training_hooks_emit_operation_nvtx_ranges(distributed_setup, monkeypatch):
    """A training step should annotate each FSDP lifecycle operation."""
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = nn.Linear(4, 4, bias=False).to(distributed_setup.device)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert events == [
        NvtxEvent("push", "<root>", "forward"),
        NvtxEvent("push", "<root>", "unshard"),
        NvtxEvent("pop", "<root>", "unshard"),
        NvtxEvent("push", "<root>", "reshard"),
        NvtxEvent("pop", "<root>", "reshard"),
        NvtxEvent("pop", "<root>", "forward"),
        NvtxEvent("push", "<root>", "backward"),
        NvtxEvent("push", "<root>", "unshard"),
        NvtxEvent("pop", "<root>", "unshard"),
        NvtxEvent("push", "<root>", "reshard"),
        NvtxEvent("pop", "<root>", "reshard"),
        NvtxEvent("push", "<root>", "gradient_reduce"),
        NvtxEvent("pop", "<root>", "gradient_reduce"),
        NvtxEvent("pop", "<root>", "backward"),
    ]


def test_fsdp_sibling_roots_emit_root_nvtx_ranges_after_training_step(
    distributed_setup, monkeypatch
):
    """Independent FSDP roots should each emit root-labeled NVTX ranges."""
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = NestedLinearModel(dim=4).to(distributed_setup.device)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert _lifecycle_events(events) == [
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
    ]


def test_fsdp_training_hooks_emit_stacked_nvtx_ranges(distributed_setup, monkeypatch):
    """Nested training hooks should emit concise NVTX ranges."""
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = NestedLinearModel(dim=4).to(distributed_setup.device)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert _lifecycle_events(events) == [
        ("push", "<root>", "forward"),
        ("push", "layers.0", "forward"),
        ("pop", "layers.0", "forward"),
        ("push", "layers.1", "forward"),
        ("pop", "layers.1", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("push", "layers.1", "backward"),
        ("pop", "layers.1", "backward"),
        ("push", "layers.0", "backward"),
        ("pop", "layers.0", "backward"),
        ("pop", "<root>", "backward"),
    ]


def test_fsdp_frozen_parameters_emit_balanced_backward_nvtx_range(distributed_setup, monkeypatch):
    """Frozen FSDP units should still balance backward NVTX ranges."""
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = nn.Linear(4, 4, bias=False).to(distributed_setup.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.ones(2, 4, device=distributed_setup.device, requires_grad=True)
    model(x).sum().backward()

    assert _lifecycle_events(events) == [
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
    ]


def test_fsdp_frozen_child_without_grad_inputs_skips_backward_nvtx_range(
    distributed_setup, monkeypatch
):
    """Frozen FSDP children outside the backward graph should not emit backward ranges."""
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = FrozenFirstLayerModel(dim=4).to(distributed_setup.device)
    for parameter in model.layers[0].parameters():
        parameter.requires_grad_(False)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
        fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert _lifecycle_events(events) == [
        ("push", "<root>", "forward"),
        ("push", "layers.0", "forward"),
        ("pop", "layers.0", "forward"),
        ("push", "layers.1", "forward"),
        ("pop", "layers.1", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("push", "layers.1", "backward"),
        ("pop", "layers.1", "backward"),
        ("pop", "<root>", "backward"),
    ]


def test_tied_child_parameters_complete_backward_once_per_cycle(distributed_setup, monkeypatch):
    """Tied parameters should complete balanced backward ranges across training cycles."""
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = TiedLM()
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    with fully_shard_context(device=distributed_setup.device):
        fully_shard(model, mesh=mesh, placements=_flat_placements())

    token_ids = torch.arange(8, device=distributed_setup.device).reshape(2, 4)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        model(token_ids).backward()

    assert _lifecycle_events(events) == [
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
    ]
