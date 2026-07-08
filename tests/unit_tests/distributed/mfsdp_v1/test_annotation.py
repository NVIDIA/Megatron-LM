# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for experimental Megatron-FSDP annotations."""

import re
from typing import Literal, NamedTuple

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

_NVTX_LABEL_PATTERN = re.compile(r"MFSDP (.+) (forward|backward)")


class NvtxEvent(NamedTuple):
    kind: Literal["push", "pop"]
    name: str
    phase: str


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


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def _setup_nvtx_recording(monkeypatch: pytest.MonkeyPatch, events: list[NvtxEvent]) -> None:
    label_stack: list[tuple[str, str]] = []

    def parse_nvtx_label(label: str) -> tuple[str, str]:
        match = _NVTX_LABEL_PATTERN.fullmatch(label)
        assert match is not None
        return match.groups()

    def record_push(label: str) -> None:
        name, phase = parse_nvtx_label(label)
        label_stack.append((name, phase))
        events.append(NvtxEvent("push", name, phase))

    def record_pop() -> None:
        name, phase = label_stack.pop()
        events.append(NvtxEvent("pop", name, phase))

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", record_push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", record_pop)


def _get_distributed_setup(request: pytest.FixtureRequest):
    try:
        return request.getfixturevalue("distributed_setup")
    except pytest.FixtureLookupError:
        pytest.skip("distributed_setup fixture is only available in the Megatron-FSDP test bucket")


def test_fsdp_sibling_roots_emit_root_nvtx_ranges_after_training_step(request, monkeypatch):
    """Independent FSDP roots should each emit root-labeled NVTX ranges."""
    distributed_setup = _get_distributed_setup(request)
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = NestedLinearModel(dim=4).to(distributed_setup.device)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
    fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert [(event.kind, event.name, event.phase) for event in events] == [
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
    ]


def test_fsdp_training_hooks_emit_stacked_nvtx_ranges(request, monkeypatch):
    """Nested training hooks should emit concise NVTX ranges."""
    distributed_setup = _get_distributed_setup(request)
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = NestedLinearModel(dim=4).to(distributed_setup.device)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
    fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert [(event.kind, event.name, event.phase) for event in events] == [
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


def test_fsdp_frozen_parameters_emit_balanced_backward_nvtx_range(request, monkeypatch):
    """Frozen FSDP units should still balance backward NVTX ranges."""
    distributed_setup = _get_distributed_setup(request)
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = nn.Linear(4, 4, bias=False).to(distributed_setup.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    x = torch.ones(2, 4, device=distributed_setup.device, requires_grad=True)
    model(x).sum().backward()

    assert [(event.kind, event.name, event.phase) for event in events] == [
        ("push", "<root>", "forward"),
        ("pop", "<root>", "forward"),
        ("push", "<root>", "backward"),
        ("pop", "<root>", "backward"),
    ]


def test_fsdp_frozen_child_without_grad_inputs_skips_backward_nvtx_range(request, monkeypatch):
    """Frozen FSDP children outside the backward graph should not emit backward ranges."""
    distributed_setup = _get_distributed_setup(request)
    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model = FrozenFirstLayerModel(dim=4).to(distributed_setup.device)
    for parameter in model.layers[0].parameters():
        parameter.requires_grad_(False)
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    fully_shard(model.layers[0], mesh=mesh, placements=_flat_placements())
    fully_shard(model.layers[1], mesh=mesh, placements=_flat_placements())
    fully_shard(model, mesh=mesh, placements=_flat_placements())

    model(torch.ones(2, 4, device=distributed_setup.device)).sum().backward()

    assert [(event.kind, event.name, event.phase) for event in events] == [
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
