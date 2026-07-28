# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph tests for Megatron-FSDP."""

import logging

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

logger = logging.getLogger(__name__)


class NestedModel(nn.Module):
    """Model with a root FSDP unit and multiple child FSDP units."""

    def __init__(self, dim: int, num_children: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(num_children)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run through every child layer with a root-owned bias."""
        x = x + self.bias
        for layer in self.layers:
            x = layer(x)
        return x


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_captures_full_iteration(distributed_setup):
    """A full training iteration should be CUDA-graphable."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    dim = 8
    model = NestedModel(dim=dim, num_children=2).to(device)

    static_input = torch.eye(dim, device=device)
    static_target = torch.zeros_like(static_input)

    placements = _flat_placements()
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements)
    fully_shard(model, mesh=mesh, placements=placements)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        output = model(static_input)
        loss = torch.nn.functional.mse_loss(output, static_target)
        loss.backward()
        optimizer.step()
        return loss.detach()

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    # Warmup
    with torch.cuda.stream(capture_stream):
        # See: https://docs.nvidia.com/dl-cuda-graph/troubleshooting/memory-issues.html#gradient-accumulator-cross-stream-memory-growth
        # Warm up on the same stream used for capture so autograd's accumulation
        # path does not create cross-stream gradient-memory growth.
        # The first warmup installs the reusable sharded gradient views; subsequent
        # iterations zero them in place for CUDA graph replay.
        for _ in range(3):
            train_iteration()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        static_loss = train_iteration()
    torch.cuda.current_stream().wait_stream(capture_stream)

    # Replay
    losses = []
    for _ in range(5):
        graph.replay()
        # Each replay rewrites static_loss's fixed graph output storage; clone
        # keeps a per-replay GPU snapshot without the CPU sync from .item().
        losses.append(static_loss.clone())
    loss_values = torch.stack(losses).tolist()

    logger.info("CUDA graph replay losses: %s", loss_values)
    assert loss_values[-1] < loss_values[0], (
        "CUDA graph replay did not reduce the fixed-input loss: "
        f"first={loss_values[0]:.6f}, "
        f"last={loss_values[-1]:.6f}, trace={loss_values}"
    )
