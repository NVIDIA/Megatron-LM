# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph tests for Megatron-FSDP."""

import logging

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

logger = logging.getLogger(__name__)


def _flat_placements() -> Placements:
    return Placements(dp_axes=[0], parameter=[Flat()], gradient=[Flat()], optimizer=[Flat()])


def test_captures_full_iteration(distributed_setup):
    """A full training iteration should be CUDA-graphable."""
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if world_size < 2:
        pytest.skip("This test requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    model = nn.Linear(4, 2, bias=False).to(device)

    fully_shard(model, mesh=mesh, placements=_flat_placements())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)

    static_input = torch.eye(4, device=device)
    static_target = torch.tensor(
        [[1.0, -0.5], [-0.25, 0.75], [0.5, 0.25], [-0.75, -1.0]], device=device
    )

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        output = model(static_input)
        loss = torch.nn.functional.mse_loss(output, static_target)
        loss.backward()
        optimizer.step()
        return loss.detach()

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    # Warm up before capture. torch.cuda.graph() uses an internal side stream
    # when `stream` is omitted, so `stream=` is only needed when callers must
    # control the capture stream, such as when reusing an explicit stream with
    # a shared graph memory pool across captures.
    with torch.cuda.stream(warmup_stream):
        # The first warmup installs the reusable sharded gradient views; subsequent
        # iterations zero them in place for CUDA graph replay.
        for _ in range(3):
            train_iteration()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_loss = train_iteration()

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
