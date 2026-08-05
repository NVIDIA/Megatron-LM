# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CUDA graph tests for Megatron-FSDP."""

import logging

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    Flat,
    Placements,
    fully_shard,
)

logger = logging.getLogger(__name__)

# Sized so NCCL selects its symmetric-memory (ncclSymk*) kernels over ring for the
# sharded Linear collectives; see test_symmetric_memory.py for the rationale.
_HIDDEN = 1024


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


@pytest.mark.parametrize("use_symm_mem", [False, True], ids=["default", "symmetric_memory"])
def test_captures_full_iteration(distributed_setup, use_symm_mem):
    """A full training iteration should be CUDA-graphable, with and without symmetric memory.

    With ``use_symm_mem=True`` the FSDP collective path changes: unshard allocates the
    all-gather buffer from a symmetric-memory ``MemPool`` and calls
    ``symm_mem.rendezvous()``, and the backward reduce-scatter does the same for the
    partial-gradient buffer. Both allocation-from-pool and rendezvous are normally
    illegal inside CUDA-graph capture, so this confirms the warmup establishes reusable,
    already-rendezvoused buffers that capture and replay cleanly -- reproducing the
    non-symmetric-memory loss trajectory exactly.
    """
    world_size = distributed_setup.world_size
    device = distributed_setup.device
    if use_symm_mem and world_size < 2:
        pytest.skip("Symmetric memory requires at least 2 ranks.")

    mesh = init_device_mesh(device.type, (world_size,))
    torch.manual_seed(1234)
    dim = _HIDDEN
    # bf16 compute weights: FSDP's default mixed-precision policy keeps fp32 optimizer
    # master weights, so SGD updates accumulate in fp32 and the loss keeps decreasing
    # across replays instead of stalling within bf16 precision.
    model = NestedModel(dim=dim, num_children=2).to(device=device, dtype=torch.bfloat16)

    # A zero regression target drives ``mean(output**2)`` toward zero so the loss
    # decreases by a robust margin (unlike a random target, whose loss floors near its
    # own variance and barely moves within bf16 precision).
    static_input = torch.randn(256, dim, device=device, dtype=torch.bfloat16)
    static_target = torch.zeros(256, dim, device=device, dtype=torch.bfloat16)

    placements = _flat_placements()
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, placements=placements, use_symm_mem=use_symm_mem)
    fully_shard(model, mesh=mesh, placements=placements, use_symm_mem=use_symm_mem)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.25, foreach=False)

    def train_iteration() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=False)
        output = model(static_input)
        loss = torch.nn.functional.mse_loss(output, static_target)
        loss.backward()
        optimizer.step()
        return loss.detach()

    # Warm the data-parallel communicator before capture. This is a no-op for the
    # default path, but NCCL symmetric-memory window registration can fail when a
    # rendezvous is the first collective on a communicator, so establish it here.
    dist.barrier(group=mesh.get_group(0), device_ids=[device.index])

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())

    # See: https://docs.nvidia.com/dl-cuda-graph/troubleshooting/memory-issues.html#gradient-accumulator-cross-stream-memory-growth
    # Warm up on the same stream used for capture so autograd's accumulation path does
    # not create cross-stream gradient-memory growth. The first warmup installs the
    # reusable sharded gradient views (and, with symmetric memory, allocates and
    # rendezvouses the staging buffers) so the captured region reuses them for replay.
    with torch.cuda.stream(capture_stream):
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
