# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the experimental Megatron-FSDP QwenImage example."""

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from examples.megatron_fsdp.train_qwen_image_experimental import (
    flat_dp_placements,
    fully_shard_qwen_image_transformer,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import fully_shard_optimizer


class TinyQwenImageTransformer(nn.Module):
    """Small QwenImage-shaped module for testing bottom-up FSDP application."""

    def __init__(self) -> None:
        super().__init__()
        self.img_in = nn.Linear(8, 8)
        self.transformer_blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(8, 8), nn.SiLU()) for _ in range(2)]
        )
        self.proj_out = nn.Linear(8, 8)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the tiny transformer."""
        hidden_states = self.img_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)
        return self.proj_out(hidden_states)


def test_qwen_image_helper_shards_blocks_bottom_up(distributed_setup):
    """The QwenImage root and every repeated transformer block should be FSDP units."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,), mesh_dim_names=("dp",))
    model = TinyQwenImageTransformer().to(device)
    fully_shard_qwen_image_transformer(model, mesh=mesh, placements=flat_dp_placements())

    assert hasattr(model, "parameter_groups")
    assert all(hasattr(block, "parameter_groups") for block in model.transformer_blocks)

    model_input = torch.randn(2, 8, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, foreach=False)
    fully_shard_optimizer(optimizer)
    first_output = model(model_input).detach()
    optimizer.zero_grad(set_to_none=True)
    loss = model(model_input).square().mean()
    loss.backward()
    optimizer.step()

    second_output = model(model_input).detach()
    assert not torch.equal(first_output, second_output)
