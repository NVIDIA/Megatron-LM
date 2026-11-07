# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""An unregistered layer defined outside Core, shared by extension contract tests."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

from megatron.core import dist_checkpointing
from megatron.core.models.hybrid.hybrid_layer import (
    HybridLayer,
    HybridLayerBuildContext,
    HybridLayerConfig,
)
from megatron.core.process_groups_config import ProcessGroupCollection


@dataclass
class ExternalGainConfig(HybridLayerConfig):
    """A synthetic architecture with fields that Core does not know about."""

    gain_init: float = 1.25
    notes: list[str] = field(default_factory=lambda: ["external"])

    def build_layer(self, context: HybridLayerBuildContext) -> HybridLayer:
        return ExternalGainLayer(self, context)

    def estimate_flops(
        self, *, total_real_tokens_in_batch: float, seqlen_squared_sum_in_batch: float
    ) -> float:
        return total_real_tokens_in_batch * self.hidden_size


class ExternalGainLayer(HybridLayer):
    """Trainable channel scaling, without inheriting any built-in layer family."""

    def __init__(self, config: ExternalGainConfig, context: HybridLayerBuildContext) -> None:
        super().__init__(config, context)
        self.gain = torch.nn.Parameter(
            torch.full((config.hidden_size,), config.gain_init, dtype=config.params_dtype)
        )
        self.last_forward_inputs = None

    def forward(
        self,
        hidden_states,
        *,
        attention_mask=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
        padding_mask=None,
        sequence_len_offset=None,
        input_ids=None,
    ):
        self.last_forward_inputs = (
            attention_mask,
            rotary_pos_emb,
            packed_seq_params,
            padding_mask,
            sequence_len_offset,
            input_ids,
        )
        return hidden_states * self.gain


def run_gloo_checkpoint_roundtrip(checkpoint_dir: str) -> None:
    """Exercise real CPU checkpointing in a process isolated from pytest's NCCL groups."""
    torch.distributed.init_process_group(
        backend="gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
    )
    try:
        groups = ProcessGroupCollection()
        groups.tp = groups.cp = groups.dp_cp = torch.distributed.group.WORLD
        config = ExternalGainConfig(num_layers=1, hidden_size=8, num_attention_heads=2)
        context = HybridLayerBuildContext(groups, layer_number=1)
        original = config.build_layer(context)
        optimizer = torch.optim.SGD(original.parameters(), lr=0.1, momentum=0.9)
        inputs = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8) / 16

        def step(layer, current_optimizer):
            current_optimizer.zero_grad()
            loss = layer(inputs).square().mean()
            loss.backward()
            current_optimizer.step()
            return loss.detach()

        step(original, optimizer)
        Path(checkpoint_dir).mkdir()
        dist_checkpointing.save(
            {"model": original.sharded_state_dict(), "optimizer": optimizer.state_dict()},
            checkpoint_dir,
        )
        restored = ExternalGainConfig.from_config(config).build_layer(context)
        with torch.no_grad():
            restored.gain.fill_(-9)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
        checkpoint = dist_checkpointing.load(
            {"model": restored.sharded_state_dict()}, checkpoint_dir
        )
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.testing.assert_close(restored.gain, original.gain, rtol=0, atol=0)
        torch.testing.assert_close(
            step(restored, restored_optimizer), step(original, optimizer), rtol=0, atol=0
        )
        torch.testing.assert_close(restored.gain, original.gain, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()
