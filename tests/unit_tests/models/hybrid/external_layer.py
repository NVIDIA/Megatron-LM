# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""An unregistered layer defined outside Core, shared by extension contract tests."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch

from megatron.core import dist_checkpointing
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class ExternalGainLayer(MegatronModule):
    """Trainable channel scaling, without inheriting any built-in layer family."""

    def __init__(
        self,
        config: TransformerConfig,
        *,
        pg_collection: ProcessGroupCollection,
        layer_number: int,
        pp_layer_offset: int = 0,
        is_mtp_layer: bool = False,
        mtp_layer_number: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config)
        self.layer_number = layer_number
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.is_mtp_layer = is_mtp_layer
        self.gain = torch.nn.Parameter(
            torch.full((config.hidden_size,), config.gain_init, dtype=config.params_dtype)
        )
        self.last_forward_inputs = None

    @classmethod
    def estimate_flops(
        cls,
        config: TransformerConfig,
        *,
        total_real_tokens_in_batch: float,
        seqlen_squared_sum_in_batch: float,
    ) -> float:
        return total_real_tokens_in_batch * config.hidden_size

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


@dataclass
class ExternalGainConfig(TransformerConfig):
    """A synthetic architecture with fields that Core does not know about."""

    layer_type: ClassVar[type[MegatronModule]] = ExternalGainLayer
    gain_init: float = 1.25
    notes: list[str] = field(default_factory=lambda: ["external"])


def run_gloo_checkpoint_roundtrip(checkpoint_dir: str) -> None:
    """Exercise real CPU checkpointing in a process isolated from pytest's NCCL groups."""
    torch.distributed.init_process_group(
        backend="gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
    )
    try:
        groups = ProcessGroupCollection()
        groups.tp = groups.cp = groups.dp_cp = torch.distributed.group.WORLD
        config = ExternalGainConfig(num_layers=1, hidden_size=8, num_attention_heads=2)
        original = config.layer_type(config=config, pg_collection=groups, layer_number=1)
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
        metadata = {"dp_cp_group": groups.dp_cp}
        dist_checkpointing.save(
            {
                "model": original.sharded_state_dict(metadata=metadata),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_dir,
        )
        restored_config = ExternalGainConfig.from_config(config)
        restored = restored_config.layer_type(
            config=restored_config, pg_collection=groups, layer_number=1
        )
        with torch.no_grad():
            restored.gain.fill_(-9)
        restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1, momentum=0.9)
        checkpoint = dist_checkpointing.load(
            {"model": restored.sharded_state_dict(metadata=metadata)}, checkpoint_dir
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
