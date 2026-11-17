# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Exercise the conversion checkpoint envelope using real distributed checkpoint I/O."""

import pytest
import torch

from megatron.bridge.models.conversion.checkpoint import (
    load_megatron_model,
    load_model_config,
    resolve_checkpoint_path,
    save_megatron_model,
)
from megatron.bridge.models.hybrid.model_config import BridgeHybridModelConfig
from megatron.bridge.models.transformer_config import TransformerConfig
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class SmallModel(torch.nn.Module):
    """Keep checkpoint testing independent of the model family's expensive kernels."""

    def __init__(self, config):
        super().__init__()
        self.model_config = config
        self.weight = torch.nn.Parameter(torch.zeros(4, 4, device="cuda"))

    def sharded_state_dict(self, metadata=None):
        return {
            "weight": ShardedTensor.from_rank_offsets(
                "weight", self.weight, replica_id=(0, 0, torch.distributed.get_rank())
            )
        }


class SmallBuilder:
    def get_model(self, config, *, load_weights, wrap_with_ddp):
        assert load_weights is False
        assert wrap_with_ddp is False
        return [SmallModel(config)]


def test_native_weights_and_model_config_round_trip_without_training_args(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = BridgeHybridModelConfig(
            transformer=TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=2),
            vocab_size=128,
            seq_length=64,
        )
        model = SmallModel(config)
        with torch.no_grad():
            model.weight.copy_(torch.arange(16, device="cuda").reshape(4, 4))

        with TempNamedDir(tmp_path_dist_ckpt / "bridge-roundtrip") as directory:
            save_megatron_model([model], directory)
            checkpoint = resolve_checkpoint_path(directory)
            restored_config, _ = load_model_config(checkpoint)
            assert type(restored_config) is type(config)
            assert restored_config.vocab_size == 128
            assert restored_config.seq_length == 64
            [restored] = load_megatron_model(SmallBuilder(), directory)
            assert torch.equal(restored.weight, model.weight)

            # Publishing into an existing iteration must not replace a prior export.
            with pytest.raises(RuntimeError, match="FileExistsError"):
                save_megatron_model([model], directory)
    finally:
        Utils.destroy_model_parallel()
