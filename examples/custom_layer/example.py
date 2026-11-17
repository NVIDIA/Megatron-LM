# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compose a built-in attention layer and a custom layer in a small HybridModel."""

import logging
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig

logger = logging.getLogger(__name__)


class GainLayer(MegatronModule):
    """Multiply each hidden channel by a learned scale."""

    def __init__(
        self,
        config: "GainConfig",
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
        self.tp_group = pg_collection.tp
        self.gain = torch.nn.Parameter(
            torch.full((config.hidden_size,), config.gain_init, dtype=config.params_dtype)
        )

    @classmethod
    def estimate_flops(
        cls,
        config: "GainConfig",
        *,
        total_real_tokens_in_batch: float,
        seqlen_squared_sum_in_batch: float,
    ) -> float:
        """Count one forward multiplication per token and hidden channel."""
        return total_real_tokens_in_batch * config.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        rotary_pos_emb: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Preserve the input shape; channel scaling needs no attention metadata."""
        return hidden_states * self.gain


@dataclass
class GainConfig(TransformerConfig):
    """Bind the custom layer directly and add its initialization setting."""

    layer_type: ClassVar[type[MegatronModule]] = GainLayer
    """Implementation constructed for each occurrence of this config in the model."""

    gain_init: float = 1.0
    """Initial value of each channel's scale."""


def build_model(pg_collection: ProcessGroupCollection) -> HybridModel:
    """Build two layers: Core attention followed by the custom channel scaling."""
    config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_backend=AttnBackend.unfused,
    )
    return HybridModel(
        config=config,
        vocab_size=64,
        max_sequence_length=8,
        hybrid_layer_config_list=[
            AttentionLayerConfig.from_config(config),
            GainConfig.from_config(config),
        ],
        position_embedding_type="rope",
        pg_collection=pg_collection,
    )


def main() -> None:
    """Run one single-GPU optimization step using synthetic next-token labels."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    try:
        parallel_state.initialize_model_parallel()
        # Materialize the global grid at bootstrap; pass it explicitly to the model.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234, pg_collection=pg_collection)

        model = build_model(pg_collection).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        custom_layer = model.decoder.layers[1]
        gain_before = custom_layer.gain.detach().clone()

        tokens = torch.arange(16, device="cuda", dtype=torch.long).reshape(2, 8)
        positions = torch.arange(8, device="cuda", dtype=torch.long).expand(2, -1)
        labels = (tokens + 1) % model.vocab_size
        attention_mask = torch.triu(torch.ones(1, 1, 8, 8, device="cuda", dtype=torch.bool), 1)

        loss = model(tokens, positions, attention_mask, labels=labels).float().mean()
        assert torch.isfinite(loss), "Loss must be finite"
        loss.backward()
        gain_grad = custom_layer.gain.grad
        assert gain_grad is not None and torch.isfinite(gain_grad).all()
        assert gain_grad.count_nonzero() > 0, "The custom layer must receive gradients"
        optimizer.step()
        assert torch.isfinite(custom_layer.gain).all()
        assert not torch.equal(gain_before, custom_layer.gain), "The custom layer must update"

        logger.info("loss=%.6f, custom_layer_grad_norm=%.6f", loss.item(), gain_grad.norm().item())
        logger.info("Success: HybridModel ran forward/backward and updated the custom layer.")
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
