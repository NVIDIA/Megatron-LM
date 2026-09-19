# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4.1 model using HybridModel's embedding, head and checkpoint interface."""

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_csa2_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel


class DeepSeekV41Model(HybridModel):
    """V4.1 text backbone on the HybridModel distributed interface."""

    def __init__(self, config, vocab_size, max_sequence_length, *, pg_collection, **kwargs) -> None:
        if any(
            (
                getattr(config, name, None) is not None
                for name in ("engram_config", "vision_config", "dspark_config")
            )
        ):
            raise NotImplementedError(
                "This composition does not include the requested conditional modules"
            )
        if kwargs.get("share_embeddings_and_output_weights", False):
            raise ValueError("The released V4.1 architecture uses untied embedding/output weights")
        super().__init__(
            config=config,
            hybrid_stack_spec=hybrid_csa2_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern=config.hybrid_pattern,
            position_embedding_type="none",
            pg_collection=pg_collection,
            **kwargs,
        )

    def forward(self, input_ids, position_ids, attention_mask=None, **kwargs):
        """Run the backbone with an optional ordinary causal mask."""
        return super().forward(input_ids, position_ids, attention_mask, **kwargs)
