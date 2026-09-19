# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""DeepSeek-V4.1 model using HybridModel's embedding, head and checkpoint interface."""

import torch

from megatron.core.models.deepseek_v41.engram import Engram, EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import build_compressed_token_map
from megatron.core.models.deepseek_v41.stack import deepseek_v41_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel


class DeepSeekV41Model(HybridModel):
    """V4.1 text backbone with trainable conditional n-gram memory."""

    def __init__(
        self,
        config,
        vocab_size,
        max_sequence_length,
        *,
        pg_collection,
        token_map=None,
        tokenizer=None,
        **kwargs
    ) -> None:
        if any(
            (getattr(config, name, None) is not None for name in ("vision_config", "dspark_config"))
        ):
            raise NotImplementedError(
                "This composition does not include the requested conditional modules"
            )
        if kwargs.get("share_embeddings_and_output_weights", False):
            raise ValueError("The released V4.1 architecture uses untied embedding/output weights")
        hasher = None
        if getattr(config, "engram_config", None):
            if token_map is None:
                if tokenizer is None:
                    raise ValueError(
                        "Engram requires the released tokenizer or an explicit compressed token map"
                    )
                token_map, _ = build_compressed_token_map(tokenizer)
                token_map = torch.tensor(token_map, dtype=torch.long)
            hasher = EngramHasher(config.engram_config, token_map)
            ids = hasher.layout.layer_ids
            if len(set(ids)) != len(ids) or any(
                (i < 0 or i >= len(config.csa_compress_ratios) for i in ids)
            ):
                raise ValueError("Engram layers must be distinct zero-based backbone blocks")
        super().__init__(
            config=config,
            hybrid_stack_spec=deepseek_v41_stack_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            hybrid_layer_pattern="VE" * len(config.csa_compress_ratios),
            position_embedding_type="none",
            pg_collection=pg_collection,
            **kwargs
        )
        self.engram_hash = hasher
        if hasher is not None:
            for i in self.engram_hash.layout.layer_ids:
                self.decoder.layers[i].engram = Engram(
                    config, self.engram_hash.layout, i, pg_collection
                )

    def forward_features(self, input_ids, position_ids, *, attention_mask=None, padding_mask=None):
        """Embed tokens and apply n-gram memory before the selected attention branches."""
        hidden = self.embedding(input_ids, position_ids)
        hashes = self.engram_hash(input_ids) if self.engram_hash is not None else None
        return self.decoder(hidden, attention_mask, padding_mask=padding_mask, engram_hashes=hashes)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        *,
        labels=None,
        padding_mask=None,
        packed_seq_params=None,
        **kwargs
    ):
        """Return per-token losses or batch-major logits for unpacked training sequences."""
        if packed_seq_params is not None or kwargs:
            raise NotImplementedError("V4.1 currently supports unpacked training forwards")
        hidden = self.forward_features(
            input_ids, position_ids, attention_mask=attention_mask, padding_mask=padding_mask
        )
        logits, _ = self.output_layer(hidden)
        return (
            self.compute_language_model_loss(labels, logits)
            if labels is not None
            else logits.transpose(0, 1).contiguous()
        )
