# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Engram-augmented GPT Model.

Extends GPTModel to support DeepSeek's Engram n-gram hash embedding mechanism.
Before each forward pass, the model pre-computes n-gram hash embeddings from
input_ids and distributes them to the relevant EngramTransformerLayers.
"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from torch import Tensor

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.core.models.engram.engram_module import EngramConfig, NgramHashMapping

logger = logging.getLogger(__name__)


class EngramGPTModel(GPTModel):
    """GPT model augmented with Engram n-gram hash embeddings.

    This model extends GPTModel by:
      1. Maintaining a shared NgramHashMapping for deterministic n-gram hashing.
      2. Before each forward pass, pre-computing hash embeddings from input_ids
         and distributing them to EngramTransformerLayers in the decoder.

    Args:
        engram_config: Configuration for the Engram module.
        All other args are forwarded to GPTModel.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        engram_config: EngramConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

        self.engram_config = engram_config

        self.ngram_hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_config.engram_vocab_size,
            max_ngram_size=engram_config.max_ngram_size,
            n_embed_per_ngram=engram_config.n_embed_per_ngram,
            n_head_per_ngram=engram_config.n_head_per_ngram,
            layer_ids=engram_config.engram_layer_ids,
            tokenizer_name_or_path=engram_config.tokenizer_name_or_path,
            pad_id=engram_config.pad_id,
            seed=engram_config.seed,
        )

    def _precompute_engram_hashes(self, input_ids: Tensor) -> None:
        """Pre-compute n-gram hash embeddings and distribute to engram layers.

        Args:
            input_ids: [B, S] token IDs tensor.
        """
        if input_ids is None:
            return

        # input_ids is [B, S] — compute hashes for all engram layers at once
        hash_ids_all_layers = self.ngram_hash_mapping.hash(input_ids)

        device = next(self.decoder.parameters()).device

        for layer in self.decoder.layers:
            if hasattr(layer, 'engram') and layer.engram is not None:
                layer_id = layer.engram.layer_id
                if layer_id in hash_ids_all_layers:
                    layer.engram.precompute_embeddings(
                        hash_ids_all_layers[layer_id], device
                    )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params=None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with Engram pre-computation.

        Pre-computes n-gram hash embeddings from input_ids before running the
        standard GPT forward pass. The embeddings are distributed to each
        EngramTransformerLayer and consumed during their forward calls.
        """
        if self.pre_process:
            self._precompute_engram_hashes(input_ids)

        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
        )
