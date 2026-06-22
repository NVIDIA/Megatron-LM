# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from typing import Optional

import torch
from torch import Tensor

from megatron.core.models.gemma4.gemma4_block import Gemma4TransformerBlock
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.gemma4_mask import (
    build_causal_mask,
    build_sliding_window_causal_mask,
)
from megatron.core.transformer.gemma4_ple import Gemma4PLE
from megatron.core.transformer.gemma4_rope import Gemma4RotaryEmbedding
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module


class Gemma4Model(GPTModel):
    """Gemma 4 E4B model (HF ``Gemma4ForCausalLM``, modeling_gemma4.py:1646-1905).

    Subclasses :class:`GPTModel` and adds the model-level Gemma4 specifics, all kept
    out of the shared base (R4):

    * **sqrt(H) embedding scaling** applied inside ``forward`` (bf16-rounded scalar, so
      sqrt(2560) rounds in bf16 before the multiply, matching HF
      ``Gemma4TextScaledWordEmbedding``).
    * **Per-Layer Embeddings (PLE)** precomputed into ``per_layer_inputs`` [B,S,L,P] and
      fed one slice per decoder layer via :class:`Gemma4TransformerBlock`.
    * **final logit softcap** ``30 * tanh(logits / 30)`` applied directly to the logits
      so it reaches the loss (``_scale_logits`` no-ops unless ``use_mup``).

    Uses :class:`Gemma4TransformerBlock` as its decoder, ``position_embedding_type=
    'none'`` (the base RoPE is skipped; a :class:`Gemma4RotaryEmbedding` supplies the
    per-layer-type cos/sin), and tied input/output embeddings.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        **kwargs,
    ):
        kwargs.setdefault("position_embedding_type", "none")
        kwargs.setdefault("share_embeddings_and_output_weights", True)
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )

        # Replace the base decoder with the gemma4 block (owns KV bus + PLE threading).
        self.decoder = Gemma4TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

        self.rotary_emb = Gemma4RotaryEmbedding(
            sliding_head_dim=self.config.sliding_kv_channels,
            full_head_dim=self.config.full_kv_channels,
            sliding_base=self.config.sliding_rotary_base,
            full_base=self.config.full_rotary_base,
        )
        self.ple = Gemma4PLE(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            ple_dim=self.config.hidden_size_per_layer_input,
            vocab_size_per_layer_input=self.config.vocab_size_per_layer_input,
            eps=self.config.layernorm_epsilon,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Gemma4 forward. Returns logits [b, s, V] (labels=None) or loss [b, s]."""
        b, s = input_ids.shape
        device = input_ids.device

        if position_ids is None:
            position_ids = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)

        # Embedding [s, b, h] (MLM seq-first) with bf16-rounded sqrt(H) scaling.
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        if self.config.apply_embedding_scaling:
            embed_dtype = self.embedding.word_embeddings.weight.dtype
            scale = torch.tensor(self.config.hidden_size**0.5).to(embed_dtype)
            decoder_input = decoder_input * scale

        # PLE precompute. The foundation Gemma4PLE consumes inputs_embeds in [B,S,H];
        # decoder_input is [s,b,h] so transpose. Result [B,S,L,P] -> [s,b,L,P] for the
        # seq-first decoder.
        inputs_embeds = decoder_input.transpose(0, 1).contiguous()  # [b, s, h]
        per_layer_inputs = self.ple(input_ids, inputs_embeds)  # [b, s, L, P]
        per_layer_inputs = per_layer_inputs.transpose(0, 1).contiguous()  # [s, b, L, P]

        # Per-layer-type rope cos/sin and additive masks (finfo.min fill).
        dtype = decoder_input.dtype
        rotary_cos_sin_by_type = {
            "sliding": self.rotary_emb(decoder_input, position_ids, "sliding"),
            "full": self.rotary_emb(decoder_input, position_ids, "full"),
        }
        attention_mask_by_type = {
            "full": build_causal_mask(s, dtype, device),
            "sliding": build_sliding_window_causal_mask(
                s, self.config.sliding_window, dtype, device
            ),
        }

        hidden_states = self.decoder(
            decoder_input,
            per_layer_inputs=per_layer_inputs,
            rotary_cos_sin_by_type=rotary_cos_sin_by_type,
            attention_mask_by_type=attention_mask_by_type,
        )

        # LM head (tied) -> final logit softcap -> loss/return.
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = softcap * torch.tanh(logits / softcap)

        if labels is None:
            return logits.transpose(0, 1).contiguous()  # [s, b, V] -> [b, s, V]
        return self.compute_language_model_loss(labels, logits)
