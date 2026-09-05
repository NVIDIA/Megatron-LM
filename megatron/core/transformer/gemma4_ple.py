# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.transformer.gemma4_norm import Gemma4RMSNorm


class Gemma4PLE(torch.nn.Module):
    """Per-Layer Embeddings for Gemma 4 (modeling_gemma4.py:1615-1630, 1737-1814).

    Computes ``per_layer_inputs`` [B, S, L, P] fed one slice per decoder layer:

        E_tok  = embed_tokens_per_layer(ids) * sqrt(P)        # bf16-rounded scale
        E_proj = per_layer_projection_norm(
                     per_layer_model_projection(inputs_embeds) * (1 / sqrt(H)))
        per_layer_inputs = (E_proj + E_tok) * (1 / sqrt(2))

    ``inputs_embeds`` are the already-√H-scaled token embeddings. Embedding scales
    are cast to the embedding weight dtype before the multiply, matching HF's
    ``Gemma4TextScaledWordEmbedding`` bf16 rounding.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        ple_dim: int = 256,
        vocab_size_per_layer_input: int = 262144,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.ple_dim = ple_dim
        self.embed_scale = ple_dim**0.5
        self.projection_scale = hidden_size**-0.5
        self.input_scale = 2.0**-0.5

        self.embed_tokens_per_layer = torch.nn.Embedding(vocab_size_per_layer_input, num_layers * ple_dim)
        self.per_layer_model_projection = torch.nn.Linear(hidden_size, num_layers * ple_dim, bias=False)
        self.per_layer_projection_norm = Gemma4RMSNorm(ple_dim, eps=eps)

    def forward(self, input_ids: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Token-identity component (E_tok).
        scale = torch.tensor(self.embed_scale).to(self.embed_tokens_per_layer.weight.dtype)
        e_tok = self.embed_tokens_per_layer(input_ids) * scale
        e_tok = e_tok.reshape(*input_ids.shape, self.num_layers, self.ple_dim)

        # Context component (E_proj).
        e_proj = self.per_layer_model_projection(inputs_embeds) * self.projection_scale
        e_proj = e_proj.reshape(*inputs_embeds.shape[:-1], self.num_layers, self.ple_dim)
        e_proj = self.per_layer_projection_norm(e_proj)

        return (e_proj + e_tok) * self.input_scale
