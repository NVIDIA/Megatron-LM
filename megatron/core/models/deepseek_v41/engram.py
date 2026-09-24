# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Trainable conditional memory, sharded over the explicit expert parallel group."""

from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch import nn

from megatron.core.models.deepseek_v41.engram_hash import EngramLayout, compute_hash_multipliers
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.transformer.module import MegatronModule


class EngramHasher(nn.Module):
    """Stateless full-sequence hashing, resetting history at images and sequence boundaries."""

    def __init__(self, options, token_map: torch.Tensor) -> None:
        super().__init__()
        args = SimpleNamespace(**{"engram_" + k: v for k, v in asdict(options).items()})
        self.layout = EngramLayout.from_args(args)
        vocab = options.compressed_vocab_size
        if token_map.ndim != 1 or token_map.min() < 0 or token_map.max() + 1 != vocab:
            raise ValueError("Engram token map does not match the compressed vocabulary")
        self.pad_id = int(token_map[options.pad_token_id])
        self.register_buffer("token_map", token_map.long())
        self.register_buffer("primes", torch.tensor(self.layout.primes, dtype=torch.long))
        flat = self.primes.flatten(1)
        offsets = torch.cat((torch.zeros_like(flat[:, :1]), flat.cumsum(-1)[:, :-1]), -1)
        if tuple(flat.sum(-1).tolist()) != self.layout.num_embeddings:
            raise ValueError("Engram table row counts must equal their prime bucket layout")
        self.register_buffer("offsets", offsets)
        self.register_buffer(
            "multipliers",
            compute_hash_multipliers(self.layout.layer_ids, self.layout.max_ngram_size, vocab),
        )

    def forward(self, token_ids, token_mask=None, sequence_ids=None):
        """Return [batch, sequence, layer, hash-head] addresses, with causal padding."""
        compressed = self.token_map[token_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, -1)
        batch, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device).expand(batch, -1)
        blocked = torch.zeros_like(token_ids, dtype=torch.bool)
        tokens = []
        for shift in range(self.layout.max_ngram_size):
            source_pos = (positions - shift).clamp_min(0)
            source = compressed.gather(1, source_pos)
            blocked = blocked | (positions < shift) | (source == -1)
            if sequence_ids is not None:
                blocked = blocked | (sequence_ids.gather(1, source_pos) != sequence_ids)
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, -1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, -1) + self.offsets


class EngramEmbedding(EPShardedMultiTableEmbedding):
    """V4.1's fused table, using variable-split EP lookups and exact logical row sharding."""

    def __init__(self, config, rows, width, pg_collection) -> None:
        super().__init__(
            config,
            (rows,),
            width,
            config.init_method,
            ep_group=pg_collection.ep,
            tp_group=pg_collection.tp,
            expt_dp_group=pg_collection.expt_dp,
        )
        self.tp_group = pg_collection.tp

    def forward(self, ids):
        """Treat all V4.1 n-gram/head offsets as addresses in one fused table."""
        return super().forward(ids.unsqueeze(-1)).squeeze(-2)


class Engram(MegatronModule):
    """Context-gated n-gram memory added to all mHC streams before attention."""

    def __init__(self, config, layout, layer_idx, pg_collection) -> None:
        super().__init__(config)
        self.tp_group = pg_collection.tp
        self.hash_index = layout.layer_ids.index(layer_idx)
        self.n = config.mhc_num_residual_streams
        self.hidden_size = config.hidden_size
        self._cudnn_gate = None
        if config.engram_gate_backend == "cudnn":
            from megatron.core.fusions.cudnn_engram import CudnnEngramGate

            self._cudnn_gate = CudnnEngramGate(config.layernorm_epsilon)
        self.embed = EngramEmbedding(
            config, layout.num_embeddings[self.hash_index], layout.head_dim, pg_collection
        )
        width = (layout.max_ngram_size - 1) * layout.n_heads * layout.head_dim
        self.wkv = nn.Linear(
            width, (self.n + 1) * config.hidden_size, bias=False, dtype=config.params_dtype
        )
        self.q_weight = nn.Parameter(
            torch.ones(self.n, config.hidden_size, dtype=config.params_dtype)
        )
        self.k_weight = nn.Parameter(torch.ones_like(self.q_weight))
        if config.perform_initialization:
            config.init_method(self.wkv.weight)

    def forward(self, hidden_states, hash_ids, token_mask=None):
        """Preserve FP32 gating arithmetic and the reference signed-square-root gate."""
        addresses = hash_ids[:, :, self.hash_index]
        kv = self.wkv(self.embed(addresses).flatten(-2)).transpose(0, 1)
        if self._cudnn_gate is not None:
            return self._cudnn_gate(hidden_states, kv, self.q_weight, self.k_weight, token_mask)
        key, value = kv.split([self.n * self.hidden_size, self.hidden_size], dim=-1)
        key = key.float().unflatten(-1, (self.n, self.hidden_size))
        h = hidden_states.float().unflatten(-1, (self.n, self.hidden_size))
        eps = self.config.layernorm_epsilon
        rstd = (h.square().mean(-1) + eps).rsqrt() * (key.square().mean(-1) + eps).rsqrt()
        dot = (h * key * self.q_weight.float() * self.k_weight.float()).sum(-1)
        dot = dot * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.transpose(0, 1).unsqueeze(-1), 0)
        return (
            (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2))
            .flatten(-2)
            .to(hidden_states.dtype)
        )
