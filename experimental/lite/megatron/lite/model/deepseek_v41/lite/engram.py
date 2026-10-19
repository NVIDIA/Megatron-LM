# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1-only hash/gate; table sharding and collectives use native Core."""
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import async_gather_from_sequence_parallel_region


def hash_multipliers(layer_ids, max_ngram_size, vocab_size):
    if vocab_size < 1 or max_ngram_size < 2:
        raise ValueError("Require nonempty compressed vocabulary and ngram order >= 2")
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)

    def multiplier(layer):
        rng = np.random.default_rng(10007 * layer)
        values = rng.integers(0, bound, size=max_ngram_size, dtype=np.int64)
        return torch.from_numpy(values * 2 + 1)

    return torch.stack([multiplier(layer) for layer in layer_ids])


def prime_buckets(layer_ids, max_ngram_size, heads, vocab_size):
    from sympy import nextprime

    current = vocab_size - 1

    def bucket():
        nonlocal current
        current = int(nextprime(current))
        return current

    return torch.tensor(
        [[[bucket() for _ in range(heads)] for _ in range(max_ngram_size - 1)] for _ in layer_ids],
        dtype=torch.int64,
    )


class NgramHash(nn.Module):
    """Hash a full local batch; packed boundaries restart every lookback.

    No process-global cache: each call is independent of previous microbatches
    and safe to recompute. Distributed history transport belongs to the caller.
    """

    def __init__(self, token_map, pad_id, multipliers, primes):
        super().__init__()
        mapping = torch.as_tensor(token_map, dtype=torch.int64)
        if multipliers.ndim != 2 or primes.ndim != 3:
            raise ValueError(
                "Expected multipliers [layers,order] and primes [layers,order-1,heads]"
            )
        if primes.shape[:2] != (multipliers.shape[0], multipliers.shape[1] - 1):
            raise ValueError("Hash layout and multiplier shape mismatch")
        self.pad_id = int(mapping[pad_id])
        self.register_buffer("token_map", mapping, persistent=False)
        self.register_buffer("multipliers", multipliers.to(torch.int64), persistent=False)
        self.register_buffer("primes", primes.to(torch.int64), persistent=False)
        flat = primes.flatten(1)
        self.register_buffer("offsets", flat.cumsum(-1) - flat, persistent=False)

    def forward(self, input_ids, token_mask=None, *, cu_seqlens=None):
        if input_ids.ndim != 2:
            raise ValueError("Expected input IDs [B,S]")
        b, length = input_ids.shape
        if token_mask is not None and (
            token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool
        ):
            raise ValueError("Expected boolean token mask [B,S]")
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = compressed.masked_fill(~token_mask, -1)
        positions = torch.arange(length, device=input_ids.device).expand(b, -1)
        starts = torch.zeros_like(positions)
        if cu_seqlens is not None:
            if b != 1:
                raise ValueError("THD packing requires B=1")
            bounds = cu_seqlens.tolist()
            if bounds[0] != 0 or bounds[-1] != length:
                raise ValueError("cu_seqlens must cover the packed sequence")
            for begin, end in zip(bounds[:-1], bounds[1:]):
                starts[:, begin:end] = begin
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        history = []
        for shift in range(self.multipliers.shape[1]):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions - shift < starts) | (source == -1)
            history.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(history, -1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.multipliers.shape[1]):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, -1) + self.offsets


class Engram(nn.Module):
    def __init__(self, hidden_size, copies, embedding, projection, *, eps=1e-6):
        super().__init__()
        self.dim = hidden_size
        self.copies = copies
        self.eps = eps
        self.embed = embedding
        self.wkv = projection
        self.q_weight = nn.Parameter(torch.ones(copies, hidden_size))
        self.k_weight = nn.Parameter(torch.ones(copies, hidden_size))

    def forward(self, hidden, hash_ids, token_mask=None):
        if hidden.ndim != 4 or hidden.shape[-2:] != (self.copies, self.dim):
            raise ValueError("Expected residual stream [B,S,HC,D]")
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.copies * self.dim, self.dim], -1)
        key = key.float().unflatten(-1, (self.copies, self.dim))
        h = hidden.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(
            key.square().mean(-1) + self.eps
        )
        dot = (
            (h * key * self.q_weight.float() * self.k_weight.float()).sum(-1)
            * rstd
            * self.dim**-0.5
        )
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden.dtype)


class ShardedFrozenTable(nn.Module):
    """Core row-sharded embedding with a persistent BF16 buffer, not a parameter.

    The Core embedding group can span data/context ranks with different query
    IDs. Native mappings first give every participant identical padded IDs;
    Core's embedding then performs its ordinary row-shard reduction. No custom
    collective or all-to-all implementation is introduced.
    """

    def __init__(self, rows, width, config, group, *, frozen=True):
        super().__init__()
        self.group, self.rows = group, rows
        padded_rows = ((rows + group.size() - 1) // group.size()) * group.size()
        cfg = copy.copy(config)
        cfg.use_cpu_initialization = False
        self.embedding = VocabParallelEmbedding(
            padded_rows, width, config=cfg, init_method=cfg.init_method, tp_group=group
        )
        shard_rows = padded_rows // group.size()
        self.embedding.lookup = SimpleNamespace(
            group=group, boundaries=tuple(i * shard_rows for i in range(group.size() + 1))
        )
        if frozen:
            value = self.embedding.weight.detach()
            self.embedding._parameters.pop("weight")
            self.embedding.register_buffer("weight", value, persistent=True)

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, ids):
        rows = ids.shape[0]
        if self.group.size() == 1:
            return self.embedding(ids)
        sizes = async_gather_from_sequence_parallel_region(
            ids.new_tensor([rows]), group=self.group
        ).wait()
        maximum = int(sizes.max())
        padded = F.pad(ids, (0, 0, 0, maximum - rows))
        all_ids = async_gather_from_sequence_parallel_region(padded, group=self.group).wait()
        value = self.embedding(all_ids)
        return value[self.group.rank() * maximum : self.group.rank() * maximum + rows]


class DS41Engram(Engram):
    def __init__(self, release, config, layer, table_group, *, frozen=True):
        position = release.engram_layer_ids.index(layer)
        rows = release.engram_num_embeddings[position]
        table = ShardedFrozenTable(
            rows, release.engram_head_dim, config, table_group, frozen=frozen
        )
        width = (
            (release.engram_max_ngram_size - 1) * release.engram_n_heads * release.engram_head_dim
        )
        projection = nn.Linear(
            width,
            (release.hc_mult + 1) * release.hidden_size,
            bias=False,
            dtype=config.params_dtype,
        )
        super().__init__(
            release.hidden_size, release.hc_mult, table, projection, eps=release.rms_norm_eps
        )
        self.layer_hash_index = position
        self.global_rows = rows
