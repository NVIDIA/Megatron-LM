# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Engram: hashed n-gram memory added to the DeepSeek-V4.1 residual streams.

At a few layers the model looks up, for every position, the 2-gram ... ``max_ngram``-gram
ending there. Each n-gram is hashed into ``n_heads`` prime-sized bucket ranges of one large
table; the fetched rows are projected into one key per residual stream plus one shared
value, and the value is added to every stream through a gate that measures how well the
stream matches the key.

This module implements the *frozen* Engram of the first training phase: the table and the
projection participate in the forward pass only. Table rows are sharded over a process
group with an all-to-all lookup, because the ranks of a group hold different tokens.

Hashing must reproduce the released model bit for bit (compressed token map, prime
bucket layout, per-layer multipliers drawn from ``numpy.random.default_rng(10007 * layer)``),
so those definitions follow the official ``inference/engram.py`` exactly; the code is an
independent implementation.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

# Marker for a position that takes no part in any n-gram (image spans in the VL model).
DEAD_TOKEN = -1


# ---------------------------------------------------------------------------------------
# Bucket layout
# ---------------------------------------------------------------------------------------

_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin, exact for every ``n < 3.3e24`` (far beyond table sizes)."""
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def next_unused_prime(start: int, used: set) -> int:
    """Smallest prime strictly above ``start`` that is not in ``used``."""
    candidate = start + 1
    while candidate in used or not is_prime(candidate):
        candidate += 1
    return candidate


@dataclass(frozen=True)
class EngramTableLayout:
    """Prime bucket ranges of every Engram table.

    ``primes[layer_index][ngram_index][head]`` is the modulus of one hash column; the
    columns of a layer are laid out consecutively in its table, 2-grams first.
    """

    layer_ids: Tuple[int, ...]
    num_embeddings: Tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    head_dim: int
    primes: Tuple[Tuple[Tuple[int, ...], ...], ...]

    @property
    def n_hash_cols(self) -> int:
        """Hash columns per position: one per (n-gram size >= 2, head)."""
        return (self.max_ngram_size - 1) * self.n_heads

    def flat_primes(self, layer_index: int) -> List[int]:
        """Column moduli of one layer in table order."""
        return [p for per_ngram in self.primes[layer_index] for p in per_ngram]

    def offsets(self, layer_index: int) -> List[int]:
        """Row offset of every hash column inside the layer's table."""
        sizes = self.flat_primes(layer_index)
        offsets, acc = [], 0
        for size in sizes:
            offsets.append(acc)
            acc += size
        return offsets

    def rows_required(self, layer_index: int) -> int:
        """Rows addressed by the layout (must not exceed the table size)."""
        return sum(self.flat_primes(layer_index))

    @classmethod
    def build(
        cls,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        bucket_size: int,
    ) -> "EngramTableLayout":
        """Draw the primes: every (layer, n-gram size, head) takes the next unused prime
        above ``bucket_size - 1``, so all ranges are disjoint and deterministic."""
        if len(layer_ids) != len(num_embeddings):
            raise ValueError("engram_num_embeddings must match engram_layer_ids")
        used: set = set()
        primes = []
        for _ in layer_ids:
            per_layer = []
            for _ in range(max_ngram_size - 1):
                per_ngram = []
                current = bucket_size - 1
                for _ in range(n_heads):
                    current = next_unused_prime(current, used)
                    used.add(current)
                    per_ngram.append(current)
                per_layer.append(tuple(per_ngram))
            primes.append(tuple(per_layer))
        layout = cls(
            layer_ids=tuple(int(x) for x in layer_ids),
            num_embeddings=tuple(int(x) for x in num_embeddings),
            max_ngram_size=int(max_ngram_size),
            n_heads=int(n_heads),
            head_dim=int(head_dim),
            primes=tuple(primes),
        )
        for i, rows in enumerate(layout.num_embeddings):
            needed = layout.rows_required(i)
            if needed > rows:
                raise ValueError(
                    f"Engram table for model layer {layout.layer_ids[i]} has {rows} rows but the "
                    f"bucket layout addresses {needed}"
                )
        return layout


def hash_multipliers(
    layer_ids: Sequence[int], max_ngram_size: int, compressed_vocab_size: int
) -> torch.Tensor:
    """Odd per-(layer, look-back) multipliers, bounded so ``id * multiplier`` fits int64.

    Drawn from ``numpy.random.default_rng(10007 * layer_id)`` as in the released model.
    """
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        rng = np.random.default_rng(10007 * int(layer_id))
        draw = rng.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.from_numpy(draw * 2 + 1))
    return torch.stack(rows)


# ---------------------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------------------


class NgramHasher(nn.Module):
    """Maps token ids to the Engram row ids of the n-grams ending at each position.

    Training-time variant: whole sequences, no decode cache. Context parallelism needs the
    ``max_ngram_size - 1`` tokens before the local chunk; that halo arrives with M1.
    """

    def __init__(
        self,
        layout: EngramTableLayout,
        token_map: torch.Tensor,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.layout = layout
        if token_map.dim() != 1:
            raise ValueError("token_map must be a 1-D tensor indexed by token id")
        if int(token_map.max()) >= compressed_vocab_size:
            raise ValueError(
                "token_map addresses ids beyond engram_compressed_vocab_size "
                f"({int(token_map.max())} >= {compressed_vocab_size})"
            )
        self.register_buffer("token_map", token_map.to(torch.int64), persistent=False)
        self.pad_id = int(token_map[pad_token_id])
        primes = torch.tensor(layout.primes, dtype=torch.int64)  # [L, ngram-1, H]
        offsets = torch.tensor(
            [layout.offsets(i) for i in range(len(layout.layer_ids))], dtype=torch.int64
        )  # [L, n_cols]
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer(
            "multipliers",
            hash_multipliers(layout.layer_ids, layout.max_ngram_size, compressed_vocab_size),
            persistent=False,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``input_ids [b, s]`` -> row ids ``[s, b, n_layers, n_hash_cols]`` (int64).

        ``cu_seqlens`` (packed layout, ``b == 1``) makes the look-back stop at segment starts,
        so an n-gram never spans two packed sequences.
        """
        b, s = input_ids.shape
        compressed = self.token_map[input_ids]
        if token_mask is not None:
            compressed = torch.where(
                token_mask, compressed, torch.full_like(compressed, DEAD_TOKEN)
            )

        if cu_seqlens is None:
            positions = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        else:
            if b != 1:
                raise ValueError("packed Engram hashing expects a single batch row")
            cu = cu_seqlens.to(torch.int64)
            rows = torch.arange(s, device=input_ids.device, dtype=torch.int64)
            seg = torch.bucketize(rows, cu[1:], right=True).clamp_max(cu.numel() - 2)
            positions = (rows - cu[seg]).clamp_min(0).unsqueeze(0)
        blocked = torch.zeros(b, s, dtype=torch.bool, device=input_ids.device)
        history = []
        row_index = torch.arange(s, device=input_ids.device, dtype=torch.int64).unsqueeze(0)
        for shift in range(self.layout.max_ngram_size):
            # Token ``shift`` rows back; rows before the start read row 0 and are blocked below
            # (``positions < shift`` also covers packed segment starts), so any length works.
            source = compressed.gather(1, (row_index - shift).clamp_min(0).expand(b, s))
            # Once the look-back leaves the sequence or hits a dead token, this and every
            # longer n-gram fall back to the pad id.
            blocked = blocked | (positions < shift) | (source == DEAD_TOKEN)
            history.append(torch.where(blocked, torch.full_like(source, self.pad_id), source))
        history = torch.stack(history, dim=-1)  # [b, s, max_ngram]

        mixed = history.unsqueeze(2) * self.multipliers  # [b, s, L, max_ngram]
        rolling = mixed[..., 0]
        columns = []
        for i in range(1, self.layout.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, mixed[..., i])
            columns.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])  # [b, s, L, H]
        rows = torch.cat(columns, dim=-1) + self.offsets  # [b, s, L, n_cols]
        return rows.permute(1, 0, 2, 3).contiguous()


# ---------------------------------------------------------------------------------------
# Memory module
# ---------------------------------------------------------------------------------------


class EngramMemory(MegatronModule):
    """One Engram layer: sharded table lookup, key/value projection, gated residual add."""

    def __init__(
        self,
        config: TransformerConfig,
        layout: EngramTableLayout,
        layer_index: int,
        shard_group: Optional[dist.ProcessGroup] = None,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        self.layout = layout
        self.layer_index = layer_index
        self.model_layer_id = layout.layer_ids[layer_index]
        self.n_streams = config.num_residual_streams
        self.hidden_size = config.hidden_size
        self.head_dim = layout.head_dim
        self.n_hash_cols = layout.n_hash_cols
        self.eps = config.layernorm_epsilon
        self.gate_clamp = 1e-6

        self.shard_group = shard_group
        self.shard_size = dist.get_world_size(shard_group) if shard_group is not None else 1
        self.shard_rank = dist.get_rank(shard_group) if shard_group is not None else 0
        total_rows = layout.num_embeddings[layer_index]
        self.total_rows = total_rows
        self.rows_per_shard = math.ceil(total_rows / self.shard_size)
        self.row_start = self.shard_rank * self.rows_per_shard

        dtype = config.params_dtype
        self.embedding_rows = nn.Parameter(
            torch.empty(self.rows_per_shard, self.head_dim, dtype=dtype)
        )
        config.init_method(self.embedding_rows)
        if self.shard_size > 1:
            # Each rank holds a different row shard: exclude it from data-parallel
            # broadcast / all-reduce (same marker as expert-parallel MoE weights).
            self.embedding_rows.allreduce = False
        self.linear_wkv = nn.Linear(
            self.n_hash_cols * self.head_dim,
            self.hidden_size * (self.n_streams + 1),
            bias=False,
            dtype=dtype,
        )
        config.init_method(self.linear_wkv.weight)
        self.q_weight = mark_keep_in_fp32(
            nn.Parameter(torch.ones(self.n_streams, self.hidden_size, dtype=torch.float32))
        )
        self.k_weight = mark_keep_in_fp32(
            nn.Parameter(torch.ones(self.n_streams, self.hidden_size, dtype=torch.float32))
        )

        self.frozen = bool(getattr(config, "engram_frozen", True))
        if self.frozen:
            for param in self.parameters():
                param.requires_grad_(False)
        elif self.shard_size > 1:
            raise NotImplementedError("trainable Engram tables with row sharding are not supported")

    # ---- lookup ---------------------------------------------------------------------------

    def _lookup_local(self, row_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(row_ids, self.embedding_rows)

    @torch.no_grad()
    def _lookup_sharded(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Fetch rows owned by other ranks with an all-to-all exchange of ids and rows."""
        flat = row_ids.reshape(-1)
        owner = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        order = torch.argsort(owner, stable=True)
        sorted_ids = flat[order]
        send_counts = torch.bincount(owner, minlength=self.shard_size)
        if send_counts.numel() != self.shard_size:
            # A row id outside the table would make the count vector longer than the shard
            # group; the all-to-all below would then exchange mismatched sizes and return
            # garbage counts (seen as huge `new_empty` sizes) instead of failing cleanly.
            raise RuntimeError(
                f"Engram row ids outside the table: max id {int(flat.max())}, table rows "
                f"{self.rows_per_shard * self.shard_size} ({self.shard_size} shards of "
                f"{self.rows_per_shard}); min id {int(flat.min())}"
            )
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.shard_group)
        send_list = send_counts.tolist()
        recv_list = recv_counts.tolist()
        if min(recv_list) < 0 or sum(recv_list) > 2**40:
            raise RuntimeError(
                f"Engram shard exchange returned invalid counts {recv_list[:8]}...: the ranks of "
                f"the shard group (size {self.shard_size}) called the exchange with different "
                f"shapes or on different layers"
            )

        requested = sorted_ids.new_empty(sum(recv_list))
        dist.all_to_all_single(
            requested,
            sorted_ids,
            output_split_sizes=recv_list,
            input_split_sizes=send_list,
            group=self.shard_group,
        )
        served = F.embedding(requested - self.row_start, self.embedding_rows)
        returned = served.new_empty(flat.numel(), self.head_dim)
        dist.all_to_all_single(
            returned,
            served,
            output_split_sizes=send_list,
            input_split_sizes=recv_list,
            group=self.shard_group,
        )
        rows = torch.empty_like(returned)
        rows[order] = returned
        return rows.view(*row_ids.shape, self.head_dim)

    def lookup(self, row_ids: torch.Tensor) -> torch.Tensor:
        """``[s, b, n_hash_cols]`` row ids -> ``[s, b, n_hash_cols, head_dim]`` rows."""
        if self.shard_size == 1:
            return self._lookup_local(row_ids)
        return self._lookup_sharded(row_ids)

    # ---- forward ----------------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        row_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add the gated n-gram value to every residual stream.

        Args:
            hidden_states: ``[s, b, n_streams * hidden]``.
            row_ids: ``[s, b, n_hash_cols]`` from :class:`NgramHasher` (this layer's slice).
            token_mask: ``[s, b]`` bool, False shuts the gate for that position.
        """
        s, b, _ = hidden_states.shape
        n, c = self.n_streams, self.hidden_size

        rows = self.lookup(row_ids).flatten(-2)  # [s, b, n_hash_cols * head_dim]
        kv = self.linear_wkv(rows.to(self.linear_wkv.weight.dtype))
        key, value = kv.split([n * c, c], dim=-1)
        key = key.float().view(s, b, n, c)
        streams = hidden_states.float().view(s, b, n, c)

        weight = self.q_weight.float() * self.k_weight.float()  # [n, c]
        stream_rstd = torch.rsqrt(streams.square().mean(-1) + self.eps)
        key_rstd = torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (streams * weight * key).sum(-1) * stream_rstd * key_rstd * c**-0.5  # [s, b, n]
        # Signed square root keeps the gate sensitive around zero (training-kernel convention).
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.gate_clamp).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0.0)
        out = streams + gate.unsqueeze(-1) * value.float().unsqueeze(2)
        return out.to(hidden_states.dtype).view(s, b, n * c)

    # ---- checkpointing ------------------------------------------------------------------------

    def sharded_state_dict(self, prefix: str = "", sharded_offsets: tuple = (), metadata=None):
        """Replicated parameters use the default path; the row-sharded table is split on dim 0."""
        sharded_sd = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.shard_size > 1:
            from megatron.core import parallel_state
            from megatron.core.dist_checkpointing.mapping import ShardedTensor

            key = f"{prefix}embedding_rows"
            # The stored global shape is ceil(rows / shards) * shards, i.e. it depends on the
            # sharding degree like Megatron's vocab-padded embeddings; allow_shape_mismatch
            # follows that convention so tables reload across degrees (padding rows are
            # never addressed by the hash layout). Cross-degree reload is exercised in M2.
            sharded_sd[key] = ShardedTensor.from_rank_offsets(
                key,
                self.embedding_rows,
                *sharded_offsets,
                (0, self.shard_rank, self.shard_size),
                replica_id=(0, 0, parallel_state.get_expert_data_parallel_rank()),
                allow_shape_mismatch=True,
            )
        return sharded_sd


def load_token_map(path: Optional[str], vocab_size: int) -> torch.Tensor:
    """Load the token -> compressed id map, or return the identity map for tests."""
    if path is None:
        return torch.arange(vocab_size, dtype=torch.int64)
    payload = torch.load(path, map_location="cpu")
    token_map = payload["token_map"] if isinstance(payload, dict) else payload
    return torch.as_tensor(token_map, dtype=torch.int64)
