# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""
Engram module for Megatron-LM.

Ported from DeepSeek's Engram reference implementation. The Engram module augments
transformer layers with n-gram hash-based embeddings that are gated against the
hidden states via a multi-head key-query mechanism.

The module operates in two phases:
  1. Pre-compute: Hash input_ids into n-gram IDs and look up embeddings (called once
     per forward pass at the model level).
  2. Forward: Gate the pre-computed embeddings against hidden states and apply a
     short causal convolution (called per layer).

Reference: https://github.com/deepseek-ai/Engram
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sympy import isprime
from tokenizers import Regex, normalizers
from transformers import AutoTokenizer


@dataclass
class EngramConfig:
    """Configuration for the Engram module."""

    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    engram_layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    hc_mult: int = 4
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"

    def __post_init__(self) -> None:
        expected_vocab_sizes = self.max_ngram_size - 1
        if len(self.engram_vocab_size) != expected_vocab_sizes:
            raise ValueError(
                "engram_vocab_size must provide one entry per n-gram size from 2 to "
                f"{self.max_ngram_size}, but got {len(self.engram_vocab_size)} entries."
            )
        if self.n_head_per_ngram <= 0:
            raise ValueError("n_head_per_ngram must be positive.")
        if self.n_embed_per_ngram % self.n_head_per_ngram != 0:
            raise ValueError("n_embed_per_ngram must be divisible by n_head_per_ngram.")
        if any(layer_id < 1 for layer_id in self.engram_layer_ids):
            raise ValueError("engram_layer_ids must be 1-based positive layer indices.")


class CompressedTokenizer:
    """Normalizes tokens into a compressed vocabulary via unicode normalization.

    Reduces the effective vocabulary size by mapping visually/semantically similar
    tokens to the same compressed ID. This is used as a preprocessing step before
    n-gram hashing.
    """

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def __call__(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out


def _find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


def _calculate_vocab_size_across_layers(
    engram_vocab_size: List[int],
    max_ngram_size: int,
    n_head_per_ngram: int,
    layer_ids: List[int],
) -> Dict[int, List[List[int]]]:
    seen_primes: set = set()
    vocab_size_across_layers: Dict[int, List[List[int]]] = {}

    for layer_id in layer_ids:
        all_ngram_vocab_sizes: List[List[int]] = []
        for ngram in range(2, max_ngram_size + 1):
            current_ngram_heads_sizes: List[int] = []
            vocab_size = engram_vocab_size[ngram - 2]
            current_prime_search_start = vocab_size - 1

            for _ in range(n_head_per_ngram):
                found_prime = _find_next_prime(current_prime_search_start, seen_primes)
                seen_primes.add(found_prime)
                current_ngram_heads_sizes.append(found_prime)
                current_prime_search_start = found_prime

            all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
        vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

    return vocab_size_across_layers


def calculate_engram_vocab_size_across_layers(
    engram_config: EngramConfig,
) -> Dict[int, List[List[int]]]:
    """Return deterministic per-layer hash vocab sizes for an Engram configuration."""

    return _calculate_vocab_size_across_layers(
        engram_vocab_size=engram_config.engram_vocab_size,
        max_ngram_size=engram_config.max_ngram_size,
        n_head_per_ngram=engram_config.n_head_per_ngram,
        layer_ids=engram_config.engram_layer_ids,
    )


class NgramHashMapping:
    """Deterministic n-gram hash mapping for Engram.

    Computes hash IDs for n-grams (bigrams, trigrams, ...) using random multipliers
    and modular arithmetic with prime-sized hash tables. Each n-gram level has multiple
    heads, each mapping to a distinct prime-sized vocabulary.
    """

    def __init__(
        self,
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_embed_per_ngram: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_name_or_path: str,
        pad_id: int,
        seed: int,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers: Dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self._calculate_vocab_size_across_layers()

    def _calculate_vocab_size_across_layers(self) -> Dict[int, List[List[int]]]:
        return _calculate_vocab_size_across_layers(
            engram_vocab_size=self.vocab_size_per_ngram,
            max_ngram_size=self.max_ngram_size,
            n_head_per_ngram=self.n_head_per_ngram,
            layer_ids=self.layer_ids,
        )

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(x, ((0, 0), (k, 0)), mode='constant', constant_values=self.pad_id)[
                :, :T
            ]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids) -> Dict[int, np.ndarray]:
        """Compute n-gram hash IDs for all engram layers.

        Args:
            input_ids: Token IDs of shape [B, T] (numpy array or tensor).

        Returns:
            Dict mapping layer_id -> hash IDs of shape [B, T, num_total_heads].
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers


class MultiHeadEmbedding(nn.Module):
    """Packs multiple embedding tables (one per hash head) into a single table.

    Uses per-head offsets so that each head's IDs map to a non-overlapping region
    of a single large nn.Embedding.
    """

    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T, num_heads] hash IDs.

        Returns:
            [B, T, num_heads, D] embeddings.
        """
        shifted_input_ids = input_ids + self.offsets
        return self.embedding(shifted_input_ids)


class ShortConv(nn.Module):
    """Depthwise causal 1D convolution with per-group RMSNorm.

    Operates on tensors of shape [B, L, HC_MULT, D], applying a grouped depthwise
    convolution across the sequence dimension with causal padding.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, HC_MULT, D]

        Returns:
            [B, L, HC_MULT, D]
        """
        B, T, G, C = x.shape
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1)  # [B, T, G*C]
        x_bct = x_norm.transpose(1, 2)  # [B, G*C, T]
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]  # causal: trim future padding

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

        return y


class EngramModule(nn.Module):
    """Core Engram module that augments transformer hidden states with n-gram
    hash-based embeddings via multi-head gating and causal convolution.

    This module operates in two phases:
      1. ``precompute_embeddings(hash_ids, device)``: Called once per forward pass at
         the model level to convert hash IDs into embeddings.
      2. ``forward(hidden_states)``: Called per layer to gate embeddings against
         hidden states.

    The hidden states are internally expanded to [B, S, HC_MULT, D] for multi-slot
    gating, then collapsed back by averaging across the HC dimension.
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        engram_config: EngramConfig,
        vocab_size_for_layer: List[List[int]],
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.hc_mult = engram_config.hc_mult
        self.max_ngram_size = engram_config.max_ngram_size
        self.n_embed_per_ngram = engram_config.n_embed_per_ngram
        self.n_head_per_ngram = engram_config.n_head_per_ngram

        head_vocab_sizes = [x for ngram_sizes in vocab_size_for_layer for x in ngram_sizes]
        per_head_dim = engram_config.n_embed_per_ngram // engram_config.n_head_per_ngram

        self.multi_head_embedding = MultiHeadEmbedding(list_of_N=head_vocab_sizes, D=per_head_dim)

        self.short_conv = ShortConv(
            hidden_size=hidden_size,
            kernel_size=engram_config.kernel_size,
            dilation=engram_config.max_ngram_size,
            hc_mult=engram_config.hc_mult,
        )

        engram_hidden_size = (engram_config.max_ngram_size - 1) * engram_config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size, hidden_size) for _ in range(engram_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(engram_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(engram_config.hc_mult)])

        self._cached_embeddings: Optional[torch.Tensor] = None

    def precompute_embeddings(self, hash_ids: np.ndarray, device: torch.device) -> None:
        """Pre-compute embeddings from hash IDs. Called once per forward pass.

        Args:
            hash_ids: [B, T, num_heads] numpy array of hash IDs for this layer.
            device: Target device for the embedding tensor.
        """
        hash_tensor = torch.from_numpy(hash_ids).to(device)
        embeddings = self.multi_head_embedding(hash_tensor)  # [B, T, num_heads, D_head]
        self._cached_embeddings = embeddings.flatten(start_dim=-2)  # [B, T, engram_hidden]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply engram gating to hidden states using pre-computed embeddings.

        Args:
            hidden_states: [S, B, H] tensor in Megatron's sequence-first format.

        Returns:
            [S, B, H] engram output to be added as a residual.
        """
        assert (
            self._cached_embeddings is not None
        ), "Must call precompute_embeddings() before forward()"
        embeddings = self._cached_embeddings  # [B, T, engram_hidden]

        # Megatron uses [S, B, H]; convert to [B, S, H] for engram processing
        hidden_states_bsh = hidden_states.transpose(0, 1).contiguous()

        # Expand to HC_MULT slots: [B, S, HC_MULT, H]
        hidden_states_hc = hidden_states_bsh.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)

        # Compute gates per HC slot
        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states_hc[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates = torch.stack(gates, dim=2)  # [B, S, HC_MULT, 1]

        # Gated value with short convolution
        value = gates * self.value_proj(embeddings).unsqueeze(2)  # [B, S, HC_MULT, H]
        output = value + self.short_conv(value)  # [B, S, HC_MULT, H]

        # Collapse HC dimension by averaging, convert back to [S, B, H]
        output = output.mean(dim=2)  # [B, S, H]
        output = output.transpose(0, 1).contiguous()  # [S, B, H]

        self._cached_embeddings = None

        return output
