# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023 DeepSeek
#
# Portions adapted from https://github.com/deepseek-ai/Engram.
# Upstream commit: fb7f84a21f91223715394a33a1dc24bbfb7f788e (engram_demo_v1.py).
# Modified for PyTorch tensor hashing, registered state, explicit token mapping,
# and globally resolved table capacities before pipeline partitioning.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""N-gram hashing and table addressing for Engram."""

from math import isqrt
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .tokenizer import CompressedTokenizer


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, isqrt(value) + 1, 2))


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    """Return the next prime strictly larger than ``start`` that is not in ``seen_primes``."""
    candidate = start + 1
    while True:
        if _is_prime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping(nn.Module):
    """Multiplicative-XOR n-gram mapping implemented with PyTorch."""

    def __init__(
        self,
        hash_table_min_sizes: Sequence[int],
        max_ngram_size: int,
        num_hash_heads_per_ngram: int,
        layer_ids: Sequence[int],
        pad_id: int | None,
        seed: int,
        *,
        compressed_tokenizer: CompressedTokenizer,
    ) -> None:
        super().__init__()
        self.hash_table_min_sizes = tuple(hash_table_min_sizes)
        self.max_ngram_size = max_ngram_size
        self.num_hash_heads_per_ngram = num_hash_heads_per_ngram
        self.layer_ids = tuple(layer_ids)
        # The same consumer Engram module registers this tokenizer once; the hash
        # mapping only borrows it to avoid duplicate lookup-table checkpoint entries.
        object.__setattr__(self, 'compressed_tokenizer', compressed_tokenizer)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if pad_id is None:
            self.pad_id = None
        else:
            compressed_pad = self.compressed_tokenizer(torch.tensor([pad_id], dtype=torch.long))
            self.pad_id = int(compressed_pad.item())

        max_long = torch.iinfo(torch.int64).max
        half_bound = max(1, int(max_long // self.tokenizer_vocab_size) // 2)
        for layer_id in self.layer_ids:
            # Checkpoint ABI: match the official DeepSeek Engram demo exactly.
            # NumPy is used only to initialize persistent multiplier buffers.
            random_values = np.random.default_rng(seed + 10007 * layer_id).integers(
                low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64
            )
            self.register_buffer(
                f'layer_multipliers_{layer_id}',
                torch.from_numpy(random_values * 2 + 1),
                persistent=True,
            )
        self.hash_moduli_by_layer = self._resolve_hash_moduli_by_layer()

    def _resolve_hash_moduli_by_layer(self) -> dict[int, list[list[int]]]:
        """Resolve distinct prime moduli for every layer, n-gram order, and hash head."""
        seen_primes: set[int] = set()
        result = {}
        for layer_id in self.layer_ids:
            layer_hash_moduli = []
            for ngram in range(2, self.max_ngram_size + 1):
                head_hash_moduli = []
                search_start = self.hash_table_min_sizes[ngram - 2] - 1
                for _ in range(self.num_hash_heads_per_ngram):
                    hash_modulus = find_next_prime(search_start, seen_primes)
                    seen_primes.add(hash_modulus)
                    head_hash_moduli.append(hash_modulus)
                    search_start = hash_modulus
                layer_hash_moduli.append(head_hash_moduli)
            result[layer_id] = layer_hash_moduli
        return result

    def _get_ngram_hashes(self, input_ids: Tensor, layer_id: int) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("Engram input_ids must have shape [B, T]")
        if layer_id not in self.layer_ids:
            raise ValueError(f"Layer ID {layer_id} is not configured for Engram")
        if self.pad_id is None:
            raise ValueError("Engram n-gram hashing requires pad_id")
        batch, sequence = input_ids.shape
        positions = torch.arange(sequence, device=input_ids.device)
        base_shifts = []
        for shift in range(self.max_ngram_size):
            source = positions - shift
            valid = source >= 0
            source = source.clamp_min(0)
            shifted = input_ids[:, source]
            valid = valid.unsqueeze(0).expand(batch, -1)
            base_shifts.append(torch.where(valid, shifted, self.pad_id))

        multipliers = getattr(self, f"layer_multipliers_{layer_id}")
        all_hashes = []
        for ngram in range(2, self.max_ngram_size + 1):
            mix = base_shifts[0] * multipliers[0]
            for index in range(1, ngram):
                mix = torch.bitwise_xor(mix, base_shifts[index] * multipliers[index])
            for hash_modulus in self.hash_moduli_by_layer[layer_id][ngram - 2]:
                all_hashes.append(torch.remainder(mix, hash_modulus))
        return torch.stack(all_hashes, dim=2)

    def hash(self, input_ids: Tensor) -> dict[int, Tensor]:
        """Hash token IDs into n-gram table addresses for every Engram layer."""
        compressed = self.compressed_tokenizer(input_ids)
        return {
            layer_id: self._get_ngram_hashes(
                compressed.to(getattr(self, f"layer_multipliers_{layer_id}").device), layer_id
            )
            for layer_id in self.layer_ids
        }

    def forward(self, input_ids: Tensor, layer_id: int) -> Tensor:
        """Hash token IDs into n-gram table addresses for one Engram layer."""
        compressed = self.compressed_tokenizer(input_ids)
        device = getattr(self, f"layer_multipliers_{layer_id}").device
        return self._get_ngram_hashes(compressed.to(device), layer_id)

    def forward_compressed(self, compressed_input_ids: Tensor, layer_id: int) -> Tensor:
        """Hash IDs already projected through CompressedTokenizer."""
        device = getattr(self, f"layer_multipliers_{layer_id}").device
        return self._get_ngram_hashes(compressed_input_ids.to(device), layer_id)
