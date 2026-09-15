# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023 DeepSeek
#
# Portions adapted from https://github.com/deepseek-ai/Engram.
# Upstream commit: fb7f84a21f91223715394a33a1dc24bbfb7f788e (engram_demo_v1.py).
# Modified to use the runtime tokenizer, cache construction-time projections,
# and register a device-aware PyTorch lookup buffer.
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

"""Vocabulary projection construction and runtime mapping for Engram."""

from typing import Any, Sequence
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

_LOOKUP_CACHE: WeakKeyDictionary[Any, tuple[int, Tensor]] = WeakKeyDictionary()


def get_engram_tokenizer_pad_id(tokenizer: Any, configured_pad_id: int | None = None) -> int:
    """Resolve Engram's raw pad ID against the Megatron Core tokenizer API."""
    if configured_pad_id is not None:
        return configured_pad_id

    for attribute in ("pad_id", "pad"):
        try:
            pad_id = getattr(tokenizer, attribute)
        except (AttributeError, NotImplementedError):
            continue
        if pad_id is not None and pad_id >= 0:
            return int(pad_id)

    for attribute in ("eod", "eod_id", "eos", "eos_id"):
        try:
            fallback_id = getattr(tokenizer, attribute)
        except (AttributeError, NotImplementedError):
            continue
        if fallback_id is not None and fallback_id >= 0:
            return int(fallback_id)

    raise ValueError(
        "Engram could not resolve a pad ID from the main tokenizer; "
        "set --engram-pad-id explicitly"
    )


def _unwrap_engram_tokenizer(tokenizer: Any) -> Any:
    """Unwrap MegatronTokenizerText and its Hugging Face library wrapper."""
    current = tokenizer
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (
            callable(getattr(current, "decode", None))
            and callable(getattr(current, "convert_ids_to_tokens", None))
            and hasattr(current, "__len__")
        ):
            return current
        for attribute in ("_tokenizer", "tokenizer"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                current = candidate
                break
        else:
            break
    raise TypeError(
        "Engram requires the runtime tokenizer to expose len(), decode(), and "
        "convert_ids_to_tokens(); expected MegatronTokenizerText wrapping a Hugging Face tokenizer"
    )


def build_engram_tokenizer_lookup(tokenizer: Any) -> torch.Tensor:
    """Build a normalized-token lookup from the actual runtime Hugging Face tokenizer."""
    tokenizer = _unwrap_engram_tokenizer(tokenizer)
    vocab_size = len(tokenizer)
    try:
        cached = _LOOKUP_CACHE.get(tokenizer)
    except TypeError:
        cached = None
    if cached is not None and cached[0] == vocab_size:
        return cached[1]

    from tokenizers import Regex, normalizers

    sentinel = '\ue000'
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r'[ \t\r\n]+'), ' '),
            normalizers.Replace(Regex(r'^ $'), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, ' '),
        ]
    )

    key_to_id: dict[str, int] = {}
    lookup = torch.empty(vocab_size, dtype=torch.long)
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if '�' in text:
            key = tokenizer.convert_ids_to_tokens(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        lookup[token_id] = key_to_id.setdefault(key, len(key_to_id))
    try:
        _LOOKUP_CACHE[tokenizer] = (vocab_size, lookup)
    except TypeError:
        pass
    return lookup


class CompressedTokenizer(nn.Module):
    """Apply a precomputed vocabulary projection ``P: V -> V'`` at runtime."""

    def __init__(self, lookup_table: Tensor | Sequence[int]) -> None:
        super().__init__()
        lookup = torch.as_tensor(lookup_table, dtype=torch.long).clone()
        if lookup.ndim != 1:
            raise ValueError('CompressedTokenizer lookup_table must be one-dimensional')
        self.register_buffer('lookup_table', lookup, persistent=True)
        self.num_new_token = int(torch.unique(lookup).numel())

    def __len__(self) -> int:
        return self.num_new_token

    def _compress(self, input_ids: Tensor) -> Tensor:
        input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.lookup_table.device)
        positive = input_ids >= 0
        if positive.any():
            valid_ids = input_ids[positive]
            if torch.any(valid_ids >= self.lookup_table.numel()):
                raise ValueError('Token ID is outside the CompressedTokenizer vocabulary')
            output = input_ids.clone()
            output[positive] = self.lookup_table[valid_ids]
            return output
        return input_ids.clone()

    def forward(self, input_ids: Tensor) -> Tensor:
        """Project raw token IDs into the compact Engram vocabulary."""
        return self._compress(input_ids)
