# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""KV-cache state for Gemma4 single-batch inference.

Used by the Gemma4 eval server. Single sequence, no batching, no paged
allocation, no MCore ``BaseInferenceContext`` integration -- this is the
minimum plumbing needed by ``Gemma4SelfAttention``'s inference branch to
attend over past K/V.

Cross-layer KV sharing
----------------------
Borrower layers (the last ``num_kv_shared_layers`` layers) do not store their
own K/V; they read from the producer layer's cache. The producer for a given
borrower's ``layer_type`` is the last own-layer of the same ``layer_type``
before ``first_shared_idx`` (matches HF ``first_kv_shared_layer_idx`` and
``store_full_length_kv`` from modeling_gemma4.py:1199-1204). ``producer_for``
maps 1-based ``layer_number`` to its producer's 1-based ``layer_number`` (or
to itself, for own layers).

Migrating this to a ``BaseInferenceContext`` subclass (so we can wire to
``GPTInferenceWrapper`` / ``DynamicInferenceEngine``) is a follow-up; the
small surface here (``is_decode``, ``get_kv``, ``append_kv``, ``advance``)
keeps that swap mechanical.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


class Gemma4InferenceContext:
    """Per-request KV-cache state for one Gemma4 generation."""

    def __init__(
        self,
        num_layers: int,
        layer_types: List[str],
        num_kv_shared_layers: int = 0,
    ) -> None:
        self.num_layers = num_layers
        self.producer_for: Dict[int, int] = self._compute_producer_map(
            num_layers, layer_types, num_kv_shared_layers
        )
        # cache[layer_number] -> (K, V) with layout [s, b, ng, hd] (matches
        # Gemma4SelfAttention.get_query_key_value_tensors output).
        self.cache: Dict[int, Tuple[Tensor, Tensor]] = {}
        # Number of tokens already absorbed by the cache. step == 0 marks the
        # prefill call; step > 0 marks decode steps.
        self.step: int = 0

    @staticmethod
    def _compute_producer_map(
        num_layers: int, layer_types: List[str], num_kv_shared_layers: int
    ) -> Dict[int, int]:
        first_shared_idx = num_layers - num_kv_shared_layers
        prev_types = layer_types[:first_shared_idx]
        # Producer per layer_type = last own-layer of that type before first_shared_idx.
        producer_by_type: Dict[str, int] = {}
        for t in set(prev_types):
            last_idx = len(prev_types) - 1 - prev_types[::-1].index(t)
            producer_by_type[t] = last_idx + 1  # 1-based layer_number
        producer_for: Dict[int, int] = {}
        for layer_idx in range(num_layers):
            layer_number = layer_idx + 1
            if layer_idx >= first_shared_idx:
                producer_for[layer_number] = producer_by_type[layer_types[layer_idx]]
            else:
                producer_for[layer_number] = layer_number
        return producer_for

    def is_decode(self) -> bool:
        return self.step > 0

    def get_kv(self, layer_number: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Return the (K, V) tensors this layer should attend over (own or producer's)."""
        return self.cache.get(self.producer_for[layer_number])

    def append_kv(
        self, layer_number: int, k_new: Tensor, v_new: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Concatenate new K/V into this layer's slot (own/producer only) and return the full slot.

        Borrower layers must read via ``get_kv`` and never call ``append_kv``.
        """
        assert self.producer_for[layer_number] == layer_number, (
            f"layer {layer_number} is a kv-shared (borrower) layer; "
            "borrowers must read producer's cache via get_kv, not append"
        )
        prev = self.cache.get(layer_number)
        if prev is None:
            k, v = k_new, v_new
        else:
            k = torch.cat([prev[0], k_new], dim=0)
            v = torch.cat([prev[1], v_new], dim=0)
        self.cache[layer_number] = (k, v)
        return k, v

    def advance(self, n_tokens: int) -> None:
        self.step += n_tokens

    def reset(self) -> None:
        self.cache.clear()
        self.step = 0
