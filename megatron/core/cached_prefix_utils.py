# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional, Type

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


class KVCache:
    '''Base of KV cache for prefix caching.'''

    def __init__(self, config: TransformerConfig):
        self.config = config


class KVCachePool:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.kv_cache_pool = {}
        self.block_prefixes = []

    def block_range_push(self, suffix: str):
        """Push a block range onto the stack."""

        self.block_prefixes.append(suffix)

    def block_range_pop(self, suffix: Optional[str] = None):
        """Pop a block range from the stack."""

        assert len(self.block_prefixes) > 0, "No block range to pop."
        if suffix is not None:
            assert (
                self.block_prefixes[-1] == suffix
            ), f"Expected block range suffix {suffix}, but got {self.block_prefixes[-1]}."
        self.block_prefixes.pop()

    def get_kv_cache(self, layer_idx: int, kv_cache_cls: Type[KVCache]) -> KVCache:
        """Get the KV cache for the given layer index."""

        if tuple(self.block_prefixes) not in self.kv_cache_pool:
            self.kv_cache_pool[tuple(self.block_prefixes)] = {}

        if layer_idx not in self.kv_cache_pool[tuple(self.block_prefixes)]:
            # Create a new KV cache for the given layer index.
            self.kv_cache_pool[tuple(self.block_prefixes)][layer_idx] = []
        pool_of_current_block_and_layer = self.kv_cache_pool[tuple(self.block_prefixes)][layer_idx]

        found_exist = False
        for kv_cache in pool_of_current_block_and_layer:
            if isinstance(kv_cache, kv_cache_cls):
                return kv_cache
        pool_of_current_block_and_layer.append(kv_cache_cls(self.config))
        return pool_of_current_block_and_layer[-1]


@dataclass
class CachedPrefixParams:
    '''Parameters for prefix caching.

    Args:
        prefix_seqlens: Sequence lengths of the prefixes
        this_chunk_seqlen: Sequence length of the current chunk
        max_total_seqlen: Maximum total sequence length globally
        kv_cache_pool: KV cache pool of the current micro batch
        boundary_elements_for_mtp: Boundary elements for MTP rolling.
        is_terminal: Whether the current chunk is the terminal chunk.

    Note: All seqlens here ignore the parallelism.
    For example, if we split a sequence of length 4096 into 4 evenly-sized chunks,
    then for the i-th chunk (i=0,1,2,3), whatever the TP and CP sizes are, we always have:
      - prefix_seqlens[i] = [1024] * i
      - this_chunk_seqlen = 1024
      - max_total_seqlen = 4096
    '''

    prefix_seqlens: List[int]
    this_chunk_seqlen: int
    max_total_seqlen: Optional[int]
    kv_cache_pool: KVCachePool
    boundary_elements_for_mtp: Optional[torch.Tensor]
    is_terminal: bool

    def __hash__(self):
        return hash((tuple(self.prefix_seqlens), self.this_chunk_seqlen, self.max_total_seqlen))
