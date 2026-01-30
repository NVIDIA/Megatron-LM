# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from dataclasses import dataclass
from typing import List, Optional, Type

from megatron.core.transformer.transformer_config import TransformerConfig


class KVCache:
    '''Base of KV cache for prefix caching.'''

    def __init__(self, config: TransformerConfig):
        self.config = config


class KVCachePool:
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.kv_cache_pool = {}

    def get_kv_cache(self, layer_idx: int, kv_cache_cls: Type[KVCache]) -> KVCache:
        if layer_idx not in self.kv_cache_pool:
            self.kv_cache_pool[layer_idx] = kv_cache_cls(self.config)
        assert isinstance(self.kv_cache_pool[layer_idx], kv_cache_cls), (
            f"Expected KV cache of type {kv_cache_cls}, "
            f"but got {type(self.kv_cache_pool[layer_idx])}."
        )
        return self.kv_cache_pool[layer_idx]


@dataclass
class CachedPrefixParams:
    '''Parameters for prefix caching.
    
    Args:
        prefix_seqlens: Sequence lengths of the prefixes
        this_chunk_seqlen: Sequence length of the current chunk
        max_total_seqlen: Maximum total sequence length globally
        kv_cache_pool: KV cache pool of the current micro batch
    
    Note: All seqlens here ignore the parallelism.
    For example, if we split a sequence of length 4096 into 4 evenly-sized chunks,
    then for the i-th chunk (i=0,1,2,3), whatever the TP and CP sizes are, we always have:
      - prefix_seqlens[i] = [1024] * i
      - this_chunk_seqlen = 1024
      - max_total_seqlen = 4096
    '''

    prefix_seqlens: List[int]  # Sequence lengths of the prefixes
    this_chunk_seqlen: int  # Sequence length of the current chunk
    max_total_seqlen: Optional[int]  # Maximum total sequence length globally
    kv_cache_pool: KVCachePool

    def __hash__(self):
        return hash(
            (tuple(self.prefix_seqlens), self.this_chunk_seqlen, self.max_total_seqlen)
        )
