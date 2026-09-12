# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Eager Engram memory for Megatron Core."""

from .addressing import NgramHashMapping, find_next_prime
from .config import EngramConfig
from .fusion import EngramLayer, ShortConv
from .layer import Engram, ParallelSequenceLayout
from .memory import (
    MultiHeadEmbedding,
    RowShardedMultiHeadEmbedding,
    RowShardedTableLayout,
    row_shard_bounds,
    row_shard_owner,
)
from .tokenizer import CompressedTokenizer

__all__ = [
    'CompressedTokenizer',
    'Engram',
    'EngramConfig',
    'EngramLayer',
    'MultiHeadEmbedding',
    'NgramHashMapping',
    'ParallelSequenceLayout',
    'RowShardedMultiHeadEmbedding',
    'RowShardedTableLayout',
    'ShortConv',
    'find_next_prime',
    'row_shard_bounds',
    'row_shard_owner',
]
