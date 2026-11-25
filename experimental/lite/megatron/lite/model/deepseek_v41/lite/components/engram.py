# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Release-specific hash seeds, table sizes and Engram model assembly."""
import numpy as np
import torch
from torch import nn

from megatron.lite.primitive.modules.engram import Engram, ShardedFrozenTable


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


class DS41Engram(Engram):
    def __init__(self, release, config, layer, table_group, *, frozen=True):
        if not frozen:
            raise NotImplementedError("DS4.1 Engram tables are frozen host FP8; not trainable")
        position = release.engram_layer_ids.index(layer)
        rows = release.engram_num_embeddings[position]
        init = config.perform_initialization  # False when the release is loaded
        table = ShardedFrozenTable(rows, release.engram_head_dim, table_group, initialize=init)
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
