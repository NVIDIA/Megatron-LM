# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The fused Engram hash kernel (EngramHasher, MCORE_ENGRAM_FUSED_HASH) replays bit-exactly.

The kernel is pure int64 arithmetic with one writer per output element (no reduction, no
atomics), so replays must be byte-identical; the replay runs at the DSv4.1 proxy shape (2 hashed
layers x 3 n-gram orders x 8 heads, 4096 local tokens plus the 3-token CP halo) with THD
documents, a token mask and side-stream contention, and the result must also equal the PyTorch
reference arithmetic. ``tests/unit_tests/models/test_v41_engram_hash.py`` covers the semantics.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.deepseek_v41 import engram_hash_kernels
from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.engram import EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and engram_hash_kernels.HAVE_TRITON),
    reason="needs a GPU and Triton",
)


def _proxy_hasher():
    """The proxy's two hashed layers (blocks 1 and 14), 4-grams and 8 heads."""
    args = SimpleNamespace(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[0, 0],
        engram_max_ngram_size=4,
        engram_vocab_size=1_000_003,
        engram_n_heads=8,
        engram_head_dim=4,
    )
    rows = [sum(map(sum, primes)) for primes in EngramLayout.from_args(args).primes]
    options = EngramConfig([1, 14], rows, 4, 1_000_003, 8, 4, 2, 98627)
    token_map = torch.cat((torch.arange(98627), torch.randint(0, 98627, (129280 - 98627,))))
    return EngramHasher(options, token_map[torch.randperm(129280)]).cuda()


def test_engram_hash_replays_under_contention():
    seeded()
    hasher = _proxy_hasher()
    batch, local_length, halo = 2, 4096, 3
    length = local_length + halo
    tokens = torch.randint(0, 129280, (batch, length), device="cuda")
    starts = torch.rand(batch, length, device="cuda") < 0.05
    sequence_ids = starts.long().cumsum(1)
    token_mask = torch.rand(batch, length, device="cuda") < 0.95
    inputs = (
        tokens,
        token_mask,
        sequence_ids,
        hasher.token_map,
        hasher.multipliers,
        hasher.primes,
        hasher.offsets,
        hasher.pad_id,
        local_length,
    )
    outputs, _ = assert_replays_bit_exact(
        engram_hash_kernels.fused_engram_hash,
        inputs,
        backward=False,
        replays=3,
        contention=True,
        what="engram hash",
    )
    expected = hasher._reference_hash(tokens, token_mask, sequence_ids, local_length)
    assert bytes_equal(outputs["out"], expected)
