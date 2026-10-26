# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the Triton kernels of Qwen Sparse Attention (QSA).

* indexer top-k (``triton_qsa_select.qsa_select_topk_keys``): fp32 block scores under a
  pinned ``triton.autotune`` configuration, then an exact radix select with a unique-key
  tie-break;
* sparse-attention metadata (``triton_qsa`` builders over the ``triton_qsa_select`` kernels):
  query-tile supersets and the KV-tile query CSR. The KV bit masks are filled with
  ``tl.atomic_add`` on int32 sums of distinct powers of two, which is exact in any order; the
  replay also checks them against the sort-based builders;
* the fixed-order split-partial sum that finishes the dk / dv gradients
  (``cute_qsa_attn._split_sum_cast``).

The CuTe DSL attention and indexer-loss kernels need an SM100 GPU and ``nvidia-cutlass-dsl``;
they are not replayed here.
"""

import pytest
import torch

pytest.importorskip("triton")

from megatron.core.transformer.experimental_attention_variant.ops import cute_qsa_attn
from megatron.core.transformer.experimental_attention_variant.ops import triton_qsa as tq
from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
    qsa_select_topk_keys,
)
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and Triton")

NUM_BLOCKS = CONTENTION_TOKENS // tq.BS


@pytest.mark.parametrize("packed", [False, True], ids=["one_document", "packed_documents"])
def test_indexer_topk_keys_replay(packed):
    """Top-k keys with exact score ties: duplicated block keys and all-zero relu scores."""
    seeded()
    tokens, top_k = CONTENTION_TOKENS, 512
    q = torch.randn(tokens, 4, 128, device="cuda").bfloat16()
    block_keys = torch.randn(NUM_BLOCKS, 128, device="cuda").bfloat16()
    duplicate = torch.arange(1, NUM_BLOCKS, 3, device="cuda")
    block_keys[duplicate] = block_keys[duplicate - 1]
    block_keys[NUM_BLOCKS // 3 : NUM_BLOCKS // 3 + 40] = 0
    position = torch.arange(tokens, device="cuda")
    if packed:
        # Three documents share one block buffer: row t sees blocks [m_lo, m_t).
        cuts = torch.tensor([0, tokens // 5, tokens // 2, tokens], device="cuda")
        document = torch.searchsorted(cuts[1:], position, right=True)
        block_offsets = torch.nn.functional.pad(
            (cuts[1:] - cuts[:-1]).div(tq.BS, rounding_mode="floor").cumsum(0), (1, 0)
        )
        m_lo = block_offsets[document]
        inputs = (q, block_keys, m_lo + (position - cuts[document] + 1) // tq.BS, m_lo)
    else:
        inputs = (q, block_keys, (position + 1) // tq.BS)

    def fn(q, block_keys, m_t, m_lo=None):
        return qsa_select_topk_keys(q, block_keys, m_t, top_k, m_lo=m_lo, impl="triton")

    outputs, _ = assert_replays_bit_exact(
        fn, inputs, replays=6, backward=False, contention=True, what="qsa_select_topk_keys"
    )
    assert outputs["out"].shape == (tokens, top_k)


def _selection(batch, selected):
    """A real selection: distinct global block ids per query row, valid prefix ``counts``."""
    seeded()
    ids = torch.rand(batch, CONTENTION_TOKENS, NUM_BLOCKS, device="cuda").argsort(-1)
    ids = ids[..., :selected].contiguous()
    counts = torch.randint(1, selected + 1, (batch, CONTENTION_TOKENS), device="cuda")
    bases = (torch.arange(NUM_BLOCKS, device="cuda") * tq.BS).to(torch.int32)
    return ids, counts, bases, torch.clamp(bases + tq.BS, max=CONTENTION_TOKENS)


def _assert_same_fields(got, reference, names):
    for name in names:
        assert bytes_equal(getattr(got, name), getattr(reference, name)), name


@pytest.mark.parametrize("tile_queries", [4, 16], ids=["uint8_mask", "int32_mask"])
def test_superset_metadata_replays(tile_queries):
    """Query-tile supersets: bit-mask scatter, row counts and compaction."""
    ids, counts, bases, ends = _selection(batch=2, selected=65)

    def fn(ids, counts, bases, ends):
        return tq.build_qsa_supersets_static(ids, counts, bases, ends, NUM_BLOCKS, tile_queries)

    assert_replays_bit_exact(
        fn,
        (ids, counts, bases, ends),
        replays=6,
        backward=False,
        contention=True,
        what="build_qsa_supersets_static",
    )
    _assert_same_fields(
        fn(ids, counts, bases, ends),
        tq.build_qsa_supersets_static_torch(ids, counts, bases, ends, NUM_BLOCKS, tile_queries),
        ("tile_offsets", "entry_base", "entry_end", "membership"),
    )


@pytest.mark.parametrize(
    "builder",
    [tq.build_qsa_kv_tile_csr, tq.build_qsa_kv_tile_csr_torch],
    ids=["triton_atomic_bitmask", "torch_scatter_add"],
)
def test_kv_tile_csr_metadata_replays(builder):
    """KV-tile query CSR: integer bit masks through ``tl.atomic_add`` and ``scatter_add_``."""
    ids, counts, _, _ = _selection(batch=2, selected=65)
    tile_blocks = cute_qsa_attn.DKV_TILE_BLOCKS

    def fn(ids, counts):
        return builder(ids, counts, NUM_BLOCKS, tile_blocks)

    assert_replays_bit_exact(
        fn, (ids, counts), replays=6, backward=False, contention=True, what=builder.__name__
    )
    _assert_same_fields(
        fn(ids, counts),
        tq.build_qsa_kv_tile_csr_torch(ids, counts, NUM_BLOCKS, tile_blocks),
        ("tile_offsets", "entry_query", "entry_bits"),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_partial_sum_replays(dtype):
    """dk / dv: eight fp32 split partials summed in ascending order and cast."""
    seeded()
    partials = torch.randn(8, 1, CONTENTION_TOKENS, 2, 128, device="cuda")

    outputs, _ = assert_replays_bit_exact(
        lambda partials: cute_qsa_attn._split_sum_cast(partials, dtype),
        (partials,),
        replays=6,
        backward=False,
        contention=True,
        what="_split_sum_cast",
    )
    assert outputs["out"].dtype == dtype
    reference = partials[0].clone()
    for split in partials[1:]:
        reference += split
    assert bytes_equal(outputs["out"], reference.to(dtype))
