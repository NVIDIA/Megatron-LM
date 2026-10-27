# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sort-free sparse-attention metadata (K3) vs the sort-based builders.

The fused (dense bitmask + count + compaction) supersets and KV-tile CSR must equal the
``*_torch`` sort-based builders bit for bit, on the static (fixed-capacity) and the
dynamic (exact-size) paths, and be CUDA-graph capturable.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops import triton_qsa as tq
from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
    qsa_build_kv_tile_csr_fused,
    qsa_build_supersets_fused,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_qsa_cute_attn import (
    _make_case,
)


def _random_case(b, s, S, nb, seed, distinct=True):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if distinct:
        # a real selection: distinct ids per row (own block + top-k), valid prefix
        ids = torch.rand(b, s, nb, generator=g, device="cuda").argsort(-1)[..., :S]
    else:
        ids = torch.randint(0, nb, (b, s, S), generator=g, device="cuda")
    counts = torch.randint(1, S + 1, (b, s), generator=g, device="cuda")
    return ids, counts


def _causal_case(s, S, r=4):
    """Proxy-like selection: own block + the S - 1 most recent visible blocks."""
    t = torch.arange(s, device="cuda")
    m_t = (t + 1) // r
    j = torch.arange(S - 1, device="cuda")
    ids = m_t.view(s, 1) - 1 - j.view(1, -1)
    picked = ids >= 0
    order = torch.where(picked, ids, 0)
    bi = torch.cat([m_t.view(s, 1), order], dim=-1).unsqueeze(0)
    return bi, (1 + picked.sum(-1)).unsqueeze(0), s // r + 1


def _bases(nb, T, r=4):
    bases = (torch.arange(nb, device="cuda") * r).to(torch.int32)
    return bases, torch.clamp(bases + r, max=T)


CASES = [
    # (b, s, S, nb, distinct)
    (1, 320, 9, 85, True),
    (2, 301, 17, 120, True),
    (1, 257, 33, 70, False),  # duplicates inside a row: both builders sum / OR alike
    (1, 1030, 65, 300, True),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestQSAFusedMetadata:
    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("bq", [4, 16])
    def test_supersets_match_sort_builders(self, case, bq):
        b, s, S, nb, distinct = case
        ids, counts = _random_case(b, s, S, nb, seed=s + S, distinct=distinct)
        bases, ends = _bases(nb, s)
        ref = tq.build_qsa_supersets_static_torch(ids, counts, bases, ends, nb, bq)
        to, eb, ee, mem, n_tiles, e_cap = qsa_build_supersets_fused(
            ids, counts, bases, ends, nb, bq
        )
        assert n_tiles == ref.n_tiles and e_cap == ref.num_entries
        assert torch.equal(to, ref.tile_offsets)
        assert torch.equal(eb, ref.entry_base)
        assert torch.equal(ee, ref.entry_end)
        assert torch.equal(mem, ref.membership)
        dref = tq.build_qsa_supersets_torch(ids, counts, bases, ends, nb, bq)
        dgot = tq.build_qsa_supersets(ids, counts, bases, ends, nb, bq)
        assert dgot.num_entries == dref.num_entries
        for name in ("tile_offsets", "entry_base", "entry_end", "membership"):
            a, c = getattr(dref, name), getattr(dgot, name)
            assert a.dtype == c.dtype and torch.equal(a, c), name

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("tile_blocks", [16, 4, 1])
    def test_kv_tile_csr_matches_sort_builder(self, case, tile_blocks):
        b, s, S, nb, distinct = case
        ids, counts = _random_case(b, s, S, nb, seed=3 * s + S, distinct=distinct)
        ref = tq.build_qsa_kv_tile_csr_torch(ids, counts, nb, tile_blocks)
        to, eq, eb, nt = qsa_build_kv_tile_csr_fused(ids, counts, nb, tile_blocks)
        got = tq.build_qsa_kv_tile_csr(ids, counts, nb, tile_blocks)
        assert nt == ref.n_tiles == got.n_tiles
        for a, c in ((ref.tile_offsets, to), (ref.entry_query, eq), (ref.entry_bits, eb)):
            assert a.dtype == c.dtype and a.shape == c.shape and torch.equal(a, c)
        assert torch.equal(got.entry_query, ref.entry_query)

    @pytest.mark.parametrize("s", [4096])
    def test_proxy_causal_selection(self, s):
        ids, counts, nb = _causal_case(s, 513)
        bases, ends = _bases(nb, s)
        ref = tq.build_qsa_supersets_static_torch(ids, counts, bases, ends, nb, 4)
        got = tq.build_qsa_supersets_static(ids, counts, bases, ends, nb, 4)
        for name in ("tile_offsets", "entry_base", "entry_end", "membership"):
            assert torch.equal(getattr(ref, name), getattr(got, name)), name
        ref = tq.build_qsa_kv_tile_csr_torch(ids, counts, nb, 16)
        got = tq.build_qsa_kv_tile_csr(ids, counts, nb, 16)
        for name in ("tile_offsets", "entry_query", "entry_bits"):
            assert torch.equal(getattr(ref, name), getattr(got, name)), name

    def test_graph_capture(self):
        b, s, S, nb = 1, 640, 33, 200
        ids, counts = _random_case(b, s, S, nb, seed=1)
        bases, ends = _bases(nb, s)
        tq.build_qsa_supersets_static(ids, counts, bases, ends, nb, 4)  # compile
        tq.build_qsa_kv_tile_csr(ids, counts, nb, 16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ss = tq.build_qsa_supersets_static(ids, counts, bases, ends, nb, 4)
            kv = tq.build_qsa_kv_tile_csr(ids, counts, nb, 16)
        ids2, counts2 = _random_case(b, s, S, nb, seed=2)
        ids.copy_(ids2)
        counts.copy_(counts2)
        graph.replay()
        ref = tq.build_qsa_supersets_static_torch(ids, counts, bases, ends, nb, 4)
        for name in ("tile_offsets", "entry_base", "entry_end", "membership"):
            assert torch.equal(getattr(ref, name), getattr(ss, name)), name
        ref = tq.build_qsa_kv_tile_csr_torch(ids, counts, nb, 16)
        for name in ("tile_offsets", "entry_query", "entry_bits"):
            assert torch.equal(getattr(ref, name), getattr(kv, name)), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestQSAKVTileCSR:
    def test_tile_csr_matches_brute_force(self):
        c = _make_case([3, 101, 0, 216, 1, 2, 57], 4, 2, 32, 8, torch.bfloat16, seed=5)
        nb = c["bases"].numel()
        idx, counts = c["idx"].cpu(), c["counts"].cpu()
        for kb in (1, 4, 5, 16):
            csr = tq.build_qsa_kv_tile_csr(c["idx"], c["counts"], nb, kb)
            nt = (nb + kb - 1) // kb
            assert csr.n_tiles == nt
            want = [dict() for _ in range(nt)]
            for t in range(idx.shape[1]):
                for j in idx[0, t, : counts[0, t]].tolist():
                    want[j // kb][t] = want[j // kb].get(t, 0) | (1 << (j % kb))
            off = csr.tile_offsets.cpu()
            eq, eb = csr.entry_query.cpu(), csr.entry_bits.cpu()
            for i in range(nt):
                got_q = eq[off[i] : off[i + 1]].tolist()
                assert got_q == sorted(want[i]), f"kb={kb} tile {i} queries"
                assert eb[off[i] : off[i + 1]].tolist() == [want[i][t] for t in got_q]
