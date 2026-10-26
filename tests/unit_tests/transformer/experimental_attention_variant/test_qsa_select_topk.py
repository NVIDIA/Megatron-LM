# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA indexer top-k (K4) vs the opt2 chunked int64 ``torch.topk`` merge (test oracle).

``triton`` (fp32 scores + in-register radix select), ``cudnn`` (cuDNN FE radix k-th
value + unique-key tie-break) and the ``auto`` dispatch used by the indexer must
reproduce the opt2 selection bit for bit -- including exact score ties (duplicate block
keys, all-zero relu scores), packed-document windows (``m_lo``) and rows with fewer
visible blocks than ``k`` -- and must be repeatable and CUDA-graph capturable.
"""

import types

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.qsa import QSAIndexer
from tests.unit_tests.transformer.experimental_attention_variant.qsa_topk_oracle import oracle_topk


def _has_cudnn_topk():
    try:
        from cudnn import DSA  # noqa: F401

        return hasattr(DSA, "indexer_top_k_wrapper")
    except Exception:
        return False


IMPLS = ["triton", "auto"] + (["cudnn"] if _has_cudnn_topk() else [])


def _topk(impl, q, bk, m_t, nb, k, m_lo=None):
    if impl == "torch":
        return oracle_topk(q, bk, m_t, nb, k, m_lo=m_lo)
    if impl == "auto":  # the indexer's own path
        indexer = types.SimpleNamespace(head_dim=q.shape[-1])
        return QSAIndexer._streaming_topk(indexer, q, bk, m_t, nb, k, m_lo=m_lo)
    from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
        ID_BITS,
        qsa_select_topk_keys,
    )

    id_code_max = (1 << ID_BITS) - 1
    keys = torch.stack(
        [
            qsa_select_topk_keys(
                q[:, bi].contiguous(), bk[:nb, bi].contiguous(), m_t, k, m_lo=m_lo, impl=impl
            )
            for bi in range(q.shape[1])
        ]
    )
    picked = keys > 0
    order = id_code_max - (keys & id_code_max)
    return torch.where(picked, order, torch.zeros_like(order)), picked


def _case(s, nb, b=1, ties=False, seed=0, packed=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(s, b, 4, 128, generator=g, device="cuda").to(torch.bfloat16)
    bk = torch.randn(nb, b, 128, generator=g, device="cuda").to(torch.bfloat16)
    if ties:
        # exact score ties: every third block duplicates its predecessor, and a band of
        # all-zero keys gives exact-zero relu scores
        dup = torch.arange(1, nb, 3, device="cuda")
        bk[dup] = bk[dup - 1]
        bk[nb // 3 : nb // 3 + 40] = 0
    t = torch.arange(s, device="cuda")
    if packed:
        # three documents sharing one block buffer: visible window [m_lo, m_lo + own)
        cut = torch.tensor([0, s // 5, s // 2, s], device="cuda")
        doc = torch.searchsorted(cut[1:], t, right=True)
        pos = t - cut[doc]
        blk_off = torch.nn.functional.pad(
            (cut[1:] - cut[:-1]).div(4, rounding_mode="floor").cumsum(0), (1, 0)
        )
        m_lo = blk_off[doc]
        return q, bk, m_lo + (pos + 1) // 4, m_lo
    return q, bk, (t + 1) // 4, None


SHAPES = [
    # (s, n_blocks, k, b)
    (600, 150, 32, 1),
    (1100, 275, 64, 2),
    (4096, 1024, 512, 1),
    (2000, 3000, 512, 1),  # n_blocks > s // 4: high ids never visible
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestQSASelectTopK:
    @pytest.mark.parametrize("impl", IMPLS)
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("ties", [False, True])
    def test_bit_identical_to_torch(self, impl, shape, ties):
        s, nb, k, b = shape
        q, bk, m_t, m_lo = _case(s, nb, b=b, ties=ties)
        ref = _topk("torch", q, bk, m_t, nb, k)
        got = _topk(impl, q, bk, m_t, nb, k)
        assert torch.equal(ref[1], got[1]), "picked differs"
        assert torch.equal(ref[0], got[0]), "order differs"

    @pytest.mark.parametrize("impl", IMPLS)
    @pytest.mark.parametrize("ties", [False, True])
    def test_packed_window_bit_identical(self, impl, ties):
        s, k = 3000, 128
        q, bk, m_t, m_lo = _case(s, s // 4, ties=ties, seed=3, packed=True)
        ref = _topk("torch", q, bk, m_t, s // 4, k, m_lo=m_lo)
        got = _topk(impl, q, bk, m_t, s // 4, k, m_lo=m_lo)
        assert torch.equal(ref[1], got[1])
        assert torch.equal(ref[0], got[0])

    @pytest.mark.parametrize("impl", IMPLS)
    def test_repeatable_and_graph_capturable(self, impl):
        s, nb, k = 4096, 1024, 512
        q, bk, m_t, _ = _case(s, nb, ties=True, seed=5)
        eager = _topk(impl, q, bk, m_t, nb, k)  # warm-up compiles before capture
        again = _topk(impl, q, bk, m_t, nb, k)
        assert torch.equal(eager[0], again[0]) and torch.equal(eager[1], again[1])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _topk(impl, q, bk, m_t, nb, k)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = _topk(impl, q, bk, m_t, nb, k)
        q2, bk2, _, _ = _case(s, nb, ties=False, seed=6)
        q.copy_(q2)
        bk.copy_(bk2)
        graph.replay()
        ref = _topk("torch", q, bk, m_t, nb, k)
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
