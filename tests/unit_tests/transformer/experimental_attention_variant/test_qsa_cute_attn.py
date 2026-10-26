# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA sparse attention (:func:`cute_qsa_attn.qsa_sparse_attention`) vs an fp64 gold reference.

The output and the q / k / v gradients are compared against fp64 dense masked attention on the
same bf16 inputs and selection (relative L2 error within ``ERR_BOUND``, about twice the measured
bf16 error). Also: bitwise repeatability, static vs dynamic metadata builders bit-identical, and
CUDA-graph replay equal to eager. Shapes cover GQA groups that are not powers of two (and
``G = 1``, whose 128-query tiles use the sort-based superset builder), packed THD documents with
ragged and empty blocks, and a context-parallel style query shard (TQ < TK).
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops import triton_qsa as tq
from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_attn import (
    qsa_sparse_attention,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    requires_qsa_kernels,
)

pytestmark = requires_qsa_kernels

BS = tq.BS
# rel-L2 vs fp64 gold; measured on GB200 at the proxy shape: o 2.1e-3, dq 2.4e-3, dk/dv 2.4e-3
ERR_BOUND = 5e-3


def _make_case(doc_lens, hq, h, d, budget_blocks, dtype, seed=0, q_rows=None):
    """Packed documents; every query selects its own block plus a random subset of at
    most ``budget_blocks`` visible complete blocks of its document (global block ids)."""
    g = torch.Generator().manual_seed(seed)
    s = sum(doc_lens)
    bases, ends, rows = [], [], []
    tok0 = blk0 = 0
    for sd in doc_lens:
        nb_d = sd // BS + 1
        b_d = [tok0 + BS * j for j in range(nb_d)]
        bases += b_d
        ends += [min(x + BS, tok0 + sd) for x in b_d]
        for t in range(sd):
            m_t = (t + 1) // BS
            k = min(budget_blocks, m_t)
            picked = torch.randperm(m_t, generator=g)[:k].tolist() if m_t else []
            rows.append([blk0 + m_t] + [blk0 + j for j in picked])
        tok0 += sd
        blk0 += nb_d
    S = 1 + budget_blocks
    idx = torch.zeros(1, s, S, dtype=torch.long)
    counts = torch.zeros(1, s, dtype=torch.long)
    for t, r in enumerate(rows):
        idx[0, t, : len(r)] = torch.tensor(r)
        # garbage beyond the count must be ignored
        idx[0, t, len(r) :] = torch.randint(0, blk0, (S - len(r),), generator=g)
        counts[0, t] = len(r)
    q_pos = torch.arange(s)
    if q_rows is not None:  # CP-style shard: local query rows against the full K/V
        idx, counts, q_pos = idx[:, q_rows], counts[:, q_rows], q_pos[q_rows]
    tq_ = q_pos.numel()
    q = torch.randn(1, tq_, hq, d, generator=g).to(dtype)
    k = torch.randn(1, s, h, d, generator=g).to(dtype)
    v = torch.randn(1, s, h, d, generator=g).to(dtype)
    do = torch.randn(1, tq_, hq, d, generator=g).to(dtype)
    dev = "cuda"
    return dict(
        idx=idx.to(dev),
        counts=counts.to(dev),
        bases=torch.tensor(bases, dtype=torch.int32, device=dev),
        ends=torch.tensor(ends, dtype=torch.int32, device=dev),
        q_pos=q_pos.to(dev),
        q=q.to(dev),
        k=k.to(dev),
        v=v.to(dev),
        do=do.to(dev),
    )


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _run(c, static_shapes=True):
    q, k, v = (c[n].clone().requires_grad_(True) for n in ("q", "k", "v"))
    out = qsa_sparse_attention(
        q,
        k,
        v,
        block_indices=c["idx"],
        block_counts=c["counts"],
        block_bases=c["bases"],
        block_ends=c["ends"],
        q_positions=c["q_pos"],
        static_shapes=static_shapes,
    )
    out.backward(c["do"])
    return out.detach(), q.grad, k.grad, v.grad


def _gold(c):
    """fp64 dense masked attention: (o, dq, dk, dv)."""
    q, k, v = (c[n].double().requires_grad_(True) for n in ("q", "k", "v"))
    do = c["do"].double()
    B, TQ, HQ, D = q.shape
    TK, H = k.shape[1], k.shape[2]
    G = HQ // H
    bases, ends = c["bases"].long(), c["ends"].long()
    NB = bases.numel()
    dev = q.device
    pos = bases[:, None] + torch.arange(BS, device=dev)[None, :]
    valid = pos < ends[:, None]
    tok_blk = torch.full((TK,), NB, dtype=torch.long, device=dev)
    tok_blk[pos[valid]] = torch.arange(NB, device=dev)[:, None].expand(NB, BS)[valid]
    S = c["idx"].shape[-1]
    slot_valid = torch.arange(S, device=dev) < c["counts"].unsqueeze(-1)
    sel = torch.zeros(B, TQ, NB + 1, dtype=torch.bool, device=dev)
    sel.scatter_(2, torch.where(slot_valid, c["idx"], NB), True)
    sel[..., NB] = False
    mask = sel[:, :, tok_blk] & (
        torch.arange(TK, device=dev)[None, None, :] <= c["q_pos"][None, :, None]
    )
    sc = torch.einsum("bthd,bshd->bhts", q, k.repeat_interleave(G, 2)) * D**-0.5
    p = torch.softmax(sc.masked_fill(~mask[:, None], float("-inf")), -1).nan_to_num(0.0)
    o = torch.einsum("bhts,bshd->bthd", p, v.repeat_interleave(G, 2))
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


CASES = {
    # name: (doc_lens, hq, h, d, budget_blocks, q_rows)
    "proxy_like_g12": ([1024], 24, 2, 256, 64, None),
    "g12_two_docs": ([600, 424], 24, 2, 256, 32, None),
    "g3_hd128": ([333], 6, 2, 128, 16, None),
    "g1_hd128": ([300], 2, 2, 128, 16, None),
    "thd_ragged_hd128": ([3, 101, 0, 216, 1, 2, 57], 8, 2, 128, 8, None),
    "cp_shard_hd256": ([512], 24, 2, 256, 16, list(range(0, 128)) + list(range(384, 512))),
}


class TestQSACuteAttention:
    @pytest.mark.parametrize("case", list(CASES))
    def test_matches_gold(self, case):
        doc_lens, hq, h, d, budget, q_rows = CASES[case]
        c = _make_case(doc_lens, hq, h, d, budget, torch.bfloat16, q_rows=q_rows)
        gold = _gold(c)
        got = _run(c)
        for name, x, x_g in zip(("o", "dq", "dk", "dv"), got, gold):
            assert torch.isfinite(x).all(), name
            err = _rel(x, x_g)
            print(f"QSA_ATTN_ERR case={case} {name}={err:.3e}")
            assert err <= ERR_BOUND, f"{name}: rel-L2 {err:.3e} vs fp64 gold"

    @pytest.mark.parametrize("case", ["proxy_like_g12", "g3_hd128", "cp_shard_hd256"])
    def test_bitwise_repeatable(self, case):
        doc_lens, hq, h, d, budget, q_rows = CASES[case]
        c = _make_case(doc_lens, hq, h, d, budget, torch.bfloat16, seed=3, q_rows=q_rows)
        first = _run(c)
        for _ in range(2):
            again = _run(c)
            for a, b in zip(first, again):
                assert torch.equal(a, b)

    @pytest.mark.parametrize("hq,d", [(24, 256), (2, 128)])
    def test_dynamic_and_static_builders_agree(self, hq, d):
        c = _make_case([3, 101, 0, 216, 1, 2, 57], hq, 2, d, 8, torch.bfloat16)
        a = _run(c, static_shapes=False)
        b = _run(c, static_shapes=True)
        for x, y in zip(a, b):
            assert torch.equal(x, y)

    def test_cuda_graph_capture_replays_eager(self):
        c = _make_case([700, 324], 24, 2, 256, 32, torch.bfloat16, seed=7)
        q, k, v = (c[n].clone().requires_grad_(True) for n in ("q", "k", "v"))

        def step():
            for t in (q, k, v):
                t.grad = None
            out = qsa_sparse_attention(
                q,
                k,
                v,
                block_indices=c["idx"],
                block_counts=c["counts"],
                block_bases=c["bases"],
                block_ends=c["ends"],
                q_positions=c["q_pos"],
                static_shapes=True,
            )
            out.backward(c["do"])
            return out.detach(), q.grad, k.grad, v.grad

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            eager = [t.clone() for t in step()]
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res = step()
        graph.replay()
        torch.cuda.synchronize()
        for a, b in zip(res, eager):
            assert torch.equal(a, b)
