# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.pp.decode_api import page_view

BLOCK = 256
LENGTHS = (300, 1000, 129)


def _supported_gpu():
    if not torch.cuda.is_available():
        return False
    from megatron.core.transformer.pp.api import HAVE_PP_ATTENTION

    return HAVE_PP_ATTENTION and torch.cuda.get_device_capability()[0] == 10


gpu = pytest.mark.skipif(
    not _supported_gpu(), reason="needs a compute capability 10.x GPU, cutlass-dsl and flash-attn-4"
)


def test_page_view():
    k = torch.zeros(4, BLOCK, 1, 8)
    table = torch.tensor([[3, 1], [0, -1]], dtype=torch.int32)
    k_pages, v_pages, pages = page_view(k, k.clone(), table)
    assert k_pages.shape == (8, 128, 1, 8) and v_pages.shape == k_pages.shape
    assert pages.tolist() == [[6, 7, 2, 3], [0, 1, -1, -1]]
    assert pages.dtype == torch.int32


def test_page_view_rejects_block_size():
    k = torch.zeros(2, 64, 1, 8)
    with pytest.raises(ValueError):
        page_view(k, k, torch.zeros(1, 1, dtype=torch.int32))


class _Case:
    """Random prompts in a shuffled 256-key block cache, plus dense reference outputs."""

    def __init__(self, head_dim, h_q, h_kv, super_block_tiles=8, lengths=LENGTHS):
        self.lengths = lengths
        from megatron.core.transformer.pp.api import get_pp_attention

        torch.manual_seed(0)
        dev = torch.device("cuda")
        self.attn = get_pp_attention(dev, head_dim, h_q, h_kv, super_block_tiles, None)
        blocks = [-(-n // BLOCK) for n in self.lengths]
        pool = sum(blocks) + 2
        perm = torch.randperm(pool, device=dev).to(torch.int32)
        self.table = torch.full((len(self.lengths), max(blocks)), -1, dtype=torch.int32, device=dev)
        shape = (pool, BLOCK, h_kv, head_dim)
        k_cache = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        v_cache = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        self.q, self.ref = [], []
        used = 0
        for b, n in enumerate(self.lengths):
            ids = perm[used : used + blocks[b]]
            used += blocks[b]
            self.table[b, : blocks[b]] = ids
            q = torch.randn(1, n, h_q, head_dim, device=dev, dtype=torch.bfloat16)
            k = k_cache[ids.long()].reshape(1, -1, h_kv, head_dim)[:, :n].contiguous()
            v = v_cache[ids.long()].reshape(1, -1, h_kv, head_dim)[:, :n].contiguous()
            self.q.append(q[0])
            self.ref.append(self.attn.prefill(q, k, v)[0])
        self.k, self.v, self.pages = page_view(k_cache, v_cache, self.table)

    def prefill(self, starts):
        """Prefill rows [start, len) of each prompt; keys [0, len) are in the cache."""
        rows = [self.q[b][s:] for b, s in enumerate(starts)]
        cu = [0]
        for r in rows:
            cu.append(cu[-1] + r.shape[0])
        cu = torch.tensor(cu, dtype=torch.int32, device="cuda")
        seqused = torch.tensor(self.lengths, dtype=torch.int32, device="cuda")
        out = self.attn.prefill_paged(
            torch.cat(rows).contiguous(), self.k, self.v, cu, seqused, self.pages
        )
        return out.split([r.shape[0] for r in rows])

    def decoder(self, verify_tokens=8):
        from megatron.core.transformer.pp.decode_api import get_pp_decoder

        return get_pp_decoder(self.attn, 8, 2048, verify_tokens)

    def tail(self, n):
        q = torch.stack([q[-n:] for q in self.q]).contiguous()
        ref = torch.stack([r[-n:] for r in self.ref])
        return q, ref, torch.tensor(self.lengths, dtype=torch.int32, device="cuda")


def _bits(t):
    return t.contiguous().view(torch.int16)


CASES = [(128, 8, 2), (256, 8, 2), (256, 16, 1), (128, 48, 1)]


@gpu
@pytest.mark.parametrize("head_dim,h_q,h_kv", CASES)
class TestPPAttentionInference:
    def test_prefill_matches_training_forward(self, head_dim, h_q, h_kv):
        case = _Case(head_dim, h_q, h_kv)
        for out, ref in zip(case.prefill([0] * len(LENGTHS)), case.ref):
            assert torch.equal(_bits(out), _bits(ref))

    def test_chunked_prefill(self, head_dim, h_q, h_kv):
        case = _Case(head_dim, h_q, h_kv)
        starts = [n // 2 for n in LENGTHS]
        for b, out in enumerate(case.prefill(starts)):
            assert torch.equal(_bits(out), _bits(case.ref[b][starts[b] :]))

    def test_decode(self, head_dim, h_q, h_kv):
        case = _Case(head_dim, h_q, h_kv)
        q, ref, seqused = case.tail(1)
        out = case.decoder().verify(q, case.k, case.v, case.pages, seqused)
        assert torch.equal(_bits(out), _bits(ref))

    @pytest.mark.parametrize("n", [3, 8])
    def test_verify(self, head_dim, h_q, h_kv, n):
        case = _Case(head_dim, h_q, h_kv)
        q, ref, seqused = case.tail(n)
        out = case.decoder().verify(q, case.k, case.v, case.pages, seqused)
        assert torch.equal(_bits(out), _bits(ref))

    def test_decode_cuda_graph(self, head_dim, h_q, h_kv):
        case = _Case(head_dim, h_q, h_kv)
        q, ref, seqused = case.tail(2)
        decoder = case.decoder()
        out = torch.empty_like(q)
        decoder.verify(q, case.k, case.v, case.pages, seqused, out=out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            decoder.verify(q, case.k, case.v, case.pages, seqused, out=out)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(out), _bits(ref))

    @pytest.mark.parametrize("super_block_tiles", [1, 8])
    def test_poisoned_workspaces(self, head_dim, h_q, h_kv, super_block_tiles):
        """Every workspace byte a kernel reads must be written by that launch."""
        case = _Case(head_dim, h_q, h_kv, super_block_tiles)
        decoder = case.decoder()
        for store in (case.attn.scratch, decoder._o_store, decoder._ml_store):
            store.view(torch.float32).fill_(float("nan"))
        starts = [n // 2 for n in LENGTHS]
        for b, out in enumerate(case.prefill(starts)):
            assert torch.equal(_bits(out), _bits(case.ref[b][starts[b] :]))
        for n in (1, 3):
            case.attn.scratch.view(torch.float32).fill_(float("nan"))
            q, ref, seqused = case.tail(n)
            out = decoder.verify(q, case.k, case.v, case.pages, seqused)
            assert torch.equal(_bits(out), _bits(ref))


@gpu
@pytest.mark.parametrize("super_block_tiles", [1, 8])
def test_prefill_writes_every_row(monkeypatch, super_block_tiles):
    """Paged prefill must write every output row, whatever the fresh buffer holds."""
    case = _Case(128, 4, 2, super_block_tiles, lengths=(300, 129, 517, 64))
    empty_like = torch.empty_like
    monkeypatch.setattr(
        torch, "empty_like", lambda *a, **k: empty_like(*a, **k).fill_(float("nan"))
    )
    for out, ref in zip(case.prefill([0] * 4), case.ref):
        assert torch.equal(_bits(out), _bits(ref))
