# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA on cuDNN Frontend (``qsa_attn_backend='cudnn'``, :mod:`ops.cudnn_qsa_attn`).

* The selection translations, exactly and on CPU: the Miles ``[T, NB]`` bitmap metadata
  admits exactly the keys of the gold mask, and the ``sparse_gqa`` index lists list exactly
  those keys in the kernel's layout.
* Build-time validation: unsupported geometries and combinations fail with their cause.
* On a Blackwell GPU with a cuDNN Frontend that has the QSA training kernels: the output and
  the q / k / v gradients of both backward implementations against the fp64 gold reference
  of ``test_qsa_cute_attn.py`` and against the in-tree CuTe DSL kernels, and the
  ``QSASelfAttention`` module against the dense-mask bridge.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.ops.cudnn_qsa_attn import (
    SPARSE_GQA_WIDTH,
    build_miles_metadata,
    build_sparse_gqa_index_lists,
    cudnn_qsa_support_error,
    sequence_spans,
)
from megatron.core.transformer.transformer_config import TransformerConfig

BS = 4


def _cudnn_qsa_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        return False
    try:
        import cudnn.ops as ops
        from cudnn.sdpa.bwd import sparse_gqa_bwd_wrapper_sm100  # noqa: F401

        ops.qsa_forward, ops.qsa_backward  # pylint: disable=pointless-statement
    except (ImportError, AttributeError):
        return False
    return True


requires_cudnn_qsa = pytest.mark.skipif(
    not _cudnn_qsa_available(),
    reason="needs an SM100/SM103 GPU and a cuDNN Frontend with the QSA training kernels",
)


def _selection(doc_lens, budget_blocks, seed=0, batch=1, device="cpu"):
    """Packed documents (or ``batch`` BSHD rows of one document) and a random selection in the
    QSA kernel contract: slot 0 the own block, then at most ``budget_blocks`` visible complete
    blocks of the document (as many as there are when fewer, like the indexer's top-k)."""
    g = torch.Generator().manual_seed(seed)
    s = sum(doc_lens)
    bases, ends, blk_off, tok_off = [], [], [], []
    tok0 = blk0 = 0
    for sd in doc_lens:
        nb_d = sd // BS + 1
        b_d = [tok0 + BS * j for j in range(nb_d)]
        bases += b_d
        ends += [min(x + BS, tok0 + sd) for x in b_d]
        blk_off += [blk0] * sd
        tok_off += [tok0] * sd
        tok0 += sd
        blk0 += nb_d
    K = 1 + budget_blocks
    idx = torch.zeros(batch, s, K, dtype=torch.long)
    counts = torch.zeros(batch, s, dtype=torch.long)
    for b in range(batch):
        for t in range(s):
            tl = t - tok_off[t]
            m_t = (tl + 1) // BS
            k = min(budget_blocks, m_t)
            picked = torch.randperm(m_t, generator=g)[:k].tolist() if m_t else []
            row = [blk_off[t] + m_t] + [blk_off[t] + j for j in picked]
            idx[b, t, : len(row)] = torch.tensor(row)
            idx[b, t, len(row) :] = torch.randint(0, blk0, (K - len(row),), generator=g)
            counts[b, t] = len(row)
    dev = torch.device(device)
    return dict(
        idx=idx.to(dev),
        counts=counts.to(dev),
        bases=torch.tensor(bases, dtype=torch.int32, device=dev),
        ends=torch.tensor(ends, dtype=torch.int32, device=dev),
        blk_off=torch.tensor(blk_off, dtype=torch.long, device=dev),
        tok_off=None if batch > 1 or len(doc_lens) == 1 else torch.tensor(tok_off, device=dev),
        doc_lens=doc_lens,
        batch=batch,
    )


def _gold_mask(c):
    """[B, S, S] bool: key j visible to query t (selected block, causal)."""
    idx, counts, bases, ends = c["idx"], c["counts"], c["bases"].long(), c["ends"].long()
    B, S, K = idx.shape
    NB = bases.numel()
    pos = bases[:, None] + torch.arange(BS)[None, :]
    valid = pos < ends[:, None]
    tok_blk = torch.full((S,), NB, dtype=torch.long)
    tok_blk[pos[valid]] = torch.arange(NB)[:, None].expand(NB, BS)[valid]
    slot_valid = torch.arange(K) < counts.unsqueeze(-1)
    sel = torch.zeros(B, S, NB + 1, dtype=torch.bool)
    sel.scatter_(2, torch.where(slot_valid, idx, NB), True)
    sel[..., NB] = False
    return sel[:, :, tok_blk] & (torch.arange(S)[None, None, :] <= torch.arange(S)[None, :, None])


def _miles_mask(meta, T):
    """[T, T] bool from the Miles lookup of the cuDNN QSA API documentation."""
    j = torch.arange(T)[None, :]
    lo, hi = meta.lo.long()[:, None], meta.hi.long()[:, None]
    b = meta.blk_base.long()[:, None] + torch.div(
        j - meta.tok_base.long()[:, None], BS, rounding_mode="floor"
    )
    NB = meta.sel_blocks.shape[1]
    inb = (b >= 0) & (b < NB)
    picked = torch.gather(meta.sel_blocks.long(), 1, b.clamp(0, NB - 1)) != 0
    return (lo <= j) & (j <= hi) & inb & picked


CPU_CASES = {
    "one_doc": dict(doc_lens=[301], budget_blocks=16),
    "thd_ragged": dict(doc_lens=[3, 101, 1, 216, 2, 57], budget_blocks=8),
    "bshd_batch2": dict(doc_lens=[150], budget_blocks=8, batch=2),
}


class TestSelectionTranslation:
    @pytest.mark.parametrize("case", list(CPU_CASES))
    def test_miles_metadata_admits_exactly_the_selected_keys(self, case):
        c = _selection(**CPU_CASES[case])
        meta = build_miles_metadata(
            c["idx"], c["counts"], c["bases"], c["ends"], c["blk_off"], c["tok_off"]
        )
        B, S = c["counts"].shape
        T = B * S
        gold = torch.block_diag(*_gold_mask(c).unbind(0))
        assert torch.equal(_miles_mask(meta, T), gold)
        # Miles spans: every block 1..4 tokens, inside its row, and never an empty selection.
        assert meta.sel_blocks.dtype == torch.uint8 and meta.sel_blocks.shape[0] == T
        sel_any = meta.sel_blocks.any(0).nonzero().flatten()
        span = meta.block_last.long()[sel_any] - meta.block_first.long()[sel_any] + 1
        assert ((span >= 1) & (span <= BS)).all()
        for t in (meta.lo, meta.hi, meta.blk_base, meta.tok_base, meta.block_first):
            assert t.dtype == torch.int32 and t.is_contiguous()

    @pytest.mark.parametrize("case", list(CPU_CASES))
    def test_sparse_gqa_index_lists_list_exactly_the_selected_keys(self, case):
        c = _selection(**CPU_CASES[case])
        lists = build_sparse_gqa_index_lists(c["idx"], c["counts"], c["blk_off"], c["tok_off"])
        B, S = c["counts"].shape
        assert lists.shape == (B * S, SPARSE_GQA_WIDTH) and lists.dtype == torch.int32
        gold = _gold_mask(c)
        cu = None if c["tok_off"] is None else torch.tensor([0, *_cum(c["doc_lens"])])
        for s0, n in sequence_spans(B, S, cu):
            b, o = divmod(s0, S)
            for t in range(n):
                row = lists[s0 + t].long()
                valid = row >= 0
                nvalid = int(valid.sum())
                # The kernel's layout: block tokens, then the ragged tail, then only -1.
                assert valid[:nvalid].all() and not valid[nvalid:].any()
                ntail = (t + 1) % BS
                tail = row[nvalid - ntail : nvalid]
                assert torch.equal(tail, (t + 1) // BS * BS + torch.arange(ntail))
                # Sequence-local token ids, exactly the visible keys of the gold mask.
                keys = torch.zeros(n, dtype=torch.bool)
                keys[row[valid]] = True
                assert torch.equal(keys, gold[b, o + t, o : o + n]), (case, s0, t)


def _cum(xs):
    total = 0
    for x in xs:
        total += x
        yield total


class TestSupportErrors:
    OK = dict(
        num_q_heads=24,
        num_kv_heads=2,
        head_dim=256,
        dtype=torch.bfloat16,
        block_size=4,
        num_selected_blocks=512,
        bwd_impl="qsa_api",
        device_capability=(10, 3),
    )

    @pytest.mark.parametrize("bwd_impl", ["qsa_api", "sparse_gqa"])
    @pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
    def test_proxy_geometry_is_served(self, bwd_impl, cc):
        assert (
            cudnn_qsa_support_error(**{**self.OK, "bwd_impl": bwd_impl, "device_capability": cc})
            is None
        )

    @pytest.mark.parametrize(
        "change, match",
        [
            (dict(device_capability=(9, 0)), "SM100, SM103 or SM107"),
            (dict(device_capability=None), "no CUDA device"),
            (dict(dtype=torch.float16), "bf16 only"),
            (dict(head_dim=128), "head dim 256"),
            (dict(block_size=8), "4-token blocks"),
            (dict(num_q_heads=16, num_kv_heads=2), "24 query / 2 KV"),
            (dict(num_q_heads=12, num_kv_heads=1, bwd_impl="sparse_gqa"), "even number of KV"),
            (dict(num_selected_blocks=256, bwd_impl="sparse_gqa"), "exactly 512 blocks"),
            (dict(bwd_impl="dense"), "qsa_cudnn_bwd_impl must be"),
        ],
    )
    def test_unsupported_is_named(self, change, match):
        reason = cudnn_qsa_support_error(**{**self.OK, **change})
        assert reason is not None and match in reason

    def test_tp2_geometry_is_served_by_qsa_api(self):
        assert cudnn_qsa_support_error(**{**self.OK, "num_q_heads": 12, "num_kv_heads": 1}) is None


def _qsa_config(**overrides):
    kwargs = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=24,
        num_query_groups=2,
        kv_channels=256,
        bf16=True,
        params_dtype=torch.bfloat16,
        experimental_attention_variant="qsa",
        qsa_indexer_budget=2048,
        qsa_indexer_compress_ratio=4,
        attention_output_gate=True,
        qk_layernorm=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        # The module test compares two training-mode calls; dropout would make them differ.
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


class TestConfig:
    def test_default_follows_legacy_flag(self):
        assert _qsa_config().qsa_attn_backend is None
        assert not _qsa_config().qsa_use_sparse_attention
        assert _qsa_config(qsa_use_sparse_attention=True).qsa_use_sparse_attention

    @pytest.mark.parametrize("backend, sparse", [("dense", False), ("cute", True), ("cudnn", True)])
    def test_backend_sets_the_legacy_flag(self, backend, sparse):
        assert _qsa_config(qsa_attn_backend=backend).qsa_use_sparse_attention is sparse

    def test_default_bwd_impl_is_sparse_gqa(self):
        assert _qsa_config(qsa_attn_backend="cudnn").qsa_cudnn_bwd_impl == "sparse_gqa"

    def test_dense_contradicting_legacy_flag_is_rejected(self):
        with pytest.raises(ValueError, match="contradicts"):
            _qsa_config(qsa_attn_backend="dense", qsa_use_sparse_attention=True)

    def test_unknown_bwd_impl_is_rejected(self):
        with pytest.raises(ValueError, match="qsa_cudnn_bwd_impl"):
            _qsa_config(qsa_attn_backend="cudnn", qsa_cudnn_bwd_impl="fa4")

    def test_cudnn_rejects_context_parallelism(self):
        with pytest.raises(ValueError, match="does not support context parallelism"):
            _qsa_config(qsa_attn_backend="cudnn", context_parallel_size=2)

    def test_sparse_gqa_rejects_thd_cuda_graph_capture(self):
        with pytest.raises(ValueError, match="cannot be captured"):
            _qsa_config(
                qsa_attn_backend="cudnn",
                qsa_cudnn_bwd_impl="sparse_gqa",
                cuda_graph_impl="transformer_engine",
                cuda_graph_modules=["attn"],
                thd_max_packed_sequences=4,
            )


# ---------------------------------------------------------------------------------------------
# GPU: kernels vs the fp64 gold and vs the CuTe DSL kernels.

GPU_CASES = {
    # name: (doc_lens, batch, hq, h, budget_blocks)
    "proxy_like_g12": ([1024], 1, 24, 2, 64),
    "g12_two_docs": ([600, 424], 1, 24, 2, 32),
    "thd_ragged": ([3, 101, 1, 216, 2, 57], 1, 24, 2, 8),
    "bshd_batch2": ([300], 2, 24, 2, 16),
    "tp2_12q_1kv": ([700], 1, 12, 1, 32),
}
# sparse_gqa needs exactly 512 selected blocks; documents past 2048 tokens make it sparse.
SPARSE_GQA_CASES = {
    "one_doc_sparse": ([2600], 1),
    "two_docs": ([2051, 1030], 1),
    "bshd_batch2": ([2300], 2),
}
# rel-L2 vs the fp64 gold (bf16 inputs); the CuTe DSL kernels measure ~2.4e-3 at these shapes.
ERR_BOUND = 5e-3


def _gpu_case(doc_lens, batch, hq, h, budget, seed=0):
    c = _selection(doc_lens, budget, seed=seed, batch=batch, device="cuda")
    g = torch.Generator().manual_seed(seed + 1)
    s = sum(doc_lens)
    for name, heads in (("q", hq), ("k", h), ("v", h), ("do", hq)):
        c[name] = torch.randn(batch, s, heads, 256, generator=g).bfloat16().cuda()
    c["q_pos"] = torch.arange(s, device="cuda")
    c["cu"] = (
        torch.tensor([0] + list(_cum(doc_lens)), dtype=torch.int32, device="cuda")
        if len(doc_lens) > 1
        else None
    )
    return c


def _run_cudnn(c, bwd_impl):
    from megatron.core.transformer.experimental_attention_variant.ops.cudnn_qsa_attn import (
        cudnn_qsa_sparse_attention,
    )

    q, k, v = (c[n].clone().requires_grad_(True) for n in ("q", "k", "v"))
    out = cudnn_qsa_sparse_attention(
        q,
        k,
        v,
        block_indices=c["idx"],
        block_counts=c["counts"],
        block_bases=c["bases"],
        block_ends=c["ends"],
        block_offsets=c["blk_off"],
        token_offsets=c["tok_off"],
        cu_seqlens=c["cu"],
        bwd_impl=bwd_impl,
    )
    out.backward(c["do"])
    return out.detach(), q.grad, k.grad, v.grad


def _run_cute(c):
    from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_attn import (
        qsa_sparse_attention,
    )

    q, k, v = (c[n].clone().requires_grad_(True) for n in ("q", "k", "v"))
    out = qsa_sparse_attention(
        q,
        k,
        v,
        block_indices=c["idx"],
        block_counts=c["counts"],
        block_bases=c["bases"],
        block_ends=c["ends"],
    )
    out.backward(c["do"])
    return out.detach(), q.grad, k.grad, v.grad


def _gold(c):
    from tests.unit_tests.transformer.experimental_attention_variant.test_qsa_cute_attn import (
        _gold as gold_single_row,
    )

    # The gold oracle is one row at a time (B = 1); BSHD rows are independent sequences.
    outs = []
    for b in range(c["batch"]):
        row = {n: c[n][b : b + 1] for n in ("q", "k", "v", "do", "idx", "counts")}
        row.update(bases=c["bases"], ends=c["ends"], q_pos=c["q_pos"])
        outs.append(gold_single_row(row))
    return tuple(torch.cat(parts, dim=0) for parts in zip(*outs))


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@requires_cudnn_qsa
class TestCudnnQSAKernels:
    @pytest.mark.parametrize("case", list(GPU_CASES))
    def test_qsa_api_matches_gold_and_cute(self, case):
        c = _gpu_case(*GPU_CASES[case])
        gold = _gold(c)
        got = _run_cudnn(c, "qsa_api")
        cute = _run_cute(c)
        for name, x, x_g, x_c in zip(("o", "dq", "dk", "dv"), got, gold, cute):
            assert torch.isfinite(x).all(), name
            err, err_cute = _rel(x, x_g), _rel(x, x_c)
            print(
                f"QSA_CUDNN_ERR impl=qsa_api case={case} {name}: gold={err:.3e} cute={err_cute:.3e}"
            )
            assert err <= ERR_BOUND, f"{name}: rel-L2 {err:.3e} vs fp64 gold"
            assert err_cute <= 2 * ERR_BOUND, f"{name}: rel-L2 {err_cute:.3e} vs CuTe DSL"

    @pytest.mark.parametrize("case", list(SPARSE_GQA_CASES))
    def test_sparse_gqa_matches_gold_and_qsa_api(self, case):
        doc_lens, batch = SPARSE_GQA_CASES[case]
        c = _gpu_case(doc_lens, batch, 24, 2, 512, seed=5)
        gold = _gold(c)
        got = _run_cudnn(c, "sparse_gqa")
        api = _run_cudnn(c, "qsa_api")
        for name, x, x_g, x_a in zip(("o", "dq", "dk", "dv"), got, gold, api):
            assert torch.isfinite(x).all(), name
            err, err_api = _rel(x, x_g), _rel(x, x_a)
            print(
                f"QSA_CUDNN_ERR impl=sparse_gqa case={case} {name}: gold={err:.3e} qsa_api={err_api:.3e}"
            )
            assert err <= ERR_BOUND, f"{name}: rel-L2 {err:.3e} vs fp64 gold"
        # Same forward kernel for both backward implementations.
        assert torch.equal(got[0], api[0])


@requires_cudnn_qsa
class TestCudnnQSAModule:
    """``QSASelfAttention`` with ``qsa_attn_backend='cudnn'`` vs the dense-mask bridge."""

    # Past 2048 tokens so the production budget (2048 -> 512 blocks, the only one sparse_gqa
    # serves) actually drops blocks.
    SEQ = 2600
    # bwd_impl -> indexer budget: qsa_api at a reduced budget, sparse_gqa at its 512 blocks.
    BUDGETS = {"qsa_api": 256, "sparse_gqa": 2048}

    def setup_method(self, method):
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from tests.unit_tests.test_utilities import Utils

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.rope = RotaryEmbedding(kv_channels=256, rotary_percent=0.25, rotary_base=1e7)
        g = torch.Generator().manual_seed(7)
        self.hidden = torch.randn(self.SEQ, 1, 256, generator=g).cuda().bfloat16()

    def _build(self, bwd_impl):
        from megatron.core.transformer.experimental_attention_variant.qsa_module_specs import (
            get_qsa_module_spec_for_backend,
        )
        from megatron.core.transformer.spec_utils import build_module

        self.config = _qsa_config(
            qsa_attn_backend="cudnn",
            qsa_cudnn_bwd_impl=bwd_impl,
            qsa_indexer_budget=self.BUDGETS[bwd_impl],
            layernorm_epsilon=1e-6,
        )
        self.attn = build_module(
            get_qsa_module_spec_for_backend(self.config), config=self.config, layer_number=1
        ).cuda()

    def teardown_method(self, method):
        from tests.unit_tests.test_utilities import Utils

        Utils.destroy_model_parallel()

    def _fwd_bwd(self, backend):
        self.attn.zero_grad(set_to_none=True)
        self.config.qsa_attn_backend = backend
        self.config.qsa_use_sparse_attention = backend != "dense"
        h = self.hidden.detach().clone().requires_grad_()
        out, _ = self.attn(h, attention_mask=None, rotary_pos_emb=self.rope(self.SEQ))
        out.float().square().mean().backward()
        return out.detach(), h.grad, self.attn.linear_qkv.weight.grad.clone()

    @pytest.mark.parametrize("bwd_impl", ["qsa_api", "sparse_gqa"])
    def test_matches_bridge(self, bwd_impl):
        self._build(bwd_impl)
        bridge = self._fwd_bwd("dense")
        cudnn = self._fwd_bwd("cudnn")
        for name, a, b in zip(("out", "dgrad", "wgrad"), cudnn, bridge):
            err = ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()
            print(f"QSA_CUDNN_MODULE_ERR impl={bwd_impl} {name}={err:.3e}")
            assert err < 2e-2, f"{name}: cudnn vs bridge max rel err {err:.3e}"
