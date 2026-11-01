# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""``qsa_dense_when_fits``: QSA as plain dense causal attention when every document fits.

A document of at most ``qsa_indexer_budget`` tokens selects all of its visible blocks, so
the sparse path computes dense causal attention within the document. The bypass skips the
indexer and runs TransformerEngine attention instead; it must match the sparse path
(different kernels, bf16 tolerance) in outputs and gradients, reject documents beyond the
budget, refuse configs that would silently drop the indexer loss, and be CUDA-graph
capturable.
"""

import dataclasses

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant.qsa_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    BUDGET,
    HIDDEN,
    ROPE_THETA,
    ROTARY_PERCENT,
    SPARSE_HEAD_DIM,
    make_config,
    make_sparse_config,
    max_rel_err,
    requires_qsa_kernels,
)

# Documents within BUDGET (64): the sparse selection keeps every visible block.
DOC_LENS = [40, 64, 25, 64]
TOL = 2e-2  # bf16, different attention kernels


def _psp(doc_lens):
    cu = torch.tensor([0] + list(torch.tensor(doc_lens).cumsum(0)), dtype=torch.int32).cuda()
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu,
        cu_seqlens_kv_padded=cu,
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )


def test_config_rejects_indexer_loss():
    cfg = make_config()
    dataclasses.replace(cfg, qsa_dense_when_fits=True)  # accepted without the loss
    with pytest.raises(ValueError, match="qsa_indexer_loss_coeff"):
        dataclasses.replace(cfg, qsa_dense_when_fits=True, qsa_indexer_loss_coeff=0.01)


@requires_qsa_kernels
class TestQSADenseWhenFits:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = make_sparse_config()
        spec = get_qsa_module_spec_for_backend(self.config)
        self.attn = build_module(spec, config=self.config, layer_number=1).cuda()
        with torch.no_grad():
            for norm in (self.attn.q_layernorm, self.attn.k_layernorm):
                norm.weight.uniform_(-0.1, 0.1)
        self.rope = RotaryEmbedding(
            kv_channels=SPARSE_HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _run(self, dense, h, psp=None, max_len=None, grad=True):
        self.config.qsa_dense_when_fits = dense
        self.attn.zero_grad(set_to_none=True)
        x = h.detach().clone().requires_grad_(grad)
        freqs = self.rope(max_len if max_len is not None else h.shape[0])
        out, _ = self.attn(x, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=psp)
        if not grad:
            return out.detach(), None, None
        out.float().square().mean().backward()
        wgrads = {
            n: p.grad.clone()
            for n, p in self.attn.named_parameters()
            if p.grad is not None and not n.startswith("indexer.")
        }
        return out.detach(), x.grad.clone(), wgrads

    def test_thd_matches_sparse(self):
        g = torch.Generator().manual_seed(17)
        h = torch.randn(sum(DOC_LENS), 1, HIDDEN, generator=g).cuda().bfloat16()
        psp = _psp(DOC_LENS)
        ref_out, ref_dx, ref_w = self._run(False, h, psp, max(DOC_LENS))
        out, dx, w = self._run(True, h, psp, max(DOC_LENS))
        assert max_rel_err(out, ref_out) < TOL
        assert max_rel_err(dx, ref_dx) < TOL
        assert set(w) == set(ref_w) and w
        for name in w:
            assert max_rel_err(w[name], ref_w[name]) < TOL, name
        # The bypass never touches the indexer.
        assert all(p.grad is None for p in self.attn.indexer.parameters())

    def test_bshd_matches_sparse(self):
        g = torch.Generator().manual_seed(19)
        h = torch.randn(BUDGET, 2, HIDDEN, generator=g).cuda().bfloat16()
        ref_out, ref_dx, _ = self._run(False, h)
        out, dx, _ = self._run(True, h)
        assert max_rel_err(out, ref_out) < TOL
        assert max_rel_err(dx, ref_dx) < TOL

    def test_document_beyond_budget_is_rejected(self):
        g = torch.Generator().manual_seed(23)
        lens = [30, BUDGET + 8]
        h = torch.randn(sum(lens), 1, HIDDEN, generator=g).cuda().bfloat16()
        with pytest.raises(ValueError, match="exceeds qsa_indexer_budget"):
            self._run(True, h, _psp(lens), max(lens), grad=False)

    def test_padding_rows_do_not_count(self):
        """A long dummy tail of padding rows (static THD) passes the fit check."""
        g = torch.Generator().manual_seed(29)
        lens = [40, 3 * BUDGET]
        h = torch.randn(sum(lens), 1, HIDDEN, generator=g).cuda().bfloat16()
        psp = _psp(lens)
        psp.padding_mask = torch.zeros(1, sum(lens), dtype=torch.bool, device="cuda")
        psp.padding_mask[:, lens[0] :] = True
        self._run(True, h, psp, max(lens), grad=False)

    def test_cuda_graph_replay_matches_eager(self):
        g = torch.Generator().manual_seed(31)
        h = torch.randn(sum(DOC_LENS), 1, HIDDEN, generator=g).cuda().bfloat16()
        psp = _psp(DOC_LENS)
        self.config.qsa_dense_when_fits = True
        freqs = self.rope(max(DOC_LENS))
        static_h = h.clone()
        with torch.no_grad():
            eager, _ = self.attn(static_h, None, rotary_pos_emb=freqs, packed_seq_params=psp)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.attn(static_h, None, rotary_pos_emb=freqs, packed_seq_params=psp)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out, _ = self.attn(
                    static_h, None, rotary_pos_emb=freqs, packed_seq_params=psp
                )
            static_h.copy_(h)
            graph.replay()
        torch.testing.assert_close(static_out, eager, rtol=0, atol=0)
