# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA sparse attention on static THD batches (full-iteration CUDA graphs).

The static THD contract pads a pack to a fixed token capacity with a dummy tail sequence and
pads cu_seqlens to a fixed entry count. Under attention capture QSA selects with static shapes.
These tests check that the padding changes nothing for the real tokens (attention output,
indexer loss, indexer gradients) and that one captured layer replays correctly on new packs.
"""

import itertools

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams, pad_sequence_for_thd
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.experimental_attention_variant import qsa as qsa_module
from megatron.core.transformer.experimental_attention_variant.qsa_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    HIDDEN,
    ROPE_THETA,
    ROTARY_PERCENT,
    SPARSE_HEAD_DIM,
    make_sparse_config,
    max_rel_err,
    requires_qsa_kernels,
)

TARGET, MAX_SEQS = 512, 48


def _psp(cu, max_seqlen):
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        total_tokens=int(cu[-1]),
    )


def _static_pack(hidden, doc_lens):
    """Pad a packed [T, 1, H] row to the static THD contract (TARGET tokens, MAX_SEQS + 1)."""
    cu = torch.tensor([0, *itertools.accumulate(doc_lens)], dtype=torch.int32, device="cuda")
    tokens = torch.zeros(1, hidden.shape[0], dtype=torch.int64, device="cuda")
    _, _, _, _, params, padding_mask = pad_sequence_for_thd(
        tokens, None, None, None, _psp(cu, max(doc_lens)), target_len=TARGET, max_num_seqs=MAX_SEQS
    )
    tail = torch.randn(TARGET - hidden.shape[0], 1, HIDDEN, device="cuda").to(hidden.dtype)
    return torch.cat([hidden, tail]), params, padding_mask


@requires_qsa_kernels
class TestQSAStaticTHD:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.config = make_sparse_config()
        self.config.qsa_indexer_loss_coeff = 0.01
        self.attn = build_module(
            get_qsa_module_spec_for_backend(self.config), config=self.config, layer_number=1
        ).cuda()
        self.attn.train()
        with torch.no_grad():
            for norm in (
                self.attn.q_layernorm,
                self.attn.k_layernorm,
                self.attn.indexer.q_layernorm,
                self.attn.indexer.k_layernorm,
            ):
                norm.weight.uniform_(-0.1, 0.1)
        self.rope = RotaryEmbedding(
            kv_channels=SPARSE_HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        self.losses = []
        save = qsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker

        def record(loss, **kwargs):
            self.losses.append(loss.detach().clone())
            return save(loss=loss, **kwargs)

        self._save = save
        qsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker = staticmethod(record)

    def teardown_method(self, method):
        qsa_module.DSAIndexerLossLoggingHelper.save_loss_to_tracker = staticmethod(self._save)
        Utils.destroy_model_parallel()

    def _hidden(self, doc_lens, seed):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(sum(doc_lens), 1, HIDDEN, generator=g).cuda().bfloat16()

    def _run(self, hidden, params, static_shapes):
        self.attn.indexer.static_shapes = static_shapes
        self.attn.zero_grad(set_to_none=True)
        h = hidden.detach().clone().requires_grad_()
        out, _ = self.attn(
            h,
            attention_mask=None,
            rotary_pos_emb=self.rope(params.max_seqlen_q),
            packed_seq_params=params,
        )
        out.float().square().mean().backward()
        return out.detach(), h.grad, self.attn.indexer.index_qk_proj.weight.grad.clone()

    @pytest.mark.parametrize("doc_lens", [[120, 180], [64, 33, 200, 7]])
    def test_padding_changes_nothing_for_real_tokens(self, doc_lens):
        hidden = self._hidden(doc_lens, seed=5)
        n = hidden.shape[0]
        cu = torch.tensor([0, *itertools.accumulate(doc_lens)], dtype=torch.int32, device="cuda")
        dyn_out, _, _ = self._run(hidden, _psp(cu, max(doc_lens)), static_shapes=False)
        dyn_loss = self.losses[-1]

        padded, params, padding_mask = _static_pack(hidden, doc_lens)
        assert params.padding_mask is padding_mask and bool(padding_mask[0, n:].all())
        static_out, _, _ = self._run(padded, params, static_shapes=True)
        static_loss = self.losses[-1]
        assert torch.equal(static_out[:n], dyn_out)
        torch.testing.assert_close(static_loss, dyn_loss, rtol=1e-5, atol=0)

        # Without the mask the dummy tail's rows enter the indexer loss.
        params.padding_mask = None
        self._run(padded, params, static_shapes=True)
        assert not torch.allclose(self.losses[-1], dyn_loss, rtol=1e-5, atol=0)

    def test_dynamic_select_with_more_blocks_than_the_longest_document(self):
        """Block rows lost to per-document remainders leave unused tail rows whose rotary
        positions pass the table of the longest document (it used to index out of range)."""
        doc_lens = [7] * 40 + [5]
        hidden = self._hidden(doc_lens, seed=3)
        n = hidden.shape[0]
        cu = torch.tensor([0, *itertools.accumulate(doc_lens)], dtype=torch.int32, device="cuda")
        dyn_out, _, _ = self._run(hidden, _psp(cu, max(doc_lens)), static_shapes=False)
        padded, params, _ = _static_pack(hidden, doc_lens)
        static_out, _, _ = self._run(padded, params, static_shapes=True)
        assert torch.equal(static_out[:n], dyn_out)

    def test_indexer_gradient_ignores_padding(self):
        """Real-token loss only: the indexer gradient matches the unpadded pack."""
        doc_lens = [150, 150]
        hidden = self._hidden(doc_lens, seed=9)
        n = hidden.shape[0]
        cu = torch.tensor([0, 150, 300], dtype=torch.int32, device="cuda")

        def run(h, params, static_shapes):
            self.attn.indexer.static_shapes = static_shapes
            self.attn.zero_grad(set_to_none=True)
            out, _ = self.attn(
                h,
                attention_mask=None,
                rotary_pos_emb=self.rope(params.max_seqlen_q),
                packed_seq_params=params,
            )
            out[:n].float().square().mean().backward()
            return self.attn.indexer.index_qk_proj.weight.grad.clone()

        g_dyn = run(hidden, _psp(cu, 150), static_shapes=False)
        padded, params, _ = _static_pack(hidden, doc_lens)
        g_static = run(padded, params, static_shapes=True)
        assert torch.isfinite(g_static).all() and g_static.abs().sum() > 0
        # allclose: the loss's gather backward is an atomic scatter-add.
        torch.testing.assert_close(g_static, g_dyn)

    def test_captured_layer_replays_new_packs(self):
        """One captured fwd+bwd replays on packs with different documents and boundaries."""
        packs = [[100, 200, 50], [300, 12], [7, 7, 7, 400], [512]]
        statics = [_static_pack(self._hidden(p, seed=20 + i), p) for i, p in enumerate(packs)]
        self.attn.indexer.static_shapes = True
        hidden = statics[0][0].detach().clone().requires_grad_()
        cu = statics[0][1].cu_seqlens_q.clone()
        mask = statics[0][2].clone()
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=TARGET,
            max_seqlen_kv=TARGET,
            pad_between_seqs=True,
            padding_mask=mask,
        )
        freqs = self.rope(TARGET)

        def step():
            out, _ = self.attn(
                hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=params
            )
            out.float().square().mean().backward()
            return out

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self.attn.zero_grad(set_to_none=True)
                hidden.grad = None
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.attn.zero_grad(set_to_none=True)
        hidden.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = step()
        static_dgrad = hidden.grad
        static_wgrad = self.attn.linear_qkv.weight.grad
        static_igrad = self.attn.indexer.index_qk_proj.weight.grad

        for padded, pack_params, pack_mask in statics[1:]:
            with torch.no_grad():
                hidden.copy_(padded)
                cu.copy_(pack_params.cu_seqlens_q)
                mask.copy_(pack_mask)
            del self.losses[:]
            graph.replay()
            torch.cuda.synchronize()
            got = (
                static_out.clone(),
                static_dgrad.clone(),
                static_wgrad.clone(),
                static_igrad.clone(),
            )
            out, dgrad, igrad = self._run(padded, pack_params, static_shapes=True)
            wgrad = self.attn.linear_qkv.weight.grad
            assert torch.equal(got[0], out)
            for name, a, b in (("dgrad", got[1], dgrad), ("wgrad", got[2], wgrad)):
                assert max_rel_err(a, b) < 1e-2, f"{name} {max_rel_err(a, b):.3e}"
            torch.testing.assert_close(got[3], igrad)
        del graph
