# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA static-shape (CUDA-graph capturable) path vs the dynamic per-document path.

The static path computes the packed-THD selection and the sparse-kernel metadata with
device ops at shape-only upper bounds. It must be bit-identical to the dynamic path, and a
QSA layer captured with it must replay bit-identically to eager execution on packings that
differ from the one captured.
"""

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa import (
    build_qsa_supersets,
    build_qsa_supersets_static,
)
from megatron.core.transformer.experimental_attention_variant.qsa_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.transformer.spec_utils import build_module
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    HIDDEN,
    RATIO,
    ROPE_THETA,
    ROTARY_PERCENT,
    SPARSE_HEAD_DIM,
    make_sparse_config,
    requires_qsa_kernels,
)

T = 320
MAX_SEQS = 5
# Packings of T tokens, padded with empty documents to MAX_SEQS. They cover documents that
# end mid-block, documents shorter than one block, empty documents between real ones, and
# documents short enough that fewer than block_topk blocks exist.
PACKINGS = [[T], [120, 200], [3, 101, 0, 216], [64, 64, 64, 64, 64], [1, 2, 317], [250, 70]]


def _cu(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    cu += [T] * (MAX_SEQS + 1 - len(cu))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


@requires_qsa_kernels
class TestQSAStaticShapes:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        # Capture needs a graph-safe RNG tracker (production requires --te-rng-tracker with
        # CUDA graphs); the default tracker's states are plain tensors TE cannot save.
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)
        self.config = make_sparse_config()
        self.config.qsa_indexer_loss_coeff = 0.01
        spec = get_qsa_module_spec_for_backend(self.config)
        self.attn = build_module(spec, config=self.config, layer_number=1).cuda()
        with torch.no_grad():
            for norm in (
                self.attn.q_layernorm,
                self.attn.k_layernorm,
                self.attn.indexer.q_layernorm,
                self.attn.indexer.k_layernorm,
            ):
                norm.weight.uniform_(-0.1, 0.1)
        self.attn.train()
        rope = RotaryEmbedding(
            kv_channels=SPARSE_HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        self.freqs = rope(T)
        g = torch.Generator().manual_seed(5)
        self.hiddens = [torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16() for _ in PACKINGS]

    def teardown_method(self, method):
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    def _select(self, static, hidden, cu):
        self.attn.indexer.static_shapes = static
        with torch.no_grad():
            return self.attn.indexer.select(hidden, self.freqs, cu_seqlens=cu)

    @pytest.mark.parametrize("packing", range(len(PACKINGS)))
    def test_selection_is_bit_identical(self, packing):
        cu = _cu(PACKINGS[packing])
        hidden = self.hiddens[packing]
        dyn = self._select(False, hidden, cu)
        sta = self._select(True, hidden, cu)
        for name in (
            "q",
            "order",
            "picked",
            "own_block",
            "block_offsets",
            "token_offsets",
            "attn_block_offsets",
        ):
            a, b = getattr(dyn, name), getattr(sta, name)
            assert a.dtype == b.dtype and torch.equal(a, b), name
        n_blocks = dyn.block_keys.shape[0]
        assert sta.block_keys.shape[0] == T // RATIO
        assert torch.equal(sta.block_keys[:n_blocks], dyn.block_keys)
        n_attn = dyn.attn_block_bases.shape[0]
        assert sta.attn_block_bases.shape[0] == T // RATIO + MAX_SEQS
        assert torch.equal(sta.attn_block_bases[:n_attn], dyn.attn_block_bases)
        assert torch.equal(sta.attn_block_ends[:n_attn], dyn.attn_block_ends)
        # Unused attention blocks are empty token ranges.
        assert torch.equal(sta.attn_block_bases[n_attn:], sta.attn_block_ends[n_attn:])

    def test_selection_shapes_do_not_depend_on_the_packing(self):
        shapes = set()
        for lengths, hidden in zip(PACKINGS, self.hiddens):
            sel = self._select(True, hidden, _cu(lengths))
            shapes.add(tuple(tuple(t.shape) for t in sel if isinstance(t, torch.Tensor)))
        assert len(shapes) == 1

    def test_static_kernel_metadata_matches_dynamic(self):
        g = torch.Generator(device="cuda").manual_seed(3)
        b, s, S, nb = 1, T, 9, T // RATIO + MAX_SEQS
        block_indices = torch.randint(0, nb - 3, (b, s, S), generator=g, device="cuda")
        block_counts = torch.randint(1, S + 1, (b, s), generator=g, device="cuda")
        bases = (torch.arange(nb, device="cuda") * RATIO).to(torch.int32)
        ends = torch.clamp(bases + RATIO, max=T)
        for bq in (4, 16):
            dyn = build_qsa_supersets(block_indices, block_counts, bases, ends, nb, bq)
            sta = build_qsa_supersets_static(block_indices, block_counts, bases, ends, nb, bq)
            e = dyn.num_entries
            assert dyn.n_tiles == sta.n_tiles
            assert torch.equal(dyn.tile_offsets, sta.tile_offsets)
            assert torch.equal(dyn.entry_base, sta.entry_base[:e])
            assert torch.equal(dyn.entry_end, sta.entry_end[:e])
            assert torch.equal(dyn.membership[:e], sta.membership[:e])

    def _fwd_bwd(self, fn, hidden, cu, grad_out):
        self.attn.zero_grad(set_to_none=True)
        h = hidden.detach().clone().requires_grad_(True)
        out = fn(h, cu)
        out.backward(grad_out)
        grads = {n: p.grad.detach().clone() for n, p in self.attn.named_parameters()}
        return out.detach().clone(), h.grad.detach().clone(), grads

    @staticmethod
    def _assert_matches(ref, got, ref_repeat, what):
        """Exact for everything that is deterministic run to run.

        The indexer's gradient flows through ``torch.gather`` (student scores at the
        selected blocks), whose backward accumulates with atomics, so those parameter
        gradients already differ between two identical eager runs. They are compared
        with a tolerance; every other tensor must be bit-identical.
        """
        assert torch.equal(ref[0], got[0]), f"output differs for {what}"
        assert torch.equal(ref[1], got[1]), f"input grad differs for {what}"
        for name, r in ref[2].items():
            if torch.equal(r, ref_repeat[2][name]):
                assert torch.equal(r, got[2][name]), f"{name} grad differs for {what}"
            else:
                torch.testing.assert_close(got[2][name], r, msg=f"{name} for {what}")

    def _layer_fn(self):
        attn, freqs = self.attn, self.freqs

        class _Fn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = attn

            def forward(self, hidden, cu):
                psp = PackedSeqParams(
                    qkv_format="thd",
                    cu_seqlens_q=cu,
                    cu_seqlens_kv=cu,
                    cu_seqlens_q_padded=cu,
                    cu_seqlens_kv_padded=cu,
                    max_seqlen_q=T,
                    max_seqlen_kv=T,
                    pad_between_seqs=True,
                )
                out, _ = self.attn(
                    hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=psp
                )
                return out

        return _Fn()

    def test_static_layer_forward_backward_is_bit_identical(self):
        """Attention output, input grad and every parameter grad (incl. the indexer's, via
        the sparse-KL loss) match the dynamic path exactly."""
        fn = self._layer_fn()
        g = torch.Generator().manual_seed(9)
        for lengths, hidden in zip(PACKINGS, self.hiddens):
            cu = _cu(lengths)
            grad_out = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            self.attn.indexer.static_shapes = False
            ref = self._fwd_bwd(fn, hidden, cu, grad_out)
            ref_repeat = self._fwd_bwd(fn, hidden, cu, grad_out)
            self.attn.indexer.static_shapes = True
            got = self._fwd_bwd(fn, hidden, cu, grad_out)
            self._assert_matches(ref, got, ref_repeat, lengths)

    def test_capture_replays_new_packings_bit_identically(self):
        from transformer_engine.pytorch import make_graphed_callables

        self.attn.indexer.static_shapes = True
        fn = self._layer_fn()
        g = torch.Generator().manual_seed(9)
        grad_outs = [torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16() for _ in PACKINGS]
        sample = torch.randn(T, 1, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        # Capture before the eager references: eager backward passes on the default stream
        # would leave the parameters' AccumulateGrad nodes bound to it, which capture on TE's
        # side stream rejects.
        graphed = make_graphed_callables(fn, (sample, _cu([T])), num_warmup_iters=2)
        eager = [
            (self._fwd_bwd(fn, h, _cu(lengths), go), self._fwd_bwd(fn, h, _cu(lengths), go))
            for lengths, h, go in zip(PACKINGS, self.hiddens, grad_outs)
        ]
        for lengths, h, go, (ref, ref_repeat) in zip(PACKINGS, self.hiddens, grad_outs, eager):
            got = self._fwd_bwd(graphed, h, _cu(lengths), go)
            self._assert_matches(ref, got, ref_repeat, lengths)
