# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA allgather-CP selection without host synchronization (K10).

* The device-only CP layout builder equals the DSA packed-CP builder.
* CP2 selection (order, picked, block keys, offsets) is bit-identical to the single-rank
  path at the local rows, on packings with several / empty documents, and runs with no
  synchronizing torch op.
* The CP2 sparse-KL indexer loss and its indexer gradients match the single-rank loss.
* A CP2 QSA layer captured into a CUDA graph replays new packings bit-identically to
  eager execution.
"""

import gc

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_query_positions_and_key_reorder,
)
from megatron.core.transformer.experimental_attention_variant.ops.triton_qsa_select import (
    qsa_cp_layout_static,
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
CP = 2
MAX_SEQS = 5
# per_document packings: every document length is a multiple of 2 * CP. They cover several
# documents, empty documents anywhere, documents shorter than one block per rank, and
# documents with fewer than block_topk blocks.
PACKINGS_DOC = [[T], [120, 200], [8, 104, 0, 208], [64, 64, 64, 64, 64], [4, 316], [0, 252, 68]]
# per_sequence packings: only the whole pack is a multiple of 2 * CP.
PACKINGS_SEQ = [[T], [90, 230], [3, 101, 0, 216], [1, 2, 317]]


def _cu(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    cu += [T] * (MAX_SEQS + 1 - len(cu))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("layout", ["per_document", "per_sequence"])
def test_static_cp_layout_matches_dsa_builder(layout):
    packings = PACKINGS_DOC if layout == "per_document" else PACKINGS_SEQ
    s_loc = T // CP
    for lengths in packings:
        cu = _cu(lengths)
        for rank in range(CP):
            ref_q, ref_r = build_packed_allgather_cp_query_positions_and_key_reorder(
                cu,
                cu,
                CP,
                rank,
                cu.device,
                local_output_size=s_loc,
                query_cu_seqlens_cover_output=True,
                key_cu_seqlens_cover_output=True,
                cp_packing_layout=layout,
            )
            q, r = qsa_cp_layout_static(cu, s_loc, CP, rank, layout)
            assert torch.equal(q, ref_q), (lengths, rank)
            assert torch.equal(r, ref_r), (lengths, rank)


@requires_qsa_kernels
class TestQSASelectCP:
    def teardown_method(self, method):
        # Captured graphs hold NCCL work on the CP communicator; free them (they sit in
        # reference cycles) before the groups and, at session end, the process group are
        # destroyed -- destroying a communicator still used by a live graph hangs.
        self.__dict__.clear()
        gc.collect()
        torch.cuda.synchronize()
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    def _build(self, layout="per_document", loss_coeff=None):
        if Utils.world_size < 2:
            pytest.skip("QSA CP2 tests require at least 2 distributed ranks")
        Utils.initialize_model_parallel(1, 1, context_parallel_size=CP)
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)
        config = make_sparse_config()
        config.context_parallel_size = CP
        config.cp_comm_type = "all_gather"
        config.qsa_cp_packing_layout = layout
        config.qsa_indexer_loss_coeff = loss_coeff
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()
        with torch.no_grad():
            for norm in (
                attn.q_layernorm,
                attn.k_layernorm,
                attn.indexer.q_layernorm,
                attn.indexer.k_layernorm,
            ):
                norm.weight.uniform_(-0.1, 0.1)
        import megatron.core.parallel_state as ps

        self.config, self.attn = config, attn
        self.rank = ps.get_context_parallel_rank()
        self.cp_group = ps.get_context_parallel_group()
        self.layout = layout
        rope = RotaryEmbedding(
            kv_channels=SPARSE_HEAD_DIM, rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        self.freqs = rope(T, packed_seq=True)
        world = torch.distributed.get_world_size()
        groups = [torch.distributed.new_group([r]) for r in range(world)]
        self.own_group = groups[torch.distributed.get_rank()]

    def _rows(self, cu):
        q, _ = qsa_cp_layout_static(cu, T // CP, CP, self.rank, self.layout)
        return q

    def _single_rank(self, fn):
        pg = self.attn.pg_collection
        old = pg.cp
        pg.cp = self.own_group
        try:
            return fn()
        finally:
            pg.cp = old

    @pytest.mark.parametrize("layout", ["per_document", "per_sequence"])
    def test_cp2_selection_bitwise_and_sync_free(self, layout):
        self._build(layout)
        packings = PACKINGS_DOC if layout == "per_document" else PACKINGS_SEQ
        g = torch.Generator().manual_seed(5)
        idx = self.attn.indexer
        for lengths in packings:
            cu = _cu(lengths)
            h = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            rows = self._rows(cu)
            shard = h.index_select(0, rows).contiguous()
            with torch.no_grad():
                idx.select(shard, self.freqs, cu_seqlens=cu)  # warm up (autotune)
                torch.cuda.set_sync_debug_mode("error")
                try:
                    sel = idx.select(shard, self.freqs, cu_seqlens=cu)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                for static in (False, True):
                    idx.static_shapes = static
                    ref = self._single_rank(lambda: idx.select(h, self.freqs, cu_seqlens=cu))
                    idx.static_shapes = False
                    for name in ("order", "picked"):
                        assert torch.equal(getattr(sel, name), getattr(ref, name)[:, rows]), (
                            lengths,
                            name,
                        )
                    for name in ("own_block", "block_offsets", "token_offsets"):
                        assert torch.equal(getattr(sel, name), getattr(ref, name)[rows]), (
                            lengths,
                            name,
                        )
                    # The indexer projection GEMM runs at a different M on the shard, so
                    # q / block keys are compared with a tolerance (the selection above is
                    # exact, as in TestQSAContextParallel).
                    torch.testing.assert_close(sel.q, ref.q[rows])
                    assert sel.block_keys.shape[0] == T // RATIO
                    n_real = sum(n // RATIO for n in lengths)
                    torch.testing.assert_close(sel.block_keys[:n_real], ref.block_keys[:n_real])
                    # attention blocks: global token ranges of every selected block agree
                    n_attn = ref.attn_block_bases.shape[0]
                    assert sel.attn_block_bases.shape[0] == T // RATIO + MAX_SEQS
                    assert torch.equal(sel.attn_block_bases[:n_attn], ref.attn_block_bases)
                    assert torch.equal(sel.attn_block_ends[:n_attn], ref.attn_block_ends)
                    assert torch.equal(sel.attn_block_offsets, ref.attn_block_offsets[rows])
                    assert torch.equal(sel.q_positions, rows)

    def test_cp2_selection_shapes_do_not_depend_on_the_packing(self):
        self._build()
        shapes = set()
        g = torch.Generator().manual_seed(6)
        for lengths in PACKINGS_DOC:
            cu = _cu(lengths)
            shard = torch.randn(T // CP, 1, HIDDEN, generator=g).cuda().bfloat16()
            with torch.no_grad():
                sel = self.attn.indexer.select(shard, self.freqs, cu_seqlens=cu)
            shapes.add(tuple(tuple(t.shape) for t in sel if isinstance(t, torch.Tensor)))
        assert len(shapes) == 1

    def test_cp2_kl_matches_single_rank(self):
        """CP2 sparse-KL loss (row-weighted across ranks) and the CP-summed indexer
        gradients equal the single-rank loss / gradients."""
        self._build(loss_coeff=1.0)
        self.attn.train()
        idx = self.attn.indexer
        params = [p for p in idx.parameters() if p.requires_grad]
        g = torch.Generator().manual_seed(11)
        for lengths in ([T], [120, 200], [8, 104, 0, 208]):
            cu = _cu(lengths)
            h = torch.randn(T, 1, HIDDEN, generator=g).cuda().bfloat16()
            rows = self._rows(cu)
            shard = h.index_select(0, rows).contiguous()

            def run(hidden):
                sel = idx.select(hidden, self.freqs, cu_seqlens=cu)
                loss = self.attn._compute_indexer_loss(hidden, self.freqs, sel, 1.0, cu_seqlens=cu)
                return loss, sel.picked.any(dim=-1).sum().float()

            self.attn.zero_grad(set_to_none=True)
            ref_loss, _ = self._single_rank(lambda: run(h))
            ref_loss.backward()
            ref_grads = [p.grad.detach().clone() for p in params]

            self.attn.zero_grad(set_to_none=True)
            loss, valid = run(shard)
            tot = valid.clone()
            torch.distributed.all_reduce(tot, group=self.cp_group)
            (loss * valid / tot).backward()
            kl = (loss * valid).detach()
            torch.distributed.all_reduce(kl, group=self.cp_group)
            got_loss = kl / tot
            torch.testing.assert_close(got_loss, ref_loss.detach(), rtol=1e-5, atol=0)
            for p, r in zip(params, ref_grads):
                gp = p.grad.detach().clone()
                torch.distributed.all_reduce(gp, group=self.cp_group)
                # Summation-order noise only: the CP split regroups the gradient sums, and
                # with bf16 parameters each rank's gradient and the all-reduce round to
                # bf16, so the bar is bf16 resolution (measured max |diff| ~1e-2 of max |g|).
                rel = float((gp.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30))
                assert rel <= 1e-2, f"indexer grad rel-L2 {rel:.3e}"
                torch.testing.assert_close(
                    gp, r, rtol=1.6e-2, atol=2e-2 * float(r.abs().max().clamp_min(1e-30))
                )

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

    def _fwd_bwd(self, fn, hidden, cu, grad_out):
        self.attn.zero_grad(set_to_none=True)
        h = hidden.detach().clone().requires_grad_(True)
        out = fn(h, cu)
        out.backward(grad_out)
        grads = {n: p.grad.detach().clone() for n, p in self.attn.named_parameters()}
        return out.detach().clone(), h.grad.detach().clone(), grads

    @staticmethod
    def _assert_matches(ref, got, ref_repeat, what):
        """Bit-identical except parameter grads that are not repeatable eagerly (the
        student gather backward accumulates with atomics), which use a tolerance."""
        assert torch.equal(ref[0], got[0]), f"output differs for {what}"
        assert torch.equal(ref[1], got[1]), f"input grad differs for {what}"
        for name, r in ref[2].items():
            if torch.equal(r, ref_repeat[2][name]):
                assert torch.equal(r, got[2][name]), f"{name} grad differs for {what}"
            else:
                torch.testing.assert_close(got[2][name], r, msg=f"{name} for {what}")

    def test_cp2_static_layer_matches_dynamic(self):
        """Static-shape kernel metadata (the capture path) vs the dynamic eager path."""
        self._build(loss_coeff=0.01)
        self.attn.train()
        fn = self._layer_fn()
        g = torch.Generator().manual_seed(9)
        for lengths in PACKINGS_DOC:
            cu = _cu(lengths)
            h = torch.randn(T // CP, 1, HIDDEN, generator=g).cuda().bfloat16()
            go = torch.randn(T // CP, 1, HIDDEN, generator=g).cuda().bfloat16()
            self.attn.indexer.static_shapes = False
            ref = self._fwd_bwd(fn, h, cu, go)
            ref_repeat = self._fwd_bwd(fn, h, cu, go)
            self.attn.indexer.static_shapes = True
            got = self._fwd_bwd(fn, h, cu, go)
            self._assert_matches(ref, got, ref_repeat, lengths)

    def test_cp2_capture_replays_new_packings_bit_identically(self):
        from transformer_engine.pytorch import make_graphed_callables

        self._build(loss_coeff=0.01)
        self.attn.train()
        assert self.attn.is_cuda_graph_capturable()
        self.attn.indexer.static_shapes = True
        fn = self._layer_fn()
        g = torch.Generator().manual_seed(9)
        hs = [torch.randn(T // CP, 1, HIDDEN, generator=g).cuda().bfloat16() for _ in PACKINGS_DOC]
        gos = [torch.randn(T // CP, 1, HIDDEN, generator=g).cuda().bfloat16() for _ in PACKINGS_DOC]
        sample = torch.randn(
            T // CP, 1, HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        graphed = make_graphed_callables(fn, (sample, _cu([T])), num_warmup_iters=2)
        eager = [
            (self._fwd_bwd(fn, h, _cu(lengths), go), self._fwd_bwd(fn, h, _cu(lengths), go))
            for lengths, h, go in zip(PACKINGS_DOC, hs, gos)
        ]
        try:
            for lengths, h, go, (ref, ref_repeat) in zip(PACKINGS_DOC, hs, gos, eager):
                got = self._fwd_bwd(graphed, h, _cu(lengths), go)
                self._assert_matches(ref, got, ref_repeat, lengths)
        finally:
            del graphed
            gc.collect()
            torch.cuda.synchronize()
