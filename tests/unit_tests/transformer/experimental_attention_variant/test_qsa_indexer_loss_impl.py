# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA indexer distillation loss: ``layout`` / ``fused`` implementations vs ``reference``.

All implementations compute the same loss from the same selection. Each case runs the
three implementations on one module (same weights, same inputs, same selection) in a
single forward/backward and compares the teacher targets, the loss, and the indexer
parameter gradients:

* ``layout`` (eager, no gathered-key materialization) must reproduce the reference
  targets within fp32 rounding and its gradients within the reference's own run-to-run
  noise (the reference student's ``torch.gather`` backward is a bf16 atomic scatter-add);
* ``fused`` (CuTe DSL teacher and sparse student kernels) must match within the
  documented tolerances and be bitwise repeatable.

A last test captures a whole QSA layer (selection, sparse attention, indexer loss and its
backward) with ``make_graphed_callables`` and checks replay against eager execution.
"""

import pytest
import torch

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.experimental_attention_variant.qsa_module_specs import (
    get_qsa_module_spec_for_backend,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    requires_qsa_kernels,
)

ROPE_THETA = 10000000.0
ROTARY_PERCENT = 0.25
COEFF = 0.01

# name -> (model dims, input layout). "proxy" is the Qwen3.8 perf-proxy QSA layer at its
# training shape (THD 4096, one document, budget 2048, chunk 1024).
CASES = {
    "proxy_thd_1doc": dict(
        hidden=2560, heads=24, groups=2, kv=256, budget=2048, chunk=1024, lengths=[4096]
    ),
    "thd_multidoc": dict(
        hidden=1024,
        heads=24,
        groups=2,
        kv=128,
        budget=256,
        chunk=512,
        lengths=[1000, 37, 3, 2050, 1006],
    ),
    "bshd_batch2": dict(
        hidden=1024, heads=8, groups=2, kv=128, budget=128, chunk=256, seq=1000, batch=2
    ),
}


def _config(case, dtype=torch.bfloat16):
    return TransformerConfig(
        num_layers=1,
        hidden_size=case["hidden"],
        num_attention_heads=case["heads"],
        num_query_groups=case["groups"],
        kv_channels=case["kv"],
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        normalization="RMSNorm",
        layernorm_zero_centered_gamma=True,
        layernorm_epsilon=1e-6,
        qk_layernorm=True,
        attention_output_gate=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        experimental_attention_variant="qsa",
        qsa_indexer_budget=case["budget"],
        qsa_indexer_compress_ratio=4,
        qsa_indexer_n_heads=4,
        qsa_indexer_head_dim=128,
        qsa_indexer_loss_coeff=COEFF,
        qsa_indexer_loss_chunk=case["chunk"],
        qsa_use_sparse_attention=True,
    )


def _rel_l2(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _report(case, what, value):
    # Grep-able line for the lane report (pytest -s).
    print(f"IDXLOSS_METRIC case={case} {what}={value:.3e}")


@requires_qsa_kernels
class TestQSAIndexerLossImpl:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build(self, case):
        config = _config(case)
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
        attn.train()
        g = torch.Generator().manual_seed(11)
        if "lengths" in case:
            total = sum(case["lengths"])
            hidden = torch.randn(total, 1, case["hidden"], generator=g)
            cu = [0]
            for n in case["lengths"]:
                cu.append(cu[-1] + n)
            cu = torch.tensor(cu, dtype=torch.int32, device="cuda")
            max_len = max(case["lengths"])
        else:
            hidden = torch.randn(case["seq"], case["batch"], case["hidden"], generator=g)
            cu, max_len = None, case["seq"]
        hidden = hidden.to(device="cuda", dtype=torch.bfloat16)
        rope = RotaryEmbedding(
            kv_channels=case["kv"], rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )
        return attn, hidden, rope(max_len), cu

    @staticmethod
    def _run(attn, impl, hidden, freqs, cu, deterministic=False):
        attn.config.qsa_indexer_loss_impl = impl
        attn.zero_grad(set_to_none=True)
        prev = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        try:
            sel = attn.indexer.select(hidden, freqs, cu_seqlens=cu)
            # Student inputs: their gradients isolate the loss backward from the indexer's
            # own (shared, impl-independent) projection backward.
            sel.q.retain_grad()
            sel.block_keys.retain_grad()
            loss = attn._compute_indexer_loss(hidden, freqs, sel, COEFF, cu_seqlens=cu)
            loss.backward()
        finally:
            torch.use_deterministic_algorithms(prev)
        grads = {n: p.grad.detach().float().clone() for n, p in attn.indexer.named_parameters()}
        grads["d_student_q"] = sel.q.grad.detach().float().clone()
        grads["d_block_keys"] = sel.block_keys.grad.detach().float().clone()
        with torch.no_grad():
            q_bt, k_bt = attn._indexer_teacher_qk(hidden, freqs, sel, cu)
            targets = attn._indexer_targets(q_bt, k_bt, sel, impl)
        return loss.detach().double().item(), targets, grads, sel

    @staticmethod
    def _gold_student_grads(sel, targets):
        """fp64 gradients of the KL w.r.t. the student inputs (``selection.q`` and
        ``selection.block_keys``) for fixed targets: the exact value both bf16 paths
        approximate."""
        q = sel.q.detach().double().requires_grad_(True)  # [s, b, nh, d]
        bk = sel.block_keys.detach().double().requires_grad_(True)  # [nb, b, d]
        order, picked = sel.order, sel.picked
        b, s, k = order.shape
        blk = order
        if sel.block_offsets is not None:
            blk = blk + sel.block_offsets.view(1, s, 1)
        bk_bt = bk.permute(1, 0, 2)
        d = bk_bt.shape[-1]
        bk_sel = torch.gather(
            bk_bt,
            1,
            blk.clamp_max(bk_bt.shape[1] - 1).reshape(b, s * k)[..., None].expand(-1, -1, d),
        ).view(b, s, k, d)
        st = torch.einsum("bshd,bskd->bshk", q.permute(1, 0, 2, 3), bk_sel)
        st = torch.relu(st).sum(dim=2) / d**0.5
        st = st.masked_fill(~picked, torch.finfo(torch.float64).min)
        has = picked.any(-1, keepdim=True)
        logp = torch.log_softmax(torch.where(has, st, torch.zeros_like(st)), dim=-1)
        t = targets.double()
        terms = torch.where(picked, t * (torch.log(t.clamp_min(1e-10)) - logp), 0.0)
        loss = terms.sum() / picked.any(-1).sum().clamp_min(1) * COEFF
        dq, dbk = torch.autograd.grad(loss, [q, bk])
        return {"d_student_q": dq, "d_block_keys": dbk}

    @pytest.mark.parametrize("case", list(CASES))
    def test_impls_match_reference(self, case):
        attn, hidden, freqs, cu = self._build(CASES[case])
        ref = self._run(attn, "reference", hidden, freqs, cu)
        ref2 = self._run(attn, "reference", hidden, freqs, cu)
        lay = self._run(attn, "layout", hidden, freqs, cu)
        fus = self._run(attn, "fused", hidden, freqs, cu)
        fus2 = self._run(attn, "fused", hidden, freqs, cu)
        assert ref[3].picked.any(), "the case selects nothing"

        # Reference run-to-run noise: its student backward is a bf16 atomic scatter-add.
        for n in ref[2]:
            _report(case, f"ref_vs_ref_grad_relL2[{n}]", _rel_l2(ref2[2][n], ref[2][n]))
        assert torch.equal(ref[1], ref2[1]) and ref[0] == ref2[0]

        # Targets and loss.
        for name, got, t_tol, l_tol in (("layout", lay, 1e-6, 1e-6), ("fused", fus, 1e-5, 1e-4)):
            t_err = (got[1] - ref[1]).abs().max().item()
            l_err = abs(got[0] - ref[0]) / max(abs(ref[0]), 1e-30)
            _report(case, f"{name}_targets_maxabs", t_err)
            _report(
                case, f"{name}_targets_n_gt_1e-6", float((got[1] - ref[1]).abs().gt(1e-6).sum())
            )
            _report(case, f"{name}_loss_rel", l_err)
            for n in ref[2]:
                _report(case, f"{name}_vs_ref_grad_relL2[{n}]", _rel_l2(got[2][n], ref[2][n]))
            assert t_err <= t_tol, f"{name}: targets max abs {t_err:.3e} > {t_tol:.0e}"
            assert l_err <= l_tol, f"{name}: loss rel {l_err:.3e} > {l_tol:.0e}"

        # 1a: with deterministic algorithms (sorted, non-atomic scatter-add) the layout
        # impl's gradients equal the reference's bit for bit (same student code, same
        # targets).
        ref_det = self._run(attn, "reference", hidden, freqs, cu, deterministic=True)
        ref_det2 = self._run(attn, "reference", hidden, freqs, cu, deterministic=True)
        lay_det = self._run(attn, "layout", hidden, freqs, cu, deterministic=True)
        for n in ref_det[2]:
            rep = torch.equal(ref_det[2][n], ref_det2[2][n])
            e = _rel_l2(lay_det[2][n], ref_det[2][n])
            _report(case, f"det_layout_vs_ref_grad_relL2[{n}]", e)
            if rep:
                assert e == 0.0, f"layout grad {n} differs from deterministic reference"
            else:
                assert e <= 2 * max(_rel_l2(ref_det2[2][n], ref_det[2][n]), 1e-5)

        # 1b: accuracy against the fp64 gradient of the same loss (fixed targets). The
        # reference's bf16 atomic accumulation into the block keys is the least accurate
        # step; the fused sparse student accumulates in fp32.
        gold = self._gold_student_grads(ref[3], ref[1])
        for n in gold:
            e_ref = _rel_l2(ref[2][n], gold[n])
            e_fus = _rel_l2(fus[2][n], gold[n])
            _report(case, f"ref_vs_fp64_relL2[{n}]", e_ref)
            _report(case, f"fused_vs_fp64_relL2[{n}]", e_fus)
            assert e_fus <= 1e-2, f"fused {n} rel-L2 vs fp64 {e_fus:.3e}"
            assert e_fus <= e_ref * 1.05 + 1e-5, f"fused {n} less accurate than reference"

        # The fused path is deterministic: no atomics anywhere in the loss.
        assert fus[0] == fus2[0]
        assert torch.equal(fus[1], fus2[1])
        for n in fus[2]:
            assert torch.equal(fus[2][n], fus2[2][n]), f"fused grad {n} not repeatable"

    @pytest.mark.parametrize("case", ["thd_multidoc", "bshd_batch2"])
    def test_layer_trunk_is_independent_of_impl(self, case):
        """The loss is attached through a forward-identity autoscaler and its inputs are
        detached, so the layer output and the trunk input gradient must be bitwise equal
        for every impl (no aliasing or in-place effects on the main path)."""
        attn, hidden, freqs, cu = self._build(CASES[case])
        psp = None
        if cu is not None:
            psp = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=freqs.shape[0],
                max_seqlen_kv=freqs.shape[0],
            )
        g = torch.Generator().manual_seed(4)
        grad_out = torch.randn(hidden.shape, generator=g).to(hidden.device, hidden.dtype)
        res = {}
        for impl in ("reference", "layout", "fused", "reference"):
            attn.config.qsa_indexer_loss_impl = impl
            attn.zero_grad(set_to_none=True)
            h = hidden.detach().clone().requires_grad_(True)
            out, _ = attn(h, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=psp)
            out.backward(grad_out)
            trunk = {
                n: p.grad.detach().clone()
                for n, p in attn.named_parameters()
                if "indexer" not in n and p.grad is not None
            }
            got = (out.detach().clone(), h.grad.detach().clone(), trunk)
            if impl in res:
                assert torch.equal(res[impl][0], got[0])
            res[impl] = got
        for impl in ("layout", "fused"):
            assert torch.equal(res[impl][0], res["reference"][0]), f"{impl}: output differs"
            assert torch.equal(res[impl][1], res["reference"][1]), f"{impl}: input grad differs"
            for n, gr in res["reference"][2].items():
                assert torch.equal(res[impl][2][n], gr), f"{impl}: trunk grad {n} differs"

    def test_fused_teacher_unpooled_matches_pooled(self):
        """``pool=False`` (the tensor-parallel path) returns the token mass whose block
        max is exactly the pooled kernel output."""
        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_indexer_loss import (  # pylint: disable=line-too-long
            qsa_indexer_teacher,
        )

        attn, hidden, freqs, cu = self._build(CASES["thd_multidoc"])
        with torch.no_grad():
            sel = attn.indexer.select(hidden, freqs, cu_seqlens=cu)
            q_bt, k_bt = attn._indexer_teacher_qk(hidden, freqs, sel, cu)
            kw = dict(
                compress_ratio=4,
                scale=attn.hidden_size_per_attention_head**-0.5,
                token_offsets=sel.token_offsets,
            )
            pooled = qsa_indexer_teacher(q_bt, k_bt, sel.order, sel.picked, pool=True, **kw)
            tok = qsa_indexer_teacher(q_bt, k_bt, sel.order, sel.picked, pool=False, **kw)
        assert torch.equal(tok.amax(-1).masked_fill(~sel.picked, 0.0), pooled)


T = 320
MAX_SEQS = 5
PACKINGS = [[T], [120, 200], [3, 101, 0, 216], [250, 70]]


def _cu(lengths):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    cu += [T] * (MAX_SEQS + 1 - len(cu))
    return torch.tensor(cu, dtype=torch.int32, device="cuda")


@requires_qsa_kernels
class TestQSAIndexerLossCapture:
    """The ``layout`` / ``fused`` loss paths inside a captured QSA layer (as in an
    attention-scope CUDA graph) replay to the eager results on new packings."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        initialize_rng_tracker(
            use_te_rng_tracker=True, use_cudagraphable_rng=True, force_reset=True
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        initialize_rng_tracker(force_reset=True)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("impl", ["layout", "fused"])
    def test_capture_replays_eager(self, impl):
        from transformer_engine.pytorch import make_graphed_callables

        dtype = torch.bfloat16  # the sparse kernels take bf16 / fp16
        case = dict(hidden=256, heads=8, groups=2, kv=128, budget=64, chunk=96)
        config = _config(case, dtype=dtype)
        config.qsa_indexer_loss_impl = impl
        spec = get_qsa_module_spec_for_backend(config)
        attn = build_module(spec, config=config, layer_number=1).cuda()
        attn.train()
        attn.indexer.static_shapes = True
        freqs = RotaryEmbedding(
            kv_channels=case["kv"], rotary_percent=ROTARY_PERCENT, rotary_base=ROPE_THETA
        )(T)

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

        fn = _Fn()

        def fwd_bwd(f, hidden, cu, grad_out):
            attn.zero_grad(set_to_none=True)
            h = hidden.detach().clone().requires_grad_(True)
            out = f(h, cu)
            out.backward(grad_out)
            grads = {n: p.grad.detach().clone() for n, p in attn.named_parameters()}
            return out.detach().clone(), h.grad.detach().clone(), grads

        g = torch.Generator().manual_seed(5)
        hiddens = [
            torch.randn(T, 1, case["hidden"], generator=g).cuda().to(dtype) for _ in PACKINGS
        ]
        grad_outs = [
            torch.randn(T, 1, case["hidden"], generator=g).cuda().to(dtype) for _ in PACKINGS
        ]
        sample = torch.randn(T, 1, case["hidden"], device="cuda", dtype=dtype, requires_grad=True)
        # Capture before the eager references (see test_qsa_static_shapes.py).
        graphed = make_graphed_callables(fn, (sample, _cu([T])), num_warmup_iters=2)
        eager = [
            (fwd_bwd(fn, h, _cu(lengths), go), fwd_bwd(fn, h, _cu(lengths), go))
            for lengths, h, go in zip(PACKINGS, hiddens, grad_outs)
        ]
        for lengths, h, go, (ref, ref_repeat) in zip(PACKINGS, hiddens, grad_outs, eager):
            got = fwd_bwd(graphed, h, _cu(lengths), go)
            assert torch.equal(ref[0], got[0]), f"output differs for {lengths}"
            assert torch.equal(ref[1], got[1]), f"input grad differs for {lengths}"
            n_idx = 0
            for name, r in ref[2].items():
                if "indexer" in name:
                    n_idx += 1
                    assert got[2][name].abs().sum() > 0, f"{name} got no gradient"
                if torch.equal(r, ref_repeat[2][name]):
                    assert torch.equal(r, got[2][name]), f"{name} grad differs for {lengths}"
                else:
                    # Only the layout impl keeps the reference student (atomic gather
                    # backward); the fused impl must be repeatable.
                    assert impl == "layout", f"fused {name} grad not repeatable"
                    torch.testing.assert_close(got[2][name], r, msg=f"{name} for {lengths}")
            assert n_idx > 0
