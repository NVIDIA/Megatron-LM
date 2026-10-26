# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""QSA fused indexer loss, CuTe DSL kernels (``ops/cute_qsa_indexer_loss.py``).

* Teacher (query-tile kernel for short documents plus the row kernel, and the row kernel
  alone): normalized targets within 1e-5 of the reference; both kernels agree to fp32
  summation order; bitwise repeatable; the unpooled (tensor-parallel) output max-pools to the
  pooled one.
* Sparse student: the fused loss within 1e-4 of the reference, student-input gradients
  within 1e-2 of the fp64 gold and no less accurate than the reference, bitwise repeatable.
* The fused loss reuses the main path's q/k, which equal the teacher's own re-projection.

The layer-level comparison of the implementations and the CUDA-graph capture of the fused
loss are covered by test_qsa_indexer_loss_impl.py.
"""

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant import (
    test_qsa_indexer_loss_impl as base,
)
from tests.unit_tests.transformer.experimental_attention_variant.test_attention_variant_qsa import (
    requires_qsa_kernels,
)

pytestmark = requires_qsa_kernels

CASES = base.CASES


class TestQSAIndexerLossCute:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("case", list(CASES))
    def test_teacher(self, case):
        from megatron.core.transformer.experimental_attention_variant.ops.cute_qsa_indexer_loss import (  # pylint: disable=line-too-long
            qsa_indexer_teacher,
        )

        attn, hidden, freqs, cu = base.TestQSAIndexerLossImpl._build(None, CASES[case])
        with torch.no_grad():
            sel = attn.indexer.select(hidden, freqs, cu_seqlens=cu)
            q_bt, k_bt = attn._indexer_teacher_qk(hidden, freqs, sel, cu)
            kw = dict(
                compress_ratio=4,
                scale=attn.hidden_size_per_attention_head**-0.5,
                token_offsets=sel.token_offsets,
            )
            res = {}
            for kernel in ("auto", "row"):
                for pool in (True, False):
                    res[kernel, pool] = qsa_indexer_teacher(
                        q_bt, k_bt, sel.order, sel.picked, pool=pool, kernel=kernel, **kw
                    )
            again = qsa_indexer_teacher(q_bt, k_bt, sel.order, sel.picked, **kw)
            ref = attn._indexer_targets(q_bt, k_bt, sel, "reference")
            got = attn._indexer_targets(q_bt, k_bt, sel, "fused")
        assert torch.equal(res["auto", True], again), "teacher not repeatable"
        for kernel in ("auto", "row"):
            assert torch.equal(
                res[kernel, False].amax(-1).masked_fill(~sel.picked, 0.0), res[kernel, True]
            ), kernel
        for pool in (True, False):
            err = (res["auto", pool] - res["row", pool]).abs().max().item()
            base._report(case, f"teacher_auto_vs_row_pool{int(pool)}_maxabs", err)
            assert err <= 1e-5, f"tile vs row teacher {err:.3e}"
        t_err = (got - ref).abs().max().item()
        base._report(case, "fused_targets_maxabs", t_err)
        assert t_err <= 1e-5

    @pytest.mark.parametrize("case", list(CASES))
    def test_sparse_student(self, case):
        attn, hidden, freqs, cu = base.TestQSAIndexerLossImpl._build(None, CASES[case])
        ref = base.TestQSAIndexerLossImpl._run(attn, "reference", hidden, freqs, cu)
        got = base.TestQSAIndexerLossImpl._run(attn, "fused", hidden, freqs, cu)
        got2 = base.TestQSAIndexerLossImpl._run(attn, "fused", hidden, freqs, cu)

        t_err = (got[1] - ref[1]).abs().max().item()
        l_err = abs(got[0] - ref[0]) / max(abs(ref[0]), 1e-30)
        base._report(case, "fused_targets_maxabs", t_err)
        base._report(case, "fused_loss_rel", l_err)
        assert t_err <= 1e-5
        assert l_err <= 1e-4, f"loss rel {l_err:.3e}"
        gold = base.TestQSAIndexerLossImpl._gold_student_grads(ref[3], ref[1])
        for n in gold:
            e_ref = base._rel_l2(ref[2][n], gold[n])
            e_got = base._rel_l2(got[2][n], gold[n])
            base._report(case, f"ref_vs_fp64_relL2[{n}]", e_ref)
            base._report(case, f"fused_vs_fp64_relL2[{n}]", e_got)
            assert e_got <= 1e-2, f"{n} rel-L2 vs fp64 {e_got:.3e}"
            assert e_got <= e_ref * 1.05 + 1e-5, f"{n} less accurate than reference"
        assert got[0] == got2[0]
        assert torch.equal(got[1], got2[1])
        for n in got[2]:
            assert torch.equal(got[2][n], got2[2][n]), f"grad {n} not repeatable"

    @pytest.mark.parametrize("case", ["thd_multidoc", "bshd_batch2"])
    def test_fused_loss_reuses_identical_qk(self, case):
        """The main path's q/k handed to the teacher equal its own re-projection."""
        attn, hidden, freqs, cu = base.TestQSAIndexerLossImpl._build(None, CASES[case])
        attn.config.qsa_indexer_loss_impl = "fused"
        psp = None
        if cu is not None:
            psp = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu,
                cu_seqlens_kv=cu,
                max_seqlen_q=freqs.shape[0],
                max_seqlen_kv=freqs.shape[0],
            )
        seen = []
        orig = attn._compute_indexer_loss

        def spy(hs, fr, sel, coeff, cu_seqlens=None, teacher_qk=None):
            stashed = attn._indexer_main_qk
            assert stashed is not None, "main q/k not stashed"
            seen.append((stashed, attn._indexer_teacher_qk(hs, fr, sel, cu_seqlens)))
            return orig(hs, fr, sel, coeff, cu_seqlens=cu_seqlens, teacher_qk=teacher_qk)

        attn._compute_indexer_loss = spy
        out, _ = attn(hidden, attention_mask=None, rotary_pos_emb=freqs, packed_seq_params=psp)
        out.sum().backward()
        assert len(seen) == 1
        (mq, mk), (tq, tk) = seen[0]
        assert torch.equal(mq, tq) and torch.equal(mk, tk)
        assert attn._indexer_main_qk is None
