# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bit-exact replay of the padding-aware (live-row bounded) kernels for packed sequences.

Every kernel here reads a device-side live-row bound (``padding_aware_gemm.live_rows_bound``) and
processes only the rows below it, zero-filling the padding tail. The bound is part of the inputs
of the replay, so a replay also certifies that the tail stays byte-identical (zeros) across runs:
the Triton cross entropy (one program per row, tensor-parallel statistics all-gathered), the tiled
RMSNorm, the three-way gradient add, the row arg-max, and the one-group grouped GEMM behind the
padding-aware linears (Transformer Engine ``GroupedTensor`` with a strided leading dimension).
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.fusions import fused_padding_aware_cross_entropy as pace
from megatron.core.fusions import fused_padding_aware_elementwise as paew
from megatron.core.fusions import fused_padding_aware_norm as pan
from megatron.core.transformer import padding_aware_gemm as pag
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_replays_bit_exact,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and pag.HAVE_TRITON), reason="needs a GPU and Triton"
)

ROWS = CONTENTION_TOKENS  # 4096 rows: dozens of programs per output column
LIVE = 2688  # live-row bound (a multiple of the 128-row alignment, 66 % of the buffer)


def _bound(rows=LIVE):
    return torch.tensor([rows], device="cuda", dtype=torch.int64)


class TestPaddingAwareKernels:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("bounded", [True, False], ids=["live-rows", "all-rows"])
    def test_triton_cross_entropy_replays_fwd_bwd(self, bounded):
        """Online-softmax forward plus the backward that recomputes the softmax from bf16 logits."""
        seeded()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        vocab = 32768
        logits = torch.randn(
            ROWS, 1, vocab, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        target = torch.randint(0, vocab * tp_group.size(), (ROWS, 1), device="cuda")
        bound = _bound()

        def run(logits, target):
            if bounded:
                return pace.padding_aware_cross_entropy(logits, target, bound, tp_group)
            return pace.triton_cross_entropy(logits, target, tp_group)

        outputs, grads = assert_replays_bit_exact(
            run,
            (logits, target),
            replays=3,
            contention=True,
            what=f"triton cross entropy[{'live rows' if bounded else 'all rows'}]",
        )
        if bounded:
            assert not outputs["out"][LIVE:].any() and not grads["in[0]"][LIVE:].any()

    @pytest.mark.parametrize("zero_centered", [False, True])
    @pytest.mark.parametrize("weighted", [True, False], ids=["weighted", "weightless"])
    def test_padding_aware_rms_norm_replays_fwd_bwd(self, weighted, zero_centered):
        """Tiled RMSNorm: fp32 statistics per row, weight gradient reduced over 32-row programs."""
        seeded()
        cols = 4096
        x = torch.randn(ROWS, 1, cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = (
            torch.randn(cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            if weighted
            else None
        )
        bound = _bound()

        def run(x, weight=None):
            return pan.padding_aware_rms_norm(x, weight, 1e-6, bound, zero_centered)

        inputs = (x, weight) if weighted else (x,)
        outputs, grads = assert_replays_bit_exact(
            run,
            inputs,
            replays=3,
            contention=True,
            what=f"padding-aware rms norm[{'weighted' if weighted else 'weightless'}]",
        )
        assert not outputs["out"][LIVE:].any() and not grads["in[0]"][LIVE:].any()

    def test_padding_aware_add3_replays(self):
        """The bounded three-way add of the hyper-connection backward (no autograd of its own)."""
        seeded()
        cols = 1024
        a, b, c = (torch.randn(ROWS, cols, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        bound = _bound()
        outputs, _ = assert_replays_bit_exact(
            lambda a, b, c: paew.padding_aware_add3(a, b, c, bound),
            (a, b, c),
            replays=3,
            backward=False,
            contention=True,
            what="padding-aware add3",
        )
        assert not outputs["out"][LIVE:].any()

    def test_padding_aware_row_argmax_replays(self):
        """Row arg-max over the live rows; the tail is -1 (no label matches it)."""
        seeded()
        vocab = 30016  # not a multiple of the block
        logits = torch.randn(ROWS, 1, vocab, device="cuda", dtype=torch.bfloat16)
        bound = _bound()
        outputs, _ = assert_replays_bit_exact(
            lambda l: paew.padding_aware_row_argmax(l, bound),
            (logits,),
            replays=3,
            backward=False,
            contention=True,
            what="padding-aware row argmax",
        )
        out = outputs["out"]
        assert torch.equal(out[:LIVE], torch.argmax(logits[:LIVE], dim=-1))
        assert (out[LIVE:] == -1).all()

    @pytest.mark.skipif(
        not pag.HAVE_TE_GROUPED_LEADING_DIM,
        reason="Transformer Engine without grouped-tensor leading_dim support",
    )
    @pytest.mark.parametrize("wgrad_grouped", [False, True], ids=["wgrad-dense", "wgrad-grouped"])
    def test_padding_aware_linear_replays_fwd_bwd(self, wgrad_grouped):
        """One-group grouped GEMM over the live rows: forward, input gradient, weight gradient."""
        seeded()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        in_features, out_features = 2048, 4096
        inp = torch.randn(
            ROWS, 1, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(
            out_features, in_features, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        bound = _bound()

        def run(inp, weight):
            return pag.padding_aware_linear(
                inp,
                weight,
                bound,
                gradient_accumulation_fusion=False,
                allreduce_dgrad=False,
                tp_group=tp_group,
                wgrad_grouped=wgrad_grouped,
            )

        outputs, grads = assert_replays_bit_exact(
            run,
            (inp, weight),
            replays=3,
            contention=True,
            what=f"padding-aware linear[{'grouped' if wgrad_grouped else 'dense'} wgrad]",
        )
        assert not outputs["out"][LIVE:].any() and not grads["in[0]"][LIVE:].any()
