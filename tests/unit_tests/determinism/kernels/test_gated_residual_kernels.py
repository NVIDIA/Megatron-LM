# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the gated-residual hyper-connection kernels.

Two paths compute the same sublayer call (read gate, write gate, write-back):

* the fused Triton kernels of ``gated_residual_triton.py`` (``deterministic_mode=False``):
  every reduction is a fixed-order partial-sum buffer, replayed here at kernel level (gates,
  write-back) and through ``GatedResidualModule`` with the residual gradient handed from the
  write-back to the gates' backward;
* the ``torch.compile`` path of ``gated_residual.py`` that ``deterministic_mode=True`` keeps.

With gradient-accumulation fusion the fused backward adds the parameter gradients into
``main_grad`` itself (two fp32 GEMM accumulations and the in-kernel gamma / write-gate
reductions); that path is replayed on the accumulated ``main_grad`` buffers.
"""

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import gated_residual_triton as grt
from megatron.core.transformer.gated_residual import GatedResidualModule
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import (
    CONTENTION_TOKENS,
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
needs_triton = pytest.mark.skipif(not grt.HAVE_TRITON, reason="needs Triton")

STREAMS, HIDDEN, LOWRANK, EPS = 4, 1024, 128, 1e-6


def _bf16(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).bfloat16()


@needs_triton
@pytest.mark.parametrize("with_write_gate", [True, False], ids=["sublayer", "exit_contract"])
@pytest.mark.parametrize("fp32_residual", [False, True], ids=["bf16_residual", "fp32_residual"])
def test_fused_gates_replay_fwd_bwd(with_write_gate, fp32_residual):
    """Read gate (and write gate): outputs, input gradient and all four parameter gradients.

    4096 rows span 16 row splits of the norm backward, whose per-split gamma and write-gate
    partials the reduce kernel sums in a fixed order.
    """
    seeded()
    width = STREAMS * HIDDEN
    h = torch.randn(CONTENTION_TOKENS, width, device="cuda")
    h = (h if fp32_residual else h.bfloat16()).requires_grad_()
    gamma = (0.1 * torch.randn(width, device="cuda")).requires_grad_()
    w_down = _bf16(LOWRANK, width, scale=0.02).requires_grad_()
    w_up = _bf16(width, LOWRANK, scale=0.02).requires_grad_()
    w_inject = None
    grad_outputs = {"out[0]": _bf16(CONTENTION_TOKENS, HIDDEN)}
    if with_write_gate:
        w_inject = (0.02 * torch.randn(STREAMS, width, device="cuda")).requires_grad_()
        grad_outputs["out[1]"] = _bf16(CONTENTION_TOKENS, STREAMS)

    def fn(h, gamma, w_down, w_up, w_inject):
        return grt.fused_gated_residual_gates(h, gamma, w_down, w_up, w_inject, EPS, STREAMS, True)

    outputs, grads = assert_replays_bit_exact(
        fn,
        (h, gamma, w_down, w_up, w_inject),
        replays=4,
        grad_outputs=grad_outputs,
        contention=True,
        what="fused gated-residual gates",
    )
    assert len(outputs) == 1 + with_write_gate
    assert len(grads) == 4 + with_write_gate


@needs_triton
@pytest.mark.parametrize("with_bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("fp32_residual", [False, True], ids=["bf16_residual", "fp32_residual"])
def test_fused_write_back_replays_fwd_bwd(with_bias, fp32_residual):
    """Write-back output and the residual / gate / sublayer-output / bias gradients."""
    seeded()
    residual = torch.randn(CONTENTION_TOKENS, STREAMS * HIDDEN, device="cuda")
    residual = (residual if fp32_residual else residual.bfloat16()).requires_grad_()
    gate = (2 * torch.rand(CONTENTION_TOKENS, STREAMS, device="cuda")).bfloat16().requires_grad_()
    x = _bf16(CONTENTION_TOKENS, HIDDEN).requires_grad_()
    bias = _bf16(HIDDEN).requires_grad_() if with_bias else None

    outputs, grads = assert_replays_bit_exact(
        grt.fused_gated_residual_write_back,
        (residual, gate, x, bias),
        replays=4,
        grad_outputs={"out": torch.randn_like(residual)},
        contention=True,
        what="fused gated-residual write-back",
    )
    assert outputs["out"].dtype == residual.dtype
    assert len(grads) == 3 + with_bias


class _Sublayer(torch.nn.Module):
    """One sublayer call: gates, then the write-back into the residual the gates read."""

    def __init__(self, gate: GatedResidualModule):
        super().__init__()
        self.gate = gate

    def forward(self, hidden_states, sublayer_output):
        mixed, h_res, g_write, residual = self.gate(hidden_states)
        assert h_res is None and residual is hidden_states
        out = self.gate.fused_h_res_h_post_bda(
            None, residual, g_write, (sublayer_output, None), 0.0, True, True
        )
        # One tensor, so the harness backpropagates through the write-back and the read gate.
        return torch.cat((out, mixed.to(out.dtype)), dim=-1)


class TestGatedResidualModule:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _sublayer(deterministic_mode):
        seeded()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN,
            num_attention_heads=8,
            use_cpu_initialization=True,
            is_hybrid_model=True,
            enable_mhc_connections=True,
            mhc_num_residual_streams=STREAMS,
            mhc_connection_variant="gated_residual",
            hc_lowrank=LOWRANK,
            layernorm_zero_centered_gamma=True,
            layernorm_epsilon=EPS,
            deterministic_mode=deterministic_mode,
        )
        gate = GatedResidualModule(config, layer_number=1, use_combine=True)
        with torch.no_grad():
            gate.hc_norm.weight.normal_(0.0, 0.1)
        gate = gate.cuda()
        convert_module_to_dtype_except_fp32_marked(gate, torch.bfloat16)
        hidden_states = _bf16(CONTENTION_TOKENS, 1, STREAMS * HIDDEN).requires_grad_()
        sublayer_output = _bf16(CONTENTION_TOKENS, 1, HIDDEN).requires_grad_()
        grad_output = _bf16(CONTENTION_TOKENS, 1, (STREAMS + 1) * HIDDEN)
        return _Sublayer(gate), (hidden_states, sublayer_output), grad_output

    @pytest.mark.parametrize(
        "deterministic_mode",
        [True, pytest.param(False, marks=needs_triton)],
        ids=["torch_compile", "fused_triton"],
    )
    def test_sublayer_replays_fwd_bwd(self, deterministic_mode):
        """Both module paths: output, parameter gradients, and the gradients of both inputs."""
        sublayer, inputs, grad_output = self._sublayer(deterministic_mode)
        assert sublayer.gate.use_fused_kernels is (not deterministic_mode)
        outputs, grads = assert_module_replays_bit_exact(
            sublayer,
            inputs,
            replays=4,
            grad_output=grad_output,
            contention=True,
            what=f"GatedResidualModule (deterministic_mode={deterministic_mode})",
        )
        assert len(outputs) == 1
        assert len(grads) == 4 + 2  # gamma, down, up, inject; hidden states, sublayer output
