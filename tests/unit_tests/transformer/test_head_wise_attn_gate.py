# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for per-head scalar attention gate (use_head_wise_attn_gate).

The gate weights are fused into linear_qkv as the trailing num_attention_heads
rows; in the forward pass those scalars are sliced out, passed through sigmoid,
and used to scale each attention head's output independently.
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from tests.unit_tests.test_utilities import Utils


SEQ_LEN = 16
BATCH_SIZE = 2
HIDDEN_SIZE = 128
NUM_HEADS = 4
KV_CHANNELS = HIDDEN_SIZE // NUM_HEADS  # 32
QKV_OUT_DIM = 3 * NUM_HEADS * KV_CHANNELS  # query + key + value, no GQA


def _make_config(transformer_impl: str, use_head_wise_attn_gate: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl=transformer_impl,
        use_head_wise_attn_gate=use_head_wise_attn_gate,
    )


def _make_attention(config: TransformerConfig, transformer_impl: str) -> SelfAttention:
    if transformer_impl == "transformer_engine":
        submodules = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
    else:
        submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    return SelfAttention(config, submodules, layer_number=1)


@pytest.mark.parametrize("transformer_impl", ["transformer_engine", "native"])
class TestHeadWiseAttnGateInit:
    """Verify linear_qkv is sized to include the fused gate rows iff enabled."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, transformer_impl):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        self.transformer_impl = transformer_impl
        yield
        Utils.destroy_model_parallel()

    def test_linear_qkv_includes_gate_rows_when_enabled(self):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl)
        # linear_qkv output dim = Q+K+V + num_attention_heads (gate rows).
        assert attn.linear_qkv_out_dim == QKV_OUT_DIM + NUM_HEADS
        assert attn.linear_qkv.weight.shape == (
            QKV_OUT_DIM + NUM_HEADS,
            HIDDEN_SIZE,
        )

    def test_linear_qkv_unchanged_when_disabled(self):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=False)
        attn = _make_attention(config, self.transformer_impl)
        assert attn.linear_qkv_out_dim == QKV_OUT_DIM
        assert attn.linear_qkv.weight.shape == (QKV_OUT_DIM, HIDDEN_SIZE)

    def test_gate_buffer_initialized_to_none(self):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl)
        assert attn._head_wise_gate_states is None


@pytest.mark.parametrize("transformer_impl", ["transformer_engine", "native"])
class TestHeadWiseAttnGateForward:
    """Verify forward-pass behaviour of the head-wise gate."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, transformer_impl):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        self.transformer_impl = transformer_impl
        yield
        Utils.destroy_model_parallel()

    def _run_forward(self, use_gate: bool):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=use_gate)
        attn = _make_attention(config, self.transformer_impl).cuda()
        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
        output, bias = attn(hidden_states, attention_mask)
        return output, bias

    def test_output_shape_with_gate(self):
        output, _ = self._run_forward(use_gate=True)
        assert output.shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)

    def test_output_shape_without_gate(self):
        output, _ = self._run_forward(use_gate=False)
        assert output.shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)

    def test_gate_buffer_cleared_after_forward(self):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl).cuda()
        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
        _ = attn(hidden_states, attention_mask)
        assert attn._head_wise_gate_states is None

    def test_zero_gate_halves_output(self):
        """Zeroing the trailing gate rows of linear_qkv makes pre-sigmoid 0,
        so the sigmoid is exactly 0.5 and the gated output is half the
        ungated reference output that shares all other weights."""
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl).cuda()

        # Zero only the trailing num_attention_heads rows (the gate weights);
        # leave QKV weights intact so the rest of the computation is unchanged.
        with torch.no_grad():
            attn.linear_qkv.weight[-NUM_HEADS:].zero_()

        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")

        # Reference: a gate-disabled attention layer that shares the QKV rows.
        config_no_gate = _make_config(self.transformer_impl, use_head_wise_attn_gate=False)
        attn_no_gate = _make_attention(config_no_gate, self.transformer_impl).cuda()
        with torch.no_grad():
            attn_no_gate.linear_qkv.weight.copy_(attn.linear_qkv.weight[:QKV_OUT_DIM])
            attn_no_gate.linear_proj.weight.copy_(attn.linear_proj.weight)
            if attn.linear_qkv.bias is not None:
                attn_no_gate.linear_qkv.bias.copy_(attn.linear_qkv.bias[:QKV_OUT_DIM])
            if attn.linear_proj.bias is not None:
                attn_no_gate.linear_proj.bias.copy_(attn.linear_proj.bias)

        with torch.no_grad():
            out_gated, _ = attn(hidden_states, attention_mask)
            out_plain, _ = attn_no_gate(hidden_states, attention_mask)

        torch.testing.assert_close(
            out_gated.float(),
            (out_plain * 0.5).float(),
            atol=1e-2,
            rtol=1e-2,
        )


class TestHeadWiseAttnGateNumerics:
    """Low-level tensor-math tests (no model-parallel overhead, pure PyTorch)."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        yield
        Utils.destroy_model_parallel()

    def test_gate_reshape_correctness(self):
        """Replicate the reshape logic and verify sigmoid is applied per-head."""
        sq, b, np, hn = 4, 2, NUM_HEADS, 32
        # Simulate core_attn_out: [sq, b, np*hn]
        core_attn_out = torch.arange(
            sq * b * np * hn, dtype=torch.float32
        ).reshape(sq, b, np * hn)
        # Simulate gate_states: [sq, b, np] (pre-sigmoid raw scores)
        gate_scores = torch.zeros(sq, b, np)  # sigmoid(0) = 0.5

        gate_states = gate_scores.view(sq, b, np, 1)
        out = core_attn_out.view(sq, b, np, hn)
        out = out * torch.sigmoid(gate_states)
        out = out.view(sq, b, np * hn)

        expected = core_attn_out * 0.5
        torch.testing.assert_close(out, expected)

    def test_gate_dtype_cast(self):
        """Gate computation upcast to float32, result cast back to input dtype."""
        sq, b, np, hn = 4, 2, NUM_HEADS, 32
        core_attn_out = torch.randn(sq, b, np * hn, dtype=torch.bfloat16)
        gate_scores = torch.randn(sq, b, np, dtype=torch.bfloat16)

        gate_states = gate_scores.view(sq, b, np, 1)
        out = core_attn_out.view(sq, b, np, hn)
        # Mirrors the production code: float cast for sigmoid, then cast back
        out = out * torch.sigmoid(gate_states.float()).to(out.dtype)
        out = out.view(sq, b, np * hn)

        assert out.dtype == torch.bfloat16

    @pytest.mark.parametrize("gate_value,expected_scale", [(-1e6, 0.0), (1e6, 1.0), (0.0, 0.5)])
    def test_gate_saturation(self, gate_value: float, expected_scale: float):
        """Extreme gate values saturate sigmoid to 0 or 1; midpoint gives 0.5."""
        sq, b, np, hn = 2, 1, 2, 8
        core_attn_out = torch.ones(sq, b, np * hn)
        gate_scores = torch.full((sq, b, np), gate_value)

        gate_states = gate_scores.view(sq, b, np, 1)
        out = core_attn_out.view(sq, b, np, hn)
        out = out * torch.sigmoid(gate_states.float()).to(out.dtype)
        out = out.view(sq, b, np * hn)

        torch.testing.assert_close(out, torch.full_like(out, expected_scale), atol=1e-5, rtol=1e-5)
