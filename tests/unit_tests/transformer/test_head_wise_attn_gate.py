# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for per-head scalar attention gate (use_head_wise_attn_gate / g_proj)."""

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
    """Verify that g_proj is created iff use_head_wise_attn_gate=True."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, transformer_impl):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        self.transformer_impl = transformer_impl
        yield
        Utils.destroy_model_parallel()

    def test_g_proj_exists_when_enabled(self):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl)
        assert hasattr(attn, "g_proj"), "g_proj should be created when use_head_wise_attn_gate=True"

    def test_g_proj_absent_when_disabled(self):
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=False)
        attn = _make_attention(config, self.transformer_impl)
        assert not hasattr(
            attn, "g_proj"
        ), "g_proj should not be created when use_head_wise_attn_gate=False"

    def test_g_proj_output_size(self):
        """g_proj maps hidden_size → num_attention_heads (no bias)."""
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl)
        # ColumnParallelLinear stores weight as (output, input)
        weight = attn.g_proj.weight
        assert weight.shape == (
            NUM_HEADS,
            HIDDEN_SIZE,
        ), f"Unexpected g_proj weight shape: {weight.shape}"


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
        output, bias = self._run_forward(use_gate=True)
        assert output.shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)
        assert bias.shape == (HIDDEN_SIZE,)

    def test_output_shape_without_gate(self):
        output, bias = self._run_forward(use_gate=False)
        assert output.shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)

    def test_gate_changes_output(self):
        """With identical weights/inputs, gating should change the output."""
        torch.manual_seed(0)
        out_gated, _ = self._run_forward(use_gate=True)
        torch.manual_seed(0)
        out_plain, _ = self._run_forward(use_gate=False)
        assert not torch.allclose(out_gated, out_plain), (
            "Gated and plain outputs should differ"
        )

    def test_zero_gate_suppresses_attn(self):
        """When g_proj weights are zero the gate is sigmoid(0)=0.5, not zero;
        confirm that zeroing the bias (if any) and weight gives a 0.5-scaled output."""
        config = _make_config(self.transformer_impl, use_head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl).cuda()

        # Drive g_proj to produce exactly 0 pre-activation → sigmoid = 0.5
        torch.nn.init.zeros_(attn.g_proj.weight)

        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")

        # Reference: disable the gate and run with same weights for linear_proj
        config_no_gate = _make_config(self.transformer_impl, use_head_wise_attn_gate=False)
        attn_no_gate = _make_attention(config_no_gate, self.transformer_impl).cuda()
        # Copy all shared weights so the only difference is the gate scaling
        attn_no_gate.load_state_dict(
            {k: v for k, v in attn.state_dict().items() if k in attn_no_gate.state_dict()},
            strict=False,
        )

        with torch.no_grad():
            out_gated, _ = attn(hidden_states, attention_mask)
            out_plain, _ = attn_no_gate(hidden_states, attention_mask)

        # Gate = sigmoid(0) = 0.5; gated output ≈ 0.5 * plain output
        # Use a loose tolerance because of bfloat16 rounding
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
