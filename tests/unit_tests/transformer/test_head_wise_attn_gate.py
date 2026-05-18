# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for per-head scalar attention gate (head_wise_attn_gate).

Unverified paths (follow-ups): CP, fp8, explicit backward-grad-to-gate,
packed_seq_params.qkv_format == 'thd'.
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


def _make_config(transformer_impl: str, head_wise_attn_gate: bool) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        transformer_impl=transformer_impl,
        head_wise_attn_gate=head_wise_attn_gate,
    )


def _make_attention(config: TransformerConfig, transformer_impl: str) -> SelfAttention:
    if transformer_impl == "transformer_engine":
        submodules = get_gpt_layer_with_transformer_engine_submodules().self_attention.submodules
    else:
        submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    return SelfAttention(config, submodules, layer_number=1)


class TestHeadWiseAttnGateConfigValidation:
    """TransformerConfig.__post_init__ rejects misconfigurations at config time."""

    def _config(self, **extra) -> TransformerConfig:
        return TransformerConfig(
            num_layers=1,
            hidden_size=HIDDEN_SIZE,
            num_attention_heads=NUM_HEADS,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            **extra,
        )

    def test_rejects_combination_with_attention_output_gate(self):
        with pytest.raises(ValueError, match="cannot both be enabled"):
            self._config(head_wise_attn_gate=True, attention_output_gate=True)

    def test_rejects_num_heads_indivisible_by_tp(self):
        with pytest.raises(ValueError, match="num_attention_heads"):
            self._config(
                num_attention_heads=3,
                num_query_groups=3,
                hidden_size=24,
                tensor_model_parallel_size=2,
                head_wise_attn_gate=True,
            )

    def test_rejects_num_query_groups_below_tp(self):
        with pytest.raises(ValueError, match="num_query_groups"):
            self._config(
                num_attention_heads=4,
                num_query_groups=1,
                tensor_model_parallel_size=2,
                head_wise_attn_gate=True,
            )

    def test_accepts_well_formed_config(self):
        self._config(
            num_attention_heads=4,
            num_query_groups=4,
            tensor_model_parallel_size=2,
            head_wise_attn_gate=True,
        )

    def test_disabled_does_not_constrain_other_flags(self):
        """Gate-off must not regress unrelated configs."""
        self._config(head_wise_attn_gate=False, attention_output_gate=True)
        self._config(
            num_attention_heads=4,
            num_query_groups=1,
            tensor_model_parallel_size=2,
            head_wise_attn_gate=False,
        )

    def test_rejects_fp8_misaligned_linear_qkv_out_dim(self):
        # H=8, G=8, D=128, TP=1 -> linear_qkv_out_dim=3080; 3080 % 16 = 8.
        with pytest.raises(ValueError, match=r"fp8.*multiple of 16"):
            self._config(
                num_attention_heads=8,
                num_query_groups=8,
                hidden_size=8 * 128,
                head_wise_attn_gate=True,
                fp8="hybrid",
            )

    def test_rejects_fp4_stricter_alignment(self):
        # H=16, G=2, D=128, TP=1 -> 2576; passes fp8 (16) but fails fp4 (32).
        with pytest.raises(ValueError, match=r"fp4.*multiple of 32"):
            self._config(
                num_attention_heads=16,
                num_query_groups=2,
                hidden_size=16 * 128,
                head_wise_attn_gate=True,
                fp4="e2m1",
            )


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
        config = _make_config(self.transformer_impl, head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl)
        # linear_qkv output dim = Q+K+V + num_attention_heads (gate rows).
        assert attn.linear_qkv_out_dim == QKV_OUT_DIM + NUM_HEADS
        assert attn.linear_qkv.weight.shape == (QKV_OUT_DIM + NUM_HEADS, HIDDEN_SIZE)

    def test_linear_qkv_unchanged_when_disabled(self):
        config = _make_config(self.transformer_impl, head_wise_attn_gate=False)
        attn = _make_attention(config, self.transformer_impl)
        assert attn.linear_qkv_out_dim == QKV_OUT_DIM
        assert attn.linear_qkv.weight.shape == (QKV_OUT_DIM, HIDDEN_SIZE)


@pytest.mark.parametrize("transformer_impl", ["transformer_engine", "native"])
class TestHeadWiseAttnGateForward:
    """Forward-pass shape sanity + zero-gate equivalence."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self, transformer_impl):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        self.transformer_impl = transformer_impl
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("use_gate", [True, False])
    def test_output_shape(self, use_gate: bool):
        config = _make_config(self.transformer_impl, head_wise_attn_gate=use_gate)
        attn = _make_attention(config, self.transformer_impl).cuda()
        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
        output, _ = attn(hidden_states, attention_mask)
        assert output.shape == (SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)

    def test_zero_gate_halves_output(self):
        """Zero gate rows -> sigmoid(0)=0.5 -> output = 0.5 * no-gate ref."""
        config = _make_config(self.transformer_impl, head_wise_attn_gate=True)
        attn = _make_attention(config, self.transformer_impl).cuda()
        with torch.no_grad():
            attn.linear_qkv.weight[-NUM_HEADS:].zero_()

        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")

        config_no_gate = _make_config(self.transformer_impl, head_wise_attn_gate=False)
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
            out_gated.float(), (out_plain * 0.5).float(), atol=1e-2, rtol=1e-2
        )


def _make_config_tp(
    head_wise_attn_gate: bool, add_qkv_bias: bool = False, **extra: object
) -> TransformerConfig:
    """Float32 config for TP correctness tests; local-spec backend, no dropout.

    Extra TransformerConfig kwargs (e.g. head_wise_attn_gate_init_*) can be
    passed through ``extra``.
    """
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_qkv_bias=add_qkv_bias,
        head_wise_attn_gate=head_wise_attn_gate,
        **extra,
    )


def _make_attention_local(config: TransformerConfig) -> SelfAttention:
    """Local (ColumnParallelLinear) backend so the trailing-rows layout is
    unambiguous -- TE variants may carry layer-norm params that don't matter
    for these layout checks but would complicate weight surgery."""
    submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
    return SelfAttention(config, submodules, layer_number=1).cuda()


def _copy_qkv_block_only(attn_dst: SelfAttention, attn_src: SelfAttention):
    """Copy the non-gate (QKV-only) rows of linear_qkv and the full linear_proj
    from attn_src into attn_dst. Used to build a no-gate reference that
    shares all weights with a gated layer EXCEPT the trailing gate rows. Each
    rank handles its own local slice."""
    qkv_rows = attn_dst.linear_qkv.weight.size(0)
    with torch.no_grad():
        attn_dst.linear_qkv.weight.copy_(attn_src.linear_qkv.weight[:qkv_rows])
        attn_dst.linear_proj.weight.copy_(attn_src.linear_proj.weight)
        if attn_src.linear_qkv.bias is not None and attn_dst.linear_qkv.bias is not None:
            attn_dst.linear_qkv.bias.copy_(attn_src.linear_qkv.bias[:qkv_rows])
        if attn_src.linear_proj.bias is not None and attn_dst.linear_proj.bias is not None:
            attn_dst.linear_proj.bias.copy_(attn_src.linear_proj.bias)


class TestHeadWiseAttnGateUnderTP2:
    """TP=2 forward equivalence under known constant gate values."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        if Utils.world_size < 2:
            pytest.skip(f"need world_size >= 2 (got {Utils.world_size})")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("gate_value,expected_scale", [(0.0, 0.5), (1e4, 1.0), (-1e4, 0.0)])
    def test_saturated_gate_under_tp2(self, gate_value: float, expected_scale: float):
        """Zero gate weight rows + fill gate bias rows with gate_value, then
        verify out_gated == sigmoid(gate_value) * out_no_gate. Catches layout
        bugs where the slice would land on V/K rows instead of gate rows."""
        gated_config = _make_config_tp(head_wise_attn_gate=True, add_qkv_bias=True)
        attn = _make_attention_local(gated_config)
        gate_size = attn.num_attention_heads_per_partition
        assert attn.linear_qkv.bias is not None
        with torch.no_grad():
            attn.linear_qkv.weight[-gate_size:].zero_()
            attn.linear_qkv.bias[-gate_size:].fill_(gate_value)

        ref_config = _make_config_tp(head_wise_attn_gate=False, add_qkv_bias=True)
        attn_ref = _make_attention_local(ref_config)
        _copy_qkv_block_only(attn_ref, attn)

        torch.manual_seed(0)
        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
        with torch.no_grad():
            out_gated, _ = attn(hidden_states, attention_mask)
            out_plain, _ = attn_ref(hidden_states, attention_mask)

        torch.testing.assert_close(out_gated, out_plain * expected_scale, atol=1e-4, rtol=1e-4)


class TestHeadWiseAttnGateInitMagnitude:
    """Init-time gate-row surgery: near-identity at start, knob propagation, FP8 safety."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    def _run_pair(self, gated_config: TransformerConfig) -> tuple[torch.Tensor, torch.Tensor]:
        attn = _make_attention_local(gated_config)
        ref_config = _make_config_tp(head_wise_attn_gate=False, add_qkv_bias=True)
        attn_ref = _make_attention_local(ref_config)
        _copy_qkv_block_only(attn_ref, attn)

        torch.manual_seed(0)
        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
        with torch.no_grad():
            out_gated, _ = attn(hidden_states, attention_mask)
            out_plain, _ = attn_ref(hidden_states, attention_mask)
        return out_gated, out_plain

    def test_default_init_is_near_identity(self):
        """out_gated ~= sigmoid(2.0) * out_plain at init; closer than 0.5 *
        out_plain (catches regressions that drop the init surgery)."""
        gated_config = _make_config_tp(head_wise_attn_gate=True, add_qkv_bias=True)
        out_gated, out_plain = self._run_pair(gated_config)
        expected_scale = torch.sigmoid(
            torch.tensor(gated_config.head_wise_attn_gate_init_bias)
        ).item()
        torch.testing.assert_close(out_gated, out_plain * expected_scale, atol=5e-2, rtol=5e-2)
        near_id_err = (out_gated - out_plain * expected_scale).abs().mean().item()
        half_err = (out_gated - out_plain * 0.5).abs().mean().item()
        assert near_id_err < half_err

    def test_init_knobs_reproduce_half_scaling(self):
        """Both knobs=0 reproduces sigmoid(0)=0.5 -- proves the knobs flow."""
        gated_config = _make_config_tp(
            head_wise_attn_gate=True,
            add_qkv_bias=True,
            head_wise_attn_gate_init_weight_scale=0.0,
            head_wise_attn_gate_init_bias=0.0,
        )
        out_gated, out_plain = self._run_pair(gated_config)
        torch.testing.assert_close(out_gated, out_plain * 0.5, atol=1e-4, rtol=1e-4)

    def test_init_preserves_fp8_friendly_gate_magnitude(self):
        """gate_std ~ 0.1 * qkv_std (no zero-init, no FP8 underflow risk)."""
        gated_config = _make_config_tp(head_wise_attn_gate=True, add_qkv_bias=True)
        attn = _make_attention_local(gated_config)
        gate_rows = attn.linear_qkv.weight[-attn.num_attention_heads_per_partition :]
        qkv_rows = attn.linear_qkv.weight[: -attn.num_attention_heads_per_partition]
        ratio = gate_rows.float().std().item() / qkv_rows.float().std().item()
        assert 0.05 < ratio < 0.5
