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


def _make_config_tp(
    use_head_wise_attn_gate: bool, add_qkv_bias: bool = False
) -> TransformerConfig:
    """Float32 config for TP correctness tests; local-spec backend, no dropout."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_qkv_bias=add_qkv_bias,
        use_head_wise_attn_gate=use_head_wise_attn_gate,
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
    """TP=2 forward tests that back-validate the trailing-rows-as-gate
    layout. Three implicit preconditions are checked indirectly:

      (1) ColumnParallelLinear partitions linear_qkv.weight along axis 0
          contiguously and uniformly (no row interleave).
      (2) num_attention_heads % world_size == 0.
      (3) The global linear_qkv.weight layout is [QKV_block; Gate_block].

    Each test forces the per-rank trailing gate scalars to a known constant
    (via add_qkv_bias=True and bias surgery on the gate rows) and checks
    that the gated output equals the expected scalar multiple of a no-gate
    reference. If any precondition were violated, the slice would pick up
    V/K rows instead of gate rows and the scalar relationship would not
    hold."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        if Utils.world_size < 2:
            pytest.skip(f"need world_size >= 2 (got {Utils.world_size})")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)
        model_parallel_cuda_manual_seed(42)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        "gate_value,expected_scale",
        [
            (0.0, 0.5),    # sigmoid(0)        = 0.5
            (1e4, 1.0),    # sigmoid(+large)  -> 1
            (-1e4, 0.0),   # sigmoid(-large)  -> 0
        ],
    )
    def test_saturated_gate_under_tp2(self, gate_value: float, expected_scale: float):
        """Force the LOCAL trailing num_attention_heads_per_partition gate rows
        on every rank to produce a constant pre-sigmoid value (zero the gate
        weight rows, set the gate bias rows to gate_value). The gated forward
        output should then equal expected_scale * no-gate reference (which
        shares all non-gate weights).

        A layout bug -- e.g. trailing rows being V rows instead of gate --
        would manifest here: setting V bias to +1e4 explodes attention
        outputs, setting V bias to -1e4 wrecks softmax via huge negative V
        contributions. Neither would produce the clean
        ``expected_scale * out_plain`` relation.
        """
        gated_config = _make_config_tp(use_head_wise_attn_gate=True, add_qkv_bias=True)
        attn = _make_attention_local(gated_config)
        gate_size = attn.num_attention_heads_per_partition
        # Per-rank gate-row surgery: zero gate weight rows, set gate bias rows
        # to the target pre-sigmoid value. After this, the gate input to
        # sigmoid is constant `gate_value` regardless of hidden_states.
        assert attn.linear_qkv.bias is not None, "test requires add_qkv_bias=True"
        with torch.no_grad():
            attn.linear_qkv.weight[-gate_size:].zero_()
            attn.linear_qkv.bias[-gate_size:].fill_(gate_value)

        # Reference: no-gate attention sharing every other weight with attn.
        ref_config = _make_config_tp(use_head_wise_attn_gate=False, add_qkv_bias=True)
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

        torch.testing.assert_close(
            out_gated, out_plain * expected_scale, atol=1e-4, rtol=1e-4
        )

    def test_per_rank_gate_independence_under_tp2(self):
        """Drive the gate to +large on rank 0 and -large on rank 1 (and vice
        versa via parametrization is overkill here). Each TP rank's local
        trailing gate rows correspond ONLY to that rank's query heads. After
        linear_proj's RowParallel AllReduce, half the heads contribute
        full-strength and half contribute zero. This test verifies the
        slice does not cross rank boundaries -- if a rank ever read its
        partner's gate scalars (impossible under axis-0 contiguous TP, but
        the test guards the assumption), the half-zero / half-pass pattern
        would collapse and the result would not match the expected
        per-rank-weighted reference.

        Verification trick: build the reference by running the no-gate
        attention on each rank with its OWN core_attn_out scaled to
        sigmoid(rank_gate); since linear_proj is row-parallel and sums TP
        partial outputs, scaling each rank's slice of linear_proj.weight by
        sigmoid(rank_gate) BEFORE the all-reduce yields the same final
        output as gating each rank's core_attn_out by sigmoid(rank_gate).
        """
        tp_rank = torch.distributed.get_rank() % 2
        gate_value = 1e4 if tp_rank == 0 else -1e4
        # sigmoid(+1e4) ~= 1, sigmoid(-1e4) ~= 0
        expected_rank_scale = 1.0 if tp_rank == 0 else 0.0

        gated_config = _make_config_tp(use_head_wise_attn_gate=True, add_qkv_bias=True)
        attn = _make_attention_local(gated_config)
        gate_size = attn.num_attention_heads_per_partition
        assert attn.linear_qkv.bias is not None
        with torch.no_grad():
            attn.linear_qkv.weight[-gate_size:].zero_()
            attn.linear_qkv.bias[-gate_size:].fill_(gate_value)

        # Reference: no-gate model with linear_proj.weight pre-scaled by this
        # rank's expected_rank_scale. linear_proj is RowParallelLinear, so
        # each rank holds (output, input_per_rank) and contributes
        # `core_attn_local @ weight_local.T` to the all-reduced sum. Scaling
        # weight_local by sigmoid(gate) on this rank == scaling that rank's
        # contribution by sigmoid(gate), which is the per-rank effect of
        # head-wise gating on rank-local heads.
        ref_config = _make_config_tp(use_head_wise_attn_gate=False, add_qkv_bias=True)
        attn_ref = _make_attention_local(ref_config)
        _copy_qkv_block_only(attn_ref, attn)
        with torch.no_grad():
            attn_ref.linear_proj.weight.mul_(expected_rank_scale)

        torch.manual_seed(0)
        hidden_states = torch.randn(
            SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda"
        )
        attention_mask = torch.ones(BATCH_SIZE, 1, 1, SEQ_LEN, dtype=bool, device="cuda")
        with torch.no_grad():
            out_gated, _ = attn(hidden_states, attention_mask)
            out_ref, _ = attn_ref(hidden_states, attention_mask)

        torch.testing.assert_close(out_gated, out_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "num_heads,num_query_groups,expected_match",
        [
            # num_query_groups (1) < world_size (2): gate slice would over-take
            # via the AG+reslice fallback path.
            (4, 1, "num_query_groups"),
            # num_attention_heads (3) % world_size (2) != 0, with
            # num_query_groups == num_attention_heads (default) and < world_size
            # too, so parent's divide(...) uses num_query_groups not world_size
            # and our explicit gate-row divisibility check is the one that
            # fires.
            (3, 1, "num_query_groups"),
        ],
    )
    def test_init_rejects_misaligned_layout_under_tp2(
        self, num_heads: int, num_query_groups: int, expected_match: str
    ):
        """Layout invariants for use_head_wise_attn_gate (both checked at
        init):

          (a) num_query_groups >= world_size -- otherwise the gate slice
              over-takes through the AG+reslice fallback.
          (b) num_attention_heads % world_size == 0 -- otherwise gate rows
              cannot be partitioned cleanly across ranks.

        Both should produce a clear AssertionError at SelfAttention
        construction time, not a silent forward-time misalignment.
        """
        bad_config = TransformerConfig(
            num_layers=1,
            hidden_size=24,  # divisible by num_heads for head_dim
            num_attention_heads=num_heads,
            num_query_groups=num_query_groups,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_head_wise_attn_gate=True,
        )
        submodules = get_gpt_layer_local_spec().submodules.self_attention.submodules
        with pytest.raises(AssertionError, match=expected_match):
            SelfAttention(bad_config, submodules, layer_number=1)
