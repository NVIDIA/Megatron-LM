# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KDA's FP32 gate parameters must survive BF16 model construction and casting."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.kda import HAVE_FLA_KDA
from megatron.core.ssm.kda_layer_config import KDALayerConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA KDA is not installed.")
@pytest.mark.parametrize("variant", ["kda", "kda_direct", "gdn"])
def test_gate_parameter_precision_through_bf16_wrapper(variant, monkeypatch):
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=2,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            params_dtype=torch.bfloat16,
            bf16=True,
            normalization="RMSNorm",
            activation_func=torch.nn.functional.silu,
            perform_initialization=True,
        )
        if variant.startswith("kda"):
            config = KDALayerConfig.from_config(config)
            config.kda_two_stage_gates = variant == "kda"
            config.kda_safe_gate = True
            config.kda_lower_bound = -5.0
        layer_spec = getattr(
            hybrid_stack_spec.submodules, f"{variant.removesuffix('_direct')}_layer"
        ).submodules.self_attention
        layer = layer_spec.module(
            config=config,
            submodules=layer_spec.submodules,
            layer_number=1,
            pg_collection=ProcessGroupCollection(
                tp=parallel_state.get_tensor_model_parallel_group(),
                cp=parallel_state.get_context_parallel_group(),
            ),
        )
        expected_dtype = torch.bfloat16 if variant == "gdn" else torch.float32
        expected = {}
        for name in ("A_log", "dt_bias"):
            param = getattr(layer, name)
            assert param.dtype == expected_dtype
            with torch.no_grad():
                param.fill_(0.12345678)
            expected[name] = param.detach().clone()

        Float16Module(config, layer)

        assert layer.in_proj.weight.dtype == torch.bfloat16
        for name, reference in expected.items():
            param = getattr(layer, name)
            assert param.dtype == expected_dtype
            assert param.tensor_model_parallel is True
            assert param.partition_dim == 0
            torch.testing.assert_close(param, reference, rtol=0, atol=0)

        if variant == "kda":
            torch.manual_seed(42)
            x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            gate = torch.randn_like(x, requires_grad=True)
            with torch.no_grad():
                layer.out_norm.weight.uniform_(0.5, 1.5)
            actual = layer._apply_gated_norm(x, gate)
            x_fp32 = x.float()
            expected_norm = x_fp32 * torch.rsqrt(
                x_fp32.square().mean(dim=-1, keepdim=True) + config.layernorm_epsilon
            )
            expected_output = (
                expected_norm * layer.out_norm.weight.float() * gate.float().sigmoid()
            ).to(x.dtype)
            assert (actual.float() - expected_output.float()).abs().mean() < 2e-5
            actual.float().sum().backward()
            for tensor in (x, gate, layer.out_norm.weight):
                assert tensor.grad is not None
                assert torch.isfinite(tensor.grad).all()

        if variant.startswith("kda"):
            gated_delta_rule = layer.gated_delta_rule
            gate_in_kernel_calls = []

            def record_gated_delta_rule(*args, **kwargs):
                gate_in_kernel_calls.append(kwargs.get("use_gate_in_kernel"))
                return gated_delta_rule(*args, **kwargs)

            monkeypatch.setattr(layer, "gated_delta_rule", record_gated_delta_rule)
            hidden = torch.randn(
                260, 1, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            cu_seqlens = torch.tensor([0, 129, 260], device="cuda", dtype=torch.int32)
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=131,
                max_seqlen_kv=131,
            )
            packed_output, _ = layer(hidden, attention_mask=None, packed_seq_params=packed)
            separate_output = torch.cat(
                [layer(segment, attention_mask=None)[0] for segment in hidden.split([129, 131])]
            )
            # Both KDA variants compute the decay gate inside the recurrence kernel.
            assert gate_in_kernel_calls == [True, True, True]
            torch.testing.assert_close(packed_output, separate_output, rtol=0.03, atol=0.002)
            packed_output.float().square().mean().backward()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    finally:
        Utils.destroy_model_parallel()


def _two_stage_kda_config(**overrides) -> KDALayerConfig:
    settings = dict(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=2,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        kda_two_stage_gates=True,
        kda_safe_gate=True,
        kda_lower_bound=-5.0,
    )
    settings.update(overrides)
    return KDALayerConfig(**settings)


def _build_kda_layer(config: KDALayerConfig):
    layer_spec = hybrid_stack_spec.submodules.kda_layer.submodules.self_attention
    return layer_spec.module(
        config=config,
        submodules=layer_spec.submodules,
        layer_number=1,
        pg_collection=ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        ),
    )


def test_two_stage_gates_accept_any_normalization():
    """Two-stage gates fix their output norm to a gated RMSNorm, so `normalization` (which sets
    every other norm) is not restricted, in the config or when the layer re-validates it."""
    config = _two_stage_kda_config(normalization="LayerNorm")
    config.validate_kda()


def test_out_norm_bias_pre_hook_drops_only_that_key():
    """The two-stage load hook drops the LayerNorm bias that checkpoints saved before out_norm
    became weight-only carry, and nothing else, at the module's own prefix."""
    from megatron.core.ssm.gated_delta_net.kda import _drop_unused_out_norm_bias

    prefix = "decoder.layers.0.self_attention."
    state_dict = {
        f"{prefix}out_norm.weight": torch.ones(4),
        f"{prefix}out_norm.bias": torch.full((4,), 0.5),
        "decoder.layers.1.self_attention.out_norm.bias": torch.zeros(4),
    }
    _drop_unused_out_norm_bias(None, state_dict, prefix, {}, True, [], [], [])
    assert set(state_dict) == {
        f"{prefix}out_norm.weight",
        "decoder.layers.1.self_attention.out_norm.bias",
    }
    _drop_unused_out_norm_bias(None, state_dict, prefix, {}, True, [], [], [])


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_two_stage_out_norm_is_a_weight_only_rmsnorm(normalization):
    """Two-stage gates build out_norm as a weight-only RMSNorm whatever `normalization` says, and
    the output is bit-identical to the math they always ran: FLA's gated RMSNorm with
    out_norm's weight and no bias."""
    from fla.modules.fused_norm_gate import rms_norm_gated

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _two_stage_kda_config(normalization=normalization)
        layer = _build_kda_layer(config)
        assert [name for name, _ in layer.out_norm.named_parameters()] == ["weight"]
        assert layer.out_norm.weight.is_rmsnorm_weight
        with torch.no_grad():
            layer.out_norm.weight.add_(torch.randn_like(layer.out_norm.weight) * 0.1)
            torch.manual_seed(42)
            x = torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn_like(x)
            actual = layer._apply_gated_norm(x, gate)
            expected = rms_norm_gated(
                x,
                gate,
                layer.out_norm.weight,
                None,
                activation="sigmoid",
                eps=config.layernorm_epsilon,
            )
        assert torch.equal(actual, expected)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("nested", [False, True], ids=["layer", "in_parent"])
def test_two_stage_loads_a_state_dict_with_the_old_out_norm_bias(nested):
    """A two-stage LayerNorm state dict saved before out_norm became weight-only carries an
    out_norm.bias the forward never read. It loads strictly, whatever the bias value, and the
    layer computes exactly what the saved one does."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        config = _two_stage_kda_config(normalization="LayerNorm")
        model_parallel_cuda_manual_seed(123)
        saved = _build_kda_layer(config)
        with torch.no_grad():
            saved.out_norm.weight.add_(torch.randn_like(saved.out_norm.weight) * 0.1)
        state_dict = saved.state_dict()
        old_layout = dict(state_dict)
        old_layout["out_norm.bias"] = torch.randn_like(saved.out_norm.weight)

        model_parallel_cuda_manual_seed(456)
        loaded = _build_kda_layer(config)
        if nested:
            parent = torch.nn.Module()
            parent.self_attention = loaded
            parent.load_state_dict(
                {f"self_attention.{key}": value for key, value in old_layout.items()}, strict=True
            )
        else:
            loaded.load_state_dict(old_layout, strict=True)

        for key, value in loaded.state_dict().items():
            if isinstance(value, torch.Tensor) and not key.endswith("_extra_state"):
                assert torch.equal(value, state_dict[key]), key
        torch.manual_seed(42)
        hidden = torch.randn(256, 2, config.hidden_size, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            assert torch.equal(loaded(hidden, None)[0], saved(hidden, None)[0])
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA KDA is not installed.")
@pytest.mark.parametrize("weight", [0.25, 0.0035], ids=["exact_in_bf16", "rounded_in_bf16"])
def test_two_stage_zero_centered_gain_is_added_in_fp32(weight):
    """With zero-centered gamma, the two-stage gated norm scales by out_norm.weight + 1 formed in
    FP32, as TE's norms form it. In BF16, 1.25 is exact but 1.0035 rounds to 1."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _two_stage_kda_config(layernorm_zero_centered_gamma=True)
        layer = _build_kda_layer(config)
        with torch.no_grad():
            layer.out_norm.weight.fill_(weight)
        torch.manual_seed(42)
        x = torch.randn(1 << 17, 128, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn_like(x)
        actual = layer._apply_gated_norm(x, gate)
        actual.float().sum().backward()
        assert torch.isfinite(layer.out_norm.weight.grad).all()

        x_fp32 = x.float()
        expected = (
            x_fp32
            * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + config.layernorm_epsilon)
            * (layer.out_norm.weight.detach().float() + 1)
            * gate.float().sigmoid()
        )
        mask = expected.abs() > 1e-2
        # BF16 output rounding averages out over the rows; a gain of 1 instead of 1.0035 does
        # not, and shifts the mean ratio by -3.5e-3.
        ratio = actual.detach().float()[mask] / expected[mask]
        assert abs((ratio - 1).mean().item()) < 5e-4
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA KDA is not installed.")
def test_single_stage_honors_layernorm_out_norm():
    """Direct gates run out_norm itself, so a LayerNorm out_norm keeps its mean and bias."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = _two_stage_kda_config(normalization="LayerNorm", kda_two_stage_gates=False)
        layer = _build_kda_layer(config)
        assert [name for name, _ in layer.out_norm.named_parameters()] == ["weight", "bias"]
        with torch.no_grad():
            layer.out_norm.weight.fill_(1.0)
            layer.out_norm.bias.fill_(0.5)
            torch.manual_seed(42)
            x = torch.randn(1 << 17, 128, device="cuda", dtype=torch.bfloat16) + 0.3
            gate = torch.randn_like(x)
            actual = layer._apply_gated_norm(x, gate).float()
        expected = torch.nn.functional.layer_norm(
            x.float(),
            (128,),
            layer.out_norm.weight.detach().float(),
            layer.out_norm.bias.detach().float(),
            config.layernorm_epsilon,
        ) * torch.sigmoid(gate.float())
        # BF16 rounding gives about 2e-3; RMSNorm without the bias would give about 0.25.
        assert (actual - expected).abs().mean().item() < 2e-2
    finally:
        Utils.destroy_model_parallel()
