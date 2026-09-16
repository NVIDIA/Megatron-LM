# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""KDA's FP32 gate parameters must survive BF16 model construction and casting."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.kda import HAVE_FLA_KDA
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_FLA_KDA, reason="FLA KDA is not installed.")
@pytest.mark.parametrize("variant", ["kda", "kda_direct", "gdn"])
def test_gate_parameter_precision_through_bf16_wrapper(variant):
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
            kda_two_stage_gates=variant == "kda",
            kda_safe_gate=True,
            kda_lower_bound=-5.0,
            perform_initialization=True,
        )
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
            assert layer.use_gate_in_kernel == (variant == "kda_direct")
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
            torch.testing.assert_close(packed_output, separate_output, rtol=0.03, atol=0.002)
            packed_output.float().square().mean().backward()
            assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    finally:
        Utils.destroy_model_parallel()
