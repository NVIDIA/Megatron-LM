# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed GDN parity for runtime CP, including static and TP/SP controls."""

import pytest
import torch.nn.functional as F

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.ssm.gated_delta_net import HAVE_FLA, HAVE_FLA_GDN2
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.transformer.parallel_test_utils import _test_parallel_attention_correctness

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            "gated_delta_net", marks=pytest.mark.skipif(not HAVE_FLA, reason="requires FLA")
        ),
        pytest.param(
            "gdn2", marks=pytest.mark.skipif(not HAVE_FLA_GDN2, reason="requires FLA GDN2")
        ),
    ],
    ids=["gdn", "gdn2"],
)
@pytest.mark.parametrize(
    ("tp", "sp", "dynamic_cp"),
    [
        pytest.param(1, False, False, id="static_cp2"),
        pytest.param(1, False, True, id="runtime_cp2"),
        pytest.param(2, True, True, id="runtime_tp2_sp_cp2"),
    ],
)
def test_gdn_runtime_parallel_correctness(
    tmp_path_dist_ckpt, variant, tp, sp, dynamic_cp, run_config
):
    config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant=variant,
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )
    spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=config, vp_stage=None, pp_rank=0
    )
    is_gdn2 = variant == "gdn2"
    tolerance = 3e-2 if is_gdn2 else 5e-3

    def static_control_factory(static_config, static_groups):
        static_spec = get_gated_delta_net_module_spec(static_config)
        return (
            static_spec.module(
                static_config, static_spec.submodules, layer_number=1, pg_collection=static_groups
            )
            .cuda()
            .bfloat16()
        )

    _test_parallel_attention_correctness(
        transformer_config=config,
        transformer_layer_spec=spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=tolerance,
        rtol=tolerance,
        tp=tp,
        sp=sp,
        cp=2,
        seed=42 if is_gdn2 else 123,
        sequence_length=512 if is_gdn2 else 256,
        micro_batch_size=2 if is_gdn2 else 4,
        sequence_packing=True,
        dynamic_cp=dynamic_cp,
        check_parameter_gradients=True,
        # Local BF16 projection gradients round before the FP32 CP/SP reduction;
        # GDN2 shows larger serial-vs-partitioned differences on GB200 and A6000.
        parameter_atol=1e-1 if is_gdn2 else 2e-2,
        parameter_rtol=2e-2,
        static_control_factory=static_control_factory,
    )
