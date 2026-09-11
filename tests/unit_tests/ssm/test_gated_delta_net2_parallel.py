# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch.nn.functional as F

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.ssm.gated_delta_net import HAVE_FLA_GDN2
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [(4, True, 1), (1, False, 2), (2, True, 2)],  # TP w/ SP  # CP  # TP w/ SP + CP
)
@pytest.mark.skipif(not HAVE_FLA_GDN2, reason="FLA with GDN2 support is not installed.")
def test_parallel_gated_delta_net2_correctness(tmp_path_dist_ckpt, sequence_packing, tp, sp, cp):
    transformer_config = TransformerConfig(
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
        experimental_attention_variant="gdn2",
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    atol = rtol = 3e-2 if cp > 1 else 2e-2
    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=42,
        sequence_length=512,
        micro_batch_size=2,
        sequence_packing=sequence_packing,
    )
