# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN parallel correctness in a separate GB200 CI bucket."""

import pytest
import torch.nn.functional as F

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.ssm.gated_delta_net import HAVE_FLA, HAVE_TE_GDN
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.transformer.test_attention import _test_parallel_attention_correctness

pytestmark = pytest.mark.launch_on_gb200


def _have_cudnn_frontend_min_version(min_version: str = "1.29.0") -> bool:
    """Check whether the installed nvidia-cudnn-frontend package meets a minimum version."""
    try:
        from importlib.metadata import version as pkg_version

        from packaging.version import Version

        return Version(pkg_version("nvidia-cudnn-frontend")) >= Version(min_version)
    except Exception:
        return False


@pytest.mark.parametrize("sequence_packing", [False, True])
@pytest.mark.parametrize(
    ("tp", "sp", "cp"),
    [
        (4, False, 1),  # TP w/o SP
        (4, True, 1),  # TP w/ SP
        (1, False, 2),  # CP
        (2, False, 2),  # TP w/o SP + CP
        (2, True, 2),  # TP w/ SP + CP
    ],
)
@pytest.mark.parametrize(
    "gdn_kernel_backend", ["fla", pytest.param("transformer_engine", marks=pytest.mark.internal)]
)
@pytest.mark.parametrize("head_dim", [32, 128])
def test_parallel_gated_delta_net_correctness(
    tmp_path_dist_ckpt, sequence_packing, tp, sp, cp, gdn_kernel_backend, head_dim, run_config
):
    if gdn_kernel_backend == "fla" and not HAVE_FLA:
        pytest.skip("FLA is not installed.")
    if gdn_kernel_backend == "transformer_engine" and not HAVE_TE_GDN:
        pytest.skip("TransformerEngine GDN is not available.")
    if (
        head_dim == 32
        and gdn_kernel_backend == "transformer_engine"
        and not _have_cudnn_frontend_min_version("1.29.0")
    ):
        pytest.skip("nvidia-cudnn-frontend >= 1.29.0 is required for head dim 32.")

    transformer_config = TransformerConfig(
        hidden_size=128,
        linear_conv_kernel_dim=2,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        num_layers=1,
        normalization="RMSNorm",
        use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True,
        num_attention_heads=8,
        activation_func=F.silu,
        bf16=True,
        experimental_attention_variant="gated_delta_net",
        gdn_kernel_backend=gdn_kernel_backend,
        linear_attention_freq=[1],
        transformer_impl="transformer_engine",
    )

    transformer_layer_spec = get_transformer_block_with_experimental_attention_variant_spec(
        config=transformer_config, vp_stage=None, pp_rank=0
    )

    if cp:
        atol, rtol = 5e-3, 5e-3
    else:
        atol, rtol = 5e-4, 5e-4

    _test_parallel_attention_correctness(
        transformer_config=transformer_config,
        transformer_layer_spec=transformer_layer_spec,
        tmp_path_dist_ckpt=tmp_path_dist_ckpt,
        atol=atol,
        rtol=rtol,
        tp=tp,
        sp=sp,
        cp=cp,
        seed=123,
        sequence_length=256,
        micro_batch_size=4,
        sequence_packing=sequence_packing,
    )
