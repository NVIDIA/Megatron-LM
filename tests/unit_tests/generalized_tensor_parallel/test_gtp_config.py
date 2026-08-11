# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU-only tests for GTP runtime configuration."""

import pytest

import megatron.core.tensor_parallel.generalized_tensor_parallelism as gtp_module


@pytest.fixture(autouse=True)
def _restore_gtp_config():
    original = {
        "pad_for_alignment": gtp_module.GTP_CONFIG.pad_for_alignment,
        "check_param_states": gtp_module.GTP_CONFIG.check_param_states,
        "weight_prefetch": gtp_module.GTP_CONFIG.weight_prefetch,
        "async_reduction": gtp_module.GTP_CONFIG.async_reduction,
        "calculate_per_token_loss": gtp_module.GTP_CONFIG.calculate_per_token_loss,
    }
    try:
        yield
    finally:
        gtp_module.update_gtp_config(**original)


@pytest.mark.parametrize(
    "recipe_kwargs,expected_pad",
    [
        pytest.param({"fp4": True}, 16, id="fp4"),
        pytest.param({"fp8_recipe": "mxfp8"}, 32, id="mxfp8"),
        pytest.param({"fp8": True}, 16, id="fp8"),
    ],
)
def test_recipe_defaults_set_pad_for_alignment(recipe_kwargs, expected_pad):
    """With no explicit value, each quantized recipe keeps its historical alignment pad."""
    gtp_module.configure_gtp_remat_from_recipe(**recipe_kwargs)
    assert gtp_module.GTP_CONFIG.pad_for_alignment == expected_pad


def test_bf16_recipe_leaves_pad_for_alignment_untouched():
    """No quantized branch fires for bf16; the pre-existing value must survive."""
    gtp_module.update_gtp_config(pad_for_alignment=48)
    gtp_module.configure_gtp_remat_from_recipe()
    assert gtp_module.GTP_CONFIG.pad_for_alignment == 48


@pytest.mark.parametrize(
    "recipe_kwargs",
    [
        pytest.param({}, id="bf16"),
        pytest.param({"fp4": True}, id="fp4"),
        pytest.param({"fp8_recipe": "mxfp8"}, id="mxfp8"),
    ],
)
def test_explicit_pad_for_alignment_wins_over_every_recipe(recipe_kwargs):
    """--gtp-remat-pad-for-alignment pins the pad; 0 disables padding entirely."""
    gtp_module.configure_gtp_remat_from_recipe(pad_for_alignment=0, **recipe_kwargs)
    assert gtp_module.GTP_CONFIG.pad_for_alignment == 0
