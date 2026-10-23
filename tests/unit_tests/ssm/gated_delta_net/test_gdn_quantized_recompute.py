# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net import GatedDeltaNet, KimiDeltaAttention
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


@pytest.fixture
def process_groups():
    Utils.initialize_model_parallel(1, 1)
    try:
        yield ProcessGroupCollection(
            tp=parallel_state.get_tensor_model_parallel_group(),
            cp=parallel_state.get_context_parallel_group(),
        )
    finally:
        Utils.destroy_model_parallel()


def _make_layer(layer_cls, process_groups, precision, recompute, observations):
    # Exercise the production forward/checkpoint boundary without depending on FLA kernels.
    layer = layer_cls.__new__(layer_cls)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(
        fp8="e4m3" if precision.startswith("fp8") or precision == "mxfp8" else None,
        fp4="e2m1" if precision.startswith("fp4") else None,
        linear_cp_mode="headwise",
        sequence_parallel=False,
        distribute_saved_activations=False,
    )
    layer.pg_collection = process_groups
    layer.sp_size = 1
    layer.recompute_gdn = recompute
    layer.projection = te.Linear(
        128, 128, bias=False, params_dtype=torch.bfloat16, init_method=torch.nn.init.xavier_uniform_
    )

    def compute(hidden_states, *unused):
        enabled = FP8GlobalStateManager.is_fp8_enabled()
        observations.append(
            (enabled, type(FP8GlobalStateManager.get_fp8_recipe()) if enabled else None)
        )
        return layer.projection(F.dropout(hidden_states, p=0.25, training=True)), None

    layer._forward_compute = compute
    return layer.cuda()


@pytest.mark.internal
@pytest.mark.parametrize("layer_cls", [GatedDeltaNet, KimiDeltaAttention])
@pytest.mark.parametrize(
    "precision", ["bf16", "fp8", "mxfp8", "fp4", "fp8_disabled", "fp4_disabled"]
)
def test_whole_layer_recompute_preserves_quantization_context(layer_cls, precision, process_groups):
    capability = torch.cuda.get_device_capability()
    if (precision.startswith("fp4") or precision == "mxfp8") and capability < (10, 0):
        pytest.skip("MXFP8 and NVFP4 require Blackwell.")
    if precision.startswith("fp8") and capability < (8, 9):
        pytest.skip("FP8 is not supported on this GPU.")

    if precision == "mxfp8":
        quant_recipe = recipe.MXFP8BlockScaling()
    elif precision.startswith("fp4"):
        # Isolate context/RNG replay from stochastic quantization noise.
        quant_recipe = recipe.NVFP4BlockScaling(disable_stochastic_rounding=True, disable_rht=True)
    else:
        quant_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)
    enabled = precision != "bf16" and not precision.endswith("_disabled")
    reference_observations, recompute_observations = [], []
    reference = _make_layer(layer_cls, process_groups, precision, False, reference_observations)
    recomputed = _make_layer(layer_cls, process_groups, precision, True, recompute_observations)
    with torch.no_grad():
        recomputed.projection.weight.copy_(reference.projection.weight)
    hidden_states = torch.randn(128, 1, 128, device="cuda", dtype=torch.bfloat16)

    def run(layer):
        model_parallel_cuda_manual_seed(123)
        x = hidden_states.detach().clone().requires_grad_(True)
        with te.fp8_autocast(enabled=enabled, fp8_recipe=quant_recipe):
            output, _ = layer(x, None)
        # Recompute after leaving the original quantization context.
        output.float().square().mean().backward()
        return output.detach(), x.grad, layer.projection.weight.grad

    reference_values = run(reference)
    recomputed_values = run(recomputed)
    expected = (enabled, type(quant_recipe) if enabled else None)
    assert reference_observations == [expected]
    assert recompute_observations == [expected, expected]
    torch.testing.assert_close(recomputed_values[0], reference_values[0], atol=0, rtol=0)
    torch.testing.assert_close(recomputed_values[1], reference_values[1], atol=0, rtol=0)
    torch.testing.assert_close(recomputed_values[2], reference_values[2], atol=2e-4, rtol=5e-3)
