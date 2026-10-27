# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""FP8 / FP4 quantization-recipe determinism check.

Four recipes, all at TP=2 (representative composite — quantization failure
modes don't depend on parallelism degree):

* ``fp8-tensorwise`` — per-tensor current scaling; amax recomputed every step.
* ``fp8-delayed``    — TE default, scale derived from amax history. Requires
                       the runner's ``_reset_quantizer_state`` between runs
                       A and B (per-module ``fp8_meta`` carries amax across
                       forward passes).
* ``fp8-mxfp8``      — Blackwell-only microscaling FP8; capability-skipped on Hopper.
* ``fp4-nvfp4``      — Blackwell-only NVFP4 block scaling; capability-skipped on Hopper.
"""

import inspect

import pytest
import torch

from tests.unit_tests.determinism.correctness.test_gpt_model import make_gpt_runner
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

# Hopper = SM 9.0, Blackwell = SM 10.0+. mxfp8 + nvfp4 need Blackwell.
_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10

RUNNER = make_gpt_runner(supports_pp=False)

_QUANT_RECIPES = [
    pytest.param({"fp8": "hybrid", "fp8_recipe": "tensorwise"}, id="fp8-tensorwise"),
    pytest.param({"fp8": "hybrid", "fp8_recipe": "delayed"}, id="fp8-delayed"),
    pytest.param(
        {"fp8": "hybrid", "fp8_recipe": "mxfp8"},
        id="fp8-mxfp8",
        marks=pytest.mark.skipif(not _IS_BLACKWELL, reason="mxfp8 requires Blackwell"),
    ),
    pytest.param(
        {"fp4": "e2m1", "fp4_recipe": "nvfp4"},
        id="fp4-nvfp4",
        marks=pytest.mark.skipif(not _IS_BLACKWELL, reason="nvfp4 requires Blackwell"),
    ),
]


class TestQuantizationDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("quant_overrides", _QUANT_RECIPES)
    def test_bit_exact_under_quantization(self, quant_overrides):
        RUNNER.run(quant_overrides, {"TP": 2})

    @pytest.mark.internal
    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(not _IS_BLACKWELL, reason="grouped MXFP8 storage needs Blackwell")
    def test_grouped_mxfp8_parameter_updates_replay_bit_exact(self, monkeypatch):
        """Refit-style writes must update cached grouped members deterministically."""
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import MXFP8BlockScaling

        from megatron.core.fp8_utils import (
            copy_tensor_to_quantized_param,
            get_grouped_quantized_members,
        )

        if "single_grouped_weight" not in inspect.signature(te.GroupedLinear).parameters:
            pytest.skip("Transformer Engine lacks single grouped parameters")

        monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
        with te.fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
            linear = te.GroupedLinear(
                2,
                128,
                64,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                single_grouped_weight=True,
            )

        members = get_grouped_quantized_members(linear.weight, create_if_missing=True)
        source = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)

        def update(source):
            copy_tensor_to_quantized_param(linear.weight, source)
            return tuple((member._rowwise_data, member._rowwise_scale_inv) for member in members)

        assert_replays_bit_exact(
            update,
            (source,),
            replays=3,
            contention=True,
            backward=False,
            what="grouped MXFP8 parameter update",
        )
