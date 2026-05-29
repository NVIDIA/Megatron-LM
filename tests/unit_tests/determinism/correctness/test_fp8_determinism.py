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

import pytest
import torch

from tests.unit_tests.determinism.correctness._gpt_shared import make_gpt_runner

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
