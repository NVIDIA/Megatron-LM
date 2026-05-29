# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for GPTModel.

Adding a new parallelism cell is a one-line append to
``configs.PARALLELISM_CONFIGS`` — this file does not need to change.
"""

import pytest

from tests.unit_tests.determinism.configs import GPT_CONFIGS, PARALLELISM_CONFIGS
from tests.unit_tests.determinism.correctness._gpt_shared import make_gpt_runner

RUNNER = make_gpt_runner(supports_pp=True)


class TestGPTModelDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", GPT_CONFIGS)
    def test_bit_exact_under_parallelism(self, cfg_overrides, parallelism):
        RUNNER.run(cfg_overrides, parallelism)
