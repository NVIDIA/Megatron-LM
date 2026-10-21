# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Block full-recompute chunk schedules; retains the original per-case setup and assertions."""

import pytest

from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_chunk_1f1b import (
    _A2AOverlapTestBase,
    run_full_recompute_case,
)
from tests.unit_tests.a2a_overlap.utils import get_valid_dispatcher_configs, get_valid_fp8_flags


class TestA2AOverlap(_A2AOverlapTestBase):
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("mtp_layers", [0, 1])
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize('recompute_method,recompute_num_layers', [('block', 1), ('block', 3)])
    def test_1f1b_schedule_model_chunk_full_recompute(
        self,
        mtp_layers,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        recompute_method,
        recompute_num_layers,
    ):
        run_full_recompute_case(
            mtp_layers,
            dispatcher_type,
            flex_backend,
            fp8_flag,
            recompute_method,
            recompute_num_layers,
        )
