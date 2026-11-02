# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Layer-level full recompute over four 1F1B microbatches, two-chunk parity.

Split out of ``test_schedule_chunk_1f1b.py`` so that each CI bucket of the chunk-schedule
parity matrix fits its time budget (the original file had grown to 189 cases, about an
hour on one H100 bucket).
"""

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
    @pytest.mark.parametrize(
        "recompute_method,recompute_num_layers", [("uniform", 1), ("block", 1)]
    )
    def test_1f1b_schedule_model_chunk_full_recompute_multi_microbatch(
        self,
        mtp_layers,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        recompute_method,
        recompute_num_layers,
    ):
        """Same parity, but over four microbatches, as in a real 1F1B steady state.

        One microbatch never runs the same modules through forward -> recompute ->
        backward twice, so it cannot catch state that the replay leaves behind and the
        next microbatch then reads: FP8 scaling metadata, the segments' RNG snapshots, or
        a node whose ``forward_no_grad`` was not restored. Grads are accumulated across
        the four microbatches on both paths and compared bitwise, so any per-round drift
        shows up. The segmentation axis is trimmed to one segment-per-layer case and one
        recomputed -> eager hand-off case; the full matrix in
        ``test_schedule_chunk_1f1b_recompute_{uniform,block}.py`` covers the rest at
        ``microbatches=1``.
        """
        run_full_recompute_case(
            mtp_layers,
            dispatcher_type,
            flex_backend,
            fp8_flag,
            recompute_method,
            recompute_num_layers,
            microbatches=4,
        )
