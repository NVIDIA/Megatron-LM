# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Padding-mask chunk schedules with the original TP/EP matrix."""

import pytest

from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.test_schedule_chunk_1f1b import (
    _A2AOverlapTestBase,
    run_two_chunk_parity,
)
from tests.unit_tests.a2a_overlap.utils import (
    apply_flex_backend_kwargs,
    get_valid_dispatcher_configs,
)
from tests.unit_tests.test_utilities import Utils


class TestA2AOverlap(_A2AOverlapTestBase):
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("layers", [[2, 1], [1, 1]])
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_1f1b_schedule_model_chunk_with_padding_mask(
        self, dispatcher_type, flex_backend, layers, tp_size
    ):
        """
        Verifies all-to-all overlap optimization with padding_mask produces
        the same results as the reference implementation with various TP/EP/CP combinations.
        """
        # Re-initialize model parallel with the specified configuration
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
        )
        set_streams()

        # create TransformerConfig
        extra_kwargs = {"tensor_model_parallel_size": tp_size, "sequence_parallel": tp_size > 1}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        run_two_chunk_parity(layers, extra_kwargs, use_padding_mask=True)
