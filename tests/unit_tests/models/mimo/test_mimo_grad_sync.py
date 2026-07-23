# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real-distributed test for the grad_sync vision partial-participation correction.

The dual-finalize per-token-mean path is validated end-to-end by
test_mimo_colocated_correctness (which wires configure_grad_sync into its
dp1-reference oracle). This file covers the participation-count helper directly
on grid-derived process groups (no parallel_state).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from examples.mimo.training.grad_sync import (
    _vision_participation_count,
    mark_modality_participation,
    reset_modality_participation,
)
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_hypercomm_grid,
    destroy_all_grids,
)
from tests.unit_tests.test_utilities import Utils


class TestVisionParticipation:
    @classmethod
    def setup_class(cls):
        Utils.initialize_distributed()
        cls.world_size = dist.get_world_size()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    def test_vision_participation_correction(self):
        """Partial participation: text-only ranks upscale present ranks.

        With only some DP ranks holding image input, the participation count is
        < dp_size and the correction factor dp_size/participation is applied.
        """
        if self.world_size != 8:
            pytest.skip(f"Requires 8 GPUs, got {self.world_size}")

        grid = create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=self.world_size)
        vision_dp = grid.get_pg("dp")
        dp_size = dist.get_world_size(vision_dp)

        submodule = SimpleNamespace()
        fake_model = SimpleNamespace(modality_submodules={"images": submodule})

        rank = dist.get_rank(vision_dp)
        has_image = rank < dp_size // 2
        batch = (
            {"modality_inputs": {"images": {"hidden_states": torch.ones(1, device="cuda")}}}
            if has_image
            else {"modality_inputs": {}}
        )
        reset_modality_participation(fake_model)
        mark_modality_participation(fake_model, batch)

        count = _vision_participation_count(submodule, vision_dp)
        assert count == float(dp_size // 2)
        factor = dp_size / count
        assert factor == pytest.approx(2.0)

        reset_modality_participation(fake_model)
        assert getattr(submodule, "_mimo_rank_processed_input") is False
