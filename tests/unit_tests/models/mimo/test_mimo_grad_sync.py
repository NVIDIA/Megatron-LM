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
    _token_normalization_scale,
    _vision_participation_count,
    _vision_participation_scale,
    configure_grad_sync,
    mark_modality_participation,
    reset_modality_participation,
)
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from tests.unit_tests.models.mimo.test_mimo_1f1b_schedule import (
    create_hypercomm_grid,
    destroy_all_grids,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(("num_tokens", "expected"), [(0.0, 1.0), (4.0, 0.25)])
def test_token_normalization_scale(num_tokens, expected):
    scale = _token_normalization_scale(torch.tensor(num_tokens))

    assert scale.ndim == 0
    torch.testing.assert_close(scale, torch.tensor(expected))


@pytest.mark.parametrize(("participation", "expected"), [(0.0, 1.0), (4.0, 2.0), (8.0, 1.0)])
def test_vision_participation_scale(participation, expected):
    scale = _vision_participation_scale(torch.tensor(participation), vision_dp_size=8)

    assert scale.ndim == 0
    torch.testing.assert_close(scale, torch.tensor(expected))


def test_configure_grad_sync_installs_production_overlap_hooks():
    def no_sync():
        return None

    def start_grad_sync(*_unused):
        return None

    config = SimpleNamespace(no_sync_func=None, grad_sync_func=None)
    model = SimpleNamespace(
        config=config,
        no_sync=no_sync,
        start_grad_sync=start_grad_sync,
        language_model=None,
        modality_submodules={},
    )
    language_grid = SimpleNamespace(get_rank_enum=lambda _name: [[0]])
    topology = SimpleNamespace(grids={MIMO_LANGUAGE_MODULE_KEY: language_grid}, module_pgs={})

    configure_grad_sync(
        SimpleNamespace(overlap_grad_reduce=True, align_grad_reduce=True), model, topology
    )

    assert config.no_sync_func is no_sync
    assert config.grad_sync_func is start_grad_sync


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

    def test_scalar_fp32_scale_applies_to_bf16_gradients(self):
        grad = torch.ones(4, dtype=torch.bfloat16, device="cuda")
        scale = _token_normalization_scale(torch.tensor(2.0, dtype=torch.float32, device="cuda"))

        grad.mul_(scale)

        assert scale.ndim == 0
        torch.testing.assert_close(grad, torch.full_like(grad, 0.5))

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
        assert count.ndim == 0
        torch.testing.assert_close(
            count, torch.tensor(float(dp_size // 2), dtype=count.dtype, device=count.device)
        )
        factor = dp_size / count
        torch.testing.assert_close(
            factor, torch.tensor(2.0, dtype=factor.dtype, device=factor.device)
        )

        reset_modality_participation(fake_model)
        assert getattr(submodule, "_mimo_rank_processed_input") is False
