# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.fp8_param_rounding import (
    E4M3_MAX,
    _blockwise_scales,
    _e4m3_cell,
    _threshold_offsets,
    fixed_threshold_e4m3,
    fp8_master_census,
    snap_master_shard,
)

BLOCK = 128


def _nearest_cast(flat, scale, rows, cols, start, block_len=BLOCK):
    """CPU stand-in for tex.fp8_block_scaling_partial_cast: round master * scale to nearest E4M3."""
    index = torch.arange(start, start + flat.numel())
    s = scale[(index // cols) // block_len, (index % cols) // block_len]
    return (flat.float() * s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def _full(scale, rows, cols):
    return scale.repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)[:, :cols]


@pytest.fixture
def bf16_master():
    torch.manual_seed(0)
    return (torch.randn(300, 520) * 0.02).to(torch.bfloat16).float()


class TestFixedThresholdRounding:
    def test_snap_then_nearest_is_fixed_threshold(self, bf16_master):
        rows, cols = bf16_master.shape
        scale = _blockwise_scales(bf16_master, BLOCK, power_of_two=True)
        offsets = _threshold_offsets(7, rows, cols, torch.arange(rows * cols)).view(rows, cols)
        expected = fixed_threshold_e4m3(bf16_master * _full(scale, rows, cols), offsets)
        snapped = snap_master_shard(bf16_master.reshape(-1), scale, rows, cols, 0, BLOCK, seed=7)
        assert torch.equal(_nearest_cast(snapped, scale, rows, cols, 0).view(rows, cols), expected)

    def test_shards_agree_with_full_cast(self, bf16_master):
        rows, cols = bf16_master.shape
        scale = _blockwise_scales(bf16_master, BLOCK, power_of_two=True)
        flat, cut = bf16_master.reshape(-1), 77_777
        full = _nearest_cast(
            snap_master_shard(flat, scale, rows, cols, 0, BLOCK, 7), scale, rows, cols, 0
        )
        first = _nearest_cast(
            snap_master_shard(flat[:cut], scale, rows, cols, 0, BLOCK, 7), scale, rows, cols, 0
        )
        second = _nearest_cast(
            snap_master_shard(flat[cut:], scale, rows, cols, cut, BLOCK, 7), scale, rows, cols, cut
        )
        assert torch.equal(torch.cat([first, second]), full)

    def test_grid_points_are_fixed(self, bf16_master):
        rows, cols = bf16_master.shape
        scale = _blockwise_scales(bf16_master, BLOCK, power_of_two=True)
        on_grid = _nearest_cast(bf16_master.reshape(-1), scale, rows, cols, 0) / _full(
            scale, rows, cols
        ).reshape(-1)
        snapped = snap_master_shard(on_grid, scale, rows, cols, 0, BLOCK, 7)
        assert torch.equal(snapped, on_grid)

    def test_unbiased_over_seeds(self):
        torch.manual_seed(1)
        master = torch.randn(64, 128) * 0.02
        scale = _blockwise_scales(master, BLOCK, power_of_two=True)
        full = _full(scale, 64, 128)
        mean = torch.stack(
            [
                _nearest_cast(
                    snap_master_shard(master.reshape(-1), scale, 64, 128, 0, BLOCK, seed),
                    scale,
                    64,
                    128,
                    0,
                ).view(64, 128)
                / full
                for seed in range(400)
            ]
        ).mean(0)
        relative_bias = ((mean - master).abs() / master.abs().clamp_min(1e-6)).median()
        assert relative_bias < 0.01

    def test_small_drift_changes_codes_in_proportion(self):
        """From a grid start, round to nearest moves no code for drift below half a cell; fixed threshold moves
        about drift / cell of them."""
        torch.manual_seed(2)
        master = torch.randn(256, 256) * 0.02
        scale = _blockwise_scales(master, BLOCK, power_of_two=True)
        on_grid = _nearest_cast(master.reshape(-1), scale, 256, 256, 0)
        start = on_grid / _full(scale, 256, 256).reshape(-1)
        _, width = _e4m3_cell(on_grid.abs())
        drifted = start + torch.sign(start) * 0.1 * width / _full(scale, 256, 256).reshape(-1)
        nearest_changed = (_nearest_cast(drifted, scale, 256, 256, 0) != on_grid).float().mean()
        fixed = _nearest_cast(
            snap_master_shard(drifted, scale, 256, 256, 0, BLOCK, 3), scale, 256, 256, 0
        )
        fixed_changed = (fixed != on_grid).float().mean()
        assert nearest_changed == 0
        assert 0.07 < fixed_changed < 0.13

    def test_census(self, bf16_master):
        rows, cols = bf16_master.shape
        scale = _blockwise_scales(bf16_master, BLOCK, power_of_two=True)
        grid_master = _nearest_cast(bf16_master.reshape(-1), scale, rows, cols, 0).view(
            rows, cols
        ) / _full(scale, rows, cols)
        assert 0.05 < fp8_master_census([bf16_master])["on_tie"] < 0.075
        assert fp8_master_census([bf16_master], power_of_two=False)["on_tie"] < 0.02
        assert fp8_master_census([grid_master])["on_grid"] > 0.99

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="blockwise FP8 needs Hopper or newer",
    )
    def test_matches_te_kernel(self, bf16_master):
        """The snapped shard through TE's own kernel gives the same E4M3 bytes as the CPU reference."""
        import transformer_engine.pytorch.tensor.utils as te_tensor_utils
        from transformer_engine.pytorch.constants import TE_DType

        tex = te_tensor_utils.tex

        master = bf16_master.cuda()
        rows, cols = master.shape
        scale = _blockwise_scales(master, BLOCK, power_of_two=True).cuda()
        out = torch.empty(
            rows * cols, dtype=torch.uint8, device="cuda"
        )  # whole tensor, start offset 0
        snapped = snap_master_shard(master.reshape(-1), scale, rows, cols, 0, BLOCK, seed=7)
        tex.fp8_block_scaling_partial_cast(
            snapped, out, scale.contiguous(), rows, cols, 0, BLOCK, TE_DType[torch.float8_e4m3fn]
        )
        te_values = out.view(torch.float8_e4m3fn).float().cpu()
        reference = _nearest_cast(snapped.cpu(), scale.cpu(), rows, cols, 0)
        assert torch.equal(te_values, reference)
