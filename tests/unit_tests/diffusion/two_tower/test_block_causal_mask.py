# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for :func:`create_block_causal_mask`.

Verify the element-wise attend / mask pattern for various block counts
and block sizes.  An incorrect mask would silently leak future context
into earlier denoiser blocks, corrupting the training objective.
"""

import torch

from megatron.diffusion.two_tower.mamba_model import create_block_causal_mask


class TestBlockCausalMask:
    """Element-wise correctness tests for :func:`create_block_causal_mask`."""

    def test_single_block(self):
        """With one block there is no past context; only the denoiser self-block is attended."""
        mask = create_block_causal_mask(1, 2, "cpu", torch.float32)
        mask_2d = mask.squeeze(0).squeeze(0)
        # Context columns (0:2) should be fully masked
        assert (mask_2d[:, :2] < -1e30).all()
        # Denoiser columns (2:4) should be fully attended (block 0)
        assert (mask_2d[:, 2:4] == 0.0).all()

    def test_two_blocks_structure(self):
        """Block 0 sees only its own denoiser KV; block 1 sees context block 0 plus self."""
        mask = create_block_causal_mask(2, 1, "cpu", torch.float32)
        mask_2d = mask.squeeze(0).squeeze(0)
        # mask_2d shape: [2, 4] (q=2, kv=4=[context(2)|denoiser(2)])
        # Block 0 (row 0): context[0] masked, context[1] masked, denoiser[0] attend, denoiser[1] masked
        assert mask_2d[0, 0] < -1e30  # context block 0: not past
        assert mask_2d[0, 1] < -1e30  # context block 1: not past
        assert mask_2d[0, 2] == 0.0  # denoiser block 0: self
        assert mask_2d[0, 3] < -1e30  # denoiser block 1: not self

        # Block 1 (row 1): context[0] attend, context[1] masked, denoiser[0] masked, denoiser[1] attend
        assert mask_2d[1, 0] == 0.0  # context block 0: past
        assert mask_2d[1, 1] < -1e30  # context block 1: not past (equal, strict <)
        assert mask_2d[1, 2] < -1e30  # denoiser block 0: not self
        assert mask_2d[1, 3] == 0.0  # denoiser block 1: self

    def test_four_blocks_block_size_two(self):
        """Four blocks of size 2: full element-wise check at token granularity."""
        mask = create_block_causal_mask(4, 2, "cpu", torch.float32)
        mask_2d = mask.squeeze(0).squeeze(0)
        S = 8
        # Row 0,1 (block 0): only denoiser block 0 columns attended
        for q in range(2):
            for kv in range(2 * S):
                if 8 <= kv < 10:  # denoiser block 0
                    assert mask_2d[q, kv] == 0.0, f"q={q},kv={kv} should attend"
                else:
                    assert mask_2d[q, kv] < -1e30, f"q={q},kv={kv} should be masked"

        # Row 6,7 (block 3): context blocks 0,1,2 + denoiser block 3
        for q in range(6, 8):
            for kv in range(2 * S):
                if kv < 6:  # context blocks 0,1,2
                    assert mask_2d[q, kv] == 0.0, f"q={q},kv={kv} should attend (past ctx)"
                elif 6 <= kv < 8:  # context block 3 (same block, not past)
                    assert mask_2d[q, kv] < -1e30, f"q={q},kv={kv} should be masked"
                elif 14 <= kv < 16:  # denoiser block 3
                    assert mask_2d[q, kv] == 0.0, f"q={q},kv={kv} should attend (self)"
                else:
                    assert mask_2d[q, kv] < -1e30, f"q={q},kv={kv} should be masked"
