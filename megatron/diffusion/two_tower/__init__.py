# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Two-tower Mamba-hybrid model for block-wise diffusion language modelling.

Public API::

    from megatron.diffusion.two_tower import TwoTowerMambaModel, create_block_causal_mask
"""

from megatron.diffusion.two_tower.mamba_model import TwoTowerMambaModel, create_block_causal_mask

__all__ = ["TwoTowerMambaModel", "create_block_causal_mask"]
