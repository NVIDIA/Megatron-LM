# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import logging
import math
from functools import lru_cache

import torch
from torch import Tensor

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

logger = logging.getLogger(__name__)


class YarnRotaryEmbedding(RotaryEmbedding):
    """Yarn Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from
            transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for
            longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (float, optional): Base period for rotary position embeddings. Defaults to
            10000.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly on
            the GPU. Defaults to False
        scaling_factor (float, optional): Scaling factor for Yarn RoPE. Defaults to 1.0.
        original_max_position_embeddings (int, optional): Original maximum position embeddings
            length. Defaults to 4096.
        beta_fast (float, optional): Fast beta value for Yarn RoPE. Defaults to 32.
        beta_slow (float, optional): Slow beta value for Yarn RoPE. Defaults to 1.
        mscale (float, optional): Mscale value for Yarn RoPE. Defaults to 1.
        mscale_all_dim (float, optional): Mscale all dim value for Yarn RoPE. Defaults to 0.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: float = 10000.0,
        use_cpu_initialization: bool = False,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        self.dim = kv_channels
        self.rotary_base = rotary_base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim

        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()
        self.inv_freq_extra = 1.0 / (
            self.rotary_base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.inv_freq_inter = 1.0 / (
            self.scaling_factor
            * self.rotary_base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        super().__init__(
            kv_channels,
            rotary_percent,
            rotary_interleaved,
            seq_len_interpolation_factor,
            rotary_base,
            use_cpu_initialization,
        )

    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Forward pass of Yarn Rotary Embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying Yarn RoPE.
        """
        assert (
            not self.rotary_interleaved
        ), "Yarn RoPE does not support interleaved rotary embeddings"

        if self.inv_freq_extra.device.type == 'cpu':
            # move `inv_freq_extra` to GPU once at the first micro-batch forward pass
            self.inv_freq_extra = self.inv_freq_extra.to(device=torch.cuda.current_device())

        if self.inv_freq_inter.device.type == 'cpu':
            # move `inv_freq_inter` to GPU once at the first micro-batch forward pass
            self.inv_freq_inter = self.inv_freq_inter.to(device=torch.cuda.current_device())

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.rotary_base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2).to(
            device=self.inv_freq_extra.device, dtype=torch.float32
        )
        inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask

        seq = (
            torch.arange(
                max_seq_len, device=self.inv_freq_extra.device, dtype=self.inv_freq_extra.dtype
            )
            + offset
        )

        freqs = torch.outer(seq, inv_freq)

        _mscale = float(
            _yarn_get_mscale(self.scaling_factor, self.mscale)
            / _yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension
            # and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb, _mscale


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
) -> tuple[int, int]:
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min: float, max: float, dim: int) -> Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0
