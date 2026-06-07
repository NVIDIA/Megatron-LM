# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.rope_utils import get_pos_emb_on_this_cp_rank

__all__ = ['BagelRotaryEmbedding']


class BagelRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling (bool, optional): Apply rope scaling as used in llama 3.x.
        rope_scaling_factor (float, optional): rope scaling factor in llama 3.x. Defaults to 8.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        cp_group (torch.distributed.ProcessGroup, optional): Process group for context parallel.
            Defaults to None.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__(kv_channels, 
                         rotary_percent, 
                         rotary_interleaved, 
                         seq_len_interpolation_factor, 
                         rotary_base, 
                         rope_scaling, 
                         rope_scaling_factor, 
                         use_cpu_initialization, 
                         cp_group,
                         )

    def get_freqs_from_position_ids(self, position_ids: Tensor) -> Tensor:
        """Generates matrix of frequencies based on actual position_ids.

        Args:
            position_ids: Tensor of shape [seq_len] containing actual positions

        Returns:
            freqs: Tensor of shape [seq_len, dim/2]
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        # position_ids: [seq_len]
        # inv_freq: [dim/2]
        # freqs = position_ids @ inv_freq -> [seq_len, dim/2]
        position_ids = position_ids.to(dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            position_ids = position_ids * (1 / self.seq_len_interpolation_factor)

        freqs = torch.outer(position_ids, self.inv_freq)  # [seq_len, dim/2]
        return freqs

    def forward_with_position_ids(
        self,
        position_ids: Tensor,
    ) -> Tensor:
        """Forward pass with actual position_ids.

        Args:
            position_ids: Tensor of shape [seq_len] containing actual positions
            packed_seq: Whether using packed sequence

        Returns:
            Tensor: Rotary embeddings of shape [seq_len, 1, 1, dim]
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        freqs = self.get_freqs_from_position_ids(position_ids)

        # first part even vector components, second part odd vector components
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )

        # emb [seq_length, dim] -> [seq_length, 1, 1, dim]
        emb = emb[:, None, None, :]

        return emb

    def get_cos_sin_from_position_ids(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Get cos and sin based on actual position_ids (qwen2 style).

        Args:
            position_ids: Tensor of shape [seq_len] containing actual positions

        Returns:
            cos: Tensor of shape [seq_len, dim]
            sin: Tensor of shape [seq_len, dim]
        """
        if self.inv_freq.device.type == 'cpu':
            self.inv_freq = self.inv_freq.to(device=position_ids.device)

        freqs = self.get_freqs_from_position_ids(position_ids)

        # Duplicate freqs like qwen2: emb = cat((freqs, freqs), dim=-1)
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )

        cos = emb.cos()
        sin = emb.sin()

        return cos, sin
