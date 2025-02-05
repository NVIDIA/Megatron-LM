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


class LongRoPERotaryEmbedding(RotaryEmbedding):
    """LongRoPE Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained
            from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position
            embeddings.
        short_factor (list[int]): RoPE factors used when current seq_len <= original_seq_len.
        long_factor (list[int]): RoPE factors used when current seq_len > original_seq_len.
        rotary_interleaved (bool, optional): If True, interleaved rotary position embeddings.
            Defaults to False.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE
            for longer sequences. The value must be a float larger than 1.0. Defaults to None
        rotary_base (int, optional): Base period for rotary position embeddings. Defaults to
            10000.
        rope_scaling_factor (float, optional): rope scaling factor in llama 3.x. Defaults to 8.
        use_cpu_initialization (bool, optional): If False, initialize the inv_freq directly
            on the GPU. Defaults to False
        original_max_position_embeddings (int, optional): Original maximum position embeddings
            length. Defaults to 4096.
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        short_factor: list[int],
        long_factor: list[int],
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        rope_scaling_factor: float = 8.0,
        use_cpu_initialization: bool = False,
        original_max_position_embeddings: int = 4096,
    ) -> None:
        super().__init__(
            kv_channels,
            rotary_percent,
            rotary_interleaved,
            seq_len_interpolation_factor,
            rotary_base,
            use_cpu_initialization,
        )
        self.dim = kv_channels
        self.rotary_base = rotary_base
        self.original_max_position_embeddings = original_max_position_embeddings

        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()
        self.inf_freq_short, self.inv_freq_long, self._mscale = self._compute_longrope_parameters(
            self, 
            device,
            rope_scaling_factor,
            original_max_position_embeddings,
            rotary_base,
            short_factor,
            long_factor
        )
    
    def _compute_longrope_parameters(
            self, 
            device,
            rope_scaling_factor,
            original_max_position_embeddings,
            rotary_base,
            short_factor,
            long_factor
        ):

        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/c772bff31a65c9c6002d0e74797cb130959a3716/src/transformers/modeling_rope_utils.py#L242

        if rope_scaling_factor <= 1.0:
            attention_factor = 1.0
        else:
            attention_factor = math.sqrt(1 + math.log(rope_scaling_factor) / math.log(original_max_position_embeddings))

        # Compute the inverse frequencies -- scaled based on the target sequence length
        inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64, device=device).float() / self.dim

        ext_factors_short = torch.tensor(short_factor, dtype=torch.float32, device=device)
        inv_freq_short = 1.0 / (ext_factors_short * rotary_base**inv_freq_shape)

        ext_factors_long = torch.tensor(long_factor, dtype=torch.float32, device=device)
        inv_freq_long = 1.0 / (ext_factors_long * rotary_base**inv_freq_shape)

        return inv_freq_short, inv_freq_long, attention_factor


    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Forward pass of LongRoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

        Returns:
            Tensor: Embeddings after applying Long RoPE.
        """
        if max_seq_len < self.original_max_position_embeddings:
            self.inv_freq = self.inv_freq_short
        else:
            self.inv_freq = self.inv_freq_long

        return super().forward(self, max_seq_len, offset, packed_seq)
