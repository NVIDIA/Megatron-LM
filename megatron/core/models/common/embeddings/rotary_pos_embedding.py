# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock
    from megatron.core.inference.contexts import BaseInferenceContext
    from megatron.core.packed_seq_params import PackedSeqParams

import logging
import math
from functools import lru_cache

import torch
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (  # for backward compatibility; pylint: disable=unused-import
    _apply_rotary_pos_emb_bshd,
    _apply_rotary_pos_emb_thd,
    _rotate_half,
    apply_rotary_pos_emb,
    get_pos_emb_on_this_cp_rank,
)
from megatron.core.utils import deprecate_inference_params

logger = logging.getLogger(__name__)


__all__ = ['RotaryEmbedding', 'MultimodalRotaryEmbedding']


class RotaryEmbedding(nn.Module):
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
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        device = 'cpu' if use_cpu_initialization else torch.cuda.current_device()
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)

        self.cp_group = (
            cp_group
            if cp_group is not None
            else parallel_state.get_context_parallel_group(check_initialized=False)
        )

    def _apply_scaling(
        self,
        freqs,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    ):
        # This implementation is adapted from:
        # https://github.com/huggingface/transformers/blob/2a5a6ad18aa22e98429bb5ecb880660328030ea0/src/transformers/modeling_rope_utils.py#L303-L343

        factor = factor  # `8` in the original implementation
        low_freq_factor = low_freq_factor  # `1` in the original implementation
        high_freq_factor = high_freq_factor  # `4` in the original implementation
        old_context_len = original_max_position_embeddings  # `8192` in the original implementation

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / freqs
        # wavelen < high_freq_wavelen: do nothing
        # wavelen > low_freq_wavelen: divide by factor
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        # otherwise: interpolate between the two, using a smooth factor
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        return inv_freq_llama

    def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Generates matrix of frequencies based on positions in the sequence,
        used to create positional encodings"""
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        freqs = torch.outer(seq, self.inv_freq)  # [seq len, dim]

        return freqs

    def get_cos_sin(self, max_seq_len: int, offset: int = 0) -> (Tensor, Tensor):
        """Cosine and sine values for RoPE are precomputed for all positions up to the maximum
        sequence length"""
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

    @lru_cache(maxsize=32)
    def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        if self.inv_freq.device.type == 'cpu':
            # move `inv_freq` to GPU once at the first micro-batch forward pass
            self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())

        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if self.cp_group is not None and self.cp_group.size() > 1 and not packed_seq:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        inference_context: BaseInferenceContext,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
        packed_seq_params: PackedSeqParams,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_context : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used
                by the model
            transformer_input (Tensor): Input tensor to the transformer
            transformer_config (TransformerConfig): Transformer config used by the model
            packed_seq_params (PackedSeqParams): Packed sequence params

        Returns:
            float: The rotary sequence length
        """

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if packed_seq_params is not None:
            # max_seqlen are the max sequence length in the packed sequence before being divived
            # by the tp and cp size.
            return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)
        elif inference_context is not None:
            rotary_seq_len = inference_context.max_sequence_length
        else:
            if transformer is not None and transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


class MultimodalRotaryEmbedding(nn.Module):
    """Multimodal Rotary Embedding for language model.
    Based on https://github.com/alibaba/Pai-Megatron-Patch/blob/
    efa5a752e845267936db9ae7df1b6aba92e9ff9a/megatron_patch/model/qwen2_vl/rotary_pos_embedding.py
    Copyright (c) 2025 alibaba/Pai-Megatron-Patch. Apache 2.0 license.

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
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: int = 10000,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)
        self.rotary_interleaved = rotary_interleaved

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.inv_freq = 1.0 / (
            rotary_base
            ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=torch.cuda.current_device())
                / dim
            )
        )

    def forward(self, position_ids: torch.Tensor, mrope_section: List[int]) -> Tensor:
        """Forward pass of multimodal RoPE embedding.

        Args:
            position_ids (torch.Tensor): A postion_id tensor with shape [3, batchsize, seqlens]
            mrope_section (list[int]): Multimodal rope section is for channel dimension of temporal,
                height and width in rope calculation.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = position_ids.to(device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor

        # shape (3, bs, dim, 1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, seq.shape[1], -1, 1)
        # shape (3, bs, 1, seq_length)
        seq_expanded = seq[:, :, None, :].float()
        # shape (3, bs, seq_length, dim)
        freqs = (inv_freq_expanded @ seq_expanded).transpose(2, 3)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)  # shape (3, bs, seq_length, 2 * dim)
        else:
            bs = freqs.shape[1]
            emb = torch.stack((freqs.view(3, bs, -1, 1), freqs.view(3, bs, -1, 1)), dim=-1).view(
                3, bs, freqs.shape[0], -1
            )

        # generate freqs with mrope_section
        # shape (bs, seq_length, 2 * dim)
        mrope_section = mrope_section * 2
        emb = torch.cat([m[i % 3] for i, m in enumerate(emb.split(mrope_section, dim=-1))], dim=-1)

        # shape (seq_length, bs, 1, 2 * dim)
        emb = emb[..., None, :].transpose(0, 1).contiguous()
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current
            # CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 1)
        return emb
