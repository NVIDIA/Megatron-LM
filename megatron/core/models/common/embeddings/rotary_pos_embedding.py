# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.transformer_block import TransformerBlock

import torch
from torch import Tensor, einsum, nn

from megatron.core import parallel_state

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


def get_pos_emb_on_this_cp_rank(pos_emb, seq_dim):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_idx = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=pos_emb.device)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


class RotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels (int): Projection weights dimension in multi-head attention. Obtained from transformer config
        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
        seq_len_interpolation_factor (float, optional): scale of linearly interpolating RoPE for longer sequences. The value must be a float larger than 1.0. Defaults to None
    """

    def __init__(
        self, kv_channels: int, rotary_percent: float, seq_len_interpolation_factor: float = None
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, max_seq_len: int, offset: int = 0) -> Tensor:
        """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): _description_. Defaults to 0.

        Returns:
            Tensor: Embeddings after applying RoPE.
        """
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        emb = emb[:, None, None, :]
        if parallel_state.get_context_parallel_world_size() > 1:
            # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
            emb = get_pos_emb_on_this_cp_rank(emb, 0)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def get_rotary_seq_len(
        self,
        inference_params,
        transformer: TransformerBlock,
        transformer_input: Tensor,
        transformer_config: TransformerConfig,
    ) -> float:
        """Function to get the rotary sequence length.

        Args:
            inference_params : Used during Inference time
            transformer (TransformerBlock): The transformer block (decoder/encoder) used by the model
            transformer_input (Tensor): _description_
            transformer_config (TransformerConfig): Transformer config used by the model

        Returns:
            float: The rotary sequence length
        """
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            if transformer.input_tensor is not None:
                rotary_seq_len = transformer.input_tensor.size(0)
            else:
                rotary_seq_len = transformer_input.size(0)

            if transformer_config.sequence_parallel:
                rotary_seq_len *= transformer_config.tensor_model_parallel_size

        rotary_seq_len *= transformer_config.context_parallel_size

        return rotary_seq_len


def _rotate_half(x: Tensor) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """

    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
