# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib.util

import torch
from torch import einsum, nn

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, seq_len_interpolation_factor=None):
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, offset=0):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb[:, None, None, :]


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
