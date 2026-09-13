# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Rotary embedding modules for MLite primitives."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import torch
from torch import Tensor, nn

from megatron.lite.primitive.utils.rope import get_pos_emb_on_this_cp_rank


def _default_rope_device(use_cpu_initialization: bool) -> str | torch.device:
    if use_cpu_initialization or not torch.cuda.is_available():
        return "cpu"
    return torch.device("cuda", torch.cuda.current_device())


class RotaryEmbedding(nn.Module):
    """Rotary embedding with optional context-parallel slicing."""

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float | None = None,
        rotary_base: float = 10000,
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
        device = _default_rope_device(use_cpu_initialization)
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)
        self.cp_group = cp_group

    def _apply_scaling(
        self,
        freqs: Tensor,
        factor: float = 8,
        low_freq_factor: float = 1,
        high_freq_factor: float = 4,
        original_max_position_embeddings: int = 8192,
    ) -> Tensor:
        low_freq_wavelen = original_max_position_embeddings / low_freq_factor
        high_freq_wavelen = original_max_position_embeddings / high_freq_factor

        wavelen = 2 * math.pi / freqs
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)
        smooth_factor = (original_max_position_embeddings / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed_inv_freq = (
            1 - smooth_factor
        ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
        seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )
        if self.seq_len_interpolation_factor is not None:
            seq *= 1 / self.seq_len_interpolation_factor
        return torch.outer(seq, self.inv_freq)

    def get_cos_sin(self, max_seq_len: int, offset: int = 0) -> tuple[Tensor, Tensor]:
        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        return torch.cos(freqs), torch.sin(freqs)

    def get_emb(self, max_seq_len: int, offset: int = 0) -> Tensor:
        if self.inv_freq.device.type == "cpu" and torch.cuda.is_available():
            self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())

        freqs = self.get_freqs_non_repeated(max_seq_len, offset)
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
        return emb[:, None, None, :]

    @lru_cache(maxsize=32)
    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Tensor:
        emb = self.get_emb(max_seq_len, offset)
        if cp_group is None:
            cp_group = self.cp_group
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f"{prefix}inv_freq", None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class YarnRotaryEmbedding(RotaryEmbedding):
    """YARN rotary embedding variant used by MLA-style models."""

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: Optional[float] = None,
        rotary_base: float = 10000.0,
        use_cpu_initialization: bool = False,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        correction_range_round_to_int: bool = True,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.dim = kv_channels
        self.rotary_base = rotary_base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.correction_range_round_to_int = correction_range_round_to_int

        device = _default_rope_device(use_cpu_initialization)
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
            kv_channels=kv_channels,
            rotary_percent=rotary_percent,
            rotary_interleaved=rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base,
            use_cpu_initialization=use_cpu_initialization,
            cp_group=cp_group,
        )
        self._set_cos_sin_cache(
            self.original_max_position_embeddings, offset=0, dtype=torch.get_default_dtype()
        )
        self.forward.cache_clear()

    def get_emb(self, max_seq_len: int, offset: int = 0) -> tuple[Tensor, float]:
        if self.rotary_interleaved:
            raise AssertionError("YARN RoPE does not support interleaved rotary embeddings")
        if self.inv_freq_extra.device.type == "cpu" and torch.cuda.is_available():
            self.inv_freq_extra = self.inv_freq_extra.to(device=torch.cuda.current_device())
        if self.inv_freq_inter.device.type == "cpu" and torch.cuda.is_available():
            self.inv_freq_inter = self.inv_freq_inter.to(device=torch.cuda.current_device())

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.dim,
            self.rotary_base,
            self.original_max_position_embeddings,
            self.correction_range_round_to_int,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(
            low, high, self.dim // 2, device=self.inv_freq_extra.device
        ).to(dtype=torch.float32)
        inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask
        seq = (
            torch.arange(
                max_seq_len, device=self.inv_freq_extra.device, dtype=self.inv_freq_extra.dtype
            )
            + offset
        )
        freqs = torch.outer(seq, inv_freq)
        concentration = _yarn_get_concentration_factor(
            self.scaling_factor, self.mscale, self.mscale_all_dim
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[:, None, None, :], concentration

    @lru_cache(maxsize=32)
    def forward(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[Tensor, float]:
        emb, concentration = self.get_emb(max_seq_len, offset)
        if cp_group is None:
            cp_group = self.cp_group
        if cp_group is not None and cp_group.size() > 1 and not packed_seq:
            emb = get_pos_emb_on_this_cp_rank(emb, 0, cp_group)
        return emb, concentration

    def _set_cos_sin_cache(self, seq_len, offset, dtype, packed_seq=False, cp_group=None):
        self.max_seq_len_cached = seq_len
        self.offset_cached = offset
        self.dtype_cached = dtype
        self.packed_seq_cached = packed_seq
        emb, concentration = self.forward(seq_len, offset, packed_seq=packed_seq, cp_group=cp_group)
        self.register_buffer(
            "cos_cached", (emb.cos() * concentration).to(dtype).contiguous(), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * concentration).to(dtype).contiguous(), persistent=False
        )

    def get_cached_cos_sin(
        self, seq_len, offset=0, dtype=torch.get_default_dtype(), packed_seq=False, cp_group=None
    ):
        if (
            seq_len > self.max_seq_len_cached
            or offset != self.offset_cached
            or dtype != self.dtype_cached
            or packed_seq != self.packed_seq_cached
        ):
            self._set_cos_sin_cache(seq_len, offset, dtype, packed_seq, cp_group)
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, rotary_base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(rotary_base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    rotary_base: float = 10000,
    max_position_embeddings: int = 2048,
    round_to_int: bool = True,
) -> tuple[int, int]:
    low = _yarn_find_correction_dim(low_rot, dim, rotary_base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, rotary_base, max_position_embeddings)
    if round_to_int:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(
    minimum: float, maximum: float, dim: int, device: torch.device
) -> Tensor:
    if minimum == maximum:
        maximum += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32, device=device) - minimum) / (
        maximum - minimum
    )
    return torch.clamp(linear_func, 0, 1)


def _yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


@lru_cache(maxsize=8)
def _yarn_get_concentration_factor(
    scaling_factor: float, mscale: Optional[float], mscale_all_dim: Optional[float]
) -> float:
    if mscale is None or mscale_all_dim is None:
        return _yarn_get_mscale(scaling_factor)
    return float(
        _yarn_get_mscale(scaling_factor, mscale) / _yarn_get_mscale(scaling_factor, mscale_all_dim)
    )


__all__ = ["RotaryEmbedding", "YarnRotaryEmbedding", "_yarn_get_mscale"]
