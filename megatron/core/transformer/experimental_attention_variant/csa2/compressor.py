# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DeepSeek-V4.1 KV compressor.

A KV-source layer turns every group of ``compress_ratio`` consecutive tokens into one
latent of width ``head_dim``:

* ``compress_ratio == 1``: a plain projection followed by RMSNorm (checkpoint dtype).
* ``compress_ratio > 1``: two fp32 projections; the second one gates the first with a
  softmax over the group, the gated sum is normalised. Trailing tokens that do not fill a
  group produce no latent (they are covered by the sliding window).

The latent is returned *before* RoPE because the indexer derives its keys from the
unrotated latent; the attention module rotates afterwards.

Differences to the DeepSeek-V4 compressor in ``csa.py``: no overlapping windows, no
additive position embedding inside the group, fp32 gating weights. Semantics follow the
official ``Compressor`` in ``inference/model.py``; independent implementation.
"""

import copy
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn

from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    pool_groups_with_softmax_gate,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class CSA2CompressorSubmodules:
    """Submodule specs for :class:`CSA2Compressor`. Only the norm is spec-driven; the two
    projections are plain ``torch.nn.Linear`` so their dtype can follow the reference
    (fp32 for gated pooling, checkpoint dtype for ratio 1)."""

    norm: Union[ModuleSpec, type] = None


class CSA2Compressor(MegatronModule):
    """Softmax-gated group pooling (ratio > 1) or projection (ratio 1) into one KV latent."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CSA2CompressorSubmodules,
        compress_ratio: int,
        head_dim: int,
        name: str | None = None,
    ) -> None:
        super().__init__(config=config)
        if compress_ratio < 1:
            raise ValueError(f"CSA2Compressor needs compress_ratio >= 1, got {compress_ratio}")
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.gated = compress_ratio > 1

        proj_dtype = torch.float32 if self.gated else config.params_dtype
        self.linear_wkv = nn.Linear(config.hidden_size, head_dim, bias=False, dtype=proj_dtype)
        config.init_method(self.linear_wkv.weight)
        if self.gated:
            self.linear_wgate = nn.Linear(
                config.hidden_size, head_dim, bias=False, dtype=proj_dtype
            )
            config.init_method(self.linear_wgate.weight)
            mark_keep_in_fp32(self.linear_wkv.weight)
            mark_keep_in_fp32(self.linear_wgate.weight)
        else:
            self.linear_wgate = None

        norm_config = copy.copy(config)
        norm_config.normalization = "RMSNorm"
        self.norm = build_module(
            submodules.norm, config=norm_config, hidden_size=head_dim, eps=config.layernorm_epsilon
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress ``[s, b, hidden]`` into ``[s // ratio, b, head_dim]`` (pre-RoPE)."""
        if not self.gated:
            return self.norm(self.linear_wkv(hidden_states))

        x = hidden_states.float()
        latent = self.linear_wkv(x)
        gate = self.linear_wgate(x)
        pooled = pool_groups_with_softmax_gate(latent, gate, self.compress_ratio)
        return self.norm(pooled.to(hidden_states.dtype))

    def forward_packed(self, hidden_rows: torch.Tensor, group_rows: torch.Tensor) -> torch.Tensor:
        """Packed variant: pool explicit row groups.

        Args:
            hidden_rows: ``[rows, hidden]`` flat rows (may include halo rows).
            group_rows: ``[n_groups, ratio]`` indices into ``hidden_rows``; group ``g`` pools
                rows ``group_rows[g]`` (consecutive tokens of one segment).

        Returns:
            ``[n_groups, head_dim]`` pre-RoPE latents.
        """
        if group_rows.numel() == 0:
            # No group ends on this rank. The result must still depend on the inputs and the
            # trainable weights: the all-gather of owned entries is autograd aware on every
            # rank, so a detached empty tensor here would leave this rank out of the
            # reduce-scatter backward and desynchronise the collective.
            empty = hidden_rows[:0]
            empty = empty.float() if self.gated else empty
            latent = self.linear_wkv(empty)
            if self.gated:
                latent = latent + self.linear_wgate(empty)
            return latent.to(hidden_rows.dtype).view(0, self.head_dim)
        if not self.gated:
            return self.norm(self.linear_wkv(hidden_rows[group_rows[:, 0]]))

        gathered = hidden_rows[group_rows.reshape(-1)].float()  # [n * ratio, hidden]
        latent = self.linear_wkv(gathered).view(*group_rows.shape, self.head_dim)
        gate = self.linear_wgate(gathered).view(*group_rows.shape, self.head_dim)
        pooled = (latent * torch.softmax(gate, dim=1)).sum(dim=1)
        return self.norm(pooled.to(hidden_rows.dtype))
