# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2023 DeepSeek
#
# Portions adapted from https://github.com/deepseek-ai/Engram.
# Upstream commit: fb7f84a21f91223715394a33a1dc24bbfb7f788e (engram_demo_v1.py).
# Modified for a single residual stream, zero-initialized convolution, explicit
# process groups, distributed tables, and Megatron checkpoint integration.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gated fusion and causal convolution for retrieved Engram embeddings."""

import math

import torch
from torch import Tensor, nn

from megatron.core.transformer.module import MegatronModule

from .config import EngramConfig


class ShortConv(nn.Module):
    """Pre-normalized causal depthwise convolution over one continuous sequence."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        activation: bool = True,
        *,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.activation = activation
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
            dtype=dtype,
        )
        nn.init.zeros_(self.conv.weight)
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps, dtype=dtype)])
        if activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply pre-normalization and causal convolution to [batch, sequence, hidden]."""
        sequence = x.size(1)
        normalized = self.norms[0](x)
        output = self.conv(normalized.transpose(1, 2))[..., :sequence]
        output = output.transpose(1, 2).contiguous()
        return self.act_fn(output) if self.activation else output


class EngramLayer(nn.Module):
    """Own a lookup module and fuse retrieved embeddings into hidden states."""

    def __init__(
        self, engram_config: EngramConfig, multi_head_embedding: nn.Module, pg_collection=None
    ):
        super().__init__()
        self.engram_config = engram_config
        self.pg_collection = pg_collection
        cfg = engram_config
        self.tp_group = getattr(pg_collection, 'tp', None)
        self.multi_head_embedding = multi_head_embedding
        device = (
            torch.device("cpu")
            if cfg.use_cpu_initialization
            else torch.device("cuda", torch.cuda.current_device())
        )
        with torch.device(device if cfg.perform_initialization else "meta"):
            self.short_conv = ShortConv(
                hidden_size=cfg.hidden_size,
                kernel_size=cfg.kernel_size,
                dilation=cfg.max_ngram_size,
                dtype=cfg.params_dtype,
            )
            self.value_proj = nn.Linear(
                cfg.engram_hidden_size, cfg.hidden_size, dtype=cfg.params_dtype
            )
            self.key_projs = nn.ModuleList(
                [nn.Linear(cfg.engram_hidden_size, cfg.hidden_size, dtype=cfg.params_dtype)]
            )
            self.norm1 = nn.ModuleList([nn.RMSNorm(cfg.hidden_size, dtype=cfg.params_dtype)])
            self.norm2 = nn.ModuleList([nn.RMSNorm(cfg.hidden_size, dtype=cfg.params_dtype)])
        if not cfg.perform_initialization:
            # Materialize only fusion parameters; table initialization is independent.
            for module in (
                self.short_conv,
                self.value_proj,
                self.key_projs,
                self.norm1,
                self.norm2,
            ):
                module.to_empty(device=device)

    def _gate(self, score: Tensor) -> Tensor:
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        return score.sigmoid()

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Use native recursion while preserving explicit process-group metadata."""
        metadata = dict(metadata or {})
        if metadata.get('dp_cp_group') is None:
            metadata['dp_cp_group'] = getattr(self.pg_collection, 'dp_cp', None)
        return MegatronModule.sharded_state_dict(self, prefix, sharded_offsets, metadata)

    def forward(self, hidden_states: Tensor, embeddings: Tensor) -> Tensor:
        """Gate retrieved values and add their convolutional residual to hidden states."""
        cfg = self.engram_config
        compute_dtype = self.value_proj.weight.dtype
        query = hidden_states.transpose(0, 1).to(compute_dtype)
        embeddings = embeddings.to(compute_dtype)
        key = self.key_projs[0](embeddings)
        score = (self.norm1[0](key) * self.norm2[0](query)).sum(dim=-1) / math.sqrt(cfg.hidden_size)
        value = self._gate(score).unsqueeze(-1) * self.value_proj(embeddings)
        delta = value + self.short_conv(value)
        return hidden_states + delta.transpose(0, 1).contiguous()
