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

"""Engram memory fusion into a standard Transformer residual stream."""

import math
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.utils import (
    make_sharded_tensors_for_checkpoint,
    sharded_state_dict_default,
)

from .config import EngramConfig, _parallel_world_size
from .memory import MultiHeadEmbedding, RowShardedMultiHeadEmbedding


class ShortConv(nn.Module):
    """Official causal depthwise short convolution over one continuous sequence."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        nn.init.zeros_(self.conv.weight)
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        if activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply RMSNorm and a causal depthwise convolution to [batch, sequence, hidden]."""
        sequence = x.shape[1]
        conv_input = self.norm(x).transpose(1, 2)
        output = self.conv(conv_input)[..., :sequence]
        output = output.transpose(1, 2).contiguous()
        return self.act_fn(output) if self.activation else output


class EngramLayer(nn.Module):
    """Layer-local Engram memory and fusion parameters."""

    def __init__(
        self,
        engram_config: EngramConfig,
        layer_id: int,
        table_sizes: Sequence[int],
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__()
        self.engram_config = engram_config
        self.layer_id = layer_id
        self.pg_collection = pg_collection
        cfg = engram_config
        tp_group = getattr(pg_collection, 'tp', None)
        dp_group = getattr(pg_collection, 'dp', None)
        dp_cp_group = getattr(pg_collection, 'dp_cp', None)
        table_group = getattr(pg_collection, 'tp_dp_cp', None)
        model_parallel_group = getattr(pg_collection, 'mp', None)
        if cfg.table_backend == "row_a2a":
            self.multi_head_embedding = RowShardedMultiHeadEmbedding(
                table_sizes,
                cfg.embedding_dim,
                layer_id=layer_id,
                seed=cfg.seed,
                init_method=cfg.init_method,
                params_dtype=cfg.params_dtype,
                use_cpu_initialization=cfg.use_cpu_initialization,
                perform_initialization=cfg.perform_initialization,
                calculate_per_token_loss=cfg.calculate_per_token_loss,
                table_group=table_group,
                tp_group=tp_group,
                dp_group=dp_group,
                dp_cp_group=dp_cp_group,
                model_parallel_group=model_parallel_group,
            )
        else:
            tensor_parallel_size = (
                torch.distributed.get_world_size(group=tp_group)
                if tp_group is not None
                else _parallel_world_size(parallel_state.get_tensor_model_parallel_world_size)
            )
            self.multi_head_embedding = MultiHeadEmbedding(
                table_sizes,
                cfg.embedding_dim,
                tensor_parallel_size=tensor_parallel_size,
                init_method=cfg.init_method,
                params_dtype=cfg.params_dtype,
                use_cpu_initialization=cfg.use_cpu_initialization,
                perform_initialization=cfg.perform_initialization,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            )
        self.short_conv = ShortConv(
            hidden_size=cfg.hidden_size, kernel_size=cfg.kernel_size, dilation=cfg.max_ngram_size
        )
        self.value_proj = nn.Linear(cfg.engram_hidden_size, cfg.hidden_size)
        self.key_proj = nn.Linear(cfg.engram_hidden_size, cfg.hidden_size)
        self.norm1 = nn.RMSNorm(cfg.hidden_size)
        self.norm2 = nn.RMSNorm(cfg.hidden_size)

    def _gate(self, score: Tensor) -> Tensor:
        score = score.abs().clamp_min(1e-6).sqrt() * score.sign()
        return score.sigmoid()

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> dict[str, Any]:
        """Recurse so MultiHeadEmbedding can expose its TP width shard."""
        result = {}
        tp_group = getattr(self.pg_collection, 'tp', None)
        # Compatibility for standalone layers; Hybrid passes a ProcessGroupCollection.
        if tp_group is None and parallel_state.model_parallel_is_initialized():
            tp_group = parallel_state.get_tensor_model_parallel_group()
        dp_cp_group = (metadata or {}).get('dp_cp_group')
        if dp_cp_group is None:
            dp_cp_group = getattr(self.pg_collection, 'dp_cp', None)
        local_state: dict[str, Any] = {}
        self._save_to_state_dict(local_state, '', keep_vars=True)
        result.update(
            make_sharded_tensors_for_checkpoint(
                local_state,
                prefix,
                sharded_offsets=sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=dp_cp_group,
            )
        )
        for name, module in self.named_children():
            result.update(
                sharded_state_dict_default(
                    module, f'{prefix}{name}.', sharded_offsets, metadata, tp_group=tp_group
                )
            )
        return result

    def forward(self, hidden_states: Tensor, embeddings: Tensor) -> Tensor:
        """Gate retrieved memory into [sequence, batch, hidden] residual states."""
        cfg = self.engram_config
        if hidden_states.ndim != 3 or hidden_states.size(-1) != cfg.hidden_size:
            raise ValueError("Engram hidden states must have shape [sequence, batch, hidden_size]")
        embeddings = embeddings.to(hidden_states.dtype)
        key = self.key_proj(embeddings)
        query = hidden_states.transpose(0, 1)
        score = (self.norm1(key) * self.norm2(query)).sum(dim=-1) / math.sqrt(cfg.hidden_size)
        value = self._gate(score).unsqueeze(-1) * self.value_proj(embeddings)
        delta = value + self.short_conv(value)
        return hidden_states + delta.transpose(0, 1).contiguous()
