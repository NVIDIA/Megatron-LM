# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Core attention module of the exact attention backend."""

import math
from typing import Optional

import torch
from torch import Tensor

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide

from .api import get_exact_attention
from .autograd import a2a_attention, cp_all_gather_attention, dense_attention, varlen_attention

CP_COMM_TYPES = ("all_gather", "a2a")


class ExactDotProductAttention(MegatronModule):
    """Causal attention with the same bytes in training, prefill, decode and verify.

    Takes the arguments of TEDotProductAttention. Forward is the exact kernel; backward
    is FlashAttention-4 fed with the exact output and LSE.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: Optional[str] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config)
        if attn_mask_type not in (AttnMaskType.causal,):
            raise ValueError(f"exact attention supports causal masks only, got {attn_mask_type}")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        tp_size = pg_collection.tp.size()
        self.cp_group = getattr(pg_collection, 'cp', None)
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1
        self.cp_comm_type = cp_comm_type or "all_gather"
        if self.cp_size > 1 and self.cp_comm_type not in CP_COMM_TYPES:
            raise ValueError(
                f"exact attention supports cp_comm_type in {CP_COMM_TYPES}, got {cp_comm_type!r}"
            )
        self.head_dim = config.kv_channels
        self.num_q_heads = divide(config.num_attention_heads, tp_size)
        self.num_kv_heads = divide(config.num_query_groups or config.num_attention_heads, tp_size)
        if self.cp_size > 1 and self.cp_comm_type == "a2a":
            if self.num_q_heads % self.cp_size or self.num_kv_heads % self.cp_size:
                raise ValueError("exact attention a2a needs heads divisible by the cp size")
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(self.head_dim)
        )
        self.super_block_tiles = config.exact_attention_super_block_tiles

    def _attn(self, device, heads_div=1):
        return get_exact_attention(
            device,
            self.head_dim,
            self.num_q_heads // heads_div,
            self.num_kv_heads // heads_div,
            self.super_block_tiles,
            self.softmax_scale,
        )

    def _packed(self, q, k, v, cu_seqlens):
        if self.cp_size == 1:
            return varlen_attention(q, k, v, self._attn(q.device), cu_seqlens)
        if self.cp_comm_type == "a2a":
            return a2a_attention(
                q, k, v, self._attn(q.device, self.cp_size), cu_seqlens, self.cp_group
            )
        return cp_all_gather_attention(q, k, v, self._attn(q.device), cu_seqlens, self.cp_group)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor],
        attn_mask_type: Optional[AttnMaskType] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Query/key/value are [s, b, h, d], or [t, h, d] with qkv_format 'thd'."""
        if attn_mask_type not in (None, AttnMaskType.causal):
            raise ValueError(f"exact attention supports causal masks only, got {attn_mask_type}")
        if attention_bias is not None:
            raise ValueError("exact attention does not support attention_bias")
        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            cu = packed_seq_params.cu_seqlens_q_padded
            if cu is None:
                cu = packed_seq_params.cu_seqlens_q
            out = self._packed(query, key, value, cu)
            return out.reshape(out.shape[0], -1)
        s, b = query.shape[:2]
        if self.cp_size == 1:
            to_bshd = lambda x: x.transpose(0, 1).contiguous()
            out = dense_attention(
                to_bshd(query), to_bshd(key), to_bshd(value), self._attn(query.device)
            )
            return out.transpose(0, 1).reshape(s, b, -1)
        to_thd = lambda x: x.transpose(0, 1).reshape(b * s, x.shape[2], x.shape[3])
        length = s * self.cp_size
        cu = torch.arange(b + 1, dtype=torch.int32, device=query.device) * length
        out = self._packed(to_thd(query), to_thd(key), to_thd(value), cu)
        return out.view(b, s, -1).transpose(0, 1).contiguous()
