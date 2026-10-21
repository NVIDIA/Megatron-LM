# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Core attention module of the PP attention backend."""

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

from .api import get_pp_attention
from .autograd import a2a_attention, cp_all_gather_attention, dense_attention, varlen_attention
from .decode_api import get_pp_decoder, page_view


class PPDotProductAttention(MegatronModule):
    """Causal attention with the same bytes in training, prefill, decode and verify.

    Takes the arguments of TEDotProductAttention. Forward is the PP kernel; backward
    is FlashAttention-4 fed with the PP output and LSE.
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
            raise ValueError(f"PP attention supports causal masks only, got {attn_mask_type}")
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        tp_size = pg_collection.tp.size()
        self.cp_group = getattr(pg_collection, 'cp', None)
        self.cp_size = self.cp_group.size() if self.cp_group is not None else 1
        self.cp_comm_type = cp_comm_type
        self.head_dim = config.kv_channels
        self.num_q_heads = divide(config.num_attention_heads, tp_size)
        self.num_kv_heads = divide(config.num_query_groups or config.num_attention_heads, tp_size)
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(self.head_dim)
        )
        self.super_block_tiles = config.pp_attention_super_block_tiles

    def _attn(self, device, heads_div=1):
        return get_pp_attention(
            device,
            self.head_dim,
            self.num_q_heads // heads_div,
            self.num_kv_heads // heads_div,
            self.super_block_tiles,
            self.softmax_scale,
        )

    def decode_and_prefill(self, context, q, k, v, cu_seqlens_q, seqlens_k, block_table):
        """One dynamic-engine step over the paged KV cache, with the bytes of the training forward.

        Mixed steps run the paged prefill kernel; decode-only steps run split decode
        (one token per request) or verify (several tokens per request).
        """
        assert not self.training
        attn = self._attn(q.device)
        k_pages, v_pages, page_table = page_view(k, v, block_table)
        seqused_k = seqlens_k.to(torch.int32)
        if not context.is_decode_only():
            if context.using_cuda_graph_this_step():
                raise ValueError("PP attention does not support CUDA graphs for prefill steps")
            out = attn.prefill_paged(
                q.squeeze(1), k_pages, v_pages, cu_seqlens_q.to(torch.int32), seqused_k, page_table
            )
            return out.unsqueeze(1)
        num_requests = seqlens_k.shape[0]
        tokens = context.num_speculative_tokens + 1
        rows = num_requests * tokens
        decoder = get_pp_decoder(attn, context.max_requests, context.max_sequence_length, tokens)
        out = q.new_zeros(q.shape) if rows < q.shape[0] else torch.empty_like(q)
        decoder.verify(
            q[:rows].reshape(num_requests, tokens, q.shape[2], q.shape[3]),
            k_pages,
            v_pages,
            page_table,
            seqused_k,
            out=out[:rows].view(num_requests, tokens, q.shape[2], q.shape[3]),
        )
        return out

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
        if attention_bias is not None:
            raise ValueError("PP attention does not support attention_bias")
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
