# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DS4.1 projections over Core's native attention/THD/CP execution."""
import torch

from megatron.core.models.common.embeddings import YarnRotaryEmbedding
from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
    DSv4HybridSelfAttention,
)


class DS41SelfAttention(DSv4HybridSelfAttention):
    """Core dev 7b150acf get_query_key_value_tensors, with DS4.1 Q semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._dsv4_compress_ratio == 1:
            self._use_compressed_rope()

    def _use_compressed_rope(self):
        """Core dev 7b150acf: Ratio one rotates like every compressing layer: compress theta with YaRN."""
        config, old = self.config, self.rotary_pos_emb
        new = YarnRotaryEmbedding(
            config.qk_pos_emb_head_dim,
            rotary_base=config.csa_compress_rotary_base,
            scaling_factor=config.rotary_scaling_factor,
            original_max_position_embeddings=config.original_max_position_embeddings,
            beta_fast=config.beta_fast,
            beta_slow=config.beta_slow,
            mscale=config.mscale,
            mscale_all_dim=config.mscale_all_dim,
            cp_group=self.pg_collection.cp,
        )
        for module in list(self.modules()):
            for table in (module._modules, module.__dict__):
                for name in [name for name, value in table.items() if value is old]:
                    table[name] = new
        self._dsv4_rope_base = config.csa_compress_rotary_base
        self._dsv4_uses_yarn_rope = True

    def get_query_key_value_tensors(
        self,
        hidden_states,
        key_value_states=None,
        position_ids=None,
        packed_seq_params=None,
        inference_context=None,
        *,
        inference_params=None,
        boundary_hidden=None,
    ):
        if (
            key_value_states is not None
            or inference_context is not None
            or inference_params is not None
        ):
            raise ValueError(
                "DS4.1 standard training supports self-attention without inference caches"
            )
        if packed_seq_params is None or packed_seq_params.qkv_format != "thd":
            raise ValueError("DS4.1 standard training requires packed THD inputs")
        if self.config.apply_rope_fusion or self.recompute_up_proj:
            raise ValueError(
                "DS4.1 projection contract requires unfused RoPE without up-projection recompute"
            )
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError("DS4.1 THD hidden states must have shape [tokens, 1, hidden]")
        params = packed_seq_params
        prefix = params.cu_seqlens_q_padded
        if prefix is None:
            prefix = params.cu_seqlens_q
        if params.max_seqlen_q is None:
            raise ValueError("DS4.1 THD requires max_seqlen_q")
        maximum = int(params.max_seqlen_q)
        frequencies = self.rotary_pos_emb(maximum, packed_seq=True)
        if isinstance(frequencies, tuple):
            frequencies = frequencies[0]
        low_q, _ = self.linear_q_down_proj(hidden_states.squeeze(1))
        low_q = self.q_layernorm(low_q)
        query, _ = self.linear_q_up_proj(low_q)
        query = query.reshape(-1, self.num_attention_heads_per_partition, self.q_head_dim)
        global_start = self.pg_collection.cp.rank() * hidden_states.shape[0]
        content_dim = self.q_head_dim - self.config.qk_pos_emb_head_dim
        query = cp_utils.apply_thd_cp_local_rope_unfused(
            query,
            frequencies,
            content_dim,
            self.config.qk_pos_emb_head_dim,
            prefix,
            global_start,
            self.config,
        )
        halo_rows = 0 if boundary_hidden is None else boundary_hidden.shape[0]
        projection_input = (
            hidden_states
            if boundary_hidden is None
            else torch.cat((boundary_hidden, hidden_states), dim=0)
        )
        kv, _ = self.linear_kv_proj(projection_input.squeeze(1))
        kv = self.kv_layernorm(kv).unsqueeze(1)
        kv = cp_utils.apply_thd_cp_local_rope_unfused(
            kv,
            frequencies,
            content_dim,
            self.config.qk_pos_emb_head_dim,
            prefix,
            global_start - halo_rows,
            self.config,
        )
        boundary_kv = None if boundary_hidden is None else kv[:halo_rows]
        kv = kv[halo_rows:]
        result = (
            query.contiguous(),
            kv.contiguous(),
            kv.contiguous(),
            low_q,
            hidden_states.squeeze(1),
        )
        return result if boundary_kv is None else result + (boundary_kv.contiguous(),)
