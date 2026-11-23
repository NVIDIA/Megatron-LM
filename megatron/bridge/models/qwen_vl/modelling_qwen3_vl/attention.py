# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/0f06205924d4e19b0723aa8c33e34a1959ca17fe/src/megatron/bridge/models/qwen_vl/modelling_qwen3_vl/attention.py
# Source license: Apache-2.0.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Alibaba PAI Team.
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

import torch
from einops import rearrange
from torch import Tensor

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import apply_rotary_pos_emb_absolute
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer.attention import (
    HAVE_FA3,
    BaseInferenceContext,
    PackedSeqParams,
    SelfAttention,
    deprecate_inference_params,
    is_fa_min_version,
    nvtx_range_pop,
    nvtx_range_push,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention


def _qwen_attention_mask_for_core_attention(
    core_attention: object, attention_mask: Tensor | None, packed_seq_params: PackedSeqParams | None
) -> Tensor | None:
    """Translate Qwen's 2D valid-token mask to TE's masked-token convention."""
    if (
        not isinstance(core_attention, TEDotProductAttention)
        or packed_seq_params is not None
        or not isinstance(attention_mask, Tensor)
        or attention_mask.ndim != 2
    ):
        return attention_mask

    if attention_mask.dtype == torch.bool:
        valid_mask = attention_mask
    elif attention_mask.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        valid_mask = attention_mask != 0
    else:
        return attention_mask

    return (~valid_mask).unsqueeze(1).unsqueeze(1)


class Qwen3VLSelfAttention(SelfAttention):
    """
    Overrides the SelfAttention class, the difference is that qwen3vl uses apply_rotary_pos_emb_absolute
    instead of apply_rotary_pos_emb
    """

    @staticmethod
    def _local_causal_attention(query: Tensor, key: Tensor, value: Tensor, scale: float) -> Tensor:
        """Causal attention fallback for local padded fixed-score paths."""
        if query.shape[2] != key.shape[2]:
            if query.shape[2] % key.shape[2] != 0:
                raise ValueError(
                    f"Query heads ({query.shape[2]}) must be divisible by key heads ({key.shape[2]})."
                )
            repeat = query.shape[2] // key.shape[2]
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)

        scores = (
            torch.einsum("sbhd,tbhd->bhst", query.to(torch.float32), key.to(torch.float32)) * scale
        )
        sq, sk = scores.shape[-2], scores.shape[-1]
        if sq > 1:
            causal_mask = torch.ones((sq, sk), dtype=torch.bool, device=scores.device).triu(
                1 + max(sk - sq, 0)
            )
            scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)

        probs = torch.softmax(scores, dim=-1).to(value.dtype)
        context = torch.einsum("bhst,tbhd->sbhd", probs, value)
        return context.reshape(context.shape[0], context.shape[1], -1).contiguous()

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        key_value_states: Tensor | None = None,
        inference_context: BaseInferenceContext | None = None,
        rotary_pos_emb: (Tensor | tuple[Tensor, Tensor]) | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: BaseInferenceContext | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """
        # Check if we need to skip RoPE
        # no_rope is 0-indexed array and self.layer_number is 1-indexed
        no_rope = (
            self.config.no_rope_freq[self.layer_number - 1] if self.config.no_rope_freq else False
        )
        if no_rope:
            rotary_pos_emb = None

        inference_context = deprecate_inference_params(inference_context, inference_params)

        if inference_context and inference_context.is_dynamic_batching():
            assert HAVE_FA3 or is_fa_min_version(
                "2.7.3"
            ), "flash attn verion v2.7.3 and above is required for dynamic batching."

        # hidden_states: [sq, b, h]
        if self.config.flash_decode and not self.training and inference_context is not None:
            rotary_pos_emb = None
        else:
            assert rotary_pos_cos is None and rotary_pos_sin is None

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        nvtx_range_push(suffix="qkv")
        gate = None
        if self.config.attention_output_gate:
            query, key, value, gate = self.get_query_key_value_tensors(
                hidden_states, key_value_states, output_gate=True
            )
        else:
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        nvtx_range_pop(suffix="qkv")

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================

        in_decode_mode = (
            inference_context is not None
            and inference_context.is_decode_only()
            and not self.training
        )

        # This branch only runs in the decode phase of flash decoding and returns after the linear
        # projection. This conditional is not used in the prefill phase or non-flash-decoding cases.
        nvtx_range_push(suffix="adjust_key_value")
        if in_decode_mode and self.config.flash_decode:
            assert self.layer_number in inference_context.key_value_memory_dict
            assert inference_context.sequence_len_offset is not None
            inference_key_memory, inference_value_memory = inference_context.key_value_memory_dict[
                self.layer_number
            ]
            output = self.flash_decode(
                sequence_len_offset=sequence_len_offset,
                query_layer=query,
                key_layer=key,
                value_layer=value,
                inference_key_memory=inference_key_memory,
                inference_value_memory=inference_value_memory,
                rotary_cos=rotary_pos_cos,
                rotary_sin=rotary_pos_sin,
                rotary_interleaved=self.config.rotary_interleaved,
            )
            out = output.transpose(0, 1).contiguous()
            context_layer = out.view(out.size(0), out.size(1), -1)
            if gate is not None:
                context_layer = self._apply_output_gate(context_layer, gate)
            output, bias = self.linear_proj(context_layer)
            return output, bias

        if (
            in_decode_mode
            and self.config.enable_cuda_graph
            and inference_context.is_static_batching()
        ):
            raise ValueError("CUDA graphs must use flash decode with static batching!")

        query, key, value, rotary_pos_emb, attn_mask_type, block_table = (
            self._adjust_key_value_for_inference(
                inference_context,
                query,
                key,
                value,
                rotary_pos_emb,
                rotary_pos_cos,
                rotary_pos_sin,
                sequence_len_offset,
            )
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)
        nvtx_range_pop(suffix="adjust_key_value")

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        nvtx_range_push(suffix="rotary_pos_emb")
        if rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                # TODO VIJAY: simplify
                if inference_context is None or inference_context.is_static_batching():
                    query = apply_rotary_pos_emb_absolute(
                        query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q
                    )
                else:
                    query = inference_context.apply_rotary_emb_query(
                        query, q_pos_emb, self.config, cu_seqlens_q, self.model_comm_pgs.cp
                    )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb_absolute(
                    key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)
        nvtx_range_pop(suffix="rotary_pos_emb")

        # ==================================
        # core attention computation
        # ==================================

        nvtx_range_push(suffix="core_attention")
        core_attention_mask = _qwen_attention_mask_for_core_attention(
            self.core_attention, attention_mask, packed_seq_params
        )
        if (
            isinstance(self.core_attention, DotProductAttention)
            and packed_seq_params is None
            and attention_bias is None
            and (inference_context is None or inference_context.is_static_batching())
        ):
            scale = float(getattr(self.core_attention, "softmax_scale", query.shape[-1] ** -0.5))
            core_attn_out = self._local_causal_attention(query, key, value, scale)
        elif self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                core_attention_mask,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
            )
        else:
            if inference_context is None or inference_context.is_static_batching():
                # Static batching attention kernel.
                core_attn_out = self.core_attention(
                    query,
                    key,
                    value,
                    core_attention_mask,
                    attn_mask_type=attn_mask_type,
                    attention_bias=attention_bias,
                    packed_seq_params=packed_seq_params,
                )

            else:
                # Dynamic batching attention kernel.
                q, k, v = (query, key, value)
                cu_query_lengths, max_seqlen_q = inference_context.cu_query_lengths()
                cu_kv_lengths, kv_lengths, kv_lengths_decode_only, max_seqlen_k = (
                    inference_context.cu_kv_lengths()
                )

                core_attn_out = self.flash_decode_and_prefill(
                    q,
                    k,
                    v,
                    max_seqlen_q,
                    max_seqlen_k,
                    cu_query_lengths,
                    cu_kv_lengths,
                    kv_lengths,
                    kv_lengths_decode_only,
                    block_table,
                )
                core_attn_out = rearrange(core_attn_out, "s b h d -> s b (h d)")

        if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
        nvtx_range_pop(suffix="core_attention")

        # Output gate (for Gated Attention in hybrid architectures like Qwen3.5/3.6).
        if gate is not None:
            core_attn_out = self._apply_output_gate(core_attn_out, gate)

        # =================
        # Output. [sq, b, h]
        # =================

        nvtx_range_push(suffix="linear_proj")
        output, bias = self.linear_proj(core_attn_out)
        nvtx_range_pop(suffix="linear_proj")

        return output, bias
