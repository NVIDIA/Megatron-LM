# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

from megatron.core.models.gemma4.gemma4_inference_context import Gemma4InferenceContext
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module


@dataclass
class Gemma4SelfAttentionSubmodules(SelfAttentionSubmodules):
    """:class:`SelfAttentionSubmodules` plus the scaleless per-head V norm."""

    v_layernorm: Union[ModuleSpec, type, None] = None


class Gemma4SelfAttention(SelfAttention):
    """Gemma 4 self-attention with scaleless v_norm and cross-layer KV sharing.

    Differences from the base :class:`SelfAttention` (all per HF
    ``Gemma4TextAttention``, modeling_gemma4.py:1229-1290):

    * ``softmax_scale = 1.0`` for both layer types (no ``1/sqrt(head_dim)``); set on
      the per-layer config.
    * q/k norm via the spec's ``q_layernorm``/``k_layernorm`` (``Gemma4RMSNorm`` over
      head_dim, applied pre-RoPE) and a NEW scaleless ``v_layernorm`` applied to V.
    * RoPE applied with externally-supplied per-layer-type cos/sin (full-width
      ``rotate_half``), POST q/k norm.
    * KV bus: producer layers write their post-norm/post-RoPE (k, v) into a shared
      dict; borrower layers read (k, v) from it and skip their own k/v processing.

    The producer/borrower role is derived from the config (``num_kv_shared_layers``)
    and this layer's 1-based ``layer_number``, matching HF's
    ``first_kv_shared_layer_idx`` / ``store_full_length_kv`` logic exactly.

    ``forward`` is overridden with a clean eager path (no flash-decode / packed-
    sequence / CP machinery; this port is DDP=1, all parallelism = 1) so the
    bitwise islands (fp32 softmax, ``finfo.min`` additive mask, scale=1.0,
    scaleless v_norm) are explicit. Optionally accepts a
    :class:`Gemma4InferenceContext` to enable single-batch KV-cache decode
    (prefill stores per-layer post-norm/post-RoPE K/V into the context;
    decode appends one step and reads from the context, with shared layers
    aliasing the producer's slot).
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Gemma4SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            **kwargs,
        )

        # Scaleless per-head V RMSNorm (no weight) over head_dim. Built only on
        # producer/own layers; borrowers reuse a producer's already-normed V.
        # The v_layernorm submodule is a norm BUILDER (a plain function, like
        # q/k_layernorm), so it is called directly rather than via build_module
        # (build_module returns a FunctionType unchanged instead of invoking it).
        self.v_layernorm = (
            submodules.v_layernorm(
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
            if submodules.v_layernorm is not None
            else None
        )

        # "full" layers drop the sliding window (get_config_for_layer sets it None).
        self.layer_type = "full" if self.config.window_size is None else "sliding"

        # Cross-layer KV-share role (HF modeling_gemma4.py:1199-1204). layer_number is
        # 1-based; HF layer_idx is 0-based.
        num_shared = getattr(self.config, "num_kv_shared_layers", 0)
        first_shared_idx = self.config.num_layers - num_shared
        layer_idx = self.layer_number - 1
        self.is_kv_shared_layer = num_shared > 0 and layer_idx >= first_shared_idx
        # Producer = last own-layer of this layer_type among layers [0, first_shared_idx).
        prev_types = self.config.layer_types[:first_shared_idx]
        self.store_full_length_kv = (
            not self.is_kv_shared_layer
            and layer_idx == len(prev_types) - 1 - prev_types[::-1].index(self.layer_type)
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        *,
        rotary_cos_sin: Optional[tuple] = None,
        kv_bus: Optional[dict] = None,
        inference_context: Optional[Gemma4InferenceContext] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Clean eager forward. ``hidden_states`` is [s, b, h] (MLM seq-first).

        ``attention_mask`` is the additive (``finfo.min``) mask for this layer type,
        broadcastable to [b, np, sq, sk]. ``rotary_cos_sin`` is (cos, sin) of width
        head_dim for this layer type, shape [b, sq, head_dim]. ``kv_bus`` is the
        per-forward shared K/V dict keyed by layer_type (training-only path).

        If ``inference_context`` is provided, the cross-layer KV bus is bypassed in
        favor of the context's persistent per-producer-layer cache: own / producer
        layers append their new (post-norm, post-RoPE) K/V into the context;
        borrower layers read the producer's slot via ``inference_context.get_kv``.
        Sliding-window decode-step queries see only the last ``sliding_window``
        cached keys (cache is stored full-length to match HF producer semantics).
        """
        # Fused QKV projection -> q [s, b, np, hd], k/v [s, b, ng, hd]. q_norm/k_norm
        # are applied inside get_query_key_value_tensors (per-head over head_dim).
        query, key, value = self.get_query_key_value_tensors(hidden_states)

        cos, sin = rotary_cos_sin
        # cos/sin are [b, s, hd]; MLM tensors are [s, b, h, hd] -> move cos/sin to
        # seq-first and broadcast over (b, heads). apply_rotary_pos_emb expects
        # [..., hd] with cos/sin unsqueezed on the head axis.
        cos = cos.transpose(0, 1).unsqueeze(2)  # [s, b, 1, hd]
        sin = sin.transpose(0, 1).unsqueeze(2)

        # Query is always projected, normed, and roped on this layer.
        query = query * cos + _rotate_half(query) * sin

        if inference_context is not None:
            if self.is_kv_shared_layer:
                # Borrower: read producer's persistent slot. Producer runs earlier
                # in this same forward, so its new K/V is already appended.
                producer_kv = inference_context.get_kv(self.layer_number)
                assert producer_kv is not None, (
                    f"layer {self.layer_number} (kv-shared borrower) found no "
                    f"producer cache for layer_type={self.layer_type}; producer "
                    "must run earlier in the forward"
                )
                key, value = producer_kv
            else:
                # Own/producer layer: project, norm, RoPE, then append to slot.
                key = key * cos + _rotate_half(key) * sin
                if self.v_layernorm is not None:
                    value = apply_module(self.v_layernorm)(value)
                key, value = inference_context.append_kv(self.layer_number, key, value)

            # Decode-step (s_q == 1) sees the entire cached range. For sliding
            # layers, clip to the last `sliding_window` keys -- equivalent to the
            # sliding causal mask but does not require building one for s_q == 1.
            attn_mask = attention_mask
            if self.layer_type == "sliding" and inference_context.is_decode():
                window = self.config.sliding_window
                if key.size(0) > window:
                    key = key[-window:]
                    value = value[-window:]
                attn_mask = None
            elif inference_context.is_decode():
                # Full layer, decode-step: every cached key is visible to s_q == 1.
                attn_mask = None

            context = self._gemma4_core_attention(query, key, value, attn_mask)
        else:
            # Training path: per-forward cross-layer kv_bus (rebuilt every forward).
            if self.is_kv_shared_layer:
                # Borrow producer's post-norm/post-RoPE K/V; discard own k/v projections.
                key, value = kv_bus[self.layer_type]
            else:
                key = key * cos + _rotate_half(key) * sin
                if self.v_layernorm is not None:
                    value = apply_module(self.v_layernorm)(value)
                if self.store_full_length_kv:
                    kv_bus[self.layer_type] = (key, value)

            context = self._gemma4_core_attention(query, key, value, attention_mask)

        # [s, b, np, hd] -> [s, b, np*hd] -> o_proj.
        context = context.reshape(context.size(0), context.size(1), -1)
        output, bias = apply_module(self.linear_proj)(context)
        return output, bias

    def _gemma4_core_attention(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Optional[Tensor]
    ) -> Tensor:
        """Eager attention mirroring HF ``eager_attention_forward`` (scale=1.0, fp32 softmax).

        query [s, b, np, hd]; key/value [s, b, ng, hd]. Returns context [s, b, np, hd].
        """
        s_q, b, np_, hd = query.shape
        ng = key.size(2)
        s_k = key.size(0)
        groups = np_ // ng

        # [s, b, n, hd] -> [b, n, s, hd].
        q = query.permute(1, 2, 0, 3)
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        # GQA: repeat K/V groups to match query heads (HF repeat_kv).
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)

        # (q @ k^T) * 1.0 ; softmax_scale = 1.0 for both layer types (HF scaling=1.0).
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.config.softmax_scale
        if attention_mask is not None:
            scores = scores + attention_mask  # additive finfo.min mask

        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        context = torch.matmul(probs, v)  # [b, np, s, hd]

        # [b, np, s, hd] -> [s, b, np, hd].
        return context.permute(2, 0, 1, 3).contiguous()


def _rotate_half(x: Tensor) -> Tensor:
    """Full-width rotate_half (split at hd/2), matching gemma4_rope.rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
