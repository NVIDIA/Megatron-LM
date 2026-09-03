# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Transformer Engine port of the RoPE NeMo audio transformer.

This module intentionally preserves the parameter layout used by the NeMo
``transformer_encoder.TransformerEncoder`` implementation used to train the
600M Granary encoder.  In particular, its state dict has keys such as
``pre_encode.proj.weight`` and ``layers.0.attn.w_qkv.weight``.  That lets a
converted NeMo archive load the encoder weights strictly without a tensor-key
conversion step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class FeatureStacking(nn.Module):
    """Stack audio frames then project, matching NeMo's FeatureStacking."""

    def __init__(self, subsampling_factor: int, feat_in: int, feat_out: int):
        super().__init__()
        self.subsampling_factor = subsampling_factor
        self.proj = nn.Linear(subsampling_factor * feat_in, feat_out, bias=False)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack and project a dense batch of padded audio features."""
        # x: [batch, features, time] -> [batch, time, features]
        x = x.transpose(1, 2)
        batch, time, features = x.shape
        pad_size = (-time) % self.subsampling_factor
        if pad_size:
            x = nn.functional.pad(x, (0, 0, 0, pad_size))
        x = x.reshape(
            batch, (time + pad_size) // self.subsampling_factor, features * self.subsampling_factor
        )
        x = self.proj(x)
        lengths = torch.div(
            lengths + self.subsampling_factor - 1, self.subsampling_factor, rounding_mode="floor"
        )
        return x, lengths

    def forward_packed(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack only valid frames and return contiguous per-audio token runs.

        The regular dense path stacks the shared padded batch dimension.  That
        is correct but particularly wasteful for long, variably-sized audio
        inputs: it creates feature-stacked frames for every suffix pad before
        attention has an opportunity to ignore them.  This path first crops
        each sample to its real frame length, pads only its final partial stack
        with zeros (the dense preprocessor's semantics), then concatenates
        the resulting runs before the common projection.
        """
        if x.ndim == 2:
            features = x.shape[1]
            lengths = lengths.to(dtype=torch.int64, device=x.device).reshape(-1)
            if int(lengths.sum().item()) != x.shape[0]:
                raise ValueError(
                    f"Packed features have {x.shape[0]} frames but "
                    f"sum(lengths)={int(lengths.sum().item())}"
                )
            padded_lengths = (
                torch.div(
                    lengths + self.subsampling_factor - 1,
                    self.subsampling_factor,
                    rounding_mode="floor",
                )
                * self.subsampling_factor
            )
            output_lengths = padded_lengths // self.subsampling_factor
            if x.shape[0] == 0:
                return x.new_zeros((0, self.proj.out_features)), output_lengths

            source_starts = lengths.cumsum(0) - lengths
            destination_starts = padded_lengths.cumsum(0) - padded_lengths
            frame_ids = torch.arange(x.shape[0], device=x.device)
            within_clip = frame_ids - torch.repeat_interleave(source_starts, lengths)
            destination = torch.repeat_interleave(destination_starts, lengths) + within_clip
            canvas = x.new_zeros((int(padded_lengths.sum().item()), features))
            canvas[destination] = x
            stacked = canvas.reshape(-1, features * self.subsampling_factor)
            return self.proj(stacked), output_lengths

        if x.ndim != 3:
            raise ValueError(
                f"Expected [B, F, T] or packed [FTotal, F] audio features, got {tuple(x.shape)}"
            )
        batch, features, max_time = x.shape
        if lengths.numel() != batch:
            raise ValueError(f"Expected {batch} input lengths, got {lengths.numel()}")

        x_btf = x.transpose(1, 2)
        lengths = lengths.to(dtype=torch.int64, device=x.device)
        output_lengths = torch.div(
            lengths + self.subsampling_factor - 1, self.subsampling_factor, rounding_mode="floor"
        )
        stacked_chunks = []
        for index, frame_count in enumerate(lengths.tolist()):
            if frame_count < 0 or frame_count > max_time:
                raise ValueError(
                    f"Invalid input length {frame_count} for audio with {max_time} frames"
                )
            if frame_count == 0:
                continue
            sample = x_btf[index, :frame_count]
            pad_size = (-frame_count) % self.subsampling_factor
            if pad_size:
                sample = F.pad(sample, (0, 0, 0, pad_size))
            stacked_chunks.append(sample.reshape(-1, features * self.subsampling_factor))
        if stacked_chunks:
            stacked = torch.cat(stacked_chunks, dim=0)
            return self.proj(stacked), output_lengths
        return x.new_zeros((0, self.proj.out_features)), output_lengths


class RotaryPositionalEncoding(nn.Module):
    """RoPE implementation and buffer layout used by the source encoder."""

    def __init__(self, d_k: int, rotary_fraction: float = 1.0, rope_base: float = 10000.0):
        super().__init__()
        if not 0 < rotary_fraction <= 1.0:
            raise ValueError(f"rotary_fraction must be in (0, 1], got {rotary_fraction}")
        d_k_rot = int(d_k * rotary_fraction)
        if d_k_rot < 2 or d_k_rot % 2:
            raise ValueError(
                "Effective rotary dim (d_k * rotary_fraction) must be a positive even number, "
                f"got {d_k_rot} from d_k={d_k} and rotary_fraction={rotary_fraction}"
            )
        self.d_k = d_k
        self.d_k_rot = d_k_rot
        self.rope_base = rope_base
        inv_freq = 1.0 / (rope_base ** (torch.arange(0, d_k_rot, 2, dtype=torch.float32) / d_k_rot))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def _apply_rotary(
        self, tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        if self.d_k_rot == tensor.shape[-1]:
            return tensor * cos + self._rotate_half(tensor) * sin
        rotated = tensor[..., : self.d_k_rot]
        passthrough = tensor[..., self.d_k_rot :]
        rotated = rotated * cos + self._rotate_half(rotated) * sin
        return torch.cat((rotated, passthrough), dim=-1)

    def extend_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> None:
        """Materialize RoPE buffers through ``length`` on the requested device."""
        if hasattr(self, "cos") and self.cos.size(0) >= length:
            return
        positions = torch.arange(0, length, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)
        if hasattr(self, "cos"):
            self.cos = cos
            self.sin = sin
        else:
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward_packed(
        self, query: torch.Tensor, key: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to THD-style variable-length audio sequences.

        ``query`` and ``key`` are laid out as ``[Ttotal, H, D]``.  The
        cumulative sequence lengths reset positions at every audio boundary,
        exactly matching separate dense encoder calls while avoiding padding.
        """
        if query.ndim != 3 or key.shape != query.shape:
            raise ValueError(
                "Expected packed query/key [Ttotal, H, D] with matching shapes, got "
                f"query={tuple(query.shape)}, key={tuple(key.shape)}"
            )
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(dtype=torch.long)
        if int(lengths.sum().item()) != query.shape[0]:
            raise ValueError(
                "Packed RoPE cumulative lengths do not match token count: "
                f"{int(lengths.sum().item())} != {query.shape[0]}"
            )
        if query.shape[0] == 0:
            return query, key

        starts = torch.repeat_interleave(cu_seqlens[:-1].to(dtype=torch.long), lengths)
        positions = torch.arange(query.shape[0], device=query.device) - starts
        cos = self.cos.index_select(0, positions).view(query.shape[0], 1, self.d_k_rot)
        sin = self.sin.index_select(0, positions).view(query.shape[0], 1, self.d_k_rot)
        query = self._apply_rotary(query, cos.to(query.dtype), sin.to(query.dtype))
        key = self._apply_rotary(key, cos.to(key.dtype), sin.to(key.dtype))
        return query, key


@dataclass(frozen=True)
class RopeTransformerEncoderConfig:
    """Configuration for the checkpoint-compatible RoPE audio encoder."""

    n_mels: int
    d_model: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    qk_norm: bool
    ff_expansion: float
    pre_block_norm: bool
    subsampling_factor: int
    rope_base: float
    rotary_fraction: float
    # Number of previous encoder positions visible to each causal query.
    # None or a negative value preserves unlimited-left causal attention.
    left_context: int | None = None
    # Runtime-only activation checkpointing toggle. This mirrors the legacy
    # NeMo transformer encoder so ``--recompute-audio`` has identical meaning
    # for both checkpoint architectures.
    recompute_layers: bool = False


class _PackedAttentionPolicy(TypedDict):
    sequence_ids: torch.Tensor
    mask_type: str
    window_size: tuple[int, int]


@dataclass(frozen=True)
class _PackedAttentionPolicies:
    """Per-sequence TE mask metadata shared by every encoder layer."""

    policies: list[_PackedAttentionPolicy]


class FeedForward(nn.Module):
    """Position-wise feed-forward network used by each audio encoder block."""

    def __init__(self, config: RopeTransformerEncoderConfig):
        super().__init__()
        ff_hidden = int(config.ff_expansion * config.d_model)
        self.net = nn.Sequential(
            nn.Linear(config.d_model, ff_hidden),
            nn.GELU(),
            nn.Dropout(config.drop_rate),
            nn.Linear(ff_hidden, config.d_model),
            nn.Dropout(config.drop_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward network to packed audio tokens."""
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """RoPE attention with scalar and per-sequence TransformerEngine policies."""

    def __init__(self, config: RopeTransformerEncoderConfig, rope: RotaryPositionalEncoding):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.rope = rope
        self.w_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.qk_norm = config.qk_norm
        left_context = config.left_context
        if left_context is not None:
            left_context = int(left_context)
            if left_context < 0:
                left_context = None
        self.causal_window_size = None if left_context is None else (left_context, 0)
        if config.qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)

    def _packed_te_attention(self):
        """Create TE's stateless THD attention lazily after checkpoint load."""
        name = "_packed_te_attention_impl"
        attention = getattr(self, name, None)
        if attention is None:
            try:
                from transformer_engine.pytorch import DotProductAttention
            except ImportError as exc:
                raise ImportError(
                    "Packed RoPE audio encoding requires transformer-engine. "
                    "Set --nemo-transformer-audio-attn-impl=te in a TE container."
                ) from exc
            attention_kwargs = dict(
                num_attention_heads=self.n_heads,
                kv_channels=self.head_dim,
                attention_dropout=0.0,
                qkv_format="thd",
                attn_mask_type="padding",
                tp_size=1,
                tp_group=None,
                layer_number=1,
            )
            attention = DotProductAttention(**attention_kwargs)
            attention.train(self.training)
            # DotProductAttention owns no trainable encoder weights. Keep the
            # runtime kernel object out of this module's state_dict so a
            # checkpoint saved after the first packed forward reloads strictly
            # before its lazy attention objects have been recreated.
            object.__setattr__(self, name, attention)
        return attention

    def _run_packed_te_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        is_causal: bool,
        attention_policies: _PackedAttentionPolicies | None = None,
    ) -> torch.Tensor:
        """Run scalar or per-sequence THD attention without reordering tokens."""
        # ``value`` is a view of the fused QKV projection.  A scalar THD call
        # with separately allocated RoPE Q/K requires all three tensors to
        # have a supported standalone layout; materializing V here makes that
        # layout explicit instead of relying on TE's fallback copy.
        value = value.contiguous()
        window_size = (-1, -1)
        if is_causal:
            window_size = self.causal_window_size or (-1, 0)
        attention_kwargs = {
            "attn_mask_type": "padding_causal" if is_causal else "padding",
            "window_size": window_size,
        }
        if attention_policies is not None:
            attention_kwargs = {
                "thd_attention_policies": attention_policies.policies,
                "thd_attention_policy_dispatch": "grouped",
            }
        return self._packed_te_attention()(
            query,
            key,
            value,
            None,
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            **attention_kwargs,
        )

    def forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        is_causal: bool,
        attention_policies: _PackedAttentionPolicies | None = None,
    ) -> torch.Tensor:
        """Attention over packed ``[Ttotal, D]`` audio tokens using TE THD."""
        token_count = x.shape[0]
        # TE's THD kernel requires individually contiguous Q/K/V tensors. A
        # view obtained by unbinding a fused [T, 3, H, D] projection has a
        # stride through the interleaved QKV dimension and is rejected for
        # multi-head inputs. Slice the checkpoint-compatible fused projection
        # weights instead, producing contiguous [T, H, D] tensors directly.
        q_weight, k_weight, v_weight = self.w_qkv.weight.chunk(3, dim=0)
        if self.w_qkv.bias is None:
            q_bias = k_bias = v_bias = None
        else:
            q_bias, k_bias, v_bias = self.w_qkv.bias.chunk(3, dim=0)
        query = F.linear(x, q_weight, q_bias).view(token_count, self.n_heads, self.head_dim)
        key = F.linear(x, k_weight, k_bias).view(token_count, self.n_heads, self.head_dim)
        value = F.linear(x, v_weight, v_bias).view(token_count, self.n_heads, self.head_dim)
        if self.qk_norm and token_count:
            query = self.q_norm(query).to(value.dtype)
            key = self.k_norm(key).to(value.dtype)
        if token_count == 0:
            output = value.new_zeros((0, self.n_heads, self.head_dim))
        else:
            query, key = self.rope.forward_packed(query, key, cu_seqlens)
            output = self._run_packed_te_attention(
                query,
                key,
                value,
                cu_seqlens,
                max_seqlen,
                is_causal=is_causal,
                attention_policies=attention_policies,
            )
        return self.out_proj(output.reshape(token_count, self.d_model))


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block operating on packed audio tokens."""

    def __init__(self, config: RopeTransformerEncoderConfig, rope: RotaryPositionalEncoding):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config, rope=rope)
        self.drop = nn.Dropout(config.drop_rate)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward_packed(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        is_causal: bool,
        attention_policies: _PackedAttentionPolicies | None = None,
    ) -> torch.Tensor:
        """Apply attention and feed-forward residuals to packed audio tokens."""
        x = x + self.drop(
            self.attn.forward_packed(
                self.norm1(x),
                cu_seqlens,
                max_seqlen,
                is_causal=is_causal,
                attention_policies=attention_policies,
            )
        )
        return x + self.drop(self.ffn(self.norm2(x)))


class RopeTransformerEncoder(nn.Module):
    """The non-causal source encoder, with a runtime causal-mask override."""

    def __init__(self, config: RopeTransformerEncoderConfig):
        super().__init__()
        self.config = config
        self._feat_in = config.n_mels
        self.d_model = config.d_model
        self.subsampling_factor = config.subsampling_factor
        left_context = config.left_context
        if left_context is not None:
            left_context = int(left_context)
            if left_context < 0:
                left_context = None
        self.causal_window_size = (-1, 0) if left_context is None else (left_context, 0)
        self.pre_encode = FeatureStacking(config.subsampling_factor, config.n_mels, config.d_model)
        self.embed_norm = nn.LayerNorm(config.d_model) if config.pre_block_norm else nn.Identity()
        self.pos_enc = RotaryPositionalEncoding(
            config.d_model // config.n_heads,
            rotary_fraction=config.rotary_fraction,
            rope_base=config.rope_base,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, rope=self.pos_enc) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        audio_signal: torch.Tensor,
        lengths: torch.Tensor,
        *,
        causal_mask: bool = False,
        thd_sequence_is_causal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a dense audio batch and return padded states and output lengths."""
        if audio_signal.ndim != 3 or audio_signal.shape[1] != self._feat_in:
            raise ValueError(
                f"Expected audio_signal [batch, {self._feat_in}, time], "
                f"got {tuple(audio_signal.shape)}"
            )
        x_packed, output_lengths = self.forward_packed(
            audio_signal,
            lengths,
            causal_mask=causal_mask,
            thd_sequence_is_causal=thd_sequence_is_causal,
        )
        batch = audio_signal.shape[0]
        max_output_frames = (
            audio_signal.shape[-1] + self.subsampling_factor - 1
        ) // self.subsampling_factor
        output = x_packed.new_zeros((batch, max_output_frames, self.d_model))
        valid = torch.arange(max_output_frames, device=output.device).unsqueeze(
            0
        ) < output_lengths.unsqueeze(1)
        output[valid] = x_packed
        return output.transpose(1, 2), output_lengths

    def forward_packed(
        self,
        audio_signal: torch.Tensor,
        lengths: torch.Tensor,
        *,
        causal_mask: bool = False,
        thd_sequence_is_causal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the complete RoPE encoder on true packed audio sequences.

        This uses Transformer Engine's THD variable-length attention so both
        self-attention and the feed-forward stack skip audio padding.  The
        returned token order is one contiguous run per input audio item.
        """
        x, output_lengths = self.pre_encode.forward_packed(
            audio_signal, lengths.to(dtype=torch.int64)
        )
        output_lengths = output_lengths.to(dtype=torch.int64, device=x.device)
        attention_policies = None
        if thd_sequence_is_causal is not None:
            thd_sequence_is_causal = thd_sequence_is_causal.to(device=x.device, dtype=torch.bool)
            if thd_sequence_is_causal.ndim != 1 or (
                thd_sequence_is_causal.numel() != output_lengths.numel()
            ):
                raise ValueError(
                    "thd_sequence_is_causal must have one boolean value per input audio: "
                    f"got shape {tuple(thd_sequence_is_causal.shape)} for "
                    f"{output_lengths.numel()} audios"
                )
        if x.shape[0] == 0:
            return x, output_lengths

        packed_lengths = output_lengths
        if thd_sequence_is_causal is not None:
            attention_policies = _PackedAttentionPolicies(
                policies=[
                    {
                        "sequence_ids": torch.nonzero(
                            ~thd_sequence_is_causal, as_tuple=False
                        ).flatten(),
                        "mask_type": "padding",
                        "window_size": (-1, -1),
                    },
                    {
                        "sequence_ids": torch.nonzero(
                            thd_sequence_is_causal, as_tuple=False
                        ).flatten(),
                        "mask_type": "padding_causal",
                        "window_size": self.causal_window_size,
                    },
                ]
            )

        max_seqlen = int(packed_lengths.max().item()) if packed_lengths.numel() else 0
        cu_seqlens = F.pad(packed_lengths.cumsum(0).to(dtype=torch.int32), (1, 0))

        self.pos_enc.extend_pe(max_seqlen, x.device, x.dtype)
        x = self.embed_norm(x)
        for layer in self.layers:
            if self.config.recompute_layers and self.training and x.requires_grad:
                x = checkpoint(
                    lambda hidden, layer=layer: layer.forward_packed(
                        hidden,
                        cu_seqlens,
                        max_seqlen,
                        is_causal=causal_mask,
                        attention_policies=attention_policies,
                    ),
                    x,
                    use_reentrant=False,
                )
            else:
                x = layer.forward_packed(
                    x,
                    cu_seqlens,
                    max_seqlen,
                    is_causal=causal_mask,
                    attention_policies=attention_policies,
                )
        x = self.final_norm(x)
        return x, output_lengths
