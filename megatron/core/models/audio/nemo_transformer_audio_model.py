# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Tuple

import torch

from megatron.core.transformer.module import MegatronModule

from .nemo_transformer_encoder import TransformerEncoder
from .packed_audio import PackedAudioEmbeddings


@dataclass
class NemoTransformerAudioConfig:
    """Hyperparameters for the vendored NeMo-style TransformerEncoder audio tower."""

    n_mels: int = 80
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 17
    drop_rate: float = 0.1
    qkv_bias: bool = False
    causal_mask: bool = False
    pre_encode: str = "conv"
    nan_debug: bool = False
    qk_norm: bool = False
    subsampling_factor: int = 4
    # Attention backend: "auto" prefers transformer_engine when importable, else SDPA.
    # Explicit values: "te" | "sdpa" | "fa". Not a structural field — checkpoints
    # trained with one backend load into another.
    attn_impl: str = "auto"
    # Runtime-only activation checkpointing toggle for the vendored audio
    # transformer layer stack. Checkpoints trained without it load unchanged.
    recompute_layers: bool = False
    # Runtime-compatible attention shape control. None or negative keeps the
    # original unlimited-left causal attention. Non-negative values require
    # causal_mask=True and limit each token to that many previous positions.
    left_context: Optional[int] = None

    @property
    def output_embedding_dim(self) -> int:
        """Return the encoder output embedding dimension (``d_model``)."""
        return self.d_model

    @property
    def encoder_time_stride(self) -> int:
        """Return the time downsampling factor of the pre-encode stage."""
        if self.pre_encode in ("conv", "depth_conv"):
            return 4
        return self.subsampling_factor

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NemoTransformerAudioConfig":
        """Build a config from a dict, keeping only keys that match config fields."""
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**kwargs)


class NemoTransformerAudioModel(MegatronModule):
    """Audio encoder matching LLaVA expectations.

    ``forward(features, mask)`` returns ``(B, T', H)`` embeddings and a bool mask.
    """

    def __init__(self, config: NemoTransformerAudioConfig) -> None:
        super().__init__(config=config)
        self.encoder = TransformerEncoder(
            n_mels=config.n_mels,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            drop_rate=config.drop_rate,
            qkv_bias=config.qkv_bias,
            causal_mask=config.causal_mask,
            pre_encode=config.pre_encode,
            nan_debug=config.nan_debug,
            qk_norm=config.qk_norm,
            subsampling_factor=config.subsampling_factor,
            attn_impl=config.attn_impl,
            recompute_layers=config.recompute_layers,
            left_context=config.left_context,
        )

    @staticmethod
    def _ceil_div(value: int, divisor: int) -> int:
        return (value + divisor - 1) // divisor

    def _post_subsample_lengths(
        self, input_seq_lengths: torch.Tensor, max_input_frames: int
    ) -> torch.Tensor:
        if self.config.pre_encode in ("conv", "depth_conv"):
            return torch.div(
                torch.div(input_seq_lengths, 2, rounding_mode="floor"), 2, rounding_mode="floor"
            )
        if self.config.pre_encode == "stacking":
            factor = int(self.config.subsampling_factor)
            return torch.div(input_seq_lengths + factor - 1, factor, rounding_mode="floor")
        raise ValueError(f"Unsupported pre_encode={self.config.pre_encode!r}")

    def _pre_encode_forward_flops(self, batch_size: int, max_input_frames: int) -> int:
        n_mels = int(self.config.n_mels)
        d_model = int(self.config.d_model)

        if self.config.pre_encode == "conv":
            t1 = max_input_frames
            t2 = self._ceil_div(t1, 2)
            t3 = self._ceil_div(t2, 2)
            return (
                2 * batch_size * t1 * d_model * n_mels * 3
                + 2 * batch_size * t2 * d_model * d_model * 3
                + 2 * batch_size * t3 * d_model * d_model * 3
            )

        if self.config.pre_encode == "depth_conv":
            t1 = max_input_frames
            t2 = self._ceil_div(t1, 2)
            t3 = self._ceil_div(t2, 2)
            return (
                2 * batch_size * t1 * d_model * n_mels * 3
                + 2 * batch_size * t2 * d_model * 3
                + 2 * batch_size * t2 * d_model * d_model
                + 2 * batch_size * t3 * d_model * 3
                + 2 * batch_size * t3 * d_model * d_model
            )

        if self.config.pre_encode == "stacking":
            factor = int(self.config.subsampling_factor)
            pad_size = (-max_input_frames) % factor
            stacked_frames = (max_input_frames + pad_size) // factor
            return 2 * batch_size * stacked_frames * (factor * n_mels) * d_model

        raise ValueError(f"Unsupported pre_encode={self.config.pre_encode!r}")

    def _attention_pair_count(self, lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(dtype=torch.float64)
        if not self.config.causal_mask:
            return (lengths * lengths).sum()

        left_context = self.config.left_context
        if left_context is not None:
            left_context = int(left_context)
            if left_context < 0:
                left_context = None

        if left_context is None:
            return (lengths * (lengths + 1.0) / 2.0).sum()

        window = float(left_context + 1)
        full_prefix = float((left_context + 1) * (left_context + 2)) / 2.0
        windowed = full_prefix + (lengths - window).clamp(min=0.0) * window
        triangular = lengths * (lengths + 1.0) / 2.0
        return torch.where(lengths <= window, triangular, windowed).sum()

    def estimate_flops(
        self,
        input_seq_lengths: torch.Tensor,
        max_input_frames: Optional[int] = None,
        include_backward: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Estimate NeMo audio tower FLOPs for a batch of mel-frame lengths.

        The estimate counts Conv/stacking pre-encode work, QKV/out projections,
        attention score/value products, and the two FeedForward linears. Norms,
        activations, dropout, masking, and softmax are intentionally omitted to
        keep the accounting comparable to Megatron's GEMM-oriented LM estimate.
        """
        if not torch.is_tensor(input_seq_lengths):
            input_seq_lengths = torch.tensor(input_seq_lengths, dtype=torch.long)
        device = input_seq_lengths.device
        zero = torch.zeros((), dtype=torch.float64, device=device)
        if input_seq_lengths.numel() == 0:
            return {
                "nemo_forward": zero,
                "nemo_train": zero,
                "pre_encode_forward": zero,
                "transformer_forward": zero,
            }

        input_seq_lengths = input_seq_lengths.to(dtype=torch.long).clamp(min=0)
        if max_input_frames is None:
            max_input_frames = int(input_seq_lengths.max().item())
        elif torch.is_tensor(max_input_frames):
            max_input_frames = int(max_input_frames.item())
        else:
            max_input_frames = int(max_input_frames)
        max_input_frames = max(0, max_input_frames)

        batch_size = int(input_seq_lengths.numel())
        pre_encode_forward = torch.tensor(
            float(self._pre_encode_forward_flops(batch_size, max_input_frames)),
            dtype=torch.float64,
            device=device,
        )

        post_lengths = self._post_subsample_lengths(input_seq_lengths, max_input_frames)
        token_count = post_lengths.to(dtype=torch.float64).sum()
        attention_pairs = self._attention_pair_count(post_lengths)

        d_model = float(self.config.d_model)
        projection_and_ffn = 24.0 * token_count * d_model * d_model
        attention_core = 4.0 * attention_pairs * d_model
        transformer_forward = float(self.config.n_layers) * (projection_and_ffn + attention_core)
        nemo_forward = pre_encode_forward + transformer_forward

        if include_backward is None:
            include_backward = self.training and any(p.requires_grad for p in self.parameters())
        nemo_train = nemo_forward * (3.0 if include_backward else 1.0)

        return {
            "nemo_forward": nemo_forward,
            "nemo_train": nemo_train,
            "pre_encode_forward": pre_encode_forward,
            "transformer_forward": transformer_forward,
        }

    def forward(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode mel features into ``(B, T', H)`` embeddings and a validity mask."""
        if input_features.ndim != 3:
            raise ValueError(
                f"Expected input_features (B, T, n_mels), got shape {tuple(input_features.shape)}"
            )

        batch_size, max_frames, feat_dim = input_features.shape
        if feat_dim != self.config.n_mels:
            raise ValueError(f"Expected last dim n_mels={self.config.n_mels}, got {feat_dim}")

        if attention_mask is None:
            lengths = torch.full(
                (batch_size,), max_frames, dtype=torch.long, device=input_features.device
            )
        else:
            lengths = attention_mask.to(dtype=torch.long).sum(dim=-1)

        audio_bct = input_features.transpose(1, 2).contiguous()
        # Cast inputs to the encoder's parameter dtype before forward.
        # The dataloader emits float32 mels, but under Megatron's bf16/fp16
        # wrapper the encoder parameters (incl. ``NGPTStackingSubsampling``'s
        # learnable ``pad_frame``) live in low precision; mismatched dtypes
        # break ``x[mask] = self.pad_frame`` (index_put requires matching
        # dtypes), so cast at the audio encoder boundary.
        encoder_dtype = next(self.encoder.parameters(), audio_bct).dtype
        if audio_bct.dtype != encoder_dtype:
            audio_bct = audio_bct.to(dtype=encoder_dtype)
        enc_out, lengths_out = self.encoder(audio_bct, lengths)

        enc_out = enc_out.transpose(1, 2).contiguous()
        max_len = enc_out.shape[1]
        steps = torch.arange(max_len, device=enc_out.device).unsqueeze(0)
        output_mask = steps < lengths_out.unsqueeze(1).to(device=enc_out.device)
        return enc_out, output_mask

    def forward_packed(
        self, input_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> PackedAudioEmbeddings:
        """Encode mel features into packed (padding-free) audio embeddings."""
        if input_features.ndim != 3:
            raise ValueError(
                f"Expected input_features (B, T, n_mels), got shape {tuple(input_features.shape)}"
            )

        batch_size, max_frames, feat_dim = input_features.shape
        if feat_dim != self.config.n_mels:
            raise ValueError(f"Expected last dim n_mels={self.config.n_mels}, got {feat_dim}")

        if attention_mask is None:
            lengths = torch.full(
                (batch_size,), max_frames, dtype=torch.long, device=input_features.device
            )
        else:
            lengths = attention_mask.to(dtype=torch.long).sum(dim=-1)

        audio_bct = input_features.transpose(1, 2).contiguous()
        encoder_dtype = next(self.encoder.parameters(), audio_bct).dtype
        if audio_bct.dtype != encoder_dtype:
            audio_bct = audio_bct.to(dtype=encoder_dtype)

        enc_out, lengths_out = self.encoder(audio_bct, lengths, return_packed=True)
        return PackedAudioEmbeddings(
            embeddings=enc_out, lengths=lengths_out.to(dtype=torch.int32, device=enc_out.device)
        )
