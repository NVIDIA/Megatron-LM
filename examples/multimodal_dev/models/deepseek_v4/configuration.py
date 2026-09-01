# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek-V4-Flash-Vision configuration and image-layout helpers."""

import math
from typing import Optional

import torch

from megatron.core.transformer.experimental_attention_variant.csa import SparseAttentionVisibility
from megatron.core.transformer.transformer_config import TransformerConfig

DEEPSEEK_V4_VOCAB_SIZE = 129_280

IMAGE_START = 0
IMAGE_PAD = 1
IMAGE = 2
IMAGE_NEW_LINE = 3
IMAGE_END = 4
NUM_IMAGE_TOKEN_TYPES = 5

VISION_PATCH_SIZE = 14
VISION_DOWNSAMPLE_RATIO = 3
VISION_MAX_IMAGE_TOKENS = 384
VISION_MIN_PIXELS = 147_456
VISION_MAX_WH_RATIO = 8
VISION_ROPE_THETA = 10_000.0
COMPRESS_PAD_TO = 4

# Forty-three decoder blocks. Each HybridModel block is one attention symbol and one MoE symbol.
DEEPSEEK_V4_FLASH_VISION_HYBRID_PATTERN = "WEWECE" + "HECE" * 20
DEEPSEEK_V4_FLASH_VISION_COMPRESS_RATIOS = [0, 0, 4] + [128, 4] * 20


def get_deepseek_v4_vision_config(
    num_layers_override: Optional[int] = None, variant: Optional[str] = None
) -> TransformerConfig:
    """Return the official DeepSeek-V4-Flash-Vision ViT configuration.

    ``variant`` is accepted for the common multimodal registry contract. ``proxy`` keeps the
    official widths and can reduce depth through ``num_layers_override`` so the aligner remains
    compatible with the 4096-wide language decoder.
    """
    if variant not in (None, "flash", "proxy"):
        raise ValueError(f"Unknown DeepSeek-V4-Vision variant '{variant}'.")

    config = TransformerConfig(
        num_layers=32 if num_layers_override is None else num_layers_override,
        hidden_size=1024,
        num_attention_heads=16,
        num_query_groups=16,
        kv_channels=64,
        ffn_hidden_size=2816,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        apply_rope_fusion=False,
        bf16=False,
    )
    # These fields describe the non-Transformer parts of the official vision tower.
    config.vision_patch_size = VISION_PATCH_SIZE
    config.vision_downsample_ratio = VISION_DOWNSAMPLE_RATIO
    config.vision_rope_theta = VISION_ROPE_THETA
    config.vision_out_hidden_size = 4096
    config.vision_max_image_tokens = VISION_MAX_IMAGE_TOKENS
    return config


def image_token_id(image_type: int, vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE) -> int:
    """Return the synthetic token ID for one DeepSeek image token type."""
    if not 0 <= image_type < NUM_IMAGE_TOKEN_TYPES:
        raise ValueError(f"Invalid image token type {image_type}.")
    return vocab_size + image_type


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int, *, device: Optional[torch.device] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build official N-layout token types and the aligner-output permutation."""
    if n_llm_h <= 0 or n_llm_w <= 0:
        raise ValueError(f"Image grid must be positive, got {n_llm_h}x{n_llm_w}.")

    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = (rows // 2 * row_len) % 2 * 2

    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.long,
        device=device,
    )
    order = (
        torch.arange(rows * row_len, device=device)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.long, device=device)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w, device=device
    ).view(n_llm_h, n_llm_w)
    permutation = image_idx[order]
    permutation = permutation[permutation >= 0]

    types = torch.cat(
        (
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.long, device=device),
            torch.tensor([IMAGE_START], dtype=torch.long, device=device),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.long, device=device),
            torch.tensor([IMAGE_END], dtype=torch.long, device=device),
        )
    )
    return types, permutation


def build_aligner_permutation(
    n_vit_h: int,
    n_vit_w: int,
    downsample_ratio: int = VISION_DOWNSAMPLE_RATIO,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the N-layout permutation for one aligned ViT grid."""
    n_llm_h = math.ceil(n_vit_h / downsample_ratio)
    n_llm_w = math.ceil(n_vit_w / downsample_ratio)
    _, permutation = build_image_block(n_llm_h, n_llm_w, 0, device=device)
    return permutation


def build_image_token_visibility(
    input_ids: torch.Tensor,
    vocab_size: int = DEEPSEEK_V4_VOCAB_SIZE,
    max_image_tokens: int = VISION_MAX_IMAGE_TOKENS,
) -> SparseAttentionVisibility:
    """Compute compact bidirectional visibility inside every image token span."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [batch, sequence], got {tuple(input_ids.shape)}.")
    seqlen = input_ids.shape[1]
    positions = torch.arange(seqlen, dtype=torch.int32, device=input_ids.device).unsqueeze(0)
    is_start = input_ids == image_token_id(IMAGE_START, vocab_size)
    is_end = input_ids == image_token_id(IMAGE_END, vocab_size)
    valid = (is_start.cumsum(1) > is_end.cumsum(1)) | is_end
    starts = torch.where(is_start, positions, 0).cummax(1)[0]
    left = (positions - starts) * valid
    ends = torch.where(is_end, positions, seqlen).flip(1).cummin(1)[0].flip(1)
    right = (ends - positions) * valid
    return SparseAttentionVisibility(
        left=left.clamp(max=max_image_tokens - 1),
        right=right.clamp(max=max_image_tokens),
        max_span=max_image_tokens,
    )
