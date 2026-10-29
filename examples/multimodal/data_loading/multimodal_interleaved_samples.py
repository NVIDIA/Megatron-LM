# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Restorable interleaved-pretraining records, independent of a model's batch schema."""

import torch

from megatron.energon import Sample
from megatron.energon.edataclass import edataclass

from .conversation_sample import ImageMedia
from .image_processing import ImageTilingParams


@edataclass
class TokenizedText:
    """Prepared text IDs in the consumer's vocabulary; never re-tokenized."""

    tokens: list[int]
    token_loss_mask: list[bool] | None = None


@edataclass
class InterleavedDocument(Sample):
    """An ordered interleaved document with optional prepared-token supervision.

    Supervision masks are supplied externally; this type does not render or parse chat roles.
    """

    parts: list[str | TokenizedText | ImageMedia]


@edataclass
class EncodedDocument(Sample):
    """Compact tokens and aligned targets, with image costs computed before decoding."""

    tokens: torch.Tensor
    labels: torch.Tensor
    image_params: list[ImageTilingParams]
    image_positions: list[int]
    length_prefix: list[int]


@edataclass
class InterleavedPackedSample(Sample):
    """A selected pack with lazy images and post-expansion sequence boundaries."""

    tokens: torch.Tensor
    labels: torch.Tensor
    image_params: list[ImageTilingParams]
    cu_seqlens: torch.Tensor
    fragments: list[tuple[str, int, int]]
