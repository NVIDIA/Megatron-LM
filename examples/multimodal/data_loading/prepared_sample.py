# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepared token parts and lazy images, before model-specific encoding."""

from megatron.energon import Sample
from megatron.energon.edataclass import edataclass

from .conversation_sample import ImageMedia


@edataclass
class TokenizedText:
    """Token IDs already encoded in the consumer's vocabulary."""

    tokens: list[int]


@edataclass
class InterleavedDocument(Sample):
    """An ordered prepared-token document with optional lazy images."""

    parts: list[TokenizedText | ImageMedia]
