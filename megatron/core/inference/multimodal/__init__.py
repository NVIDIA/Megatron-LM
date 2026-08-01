# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from megatron.core.inference.multimodal.openai_content import (
    ExtractedContent,
    extract_media_from_content,
    extract_media_from_messages,
)
from megatron.core.inference.multimodal.types import (
    MultimodalData,
    MultimodalEmbeddingProvider,
    ProcessedMultimodalPrompt,
)

__all__ = [
    "ExtractedContent",
    "MultimodalData",
    "MultimodalEmbeddingProvider",
    "ProcessedMultimodalPrompt",
    "extract_media_from_content",
    "extract_media_from_messages",
]
