# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Modality-agnostic contract between the dynamic engine and a multimodal model.

The engine knows only these three types. Everything model-specific -- how images are tiled,
which vision tower runs, how placeholder spans are laid out -- lives behind
`MultimodalEmbeddingProvider`, so adding a second multimodal model requires no engine change.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch


@dataclass
class MultimodalData:
    """Raw, undecoded media attached to a request.

    Element types are provider-defined; the Nemotron Omni provider accepts PIL images, numpy
    arrays, or file paths for `images`, a list of frames or a video path for each entry of
    `videos`, and `(waveform, sample_rate)` pairs for `audios`.
    """

    images: List[Any] = field(default_factory=list)
    videos: List[Any] = field(default_factory=list)
    audios: List[Any] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Whether any media is attached."""
        return not (self.images or self.videos or self.audios)


@dataclass
class ProcessedMultimodalPrompt:
    """A prompt whose placeholder spans have been expanded and encoded.

    Attributes:
        prompt_tokens: Expanded prompt, int64, on GPU. Placeholder spans are materialized as
            real token ids, so all downstream token accounting (block counts, chunk lengths,
            `num_tokens_to_generate`) works unchanged.
        mm_embeddings: `[N_mm, hidden_size]`, params_dtype, on GPU. Encoder rows in the same
            order as `mm_embed_positions`.
        mm_embed_positions: `[N_mm]`, int64, on CPU, sorted ascending. Indices into
            `prompt_tokens` that must receive an encoder row instead of an embedding-table
            lookup. Kept on CPU so the context can bisect it without a device sync.
        content_hash: Digest over the decoded media, stable across processes. Reserved for a
            future prefix-cache extension that chains it into the block hashes.
    """

    prompt_tokens: torch.Tensor
    mm_embeddings: torch.Tensor
    mm_embed_positions: torch.Tensor
    content_hash: Optional[int] = None

    def __post_init__(self):
        assert self.mm_embeddings.shape[0] == self.mm_embed_positions.shape[0], (
            f"{self.mm_embeddings.shape[0]} embedding rows for "
            f"{self.mm_embed_positions.shape[0]} positions"
        )
        assert (
            self.mm_embed_positions.device.type == "cpu"
        ), "mm_embed_positions must stay on CPU; the context bisects it on the host"
        if self.mm_embed_positions.numel() > 0:
            assert int(self.mm_embed_positions[-1]) < self.prompt_tokens.shape[0], (
                f"position {int(self.mm_embed_positions[-1])} is past the end of a "
                f"{self.prompt_tokens.shape[0]}-token prompt"
            )


class MultimodalEmbeddingProvider(abc.ABC):
    """Turns a text prompt plus raw media into an expanded, encoded prompt.

    Implementations run at request admission, outside the graphed per-step region: encoder
    output shape varies per request, so running a tower inside the step would break CUDA graph
    capture and would re-run the tower once per prefill chunk.
    """

    @abc.abstractmethod
    def encode(self, prompt: str, multimodal_data: MultimodalData) -> ProcessedMultimodalPrompt:
        """Expand placeholder spans and run the encoders.

        Args:
            prompt (str): Raw prompt text, containing the model's media placeholder markers.
            multimodal_data (MultimodalData): Undecoded media for this request.

        Return:
            (ProcessedMultimodalPrompt) Expanded tokens plus the encoder rows they need.
        """

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        """Language-model hidden size the encoder rows are projected to."""
