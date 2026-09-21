# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass

import torch


@dataclass
class PackedAudioEmbeddings:
    """Flat valid audio embeddings with one length per source audio."""

    embeddings: torch.Tensor
    lengths: torch.Tensor

    @property
    def cu_seqlens(self) -> torch.Tensor:
        """Return cumulative sequence lengths with a leading zero (int32)."""
        lengths = self.lengths.to(dtype=torch.int32, device=self.embeddings.device)
        return torch.nn.functional.pad(lengths.cumsum(0), (1, 0))

    def pad_to_lengths(self, target_lengths: torch.Tensor) -> "PackedAudioEmbeddings":
        """Zero-pad each packed embedding up to the given per-source target lengths."""
        target_lengths = target_lengths.to(dtype=torch.int32, device=self.embeddings.device)
        lengths = self.lengths.to(dtype=torch.int32, device=self.embeddings.device)
        target_lengths = torch.maximum(target_lengths, lengths)
        if torch.equal(target_lengths, lengths):
            return self

        chunks = []
        offset = 0
        hidden_size = self.embeddings.shape[-1]
        for length, target_length in zip(lengths.tolist(), target_lengths.tolist()):
            chunk = self.embeddings[offset : offset + length]
            if target_length > length:
                padding = self.embeddings.new_zeros(target_length - length, hidden_size)
                chunk = torch.cat([chunk, padding], dim=0)
            chunks.append(chunk)
            offset += length

        embeddings = (
            torch.cat(chunks, dim=0) if chunks else self.embeddings.new_zeros(0, hidden_size)
        )
        return PackedAudioEmbeddings(embeddings=embeddings, lengths=target_lengths)
