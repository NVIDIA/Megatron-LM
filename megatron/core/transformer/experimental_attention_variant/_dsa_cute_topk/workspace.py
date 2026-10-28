# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bounded exact top-k selection and merging for arbitrary context capacities."""

import math

import torch
import triton
import triton.language as tl

from .dispatch import CuteTopK


@triton.jit
def chunk_lengths(Lengths, Local, Candidate, CHUNKS: tl.constexpr, K: tl.constexpr):
    """Compute each bounded row length and its exact candidate-union size."""
    row = tl.program_id(0)
    n = tl.load(Lengths + row)
    total = tl.full((), 0, tl.int32)
    for chunk in tl.static_range(CHUNKS):
        length = tl.minimum(tl.maximum(n - chunk * 65536, 0), 65536)
        tl.store(Local + row * CHUNKS + chunk, length)
        total += tl.minimum(length, K)
    tl.store(Candidate + row, total)


@triton.jit
def gather_candidates(
    Scores, Local, Candidates, WIDTH: tl.constexpr, CHUNKS: tl.constexpr, K: tl.constexpr
):
    """Gather original scores for the union of disjoint chunk-local top-k supports."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    slots = tl.arange(0, K)
    index = tl.load(Local + (row * CHUNKS + chunk) * K + slots)
    value = tl.load(Scores + row * WIDTH + chunk * 65536 + index, index >= 0, float("-inf"))
    tl.store(Candidates + (row * CHUNKS + chunk) * K + slots, value)


@triton.jit
def remap(Local, Selected, Out, CHUNKS: tl.constexpr, K: tl.constexpr):
    """Map selected candidate slots back to logical key positions."""
    row = tl.program_id(0)
    slots = tl.arange(0, K)
    selected = tl.load(Selected + row * K + slots)
    local = tl.load(Local + row * CHUNKS * K + selected, selected >= 0, -1)
    tl.store(
        Out + row * K + slots,
        tl.where((selected >= 0) & (local >= 0), selected // K * 65536 + local, -1),
    )


class CuteTopKWorkspace:
    """Use disjoint bounded row chunks, then exact top-k of their union.

    The score stride must include complete 65536-element chunks for long contexts.
    Every buffer and CuTe launch is prepared before graph capture.
    """

    def __init__(self, scores: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor) -> None:
        self.batch, self.width = scores.shape
        self.k = indices.shape[1]
        self.chunks = math.ceil(self.width / 65536)
        self.indices = indices
        self.lengths = lengths
        self.scores = scores
        if self.chunks == 1:
            self.flags = torch.empty(self.batch, device=scores.device, dtype=torch.int32)
            self.select = CuteTopK(scores, lengths, indices, self.flags)
        else:
            if self.width % 65536:
                raise ValueError(
                    "Long-context score stride must be padded to complete 65536-element chunks"
                )
            self.local_lengths = torch.empty(
                self.batch * self.chunks, device=scores.device, dtype=torch.int32
            )
            self.candidate_lengths = torch.empty(
                self.batch, device=scores.device, dtype=torch.int32
            )
            self.local_indices = torch.empty(
                self.batch * self.chunks, self.k, device=scores.device, dtype=torch.int32
            )
            self.candidates = torch.empty(
                self.batch, self.chunks * self.k, device=scores.device, dtype=torch.float32
            )
            self.selected = torch.empty_like(indices)
            self.flags = torch.empty_like(self.local_lengths)
            self.merge_flags = torch.empty_like(lengths)
            self.select = CuteTopK(
                scores.view(self.batch * self.chunks, 65536),
                self.local_lengths,
                self.local_indices,
                self.flags,
            )
            self.merge = CuteTopK(
                self.candidates, self.candidate_lengths, self.selected, self.merge_flags
            )

    def __call__(self) -> None:
        """Select into stable output storage, merging chunk supports when needed."""
        if self.chunks == 1:
            self.select()
        else:
            chunk_lengths[(self.batch,)](
                self.lengths,
                self.local_lengths,
                self.candidate_lengths,
                CHUNKS=self.chunks,
                K=self.k,
            )
            self.select()
            gather_candidates[(self.batch, self.chunks)](
                self.scores,
                self.local_indices,
                self.candidates,
                WIDTH=self.width,
                CHUNKS=self.chunks,
                K=self.k,
            )
            self.merge()
            remap[(self.batch,)](
                self.local_indices, self.selected, self.indices, CHUNKS=self.chunks, K=self.k
            )
