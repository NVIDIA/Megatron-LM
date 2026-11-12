# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Logical lengths and two levels of physical padding, with no GPU dependency."""

from dataclasses import asdict, dataclass

import hashlib

import json

GEOMETRIES = ((32, 4, 128), (64, 4, 128), (16, 2, 256),
              (32, 2, 256), (32, 8, 128), (64, 8, 128))

PACK_LENGTHS = (4096, 8192, 16384)

def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value

def ceil_multiple(value, alignment):
    positive_int(alignment, "alignment")
    return (value + alignment - 1) // alignment * alignment

def cumulative(lengths):
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return tuple(result)

@dataclass(frozen=True)
class Pack:
    lengths: tuple
    sequence_alignment: int
    pack_alignment: int
    aligned_lengths: tuple
    storage_lengths: tuple
    cu_seqlens: tuple
    cu_seqlens_padded: tuple
    storage_tokens: int
    tail_padding_tokens: int

    @classmethod
    def build(cls, lengths, sequence_alignment=1, pack_alignment=4096,
              *, cp_size=1, static_slots=None):
        values = tuple(lengths)
        if not values:
            raise ValueError("a pack needs at least one real sequence")
        for value in values:
            positive_int(value, "sequence length")
        positive_int(sequence_alignment, "sequence alignment")
        positive_int(pack_alignment, "pack alignment")
        positive_int(cp_size, "CP size")
        aligned = tuple(ceil_multiple(n, sequence_alignment) for n in values)
        # Existing MCore CP packing adds 2*CP storage alignment after dataset
        # alignment. CP1 standalone accepts arbitrary per-sequence alignment.
        stored = (tuple(ceil_multiple(n, 2 * cp_size) for n in aligned)
                  if cp_size > 1 else aligned)
        capacity = ceil_multiple(sum(stored), pack_alignment)
        partitions = 2 * cp_size if cp_size > 1 else 1
        if capacity % partitions:
            raise ValueError("physical pack must divide evenly across 2*CP chunks")
        tail = capacity - sum(stored)
        final = stored[:-1] + (stored[-1] + tail,)
        logical, physical = cumulative(values), cumulative(final)
        if static_slots is not None:
            positive_int(static_slots, "static slots")
            if static_slots < len(values):
                raise ValueError("static slots cannot truncate real sequences")
            logical += (logical[-1],) * (static_slots - len(values))
            physical += (physical[-1],) * (static_slots - len(values))
        return cls(values, sequence_alignment, pack_alignment, aligned, final,
                   logical, physical, capacity, tail)

    @property
    def pad_between_seqs(self):
        # Ignore repeated metadata slots: the tail is not an internal gap.
        count = len(self.lengths)
        return self.cu_seqlens_padded[:count] != self.cu_seqlens[:count]

    @property
    def valid_ranges(self):
        return tuple((self.cu_seqlens_padded[i],
                      self.cu_seqlens_padded[i] + length)
                     for i, length in enumerate(self.lengths))

    @property
    def identity(self):
        # Equivalent executed geometry cannot leak across train/test merely
        # because the input alignment label differs.
        payload = (self.lengths, self.cu_seqlens_padded[:len(self.lengths) + 1])
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

    def record(self):
        return dict(asdict(self), pad_between_seqs=self.pad_between_seqs,
                    valid_ranges=self.valid_ranges, identity=self.identity)
