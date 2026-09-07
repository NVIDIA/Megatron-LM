"""Immutable host geometry for packed DS4 attention requests."""

from __future__ import annotations


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def padded_sequence_boundaries(
    seq_lens: tuple[int, ...], *, cp_size: int, tp_size: int = 1
) -> tuple[int, ...]:
    """Build the host-side padded THD boundaries used by DS4 contiguous CP."""
    if cp_size < 1 or tp_size < 1:
        raise ValueError("cp_size and tp_size must be positive")
    if not seq_lens or any(
        isinstance(length, bool) or not isinstance(length, int) or length <= 0
        for length in seq_lens
    ):
        raise ValueError("seq_lens must be a non-empty tuple of positive integers")
    alignment = tp_size * (2 * cp_size if cp_size > 1 else 1)
    boundaries = [0]
    for length in seq_lens:
        boundaries.append(boundaries[-1] + _round_up(length, alignment))
    return tuple(boundaries)


def compressed_sequence_boundaries(
    sequence_boundaries: tuple[int, ...], *, ratio: int
) -> tuple[int, ...]:
    """Derive request-local compressed boundaries with floor-per-request math."""
    if ratio < 1:
        raise ValueError("ratio must be positive")
    compressed = [0]
    for start, end in zip(sequence_boundaries, sequence_boundaries[1:]):
        if end <= start:
            raise ValueError("sequence boundaries must be strictly increasing")
        compressed.append(compressed[-1] + (end - start) // ratio)
    return tuple(compressed)
