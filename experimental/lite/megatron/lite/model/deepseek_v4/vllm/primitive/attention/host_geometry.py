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


def local_compressed_sequence_boundaries(
    sequence_boundaries: tuple[int, ...],
    *,
    global_start: int,
    local_rows: int,
    ratio: int,
) -> tuple[int, ...]:
    """Derive the compressed rows physically emitted by one contiguous CP rank.

    This mirrors ``CompressorInputCompact``. In particular, a sequence that ends
    at or before this rank's first row contributes no compressed groups, even
    when its end is within the compressor's left-boundary window.
    """
    if ratio < 1:
        raise ValueError("ratio must be positive")
    if global_start < 0 or local_rows < 1:
        raise ValueError("global_start must be non-negative and local_rows positive")

    d_comp = 8 if ratio == 4 else ratio
    local_end = global_start + local_rows
    first_range_group_start = global_start - d_comp
    compressed = [0]
    for seq_start, seq_end in zip(sequence_boundaries, sequence_boundaries[1:]):
        if seq_end <= seq_start:
            raise ValueError("sequence boundaries must be strictly increasing")
        local_seq_end = min(seq_end, local_end)
        if seq_start >= local_seq_end or global_start >= local_seq_end:
            compressed.append(compressed[-1])
            continue

        first_visible_numer = max(0, first_range_group_start - seq_start)
        first_visible_group = (first_visible_numer + ratio - 1) // ratio
        stop_visible_group = (local_seq_end - seq_start) // ratio
        compressed.append(
            compressed[-1] + max(0, stop_visible_group - first_visible_group)
        )
    return tuple(compressed)
