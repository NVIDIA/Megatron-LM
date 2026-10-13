"""Frozen CPU prediction kernel; provenance is recorded in ORIGIN.json."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class SequenceTile:
    sequence_index: int
    q_tokens: int
    kv_tokens: int


@dataclass(frozen=True)
class RoundGeometry:
    rank: int
    step: int
    kv_rank: int
    section: str
    q_physical_tokens: int
    kv_physical_tokens: int
    sequence_tiles: tuple[SequenceTile, ...]

    @property
    def q_valid_tokens(self) -> int:
        return sum(tile.q_tokens for tile in self.sequence_tiles)

    @property
    def kv_valid_tokens(self) -> int:
        return sum(tile.kv_tokens for tile in self.sequence_tiles)

    @property
    def active_sequences(self) -> int:
        return sum(tile.q_tokens > 0 and tile.kv_tokens > 0 for tile in self.sequence_tiles)

    @property
    def attention_area(self) -> int:
        if self.section == "diagonal":
            return sum(tile.q_tokens * (tile.q_tokens + 1) // 2 for tile in self.sequence_tiles)
        return sum(tile.q_tokens * tile.kv_tokens for tile in self.sequence_tiles)

    @property
    def cp1_equivalent_sum_squared_tokens(self) -> int:
        """Causal CP1 work unit equivalent to this round's useful work.

        CP1 causal forward uses ``sum(L_i^2)`` as its QK+PV work feature.
        A no-mask rectangular tile has twice the work of a causal triangle
        with the same Q/K area, hence the factor of two.
        """
        return 2 * self.attention_area

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.update(
            {
                "q_valid_tokens": self.q_valid_tokens,
                "kv_valid_tokens": self.kv_valid_tokens,
                "active_sequences": self.active_sequences,
                "attention_area": self.attention_area,
                "cp1_equivalent_sum_squared_tokens": (self.cp1_equivalent_sum_squared_tokens),
            }
        )
        return payload


def _positive_lengths(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or min(result) <= 0:
        raise ValueError(f"{name} must contain positive lengths")
    return result


def chunk_pair(valid_length: int, padded_length: int, cp_size: int, rank: int) -> tuple[int, int]:
    """Return valid tokens in a rank's first and mirrored second chunks."""
    if cp_size <= 0 or not 0 <= rank < cp_size:
        raise ValueError(f"invalid CP rank {rank} for CP={cp_size}")
    if padded_length < valid_length or padded_length % (2 * cp_size):
        raise ValueError(
            f"padded length {padded_length} must cover valid length {valid_length} "
            f"and be divisible by {2 * cp_size}"
        )
    chunk = padded_length // (2 * cp_size)
    first = min(max(valid_length - rank * chunk, 0), chunk)
    second_chunk_index = 2 * cp_size - rank - 1
    second = min(max(valid_length - second_chunk_index * chunk, 0), chunk)
    return first, second


def build_round_geometries(
    valid_lengths: Sequence[int], padded_lengths: Sequence[int], cp_size: int
) -> list[RoundGeometry]:
    """Build every rank/step tile exactly as TE's causal P2P forward does."""
    valid = _positive_lengths(valid_lengths, "valid_lengths")
    padded = _positive_lengths(padded_lengths, "padded_lengths")
    if len(valid) != len(padded):
        raise ValueError("valid and padded lengths must have the same size")
    if cp_size <= 1:
        raise ValueError("round geometry requires CP > 1")

    chunks = [
        [chunk_pair(length, physical, cp_size, rank) for length, physical in zip(valid, padded)]
        for rank in range(cp_size)
    ]
    physical_per_rank = sum(padded) // cp_size
    rounds = []
    for rank in range(cp_size):
        q_first = [pair[0] for pair in chunks[rank]]
        q_second = [pair[1] for pair in chunks[rank]]
        for step in range(cp_size):
            kv_rank = (rank - step) % cp_size
            kv_first = [pair[0] for pair in chunks[kv_rank]]
            kv_second = [pair[1] for pair in chunks[kv_rank]]
            if step == 0:
                section = "diagonal"
                q_lengths = [a + b for a, b in zip(q_first, q_second)]
                kv_lengths = [a + b for a, b in zip(kv_first, kv_second)]
                q_physical = physical_per_rank
                kv_physical = physical_per_rank
            elif step <= rank:
                section = "lower"
                q_lengths = [a + b for a, b in zip(q_first, q_second)]
                kv_lengths = kv_first
                q_physical = physical_per_rank
                kv_physical = physical_per_rank // 2
            else:
                section = "upper"
                q_lengths = q_second
                kv_lengths = [a + b for a, b in zip(kv_first, kv_second)]
                q_physical = physical_per_rank // 2
                kv_physical = physical_per_rank

            rounds.append(
                RoundGeometry(
                    rank=rank,
                    step=step,
                    kv_rank=kv_rank,
                    section=section,
                    q_physical_tokens=q_physical,
                    kv_physical_tokens=kv_physical,
                    sequence_tiles=tuple(
                        SequenceTile(index, q_length, kv_length)
                        for index, (q_length, kv_length) in enumerate(zip(q_lengths, kv_lengths))
                    ),
                )
            )
    return rounds


def critical_rank_work(rounds: Sequence[RoundGeometry]) -> dict[int, int]:
    work: dict[int, int] = {}
    for item in rounds:
        work[item.rank] = work.get(item.rank, 0) + item.cp1_equivalent_sum_squared_tokens
    return work
