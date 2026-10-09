# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Token layouts for independent shared-prefix hybrid execution groups."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SharedPrefixLayout:
    """One exact-prompt star packed as ``[prefix, completion_1, ..., completion_G]``."""

    prefix_len: int
    completion_lens: Sequence[int]
    logical_completion_lens: Sequence[int] | None = None
    padding_multiple: int | None = None

    def __post_init__(self) -> None:
        prefix_len = int(self.prefix_len)
        completion_lens = tuple(int(length) for length in self.completion_lens)
        if prefix_len < 1:
            raise ValueError("shared-prefix layout requires a non-empty prefix")
        if not completion_lens or any(length < 1 for length in completion_lens):
            raise ValueError("shared-prefix layout requires one or more non-empty completions")
        object.__setattr__(self, "prefix_len", prefix_len)
        object.__setattr__(self, "completion_lens", completion_lens)
        if self.logical_completion_lens is not None:
            logical_completion_lens = tuple(int(length) for length in self.logical_completion_lens)
            if len(logical_completion_lens) != len(completion_lens) or any(
                logical < 1 or logical > physical
                for logical, physical in zip(logical_completion_lens, completion_lens, strict=True)
            ):
                raise ValueError(
                    "logical completion lengths must be positive, match the physical "
                    "branch count, and not exceed physical completion lengths"
                )
            object.__setattr__(self, "logical_completion_lens", logical_completion_lens)
        if (self.logical_completion_lens is None) != (self.padding_multiple is None):
            raise ValueError(
                "logical completion lengths and padding_multiple must be provided together"
            )
        if self.padding_multiple is not None:
            if isinstance(self.padding_multiple, bool) or not isinstance(
                self.padding_multiple, int
            ):
                raise ValueError("shared-prefix padding_multiple must be an integer")
            if self.padding_multiple < 1:
                raise ValueError("shared-prefix padding_multiple must be positive")

    @property
    def total_len(self) -> int:
        """Return the physical token count of this star."""
        return self.prefix_len + sum(self.completion_lens)

    def iter_roots(self) -> Iterator[tuple[int, SharedPrefixLayout]]:
        """Yield canonical offsets and independent prompt stars."""
        yield 0, self

    @property
    def dense_branch_lengths(self) -> tuple[int, ...]:
        """Return each completion length including its independent prompt copy."""
        return tuple(self.prefix_len + length for length in self.completion_lens)

    @property
    def forest(self) -> list[tuple[int, int, list[int]]]:
        """Represent this star as a root and its completion lengths."""
        return [(0, self.prefix_len, list(self.completion_lens))]

    def completion_slices(self) -> tuple[slice, ...]:
        """Return physical slices for the disjoint completion segments."""
        slices = []
        start = self.prefix_len
        for length in self.completion_lens:
            slices.append(slice(start, start + length))
            start += length
        return tuple(slices)

    def dense_branch_indices(self, device: torch.device | str) -> tuple[Tensor, ...]:
        """Return global-star indices for conventional prompt-completion sequences.

        MTP's token shifts are defined on an ordinary causal sequence and must
        never cross between sibling completion branches.  These indices provide
        the exact inverse of prompt deduplication: the prompt is repeated once
        for each physical completion, including that completion's ordinary
        per-sequence padding.
        """
        prompt = torch.arange(self.prefix_len, device=device, dtype=torch.long)
        return tuple(
            torch.cat(
                (prompt, torch.arange(branch.start, branch.stop, device=device, dtype=torch.long))
            )
            for branch in self.completion_slices()
        )

    def position_ids(self, device: torch.device | str) -> Tensor:
        """Prefix-continued RoPE positions for the packed star."""
        pieces = [torch.arange(self.prefix_len, device=device, dtype=torch.long)]
        pieces.extend(
            torch.arange(self.prefix_len, self.prefix_len + length, device=device, dtype=torch.long)
            for length in self.completion_lens
        )
        return torch.cat(pieces)

    def padded_position_ids(self, physical_len: int, device: torch.device | str) -> Tensor:
        """Return global star positions plus inert positions for trailing CP padding."""
        physical_len = int(physical_len)
        if physical_len < self.total_len:
            raise ValueError(
                f"physical length {physical_len} is shorter than layout {self.total_len}"
            )
        positions = self.position_ids(device)
        if physical_len == self.total_len:
            return positions
        return torch.cat(
            [positions, torch.zeros(physical_len - self.total_len, device=device, dtype=torch.long)]
        )

    def padded_token_multiplicities(
        self,
        physical_len: int,
        device: torch.device | str,
        *,
        exclude_sequence_padding: bool = False,
    ) -> Tensor:
        """Dense-baseline multiplicity for each physical shared-prefix token.

        Prompt tokens occur once per completion in a conventional rollout batch.
        Branch tokens, including ordinary per-sequence padding, have unit
        multiplicity unless exclude_sequence_padding is requested. Trailing topology-only padding
        is inert.
        """
        physical_len = int(physical_len)
        if physical_len < self.total_len:
            raise ValueError(
                f"physical length {physical_len} is shorter than layout {self.total_len}"
            )
        multiplicities = torch.cat(
            [
                torch.full(
                    (self.prefix_len,),
                    len(self.completion_lens),
                    device=device,
                    dtype=torch.float32,
                ),
                torch.ones(sum(self.completion_lens), device=device, dtype=torch.float32),
            ]
        )
        if exclude_sequence_padding:
            # Hybridep's dense input mask excludes ordinary sequence padding.
            # Keep the legacy unmasked convention as the default for other callers.
            logical_lengths = self.logical_completion_lens or self.completion_lens
            offset = self.prefix_len
            for logical, physical in zip(logical_lengths, self.completion_lens):
                if logical < physical:
                    multiplicities[offset + logical : offset + physical] = 0
                offset += physical
        if physical_len == self.total_len:
            return multiplicities
        return torch.cat(
            [
                multiplicities,
                torch.zeros(physical_len - self.total_len, device=device, dtype=torch.float32),
            ]
        )

    @staticmethod
    def cp_local_indices(
        physical_len: int, cp_size: int, cp_rank: int, device: torch.device | str
    ) -> Tensor:
        """Global indices owned by one rank under standard two-chunk CP zigzag."""
        physical_len, cp_size, cp_rank = int(physical_len), int(cp_size), int(cp_rank)
        if cp_size < 1 or not 0 <= cp_rank < cp_size:
            raise ValueError(f"invalid CP geometry: {cp_size=}, {cp_rank=}")
        if physical_len % (2 * cp_size):
            raise ValueError(
                f"physical length {physical_len} must be divisible by 2 * CP size {cp_size}"
            )
        chunk = physical_len // (2 * cp_size)
        front = torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk, device=device)
        back_chunk = 2 * cp_size - cp_rank - 1
        back = torch.arange(back_chunk * chunk, (back_chunk + 1) * chunk, device=device)
        return torch.cat([front, back]).to(torch.long)


@dataclass(frozen=True)
class SharedPrefixForestLayout:
    """Independent prompt stars concatenated into one hybrid forward.

    Each branch retains its ordinary dense padding. Only the complete forest
    receives topology padding; roots need no individual alignment because the
    Mamba CP transform restores canonical token order before scanning them.

    ``mtp_loss_group_root_counts`` partitions consecutive roots into the original
    independently normalized MTP forwards. Empty counts preserve one group per
    root. Grouping loss normalization never joins causal sequences.
    """

    roots: tuple[SharedPrefixLayout, ...]
    mtp_loss_group_root_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", tuple(self.roots))
        if not self.roots or any(not isinstance(root, SharedPrefixLayout) for root in self.roots):
            raise ValueError("a shared-prefix forest requires one or more star layouts")
        if len({root.padding_multiple for root in self.roots}) != 1:
            raise ValueError("all shared-prefix roots must use the same padding multiple")
        if isinstance(self.mtp_loss_group_root_counts, (str, bytes)):
            raise ValueError("MTP loss group root counts must be a sequence of integers")
        try:
            counts = tuple(self.mtp_loss_group_root_counts)
        except TypeError:
            raise ValueError("MTP loss group root counts must be a sequence of integers") from None
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 1 for count in counts
        ):
            raise ValueError("MTP loss group root counts must be positive integers")
        if counts and sum(counts) != len(self.roots):
            raise ValueError("MTP loss group root counts must partition every forest root")
        object.__setattr__(self, "mtp_loss_group_root_counts", counts)

    @property
    def total_len(self) -> int:
        """Return the physical token count across all independent stars."""
        return sum(root.total_len for root in self.roots)

    @property
    def padding_multiple(self) -> int | None:
        """Return the shared branch-padding multiple, if specified."""
        return self.roots[0].padding_multiple

    @property
    def dense_branch_lengths(self) -> tuple[int, ...]:
        """Return dense branch lengths in forest order."""
        return tuple(length for root in self.roots for length in root.dense_branch_lengths)

    @property
    def mtp_loss_group_lengths(self) -> tuple[int, ...]:
        """Global dense-expanded token lengths of consecutive MTP loss groups.

        Ordinary per-branch padding is included; forest-only topology padding
        never enters MTP. Each length is divided by CP size after branch packing.
        """
        counts = self.mtp_loss_group_root_counts or (1,) * len(self.roots)
        lengths = []
        start = 0
        for count in counts:
            lengths.append(
                sum(sum(root.dense_branch_lengths) for root in self.roots[start : start + count])
            )
            start += count
        return tuple(lengths)

    def iter_roots(self) -> Iterator[tuple[int, SharedPrefixLayout]]:
        """Yield each root with its global token offset."""
        offset = 0
        for root in self.roots:
            yield offset, root
            offset += root.total_len

    @property
    def forest(self) -> list[tuple[int, int, list[int]]]:
        """Return root offsets and lengths for the attention forest."""
        return [
            (offset, root.prefix_len, list(root.completion_lens))
            for offset, root in self.iter_roots()
        ]

    def dense_branch_indices(self, device: torch.device | str) -> tuple[Tensor, ...]:
        """Return independent dense branches without crossing root boundaries."""
        return tuple(
            indices + offset
            for offset, root in self.iter_roots()
            for indices in root.dense_branch_indices(device)
        )

    def position_ids(self, device: torch.device | str) -> Tensor:
        """Return per-branch positions in physical forest order."""
        return torch.cat([root.position_ids(device) for root in self.roots])

    def padded_position_ids(self, physical_len: int, device: torch.device | str) -> Tensor:
        """Extend forest positions to the requested physical length."""
        if physical_len < self.total_len:
            raise ValueError("physical length is shorter than shared-prefix forest")
        return F.pad(self.position_ids(device), (0, physical_len - self.total_len))

    def padded_token_multiplicities(
        self,
        physical_len: int,
        device: torch.device | str,
        *,
        exclude_sequence_padding: bool = False,
    ) -> Tensor:
        """Count logical token copies represented by each physical forest row."""
        if physical_len < self.total_len:
            raise ValueError("physical length is shorter than shared-prefix forest")
        weights = torch.cat(
            [
                root.padded_token_multiplicities(
                    root.total_len, device, exclude_sequence_padding=exclude_sequence_padding
                )
                for root in self.roots
            ]
        )
        return F.pad(weights, (0, physical_len - self.total_len))

    cp_local_indices = staticmethod(SharedPrefixLayout.cp_local_indices)
