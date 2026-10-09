# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backend-neutral planning primitives for exact shared-prefix GRPO groups.

This module only describes how rows can be packed. It does not build tensors,
attention masks, or backend-specific model inputs. Consequently, a caller can
use the same plan for Megatron, DTensor, or an observational dry run.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from megatron.rl.tree_layout import PackedTreeLayout

MAX_SHARED_PREFIX_BRANCHES = 16
"""Default number of completion branches stored behind one prompt copy.

This is a planning policy, not a kernel limit. Planners that accept a branch
limit default to this value; larger exact-prompt groups are split evenly.
"""


def _round_up(value: int, multiple: int) -> int:
    """Round ``value`` up to a positive alignment without backend imports."""
    return ((value + multiple - 1) // multiple) * multiple


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def split_rows_evenly(
    rows: Sequence[SharedPrefixRow], *, max_completions_per_bin: int
) -> tuple[tuple[SharedPrefixRow, ...], ...]:
    """Split rows into the fewest consecutive chunks of nearly equal size.

    A group of ``max_completions_per_bin + 1`` rows becomes two shared halves
    instead of a full star plus a singleton that shares nothing.
    """
    if not _is_int(max_completions_per_bin) or max_completions_per_bin < 1:
        raise ValueError("max_completions_per_bin must be a positive integer")
    count = -(-len(rows) // max_completions_per_bin)
    base, extra = divmod(len(rows), count) if count else (0, 0)
    chunks = []
    start = 0
    for index in range(count):
        stop = start + base + (index < extra)
        chunks.append(tuple(rows[start:stop]))
        start = stop
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class SharedPrefixRow:
    """One prompt-completion row considered for shared-prefix packing.

    Attributes:
        row_index: Stable index of the row in the caller's input batch.
        group_id: Opaque GRPO rollout-group identity. ``None`` means that the
            row cannot be safely matched with peers.
        prompt_token_ids: Exact prompt token sequence. Rows only share a prefix
            when both this sequence and ``group_id`` are equal.
        completion_length: Number of contiguous completion tokens immediately
            following the prompt in the source row.
    """

    row_index: int
    group_id: str | None
    prompt_token_ids: tuple[int, ...]
    completion_length: int

    def __post_init__(self) -> None:
        if not _is_int(self.row_index) or self.row_index < 0:
            raise ValueError("row_index must be a nonnegative integer")
        if not _is_int(self.completion_length) or self.completion_length < 0:
            raise ValueError("completion_length must be a nonnegative integer")
        if self.group_id is not None and (not isinstance(self.group_id, str) or not self.group_id):
            raise ValueError("group_id must be None or a non-empty string")
        if not isinstance(self.prompt_token_ids, tuple):
            object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
        if not set(map(type, self.prompt_token_ids)) <= {int}:
            raise ValueError("prompt_token_ids must contain integers, excluding booleans")

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens in the source row."""
        return len(self.prompt_token_ids)

    @property
    def total_length(self) -> int:
        """Unpadded token count of the ordinary prompt-completion row."""
        return self.prompt_length + self.completion_length


@dataclass(frozen=True)
class SharedPrefixLayout:
    """CPU representation of one exact-prompt GRPO star.

    Only the source rows and their logical and physical completion lengths are
    stored; every token index is derived from them, so a layout cannot carry
    inconsistent maps. Derived tuples are ordinary Python values computed on
    first access. A model adapter can tensorize them on its own device without
    making the data-plane planner depend on PyTorch.

    ``token_gather_rows`` and ``token_gather_columns`` map every packed token to
    its source-batch coordinate. ``completion_positions`` identify target tokens
    in the packed sequence, while ``predecessor_positions`` identify the logits
    that predict them. ``completion_scatter_*`` map those logprobs back to the
    caller's conventional ``[row, sequence_length - 1]`` next-token view.
    """

    group_id: str
    prompt_token_ids: tuple[int, ...]
    row_indices: tuple[int, ...]
    completion_lengths: tuple[int, ...]
    physical_completion_lengths: tuple[int, ...] = ()
    tree_layout: PackedTreeLayout = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "prompt_token_ids",
            "row_indices",
            "completion_lengths",
            "physical_completion_lengths",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("shared-prefix layouts require a non-empty group_id")
        if not self.prompt_token_ids:
            raise ValueError("a shared-prefix layout requires a non-empty prompt")
        if not self.row_indices or len(self.row_indices) != len(self.completion_lengths):
            raise ValueError("each completion node must own one source row")
        if len(set(self.row_indices)) != len(self.row_indices) or any(
            not _is_int(row) or row < 0 for row in self.row_indices
        ):
            raise ValueError("shared-prefix row indices must be unique nonnegative integers")
        physical_lengths = self.physical_completion_lengths or self.completion_lengths
        if len(physical_lengths) != len(self.completion_lengths):
            raise ValueError(
                "physical_completion_lengths and completion_lengths must have equal length"
            )
        object.__setattr__(self, "physical_completion_lengths", physical_lengths)
        # The tree validates integral, positive and physically bounded spans.
        tree = PackedTreeLayout.from_shared_prefix(
            self.prompt_length, physical_lengths, logical_completion_lens=self.completion_lengths
        )
        object.__setattr__(self, "tree_layout", tree)

    def iter_roots(self) -> Iterator[tuple[int, SharedPrefixLayout]]:
        """Yield independent roots with their canonical physical offsets."""
        yield 0, self

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens stored once in this layout."""
        return len(self.prompt_token_ids)

    @property
    def total_length(self) -> int:
        """Logical prompt-once token count, excluding branch padding tails."""
        return self.prompt_length + sum(self.completion_lengths)

    @property
    def physical_total_length(self) -> int:
        """Prompt-once length including each branch's ordinary packing tail."""
        return self.tree_layout.total_len

    @property
    def branch_starts(self) -> tuple[int, ...]:
        """Packed start of each completion branch."""
        return self.tree_layout.node_start[1:]

    @cached_property
    def position_ids(self) -> tuple[int, ...]:
        """Prefix-continued position of every packed token."""
        return self.tree_layout.position_ids()

    @cached_property
    def token_gather_rows(self) -> tuple[int, ...]:
        """Source row of every packed token."""
        return (self.row_indices[0],) * self.prompt_length + tuple(
            row
            for row, length in zip(self.row_indices, self.physical_completion_lengths, strict=True)
            for _ in range(length)
        )

    @cached_property
    def token_gather_columns(self) -> tuple[int, ...]:
        """Source column of every packed token, including branch padding tails."""
        prompt_length = self.prompt_length
        return tuple(range(prompt_length)) + tuple(
            column
            for length in self.physical_completion_lengths
            for column in range(prompt_length, prompt_length + length)
        )

    @cached_property
    def completion_positions(self) -> tuple[int, ...]:
        """Packed position of every real completion token."""
        return tuple(
            position
            for start, length in zip(self.branch_starts, self.completion_lengths, strict=True)
            for position in range(start, start + length)
        )

    @cached_property
    def predecessor_positions(self) -> tuple[int, ...]:
        """Packed position whose logits predict each completion token."""
        # The first token of every branch is predicted by the last prompt token.
        return tuple(
            position
            for start, length in zip(self.branch_starts, self.completion_lengths, strict=True)
            for position in (self.prompt_length - 1, *range(start, start + length - 1))
        )

    @cached_property
    def completion_scatter_rows(self) -> tuple[int, ...]:
        """Source row of each completion token's logprob."""
        return tuple(
            row
            for row, length in zip(self.row_indices, self.completion_lengths, strict=True)
            for _ in range(length)
        )

    @cached_property
    def completion_scatter_columns(self) -> tuple[int, ...]:
        """Column of each completion logprob in the ``[row, sequence - 1]`` view."""
        first_column = self.prompt_length - 1
        return tuple(
            column
            for length in self.completion_lengths
            for column in range(first_column, first_column + length)
        )

    @cached_property
    def physical_padding_positions(self) -> tuple[int, ...]:
        """Packed positions of branch padding tails."""
        return self.tree_layout.padding_positions()


@dataclass(frozen=True)
class SharedPrefixForestLayout:
    """Multiple exact-prompt stars sharing a physical execution unit.

    Source rows stay globally addressed. Every token offset is shifted by the
    preceding roots' physical lengths, including ordinary branch padding.
    Topology padding is added once, after the complete forest.
    ``mtp_loss_group_root_counts`` optionally groups consecutive roots into
    original auxiliary-loss normalization bins; empty retains one group per root.
    """

    roots: tuple[SharedPrefixLayout, ...]
    mtp_loss_group_root_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "roots", tuple(self.roots))
        if not self.roots or any(not isinstance(root, SharedPrefixLayout) for root in self.roots):
            raise ValueError("a forest requires one or more shared-prefix stars")
        if len(set(self.row_indices)) != len(self.row_indices):
            raise ValueError("shared-prefix forest roots must own disjoint source rows")
        if not isinstance(self.mtp_loss_group_root_counts, (tuple, list)):
            raise ValueError("MTP loss group root counts must partition all forest roots")
        counts = tuple(self.mtp_loss_group_root_counts)
        if counts and (
            any(type(count) is not int or count < 1 for count in counts)
            or sum(counts) != len(self.roots)
        ):
            raise ValueError("MTP loss group root counts must partition all forest roots")
        object.__setattr__(self, "mtp_loss_group_root_counts", counts)

    def iter_roots(self) -> Iterator[tuple[int, SharedPrefixLayout]]:
        """Yield each root with its global physical token offset."""
        offset = 0
        for root in self.roots:
            yield offset, root
            offset += root.physical_total_length

    @cached_property
    def tree_layout(self) -> PackedTreeLayout:
        """Canonical multi-level descriptor, with globally rebased node parents."""
        return PackedTreeLayout.concat(tuple(root.tree_layout for root in self.roots))

    @cached_property
    def physical_total_length(self) -> int:
        return self.tree_layout.total_len

    @cached_property
    def total_length(self) -> int:
        return sum(root.total_length for root in self.roots)

    @cached_property
    def row_indices(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.row_indices)

    @cached_property
    def completion_lengths(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.completion_lengths)

    @cached_property
    def physical_completion_lengths(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.physical_completion_lengths)

    @cached_property
    def position_ids(self) -> tuple[int, ...]:
        return self.tree_layout.position_ids()

    @cached_property
    def token_gather_rows(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.token_gather_rows)

    @cached_property
    def token_gather_columns(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.token_gather_columns)

    @cached_property
    def completion_scatter_rows(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.completion_scatter_rows)

    @cached_property
    def completion_scatter_columns(self) -> tuple[int, ...]:
        return tuple(value for root in self.roots for value in root.completion_scatter_columns)

    @cached_property
    def branch_starts(self) -> tuple[int, ...]:
        return tuple(
            offset + value for offset, root in self.iter_roots() for value in root.branch_starts
        )

    @cached_property
    def completion_positions(self) -> tuple[int, ...]:
        return tuple(
            offset + value
            for offset, root in self.iter_roots()
            for value in root.completion_positions
        )

    @cached_property
    def predecessor_positions(self) -> tuple[int, ...]:
        return tuple(
            offset + value
            for offset, root in self.iter_roots()
            for value in root.predecessor_positions
        )

    @cached_property
    def physical_padding_positions(self) -> tuple[int, ...]:
        return tuple(
            offset + value
            for offset, root in self.iter_roots()
            for value in root.physical_padding_positions
        )


def pack_shared_prefix_groups(
    layouts: Sequence[SharedPrefixLayout],
    *,
    bin_capacity: int,
    dense_capacity: int | None,
    padding_multiple: int,
) -> tuple[SharedPrefixLayout | SharedPrefixForestLayout, ...]:
    """Coalesce validated stars under backbone and expanded-MTP token budgets.

    Stable first-fit decreasing uses dense cost as its primary key. Existing
    individual stars exceeding the dense budget remain singleton executions;
    merging never makes that pre-existing situation larger.
    ``dense_capacity=None`` omits the expanded MTP constraint for evaluation.
    """
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (bin_capacity, padding_multiple)
        + (() if dense_capacity is None else (dense_capacity,))
    ):
        raise ValueError("group packing capacities and padding multiple must be positive integers")
    if bin_capacity % padding_multiple or (
        dense_capacity is not None and dense_capacity % padding_multiple
    ):
        raise ValueError("group packing capacities must be aligned to the padding multiple")

    def dense_length(layout: SharedPrefixLayout) -> int:
        return len(layout.row_indices) * layout.prompt_length + sum(
            layout.physical_completion_lengths
        )

    bins: list[list[SharedPrefixLayout]] = []
    trunk_lengths: list[int] = []
    dense_lengths: list[int] = []
    for root in sorted(
        layouts, key=lambda item: (-dense_length(item), -item.physical_total_length)
    ):
        trunk = root.physical_total_length
        dense = dense_length(root)
        if _round_up(trunk, padding_multiple) > bin_capacity:
            raise ValueError("shared-prefix root exceeds the aligned backbone budget")
        for index in range(len(bins)):
            combined = trunk_lengths[index] + trunk
            if _round_up(combined, padding_multiple) <= bin_capacity and (
                dense_capacity is None or dense_lengths[index] + dense <= dense_capacity
            ):
                bins[index].append(root)
                trunk_lengths[index] = combined
                dense_lengths[index] += dense
                break
        else:
            bins.append([root])
            trunk_lengths.append(trunk)
            dense_lengths.append(dense)
    return tuple(
        roots[0] if len(roots) == 1 else SharedPrefixForestLayout(tuple(roots)) for roots in bins
    )


@dataclass(frozen=True, slots=True)
class SharedPrefixPlan:
    """Deterministic partition of input rows into shared stars and fallbacks.

    ``fallback_row_indices`` lists, in ascending order, every row that keeps
    ordinary causal packing because it has no exact-prompt peer that fits.
    """

    shared_bins: tuple[SharedPrefixLayout, ...]
    fallback_row_indices: tuple[int, ...]


def build_shared_prefix_layout(
    rows: Sequence[SharedPrefixRow],
    *,
    sequence_length_pad_multiple: int = 1,
    allow_singleton: bool = False,
) -> SharedPrefixLayout:
    """Build an exact-prompt star, or an explicitly allowed single causal row.

    The first row supplies the single stored copy of the prompt. Each completion
    supplies its own contiguous suffix. The first completion token of every
    branch is predicted by the final prompt position; later tokens are predicted
    by the preceding token in the same branch.

    Args:
        rows: Rows with the same non-null group identity and exact prompt tokens.
        allow_singleton: Allow one ordinary causal row as an independent forest
            root. This does not establish prefix reuse or change group matching.

    Returns:
        A backend-neutral star layout whose indices reference the caller's input
        row indices.

    Raises:
        ValueError: If no rows are supplied, a singleton is not explicitly
            allowed, or rows do not form a valid exact-prompt group.
    """
    if not rows or (len(rows) < 2 and not allow_singleton):
        raise ValueError("a shared-prefix layout requires at least two rows")
    if not _is_int(sequence_length_pad_multiple) or sequence_length_pad_multiple < 1:
        raise ValueError("sequence_length_pad_multiple must be a positive integer")

    first = rows[0]
    group_id = first.group_id
    if group_id is None:
        raise ValueError("shared-prefix rows must have a group_id")
    if first.prompt_length == 0:
        raise ValueError("a shared-prefix layout requires a non-empty prompt")

    seen_row_indices: set[int] = set()
    for row in rows:
        if row.row_index in seen_row_indices:
            raise ValueError(f"duplicate row_index {row.row_index}")
        seen_row_indices.add(row.row_index)
        if row.group_id != group_id:
            raise ValueError("all shared-prefix rows must have the same group_id")
        if row.prompt_token_ids != first.prompt_token_ids:
            raise ValueError("all shared-prefix rows must have identical prompt tokens")
        if row.completion_length == 0:
            raise ValueError("all shared-prefix rows must have a non-empty completion")

    prompt_length = first.prompt_length
    return SharedPrefixLayout(
        group_id=group_id,
        prompt_token_ids=first.prompt_token_ids,
        row_indices=tuple(row.row_index for row in rows),
        completion_lengths=tuple(row.completion_length for row in rows),
        physical_completion_lengths=tuple(
            _round_up(row.total_length, sequence_length_pad_multiple) - prompt_length
            for row in rows
        ),
    )


def plan_shared_prefix_bins(
    rows: Sequence[SharedPrefixRow],
    *,
    bin_capacity: int,
    max_completions_per_bin: int = MAX_SHARED_PREFIX_BRANCHES,
    sequence_length_pad_multiple: int = 1,
) -> SharedPrefixPlan:
    """Partition exact-prompt GRPO rows into shared stars and mixed fallbacks.

    Planning is invariant to the input sequence order: rows are first ordered by
    ``row_index``, exact groups are visited by their lowest row index, and each
    group uses first-fit decreasing with ``row_index`` as the tie-breaker.

    A shared bin always has at least two completions and satisfies both the token
    capacity and completion-count limit. Rows that cannot share are retained as
    explicit fallbacks instead of being silently dropped.

    Args:
        rows: Candidate prompt-completion rows.
        bin_capacity: Maximum number of deduplicated tokens in one shared bin. It
            must be a multiple of ``sequence_length_pad_multiple`` so that a
            planned star still fits after topology padding.
        max_completions_per_bin: Maximum branches behind one stored prompt. An
            exact group above it is first capped to an even share, so 17 rows
            at limit 16 become stars of 9 and 8 when the token budget allows.
        sequence_length_pad_multiple: Ordinary per-sequence packing alignment.

    Returns:
        A complete, deterministic partition of the input row indices.

    Raises:
        ValueError: If planner limits are invalid or row indices are duplicated.
    """
    if not _is_int(sequence_length_pad_multiple) or sequence_length_pad_multiple < 1:
        raise ValueError("sequence_length_pad_multiple must be a positive integer")
    if not _is_int(bin_capacity) or bin_capacity < 1 or bin_capacity % sequence_length_pad_multiple:
        raise ValueError(
            "bin_capacity must be a positive integer multiple of sequence_length_pad_multiple, "
            f"got {bin_capacity!r}"
        )
    if not _is_int(max_completions_per_bin) or max_completions_per_bin < 2:
        raise ValueError("max_completions_per_bin must be an integer of at least 2")

    def padded_row_length(row: SharedPrefixRow) -> int:
        return _round_up(row.total_length, sequence_length_pad_multiple)

    def physical_completion_length(row: SharedPrefixRow) -> int:
        return padded_row_length(row) - row.prompt_length

    ordered_rows = sorted(rows, key=lambda row: row.row_index)
    row_indices = [row.row_index for row in ordered_rows]
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("row_index values must be unique")

    fallbacks: list[int] = []
    exact_groups: dict[tuple[str, tuple[int, ...]], list[SharedPrefixRow]] = {}
    for row in ordered_rows:
        if (
            row.group_id is None
            or row.prompt_length == 0
            or row.completion_length == 0
            or padded_row_length(row) > bin_capacity
        ):
            fallbacks.append(row.row_index)
        else:
            exact_groups.setdefault((row.group_id, row.prompt_token_ids), []).append(row)

    shared_bins: list[SharedPrefixLayout] = []
    for exact_rows in exact_groups.values():
        if len(exact_rows) < 2:
            fallbacks.append(exact_rows[0].row_index)
            continue

        prompt_length = exact_rows[0].prompt_length
        # Spread a group past the branch limit evenly (17 -> 9+8, not a
        # 16-star plus an unshared singleton), as split_rows_evenly does.
        branch_limit = -(-len(exact_rows) // -(-len(exact_rows) // max_completions_per_bin))
        physical_bins: list[list[SharedPrefixRow]] = []
        bin_completion_tokens: list[int] = []
        rows_by_size = sorted(
            exact_rows,
            key=lambda row: (
                -physical_completion_length(row),
                -row.completion_length,
                row.row_index,
            ),
        )
        for row in rows_by_size:
            destination: int | None = None
            for bin_index, bin_rows in enumerate(physical_bins):
                has_branch_slot = len(bin_rows) < branch_limit
                has_token_space = (
                    prompt_length
                    + bin_completion_tokens[bin_index]
                    + physical_completion_length(row)
                    <= bin_capacity
                )
                if has_branch_slot and has_token_space:
                    destination = bin_index
                    break
            if destination is None:
                physical_bins.append([row])
                bin_completion_tokens.append(physical_completion_length(row))
            else:
                physical_bins[destination].append(row)
                bin_completion_tokens[destination] += physical_completion_length(row)

        for bin_rows in physical_bins:
            if len(bin_rows) >= 2:
                shared_bins.append(
                    build_shared_prefix_layout(
                        bin_rows, sequence_length_pad_multiple=sequence_length_pad_multiple
                    )
                )
            else:
                fallbacks.append(bin_rows[0].row_index)

    return SharedPrefixPlan(
        shared_bins=tuple(shared_bins), fallback_row_indices=tuple(sorted(fallbacks))
    )
