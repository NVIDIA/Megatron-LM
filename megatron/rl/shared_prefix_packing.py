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

import enum
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property

from megatron.rl.tree_layout import PackedTreeLayout


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
        if self.row_index < 0:
            raise ValueError("row_index must be nonnegative")
        if self.completion_length < 0:
            raise ValueError("completion_length must be nonnegative")

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens in the source row."""
        return len(self.prompt_token_ids)

    @property
    def total_length(self) -> int:
        """Unpadded token count of the ordinary prompt-completion row."""
        return self.prompt_length + self.completion_length


@dataclass(frozen=True, slots=True)
class SharedPrefixLayout:
    """CPU representation of one exact-prompt GRPO star.

    Parallel index tuples deliberately remain ordinary Python values. A model
    adapter can tensorize them on its own device without making the data-plane
    planner depend on PyTorch or a specific training backend.

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
    total_length: int
    branch_starts: tuple[int, ...]
    position_ids: tuple[int, ...]
    token_gather_rows: tuple[int, ...]
    token_gather_columns: tuple[int, ...]
    completion_positions: tuple[int, ...]
    predecessor_positions: tuple[int, ...]
    completion_scatter_rows: tuple[int, ...]
    completion_scatter_columns: tuple[int, ...]
    physical_completion_lengths: tuple[int, ...] = ()
    physical_padding_positions: tuple[int, ...] = ()
    tree_layout: PackedTreeLayout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        physical_lengths = self.physical_completion_lengths or self.completion_lengths
        if len(physical_lengths) != len(self.completion_lengths):
            raise ValueError(
                "physical_completion_lengths and completion_lengths must have equal length"
            )
        if any(
            physical < logical
            for physical, logical in zip(physical_lengths, self.completion_lengths, strict=True)
        ):
            raise ValueError(
                "physical completion lengths cannot be shorter than logical completions"
            )
        object.__setattr__(self, "physical_completion_lengths", physical_lengths)
        tree = PackedTreeLayout.from_shared_prefix(
            self.prompt_length, physical_lengths, logical_completion_lens=self.completion_lengths
        )
        if len(self.row_indices) != len(physical_lengths):
            raise ValueError("each completion node must own one source row")
        if self.branch_starts != tree.node_start[1:]:
            raise ValueError("shared-prefix branches must be positive and contiguous")
        object.__setattr__(self, "tree_layout", tree)

    def iter_roots(self) -> Iterator[tuple[int, SharedPrefixLayout]]:
        """Yield independent roots with their canonical physical offsets."""
        yield 0, self

    @property
    def prompt_length(self) -> int:
        """Number of prompt tokens stored once in this layout."""
        return len(self.prompt_token_ids)

    @property
    def baseline_length(self) -> int:
        """Token count if every completion duplicated the prompt."""
        return len(self.row_indices) * self.prompt_length + sum(self.completion_lengths)

    @property
    def physical_total_length(self) -> int:
        """Prompt-once length including each branch's ordinary packing tail."""
        return self.tree_layout.total_len

    @property
    def tokens_saved(self) -> int:
        """Prompt tokens removed relative to ordinary duplicated rows."""
        return self.baseline_length - self.total_length


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
    def baseline_length(self) -> int:
        return sum(root.baseline_length for root in self.roots)

    @property
    def tokens_saved(self) -> int:
        return self.baseline_length - self.total_length

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
        if ((trunk + padding_multiple - 1) // padding_multiple) * padding_multiple > bin_capacity:
            raise ValueError("shared-prefix root exceeds the aligned backbone budget")
        for index in range(len(bins)):
            combined = trunk_lengths[index] + trunk
            if (
                (combined + padding_multiple - 1) // padding_multiple
            ) * padding_multiple <= bin_capacity and (
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


class SharedPrefixFallbackReason(enum.Enum):
    """Reason a row was not placed in a shared-prefix star."""

    MISSING_GROUP_ID = "missing_group_id"
    EMPTY_PROMPT = "empty_prompt"
    EMPTY_COMPLETION = "empty_completion"
    SEQUENCE_EXCEEDS_BIN = "sequence_exceeds_bin"
    NO_EXACT_PROMPT_PEER = "no_exact_prompt_peer"
    PROMPT_MISMATCH = "prompt_mismatch"
    NO_CAPACITY_COMPATIBLE_PEER = "no_capacity_compatible_peer"


@dataclass(frozen=True, slots=True)
class SharedPrefixFallback:
    """One row routed away from shared-prefix packing.

    ``fits_block_diagonal_bin`` distinguishes a normal mixed fallback from an
    oversized row that the existing packer must reject or handle separately.
    """

    row: SharedPrefixRow
    reason: SharedPrefixFallbackReason
    fits_block_diagonal_bin: bool


@dataclass(frozen=True, slots=True)
class SharedPrefixPlan:
    """Deterministic partition of input rows into shared stars and fallbacks."""

    shared_bins: tuple[SharedPrefixLayout, ...]
    fallbacks: tuple[SharedPrefixFallback, ...]

    @property
    def shared_row_indices(self) -> tuple[int, ...]:
        """Input row indices covered by shared-prefix bins."""
        return tuple(row_index for layout in self.shared_bins for row_index in layout.row_indices)

    @property
    def fallback_row_indices(self) -> tuple[int, ...]:
        """Input row indices routed to the fallback path."""
        return tuple(fallback.row.row_index for fallback in self.fallbacks)


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
    if (
        isinstance(sequence_length_pad_multiple, bool)
        or not isinstance(sequence_length_pad_multiple, int)
        or sequence_length_pad_multiple < 1
    ):
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
    token_gather_rows = [first.row_index] * prompt_length
    token_gather_columns = list(range(prompt_length))
    completion_positions: list[int] = []
    predecessor_positions: list[int] = []
    completion_scatter_rows: list[int] = []
    completion_scatter_columns: list[int] = []
    physical_completion_lengths = tuple(
        (
            (row.total_length + sequence_length_pad_multiple - 1)
            // sequence_length_pad_multiple
            * sequence_length_pad_multiple
        )
        - prompt_length
        for row in rows
    )
    tree = PackedTreeLayout.from_shared_prefix(
        prompt_length,
        physical_completion_lengths,
        logical_completion_lens=tuple(row.completion_length for row in rows),
    )
    for row, packed_offset, physical_completion_length, first_predecessor in zip(
        rows, tree.node_start[1:], tree.node_len[1:], tree.first_predecessors()[1:], strict=True
    ):
        token_gather_rows.extend([row.row_index] * physical_completion_length)
        token_gather_columns.extend(
            range(prompt_length, prompt_length + physical_completion_length)
        )
        for completion_offset in range(row.completion_length):
            packed_position = packed_offset + completion_offset
            predecessor_position = (
                first_predecessor if completion_offset == 0 else packed_position - 1
            )
            completion_positions.append(packed_position)
            predecessor_positions.append(predecessor_position)
            completion_scatter_rows.append(row.row_index)
            completion_scatter_columns.append(prompt_length + completion_offset - 1)

    return SharedPrefixLayout(
        group_id=group_id,
        prompt_token_ids=first.prompt_token_ids,
        row_indices=tuple(row.row_index for row in rows),
        completion_lengths=tuple(row.completion_length for row in rows),
        total_length=prompt_length + sum(row.completion_length for row in rows),
        branch_starts=tree.node_start[1:],
        position_ids=tree.position_ids(),
        token_gather_rows=tuple(token_gather_rows),
        token_gather_columns=tuple(token_gather_columns),
        completion_positions=tuple(completion_positions),
        predecessor_positions=tuple(predecessor_positions),
        completion_scatter_rows=tuple(completion_scatter_rows),
        completion_scatter_columns=tuple(completion_scatter_columns),
        physical_completion_lengths=physical_completion_lengths,
        physical_padding_positions=tree.padding_positions(),
    )


def plan_shared_prefix_bins(
    rows: Sequence[SharedPrefixRow],
    *,
    bin_capacity: int,
    max_completions_per_bin: int = 16,
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
        bin_capacity: Maximum number of deduplicated tokens in one shared bin.
        max_completions_per_bin: Maximum branches behind one stored prompt.

    Returns:
        A complete, deterministic partition of the input row indices.

    Raises:
        ValueError: If planner limits are invalid or row indices are duplicated.
    """
    if bin_capacity < 1:
        raise ValueError("bin_capacity must be positive")
    if max_completions_per_bin < 2:
        raise ValueError("max_completions_per_bin must be at least 2")
    if (
        isinstance(sequence_length_pad_multiple, bool)
        or not isinstance(sequence_length_pad_multiple, int)
        or sequence_length_pad_multiple < 1
    ):
        raise ValueError("sequence_length_pad_multiple must be a positive integer")

    def padded_row_length(row: SharedPrefixRow) -> int:
        return (
            (row.total_length + sequence_length_pad_multiple - 1)
            // sequence_length_pad_multiple
            * sequence_length_pad_multiple
        )

    def physical_completion_length(row: SharedPrefixRow) -> int:
        return padded_row_length(row) - row.prompt_length

    ordered_rows = sorted(rows, key=lambda row: row.row_index)
    row_indices = [row.row_index for row in ordered_rows]
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("row_index values must be unique")

    fallbacks: list[SharedPrefixFallback] = []
    eligible_rows: list[SharedPrefixRow] = []
    for row in ordered_rows:
        reason: SharedPrefixFallbackReason | None = None
        if row.group_id is None:
            reason = SharedPrefixFallbackReason.MISSING_GROUP_ID
        elif row.prompt_length == 0:
            reason = SharedPrefixFallbackReason.EMPTY_PROMPT
        elif row.completion_length == 0:
            reason = SharedPrefixFallbackReason.EMPTY_COMPLETION
        elif padded_row_length(row) > bin_capacity:
            reason = SharedPrefixFallbackReason.SEQUENCE_EXCEEDS_BIN

        if reason is None:
            eligible_rows.append(row)
        else:
            fallbacks.append(
                SharedPrefixFallback(
                    row=row,
                    reason=reason,
                    fits_block_diagonal_bin=padded_row_length(row) <= bin_capacity,
                )
            )

    exact_groups: dict[tuple[str, tuple[int, ...]], list[SharedPrefixRow]] = {}
    eligible_group_sizes: dict[str, int] = {}
    for row in eligible_rows:
        assert row.group_id is not None
        exact_groups.setdefault((row.group_id, row.prompt_token_ids), []).append(row)
        eligible_group_sizes[row.group_id] = eligible_group_sizes.get(row.group_id, 0) + 1

    shared_bins: list[SharedPrefixLayout] = []
    for (group_id, _prompt_token_ids), exact_rows in exact_groups.items():
        if len(exact_rows) < 2:
            reason = (
                SharedPrefixFallbackReason.PROMPT_MISMATCH
                if eligible_group_sizes[group_id] > 1
                else SharedPrefixFallbackReason.NO_EXACT_PROMPT_PEER
            )
            fallbacks.append(
                SharedPrefixFallback(row=exact_rows[0], reason=reason, fits_block_diagonal_bin=True)
            )
            continue

        prompt_length = exact_rows[0].prompt_length
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
                has_branch_slot = len(bin_rows) < max_completions_per_bin
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
                fallbacks.append(
                    SharedPrefixFallback(
                        row=bin_rows[0],
                        reason=SharedPrefixFallbackReason.NO_CAPACITY_COMPATIBLE_PEER,
                        fits_block_diagonal_bin=True,
                    )
                )

    fallbacks.sort(key=lambda fallback: fallback.row.row_index)
    return SharedPrefixPlan(shared_bins=tuple(shared_bins), fallbacks=tuple(fallbacks))
