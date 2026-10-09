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

"""Pure group sharding and real-row execution-slot planning.

No batch transport, training configuration, or model runtime is required.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from megatron.rl.shared_prefix_packing import MAX_SHARED_PREFIX_BRANCHES, _round_up


@dataclass(frozen=True, slots=True)
class GroupCoherentShardPlan:
    """Stable row-index assignment that never splits a prompt group.

    ``rank_order_permutation`` maps each position of the rank-major
    concatenation to its source row (output position -> source row), which is
    the forward permutation ``BatchedDataDict.reorder_data`` expects to restore
    source order. ``None`` means the concatenation is already in source order.
    """

    shard_indices: tuple[tuple[int, ...], ...]
    rank_order_permutation: tuple[int, ...] | None


@dataclass(frozen=True, slots=True)
class FixedExecutionSlotPlan:
    """Conservative row-to-unit assignments with equal units per group.

    ``row_slot_ids`` is aligned with the caller's original row order. Slot IDs
    are local to a prompt group; workers identify an execution unit by
    ``(group_id, slot_id)``. ``units_per_group_by_chunk`` records the common
    slot count selected independently for each logical global-batch chunk.
    """

    row_slot_ids: tuple[int, ...]
    units_per_group_by_chunk: tuple[int, ...]


def _pack_full_length_group(
    rows: Sequence[int], *, padded_lengths: Sequence[int], bin_capacity: int, max_rows_per_slot: int
) -> list[list[int]]:
    """FFD-pack conventional full lengths for one prompt group.

    Full duplicated lengths are deliberately conservative. Any resulting bin
    also fits as one deduplicated star because removing repeated prompts can
    only shorten it; if exact-prompt validation later fails, the same bin still
    fits as one ordinary block-diagonal packed microbatch.
    """
    bins: list[list[int]] = []
    bin_work: list[int] = []
    for row_index in sorted(rows, key=lambda index: (-padded_lengths[index], index)):
        destination: int | None = None
        for bin_index, bin_rows in enumerate(bins):
            if (
                len(bin_rows) < max_rows_per_slot
                and bin_work[bin_index] + padded_lengths[row_index] <= bin_capacity
            ):
                destination = bin_index
                break
        if destination is None:
            bins.append([row_index])
            bin_work.append(padded_lengths[row_index])
        else:
            bins[destination].append(row_index)
            bin_work[destination] += padded_lengths[row_index]
    return bins


def plan_fixed_execution_slots(
    *,
    group_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    bin_capacity: int,
    batch_size: int | None = None,
    sequence_length_pad_multiple: int = 1,
    max_rows_per_slot: int = MAX_SHARED_PREFIX_BRANCHES,
) -> FixedExecutionSlotPlan:
    """Give every complete prompt group the same number of real forwards.

    The driver does not need prompt tokens or even prompt lengths. It first
    packs each group's *full*, conventionally duplicated row lengths. Within a
    logical batch chunk, ``K`` is the largest resulting bin count. Groups with
    fewer than ``K`` bins are deterministically split until every group has
    exactly ``K`` nonempty units. Splitting preserves capacity, so no masked
    dummy forward is needed to synchronize Megatron/MoE schedules.
    """
    if bin_capacity < 1:
        raise ValueError(f"bin_capacity must be positive, got {bin_capacity}")
    if sequence_length_pad_multiple < 1:
        raise ValueError(
            "sequence_length_pad_multiple must be positive, got " f"{sequence_length_pad_multiple}"
        )
    if max_rows_per_slot < 1:
        raise ValueError(f"max_rows_per_slot must be positive, got {max_rows_per_slot}")
    if len(group_ids) != len(sequence_lengths):
        raise ValueError("group_ids and sequence_lengths must have the same number of rows")
    if not group_ids:
        raise ValueError("cannot plan execution slots for an empty batch")

    num_rows = len(group_ids)
    if batch_size is None:
        batch_size = num_rows
    if batch_size < 1 or num_rows % batch_size != 0:
        raise ValueError(
            "shared-prefix batch size must be positive and divide the number "
            f"of rows exactly; got {num_rows} rows and batch_size={batch_size}"
        )

    padded_lengths: list[int] = []
    for row_index, (group_id, sequence_length) in enumerate(
        zip(group_ids, sequence_lengths, strict=True)
    ):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"shared-prefix group_ids[{row_index}] must be a non-empty string")
        normalized_length = int(sequence_length)
        if normalized_length < 1:
            raise ValueError(
                f"sequence_lengths[{row_index}] must be positive, got {sequence_length}"
            )
        padded_length = _round_up(normalized_length, sequence_length_pad_multiple)
        if padded_length > bin_capacity:
            raise ValueError(
                "shared-prefix row exceeds the execution bin capacity after "
                f"padding: row={row_index}, padded_length={padded_length}, "
                f"capacity={bin_capacity}"
            )
        padded_lengths.append(padded_length)

    row_slot_ids = [-1] * num_rows
    units_per_group_by_chunk: list[int] = []
    seen_group_chunks: dict[str, int] = {}
    for chunk_index, chunk_start in enumerate(range(0, num_rows, batch_size)):
        chunk_end = chunk_start + batch_size
        grouped_rows: dict[str, list[int]] = {}
        for row_index in range(chunk_start, chunk_end):
            group_id = group_ids[row_index]
            previous_chunk = seen_group_chunks.setdefault(group_id, chunk_index)
            if previous_chunk != chunk_index:
                raise ValueError(
                    "shared-prefix prompt groups must not cross logical global "
                    f"batch boundaries; group {group_id!r} appears in chunks "
                    f"{previous_chunk} and {chunk_index}"
                )
            grouped_rows.setdefault(group_id, []).append(row_index)

        group_sizes = {len(rows) for rows in grouped_rows.values()}
        if len(group_sizes) != 1:
            raise ValueError(
                "shared-prefix execution slots require complete equal-size "
                f"prompt groups; observed sizes {sorted(group_sizes)} in batch "
                f"chunk {chunk_index}"
            )
        bins_by_group: dict[str, list[list[int]]] = {
            group_id: _pack_full_length_group(
                rows,
                padded_lengths=padded_lengths,
                bin_capacity=bin_capacity,
                max_rows_per_slot=max_rows_per_slot,
            )
            for group_id, rows in grouped_rows.items()
        }
        units_per_group = max(len(group_bins) for group_bins in bins_by_group.values())
        units_per_group_by_chunk.append(units_per_group)

        for group_id, bins in bins_by_group.items():
            while len(bins) < units_per_group:
                candidates = [
                    (bin_index, bin_rows)
                    for bin_index, bin_rows in enumerate(bins)
                    if len(bin_rows) > 1
                ]
                if not candidates:
                    raise RuntimeError(
                        "could not split shared-prefix group into the common "
                        f"execution-unit count; group={group_id!r}, "
                        f"target={units_per_group}"
                    )
                split_index, split_bin = min(
                    candidates,
                    key=lambda item: (
                        -len(item[1]),
                        -sum(padded_lengths[index] for index in item[1]),
                        item[0],
                    ),
                )
                split_row = max(split_bin, key=lambda index: (padded_lengths[index], -index))
                bins[split_index] = [index for index in split_bin if index != split_row]
                bins.append([split_row])

            for slot_id, bin_rows in enumerate(bins):
                for row_index in bin_rows:
                    if row_slot_ids[row_index] != -1:
                        raise RuntimeError(f"row {row_index} was assigned to more than one slot")
                    row_slot_ids[row_index] = slot_id

    if any(slot_id < 0 for slot_id in row_slot_ids):
        raise RuntimeError("execution-slot planning did not cover every input row")
    return FixedExecutionSlotPlan(
        row_slot_ids=tuple(row_slot_ids), units_per_group_by_chunk=tuple(units_per_group_by_chunk)
    )


def plan_group_coherent_shards(
    *,
    group_ids: Sequence[str],
    sequence_lengths: Sequence[int],
    num_shards: int,
    batch_size: int | None = None,
) -> GroupCoherentShardPlan:
    """Balance complete equal-size prompt groups without splitting siblings.

    The first shared-prefix execution slice intentionally requires complete,
    fixed-size rollout groups and an integral number of groups per DP rank. Groups
    are assigned longest-first to the currently lightest rank, with a fixed
    group-count quota per rank. This keeps row counts equal for Megatron while
    improving token balance and retaining exact group locality. When
    ``batch_size`` is provided, each logical global batch is planned
    independently so rank-local concatenation retains Megatron's GBS
    boundaries. Single-completion groups are valid; the worker retains them as
    ordinary dense execution units because they have no prompt peer to share.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if len(group_ids) != len(sequence_lengths):
        raise ValueError("group_ids and sequence_lengths must have the same number of rows")
    if not group_ids:
        raise ValueError("cannot shard an empty shared-prefix batch")

    num_rows = len(group_ids)
    if batch_size is None:
        batch_size = num_rows
    if batch_size < 1 or num_rows % batch_size != 0:
        raise ValueError(
            "shared-prefix batch size must be positive and divide the number "
            f"of rows exactly; got {num_rows} rows and batch_size={batch_size}"
        )

    normalized_lengths: list[int] = []
    for row_index, (group_id, sequence_length) in enumerate(
        zip(group_ids, sequence_lengths, strict=True)
    ):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(f"shared-prefix group_ids[{row_index}] must be a non-empty string")
        normalized_length = int(sequence_length)
        if normalized_length < 0:
            raise ValueError(
                f"sequence_lengths[{row_index}] must be nonnegative, got " f"{sequence_length}"
            )
        normalized_lengths.append(normalized_length)

    group_chunks: dict[str, set[int]] = {}
    for row_index, group_id in enumerate(group_ids):
        group_chunks.setdefault(group_id, set()).add(row_index // batch_size)
    for group_id, chunk_indices in group_chunks.items():
        if len(chunk_indices) != 1:
            raise ValueError(
                "shared-prefix prompt groups must not cross logical global "
                f"batch boundaries; group {group_id!r} appears in chunks "
                f"{sorted(chunk_indices)}"
            )

    shard_rows: list[list[int]] = [[] for _ in range(num_shards)]
    for chunk_index, chunk_start in enumerate(range(0, num_rows, batch_size)):
        chunk_end = chunk_start + batch_size
        grouped_rows: dict[str, list[int]] = {}
        for row_index in range(chunk_start, chunk_end):
            group_id = group_ids[row_index]
            grouped_rows.setdefault(group_id, []).append(row_index)

        group_sizes = {len(rows) for rows in grouped_rows.values()}
        if len(group_sizes) != 1:
            size_summary = sorted(group_sizes)
            raise ValueError(
                "shared-prefix DP sharding requires complete equal-size prompt "
                f"groups; observed group sizes {size_summary} in batch chunk "
                f"{chunk_index}"
            )
        group_count = len(grouped_rows)
        if group_count % num_shards != 0:
            raise ValueError(
                "shared-prefix DP sharding requires an integral number of "
                f"complete groups per rank; got {group_count} groups across "
                f"{num_shards} ranks in batch chunk {chunk_index}"
            )
        groups_per_shard = group_count // num_shards

        ranked_groups = sorted(
            grouped_rows.items(),
            key=lambda item: (-sum(normalized_lengths[index] for index in item[1]), item[1][0]),
        )
        chunk_shard_groups: list[list[list[int]]] = [[] for _ in range(num_shards)]
        chunk_shard_work = [0] * num_shards
        for _group_id, rows in ranked_groups:
            candidates = [
                shard_index
                for shard_index, groups in enumerate(chunk_shard_groups)
                if len(groups) < groups_per_shard
            ]
            destination = min(candidates, key=lambda index: (chunk_shard_work[index], index))
            chunk_shard_groups[destination].append(rows)
            chunk_shard_work[destination] += sum(normalized_lengths[index] for index in rows)

        for shard_index, groups in enumerate(chunk_shard_groups):
            shard_rows[shard_index].extend(
                row_index
                for rows in sorted(groups, key=lambda group_rows: group_rows[0])
                for row_index in rows
            )

    shard_indices = tuple(tuple(indices) for indices in shard_rows)
    flat_indices = [index for shard in shard_indices for index in shard]
    expected_indices = list(range(num_rows))
    if sorted(flat_indices) != expected_indices:
        raise RuntimeError("group-coherent sharding did not cover every input row exactly once")

    return GroupCoherentShardPlan(
        shard_indices=shard_indices,
        rank_order_permutation=(None if flat_indices == expected_indices else tuple(flat_indices)),
    )


def make_repeated_group_ids(*, num_rows: int, group_size: int, namespace: str) -> list[str]:
    """Return one opaque ID per row for contiguous repeated prompt groups."""
    if not namespace:
        raise ValueError("shared-prefix group namespace must be non-empty")
    if group_size < 1:
        raise ValueError(f"shared-prefix group_size must be positive, got {group_size}")
    if num_rows < 0 or num_rows % group_size != 0:
        raise ValueError(
            f"num_rows={num_rows} must be nonnegative and divisible by group_size={group_size}"
        )
    return [
        f"{namespace}:{group_index}"
        for group_index in range(num_rows // group_size)
        for _ in range(group_size)
    ]


def get_prescribed_shared_prefix_slots(
    *, group_ids: Sequence[str], slot_ids: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Validate equal contiguous slot counts and return deterministic row sets."""
    slots = slot_ids
    if len(slots) != len(group_ids):
        raise ValueError("shared-prefix execution slots must have one entry per row")
    if any(slot < 0 for slot in slots):
        raise ValueError("shared-prefix execution slot IDs must be nonnegative")
    units: dict[tuple[str, int], list[int]] = {}
    slots_by_group: dict[str, set[int]] = {}
    for row_index, (group_id, slot_id) in enumerate(zip(group_ids, slots, strict=True)):
        if not isinstance(group_id, str) or not group_id:
            raise ValueError(
                "driver-prescribed shared-prefix execution requires a nonempty "
                f"group ID for row {row_index}"
            )
        units.setdefault((group_id, slot_id), []).append(row_index)
        slots_by_group.setdefault(group_id, set()).add(slot_id)

    expected_slots: set[int] | None = None
    for group_id, group_slots in slots_by_group.items():
        contiguous_slots = set(range(len(group_slots)))
        if group_slots != contiguous_slots:
            raise ValueError(
                "shared-prefix execution slots must be contiguous from zero; "
                f"group {group_id!r} has {sorted(group_slots)}"
            )
        if expected_slots is None:
            expected_slots = group_slots
        elif group_slots != expected_slots:
            raise ValueError(
                "every local prompt group must have the same number of execution "
                f"slots; expected {len(expected_slots)}, group {group_id!r} has "
                f"{len(group_slots)}"
            )
    return tuple(tuple(rows) for rows in units.values())
