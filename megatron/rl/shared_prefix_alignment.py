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

"""Align packed distributed forwards using each real source row exactly once."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.rl.shared_prefix_execution import SharedPrefixExecutionUnit


@dataclass(frozen=True)
class AlignedUnit:
    row_indices: tuple[int, ...]
    # Reuse this original unit, including its shared layout. None means that
    # splitting changed its rows: construct a conventional dense fallback.
    source_unit: int | None


def align_rows_to_count(
    units: Sequence[Sequence[int]],
    *,
    padded_row_lengths: Sequence[int],
    bin_capacity: int,
    target_count: int,
) -> tuple[AlignedUnit, ...]:
    """Split real execution units until every rank can use the agreed count.

    Coverage is exact: no row is copied, dropped, fabricated or masked out.
    Splitting never increases an individual unit's conventional token cost.
    Untouched units retain their original shared layouts. Changed units are
    intentionally dense, avoiding unsupported shared-layout slicing.
    """
    if type(bin_capacity) is not int or bin_capacity < 1:
        raise ValueError("bin_capacity must be a positive integer")
    if type(target_count) is not int:
        raise ValueError("target_count must be an integer")
    costs = tuple(padded_row_lengths)
    if not costs or any(type(n) is not int or n < 1 for n in costs):
        raise ValueError("padded row lengths must be positive integers")
    rows = tuple(tuple(unit) for unit in units)
    if not rows or any(not unit for unit in rows):
        raise ValueError("execution units must be nonempty")
    flat = [row for unit in rows for row in unit]
    if any(type(row) is not int for row in flat) or sorted(flat) != list(range(len(costs))):
        raise ValueError("units must cover every local row exactly once")
    if not len(rows) <= target_count <= len(costs):
        raise ValueError("target count must be between current units and real rows")

    def work(unit_rows: Sequence[int]) -> int:
        return sum(costs[row] for row in unit_rows)

    if any(work(unit) > bin_capacity for unit in rows):
        raise ValueError("every original unit must fit the expanded dense budget")
    result = [AlignedUnit(unit, index) for index, unit in enumerate(rows)]
    while len(result) < target_count:
        candidates = [i for i, unit in enumerate(result) if len(unit.row_indices) > 1]
        # The coverage/count preconditions guarantee a splittable real unit.
        index = max(candidates, key=lambda i: (work(result[i].row_indices), -i))
        original = result[index].row_indices
        total = work(original)
        prefix = 0
        best_cut = 1
        best_difference = total
        for cut, row in enumerate(original[:-1], start=1):
            prefix += costs[row]
            difference = abs(total - 2 * prefix)
            if difference < best_difference:
                best_cut, best_difference = cut, difference
        result[index : index + 1] = [
            AlignedUnit(original[:best_cut], None),
            AlignedUnit(original[best_cut:], None),
        ]
    return tuple(result)


def rebuild_subset(
    unit: "SharedPrefixExecutionUnit", selected: Sequence[int], *, padding_multiple: int
) -> "SharedPrefixExecutionUnit":
    """Retain original source-row IDs and exact prompt identities in every root."""
    from megatron.rl.shared_prefix_packing import (
        SharedPrefixForestLayout,
        SharedPrefixRow,
        build_shared_prefix_layout,
    )

    selected_rows = set(selected)
    if unit.shared_layout is None:
        raise ValueError("Rebuilding a shared subset requires a shared layout")
    roots = []
    for _, root in unit.shared_layout.iter_roots():
        rows = [
            SharedPrefixRow(index, root.group_id, root.prompt_token_ids, length)
            for index, length in zip(root.row_indices, root.completion_lengths, strict=True)
            if index in selected_rows
        ]
        if rows:
            roots.append(
                build_shared_prefix_layout(
                    rows, sequence_length_pad_multiple=padding_multiple, allow_singleton=True
                )
            )
    if not roots:
        raise ValueError("A shared subset must retain at least one real row")
    layout = roots[0] if len(roots) == 1 else SharedPrefixForestLayout(tuple(roots))
    return replace(
        unit,
        row_indices=layout.row_indices,
        shared_layout=layout,
        physical_length=layout.physical_total_length,
    )


def align_physical_units(
    units: Sequence["SharedPrefixExecutionUnit"],
    *,
    costs: Sequence[int],
    capacity: int,
    target_count: int,
    padding_multiple: int,
    rebuild: Callable[..., "SharedPrefixExecutionUnit"] = rebuild_subset,
) -> tuple[tuple["SharedPrefixExecutionUnit", ...], dict[str, int]]:
    """Split real units while preserving physical bounds and prompt sharing.

    Evaluation only: MTP must be uniformly disabled by the caller. Training
    still uses the independently qualified expanded-budget alignment path.
    """
    if (
        type(capacity) is not int
        or capacity < 1
        or type(padding_multiple) is not int
        or padding_multiple < 1
        or capacity % padding_multiple
        or type(target_count) is not int
    ):
        raise ValueError("Positive aligned capacity and integer target required")
    if not costs or any(type(n) is not int or n < 1 for n in costs):
        raise ValueError("Positive padded row lengths required")
    expected = list(range(len(costs)))

    def verify(plan: Sequence["SharedPrefixExecutionUnit"]) -> None:
        flat = [row for unit in plan for row in unit.row_indices]
        if any(type(row) is not int for row in flat) or sorted(flat) != expected:
            raise ValueError("Physical alignment must cover every real row once")
        for unit in plan:
            length = unit.physical_length
            if (
                not unit.row_indices
                or type(length) is not int
                or length < 1
                or (length + padding_multiple - 1) // padding_multiple * padding_multiple > capacity
            ):
                raise ValueError("Physical execution unit exceeds aligned capacity")
            if unit.shared_layout is None and length != sum(costs[r] for r in unit.row_indices):
                raise ValueError("Dense units must retain their expanded row cost")
            if unit.shared_layout is not None:
                if tuple(unit.shared_layout.row_indices) != tuple(unit.row_indices):
                    raise ValueError("Shared layout and execution rows disagree")
                if unit.shared_layout.physical_total_length != length:
                    raise ValueError("Shared layout and execution lengths disagree")

    result = list(units)
    verify(result)
    if not len(result) <= target_count <= len(costs):
        raise ValueError("Target must be between current unit count and real rows")
    original_ids = {id(unit) for unit in units}
    while len(result) < target_count:
        candidates = [i for i, u in enumerate(result) if len(u.row_indices) > 1]
        index = max(candidates, key=lambda i: (result[i].physical_length, -i))
        unit = result[index]
        rows = unit.row_indices
        total = sum(costs[r] for r in rows)
        prefix = 0
        best_cut, best_difference = 1, total
        for cut, row in enumerate(rows[:-1], start=1):
            prefix += costs[row]
            difference = abs(total - 2 * prefix)
            if difference < best_difference:
                best_cut, best_difference = cut, difference
        children = []
        for selected in (rows[:best_cut], rows[best_cut:]):
            if unit.shared_layout is None:
                child = replace(
                    unit, row_indices=selected, physical_length=sum(costs[r] for r in selected)
                )
            else:
                child = rebuild(unit, selected, padding_multiple=padding_multiple)
            if set(child.row_indices) != set(selected):
                raise ValueError("Rebuilt child changed its assigned source rows")
            children.append(child)
        result[index : index + 1] = children
        verify(result)
    changed = [unit for unit in result if id(unit) not in original_ids]
    return tuple(result), {
        "split_dense_units": sum(u.shared_layout is None for u in changed),
        "split_shared_units": sum(u.shared_layout is not None for u in changed),
        "rebuilt_units": len(changed),
    }


def materialize_alignment(
    units: Sequence["SharedPrefixExecutionUnit"],
    *,
    costs: Sequence[int],
    capacity: int,
    target_count: int,
) -> tuple[tuple["SharedPrefixExecutionUnit", ...], int]:
    """Preserve unmodified layouts; changed units carry conventional semantics."""
    aligned = align_rows_to_count(
        [unit.row_indices for unit in units],
        padded_row_lengths=costs,
        bin_capacity=capacity,
        target_count=target_count,
    )
    result = []
    for item in aligned:
        if item.source_unit is not None:
            result.append(units[item.source_unit])
        else:
            result.append(
                replace(
                    units[0],
                    row_indices=item.row_indices,
                    shared_layout=None,
                    physical_length=sum(costs[row] for row in item.row_indices),
                )
            )
    return tuple(result), sum(item.source_unit is None for item in aligned)


def align_training_units(
    units: Sequence["SharedPrefixExecutionUnit"],
    *,
    costs: Sequence[int],
    capacity: int,
    target_count: int,
    padding_multiple: int,
) -> tuple[tuple["SharedPrefixExecutionUnit", ...], int]:
    """Preserve prefixes in the existing expanded-budget training row cuts.

    The count, row order and cuts match ``materialize_alignment`` exactly.
    Every parent and child must fit the expanded MTP budget. Single rows keep
    conventional execution because sharing cannot save work for a singleton.
    Rebuilt forests use one auxiliary-loss normalization group, matching the
    conventional unit they replace. Parents with explicit groups are rejected.
    """
    if (
        type(padding_multiple) is not int
        or padding_multiple < 1
        or capacity % padding_multiple
        or any(cost % padding_multiple for cost in costs)
    ):
        raise ValueError("Training capacity and costs must respect topology alignment")
    aligned = align_rows_to_count(
        [unit.row_indices for unit in units],
        padded_row_lengths=costs,
        bin_capacity=capacity,
        target_count=target_count,
    )
    from megatron.rl.shared_prefix_packing import SharedPrefixForestLayout

    owners = {row: unit for unit in units for row in unit.row_indices}
    result = []
    for item in aligned:
        if item.source_unit is not None:
            result.append(units[item.source_unit])
            continue
        parent = owners[item.row_indices[0]]
        layout = parent.shared_layout
        if isinstance(layout, SharedPrefixForestLayout) and layout.mtp_loss_group_root_counts:
            raise ValueError("Training splits do not support explicit MTP loss groups")
        if parent.shared_layout is None or len(item.row_indices) == 1:
            child = replace(
                parent,
                row_indices=item.row_indices,
                shared_layout=None,
                physical_length=sum(costs[row] for row in item.row_indices),
            )
        else:
            child = rebuild_subset(parent, item.row_indices, padding_multiple=padding_multiple)
            if isinstance(child.shared_layout, SharedPrefixForestLayout):
                # The old split is one dense microbatch. Keep its one MTP
                # token-count correction even when several roots remain.
                child = replace(
                    child,
                    shared_layout=replace(
                        child.shared_layout,
                        mtp_loss_group_root_counts=(len(child.shared_layout.roots),),
                    ),
                )
        if child.row_indices != item.row_indices:
            raise ValueError("Training alignment changed source row order")
        padded_physical = (
            (child.physical_length + padding_multiple - 1) // padding_multiple * padding_multiple
        )
        if padded_physical > sum(costs[row] for row in item.row_indices):
            raise ValueError("Rebuilt shared unit exceeds its expanded token budget")
        result.append(child)
    return tuple(result), sum(item.source_unit is None for item in aligned)
