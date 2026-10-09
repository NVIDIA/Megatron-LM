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

"""Backend-neutral execution plans with physical and expanded token budgets."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Optional

from megatron.rl.shared_prefix_metadata import plan_fixed_execution_slots
from megatron.rl.shared_prefix_packing import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixRow,
    _round_up,
    build_shared_prefix_layout,
    pack_shared_prefix_groups,
    plan_shared_prefix_bins,
)


@dataclass(frozen=True, slots=True)
class SharedPrefixExecutionUnit:
    """One forward for an exact-prompt star, a forest, or a dense fallback."""

    row_indices: tuple[int, ...]
    shared_layout: Optional[SharedPrefixLayout | SharedPrefixForestLayout]
    physical_length: int


@dataclass(frozen=True, slots=True)
class SharedPrefixExecutionPlan:
    """Planned physical forwards for one local shared-prefix batch.

    ``units`` is the driver-prescribed execution order consumed by
    the caller's microbatch consumer. ``num_units`` is the rank-local
    forward count the worker barrier compares across the model world, and
    ``max_physical_length`` bounds the padded microbatch width.
    """

    units: tuple[SharedPrefixExecutionUnit, ...]

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("shared-prefix train mode received an empty local batch")

    @property
    def num_units(self) -> int:
        return len(self.units)

    @property
    def max_physical_length(self) -> int:
        return max(unit.physical_length for unit in self.units)


def validate_shared_prefix_execution_units(
    units: tuple[SharedPrefixExecutionUnit, ...], *, batch_size: int
) -> None:
    """Reject a precomputed plan that does not cover this batch exactly once."""
    if not units:
        raise ValueError("shared-prefix train mode received an empty local batch")
    covered_rows = sorted(index for unit in units for index in unit.row_indices)
    if covered_rows != list(range(batch_size)):
        raise ValueError(
            "precomputed shared-prefix execution units must cover every local "
            f"row exactly once; batch has {batch_size} rows, units cover "
            f"{covered_rows}"
        )


def plan_shared_prefix_execution_units(
    rows: Sequence[SharedPrefixRow],
    *,
    row_slots: Sequence[Sequence[int]],
    bin_capacity: int,
    padding_multiple: int,
    pack_groups: bool = False,
    repack_groups: bool = False,
    pack_dense_fallbacks: bool = False,
    merge_dense_fallbacks: bool = False,
    forward_only: bool = False,
    evaluation_packing: bool = False,
    dense_packer: Callable[[Sequence[int]], Sequence[Sequence[int]]] | None = None,
    largest_first: bool = False,
) -> tuple[SharedPrefixExecutionUnit, ...]:
    """Resolve prescribed real-row slots into shared forests or dense units.

    ``dense_packer`` is a length-only callback chosen by the caller. Training
    retains both physical and expanded MTP budgets; evaluation may use only
    the physical budget when the caller disables MTP uniformly. The caller
    must align unit counts across its distributed model world afterward.
    """
    if bin_capacity < 1 or padding_multiple < 1:
        raise ValueError("Positive capacity and padding multiple required")
    rows_by_index = {row.row_index: row for row in rows}
    covered = [index for slot in row_slots for index in slot]
    if (
        len(rows_by_index) != len(rows)
        or any(not slot for slot in row_slots)
        or sorted(covered) != sorted(rows_by_index)
    ):
        raise ValueError("Execution slots must cover every source row exactly once")
    if pack_dense_fallbacks and not pack_groups:
        raise ValueError("dense fallback repacking requires DP1 pack_groups mode")
    if merge_dense_fallbacks and not (pack_groups and pack_dense_fallbacks):
        raise ValueError("dense fallback merging requires pack_groups and pack_dense_fallbacks")
    if pack_groups and repack_groups:
        # Forest execution is guarded to DP1 by the worker. The driver's
        # equal-slot schedule protects cross-DP collectives, but in DP1 one
        # long group must not force unrelated groups into singleton fallbacks.
        # Repack each complete group with the same conventional token bound;
        # both expanded MTP work and prompt-mismatch fallbacks still fit.
        grouped_indices: dict[str | None, list[int]] = {}
        for row in rows:
            grouped_indices.setdefault(row.group_id, []).append(row.row_index)
        independent_slots: list[tuple[int, ...]] = []
        for group_id, indices in grouped_indices.items():
            if len(indices) == 1:
                independent_slots.append(tuple(indices))
                continue
            assert group_id is not None  # Validated by the prescribed-slot reader.
            if forward_only and evaluation_packing:
                # Evaluation never expands these stars for MTP. Apply the
                # physical budget before splitting a group, as well as when
                # packing stars together below. Retaining the training slot
                # cuts here would still duplicate a large prefix unnecessarily.
                evaluation_plan = plan_shared_prefix_bins(
                    [rows_by_index[index] for index in indices],
                    bin_capacity=bin_capacity,
                    max_completions_per_bin=16,
                    sequence_length_pad_multiple=padding_multiple,
                )
                independent_slots.extend(
                    layout.row_indices for layout in evaluation_plan.shared_bins
                )
                independent_slots.extend(
                    (row_index,) for row_index in evaluation_plan.fallback_row_indices
                )
                continue
            plan = plan_fixed_execution_slots(
                group_ids=[group_id] * len(indices),
                sequence_lengths=[rows_by_index[index].total_length for index in indices],
                bin_capacity=bin_capacity,
                sequence_length_pad_multiple=padding_multiple,
            )
            for slot_id in range(plan.units_per_group_by_chunk[0]):
                independent_slots.append(
                    tuple(
                        index
                        for index, assigned_slot in zip(indices, plan.row_slot_ids, strict=True)
                        if assigned_slot == slot_id
                    )
                )
        row_slots = tuple(independent_slots)
    units: list[SharedPrefixExecutionUnit] = []
    for row_indices in row_slots:
        slot_rows = [rows_by_index[index] for index in row_indices]
        candidate = plan_shared_prefix_bins(
            slot_rows,
            bin_capacity=bin_capacity,
            max_completions_per_bin=16,
            sequence_length_pad_multiple=padding_multiple,
        )
        if (
            len(candidate.shared_bins) == 1
            and not candidate.fallback_row_indices
            and set(candidate.shared_bins[0].row_indices) == set(row_indices)
        ):
            layout = candidate.shared_bins[0]
            units.append(
                SharedPrefixExecutionUnit(
                    row_indices=layout.row_indices,
                    shared_layout=layout,
                    physical_length=layout.physical_total_length,
                )
            )
            continue

        # ``SharedPrefixRow.total_length`` is exactly the validated input
        # length of that row, so no second host copy of ``input_lengths`` and
        # no per-row ``.item()`` are needed here.
        fallback_length = sum(
            _round_up(rows_by_index[index].total_length, padding_multiple) for index in row_indices
        )
        if fallback_length > bin_capacity:
            raise RuntimeError(
                "driver-prescribed shared-prefix fallback exceeds its bin "
                f"capacity: rows={row_indices}, padded_length={fallback_length}, "
                f"capacity={bin_capacity}"
            )
        units.append(
            SharedPrefixExecutionUnit(
                row_indices=row_indices, shared_layout=None, physical_length=fallback_length
            )
        )
    if not units:
        raise ValueError("shared-prefix train mode received an empty local batch")
    if pack_groups:
        # Exact stars can share a forward; all roots retain their source row IDs.
        layouts = pack_shared_prefix_groups(
            [unit.shared_layout for unit in units if unit.shared_layout is not None],
            bin_capacity=bin_capacity,
            # Shared-prefix evaluation skips MTP. Its physical/logit budget
            # still applies; only training needs the expanded branch budget.
            dense_capacity=(None if forward_only and evaluation_packing else bin_capacity),
            padding_multiple=padding_multiple,
        )
        units = [unit for unit in units if unit.shared_layout is None] + [
            SharedPrefixExecutionUnit(layout.row_indices, layout, layout.physical_total_length)
            for layout in layouts
        ]
        if pack_dense_fallbacks:
            # Fallback rows have ordinary causal semantics. Pack them across
            # prompt groups instead of retaining the driver's equal-slot cuts.
            # This also recovers conventional batching when every slot falls
            # back, while leaving the shared attention layout unchanged.
            fallback_rows = sorted(
                index for unit in units if unit.shared_layout is None for index in unit.row_indices
            )
            if fallback_rows:
                lengths = [
                    _round_up(rows_by_index[index].total_length, padding_multiple)
                    for index in fallback_rows
                ]
                if dense_packer is None:
                    raise ValueError("Dense fallback repacking requires a length-only packer")
                bins = [list(indices) for indices in dense_packer(lengths)]
                if largest_first:
                    bins.sort(
                        key=lambda indices: sum(lengths[index] for index in indices), reverse=True
                    )
                dense_units = [
                    SharedPrefixExecutionUnit(
                        row_indices=tuple(fallback_rows[index] for index in indices),
                        shared_layout=None,
                        physical_length=sum(lengths[index] for index in indices),
                    )
                    for indices in bins
                ]
                units = dense_units + [unit for unit in units if unit.shared_layout is not None]
        if merge_dense_fallbacks and not forward_only:
            units = _merge_dense_fallback_execution_units(
                units,
                rows_by_index=rows_by_index,
                bin_capacity=bin_capacity,
                padding_multiple=padding_multiple,
            )
    if sorted(index for unit in units for index in unit.row_indices) != sorted(rows_by_index):
        raise ValueError("Execution packing must cover every source row exactly once")
    return tuple(units)


def _merge_dense_fallback_execution_units(
    units: list[SharedPrefixExecutionUnit],
    *,
    rows_by_index: dict[int, SharedPrefixRow],
    bin_capacity: int,
    padding_multiple: int,
) -> list[SharedPrefixExecutionUnit]:
    """Absorb whole dense bins without changing their MTP normalization groups.

    Existing shared units remain intact. Each fallback bin is indivisible and
    becomes consecutive independent causal roots with one auxiliary-loss group.
    A bin that cannot fit both budgets retains its conventional execution.
    """
    shared_indices = [index for index, unit in enumerate(units) if unit.shared_layout is not None]
    if not shared_indices:
        return units
    merged = list(units)
    expanded_lengths: dict[int, int] = {}
    for index in shared_indices:
        layout = units[index].shared_layout
        assert layout is not None
        expanded_lengths[index] = sum(
            len(root.row_indices) * root.prompt_length + sum(root.physical_completion_lengths)
            for _, root in layout.iter_roots()
        )
    absorbed: set[int] = set()
    fallback_indices = sorted(
        (index for index, unit in enumerate(units) if unit.shared_layout is None),
        key=lambda index: -units[index].physical_length,
    )
    for fallback_index in fallback_indices:
        fallback = units[fallback_index]
        fallback_rows = [rows_by_index[index] for index in fallback.row_indices]
        if any(row.prompt_length == 0 or row.completion_length == 0 for row in fallback_rows):
            continue
        for target_index in shared_indices:
            target = merged[target_index]
            combined = target.physical_length + fallback.physical_length
            if (
                _round_up(combined, padding_multiple) > bin_capacity
                or expanded_lengths[target_index] + fallback.physical_length > bin_capacity
            ):
                continue
            layout = target.shared_layout
            assert layout is not None
            target_roots = tuple(root for _, root in layout.iter_roots())
            target_counts = (
                layout.mtp_loss_group_root_counts or (1,) * len(target_roots)
                if isinstance(layout, SharedPrefixForestLayout)
                else (1,)
            )
            fallback_roots = tuple(
                build_shared_prefix_layout(
                    [row], sequence_length_pad_multiple=padding_multiple, allow_singleton=True
                )
                for row in fallback_rows
            )
            forest = SharedPrefixForestLayout(
                target_roots + fallback_roots,
                mtp_loss_group_root_counts=target_counts + (len(fallback_roots),),
            )
            merged[target_index] = SharedPrefixExecutionUnit(
                row_indices=forest.row_indices,
                shared_layout=forest,
                physical_length=forest.physical_total_length,
            )
            expanded_lengths[target_index] += fallback.physical_length
            absorbed.add(fallback_index)
            break
    return [unit for index, unit in enumerate(merged) if index not in absorbed]
