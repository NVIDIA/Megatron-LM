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

from megatron.rl.shared_prefix_packing import (
    MAX_SHARED_PREFIX_BRANCHES,
    SharedPrefixForestLayout,
    SharedPrefixLayout,
    SharedPrefixRow,
    _is_int,
    _round_up,
    build_shared_prefix_layout,
    pack_shared_prefix_groups,
    plan_shared_prefix_bins,
)

DensePacker = Callable[[Sequence[int]], Sequence[Sequence[int]]]


@dataclass(frozen=True, slots=True)
class SharedPrefixExecutionUnit:
    """One forward for an exact-prompt star, a forest, or a dense fallback."""

    row_indices: tuple[int, ...]
    shared_layout: SharedPrefixLayout | SharedPrefixForestLayout | None
    physical_length: int


@dataclass(frozen=True, slots=True)
class SharedPrefixExecutionPlan:
    """Planned physical forwards for one local shared-prefix batch.

    ``units`` lists the forwards in execution order. ``num_units`` is this
    rank's forward count; distributed callers must make it equal on every rank
    that shares a collective before the first forward.
    """

    units: tuple[SharedPrefixExecutionUnit, ...]

    def __post_init__(self) -> None:
        if not self.units:
            raise ValueError("shared-prefix train mode received an empty local batch")

    @property
    def num_units(self) -> int:
        """Number of physical forwards in this plan."""
        return len(self.units)

    @property
    def max_physical_length(self) -> int:
        """Longest physical forward, before the caller's TP/CP padding.

        The padded microbatch width is this value rounded up to the caller's
        TP/CP padding multiple.
        """
        return max(unit.physical_length for unit in self.units)


def validate_shared_prefix_execution_units(
    units: tuple[SharedPrefixExecutionUnit, ...], *, batch_size: int
) -> None:
    """Reject a precomputed plan that does not cover this batch exactly once.

    A shared unit must also execute exactly its layout: the layout's rows in layout order and
    the layout's physical length.
    """
    if not units:
        raise ValueError("shared-prefix train mode received an empty local batch")
    covered_rows = sorted(index for unit in units for index in unit.row_indices)
    if covered_rows != list(range(batch_size)):
        raise ValueError(
            "precomputed shared-prefix execution units must cover every local "
            f"row exactly once; batch has {batch_size} rows, units cover "
            f"{covered_rows}"
        )
    for unit in units:
        layout = unit.shared_layout
        if layout is None:
            continue
        if tuple(unit.row_indices) != tuple(layout.row_indices):
            raise ValueError(
                "shared-prefix execution unit rows must equal its layout's rows in layout "
                f"order; unit has {tuple(unit.row_indices)}, layout has "
                f"{tuple(layout.row_indices)}"
            )
        if unit.physical_length != layout.physical_total_length:
            raise ValueError(
                "shared-prefix execution unit physical length must equal its layout's; "
                f"unit has {unit.physical_length}, layout has {layout.physical_total_length}"
            )


def pack_dense_rows(
    costs: Sequence[int], *, bin_capacity: int, dense_packer: DensePacker
) -> tuple[tuple[int, ...], ...]:
    """Run a caller's length-only packer and validate the bins it returns.

    Every model-parallel rank plans its own microbatches, so ``dense_packer``
    must be a deterministic, pure function of ``costs``: TP/CP peers that
    agree on the forward count but not on bin composition would run
    collectives over different tokens. The packer is therefore called twice
    on fresh copies and must return identical bins. Each bin must be nonempty
    and fit ``bin_capacity``, and the bins must cover every row exactly once.
    """
    bins = tuple(tuple(indices) for indices in dense_packer(list(costs)))
    if tuple(tuple(indices) for indices in dense_packer(list(costs))) != bins:
        raise ValueError(
            "dense_packer must be deterministic: every model-parallel rank packs "
            "independently and must produce identical bins"
        )
    if sorted(index for indices in bins for index in indices) != list(range(len(costs))):
        raise ValueError("Dense packer bins must cover every source row exactly once")
    if any(
        not indices or sum(costs[index] for index in indices) > bin_capacity for indices in bins
    ):
        raise ValueError("Dense packer bins must be nonempty and within the token budget")
    return bins


def _pack_full_length_group(
    rows: Sequence[SharedPrefixRow],
    *,
    bin_capacity: int,
    padding_multiple: int,
    max_rows_per_slot: int,
) -> list[tuple[int, ...]]:
    """FFD-pack one prompt group's conventional full lengths into execution slots.

    Full duplicated lengths are deliberately conservative. Any resulting slot
    also fits as one deduplicated star because removing repeated prompts can
    only shorten it; if exact-prompt validation later fails, the same slot
    still fits as one ordinary block-diagonal packed microbatch.

    A group larger than ``max_rows_per_slot`` is spread over the fewest slots
    that respect it, with sizes that differ by at most one (17 -> 9+8 with the
    default 16, not a 16-row slot plus an unshared singleton), as
    ``plan_shared_prefix_bins`` does. Each slot lists its source row indices in
    input order. Every row must fit ``bin_capacity`` on its own.
    """
    padded_lengths = [_round_up(row.total_length, padding_multiple) for row in rows]
    num_bins = -(-len(rows) // max_rows_per_slot)
    # ``extra`` slots of ``base + 1`` rows and the rest of ``base``.
    base, extra = divmod(len(rows), num_bins) if num_bins else (max_rows_per_slot, 0)
    large_bins = 0
    bins: list[list[int]] = []
    bin_work: list[int] = []
    for position in sorted(range(len(rows)), key=lambda index: (-padded_lengths[index], index)):
        destination: int | None = None
        for bin_index, bin_positions in enumerate(bins):
            has_row_slot = len(bin_positions) < base or (
                len(bin_positions) == base and large_bins < extra
            )
            if has_row_slot and bin_work[bin_index] + padded_lengths[position] <= bin_capacity:
                destination = bin_index
                break
        if destination is None:
            bins.append([position])
            bin_work.append(padded_lengths[position])
        else:
            large_bins += len(bins[destination]) == base
            bins[destination].append(position)
            bin_work[destination] += padded_lengths[position]
    return [tuple(rows[position].row_index for position in sorted(positions)) for positions in bins]


def plan_shared_prefix_execution_units(
    rows: Sequence[SharedPrefixRow],
    *,
    bin_capacity: int,
    padding_multiple: int,
    pack_groups: bool = False,
    pack_dense_fallbacks: bool = False,
    merge_dense_fallbacks: bool = False,
    forward_only: bool = False,
    evaluation_packing: bool = False,
    dense_packer: DensePacker | None = None,
    largest_first: bool = False,
    max_completions_per_bin: int = MAX_SHARED_PREFIX_BRANCHES,
) -> tuple[SharedPrefixExecutionUnit, ...]:
    """Plan one rank's local batch as exact-prompt stars, forests and dense fallbacks.

    Each prompt group is split locally into execution slots: conventional
    full-length first-fit decreasing under ``bin_capacity``, with at most
    ``max_completions_per_bin`` rows per slot. Rows without a group ID and
    single-row groups get a slot each. A slot whose rows all share one exact
    prompt runs as one star. Every rank plans only its own rows, so forward
    counts can differ between ranks; a distributed caller must agree on one
    count and align to it before the first forward. Flags are validated
    before any row is read, so an invalid combination raises on every rank.

    Args:
        rows: Planner rows of this rank's local batch. ``row_index`` addresses
            the caller's batch and must be unique.
        bin_capacity: Token budget of one forward. It bounds every unit's
            physical length after padding, and in training also its expanded
            length (one prompt copy per row), which MTP processes.
        padding_multiple: Per-row TP/CP padding quantum.
        pack_groups: Allow several forwards per slot and coalesce stars into
            forests. A slot whose rows do not all share then keeps its stars
            and runs only the other rows dense. Without it, every slot is
            exactly one forward: a star, or a dense unit when any of its rows
            cannot share.
        pack_dense_fallbacks: Repack all dense rows across groups with
            ``dense_packer``. Requires ``pack_groups`` and ``dense_packer``.
        merge_dense_fallbacks: Training only: fold whole dense bins into
            forests as independent causal roots with their own MTP loss group.
            Requires ``pack_dense_fallbacks``; ignored with ``forward_only``.
        forward_only: Plan for a no-grad forward.
        evaluation_packing: With ``forward_only``, apply only the physical
            budget; the caller must disable MTP uniformly. Requires
            ``pack_groups``.
        dense_packer: Deterministic length-only packer for dense rows; see
            :func:`pack_dense_rows` for its contract.
        largest_first: Return the units in nonincreasing ``physical_length``
            order. Ties keep the planning order.
        max_completions_per_bin: Branch limit of one star.

    Returns:
        Execution units in execution order, covering every row exactly once.

    Raises:
        ValueError: If a limit or flag combination is invalid, a row index
            repeats, a row is empty or longer than ``bin_capacity`` after
            padding, the batch is empty, or ``dense_packer`` returns invalid
            bins.
        RuntimeError: If a planned dense fallback exceeds ``bin_capacity``,
            which valid rows cannot cause.
    """
    if bin_capacity < 1 or padding_multiple < 1:
        raise ValueError("Positive capacity and padding multiple required")
    if not _is_int(max_completions_per_bin) or max_completions_per_bin < 2:
        raise ValueError("max_completions_per_bin must be an integer of at least 2")
    if pack_dense_fallbacks and not pack_groups:
        raise ValueError("dense fallback repacking requires pack_groups")
    if pack_dense_fallbacks and dense_packer is None:
        raise ValueError("Dense fallback repacking requires a length-only packer")
    if merge_dense_fallbacks and not (pack_groups and pack_dense_fallbacks):
        raise ValueError("dense fallback merging requires pack_groups and pack_dense_fallbacks")
    if evaluation_packing and not pack_groups:
        raise ValueError("evaluation packing requires pack_groups")
    rows_by_index = {row.row_index: row for row in rows}
    if len(rows_by_index) != len(rows):
        raise ValueError("Execution planning requires unique source row indices")
    for row in rows:
        if row.total_length < 1:
            raise ValueError(
                f"shared-prefix row {row.row_index} must have a positive length, "
                f"got {row.total_length}"
            )
        padded_length = _round_up(row.total_length, padding_multiple)
        if padded_length > bin_capacity:
            raise ValueError(
                "shared-prefix row exceeds the execution bin capacity after "
                f"padding: row={row.row_index}, padded_length={padded_length}, "
                f"capacity={bin_capacity}"
            )
    # Split each complete group locally with the same conventional token
    # bound, so one long group cannot force unrelated groups into singleton
    # fallbacks; both expanded MTP work and prompt-mismatch fallbacks still
    # fit. Forward counts can then differ between data-parallel ranks, so a
    # distributed caller must align them.
    grouped_indices: dict[str | None, list[int]] = {}
    for row in rows:
        grouped_indices.setdefault(row.group_id, []).append(row.row_index)
    row_slots: list[tuple[int, ...]] = []
    for group_id, indices in grouped_indices.items():
        if group_id is None or len(indices) == 1:
            # Rows without a group ID never share a prompt, however many there
            # are: each one is its own dense slot.
            row_slots.extend((index,) for index in indices)
            continue
        if forward_only and evaluation_packing:
            # Evaluation never expands these stars for MTP. Apply the
            # physical budget before splitting a group, as well as when
            # packing stars together below. Retaining the training slot
            # cuts here would still duplicate a large prefix unnecessarily.
            evaluation_plan = plan_shared_prefix_bins(
                [rows_by_index[index] for index in indices],
                bin_capacity=bin_capacity,
                max_completions_per_bin=max_completions_per_bin,
                sequence_length_pad_multiple=padding_multiple,
            )
            row_slots.extend(layout.row_indices for layout in evaluation_plan.shared_bins)
            row_slots.extend((row_index,) for row_index in evaluation_plan.fallback_row_indices)
            continue
        row_slots.extend(
            _pack_full_length_group(
                [rows_by_index[index] for index in indices],
                bin_capacity=bin_capacity,
                padding_multiple=padding_multiple,
                max_rows_per_slot=max_completions_per_bin,
            )
        )
    units: list[SharedPrefixExecutionUnit] = []
    for row_indices in row_slots:
        candidate = plan_shared_prefix_bins(
            [rows_by_index[index] for index in row_indices],
            bin_capacity=bin_capacity,
            max_completions_per_bin=max_completions_per_bin,
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

        dense_rows = row_indices
        if pack_groups and candidate.shared_bins:
            # Group packing already allows several forwards per slot, so rows
            # that cannot share (a prompt mismatch, an empty placeholder) run
            # dense without taking the slot's stars with them. Each star is a
            # subset of a slot that fits as dense rows, so it fits both budgets.
            units.extend(
                SharedPrefixExecutionUnit(
                    row_indices=layout.row_indices,
                    shared_layout=layout,
                    physical_length=layout.physical_total_length,
                )
                for layout in candidate.shared_bins
            )
            fallback = set(candidate.fallback_row_indices)
            dense_rows = tuple(index for index in row_indices if index in fallback)
            if not dense_rows:
                continue
        fallback_length = sum(
            _round_up(rows_by_index[index].total_length, padding_multiple) for index in dense_rows
        )
        if fallback_length > bin_capacity:
            raise RuntimeError(
                "shared-prefix dense fallback exceeds its bin capacity: "
                f"rows={dense_rows}, padded_length={fallback_length}, capacity={bin_capacity}"
            )
        units.append(
            SharedPrefixExecutionUnit(
                row_indices=dense_rows, shared_layout=None, physical_length=fallback_length
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
            # prompt groups instead of keeping each group's slot cuts. This
            # also recovers conventional batching when every slot falls back,
            # while leaving the shared attention layout unchanged.
            fallback_rows = sorted(
                index for unit in units if unit.shared_layout is None for index in unit.row_indices
            )
            if fallback_rows:
                lengths = [
                    _round_up(rows_by_index[index].total_length, padding_multiple)
                    for index in fallback_rows
                ]
                assert dense_packer is not None  # Checked on entry.
                dense_units = [
                    SharedPrefixExecutionUnit(
                        row_indices=tuple(fallback_rows[index] for index in indices),
                        shared_layout=None,
                        physical_length=sum(lengths[index] for index in indices),
                    )
                    for indices in pack_dense_rows(
                        lengths, bin_capacity=bin_capacity, dense_packer=dense_packer
                    )
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
    if largest_first:
        # Stable, so equal lengths keep their deterministic planning order.
        units = sorted(units, key=lambda unit: -unit.physical_length)
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
        expanded_lengths[index] = layout.dense_length
    absorbed: set[int] = set()
    fallback_indices = sorted(
        (index for index, unit in enumerate(units) if unit.shared_layout is None),
        key=lambda index: -units[index].physical_length,
    )
    for fallback_index in fallback_indices:
        fallback = units[fallback_index]
        fallback_rows = [rows_by_index[index] for index in fallback.row_indices]
        if any(
            row.group_id is None or row.prompt_length == 0 or row.completion_length == 0
            for row in fallback_rows
        ):
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
