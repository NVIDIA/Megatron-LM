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

"""Reconstruct shared prefixes within already aligned conventional training bins."""

from collections.abc import Sequence

from megatron.rl.shared_prefix_execution import (
    DensePacker,
    SharedPrefixExecutionUnit,
    pack_dense_rows,
)
from megatron.rl.shared_prefix_packing import (
    MAX_SHARED_PREFIX_BRANCHES,
    SharedPrefixForestLayout,
    SharedPrefixRow,
    build_shared_prefix_layout,
    split_rows_evenly,
)


def _validate_dense_partition(
    units: Sequence[SharedPrefixExecutionUnit], costs: Sequence[int], capacity: int
) -> None:
    if sorted(row for unit in units for row in unit.row_indices) != list(range(len(costs))):
        raise ValueError("Dense training bins must cover every source row exactly once")
    for unit in units:
        expanded = sum(costs[row] for row in unit.row_indices)
        if (
            unit.shared_layout is not None
            or not unit.row_indices
            or not 0 < expanded <= capacity
            or unit.physical_length != expanded
        ):
            raise ValueError("Expected conventional training bins within the token budget")


def plan_dense_training_bins(
    *, costs: Sequence[int], bin_capacity: int, dense_packer: DensePacker
) -> tuple[SharedPrefixExecutionUnit, ...]:
    """Pack expanded rows before distributed forward-count alignment.

    ``dense_packer`` must be deterministic; see
    :func:`~megatron.rl.shared_prefix_execution.pack_dense_rows`.
    """
    if any(cost <= 0 or cost > bin_capacity for cost in costs):
        raise ValueError("Training rows must fit the expanded token budget")
    bins = pack_dense_rows(costs, bin_capacity=bin_capacity, dense_packer=dense_packer)
    units = tuple(
        SharedPrefixExecutionUnit(
            row_indices=tuple(indices),
            shared_layout=None,
            physical_length=sum(costs[row] for row in indices),
        )
        for indices in bins
    )
    _validate_dense_partition(units, costs, bin_capacity)
    return units


def share_prefixes_in_dense_training_bins(
    rows: Sequence[SharedPrefixRow],
    units: Sequence[SharedPrefixExecutionUnit],
    *,
    costs: Sequence[int],
    padding_multiple: int,
    bin_capacity: int,
    max_completions_per_bin: int = MAX_SHARED_PREFIX_BRANCHES,
) -> tuple[SharedPrefixExecutionUnit, ...]:
    """Share exact prompts in aligned dense bins without changing MTP groups.

    Each output bin retains one auxiliary-loss normalization group, source-row
    coverage, and independent causal boundaries. Ineligible bins stay dense.
    An exact-prompt group with more than ``max_completions_per_bin`` rows is
    split evenly into several roots.
    """
    multiple = padding_multiple
    _validate_dense_partition(units, costs, bin_capacity)
    result = []
    for unit in units:
        selected = [rows[index] for index in unit.row_indices]
        if any(
            row.group_id is None or row.prompt_length == 0 or row.completion_length == 0
            for row in selected
        ):
            result.append(unit)
            continue
        groups = {}
        for row in selected:
            groups.setdefault((row.group_id, row.prompt_token_ids), []).append(row)
        if all(len(group) == 1 for group in groups.values()):
            result.append(unit)
            continue
        roots = tuple(
            build_shared_prefix_layout(
                chunk, sequence_length_pad_multiple=multiple, allow_singleton=True
            )
            for group in groups.values()
            for chunk in split_rows_evenly(group, max_completions_per_bin=max_completions_per_bin)
        )
        forest = SharedPrefixForestLayout(roots, mtp_loss_group_root_counts=(len(roots),))
        expanded = sum(
            len(root.row_indices) * root.prompt_length + sum(root.physical_completion_lengths)
            for root in roots
        )
        padded_physical = (forest.physical_total_length + multiple - 1) // multiple * multiple
        if (
            expanded != unit.physical_length
            or padded_physical > expanded
            or sorted(forest.row_indices) != sorted(unit.row_indices)
        ):
            raise ValueError("Shared reconstruction changed the dense training bin")
        result.append(
            SharedPrefixExecutionUnit(
                row_indices=forest.row_indices,
                shared_layout=forest,
                physical_length=forest.physical_total_length,
            )
        )
    return tuple(result)
