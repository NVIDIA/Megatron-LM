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

"""Pure group-coherent sharding of shared-prefix batches.

No batch transport, training configuration, or model runtime is required.
"""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GroupCoherentShardPlan:
    """Stable row-index assignment that never splits a prompt group.

    ``rank_order_permutation`` maps each position of the rank-major
    concatenation to its source row (output position -> source row), which is
    the forward permutation NeMo RL's ``BatchedDataDict.reorder_data`` expects to restore
    source order. ``None`` means the concatenation is already in source order.
    """

    shard_indices: tuple[tuple[int, ...], ...]
    rank_order_permutation: tuple[int, ...] | None


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
