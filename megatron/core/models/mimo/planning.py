# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Plan projected media rows independently of decoder packing and CP ownership.

These helpers operate on replicated descriptors, not media tensors. Callers own
media preparation and cost estimates; the existing packing scheduler owns sample
assignments. Sample and media identities need only be hashable, not contiguous
integers. Every rank must supply descriptors in the same order.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any, Hashable, Mapping, Sequence

from megatron.core.models.mimo.comm.pack_bridge import FeatureSlice, PackFeaturePlan


@dataclass(frozen=True)
class MediaTask:
    """One independently encoded media item with a caller-provided cost estimate.

    ``length`` is the number of projected rows consumed by the decoder. Tasks
    have unique identities; a sample may reference the same task more than once.
    """

    media_id: Hashable
    length: int
    cost: float = 1.0


def assign_media_to_producers(
    tasks: Sequence[MediaTask], producer_ranks: Sequence[int]
) -> tuple[dict[int, list[MediaTask]], dict[Hashable, FeatureSlice]]:
    """Greedily balance whole media tasks and establish each producer's row offsets.

    Equal costs use input order, so identities never need to be sortable. Each
    producer encodes its assigned tasks in their original input order and must
    concatenate projected rows in that order.
    """
    producer_ranks = tuple(producer_ranks)
    if len(set(producer_ranks)) != len(producer_ranks):
        raise ValueError("Encoder producer ranks must be unique")
    if tasks and not producer_ranks:
        raise ValueError("Media tasks require at least one encoder producer")
    if len({task.media_id for task in tasks}) != len(tasks):
        raise ValueError("Each encoder media task must have a unique media ID")
    if any(
        not isinstance(task.length, int)
        or task.length <= 0
        or not isfinite(task.cost)
        or task.cost < 0
        for task in tasks
    ):
        raise ValueError("Projected row counts must be positive and costs finite and nonnegative")

    indices_by_rank = {rank: [] for rank in producer_ranks}
    loads = [0.0] * len(producer_ranks)
    for index in sorted(range(len(tasks)), key=lambda index: (-tasks[index].cost, index)):
        producer = min(range(len(producer_ranks)), key=lambda rank: (loads[rank], rank))
        indices_by_rank[producer_ranks[producer]].append(index)
        loads[producer] += tasks[index].cost

    assignments, feature_slices = {}, {}
    for rank, indices in indices_by_rank.items():
        local_tasks = [tasks[index] for index in sorted(indices)]
        assignments[rank] = local_tasks
        offset = 0
        for task in local_tasks:
            feature_slices[task.media_id] = FeatureSlice(rank, offset, task.length)
            offset += task.length
    return assignments, feature_slices


def build_round_plans(
    sample_ids_by_rank: Sequence[Sequence[Hashable]],
    samples: Mapping[Hashable, Mapping[str, Any]],
    feature_slices: Mapping[Hashable, FeatureSlice],
    domain_ranks: Sequence[int],
) -> tuple[PackFeaturePlan, ...]:
    """Join one scheduled decoder round with ordered media references.

    Identical ordered sample tuples identify ranks sharing one decoder pack.
    ``samples[sample_id]['media_ids']`` follows placeholder order within that
    sample; repeated references are preserved. Runtime group selection and
    validation against actual process-group ranks remain the caller's job.
    """
    domain_ranks = tuple(domain_ranks)
    if len(sample_ids_by_rank) != len(domain_ranks) or len(set(domain_ranks)) != len(domain_ranks):
        raise ValueError("Round assignments must match a domain of unique global ranks")
    packs = {}
    for rank, sample_ids in zip(domain_ranks, sample_ids_by_rank):
        sample_ids = tuple(sample_ids)
        if not sample_ids or any(sample_id not in samples for sample_id in sample_ids):
            raise ValueError("Each decoder rank needs a nonempty pack of known sample IDs")
        packs.setdefault(sample_ids, []).append(rank)
    plans = []
    for sample_ids, ranks in packs.items():
        media_ids = [media_id for sid in sample_ids for media_id in samples[sid]['media_ids']]
        if any(media_id not in feature_slices for media_id in media_ids):
            raise ValueError("Every referenced media item needs a producer feature slice")
        plans.append(
            PackFeaturePlan(tuple(ranks), tuple(feature_slices[media_id] for media_id in media_ids))
        )
    return tuple(plans)
