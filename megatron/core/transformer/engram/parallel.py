# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Engram row ownership and replica groups, shared by all layers of a model."""

from dataclasses import dataclass
from typing import Sequence, Tuple
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather

from megatron.core import parallel_state, tensor_parallel

from .config import _parallel_world_size

_GROUP_CACHE = WeakKeyDictionary()


@dataclass(frozen=True)
class EngramParallelGroups:
    """Process groups for an E-column, R-row rank grid.

    Each grid row owns a complete table partitioned across E ranks. Columns
    contain replicas of the same row shard; native DistributedOptimizer shards
    their Adam state across R ranks. The full grid counts gradient shards once.
    """

    table_group: dist.ProcessGroup
    replica_group: dist.ProcessGroup
    stats_group: dist.ProcessGroup
    participant_group: dist.ProcessGroup
    shard_size: int
    replica_size: int
    shard_rank: int
    replica_rank: int

    @classmethod
    def create(
        cls,
        participants: dist.ProcessGroup,
        shard_size: int | None = None,
        *,
        stats_group: dist.ProcessGroup | None = None,
    ) -> 'EngramParallelGroups':
        """Create an E/R grid within one PP stage and reuse it across VPP chunks.

        All stage participants construct the same disjoint rows and columns in
        the same order. Local synchronization avoids collectives on other PP stages.
        Statistical reductions include every stage through the explicit stats group.
        """
        ranks = dist.get_process_group_ranks(participants)
        world_size = len(ranks)
        shard_size = world_size if shard_size is None else shard_size
        if shard_size < 1 or world_size % shard_size:
            raise ValueError('Engram row parallel size must divide the stage participant count')
        stats_group = participants if stats_group is None else stats_group
        cache = _GROUP_CACHE.setdefault(participants, {})
        key = (shard_size, stats_group)
        if key in cache:
            return cache[key]
        replica_size = world_size // shard_size
        replica_rank, shard_rank = divmod(dist.get_rank(group=participants), shard_size)
        backend = dist.get_backend(participants)
        table_group = replica_group = None
        for row in range(replica_size):
            group = (
                participants
                if replica_size == 1
                else dist.new_group(
                    ranks[row * shard_size : (row + 1) * shard_size],
                    backend=backend,
                    use_local_synchronization=True,
                    group_desc='engram_table',
                )
            )
            if row == replica_rank:
                table_group = group
        for column in range(shard_size):
            group = (
                participants
                if shard_size == 1
                else dist.new_group(
                    ranks[column::shard_size],
                    backend=backend,
                    use_local_synchronization=True,
                    group_desc='engram_replica',
                )
            )
            if column == shard_rank:
                replica_group = group
        result = cls(
            table_group,
            replica_group,
            stats_group,
            participants,
            shard_size,
            replica_size,
            shard_rank,
            replica_rank,
        )
        cache[key] = result
        return result


class ParallelSequenceLayout:
    """Restore and repartition Megatron SP/CP sequence layouts around Engram."""

    @staticmethod
    def _restore_cp(tensors: Sequence[Tensor], sequence_dim: int) -> Tensor:
        chunks = [None] * (2 * len(tensors))
        for rank, tensor in enumerate(tensors):
            first, second = tensor.chunk(2, dim=sequence_dim)
            chunks[rank] = first
            chunks[2 * len(tensors) - rank - 1] = second
        return torch.cat(chunks, dim=sequence_dim)

    @staticmethod
    def _select_cp(tensor: Tensor, sequence_dim: int, cp_group=None) -> Tensor:
        cp_size = (
            torch.distributed.get_world_size(group=cp_group)
            if cp_group is not None
            else _parallel_world_size(parallel_state.get_context_parallel_world_size)
        )
        if cp_size == 1:
            return tensor
        cp_rank = (
            torch.distributed.get_rank(group=cp_group)
            if cp_group is not None
            else parallel_state.get_context_parallel_rank()
        )
        chunks = tensor.chunk(2 * cp_size, dim=sequence_dim)
        return torch.cat((chunks[cp_rank], chunks[2 * cp_size - cp_rank - 1]), dim=sequence_dim)

    @classmethod
    def gather(
        cls,
        hidden_states: Tensor,
        compressed_input_ids: Tensor,
        sequence_parallel: bool,
        tp_group=None,
        cp_group=None,
    ) -> Tuple[Tensor, Tensor]:
        """Restore complete sequences and token IDs around the Engram operation."""
        if sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=tp_group
            )

        cp_size = (
            torch.distributed.get_world_size(group=cp_group)
            if cp_group is not None
            else _parallel_world_size(parallel_state.get_context_parallel_world_size)
        )
        if cp_size == 1:
            return hidden_states, compressed_input_ids

        cp_group = parallel_state.get_context_parallel_group() if cp_group is None else cp_group
        hidden_states = cls._restore_cp(differentiable_all_gather(hidden_states, group=cp_group), 0)

        def gather_ids(value: Tensor) -> Tensor:
            gathered = [torch.empty_like(value) for _ in range(cp_size)]
            torch.distributed.all_gather(gathered, value, group=cp_group)
            return cls._restore_cp(gathered, 1)

        return hidden_states, gather_ids(compressed_input_ids)

    @classmethod
    def scatter(
        cls, hidden_states: Tensor, sequence_parallel: bool, tp_group=None, cp_group=None
    ) -> Tensor:
        """Return fused states in the caller's CP/SP sequence partition."""
        hidden_states = cls._select_cp(hidden_states, 0, cp_group=cp_group)
        if sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(
                hidden_states, group=tp_group
            )
        return hidden_states
