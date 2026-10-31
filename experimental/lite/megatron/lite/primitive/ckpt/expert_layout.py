# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Redistribute expert deployment tensors without gathering all experts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ExpertLayoutPlan:
    """Map contiguous expert shards within one explicit exchange group.

    Args:
        num_experts: Number of global experts in each layer.
        source_ep_size: Number of source expert shards.
        target_ep_size: Number of destination expert shards.
        global_ranks: Ordered global ranks in the exchange group. Group position
            modulo EP size identifies the source or destination shard. Repeated
            shard positions represent replicas, not tensor-parallel slices.
    """

    num_experts: int
    source_ep_size: int
    target_ep_size: int
    global_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        for size in (self.num_experts, self.source_ep_size, self.target_ep_size):
            if size <= 0:
                raise ValueError('Expert counts and EP sizes must be positive')
        if not self.global_ranks or len(set(self.global_ranks)) != len(self.global_ranks):
            raise ValueError('Exchange ranks must be nonempty and unique')
        for size in (self.source_ep_size, self.target_ep_size):
            if self.num_experts % size or len(self.global_ranks) % size:
                raise ValueError(
                    'Expert count and exchange group size must be divisible by EP sizes'
                )

    def expert_owner(self, expert_id: int, *, target: bool = False) -> int:
        """Return the logical source or target EP owner of a global expert."""
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(f'Expert {expert_id} is outside [0, {self.num_experts})')
        ep_size = self.target_ep_size if target else self.source_ep_size
        return expert_id // (self.num_experts // ep_size)

    def source_rank(self, expert_id: int) -> int:
        """Choose one source replica for an expert, avoiding duplicate sends."""
        owner = self.expert_owner(expert_id)
        holders = self.global_ranks[owner :: self.source_ep_size]
        return holders[self.expert_owner(expert_id, target=True) % len(holders)]

    def destination_ranks(self, expert_id: int) -> tuple[int, ...]:
        """Return every destination replica that needs an expert."""
        owner = self.expert_owner(expert_id, target=True)
        return self.global_ranks[owner :: self.target_ep_size]


def _all_gather_metadata(value, group: dist.ProcessGroup):
    gathered = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _next_bucket(iterator, pending, plan, rank, expert_id_from_name, limit, seen):
    bucket = []
    nbytes = 0
    while True:
        if pending is not None:
            name, tensor, expert = pending
            pending = None
        else:
            item = next(iterator, None)
            if item is None:
                break
            name, tensor = item
            if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
                raise TypeError('Expected (parameter name, Tensor) entries')
            expert = expert_id_from_name(name)
            owner = plan.expert_owner(expert)
            if plan.global_ranks.index(rank) % plan.source_ep_size != owner:
                raise ValueError(f'Rank {rank} does not own expert {expert}: {name}')
            if name in seen:
                raise ValueError(f'Duplicate parameter name: {name}')
            seen.add(name)
            if rank != plan.source_rank(expert):
                continue
            if tensor.layout != torch.strided:
                raise ValueError('Only strided deployment tensors are supported')
            tensor = tensor.detach().contiguous()
        size = tensor.numel() * tensor.element_size()
        if bucket and nbytes + size > limit:
            pending = (name, tensor, expert)
            break
        bucket.append((name, tensor, expert))
        nbytes += size
        if nbytes >= limit:
            break
    return bucket, pending


def exchange_expert_layout_tensors(
    weights: Iterable[tuple[str, torch.Tensor]],
    plan: ExpertLayoutPlan,
    *,
    group: dist.ProcessGroup,
    device: torch.device,
    expert_id_from_name: Callable[[str], int],
    buffer_max_size_bytes: int = 64 * 1024**2,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Exchange expert weights and scales as opaque bytes in collective rounds.

    Every group member must fully consume this iterator, even when its local
    stream is empty. All ranks must use the same plan and byte budget. Source
    replicas must contain identical deployment values. Inputs contain only
    expert tensors, with globally unique names that include the expert ID;
    ordinary parameters remain on the caller's existing synchronization path.

    Args:
        weights: Actor-local, materialized expert tensors, including scales.
        plan: Contiguous source and destination expert ownership.
        group: Explicit process group in exactly ``plan.global_ranks`` order.
        device: Collective device: CPU for Gloo, local CUDA device for NCCL.
        expert_id_from_name: Extract the global expert ID from each tensor name.
        buffer_max_size_bytes: Source bytes per round before destination
            replication. A single larger tensor is sent alone. Receiver buffers
            may contain contributions from multiple sources.

    Yields:
        Globally named tensors for the current rank's destination EP shard.
        Shapes, dtypes and bytes are preserved; no quantization is performed.
    """
    device = torch.device(device)
    actual_ranks = tuple(dist.get_process_group_ranks(group))
    contract = (plan, actual_ranks, buffer_max_size_bytes, device.type)
    contracts = _all_gather_metadata(contract, group)
    if any(value != contracts[0] for value in contracts):
        raise ValueError('All ranks must use the same expert layout, budget and device type')
    if actual_ranks != plan.global_ranks:
        raise ValueError('Process group rank order does not match the expert layout')
    if buffer_max_size_bytes <= 0:
        raise ValueError('buffer_max_size_bytes must be positive')
    backend = dist.get_backend(group)
    if (backend == 'nccl' and device.type != 'cuda') or (
        backend == 'gloo' and device.type != 'cpu'
    ):
        raise ValueError('NCCL requires CUDA tensors; Gloo requires CPU tensors')

    rank = dist.get_rank()
    iterator = iter(weights)
    pending = None
    local_names = set()
    exchanged_names = set()
    while True:
        error = None
        try:
            bucket, pending = _next_bucket(
                iterator,
                pending,
                plan,
                rank,
                expert_id_from_name,
                buffer_max_size_bytes,
                local_names,
            )
            entries = [
                (
                    name,
                    expert,
                    tensor.dtype,
                    tuple(tensor.shape),
                    tensor.numel() * tensor.element_size(),
                )
                for name, tensor, expert in bucket
            ]
        except (ValueError, TypeError, RuntimeError) as exc:
            bucket, entries = [], []
            error = str(exc)
        headers = _all_gather_metadata((entries, error), group)
        errors = [message for _, message in headers if message is not None]
        if errors:
            raise ValueError(f'Invalid expert stream: {errors[0]}')
        if not any(entries for entries, _ in headers):
            return
        # Validate names on every rank before any tensor collective. Replicated
        # destinations are derived from the plan, so one source header suffices.
        for source_index, (entries, _) in enumerate(headers):
            for name, expert, _, _, _ in entries:
                if plan.source_rank(expert) != actual_ranks[source_index]:
                    raise ValueError(f'Incorrect source for expert {expert}')
                if name in exchanged_names:
                    raise ValueError(f'Duplicate parameter name: {name}')
                exchanged_names.add(name)

        parts = [[] for _ in actual_ranks]
        for _, tensor, expert in bucket:
            # Flatten before viewing bytes so scalar scales work too.
            payload = tensor.to(device).reshape(-1).view(torch.uint8)
            for destination in plan.destination_ranks(expert):
                parts[actual_ranks.index(destination)].append(payload)
        input_splits = [sum(part.numel() for part in chunks) for chunks in parts]
        output_splits = [
            sum(size for _, expert, _, _, size in entries if rank in plan.destination_ranks(expert))
            for entries, _ in headers
        ]
        send = (
            torch.cat([part for chunks in parts for part in chunks])
            if sum(input_splits)
            else torch.empty(0, dtype=torch.uint8, device=device)
        )
        received = torch.empty(sum(output_splits), dtype=torch.uint8, device=device)
        dist.all_to_all_single(
            received,
            send,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        offset = 0
        for entries, _ in headers:
            for name, expert, dtype, shape, size in entries:
                if rank not in plan.destination_ranks(expert):
                    continue
                # Copy the byte slice to aligned storage before viewing a wider
                # dtype: a preceding uint8 tensor can leave an odd byte offset.
                tensor = received[offset : offset + size].clone().view(dtype).reshape(shape)
                offset += size
                yield name, tensor
