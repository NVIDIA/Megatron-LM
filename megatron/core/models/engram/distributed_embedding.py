# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native differentiable EP-sharded embedding lookup for Engram."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size, nvtx_range_pop, nvtx_range_push


class _DifferentiableEngramAllToAll(torch.autograd.Function):
    """Variable-split all-to-all with Engram-specific forward/backward NVTX ranges."""

    @staticmethod
    def _exchange(group, input_: Tensor, output_splits: list[int], input_splits: list[int]):
        output = input_.new_empty((sum(output_splits), *input_.shape[1:]))
        torch.distributed.all_to_all_single(
            output,
            input_.contiguous(),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return output

    @staticmethod
    def forward(ctx, group, input_: Tensor, output_splits: list[int], input_splits: list[int]):
        ctx.group = group
        ctx.output_splits = output_splits
        ctx.input_splits = input_splits
        message = "engram.lookup.return-a2a.forward"
        nvtx_range_push(message)
        try:
            return _DifferentiableEngramAllToAll._exchange(
                group, input_, output_splits, input_splits
            )
        finally:
            nvtx_range_pop(message)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        message = "engram.lookup.return-a2a.backward"
        nvtx_range_push(message)
        try:
            grad_input = _DifferentiableEngramAllToAll._exchange(
                ctx.group, grad_output, ctx.input_splits, ctx.output_splits
            )
        finally:
            nvtx_range_pop(message)
        return None, grad_input, None, None


class _DeterministicEmbedding(torch.autograd.Function):
    """Embedding lookup with sorted, segmented accumulation for repeated-row gradients."""

    @staticmethod
    def forward(ctx, weight: Tensor, row_ids: Tensor) -> Tensor:
        ctx.save_for_backward(row_ids)
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return weight.index_select(0, row_ids)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        message = "engram.embedding.backward"
        nvtx_range_push(message)
        try:
            (row_ids,) = ctx.saved_tensors
            accumulation_dtype = (
                torch.float32
                if grad_output.dtype in (torch.float16, torch.bfloat16)
                else grad_output.dtype
            )
            # The table shard is large, so allocate the dense gradient once in its final dtype
            # and keep the higher-precision accumulation to the touched rows.
            grad_weight = torch.zeros(
                ctx.weight_shape, dtype=ctx.weight_dtype, device=grad_output.device
            )
            if row_ids.numel() > 0:
                order = torch.argsort(row_ids, stable=True)
                sorted_rows = row_ids.index_select(0, order)
                sorted_grads = grad_output.index_select(0, order).to(accumulation_dtype)
                unique_rows, counts = torch.unique_consecutive(sorted_rows, return_counts=True)
                row_grads = torch.segment_reduce(sorted_grads, "sum", lengths=counts)
                grad_weight.index_copy_(0, unique_rows, row_grads.to(ctx.weight_dtype))
            return grad_weight, None
        finally:
            nvtx_range_pop(message)


def get_contiguous_row_range(global_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return the balanced, unpadded contiguous row interval owned by ``rank``."""
    if global_rows < 0:
        raise ValueError(f"global_rows must be nonnegative, got {global_rows}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")
    base, remainder = divmod(global_rows, world_size)
    start = rank * base + min(rank, remainder)
    local_rows = base + int(rank < remainder)
    return start, start + local_rows


class EPShardedEmbeddingTable(MegatronModule):
    """One prime-sized embedding table sharded over an explicit EP group."""

    def __init__(
        self,
        config,
        global_num_embeddings: int,
        embedding_dim: int,
        init_method: Callable[[Tensor], None],
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
    ) -> None:
        super().__init__(config=config)
        self.global_num_embeddings = global_num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_group = tp_group
        self.expt_dp_group = expt_dp_group
        self.deterministic_mode = getattr(config, "deterministic_mode", False)
        self.ep_rank = get_pg_rank(ep_group)
        self.ep_size = get_pg_size(ep_group)
        self.row_start, self.row_end = get_contiguous_row_range(
            global_num_embeddings, self.ep_rank, self.ep_size
        )

        device = (
            torch.device("cpu") if config.use_cpu_initialization else torch.cuda.current_device()
        )
        self.weight = nn.Parameter(
            torch.empty(
                (self.row_end - self.row_start, embedding_dim),
                device=device,
                dtype=config.params_dtype,
            )
        )
        if config.perform_initialization:
            self._initialize_weight(init_method)

        # The row shard is distinct across EP and synchronized only over expert-DP.
        self.weight.allreduce = False
        self.weight.is_engram_embedding = True
        self.weight.is_embedding_or_output_parameter = True
        if config.sequence_parallel:
            self.weight.sequence_parallel = True

    def _initialize_weight(self, init_method: Callable[[Tensor], None]) -> None:
        """Initialize the row shard on a forked RNG stream keyed by (table, EP rank).

        EP ranks own different row counts (prime table sizes never divide evenly), so drawing
        from the default generator would desynchronize it across EP ranks and silently diverge
        every subsequently initialized replicated parameter across data-parallel peers. The
        fork keeps the default stream untouched; the shard seed varies only with the globally
        unique prime table size and the EP rank, so TP and expert-DP replicas stay identical
        while EP shards are decorrelated.
        """
        base_seed = torch.initial_seed() % (2**31)
        shard_seed = base_seed + 100003 * self.global_num_embeddings + self.ep_rank
        devices = [self.weight.device] if self.weight.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if devices:
                torch.cuda.manual_seed(shard_seed)
            else:
                # Seed only the CPU generator: torch.manual_seed would also reseed every CUDA
                # default generator with the EP-rank-dependent shard seed, and fork_rng([])
                # restores only the CPU state.
                torch.default_generator.manual_seed(shard_seed)
            init_method(self.weight)

    @property
    def local_num_embeddings(self) -> int:
        """Number of rows owned by this rank."""
        return self.row_end - self.row_start

    def forward(self, local_row_ids: Tensor) -> Tensor:
        """Look up owner-local row IDs."""
        if self.deterministic_mode:
            return _DeterministicEmbedding.apply(self.weight, local_row_ids)
        return F.embedding(local_row_ids, self.weight)

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Represent this uneven EP row shard with its exact global logical shape."""
        del metadata
        prepend_axis_num = len(sharded_offsets)
        global_shape = [1] * prepend_axis_num + [self.global_num_embeddings, self.embedding_dim]
        global_offset = [0] * prepend_axis_num + [self.row_start, 0]
        for axis, rank_offset, axis_fragmentation in sharded_offsets:
            global_shape[axis] = axis_fragmentation
            global_offset[axis] = rank_offset

        replica_id = (0, get_pg_rank(self.tp_group), get_pg_rank(self.expt_dp_group))
        key = f"{prefix}weight"
        return {
            key: ShardedTensor(
                key=key,
                data=self.weight,
                dtype=self.weight.dtype,
                local_shape=tuple(self.weight.shape),
                global_shape=tuple(global_shape),
                global_offset=tuple(global_offset),
                axis_fragmentations=None,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            )
        }


class EPShardedMultiTableEmbedding(MegatronModule):
    """Batch variable-size requests for multiple independently EP-sharded tables."""

    def __init__(
        self,
        config,
        table_sizes: tuple[int, ...],
        embedding_dim: int,
        init_method: Callable[[Tensor], None],
        ep_group=None,
        tp_group=None,
        expt_dp_group=None,
    ) -> None:
        super().__init__(config=config)
        self.table_sizes = tuple(table_sizes)
        self.embedding_dim = embedding_dim
        self.ep_group = ep_group
        self.ep_size = get_pg_size(ep_group)
        self.tables = nn.ModuleList(
            [
                EPShardedEmbeddingTable(
                    config=config,
                    global_num_embeddings=table_size,
                    embedding_dim=embedding_dim,
                    init_method=init_method,
                    ep_group=ep_group,
                    tp_group=tp_group,
                    expt_dp_group=expt_dp_group,
                )
                for table_size in self.table_sizes
            ]
        )

        # Per-table balanced-partition constants let forward resolve the owning EP rank of any
        # global row in closed form, without materializing per-request [N, ep_size] boundaries.
        device = self.tables[0].weight.device
        bases_and_remainders = [divmod(table_size, self.ep_size) for table_size in self.table_sizes]
        self.register_buffer(
            "table_row_base",
            torch.tensor(
                [base for base, _ in bases_and_remainders], dtype=torch.int64, device=device
            ),
            persistent=False,
        )
        self.register_buffer(
            "table_row_remainder",
            torch.tensor(
                [remainder for _, remainder in bases_and_remainders],
                dtype=torch.int64,
                device=device,
            ),
            persistent=False,
        )

    @property
    def local_parameter_count(self) -> int:
        """Local sparse parameter count."""
        return sum(table.weight.numel() for table in self.tables)

    @property
    def global_parameter_count(self) -> int:
        """Global logical sparse parameter count."""
        return sum(self.table_sizes) * self.embedding_dim

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Preserve each table's irregular EP row metadata through the ModuleList container."""
        sharded_state_dict = {}
        for table_id, table in enumerate(self.tables):
            sharded_state_dict.update(
                table.sharded_state_dict(
                    prefix=f"{prefix}tables.{table_id}.",
                    sharded_offsets=sharded_offsets,
                    metadata=metadata,
                )
            )
        return sharded_state_dict

    def _lookup_received_requests(self, requests: Tensor, table_counts: list[int]) -> Tensor:
        """Look up owner-local rows, grouping the received requests by table.

        ``table_counts`` comes from the request-count exchange, so no extra host
        synchronization is needed on the owner-side critical path.
        """
        order = torch.argsort(requests[:, 0], stable=True)
        sorted_rows = requests[:, 1].index_select(0, order)
        values = []
        start = 0
        for table, count in zip(self.tables, table_counts):
            values.append(table(sorted_rows.narrow(0, start, count)))
            start += count
        output = self.tables[0].weight.new_empty((requests.shape[0], self.embedding_dim))
        output.index_copy_(0, order, torch.cat(values))
        return output

    def forward(self, hash_ids: Tensor) -> Tensor:
        """Route global table rows to owners and restore token/head ordering.

        Args:
            hash_ids: Per-table row IDs with shape ``[..., num_tables]``.

        Returns:
            Retrieved embeddings with shape ``[..., num_tables, embedding_dim]``.
        """
        if hash_ids.shape[-1] != len(self.tables):
            raise ValueError(
                f"Expected {len(self.tables)} Engram hash heads, got {hash_ids.shape[-1]}."
            )
        original_shape = hash_ids.shape

        if self.ep_size == 1:
            # Per-table layout is statically known: no routing, sorting, or host sync needed.
            values = [
                table(hash_ids[..., table_id].reshape(-1).to(torch.int64))
                for table_id, table in enumerate(self.tables)
            ]
            return torch.stack(values, dim=-2).view(*original_shape, self.embedding_dim)

        rows = hash_ids.reshape(-1).to(torch.int64)
        table_ids = (
            torch.arange(len(self.tables), device=hash_ids.device, dtype=torch.int64)
            .view(*([1] * (hash_ids.ndim - 1)), -1)
            .expand(original_shape)
            .reshape(-1)
        )

        message = "engram.lookup.owner-map"
        nvtx_range_push(message)
        try:
            # Balanced contiguous partition: ranks below `remainder` own `base + 1` rows and the
            # rest own `base`, so the owner of a global row follows in O(1) per request.
            base = self.table_row_base.index_select(0, table_ids)
            remainder = self.table_row_remainder.index_select(0, table_ids)
            large_region_end = (base + 1) * remainder
            owners = torch.where(
                rows < large_region_end,
                rows // (base + 1),
                remainder + (rows - large_region_end) // base.clamp_min(1),
            )
            owner_starts = owners * base + torch.minimum(owners, remainder)
            local_rows = rows - owner_starts

            owner_order = torch.argsort(owners, stable=True)
            sorted_owners = owners.index_select(0, owner_order)
            requests = torch.stack(
                (table_ids.index_select(0, owner_order), local_rows.index_select(0, owner_order)),
                dim=-1,
            )
            # Count per (peer, table) rather than per peer: the same exchange then yields the
            # all-to-all splits and the owner-side per-table counts from one host transfer.
            num_tables = len(self.tables)
            send_counts = torch.bincount(
                sorted_owners * num_tables + requests[:, 0], minlength=self.ep_size * num_tables
            ).to(torch.int64)
            recv_counts = torch.empty_like(send_counts)
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.count-a2a"
        nvtx_range_push(message)
        try:
            torch.distributed.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        finally:
            nvtx_range_pop(message)
        sent, received = torch.stack((send_counts, recv_counts)).tolist()
        peer_slices = [
            slice(peer * num_tables, (peer + 1) * num_tables) for peer in range(self.ep_size)
        ]
        send_splits = [sum(sent[peer]) for peer in peer_slices]
        recv_splits = [sum(received[peer]) for peer in peer_slices]
        table_counts = [
            sum(received[peer * num_tables + table] for peer in range(self.ep_size))
            for table in range(num_tables)
        ]

        received_requests = requests.new_empty((sum(recv_splits), 2))
        message = "engram.lookup.request-a2a"
        nvtx_range_push(message)
        try:
            torch.distributed.all_to_all_single(
                received_requests,
                requests.contiguous(),
                output_split_sizes=recv_splits,
                input_split_sizes=send_splits,
                group=self.ep_group,
            )
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.local-embedding"
        nvtx_range_push(message)
        try:
            owner_embeddings = self._lookup_received_requests(received_requests, table_counts)
        finally:
            nvtx_range_pop(message)
        sorted_embeddings = _DifferentiableEngramAllToAll.apply(
            self.ep_group, owner_embeddings, send_splits, recv_splits
        )
        output = torch.empty_like(sorted_embeddings)
        output.index_copy_(0, owner_order, sorted_embeddings)
        return output.view(*original_shape, self.embedding_dim)
