# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native differentiable EP-sharded embedding lookup for Engram."""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_rank, get_pg_size, nvtx_range_pop, nvtx_range_push


def _check_bucket_capacity(peak_requests: Tensor, capacity: int, num_tokens: int, ep_size: int):
    """Fail when a fixed-capacity bucket received more requests than it holds.

    Eagerly (graph warmup) this reads the peak on the host and raises with the factor that
    would have fit. Under CUDA graph capture the host cannot read it, so a device-side assert is
    recorded instead and every replay checks it; an overflowing replay aborts the process.
    """
    message = (
        "Engram fixed-capacity EP lookup overflowed a bucket; raise "
        f"--engram-ep-capacity-factor (no overflow is possible at >= the EP size, {ep_size})."
    )
    if torch.cuda.is_current_stream_capturing():
        torch._assert_async(peak_requests <= capacity, message)
        return
    peak = int(peak_requests.item())
    if peak > capacity:
        needed = peak * ep_size / num_tokens
        raise ValueError(
            f"{message} A bucket received {peak} requests for {capacity} slots; a factor of "
            f"at least {needed:.2f} would have fit this step."
        )


class _DifferentiableEngramAllToAll(torch.autograd.Function):
    """All-to-all with Engram-specific forward/backward NVTX ranges.

    ``None`` splits mean equal splits (the fixed-capacity exchange).
    """

    @staticmethod
    def _exchange(
        group, input_: Tensor, output_splits: Optional[list[int]], input_splits: Optional[list[int]]
    ):
        if output_splits is None:
            output = torch.empty_like(input_)
        else:
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
        """All-to-all ``input_`` with the given splits."""
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
        """Return the gradient through the reverse all-to-all."""
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
        """Gather ``weight`` rows at ``row_ids``."""
        ctx.save_for_backward(row_ids)
        ctx.weight_shape = weight.shape
        ctx.weight_dtype = weight.dtype
        return weight.index_select(0, row_ids)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        """Accumulate row gradients deterministically (sorted, segmented)."""
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
        capacity_factor: Optional[float] = None,
    ) -> None:
        super().__init__(config=config)
        if capacity_factor is not None and capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {capacity_factor}.")
        # Set: exchange through fixed-size (owner, table) buckets with no host synchronization,
        # as a captured CUDA graph requires. None: exact, variable-split exchange.
        self.capacity_factor = capacity_factor
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

    def _table_ids(self, shape: torch.Size, device) -> Tensor:
        """Table index of every flattened request of a ``[..., num_tables]`` hash tensor."""
        return (
            torch.arange(len(self.tables), device=device, dtype=torch.int64)
            .view(*([1] * (len(shape) - 1)), -1)
            .expand(shape)
            .reshape(-1)
        )

    def _owner_map(self, rows: Tensor, table_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return each request's owning EP rank and its row index within the owner's shard."""
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
        return owners, rows - owner_starts

    def bucket_capacity(self, num_tokens: int) -> int:
        """Slots per (owner, table) bucket for a lookup over ``num_tokens`` tokens."""
        return max(1, math.ceil(self.capacity_factor * num_tokens / self.ep_size))

    def _forward_fixed_capacity(self, hash_ids: Tensor, valid: Tensor | None = None) -> Tensor:
        """EP lookup through fixed-size buckets; every shape follows from ``hash_ids.shape``.

        Each (owner, table) bucket holds ``bucket_capacity`` requests. Requests travel in a
        ``[ep, num_tables, capacity]`` layout, so the table of a slot is implicit in its position
        and the owner gathers each table's slots with one embedding call. Unused slots look up
        row 0 and are never read back, so they contribute no gradient. A bucket that overflows
        is an error (see ``_check_bucket_capacity``): the capacity factor is too small. Requests
        of tokens where ``valid`` is False take no slot; their returned rows are arbitrary and
        must be discarded by the caller.
        """
        original_shape = hash_ids.shape
        num_tables = len(self.tables)
        rows = hash_ids.reshape(-1).to(torch.int64)
        num_requests = rows.numel()
        capacity = self.bucket_capacity(num_requests // num_tables)
        num_slots = self.ep_size * num_tables * capacity
        device = hash_ids.device

        message = "engram.lookup.owner-map"
        nvtx_range_push(message)
        try:
            table_ids = self._table_ids(original_shape, device)
            owners, local_rows = self._owner_map(rows, table_ids)
            num_buckets = self.ep_size * num_tables
            buckets = owners * num_tables + table_ids
            if valid is not None:
                # Invalid requests go to one extra bucket that is never sent.
                valid = valid.unsqueeze(-1).expand(original_shape).reshape(-1)
                buckets = torch.where(valid, buckets, num_buckets)
            # Rank of each request within its bucket, from a stable sort and per-bucket starts.
            order = torch.argsort(buckets, stable=True)
            counts = torch.zeros(num_buckets + 1, dtype=torch.int64, device=device)
            counts.scatter_add_(0, buckets, torch.ones_like(buckets))
            starts = torch.cumsum(counts, 0) - counts
            sorted_positions = torch.arange(
                num_requests, device=device, dtype=torch.int64
            ) - starts.index_select(0, buckets.index_select(0, order))
            positions = torch.empty_like(sorted_positions).scatter_(0, order, sorted_positions)
            fits = positions < capacity
            if valid is not None:
                fits = fits & valid
                buckets = torch.where(valid, buckets, 0)
            _check_bucket_capacity(
                counts[:num_buckets].max(), capacity, num_requests // num_tables, self.ep_size
            )
            slots = buckets * capacity + positions.clamp_max(capacity - 1)
            # Invalid (and, before the capacity check fails, excess) requests write to one
            # spare slot past the end that is never sent.
            send = torch.zeros(num_slots + 1, dtype=torch.int64, device=device)
            send.scatter_(0, torch.where(fits, slots, num_slots), local_rows)
            send = send[:num_slots]
            received = torch.empty_like(send)
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.request-a2a"
        nvtx_range_push(message)
        try:
            torch.distributed.all_to_all_single(received, send, group=self.ep_group)
        finally:
            nvtx_range_pop(message)

        message = "engram.lookup.local-embedding"
        nvtx_range_push(message)
        try:
            received = received.view(self.ep_size, num_tables, capacity)
            owner_embeddings = torch.stack(
                [
                    table(received[:, table_id].reshape(-1)).view(
                        self.ep_size, capacity, self.embedding_dim
                    )
                    for table_id, table in enumerate(self.tables)
                ],
                dim=1,
            ).view(num_slots, self.embedding_dim)
        finally:
            nvtx_range_pop(message)
        responses = _DifferentiableEngramAllToAll.apply(self.ep_group, owner_embeddings, None, None)
        return responses.index_select(0, slots).view(*original_shape, self.embedding_dim)

    def forward(self, hash_ids: Tensor, valid: Tensor | None = None) -> Tensor:
        """Route global table rows to owners and restore token/head ordering.

        Args:
            hash_ids: Per-table row IDs with shape ``[..., num_tables]``.
            valid: Optional bool mask of shape ``hash_ids.shape[:-1]``. The fixed-capacity
                exchange gives tokens where it is False no bucket slot; their returned rows
                are arbitrary and must be discarded by the caller.

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

        if self.capacity_factor is not None:
            return self._forward_fixed_capacity(hash_ids, valid)

        rows = hash_ids.reshape(-1).to(torch.int64)
        table_ids = self._table_ids(original_shape, hash_ids.device)

        message = "engram.lookup.owner-map"
        nvtx_range_push(message)
        try:
            owners, local_rows = self._owner_map(rows, table_ids)
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
