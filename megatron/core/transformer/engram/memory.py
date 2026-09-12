# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Replicated and row-sharded Engram embedding tables."""

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

from .config import ENGRAM_TABLE_LR_MULTIPLIER, ENGRAM_TABLE_WEIGHT_DECAY


@dataclass(frozen=True)
class EngramTableParameterMetadata:
    """Optimizer and checkpoint contract for an Engram table parameter."""

    row_parallel: bool = False
    layer_id: int | None = None
    table_sizes: tuple[int, ...] = ()
    local_bounds: tuple[tuple[int, int], ...] = ()
    local_offsets: tuple[int, ...] = ()
    table_group: torch.distributed.ProcessGroup | None = None
    tp_group: torch.distributed.ProcessGroup | None = None
    model_parallel_group: torch.distributed.ProcessGroup | None = None
    lr_multiplier: float = ENGRAM_TABLE_LR_MULTIPLIER
    weight_decay: float = ENGRAM_TABLE_WEIGHT_DECAY


def mark_engram_table_parameter(
    parameter: nn.Parameter, metadata: EngramTableParameterMetadata | None = None
) -> None:
    """Attach the single metadata object consumed by Engram optimizer paths."""
    parameter.engram_table_metadata = metadata or EngramTableParameterMetadata()
    if parameter.engram_table_metadata.row_parallel:
        parameter.skip_param_and_grad_buffer = True


def row_shard_bounds(num_rows: int, rank: int, world_size: int) -> tuple[int, int]:
    """Canonical Engram row ownership interval for one logical table."""
    if num_rows < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            "Invalid row shard arguments: "
            f"num_rows={num_rows}, rank={rank}, world_size={world_size}"
        )
    return (num_rows * rank // world_size, num_rows * (rank + 1) // world_size)


def row_shard_owner(num_rows: int, row: int, world_size: int) -> int:
    """Return the unique owner defined by :func:`row_shard_bounds`."""
    if not 0 <= row < num_rows:
        raise ValueError(f"Row {row} is outside table with {num_rows} rows")
    return ((row + 1) * world_size - 1) // num_rows


def _row_shard_owners(num_rows: Tensor, rows: Tensor, world_size: int) -> Tensor:
    """Tensor form of row_shard_owner; all routing uses this implementation."""
    return torch.div((rows + 1) * world_size - 1, num_rows, rounding_mode='floor')


@dataclass(frozen=True)
class RowShardedTableLayout:
    """Static mapping between logical/global rows and a rank-local packed parameter."""

    table_sizes: tuple[int, ...]
    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if not self.table_sizes or any(size <= 0 for size in self.table_sizes):
            raise ValueError("Engram logical table sizes must be positive")
        row_shard_bounds(0, self.rank, self.world_size)

    @property
    def global_offsets(self) -> tuple[int, ...]:
        """Prefix sums of logical table sizes used to flatten global row ids."""
        offsets = [0]
        for size in self.table_sizes[:-1]:
            offsets.append(offsets[-1] + size)
        return tuple(offsets)

    @property
    def local_bounds(self) -> tuple[tuple[int, int], ...]:
        """Owner-local [start, end) interval for each logical table."""
        return tuple(
            row_shard_bounds(size, self.rank, self.world_size) for size in self.table_sizes
        )

    @property
    def local_offsets(self) -> tuple[int, ...]:
        """Prefix sums of owner-local shard lengths in the packed parameter."""
        offsets = [0]
        for start, end in self.local_bounds[:-1]:
            offsets.append(offsets[-1] + end - start)
        return tuple(offsets)

    @property
    def local_num_rows(self) -> int:
        """Total number of packed rows owned by this rank."""
        return sum(end - start for start, end in self.local_bounds)

    def owners(self, global_rows: Tensor) -> Tensor:
        """Return the table-group owner rank for each flattened global row id."""
        ends = global_rows.new_tensor(
            [offset + size for offset, size in zip(self.global_offsets, self.table_sizes)]
        )
        table_indices = torch.bucketize(global_rows, ends, right=True)
        offsets = global_rows.new_tensor(self.global_offsets)[table_indices]
        sizes = global_rows.new_tensor(self.table_sizes)[table_indices]
        return _row_shard_owners(sizes, global_rows - offsets, self.world_size)

    def to_local_rows(self, global_rows: Tensor) -> Tensor:
        """Map flattened global row ids onto this rank's packed local rows."""
        ends = global_rows.new_tensor(
            [offset + size for offset, size in zip(self.global_offsets, self.table_sizes)]
        )
        table_indices = torch.bucketize(global_rows, ends, right=True)
        table_offsets = global_rows.new_tensor(self.global_offsets)[table_indices]
        sizes = global_rows.new_tensor(self.table_sizes)[table_indices]
        rows = global_rows - table_offsets
        owners = _row_shard_owners(sizes, rows, self.world_size)
        if torch.any(owners != self.rank):
            raise RuntimeError("Received an Engram row on a non-owner table rank")
        starts = torch.div(sizes * self.rank, self.world_size, rounding_mode='floor')
        local_offsets = global_rows.new_tensor(self.local_offsets)[table_indices]
        return local_offsets + rows - starts


class MultiHeadEmbedding(nn.Module):
    """Official layer-local concatenation of independently sized hash tables."""

    def __init__(
        self,
        list_of_N: Sequence[int],
        D: int,
        tensor_parallel_size: int = 1,
        *,
        init_method: Callable | None = None,
        params_dtype: torch.dtype = torch.float32,
        use_cpu_initialization: bool = True,
        perform_initialization: bool = True,
        tp_group: torch.distributed.ProcessGroup | None = None,
        dp_cp_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.tensor_parallel_size = tensor_parallel_size
        self.tp_group = tp_group
        self.dp_cp_group = dp_cp_group
        self.local_embedding_dim = D // tensor_parallel_size
        offsets = [0]
        for size in list_of_N[:-1]:
            offsets.append(offsets[-1] + size)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        num_embeddings = sum(list_of_N)
        device = None if use_cpu_initialization else torch.cuda.current_device()
        self.embedding = nn.Embedding(
            num_embeddings, self.local_embedding_dim, device=device, dtype=params_dtype
        )
        mark_engram_table_parameter(self.embedding.weight)
        if init_method is not None and perform_initialization:
            if tensor_parallel_size == 1:
                init_method(self.embedding.weight)
            elif use_cpu_initialization:
                _initialize_affine_weight_cpu(
                    self.embedding.weight,
                    num_embeddings,
                    D,
                    self.local_embedding_dim,
                    1,
                    init_method,
                    params_dtype=params_dtype,
                    rank=(
                        torch.distributed.get_rank(group=tp_group)
                        if tp_group is not None
                        else parallel_state.get_tensor_model_parallel_rank()
                    ),
                    world_size=(tensor_parallel_size),
                )
            else:
                _initialize_affine_weight_gpu(self.embedding.weight, init_method, partition_dim=1)
        elif tensor_parallel_size > 1:
            tensor_parallel.set_tensor_model_parallel_attributes(self.embedding.weight, True, 1, 1)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Look up replicated local tables and gather the full embedding width."""
        output = self.embedding(input_ids + self.offsets)
        if self.tensor_parallel_size > 1:
            output = tensor_parallel.gather_from_tensor_model_parallel_region(
                output, group=self.tp_group
            )
        return output

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> dict[str, Any]:
        """Shard replicated table offsets with the standard checkpoint helper."""
        state_dict = self.state_dict(prefix='', keep_vars=True)
        result = make_sharded_tensors_for_checkpoint(
            {'offsets': state_dict['offsets']},
            prefix,
            sharded_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )
        weight_key = f'{prefix}embedding.weight'
        result[weight_key] = make_tp_sharded_tensor_for_checkpoint(
            state_dict['embedding.weight'],
            weight_key,
            tp_axis=1,
            prepend_offsets=sharded_offsets,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )
        return result


def _all_to_all_splits(
    input_splits: Tensor, group: torch.distributed.ProcessGroup | None
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Exchange all-to-all-v counts and return Python split tuples."""
    world_size = input_splits.numel()
    if world_size == 1:
        value = int(input_splits.item())
        return (value,), (value,)
    output_splits = torch.empty_like(input_splits)
    torch.distributed.all_to_all_single(output_splits, input_splits, group=group)
    return tuple(input_splits.cpu().tolist()), tuple(output_splits.cpu().tolist())


class _RowShardedEmbedding(torch.autograd.Function):
    """All-to-all row lookup whose weight gradient contains touched owner rows only."""

    @staticmethod
    def forward(
        ctx: Any,
        weight: Tensor,
        global_rows: Tensor,
        layout: RowShardedTableLayout,
        group: torch.distributed.ProcessGroup,
        grad_scale: float,
    ) -> Tensor:
        """All-to-all fetch owner rows and return values aligned to the request order."""
        flat_rows = global_rows.reshape(-1)
        unique_rows, sender_inverse = torch.unique(flat_rows, sorted=True, return_inverse=True)
        owners = layout.owners(unique_rows)
        send_order = torch.argsort(owners, stable=True)
        send_rows = unique_rows[send_order].contiguous()
        input_counts = torch.bincount(owners, minlength=layout.world_size).to(torch.int64)
        input_splits, output_splits = _all_to_all_splits(input_counts, group)

        if layout.world_size == 1:
            received_rows = send_rows
        else:
            received_rows = tensor_parallel.mappings.all_to_all(
                group, send_rows, output_splits, input_splits
            )
        received_local_rows = layout.to_local_rows(received_rows)
        owner_unique_rows, owner_inverse = torch.unique(
            received_local_rows, sorted=True, return_inverse=True
        )
        owner_values = weight[owner_unique_rows][owner_inverse]
        if layout.world_size == 1:
            returned_values = owner_values
        else:
            returned_values = tensor_parallel.mappings.all_to_all(
                group, owner_values, input_splits, output_splits
            )
        unique_values = weight.new_empty((unique_rows.numel(), weight.shape[1]))
        unique_values[send_order] = returned_values
        output = unique_values[sender_inverse]

        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(sender_inverse, send_order, owner_inverse, owner_unique_rows)
        return output.view(*global_rows.shape, weight.shape[1])

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        """Route output grads back to owner rows as a sparse COO weight gradient."""
        sender_inverse, send_order, owner_inverse, owner_unique_rows = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).float()
        sender_unique_grads = grad_output.new_zeros((send_order.numel(), grad_output.shape[-1]))
        sender_unique_grads.index_add_(0, sender_inverse, grad_output)
        send_grads = sender_unique_grads[send_order].contiguous()

        if len(ctx.input_splits) == 1:
            received_grads = send_grads
        else:
            received_grads = tensor_parallel.mappings.all_to_all(
                ctx.group, send_grads, ctx.output_splits, ctx.input_splits
            )
        owner_grads = received_grads.new_zeros(
            (owner_unique_rows.numel(), received_grads.shape[-1])
        )
        owner_grads.index_add_(0, owner_inverse, received_grads)
        owner_grads.mul_(ctx.grad_scale)
        owner_grads = owner_grads.to(ctx.weight_dtype)
        sparse_grad = torch.sparse_coo_tensor(
            owner_unique_rows.unsqueeze(0),
            owner_grads,
            size=ctx.weight_shape,
            dtype=owner_grads.dtype,
            device=owner_grads.device,
            check_invariants=False,
        ).coalesce()
        return sparse_grad, None, None, None, None


class RowShardedMultiHeadEmbedding(nn.Module):
    """One packed owner-local parameter for all logical tables in an Engram layer."""

    def __init__(
        self,
        list_of_N: Sequence[int],
        D: int,
        *,
        layer_id: int,
        seed: int = 0,
        init_method: Callable | None = None,
        params_dtype: torch.dtype = torch.float32,
        use_cpu_initialization: bool = True,
        perform_initialization: bool = True,
        calculate_per_token_loss: bool = False,
        table_group: torch.distributed.ProcessGroup | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
        dp_group: torch.distributed.ProcessGroup | None = None,
        dp_cp_group: torch.distributed.ProcessGroup | None = None,
        model_parallel_group: torch.distributed.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        if table_group is None and not parallel_state.model_parallel_is_initialized():
            raise RuntimeError("Engram row_a2a requires initialized Megatron parallel groups")
        # Compatibility for standalone tables constructed without explicit groups.
        # Model-owned tables receive every group from the Hybrid provider.
        self.table_group = (
            parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
            if table_group is None
            else table_group
        )
        self.tp_group = (
            parallel_state.get_tensor_model_parallel_group() if tp_group is None else tp_group
        )
        self.dp_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=False)
            if dp_group is None
            else dp_group
        )
        self.dp_cp_group = (
            parallel_state.get_data_parallel_group(with_context_parallel=True)
            if dp_cp_group is None
            else dp_cp_group
        )
        self.model_parallel_group = (
            parallel_state.get_model_parallel_group()
            if model_parallel_group is None
            else model_parallel_group
        )
        rank = torch.distributed.get_rank(group=self.table_group)
        world_size = torch.distributed.get_world_size(group=self.table_group)
        self.layout = RowShardedTableLayout(tuple(list_of_N), rank, world_size)
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.layer_id = layer_id
        self.calculate_per_token_loss = calculate_per_token_loss
        self.register_buffer(
            "offsets", torch.tensor(self.layout.global_offsets, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "local_offsets",
            torch.tensor(self.layout.local_offsets, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "local_starts",
            torch.tensor([start for start, _ in self.layout.local_bounds], dtype=torch.long),
            persistent=False,
        )
        device = None if use_cpu_initialization else torch.cuda.current_device()
        fork_devices = [] if use_cpu_initialization else [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            initialization_seed = int(seed + 10007 * layer_id + rank)
            torch.random.default_generator.manual_seed(initialization_seed)
            if not use_cpu_initialization:
                torch.cuda.manual_seed(initialization_seed)
            self.embedding = nn.Embedding(
                self.layout.local_num_rows, D, sparse=True, device=device, dtype=params_dtype
            )
            if init_method is not None and perform_initialization:
                init_method(self.embedding.weight)
        weight = self.embedding.weight
        mark_engram_table_parameter(
            weight,
            EngramTableParameterMetadata(
                row_parallel=True,
                layer_id=layer_id,
                table_sizes=self.layout.table_sizes,
                local_bounds=self.layout.local_bounds,
                local_offsets=self.layout.local_offsets,
                table_group=self.table_group,
                tp_group=self.tp_group,
                model_parallel_group=self.model_parallel_group,
            ),
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Look up row-sharded tables and route gradients to their owning ranks."""
        global_rows = input_ids + self.offsets
        dp_cp_size = torch.distributed.get_world_size(group=self.dp_cp_group)
        tp_size = torch.distributed.get_world_size(group=self.tp_group)
        grad_scale = 1.0 / tp_size
        if not self.calculate_per_token_loss:
            grad_scale /= dp_cp_size
        return _RowShardedEmbedding.apply(
            self.embedding.weight, global_rows, self.layout, self.table_group, grad_scale
        )

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: dict | None = None
    ) -> dict[str, Any]:
        """Checkpoint logical table weights and table offsets."""
        if sharded_offsets:
            raise ValueError("row_a2a table weights do not accept parent sharded offsets")
        state_dict = self.state_dict(prefix='', keep_vars=True)
        result = make_sharded_tensors_for_checkpoint(
            {'offsets': state_dict['offsets']},
            prefix,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )
        key = f'{prefix}embedding.weight'
        result[key] = self._row_checkpoint_factory(key, state_dict['embedding.weight'])
        return result

    def _row_checkpoint_factory(self, key: str, tensor: Tensor) -> ShardedTensorFactory:
        """Shard table weights using their logical row coordinates."""
        layout = self.layout

        def build_fn(factory_key, data, replica_id, flattened_range):
            if flattened_range is not None:
                raise ValueError("Nested flattened row_a2a factories are unsupported")
            tables = {}
            for table_index, ((start, end), local_offset, size) in enumerate(
                zip(layout.local_bounds, layout.local_offsets, layout.table_sizes)
            ):
                local_rows = end - start
                # Empty shards can share a global offset with the following
                # non-empty shard. Registering both makes torch DCP deduplicate
                # the real chunk, leaving holes when table_size < world_size.
                if local_rows == 0:
                    continue
                table_data = data.narrow(0, local_offset, local_rows)
                tables[str(table_index)] = ShardedTensor(
                    f'{factory_key}.table_{table_index}',
                    table_data,
                    table_data.dtype,
                    tuple(table_data.shape),
                    (size, *data.shape[1:]),
                    (start, *(0 for _ in data.shape[1:])),
                    axis_fragmentations=None,
                    replica_id=replica_id,
                )
            return tables

        def merge_fn(sub_state_dict):
            values = [
                sub_state_dict[str(index)]
                for index in range(len(layout.table_sizes))
                if str(index) in sub_state_dict
            ]
            if not values:
                return tensor.new_empty((0, *tensor.shape[1:]))
            return torch.cat([value.view(-1, *tensor.shape[1:]) for value in values], dim=0)

        return ShardedTensorFactory(key, tensor, build_fn, merge_fn, replica_id=0)
