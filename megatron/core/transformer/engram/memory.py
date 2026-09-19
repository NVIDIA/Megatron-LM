# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tensor-parallel and row-sharded hash embedding tables."""

from dataclasses import dataclass
from itertools import accumulate
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from megatron.core import parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
)
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint

from .config import EngramConfig, _parallel_world_size


@dataclass(frozen=True)
class EngramTableParameterMetadata:
    """Optimizer and checkpoint contract for an Engram table parameter."""

    row_parallel: bool = False
    table_group: object = None
    tp_group: object = None
    replica_group: object = None


def mark_engram_table_parameter(
    parameter: nn.Parameter, metadata: Optional[EngramTableParameterMetadata] = None
) -> None:
    """Attach the single metadata object consumed by Engram optimizer paths."""
    parameter.engram_table_metadata = metadata or EngramTableParameterMetadata()
    parameter.use_muon = False
    if parameter.engram_table_metadata.row_parallel:
        parameter.optimizer_sharding_group = parameter.engram_table_metadata.replica_group


def build_multi_head_embedding(
    config: EngramConfig,
    layer_id: int,
    table_sizes: Sequence[int],
    pg_collection=None,
    parallel_groups=None,
) -> nn.Module:
    """Construct the lookup backend before attaching it to the fusion module."""
    tp_group = getattr(pg_collection, 'tp', None)
    common = dict(
        init_method=config.init_method,
        params_dtype=config.params_dtype,
        use_cpu_initialization=config.use_cpu_initialization,
        perform_initialization=config.perform_initialization,
        tp_group=tp_group,
        dp_cp_group=getattr(pg_collection, 'dp_cp', None),
    )
    if config.table_backend == 'row_a2a':
        return RowShardedMultiHeadEmbedding(
            table_sizes,
            config.embedding_dim,
            layer_id=layer_id,
            seed=config.seed,
            calculate_per_token_loss=config.calculate_per_token_loss,
            parallel_groups=parallel_groups,
            dp_group=getattr(pg_collection, 'dp', None),
            **common,
        )
    tp_size = (
        torch.distributed.get_world_size(group=tp_group)
        if tp_group is not None
        else _parallel_world_size(parallel_state.get_tensor_model_parallel_world_size)
    )
    return MultiHeadEmbedding(
        table_sizes, config.embedding_dim, tensor_parallel_size=tp_size, **common
    )


def row_shard_bounds(num_rows: int, rank: int, world_size: int) -> Tuple[int, int]:
    """Canonical Engram row ownership interval for one logical table."""
    if num_rows < 0 or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(
            f"Invalid row shard: rows={num_rows}, rank={rank}, world_size={world_size}"
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

    table_sizes: Tuple[int, ...]
    rank: int
    world_size: int

    def __post_init__(self):
        if not self.table_sizes or any(size <= 0 for size in self.table_sizes):
            raise ValueError("Engram logical table sizes must be positive")
        row_shard_bounds(0, self.rank, self.world_size)

    @property
    def global_offsets(self) -> Tuple[int, ...]:
        """Return the starting global row of each logical table."""
        return tuple(accumulate(self.table_sizes[:-1], initial=0))

    @property
    def local_bounds(self) -> Tuple[Tuple[int, int], ...]:
        """Return this rank's half-open row interval in each table."""
        return tuple(
            row_shard_bounds(size, self.rank, self.world_size) for size in self.table_sizes
        )

    @property
    def local_offsets(self) -> Tuple[int, ...]:
        """Return offsets of local table fragments in packed storage."""
        return tuple(accumulate((end - start for start, end in self.local_bounds[:-1]), initial=0))

    @property
    def local_num_rows(self) -> int:
        """Return the number of rows owned by this rank."""
        return sum(end - start for start, end in self.local_bounds)

    def owners(self, global_rows: Tensor) -> Tensor:
        """Map packed global row IDs to table-group owner ranks."""
        ends = global_rows.new_tensor(
            [offset + size for offset, size in zip(self.global_offsets, self.table_sizes)]
        )
        table_indices = torch.bucketize(global_rows, ends, right=True)
        offsets = global_rows.new_tensor(self.global_offsets)[table_indices]
        sizes = global_rows.new_tensor(self.table_sizes)[table_indices]
        return _row_shard_owners(sizes, global_rows - offsets, self.world_size)

    def to_local_rows(self, global_rows: Tensor) -> Tensor:
        """Validate ownership and map global rows into local packed storage."""
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
    """Concatenated hash tables with optional tensor-parallel embedding dimensions."""

    def __init__(
        self,
        list_of_N: Sequence[int],
        D: int,
        tensor_parallel_size: int = 1,
        *,
        init_method: Optional[Callable] = None,
        params_dtype: torch.dtype = torch.float32,
        use_cpu_initialization: bool = True,
        perform_initialization: bool = True,
        tp_group=None,
        dp_cp_group=None,
    ):
        super().__init__()
        self.embedding_dim = D
        self.tensor_parallel_size = tensor_parallel_size
        self.tp_group = tp_group
        self.dp_cp_group = dp_cp_group
        self.local_embedding_dim = D // tensor_parallel_size
        offsets = tuple(accumulate(list_of_N[:-1], initial=0))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        num_embeddings = sum(list_of_N)
        device = None if use_cpu_initialization else torch.cuda.current_device()
        self.embedding = nn.Embedding(
            num_embeddings,
            self.local_embedding_dim,
            _weight=torch.empty(
                num_embeddings, self.local_embedding_dim, device=device, dtype=params_dtype
            ),
        )
        mark_engram_table_parameter(self.embedding.weight)
        if perform_initialization:
            init_method = nn.init.normal_ if init_method is None else init_method
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
        """Look up local head columns and gather the complete embedding width."""
        output = self.embedding(input_ids + self.offsets)
        if self.tensor_parallel_size > 1:
            output = tensor_parallel.gather_from_tensor_model_parallel_region(
                output, group=self.tp_group
            )
        return output

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Describe table columns through the native TP checkpoint helper."""
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


def _all_to_all_splits(input_splits: Tensor, group) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Exchange all-to-all-v counts and return Python split tuples."""
    output_splits = tensor_parallel.mappings.all_to_all(group, input_splits)
    return tuple(input_splits.cpu().tolist()), tuple(output_splits.cpu().tolist())


class _RowShardedEmbedding(torch.autograd.Function):
    """All-to-all row lookup accumulating owner gradients into FP32 main_grad."""

    @staticmethod
    def forward(ctx, weight: Tensor, global_rows: Tensor, layout, group, grad_scale: float):
        """Fetch unique rows from their owners and restore the requested token layout."""
        flat_rows = global_rows.reshape(-1)
        unique_rows, sender_inverse = torch.unique(flat_rows, sorted=True, return_inverse=True)
        owners = layout.owners(unique_rows)
        send_order = torch.argsort(owners, stable=True)
        send_rows = unique_rows[send_order].contiguous()
        input_counts = torch.bincount(owners, minlength=layout.world_size).to(torch.int64)
        input_splits, output_splits = _all_to_all_splits(input_counts, group)

        received_rows = tensor_parallel.mappings.all_to_all(
            group, send_rows, output_splits, input_splits
        )
        received_local_rows = layout.to_local_rows(received_rows)
        owner_unique_rows, owner_inverse = torch.unique(
            received_local_rows, sorted=True, return_inverse=True
        )
        owner_values = weight[owner_unique_rows][owner_inverse]

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
        ctx.main_grad = getattr(weight, "main_grad", None)
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(sender_inverse, send_order, owner_inverse, owner_unique_rows)
        return output.view(*global_rows.shape, weight.shape[1])

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        """Return lookup gradients to row owners and accumulate into FP32 buffers."""
        sender_inverse, send_order, owner_inverse, owner_unique_rows = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]).float()
        sender_unique_grads = grad_output.new_zeros((send_order.numel(), grad_output.shape[-1]))
        sender_unique_grads.index_add_(0, sender_inverse, grad_output)
        send_grads = sender_unique_grads[send_order].contiguous()

        received_grads = tensor_parallel.mappings.all_to_all(
            ctx.group, send_grads, ctx.output_splits, ctx.input_splits
        )
        owner_grads = received_grads.new_zeros(
            (owner_unique_rows.numel(), received_grads.shape[-1])
        )
        owner_grads.index_add_(0, owner_inverse, received_grads)
        owner_grads.mul_(ctx.grad_scale)
        main_grad = ctx.main_grad
        if main_grad is None:
            raise RuntimeError("Engram row_a2a backward requires a preallocated FP32 main_grad")
        if (
            main_grad.dtype != torch.float32
            or tuple(main_grad.shape) != ctx.weight_shape
            or main_grad.device != owner_grads.device
        ):
            raise RuntimeError("Engram main_grad must match the local table shape/device in FP32")
        main_grad.index_add_(0, owner_unique_rows, owner_grads)
        return None, None, None, None, None


class RowShardedMultiHeadEmbedding(nn.Module):
    """One packed owner-local parameter for all logical tables in an Engram layer."""

    def __init__(
        self,
        list_of_N: Sequence[int],
        D: int,
        *,
        layer_id: int,
        seed: int = 0,
        init_method: Optional[Callable] = None,
        params_dtype: torch.dtype = torch.float32,
        use_cpu_initialization: bool = True,
        perform_initialization: bool = True,
        calculate_per_token_loss: bool = False,
        table_group=None,
        tp_group=None,
        dp_group=None,
        dp_cp_group=None,
        parallel_groups=None,
    ):
        super().__init__()
        self.parallel_groups = parallel_groups
        if parallel_groups is not None:
            table_group = parallel_groups.table_group
        self.replica_rank = 0 if parallel_groups is None else parallel_groups.replica_rank
        if table_group is None and not parallel_state.model_parallel_is_initialized():
            raise RuntimeError("Engram row_a2a requires initialized Megatron parallel groups")
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
        rank = torch.distributed.get_rank(group=self.table_group)
        world_size = torch.distributed.get_world_size(group=self.table_group)
        self.layout = RowShardedTableLayout(tuple(list_of_N), rank, world_size)
        self.embedding_dim = D
        self.calculate_per_token_loss = calculate_per_token_loss
        self.register_buffer(
            "offsets", torch.tensor(self.layout.global_offsets, dtype=torch.long), persistent=True
        )
        device = None if use_cpu_initialization else torch.cuda.current_device()
        fork_devices = [] if use_cpu_initialization else [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            init_seed = int(seed + 10007 * layer_id + rank)
            # Seed only the generators whose states this context restores.
            torch.random.default_generator.manual_seed(init_seed)
            if not use_cpu_initialization:
                torch.cuda.manual_seed(init_seed)
            self.embedding = nn.Embedding(
                self.layout.local_num_rows,
                D,
                _weight=torch.empty(
                    self.layout.local_num_rows, D, device=device, dtype=params_dtype
                ),
            )
            if perform_initialization:
                initializer = nn.init.normal_ if init_method is None else init_method
                initializer(self.embedding.weight)
        weight = self.embedding.weight
        mark_engram_table_parameter(
            weight,
            EngramTableParameterMetadata(
                row_parallel=True,
                table_group=self.table_group,
                tp_group=self.tp_group,
                replica_group=None if parallel_groups is None else parallel_groups.replica_group,
            ),
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        """Route token rows to their owners with native TP/DP/CP gradient scaling."""
        global_rows = input_ids + self.offsets
        dp_size = torch.distributed.get_world_size(group=self.dp_cp_group)
        tp_size = torch.distributed.get_world_size(group=self.tp_group)
        grad_scale = 1.0 / tp_size
        if not self.calculate_per_token_loss:
            grad_scale /= dp_size
        return _RowShardedEmbedding.apply(
            self.embedding.weight, global_rows, self.layout, self.table_group, grad_scale
        )

    def _row_sharded_factory(
        self, key: str, data: Tensor, *, replica_id=None
    ) -> ShardedTensorFactory:
        """Store each logical table flat so native optimizer shards may split a row.

        Runtime weights remain [rows, width]. Checkpoint offsets are row * width
        plus column; each table contributes at most one contiguous local interval.
        The loader writes into its own model/master/moment storage without aliasing
        the original model tensor or concatenating a complete table.
        """
        layout = self.layout
        if data.ndim != 2 or data.shape[0] != layout.local_num_rows:
            raise ValueError('Engram checkpoint weights must match the local row layout')
        width = data.shape[1]
        packed_numel = data.numel()

        def build_fn(factory_key, tensor, replica_id, flattened_range):
            interval = flattened_range or slice(0, packed_numel)
            start = 0 if interval.start is None else interval.start
            stop = packed_numel if interval.stop is None else interval.stop
            if (
                interval.step not in (None, 1)
                or not 0 <= start <= stop <= packed_numel
                or tensor.numel() != stop - start
                or not tensor.is_contiguous()
            ):
                raise ValueError('Engram checkpoint data must match a contiguous flattened range')
            flat = tensor.view(-1)
            result = {'_restore_target': LocalNonpersistentObject(tensor)}
            slices = []
            for index, ((first, last), offset, size) in enumerate(
                zip(layout.local_bounds, layout.local_offsets, layout.table_sizes)
            ):
                table_start = offset * width
                left, right = max(start, table_start), min(
                    stop, table_start + (last - first) * width
                )
                if left >= right:
                    continue
                length, destination = right - left, left - start
                entry = str(index)
                result[entry] = ShardedTensor(
                    f'{factory_key}.table_{index}',
                    flat.narrow(0, destination, length),
                    tensor.dtype,
                    (length,),
                    (size * width,),
                    (first * width + left - table_start,),
                    axis_fragmentations=None,
                    replica_id=replica_id,
                )
                slices.append((entry, destination, length, (length,)))
            result['_restore_slices'] = LocalNonpersistentObject(tuple(slices))
            return result

        def merge_fn(state):
            if '_restore_target' not in state or '_restore_slices' not in state:
                raise ValueError('Engram checkpoint restore target and slice plan are required')
            target, plan = state['_restore_target'], state['_restore_slices']
            if set(state) != {'_restore_target', '_restore_slices'} | {x[0] for x in plan}:
                raise ValueError('Engram checkpoint entries do not match the required state')
            if not isinstance(target, Tensor):
                raise TypeError('Engram checkpoint restore target must be a tensor')
            # Validate all entries before mutating any model or optimizer storage.
            for entry, offset, length, shape in plan:
                value = state[entry]
                if not isinstance(value, Tensor):
                    raise TypeError('Engram checkpoint entry must be a tensor')
                if value.shape != shape or value.dtype != target.dtype:
                    raise ValueError('Engram checkpoint restore shape/dtype mismatch')
            with torch.no_grad():
                for entry, offset, length, _ in plan:
                    target.view(-1).narrow(0, offset, length).copy_(state[entry])
            return target

        if replica_id is None:
            replica_id = (0, 0, self.replica_rank)
        return ShardedTensorFactory(key, data, build_fn, merge_fn, replica_id=replica_id)

    def sharded_state_dict(self, prefix='', sharded_offsets=(), metadata=None):
        """Describe replicated offsets and logical table intervals for native DCP."""
        if sharded_offsets:
            raise ValueError("row_a2a table weights do not accept parent sharded offsets")
        state_dict = self.state_dict(prefix='', keep_vars=True)
        result = make_sharded_tensors_for_checkpoint(
            {'offsets': state_dict['offsets']},
            prefix,
            tp_group=self.tp_group,
            dp_cp_group=self.dp_cp_group,
        )
        result[f'{prefix}embedding.weight'] = self._row_sharded_factory(
            f'{prefix}embedding.weight', state_dict['embedding.weight']
        )
        return result
