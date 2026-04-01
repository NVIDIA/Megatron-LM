import math
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch

from .allocator import TemporaryBucketAllocator


class BufferIndex:
    """Describes how params are laid out in a flat buffer, including global layout
    and per-rank shard information.

    Each DataParallelBuffer owns its own independent BufferIndex instance.
    """

    ItemIndex = namedtuple(
        "ItemIndex", ["global_data_index", "size", "item_id", "shape"]
    )
    BucketMeta = namedtuple(
        "BucketMeta", ["global_data_index", "size", "items"]
    )
    ShardMeta = namedtuple(
        "ShardMeta",
        ["global_data_index", "local_data_index", "bucket_data_index", "size"],
    )

    def __init__(
        self,
        param_shapes: List[torch.Size],
        dp_rank: int,
        dp_world_size: int,
        is_distributed: bool,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
        param_group_id: int = 0,
    ):
        self.param_group_id = param_group_id
        self.is_distributed = is_distributed
        self.item_index_map, self.bucket_meta = self._build_layout(
            param_shapes, dp_world_size, chunk_size_factor, sharding_strategy,
        )
        self.shard_meta = self._build_shard_meta(
            self.bucket_meta, is_distributed, dp_world_size, dp_rank,
        )

    # ------------------------------------------------------------------ #
    #  Layout construction (global, rank-independent)
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_layout(
        cls,
        param_shapes: List[torch.Size],
        dp_world_size: int,
        chunk_size_factor: int,
        sharding_strategy: str,
    ) -> Tuple[Dict[int, "BufferIndex.ItemIndex"], "BufferIndex.BucketMeta"]:

        def _pad(n: int, divisor: int) -> int:
            return int(math.ceil(n / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:
            if sharding_strategy != "no_shard":
                return _pad(data_index, dp_world_size * chunk_size_factor)
            return data_index

        def add_item(item_id, shape, offset, index_map):
            index_map[item_id] = cls.ItemIndex(
                global_data_index=offset,
                size=shape.numel(),
                item_id=item_id,
                shape=shape,
            )

        fragment_items = []
        regular_items = []
        for item_id, shape in enumerate(param_shapes):
            if shape.numel() < chunk_size_factor:
                fragment_items.append((item_id, shape))
            else:
                regular_items.append((item_id, shape))

        fragment_items = sorted(fragment_items, key=lambda x: -x[1].numel())

        item_index_map: Dict[int, cls.ItemIndex] = {}
        data_index = 0

        while len(regular_items) > 0:
            item_id, shape = regular_items.pop(0)
            add_item(item_id, shape, data_index, item_index_map)

            if shape.numel() % chunk_size_factor == 0:
                data_index += shape.numel()
                continue

            gap_offset = data_index + shape.numel()
            data_index += (shape.numel() // chunk_size_factor + 1) * chunk_size_factor
            remain = shape.numel() % chunk_size_factor
            space = chunk_size_factor - remain

            found_rhs = False
            for id_rhs in regular_items[:]:
                rhs_id, rhs = id_rhs
                if rhs.numel() % chunk_size_factor == 0:
                    continue
                rhs_remain = rhs.numel() % chunk_size_factor
                if remain + rhs_remain <= chunk_size_factor:
                    found_rhs = True
                    regular_items.remove(id_rhs)
                    break

            if found_rhs:
                add_item(rhs_id, rhs, data_index - rhs_remain, item_index_map)
                space -= rhs_remain
                data_index += rhs.numel() // chunk_size_factor * chunk_size_factor

            for id_frag in fragment_items[:]:
                frag_id, frag = id_frag
                if frag.numel() > space:
                    continue
                add_item(frag_id, frag, gap_offset, item_index_map)
                space -= frag.numel()
                gap_offset += frag.numel()
                fragment_items.remove(id_frag)

        for frag_id, frag in fragment_items:
            add_item(frag_id, frag, data_index, item_index_map)
            data_index += frag.numel()

        bucket_meta = cls.BucketMeta(
            global_data_index=0,
            size=_pad_if_needed(data_index),
            items=list(item_index_map.values()),
        )
        return item_index_map, bucket_meta

    # ------------------------------------------------------------------ #
    #  Shard meta construction (per-rank)
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_shard_meta(
        cls,
        bucket_meta: "BufferIndex.BucketMeta",
        is_distributed: bool,
        dp_world_size: int,
        dp_rank: int,
    ) -> "BufferIndex.ShardMeta":
        shard_size = bucket_meta.size // dp_world_size
        bucket_data_index = shard_size * dp_rank
        global_data_index = bucket_meta.global_data_index + bucket_data_index

        if is_distributed:
            return cls.ShardMeta(
                global_data_index=global_data_index,
                local_data_index=0,
                bucket_data_index=bucket_data_index,
                size=shard_size,
            )
        else:
            return cls.ShardMeta(
                global_data_index=global_data_index,
                local_data_index=global_data_index,
                bucket_data_index=bucket_data_index,
                size=shard_size,
            )

    # ------------------------------------------------------------------ #
    #  Internal index query methods
    # ------------------------------------------------------------------ #

    def _get_item_offset(self, item_id: int) -> Tuple[int, int]:
        """Return (global_data_index, size) for the given item."""
        idx = self.item_index_map[item_id]
        return (idx.global_data_index, idx.size)

    def _get_item_slice_in_shard(self, item_id: int) -> Tuple[int, int]:
        """Return the intersection of the item with the current shard,
        as coordinates relative to the item's start."""
        idx = self.item_index_map[item_id]
        item_start = idx.global_data_index
        item_end = item_start + idx.size
        shard_start = self.shard_meta.global_data_index
        shard_end = shard_start + self.shard_meta.size

        if item_start > shard_end or item_end < shard_start:
            return (0, 0)

        start = max(item_start, shard_start) - item_start
        end = min(item_end, shard_end) - item_start
        return (start, end)

    def _get_item_local_shard_index(self, item_id: int) -> Tuple[int, int]:
        """Return coordinates within self.data for the portion of the item
        that falls in this rank's shard."""
        slice_start, slice_end = self._get_item_slice_in_shard(item_id)
        if slice_start == slice_end:
            return (0, 0)

        idx = self.item_index_map[item_id]
        offset = (
            idx.global_data_index
            - self.shard_meta.global_data_index
            + self.shard_meta.local_data_index
        )
        return (offset + slice_start, offset + slice_end)

    def _get_item_local_index(self, item_id: int) -> Tuple[int, int]:
        """Unified entry: return coordinates within self.data for the item."""
        if not self.is_distributed:
            idx = self.item_index_map[item_id]
            return (idx.global_data_index, idx.global_data_index + idx.size)
        return self._get_item_local_shard_index(item_id)


class DataParallelBuffer:
    """Manages a flat buffer that stores (a shard of) a group of parameters.

    On construction it builds its own BufferIndex describing the layout and
    shard ownership.  External callers interact via init_data / set_item /
    get_item only.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_idx: Dict[torch.nn.Parameter, int],
        dtype: torch.dtype,
        device: torch.device,
        dp_group: torch.distributed.ProcessGroup,
        allocator: Optional[TemporaryBucketAllocator] = None,
        is_distributed: bool = False,
        param_group_id: int = 0,
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
    ):
        self.params = params
        self.param_idx = param_idx
        self.dtype = dtype
        self.device = device
        self.dp_group = dp_group
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.is_distributed = is_distributed
        self.gradient_scaling_factor = gradient_scaling_factor

        dp_rank = torch.distributed.get_rank(dp_group)
        dp_world_size = torch.distributed.get_world_size(dp_group)

        self.buffer_index = BufferIndex(
            param_shapes=[p.shape for p in params],
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            is_distributed=is_distributed,
            chunk_size_factor=chunk_size_factor,
            sharding_strategy=sharding_strategy,
            param_group_id=param_group_id,
        )

        if is_distributed:
            self.data_size = self.buffer_index.shard_meta.size
        else:
            self.data_size = self.buffer_index.bucket_meta.size

        self.data: Optional[torch.Tensor] = None
        self._unsharded_buffer: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def init_data(self, data: torch.Tensor) -> None:
        """Bind an externally allocated tensor as the persistent storage."""
        assert data.dtype == self.dtype, f"dtype mismatch: {data.dtype} vs {self.dtype}"
        assert data.numel() == self.data_size, (
            f"size mismatch: {data.numel()} vs {self.data_size}"
        )
        self.data = data

    def set_item(self, item_id: int, item_data: torch.Tensor) -> None:
        """Write a parameter tensor into the corresponding region of the buffer."""
        if self.is_distributed:
            slice_start, slice_end = self.buffer_index._get_item_slice_in_shard(item_id)
            item_data = item_data.flatten()[slice_start:slice_end]

        local_start, local_end = self.buffer_index._get_item_local_index(item_id)
        shard = self.data[local_start:local_end]
        if shard.numel() > 0:
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, only_shard: bool = False) -> torch.Tensor:
        """Read a parameter tensor (or its shard) from the buffer."""
        if only_shard:
            start, end = self.buffer_index._get_item_local_shard_index(item_id)
        else:
            start, end = self.buffer_index._get_item_local_index(item_id)
        return self.data[start:end]

    @torch.no_grad()
    def unshard(self, async_op: bool = True) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
        """All-gather the full buffer from all shards and rebind param.data.

        After the all-gather completes, each param's .data is pointed to
        the corresponding slice of the unsharded buffer so that forward
        computation can use the full weights directly.

        For non-distributed buffers self.data is already full, so
        (self.data, None) is returned directly without rebinding.
        """
        if not self.is_distributed:
            return (self.data, None)

        bucket = self.allocator.allocate(
            param_group_id=self.buffer_index.param_group_id,
            size=self.buffer_index.bucket_meta.size,
            dtype=self.dtype,
            device=self.device,
        )
        self._unsharded_buffer = bucket.data

        sm = self.buffer_index.shard_meta
        shard = self.data[sm.local_data_index : sm.local_data_index + sm.size]

        work = torch.distributed.all_gather_into_tensor(
            output_tensor=self._unsharded_buffer,
            input_tensor=shard,
            group=self.dp_group,
            async_op=async_op,
        )

        for p in self.params:
            item_id = self.param_idx[p]
            offset, size = self.buffer_index._get_item_offset(item_id)
            p.data = self._unsharded_buffer[offset : offset + size].view(p.shape)

        return (self._unsharded_buffer, work)

    @torch.no_grad()
    def reshard(self) -> None:
        """Release the temporary unsharded buffer allocated by unshard()."""
        if not self.is_distributed:
            return
        self.allocator.free(self.buffer_index.param_group_id)
        self._unsharded_buffer = None

    @torch.no_grad()
    def reduce_grad(self, async_op: bool = True) -> Optional[torch.distributed.Work]:
        """Reduce gradients across the data-parallel group.

        For distributed buffers: reduce-scatter the full gradient into each
        rank's shard, then accumulate into self.data.
        For non-distributed buffers: all-reduce in-place.
        """
        if not self.is_distributed:
            work = torch.distributed.all_reduce(
                self.data, group=self.dp_group, async_op=async_op,
            )
            return work

        bucket = self.allocator.allocate(
            param_group_id=self.buffer_index.param_group_id,
            size=self.buffer_index.bucket_meta.size,
            dtype=self.dtype,
            device=self.device,
        )
        full_grad = bucket.data

        sm = self.buffer_index.shard_meta
        grad_shard = full_grad[sm.bucket_data_index : sm.bucket_data_index + sm.size]

        work = torch.distributed.reduce_scatter_tensor(
            output=grad_shard,
            input=full_grad,
            group=self.dp_group,
            async_op=async_op,
        )

        if not async_op:
            self.data[sm.local_data_index : sm.local_data_index + sm.size] += grad_shard
            self.allocator.free(self.buffer_index.param_group_id)

        return work
