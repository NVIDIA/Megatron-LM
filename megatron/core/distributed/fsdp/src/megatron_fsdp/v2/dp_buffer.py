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

import logging
import math
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import torch

from .allocator import TemporaryBucketAllocator
from .utils import ParamGroupIdx

logger = logging.getLogger(__name__)


class BufferIndex:
    """Describes how params are laid out in a flat buffer, including global layout
    and per-rank shard information.

    Each DataParallelBuffer owns its own independent BufferIndex instance.
    """

    ItemIndex = namedtuple("ItemIndex", ["global_data_index", "size", "item_id", "shape"])
    BucketMeta = namedtuple("BucketMeta", ["global_data_index", "size", "items"])
    ShardMeta = namedtuple(
        "ShardMeta", ["global_data_index", "local_data_index", "bucket_data_index", "size"]
    )

    def __init__(
        self,
        param_shapes: List[torch.Size],
        dp_rank: int,
        dp_world_size: int,
        is_distributed: bool,
        param_group_id: ParamGroupIdx,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
    ):
        self.param_group_id = param_group_id
        self.is_distributed = is_distributed
        self.item_index_map, self.bucket_meta = self._build_layout(
            param_shapes, dp_world_size, chunk_size_factor, sharding_strategy
        )
        self.shard_meta = self._build_shard_meta(
            self.bucket_meta, is_distributed, dp_world_size, dp_rank
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
        """
        Compute global buffer layout for a list of parameter shapes.

        Regular parameters (numel >= chunk_size_factor) are placed first.
        When a regular parameter has a remainder modulo chunk_size_factor,
        its trailing grid is shared with either another regular parameter
        or filled with "fragment" parameters.  Leftover fragments are
        bin-packed into chunk_size_factor-sized grids so that every grid
        is aligned and can be split evenly across DP ranks.

        Returns:
            item_index_map:  Maps item_id -> ItemIndex (global position).
            bucket_meta:     Describes the full padded buffer.
        """

        def _pad(n: int, divisor: int) -> int:
            return int(math.ceil(n / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:
            if sharding_strategy != "no_shard":
                return _pad(data_index, dp_world_size * chunk_size_factor)
            return data_index

        def add_item(item_id, shape, offset, index_map):
            index_map[item_id] = cls.ItemIndex(
                global_data_index=offset, size=shape.numel(), item_id=item_id, shape=shape
            )

        # Separate regular and fragment parameters.
        regular_items: List[Tuple[int, torch.Size]] = []
        fragment_items: List[Tuple[int, torch.Size]] = []
        for item_id, shape in enumerate(param_shapes):
            if shape.numel() < chunk_size_factor:
                fragment_items.append((item_id, shape))
            else:
                regular_items.append((item_id, shape))

        # Sort fragments largest-first for best gap-filling.
        fragment_items.sort(key=lambda x: -x[1].numel())

        item_index_map: Dict[int, cls.ItemIndex] = {}
        data_index = 0

        # ---- First pass: place regular parameters, fill gaps with fragments ----
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

            # Try to pair another regular param whose remainder fits.
            rhs_found_id = None
            rhs_found_shape = None
            for id_rhs in regular_items[:]:
                rhs_id, rhs = id_rhs
                if rhs.numel() % chunk_size_factor == 0:
                    continue
                rhs_remain = rhs.numel() % chunk_size_factor
                if remain + rhs_remain <= chunk_size_factor:
                    rhs_found_id, rhs_found_shape = rhs_id, rhs
                    regular_items.remove(id_rhs)
                    break

            if rhs_found_id is not None:
                rhs_remain = rhs_found_shape.numel() % chunk_size_factor
                # Place the paired param so its LAST ``rhs_remain`` elements
                # land in the same alignment grid as the current param's remainder.
                # The bulk of the param extends backward from the grid boundary.
                add_item(rhs_found_id, rhs_found_shape, data_index - rhs_remain, item_index_map)
                space -= rhs_remain
                # Advance past the aligned portion of the paired param.
                data_index += (rhs_found_shape.numel() // chunk_size_factor) * chunk_size_factor

            # Fill remaining space in this grid with fragments.
            for id_frag in fragment_items[:]:
                frag_id, frag = id_frag
                if frag.numel() > space:
                    continue
                add_item(frag_id, frag, gap_offset, item_index_map)
                space -= frag.numel()
                gap_offset += frag.numel()
                fragment_items.remove(id_frag)

        # ---- helper: bin-pack leftover fragments into chunk_size_factor grids ----
        def pack_fragments(
            fragments: List[Tuple[int, torch.Size]], capacity: int
        ) -> List[List[Tuple[int, torch.Size]]]:
            """
            Bin-pack fragments into fixed-capacity slots (grids).

            Each slot has size *capacity* (== chunk_size_factor). Returns a
            list of slots, each containing one or more (param_id, shape) pairs.
            """
            sorted_frags = sorted(fragments, key=lambda p: -p[1].numel())
            slots: List[List[Tuple[int, torch.Size]]] = []
            for pid, pshape in sorted_frags:
                psize = pshape.numel()
                placed = False
                for slot in slots:
                    used = sum(p[1].numel() for p in slot)
                    if used + psize <= capacity:
                        slot.append((pid, pshape))
                        placed = True
                        break
                if not placed:
                    slots.append([(pid, pshape)])
            return slots

        # ---- Second pass: bin-pack any remaining fragments into aligned grids ----
        if fragment_items:
            fragment_slots = pack_fragments(fragment_items, chunk_size_factor)
            for slot in fragment_slots:
                offset_within_grid = 0
                for pid, pshape in slot:
                    add_item(pid, pshape, data_index + offset_within_grid, item_index_map)
                    offset_within_grid += pshape.numel()
                data_index += chunk_size_factor

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
        param_group_id: ParamGroupIdx,
        *,
        allocator: Optional[TemporaryBucketAllocator] = None,
        buffer_role: str = "model_weight",
        is_distributed: bool = False,
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
        self.buffer_role = buffer_role
        self.alloc_key = (param_group_id, buffer_role)
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
        assert data.numel() == self.data_size, f"size mismatch: {data.numel()} vs {self.data_size}"
        self.data = data

    def check_no_local_overlap(self, label: str = "") -> bool:
        """
        Runtime check: verify no two items' local slices overlap within ``self.data``.

        Returns True if layout is valid (no overlaps, all slices in bounds).
        Returns False and prints diagnostic info if any overlap or bound violation is found.
        """
        if self.data is None:
            return True

        items = self.buffer_index.item_index_map
        n_items = len(items)
        if n_items == 0:
            return True

        label_prefix = f"[{label}] " if label else ""

        # Collect (local_start, local_end, item_id, global_start, size) for each item
        slices = []
        for item_id in range(n_items):
            local_start, local_end = self.buffer_index._get_item_local_index(item_id)
            idx = self.buffer_index.item_index_map[item_id]
            slices.append((local_start, local_end, item_id, idx.global_data_index, idx.size))

        # Sort by local_start
        slices.sort(key=lambda x: x[0])

        valid = True
        data_nel = self.data.numel()

        for i in range(len(slices)):
            s_start, s_end, s_id, g_start, size = slices[i]
            shape = items[s_id].shape

            # Bounds check: end must not exceed data size
            if s_end > data_nel:
                logger.warning(
                    f"{label_prefix}OVERFLOW: item {s_id} shape={list(shape)} "
                    f"local=[{s_start}, {s_end}) but data.numel()={data_nel} "
                    f"(global=[{g_start}, {g_start + size}))"
                )
                valid = False

            # Overlap check with next item
            if i + 1 < len(slices):
                n_start, n_end, n_id, n_gstart, n_size = slices[i + 1]
                if s_end > n_start:
                    overlap = s_end - n_start
                    logger.warning(
                        f"{label_prefix}OVERLAP: item {s_id} shape={list(shape)} "
                        f"local=[{s_start}, {s_end}) overlaps item {n_id} "
                        f"local=[{n_start}, {n_end}) by {overlap} elements "
                        f"(global_{s_id}=[{g_start}, {g_start + size}), "
                        f"global_{n_id}=[{n_gstart}, {n_gstart + n_size}))"
                    )
                    valid = False

        return valid

    def check_no_global_overlap(self, label: str = "") -> bool:
        """
        Runtime check: verify no two items' **global** slices overlap.

        This checks the logical layout (should never fail if _build_layout is correct).
        """
        items = self.buffer_index.item_index_map
        n_items = len(items)
        if n_items == 0:
            return True

        label_prefix = f"[{label}] " if label else ""

        ranges = []
        for item_id in range(n_items):
            idx = items[item_id]
            ranges.append(
                (idx.global_data_index, idx.global_data_index + idx.size, item_id, idx.shape)
            )

        ranges.sort(key=lambda x: x[0])

        valid = True
        for i in range(len(ranges) - 1):
            a_start, a_end, a_id, a_shape = ranges[i]
            b_start, b_end, b_id, b_shape = ranges[i + 1]
            if a_end > b_start:
                logger.warning(
                    f"{label_prefix}GLOBAL OVERLAP: item {a_id} shape={list(a_shape)} "
                    f"[{a_start}, {a_end}) vs item {b_id} shape={list(b_shape)} "
                    f"[{b_start}, {b_end}) overlap={a_end - b_start}"
                )
                valid = False

        if valid:
            pass  # silent on success
        return valid

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
    def unshard(
        self, async_op: bool = True, bind_params: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
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
            key=self.alloc_key,
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
        if self._unsharded_buffer.is_cuda:
            # The temporary bucket may be released from another stream before this
            # collective finishes; record the producer stream for allocator safety.
            self._unsharded_buffer.record_stream(torch.cuda.current_stream())

        if bind_params:
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
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def fetch_unsharded_buffer(self) -> torch.Tensor:
        """Return the unsharded buffer, allocating it if needed."""
        if not self.is_distributed:
            return self.data
        if self._unsharded_buffer is None:
            bucket = self.allocator.allocate(
                key=self.alloc_key,
                size=self.buffer_index.bucket_meta.size,
                dtype=self.dtype,
                device=self.device,
            )
            self._unsharded_buffer = bucket.data
        return self._unsharded_buffer

    @torch.no_grad()
    def reduce_grad(self, grad_comm_dtype: Optional[torch.dtype] = None, async_op: bool = False):
        """Reduce gradients across the data-parallel group.

        For distributed buffers: reduce-scatter the full gradient into each
        rank's shard, then accumulate into self.data.
        For non-distributed buffers: all-reduce in-place.
        If grad_comm_dtype differs from self.dtype, communicate with a temporary
        casted tensor and cast the reduced result back before accumulation.
        """
        del async_op
        grad_comm_dtype = grad_comm_dtype or self.dtype

        if self.gradient_scaling_factor in (None, 1.0):
            op = torch.distributed.ReduceOp.SUM
            prescale = False
        elif grad_comm_dtype != torch.bfloat16:
            op = torch.distributed._make_nccl_premul_sum(self.gradient_scaling_factor)
            prescale = False
        else:
            op = torch.distributed.ReduceOp.SUM
            prescale = True

        if not self.is_distributed:
            comm_data = (
                self.data if grad_comm_dtype == self.dtype else self.data.to(grad_comm_dtype)
            )
            if prescale:
                comm_data.mul_(self.gradient_scaling_factor)
            torch.distributed.all_reduce(comm_data, group=self.dp_group, op=op)
            if comm_data is not self.data:
                self.data.copy_(comm_data.to(self.dtype))
            return

        full_grad = self.fetch_unsharded_buffer()
        comm_input = full_grad if grad_comm_dtype == self.dtype else full_grad.to(grad_comm_dtype)
        if full_grad.is_cuda:
            # Keep temporary reduce-scatter buffers tied to the stream that uses them.
            full_grad.record_stream(torch.cuda.current_stream())
        if prescale:
            comm_input.mul_(self.gradient_scaling_factor)

        sm = self.buffer_index.shard_meta
        local_grad_shard = self.data[sm.local_data_index : sm.local_data_index + sm.size]
        reduced_grad_shard = torch.empty(sm.size, dtype=grad_comm_dtype, device=self.device)

        torch.distributed.reduce_scatter_tensor(
            output=reduced_grad_shard, input=comm_input, group=self.dp_group, op=op
        )

        # Accumulate into the persistent local shard buffer. The reduce-scatter output
        # must not alias the full input buffer, otherwise the collective can clobber its
        # own input and silently corrupt gradients.
        local_grad_shard += reduced_grad_shard


def check_all_fsdp_buffers(module) -> bool:
    """
    Scan every FSDPModule in *module* and verify no local slice overlaps
    in any buffer (model_weight, main_weight, main_grad).

    Call this at any point after FSDP initialization to catch runtime
    corruption.  Returns True if all buffers are clean.
    """
    import torch.distributed as dist

    from .fsdp_module import FSDPModule

    rank = dist.get_rank() if dist.is_initialized() else -1
    all_ok = True

    for name, child in module.named_modules():
        if not isinstance(child, FSDPModule):
            continue
        for param_names, param_group in child._named_param_groups:
            gid = f"mod={name} pg={param_group.param_group_id} rank={rank}"
            if param_group.model_weight_buffer is not None:
                ok = param_group.model_weight_buffer.check_no_local_overlap(gid + " wbuf")
                all_ok = all_ok and ok
            if param_group.main_weight_buffer is not None:
                ok = param_group.main_weight_buffer.check_no_local_overlap(gid + " mbuf")
                all_ok = all_ok and ok
            if param_group.main_grad_buffer is not None:
                ok = param_group.main_grad_buffer.check_no_local_overlap(gid + " gbuf")
                all_ok = all_ok and ok

    return all_ok
