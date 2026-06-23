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

from .allocator import BucketAllocator, TemporaryBucketAllocator, _free_storage
from .mixed_precision import MixedPrecisionPolicy
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
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.chunk_size_factor = chunk_size_factor
        self.sharding_strategy = sharding_strategy
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
                # For non-distributed buffers, each rank has the full buffer, so
                # the local index is the same as the global index.
                local_data_index=global_data_index,
                bucket_data_index=bucket_data_index,
                size=shard_size,
            )

    # ------------------------------------------------------------------ #
    #  Compaction — scale indices proportionally for packed storage
    # ------------------------------------------------------------------ #

    def compact(self, factor: float, compact_shapes: List[torch.Size]) -> None:
        """Scale all indices proportionally for packed storage.

        Args:
            factor: Scale factor (0.5 for NVFP4 2-values-per-byte packing).
            compact_shapes: Per-item shapes for the packed layout (same
                length and order as the original param_shapes).
        """
        new_map: Dict[int, "BufferIndex.ItemIndex"] = {}
        for item_id, item in self.item_index_map.items():
            new_map[item_id] = self.ItemIndex(
                global_data_index=int(item.global_data_index * factor),
                size=int(item.size * factor),
                item_id=item.item_id,
                shape=compact_shapes[item_id],
            )
        self.item_index_map = new_map

        self.bucket_meta = self.BucketMeta(
            global_data_index=0,
            size=int(self.bucket_meta.size * factor),
            items=list(new_map.values()),
        )
        self.shard_meta = self._build_shard_meta(
            self.bucket_meta, self.is_distributed, self.dp_world_size, self.dp_rank
        )

    # ------------------------------------------------------------------ #
    #  Internal index query methods — three coordinate domains:
    #
    #  _get_item_self_range   → (start, end) relative to the item's own
    #                            start.  Tells what portion of this item
    #                            falls within the current rank's shard.
    #  _get_item_local_range  → (start, end) within self.data (the local
    #                            GPU buffer).  Where to read/write bytes.
    #  _get_item_global_range → (start, end) in the full logical
    #                            (unsharded) buffer, same on all
    #                            ranks.
    # ------------------------------------------------------------------ #

    def _get_item_global_range(self, item_id: int) -> Tuple[int, int]:
        """Return (start, end) in the full unsharded buffer for the given item."""
        idx = self.item_index_map[item_id]
        return (idx.global_data_index, idx.global_data_index + idx.size)

    def _get_item_self_range(self, item_id: int, *, as_shard: bool = True) -> Tuple[int, int]:
        """Return coordinates relative to the item's own start.

        When ``as_shard=True`` (default), returns the portion of the item
        that falls within this rank's shard — the slice ``(start, end)``
        within the item.  When ``as_shard=False``, returns ``(0, size)``
        representing the full item.
        """
        idx = self.item_index_map[item_id]
        if not as_shard:
            return (0, idx.size)

        item_start = idx.global_data_index
        item_end = item_start + idx.size
        shard_start = self.shard_meta.global_data_index
        shard_end = shard_start + self.shard_meta.size

        if item_start > shard_end or item_end < shard_start:
            return (0, 0)

        start = max(item_start, shard_start) - item_start
        end = min(item_end, shard_end) - item_start
        return (start, end)

    def _get_item_local_range(self, item_id: int, *, as_shard: bool = False) -> Tuple[int, int]:
        """Return coordinates within self.data for the item.

        Parameters
        ----------
        as_shard : bool
            If True, compute the shard intersection even when the buffer
            is not distributed.  Default (False) returns the full item
            range for non-distributed buffers.
        """
        if not self.is_distributed and not as_shard:
            idx = self.item_index_map[item_id]
            return (idx.global_data_index, idx.global_data_index + idx.size)

        slice_start, slice_end = self._get_item_self_range(item_id)
        if slice_start == slice_end:
            return (0, 0)

        idx = self.item_index_map[item_id]
        offset = (
            idx.global_data_index
            - self.shard_meta.global_data_index
            + self.shard_meta.local_data_index
        )
        return (offset + slice_start, offset + slice_end)


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
        mp_policy: MixedPrecisionPolicy,
        *,
        allocator: Optional[BucketAllocator] = None,
        buffer_role: str = "model_weight",
        is_distributed: bool = False,
        gradient_scaling_factor: Optional[float] = None,
        chunk_size_factor: int = 1,
        sharding_strategy: str = "no_shard",
    ):
        assert mp_policy is not None, "DataParallelBuffer requires a mixed-precision policy"
        self.params = params
        self.param_idx = param_idx
        self.dtype = dtype
        self.device = device
        self.dp_group = dp_group
        self.allocator = allocator if allocator is not None else TemporaryBucketAllocator()
        self.buffer_role = buffer_role
        self.alloc_key = (param_group_id, buffer_role)
        self.mp_policy = mp_policy
        self.is_distributed = is_distributed
        self.sharding_strategy = sharding_strategy
        self.gradient_scaling_factor = gradient_scaling_factor

        dp_rank = torch.distributed.get_rank(dp_group)
        dp_world_size = torch.distributed.get_world_size(dp_group)

        # Always build layout with logical shapes and shared chunk_size_factor
        # so that all buffers share the same proportional item-offset mapping.
        _logical_shapes = [p.shape for p in params]
        self.buffer_index = BufferIndex(
            param_shapes=_logical_shapes,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            is_distributed=is_distributed,
            chunk_size_factor=chunk_size_factor,
            sharding_strategy=sharding_strategy,
            param_group_id=param_group_id,
        )

        # Compact NVFP4 weight buffers: scale all indices proportionally so
        # the buffer holds only the packed data without fragment-binning waste.
        if buffer_role in ("model_weight", "transpose_weight") and any(
            mp_policy.is_nvfp4_param(p) for p in params
        ):
            compact_shapes = mp_policy.get_param_storage_shapes(params)
            self.buffer_index.compact(0.5, compact_shapes)

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
        if self.buffer_role in ("model_weight", "transpose_weight") and not self.is_distributed:
            self.data._dirty = False

    # ------------------------------------------------------------------ #
    #  CPU offload
    # ------------------------------------------------------------------ #

    def _is_on_cpu(self) -> bool:
        """True if ``self.data`` is resident on CPU."""
        return self.data is not None and self.data.device.type == "cpu"

    def _ensure_data_on_gpu(self) -> bool:
        """Move ``self.data`` to GPU if currently on CPU.

        Returns True if a move happened (caller must rebuild dist views).
        """
        if not self._is_on_cpu():
            return False
        self.data = self.data.to(self.device, non_blocking=True)
        return True

    def _move_data_to(
        self,
        target_device: torch.device,
        pin_memory: bool = False,
        non_blocking: bool = True,
    ) -> None:
        """Move ``self.data`` to *target_device*, optionally using pinned memory.

        Caller must call ``ParameterGroup._rebuild_dist_views()`` afterwards
        because ``dist_params._local_tensor`` views share ``self.data`` Storage.
        """
        if self.data is None or self.data.device == target_device:
            return
        if target_device.type == "cpu" and pin_memory:
            cpu_data = torch.empty(self.data.shape, dtype=self.data.dtype, pin_memory=True)
            cpu_data.copy_(self.data, non_blocking=non_blocking)
            _free_storage(self.data)
            self.data = cpu_data
        else:
            self.data = self.data.to(target_device, non_blocking=non_blocking)

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
            local_start, local_end = self.buffer_index._get_item_local_range(item_id)
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
            slice_start, slice_end = self.buffer_index._get_item_self_range(item_id)
            item_data = item_data.flatten()[slice_start:slice_end]

        local_start, local_end = self.buffer_index._get_item_local_range(item_id)
        shard = self.data[local_start:local_end]
        if shard.numel() > 0:
            shard.data.copy_(item_data.flatten())

    def get_item(self, item_id: int, *, as_shard: bool = False) -> torch.Tensor:
        """Read a parameter tensor (or its shard) from the buffer."""
        start, end = self.buffer_index._get_item_local_range(item_id, as_shard=as_shard)
        return self.data[start:end]

    def is_unsharded(self) -> bool:
        """Return whether this buffer currently has a full unsharded view."""
        full_tensor = self._unsharded_buffer if self.is_distributed else self.data
        if full_tensor is not None and not getattr(full_tensor, "_dirty", True):
            return True
        return False

    @torch.no_grad()
    def unshard(
        self,
        bind_params: bool = False,
    ) -> torch.Tensor:
        """All-gather the full buffer from all shards and bind parameter storage.

        For non-distributed buffers self.data is already full, so
        self.data is returned directly. If a replicated buffer only has this
        rank's updated shard, the shard is all-gathered into self.data first.
        """
        full_buffer = self.fetch_buffer(as_shard=False)

        if not self.is_distributed and not getattr(full_buffer, "_dirty", False):
            if bind_params:
                self._bind_buffer_to_params(full_buffer)
            return full_buffer

        sm = self.buffer_index.shard_meta
        shard_buffer = self.data[sm.local_data_index : sm.local_data_index + sm.size]
        torch.distributed.all_gather_into_tensor(
            output_tensor=full_buffer,
            input_tensor=shard_buffer,
            group=self.dp_group,
        )
        if full_buffer.is_cuda:
            # Temporary all-gather buckets may be released from another stream before
            # the collective finishes; record the producer stream for allocator safety.
            full_buffer.record_stream(torch.cuda.current_stream())

        if bind_params:
            self._bind_buffer_to_params(full_buffer)

        setattr(full_buffer, "_dirty", False)  # mark the buffer as clean (unsharded)

        return full_buffer

    def _bind_buffer_to_params(self, buffer: torch.Tensor) -> None:
        """Bind the given buffer to the params according to the layout."""
        assert buffer.numel() == self.buffer_index.bucket_meta.size, (
            f"Buffer size {buffer.numel()} does not match expected size "
            f"{self.buffer_index.bucket_meta.size}"
        )
        for p in self.params:
            item_id = self.param_idx[p]
            start, end = self.buffer_index._get_item_global_range(item_id)
            idx_shape = self.buffer_index.item_index_map[item_id].shape
            param_data = buffer[start:end].view(idx_shape)
            self.mp_policy.bind_unsharded_param(p, param_data, self.buffer_role)

    @torch.no_grad()
    def reshard(self) -> None:
        """Release the temporary unsharded buffer allocated by unshard()."""
        if not self.is_distributed:
            return
        self.allocator.free(self.alloc_key)
        self._unsharded_buffer = None

    def fetch_buffer(self, *, as_shard: bool = False) -> torch.Tensor:
        """Return the buffer, allocating the full unsharded view if needed.

        Parameters
        ----------
        as_shard : bool
            If True, return only this rank's shard slice of the full buffer.
            Default (False) returns the full unsharded buffer.

        Memory allocation always occurs on the default stream for deterministic
        caching-allocator behaviour.
        """
        if self.is_distributed:
            if self._unsharded_buffer is None:
                bucket = self.allocator.allocate(
                    key=self.alloc_key,
                    size=self.buffer_index.bucket_meta.size,
                    dtype=self.dtype,
                    device=self.device,
                )
                self._unsharded_buffer = bucket.data
            full = self._unsharded_buffer
        else:
            assert self.data is not None, "DataParallelBuffer data not initialized"
            full = self.data

        if as_shard:
            sm = self.buffer_index.shard_meta
            return full[sm.bucket_data_index : sm.bucket_data_index + sm.size]
        return full

    @torch.no_grad()
    def reduce_grad(
        self,
        overwrite_grad: bool = False,
    ):
        """Reduce gradients into the optimizer-facing local shard.

        For distributed buffers, this reduce-scatters a temporary full gradient
        and accumulates the result into the persistent local shard. For
        replicated buffers, this reduce-scatters the full accumulation buffer
        once into this rank's virtual shard for ZeRO-1 optimizer consumption.
        For no-shard buffers, this all-reduces the full gradient buffer.
        If grad_comm_dtype differs from self.dtype, communicate with a temporary
        casted tensor and cast the reduced result back before accumulation.
        """
        if self.sharding_strategy in ("no_shard", "optim"):
            overwrite_grad = True

        grad_comm_dtype = self.mp_policy.grad_comm_dtype or self.dtype

        if self.gradient_scaling_factor in (None, 1.0):
            op = torch.distributed.ReduceOp.SUM
            prescale = False
        elif grad_comm_dtype != torch.bfloat16:
            op = torch.distributed._make_nccl_premul_sum(self.gradient_scaling_factor)
            prescale = False
        else:
            op = torch.distributed.ReduceOp.SUM
            prescale = True

        sm = self.buffer_index.shard_meta
        local_grad_shard = self.data[sm.local_data_index : sm.local_data_index + sm.size]

        if not self.is_distributed and self.sharding_strategy == "no_shard":
            comm_input = (
                self.data if grad_comm_dtype == self.dtype else self.data.to(grad_comm_dtype)
            )
            if prescale:
                comm_input.mul_(self.gradient_scaling_factor)
            torch.distributed.all_reduce(comm_input, group=self.dp_group, op=op)
            if grad_comm_dtype != self.dtype:
                self.data.copy_(comm_input.to(self.dtype))
            return

        if self.is_distributed:
            # ZeRO-2/3 (optim_grads/optim_grads_params): ``self.data`` is the
            # persistent local grad shard. The full grad buffer is temporary,
            # assembled only for this reduce-scatter, and the RS result is
            # accumulated into ``local_grad_shard`` for gradient accumulation.
            input_buffer = self.fetch_buffer()
            output_offset = sm.bucket_data_index
            if input_buffer.is_cuda:
                # Keep temporary reduce-scatter buffers tied to the stream that uses them.
                input_buffer.record_stream(torch.cuda.current_stream())
        else:
            # ZeRO-1 (optim): ``self.data`` is the replicated full grad
            # accumulation buffer. The optimizer consumes only this rank's
            # virtual shard, so the one delayed RS writes directly into that
            # slice instead of accumulating into a separate shard buffer.
            input_buffer = self.fetch_buffer()
            output_offset = sm.local_data_index

        comm_input = input_buffer.to(grad_comm_dtype)
        if prescale:
            comm_input.mul_(self.gradient_scaling_factor)
        reduced_grad_shard = comm_input[output_offset : output_offset + sm.size]

        torch.distributed.reduce_scatter_tensor(
            output=reduced_grad_shard, input=comm_input, group=self.dp_group, op=op
        )

        # If the reduced shard is already in the local grad buffer, skip copy/accumulation.
        if local_grad_shard.data_ptr() == reduced_grad_shard.data_ptr():
            return

        if overwrite_grad:
            local_grad_shard.copy_(reduced_grad_shard)
        else:
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
