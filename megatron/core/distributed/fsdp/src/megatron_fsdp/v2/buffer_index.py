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

import math
from collections import namedtuple
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributed.tensor import DeviceMesh

from .utils import ParamGroupIdx


class BufferIndex:
    """Describes how params are laid out in a flat buffer, including global layout
    and per-rank shard information.

    The index always builds coordinate metadata for a 2D ``(outer, inner)``
    mesh. Callers choose which mesh dimensions shard a query via
    ``shard_layout``: ``(0, 0)`` full, ``(1, 0)`` outer, ``(0, 1)`` inner, and
    ``(1, 1)`` inner then outer.

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
        mesh: DeviceMesh,
        param_group_id: ParamGroupIdx,
        chunk_size_factor: int = 1,
    ):
        assert mesh.ndim == 2, f"BufferIndex expects a 2D mesh, got {mesh.ndim}D."
        self.param_group_id = param_group_id
        dp_rank = int(mesh.get_local_rank(mesh_dim=1))
        dp_world_size = int(mesh.size(1))
        outer_dp_rank = int(mesh.get_local_rank(mesh_dim=0))
        outer_dp_world_size = int(mesh.size(0))
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.outer_dp_rank = outer_dp_rank
        self.outer_dp_world_size = outer_dp_world_size
        self.chunk_size_factor = chunk_size_factor
        # Pad for the finest shard grid so full, inner, and outer coordinate
        # queries all preserve dim-0 row boundaries.
        layout_world_size = dp_world_size * outer_dp_world_size
        self.item_index_map, self.bucket_meta = self._build_layout(
            param_shapes, layout_world_size, chunk_size_factor
        )
        self._refresh_shard_metas()

    # ------------------------------------------------------------------ #
    #  Layout construction (global, rank-independent)
    # ------------------------------------------------------------------ #

    @classmethod
    def _build_layout(
        cls,
        param_shapes: List[torch.Size],
        layout_world_size: int,
        chunk_size_factor: int,
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
            if layout_world_size > 1:
                return _pad(data_index, layout_world_size * chunk_size_factor)
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
        parent_meta,
        num_shards: int,
        shard_id: int,
    ) -> "BufferIndex.ShardMeta":
        if parent_meta.size % num_shards != 0:
            raise ValueError(
                f"Shard parent size {parent_meta.size} is not divisible by "
                f"shard count {num_shards}."
            )
        shard_size = parent_meta.size // num_shards
        shard_offset = shard_id * shard_size
        parent_bucket_data_index = getattr(parent_meta, "bucket_data_index", 0)
        return cls.ShardMeta(
            global_data_index=parent_meta.global_data_index + shard_offset,
            local_data_index=0,
            bucket_data_index=parent_bucket_data_index + shard_offset,
            size=shard_size,
        )

    def _refresh_shard_metas(self) -> None:
        full_meta = self.ShardMeta(
            global_data_index=0,
            local_data_index=0,
            bucket_data_index=0,
            size=self.bucket_meta.size,
        )
        inner_meta = self._build_shard_meta(self.bucket_meta, self.dp_world_size, self.dp_rank)
        outer_full_meta = self._build_shard_meta(
            self.bucket_meta, self.outer_dp_world_size, self.outer_dp_rank
        )
        outer_inner_meta = self._build_shard_meta(
            inner_meta, self.outer_dp_world_size, self.outer_dp_rank
        )

        # ``shard_layout`` follows PyTorch DeviceMesh dim order:
        # mesh_dim 0 is outer-DP, mesh_dim 1 is inner-DP.
        # The cache keys below use 0/1 flags to mean unsharded/sharded.
        self.inner_shard_metas = {
            0: full_meta,
            1: inner_meta,
        }
        # Outer has four views: outer flag x inner flag, matching mesh dim order.
        self.outer_shard_metas = {
            (0, 0): full_meta,
            (0, 1): inner_meta,
            (1, 0): outer_full_meta,
            (1, 1): outer_inner_meta,
        }

        self.shard_meta = inner_meta
        self.outer_shard_meta = outer_inner_meta

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
        self._refresh_shard_metas()

    # ------------------------------------------------------------------ #
    #  Index query methods — three coordinate domains:
    #
    #  _get_item_self_range   → (start, end) relative to the item's own
    #                            start.  Tells what portion of this item
    #                            falls within the current rank's shard.
    #  _get_item_local_range  → (start, end) relative to the selected
    #                            shard_layout coordinate domain.
    #  _get_item_global_range → (start, end) in the full logical
    #                            (unsharded) buffer, same on all
    #                            ranks.
    # ------------------------------------------------------------------ #

    def _get_shard_meta(self, shard_layout: Iterable[int] | int | None):
        # shard_layout=(outer, inner): 1 means that dimension is sharded, 0 means replicated.
        if shard_layout is None:
            outer_sharded = 0
            inner_sharded = 0
        elif isinstance(shard_layout, int):
            if shard_layout not in (0, 1):
                raise ValueError(f"Unsupported shard_layout: {shard_layout}")
            outer_sharded = 0
            inner_sharded = shard_layout
        else:
            shard_layout = tuple(int(dim) for dim in shard_layout)
            if any(dim not in (0, 1) for dim in shard_layout):
                raise ValueError(f"Unsupported shard_layout: {shard_layout}")
            if len(shard_layout) == 0:
                outer_sharded = 0
                inner_sharded = 0
            elif len(shard_layout) == 1:
                outer_sharded = 0
                inner_sharded = shard_layout[0]
            elif len(shard_layout) == 2:
                outer_sharded, inner_sharded = shard_layout
            else:
                raise ValueError(f"Unsupported shard_layout: {shard_layout}")
        return self.outer_shard_metas[(outer_sharded, inner_sharded)]

    def local_slice_for(
        self,
        global_range: Tuple[int, int],
        requested_layout: Iterable[int] | int | None,
        storage_layout: Iterable[int] | int | None,
    ) -> Tuple[Optional[slice], Optional[slice]]:
        """Clip global_range to requested and storage shard layouts.

        The source slice indexes the object described by global_range relative
        to global_range[0]. The local slice indexes storage physically laid out
        as storage_layout. Returns (None, None) when the intersection is empty.
        """
        global_start, global_end = global_range
        requested_meta = self._get_shard_meta(requested_layout)
        storage_meta = self._get_shard_meta(storage_layout)
        start = max(
            global_start,
            requested_meta.global_data_index,
            storage_meta.global_data_index,
        )
        end = min(
            global_end,
            requested_meta.global_data_index + requested_meta.size,
            storage_meta.global_data_index + storage_meta.size,
        )
        if start >= end:
            return None, None

        source_slice = slice(start - global_start, end - global_start)
        local_start = (
            storage_meta.local_data_index + start - storage_meta.global_data_index
        )
        local_slice = slice(local_start, local_start + end - start)
        return source_slice, local_slice

    def _get_item_global_range(self, item_id: int) -> Tuple[int, int]:
        """Return (start, end) in the full unsharded buffer for the given item."""
        idx = self.item_index_map[item_id]
        return (idx.global_data_index, idx.global_data_index + idx.size)

    def _get_item_self_range(
        self,
        item_id: int,
        *,
        shard_layout: Iterable[int] | None = (0, 1),
    ) -> Tuple[int, int]:
        """Return coordinates relative to the item's own start.

        ``shard_layout`` selects the mesh dimensions to shard on.
        """
        idx = self.item_index_map[item_id]
        item_start = idx.global_data_index
        item_end = item_start + idx.size
        range_start = item_start
        range_end = item_end

        shard_meta = self._get_shard_meta(shard_layout)
        shard_start = shard_meta.global_data_index
        shard_end = shard_start + shard_meta.size
        range_start = max(range_start, shard_start)
        range_end = min(range_end, shard_end)

        if range_start >= range_end:
            return (0, 0)

        return (range_start - idx.global_data_index, range_end - idx.global_data_index)

    def _get_item_local_range(
        self,
        item_id: int,
        *,
        shard_layout: Iterable[int] | None = (0, 0),
    ) -> Tuple[int, int]:
        """Return item coordinates relative to the selected shard layout.

        The result is not aware of DataParallelBuffer storage.
        """
        idx = self.item_index_map[item_id]
        range_start = idx.global_data_index
        range_end = range_start + idx.size

        shard_meta = self._get_shard_meta(shard_layout)
        shard_start = shard_meta.global_data_index
        shard_end = shard_start + shard_meta.size
        range_start = max(range_start, shard_start)
        range_end = min(range_end, shard_end)

        if range_start >= range_end:
            return (0, 0)

        local_start = shard_meta.local_data_index + range_start - shard_meta.global_data_index
        return (local_start, local_start + (range_end - range_start))
