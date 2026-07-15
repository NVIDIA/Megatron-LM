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

import sys
from pathlib import Path

import pytest
import torch
from torch.distributed.tensor import DeviceMesh

sys.path.insert(0, str(Path(__file__).parents[2]))
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.buffer_index import BufferIndex
from megatron.core.distributed.fsdp.src.megatron_fsdp.v2.utils import ParamGroupIdx


CANONICAL_LAYOUTS = ((0, 0), (0, 1), (1, 0), (1, 1))


@pytest.fixture(scope="session", autouse=True)
def dist_env():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(torch.device(f"cuda:{rank % torch.cuda.device_count()}"))
    yield
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _build_hsdp_mesh():
    world_size = torch.distributed.get_world_size()
    if world_size < 4 or world_size % 2 != 0:
        pytest.skip("BufferIndex HSDP layout coverage requires an even world size >= 4")
    mesh = torch.arange(world_size, dtype=torch.int).reshape(2, world_size // 2)
    return DeviceMesh("cuda", mesh, mesh_dim_names=("dp_outer", "dp"))


class TestBufferIndex:
    def setup_method(self):
        self.mesh = _build_hsdp_mesh()
        self.param_shapes = [torch.Size([16]), torch.Size([8]), torch.Size([40])]
        self.chunk_size_factor = 4

        self.ref_item_ranges = {
            0: (0, 16),
            1: (16, 24),
            2: (24, 64),
        }
        layout_size = 64
        shard_grid = int(self.mesh.size(0) * self.mesh.size(1) * self.chunk_size_factor)
        self.ref_bucket_size = ((layout_size + shard_grid - 1) // shard_grid) * shard_grid

        outer_size = int(self.mesh.size(0))
        inner_size = int(self.mesh.size(1))
        outer_rank = int(self.mesh.get_local_rank(mesh_dim=0))
        inner_rank = int(self.mesh.get_local_rank(mesh_dim=1))

        inner_shard_size = self.ref_bucket_size // inner_size
        inner_global_start = inner_rank * inner_shard_size
        outer_full_shard_size = self.ref_bucket_size // outer_size
        outer_full_global_start = outer_rank * outer_full_shard_size
        outer_inner_shard_size = inner_shard_size // outer_size
        outer_inner_offset = outer_rank * outer_inner_shard_size

        full_meta = (0, 0, 0, self.ref_bucket_size)
        outer_meta = (
            outer_full_global_start,
            0,
            outer_full_global_start,
            outer_full_shard_size,
        )
        inner_meta = (
            inner_global_start,
            0,
            inner_global_start,
            inner_shard_size,
        )
        outer_inner_meta = (
            inner_global_start + outer_inner_offset,
            0,
            inner_global_start + outer_inner_offset,
            outer_inner_shard_size,
        )
        self.ref_shard_metas = {
            (): full_meta,
            # shard_layout=(outer, inner): (0, 0) means neither dimension is sharded.
            (0, 0): full_meta,
            0: full_meta,
            (0,): full_meta,
            # shard_layout=(outer, inner): (1, 0) means outer sharded, inner not sharded.
            (1, 0): outer_meta,
            1: inner_meta,
            (1,): inner_meta,
            # shard_layout=(outer, inner): (0, 1) means outer not sharded, inner sharded.
            (0, 1): inner_meta,
            # shard_layout=(outer, inner): (1, 1) means both dimensions are sharded.
            (1, 1): outer_inner_meta,
        }

    @pytest.mark.parametrize(
        "shard_layout",
        [
            (),
            0,
            1,
            (0,),
            (1,),
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ],
    )
    def test_shard_meta_matches_ref(self, shard_layout):
        index = BufferIndex(
            param_shapes=self.param_shapes,
            mesh=self.mesh,
            param_group_id=ParamGroupIdx(0, 0),
            chunk_size_factor=self.chunk_size_factor,
        )
        meta = index._get_shard_meta(shard_layout)
        assert (
            meta.global_data_index,
            meta.local_data_index,
            meta.bucket_data_index,
            meta.size,
        ) == self.ref_shard_metas[shard_layout]

        # shard_layout=(outer, inner): (0, 1) means outer not sharded, inner sharded.
        assert index.shard_meta == index._get_shard_meta((0, 1))
        # shard_layout=(outer, inner): (1, 1) means outer sharded, inner sharded.
        assert index.outer_shard_meta == index._get_shard_meta((1, 1))

    @pytest.mark.parametrize("item_id", [0, 1, 2])
    # shard_layout=(outer, inner): 0/1 scalar cases are legacy inner-only shorthand.
    @pytest.mark.parametrize("shard_layout", [(), 0, 1, (0,), (1,), (0, 0), (0, 1), (1, 0), (1, 1)])
    def test_item_ranges_match_ref(self, shard_layout, item_id):
        index = BufferIndex(
            param_shapes=self.param_shapes,
            mesh=self.mesh,
            param_group_id=ParamGroupIdx(0, 0),
            chunk_size_factor=self.chunk_size_factor,
        )
        item_start, item_end = self.ref_item_ranges[item_id]
        shard_global_start, shard_local_start, _, shard_size = self.ref_shard_metas[
            shard_layout
        ]
        range_start = max(item_start, shard_global_start)
        range_end = min(item_end, shard_global_start + shard_size)

        assert index._get_item_global_range(item_id) == (item_start, item_end)

        meta = index._get_shard_meta(shard_layout)
        assert (
            meta.global_data_index,
            meta.local_data_index,
            meta.bucket_data_index,
            meta.size,
        ) == self.ref_shard_metas[shard_layout]

        if range_start >= range_end:
            expected_self = (0, 0)
            expected_local = (0, 0)
        else:
            expected_self = (range_start - item_start, range_end - item_start)
            expected_local = (
                shard_local_start + range_start - shard_global_start,
                shard_local_start + range_end - shard_global_start,
            )

        assert index._get_item_self_range(item_id, shard_layout=shard_layout) == expected_self
        assert index._get_item_local_range(item_id, shard_layout=shard_layout) == expected_local

    def _expected_local_slices(self, global_range, requested_layout, storage_layout):
        global_start, global_end = global_range
        requested_start, _, _, requested_size = self.ref_shard_metas[requested_layout]
        storage_start, storage_local_start, _, storage_size = self.ref_shard_metas[
            storage_layout
        ]
        start = max(global_start, requested_start, storage_start)
        end = min(
            global_end,
            requested_start + requested_size,
            storage_start + storage_size,
        )
        if start >= end:
            return None, None

        source_slice = slice(start - global_start, end - global_start)
        local_start = storage_local_start + start - storage_start
        return source_slice, slice(local_start, local_start + end - start)

    @pytest.mark.parametrize("item_id", [0, 1, 2])
    @pytest.mark.parametrize("requested_layout", CANONICAL_LAYOUTS)
    @pytest.mark.parametrize("storage_layout", CANONICAL_LAYOUTS)
    def test_local_slice_for_item(self, item_id, requested_layout, storage_layout):
        index = BufferIndex(
            param_shapes=self.param_shapes,
            mesh=self.mesh,
            param_group_id=ParamGroupIdx(0, 0),
            chunk_size_factor=self.chunk_size_factor,
        )
        global_range = self.ref_item_ranges[item_id]

        assert index.local_slice_for(
            global_range, requested_layout, storage_layout
        ) == self._expected_local_slices(
            global_range, requested_layout, storage_layout
        )

    @pytest.mark.parametrize("requested_layout", CANONICAL_LAYOUTS)
    @pytest.mark.parametrize("storage_layout", CANONICAL_LAYOUTS)
    def test_local_slice_for_whole_bucket(self, requested_layout, storage_layout):
        index = BufferIndex(
            param_shapes=self.param_shapes,
            mesh=self.mesh,
            param_group_id=ParamGroupIdx(0, 0),
            chunk_size_factor=self.chunk_size_factor,
        )
        global_range = (0, self.ref_bucket_size)

        assert index.local_slice_for(
            global_range, requested_layout, storage_layout
        ) == self._expected_local_slices(
            global_range, requested_layout, storage_layout
        )

    @pytest.mark.parametrize(
        ("legacy_layout", "canonical_layout"),
        [
            (None, (0, 0)),
            (0, (0, 0)),
            (1, (0, 1)),
            ((), (0, 0)),
            ((0,), (0, 0)),
            ((1,), (0, 1)),
        ],
    )
    def test_local_slice_for_legacy_layout(self, legacy_layout, canonical_layout):
        index = BufferIndex(
            param_shapes=self.param_shapes,
            mesh=self.mesh,
            param_group_id=ParamGroupIdx(0, 0),
            chunk_size_factor=self.chunk_size_factor,
        )
        global_range = (0, self.ref_bucket_size)

        assert index.local_slice_for(
            global_range, legacy_layout, (0, 0)
        ) == index.local_slice_for(
            global_range, canonical_layout, (0, 0)
        )
        assert index.local_slice_for(
            global_range, (0, 0), legacy_layout
        ) == index.local_slice_for(
            global_range, (0, 0), canonical_layout
        )
