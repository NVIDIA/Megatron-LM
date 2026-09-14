# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
# This local mcore_gdn_opt CuTeDSL source is referred from FlashInfer commit e8d31317bedb4efd52559a2234f4cb9e83428cb9; keep the license above when updating.

Tile scheduler for Chunked Gated Delta Net (GDN).

Each tile = one (batch, head) pair. The assigned CTA loops over all
chunks for that tile sequentially, which is required because the
recurrent state S must be propagated chunk-by-chunk.

Persistent grid shape:     (min(B * H, max_active_clusters), 1, 1)
Non-persistent grid shape: (B, H, 1)  - one CTA per tile, 2D grid
"""

from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.cutlass_dsl import (
    Int32,
    Integer,
    dsl_user_op,
    extract_mlir_values,
    min,
    new_from_mlir_values,
)
from cutlass.utils import WorkTileInfo


class GDNTileSchedulerParams:
    def __init__(
        self,
        num_seqs: cutlass.Int32,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
        is_GQA: bool,
        is_persistent: bool = True,
        total_tiles: Optional[cutlass.Int32] = None,
        num_seqs_fdd: cute.FastDivmodDivisor = None,
        num_heads_fdd: cute.FastDivmodDivisor = None,
        *,
        loc=None,
        ip=None,
    ):
        self.num_seqs = num_seqs  # B
        self.num_q_heads = num_q_heads
        self.num_v_heads = num_v_heads
        self.is_GQA = is_GQA
        self.is_persistent = is_persistent
        # num_o_heads: HQ for GQA (grouped-query), HV for GVA/MHA
        self.num_o_heads = num_q_heads if is_GQA else num_v_heads
        if total_tiles is None:
            self.total_tiles = num_seqs * self.num_o_heads
        else:
            self.total_tiles = total_tiles
        self.num_seqs_fdd = num_seqs_fdd
        self.num_heads_fdd = num_heads_fdd
        if num_seqs_fdd is None:
            self.num_seqs_fdd = cute.fast_divmod_create_divisor(
                num_seqs, loc=loc, ip=ip
            )
        if num_heads_fdd is None:
            self.num_heads_fdd = cute.fast_divmod_create_divisor(
                self.num_o_heads, loc=loc, ip=ip
            )
        self._loc = loc

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.num_seqs,
            self.num_q_heads,
            self.num_v_heads,
            self.is_GQA,
            self.is_persistent,
            self.total_tiles,
            self.num_seqs_fdd,
            self.num_heads_fdd,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.num_seqs,
                self.num_q_heads,
                self.num_v_heads,
                self.is_GQA,
                self.is_persistent,
                self.total_tiles,
                self.num_seqs_fdd,
                self.num_heads_fdd,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]

        return GDNTileSchedulerParams(*(tuple(obj_list)), loc=self._loc)

    @dsl_user_op
    def get_grid_shape(
        self, max_active_clusters: Int32, *, loc=None, ip=None
    ) -> Tuple[Integer, Integer, Integer]:
        if cutlass.const_expr(self.is_persistent):
            return (min(self.total_tiles, max_active_clusters), 1, 1)
        else:
            return (self.num_seqs, self.num_o_heads, 1)


class GDNTileScheduler:
    """
    Tile scheduler supporting both persistent and non-persistent modes.

    Persistent:     CTAs are assigned (batch, head) tiles in round-robin order
                    and loop over multiple tiles, processing all chunks per tile
                    before moving on.
    Non-persistent: Each CTA handles exactly one (batch, head) tile; the 2D
                    grid maps blockIdx.x -> batch, blockIdx.y -> head.
    """

    def __init__(
        self,
        params: GDNTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        num_persistent_ctas: Int32,
        current_work_linear_idx: Int32,
        is_valid: bool = True,
    ):
        self.params = params
        self.block_idx = block_idx
        self.num_persistent_ctas = num_persistent_ctas
        self._current_work_linear_idx = current_work_linear_idx
        self.is_valid = is_valid

    def __extract_mlir_values__(self) -> list[ir.Value]:
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self.block_idx,
            self.num_persistent_ctas,
            self._current_work_linear_idx,
            self.is_valid,
        ]:
            obj_values = extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> "GDNTileScheduler":
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self.block_idx,
                self.num_persistent_ctas,
                self._current_work_linear_idx,
                self.is_valid,
            ],
            self._values_pos,
            strict=True,
        ):
            obj_list.append(new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return GDNTileScheduler(*obj_list)

    @staticmethod
    @dsl_user_op
    def create(
        params: GDNTileSchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        *,
        loc=None,
        ip=None,
    ) -> "GDNTileScheduler":
        num_persistent_ctas = Int32(cute.size(grid_dim, loc=loc, ip=ip))
        bidx, _, _ = block_idx
        return GDNTileScheduler(
            params,
            block_idx,
            num_persistent_ctas,
            current_work_linear_idx=Int32(bidx),
            is_valid=True,
        )

    @staticmethod
    def get_grid_shape(
        params: GDNTileSchedulerParams,
        max_active_clusters: Int32,
        *,
        loc=None,
        ip=None,
    ) -> Tuple[Integer, Integer, Integer]:
        return params.get_grid_shape(max_active_clusters, loc=loc, ip=ip)

    def _get_current_work_for_linear_idx(
        self, linear_idx: Int32, *, loc=None, ip=None
    ) -> WorkTileInfo:
        # Used only in persistent mode: decode flat tile index -> (batch, head).
        # Tile ordering: head-major, i.e. linear_idx = batch * num_o_heads + head.
        is_valid = linear_idx < cute.size(self.params.total_tiles, loc=loc, ip=ip)
        remain_work_idx, head_idx = divmod(linear_idx, self.params.num_heads_fdd)
        _, batch_idx = divmod(remain_work_idx, self.params.num_seqs_fdd)
        cur_tile_coord = tuple(Int32(x) for x in (batch_idx, head_idx, 1))
        return WorkTileInfo(cur_tile_coord, is_valid)

    @dsl_user_op
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        if self.params.is_persistent:
            return self._get_current_work_for_linear_idx(
                self._current_work_linear_idx, loc=loc, ip=ip
            )
        # Non-persistent: 2D grid -> blockIdx.x = batch, blockIdx.y = head.
        batch_idx, head_idx, _ = self.block_idx
        return WorkTileInfo(
            (batch_idx, head_idx, self.block_idx[2]), cutlass.Boolean(self.is_valid)
        )

    @dsl_user_op
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @dsl_user_op
    def advance_to_next_work(self, *, advance_count: int = 1, loc=None, ip=None):
        if self.params.is_persistent:
            self._current_work_linear_idx += Int32(advance_count) * Int32(
                self.num_persistent_ctas
            )
        else:
            # Non-persistent: each CTA handles exactly one tile; mark exhausted.
            self.is_valid = False
