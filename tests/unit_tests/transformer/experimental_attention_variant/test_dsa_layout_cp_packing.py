# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""cp_packing_layout contract invariants for zigzag THD context-parallel sharding.

Two hard invariants, checked for both layouts (CPU-only, no distributed setup):

1. Per-rank row count: every rank owns exactly ``total_tokens / cp_size`` rows and
   the positions are strictly increasing (documents map to contiguous row slices).
2. Exact partition: the union of all ranks' positions is precisely ``[0, T)`` with
   no duplicates and no gaps, and the gathered-KV reorder index restores global
   order.

Both layouts are legal for the very same ``cu_seqlens`` (divisibility is not
evidence of a layout), which is why the layout is an explicit caller contract.
"""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    build_packed_allgather_cp_local_positions,
    build_packed_allgather_cp_query_positions_and_key_reorder,
)

CASES = [
    # (cu_seqlens, cp_size, layouts that must accept it)
    ([0, 8, 20], 2, ("per_document", "per_sequence")),
    ([0, 16, 24, 40], 4, ("per_document", "per_sequence")),
    ([0, 5, 20], 2, ("per_sequence",)),  # doc len 5 not divisible by 2 * cp
    ([0, 90, 300], 2, ("per_sequence",)),
]


class TestCPPackingLayoutInvariants:

    @pytest.mark.parametrize("cu,cp,layouts", CASES)
    def test_partition_and_row_count(self, cu, cp, layouts):
        cu_t = torch.tensor(cu, dtype=torch.int64)
        total = cu[-1]
        for layout in layouts:
            per_rank = []
            for rank in range(cp):
                pos = build_packed_allgather_cp_local_positions(
                    cu_t, cp, rank, torch.device("cpu"), cp_packing_layout=layout
                )
                assert pos.numel() == total // cp, (layout, rank)
                assert bool((pos[1:] > pos[:-1]).all()), (layout, rank)
                per_rank.append(pos)
            union = torch.cat(per_rank).sort().values
            assert torch.equal(union, torch.arange(total)), layout

    @pytest.mark.parametrize("cu,cp,layouts", CASES)
    def test_key_reorder_restores_global_order(self, cu, cp, layouts):
        cu_t = torch.tensor(cu, dtype=torch.int64)
        total = cu[-1]
        for layout in layouts:
            q_pos, reorder = build_packed_allgather_cp_query_positions_and_key_reorder(
                cu_t,
                cu_t,
                cp,
                0,
                torch.device("cpu"),
                local_output_size=total // cp,
                query_cu_seqlens_cover_output=True,
                key_cu_seqlens_cover_output=True,
                cp_packing_layout=layout,
            )
            # simulate the rank-major allgather of a positions payload
            gathered = torch.cat(
                [
                    build_packed_allgather_cp_local_positions(
                        cu_t, cp, rank, torch.device("cpu"), cp_packing_layout=layout
                    )
                    for rank in range(cp)
                ]
            )
            assert torch.equal(gathered[reorder], torch.arange(total)), layout
            assert torch.equal(q_pos, gathered[: total // cp]), layout

    def test_per_document_rejects_indivisible_docs(self):
        cu_t = torch.tensor([0, 5, 20], dtype=torch.int64)
        with pytest.raises(ValueError):
            build_packed_allgather_cp_local_positions(
                cu_t, 2, 0, torch.device("cpu"), cp_packing_layout="per_document"
            )

    def test_unknown_layout_rejected(self):
        cu_t = torch.tensor([0, 8], dtype=torch.int64)
        with pytest.raises(ValueError):
            build_packed_allgather_cp_local_positions(
                cu_t, 2, 0, torch.device("cpu"), cp_packing_layout="interleaved"
            )
