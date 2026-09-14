# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU tests for the CSA2 packed-sequence (THD) helpers."""

import torch

from megatron.core.transformer.experimental_attention_variant.csa2.reference import (
    compressed_visible_counts,
    sliding_window_indices,
)
from megatron.core.transformer.experimental_attention_variant.csa2.thd import (
    compressed_cu_seqlens,
    compressed_entry_metadata,
    compressor_group_rows,
    owned_compressed_entries,
    row_metadata,
    segment_rows,
    shift_compressed_indices,
    visible_compressed_counts,
    window_indices_thd,
)

CU = torch.tensor([0, 5, 12, 14])  # three segments: 5, 7, 2 tokens


class TestRowMetadata:
    def test_segments_positions_validity(self):
        meta = row_metadata(CU, local_rows=16)  # two padding rows past the pack
        assert meta.segment_ids.tolist() == [0] * 5 + [1] * 7 + [2] * 2 + [2, 2]
        assert meta.positions.tolist() == list(range(5)) + list(range(7)) + [0, 1, 0, 0]
        assert meta.valid.tolist() == [True] * 14 + [False, False]

    def test_global_start(self):
        meta = row_metadata(CU, local_rows=4, global_start=7)
        assert meta.segment_ids.tolist() == [1, 1, 1, 1]
        assert meta.positions.tolist() == [2, 3, 4, 5]
        assert segment_rows(meta, 1).tolist() == [0, 1, 2, 3]

    def test_padded_pack(self):
        # physical starts [0, 8, 16], valid lengths [3, 6]: rows 3..7 and 14..15 are padding
        cu_phys = torch.tensor([0, 8, 16])
        lens = torch.tensor([3, 6])
        meta = row_metadata(cu_phys, local_rows=16, seq_lens=lens)
        assert meta.valid.tolist() == [True] * 3 + [False] * 5 + [True] * 6 + [False] * 2
        assert meta.positions.tolist() == [0, 1, 2, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 0, 0]
        cu_comp = compressed_cu_seqlens(cu_phys, ratio=2, seq_lens=lens)
        assert cu_comp.tolist() == [0, 1, 4]  # 3//2, 6//2
        _, grp, first = compressed_entry_metadata(cu_phys, cu_comp, ratio=2)
        assert first.tolist() == [0, 8, 10, 12]  # physical rows, never inside the padding
        assert grp.tolist() == [0, 0, 1, 2]
        idx = window_indices_thd(meta, window=3, local_row_base=0, halo=0)
        assert idx[3].tolist() == [-1, -1, -1]  # padding row attends nothing
        assert idx[8].tolist() == [8, -1, -1]  # segment 1 starts at its physical row

    def test_packed_layout_from_params(self):
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.transformer.experimental_attention_variant.csa2.thd import packed_layout

        cu = torch.tensor([0, 3, 9], dtype=torch.int32)
        starts, lens = packed_layout(
            PackedSeqParams(qkv_format='thd', cu_seqlens_q=cu, cu_seqlens_q_padded=None)
        )
        assert starts.tolist() == [0, 3, 9] and lens.tolist() == [3, 6]
        starts, lens = packed_layout(
            PackedSeqParams(
                qkv_format='thd',
                cu_seqlens_q=cu,
                cu_seqlens_q_padded=torch.tensor([0, 8, 16], dtype=torch.int32),
            )
        )
        assert starts.tolist() == [0, 8, 16] and lens.tolist() == [3, 6]


class TestCompressedLayout:
    def test_cu_and_entries(self):
        cu_comp = compressed_cu_seqlens(CU, ratio=2)
        assert cu_comp.tolist() == [0, 2, 5, 6]  # 5//2, 7//2, 2//2
        seg, grp, first = compressed_entry_metadata(CU, cu_comp, ratio=2)
        assert seg.tolist() == [0, 0, 1, 1, 1, 2]
        assert grp.tolist() == [0, 1, 0, 1, 2, 0]
        assert first.tolist() == [0, 2, 5, 7, 9, 12]
        rows = compressor_group_rows(first, ratio=2)
        assert rows.tolist() == [[0, 1], [2, 3], [5, 6], [7, 8], [9, 10], [12, 13]]
        # row 4 (tail of segment 0) and row 11 (tail of segment 1) are never pooled
        assert 4 not in rows.flatten().tolist() and 11 not in rows.flatten().tolist()

    def test_ratio_one_is_identity(self):
        cu_comp = compressed_cu_seqlens(CU, ratio=1)
        assert cu_comp.tolist() == CU.tolist()
        _, _, first = compressed_entry_metadata(CU, cu_comp, ratio=1)
        assert first.tolist() == list(range(14))

    def test_ownership_across_cp_blocks(self):
        cu_comp = compressed_cu_seqlens(CU, ratio=2)
        _, _, first = compressed_entry_metadata(CU, cu_comp, ratio=2)
        owned = [
            owned_compressed_entries(first, 2, global_start=s, local_rows=7).tolist()
            for s in (0, 7)
        ]
        # every entry owned exactly once; group [5,6] belongs to the block holding row 6
        assert [a or b for a, b in zip(*owned)] == [True] * 6
        assert not any(a and b for a, b in zip(*owned))
        assert owned[0] == [True, True, True, False, False, False]


class TestVisibilityAndIndices:
    def test_visible_counts_match_single_sequence(self):
        meta = row_metadata(CU, local_rows=14)
        cu_comp = compressed_cu_seqlens(CU, ratio=2)
        counts = visible_compressed_counts(meta, cu_comp, ratio=2)
        expected = torch.cat(
            [
                compressed_visible_counts(5, 2).clamp_max(2),
                compressed_visible_counts(7, 2).clamp_max(3),
                compressed_visible_counts(2, 2).clamp_max(1),
            ]
        )
        assert counts.tolist() == expected.tolist()

    def test_window_indices_respect_segments(self):
        meta = row_metadata(CU, local_rows=14)
        idx = window_indices_thd(meta, window=3, local_row_base=0, halo=0)
        assert idx.shape == (14, 3)
        # row 5 is position 0 of segment 1: only itself
        assert idx[5].tolist() == [5, -1, -1]
        # row 7 (position 2): rows 7, 6, 5
        assert idx[7].tolist() == [7, 6, 5]
        # single-segment pack equals the SBHD helper
        single = row_metadata(torch.tensor([0, 9]), local_rows=9)
        assert torch.equal(window_indices_thd(single, 4, 0, 0), sliding_window_indices(9, 4))

    def test_window_indices_with_halo(self):
        # rank block starting at global row 7 (segment 1 position 2) with a 3-row halo
        meta = row_metadata(CU, local_rows=5, global_start=7)
        idx = window_indices_thd(meta, window=4, local_row_base=0, halo=3)
        # local row 0 = position 2: keys at positions 2,1,0 are buffer rows 3,2,1; k=3 leaves
        # the segment
        assert idx[0].tolist() == [3, 2, 1, -1]
        # local row 4 = global 11 = position 6: buffer rows 7,6,5,4
        assert idx[4].tolist() == [7, 6, 5, 4]

    def test_shift_compressed(self):
        local = torch.tensor([[0, 2, -1]], dtype=torch.int32)
        out = shift_compressed_indices(local, segment_comp_start=5, compressed_base=100)
        assert out.tolist() == [[105, 107, -1]]
