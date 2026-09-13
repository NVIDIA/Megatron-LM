# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Whole-bin samples for pre-packed parquets (``--varlen-bins-as-samples``): CPU only."""

import numpy as np
import pytest
import torch

from megatron.training.datasets.sft_dataset import IGNORE_INDEX
from megatron.training.datasets.varlen_dataset import (
    _PackedBins,
    _shift_and_pad_sequence,
    build_packed_bin_sample,
)

EOD, PAD, MAX_LEN, GRAN = 2, 0, 256, 16


def _write_parquet(path, bins):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = {"input_ids": [], "loss_mask": [], "seq_start_id": []}
    for seqs in bins:
        ids, mask, starts = [], [], []
        for s_ids, s_mask in seqs:
            starts.append(len(ids))
            ids += list(s_ids)
            mask += list(s_mask)
        rows["input_ids"].append(ids)
        rows["loss_mask"].append(mask)
        rows["seq_start_id"].append(starts)
    pq.write_table(pa.table(rows), str(path))


def _seq(n, ends_with_eod=True):
    ids = list(range(10, 10 + n))
    if ends_with_eod:
        ids[-1] = EOD
    mask = [0] * (n // 3) + [1] * (n - n // 3)  # prompt masked, answer supervised
    return ids, mask


def test_whole_bins_index_and_items(tmp_path):
    bins = [[_seq(64), _seq(32)], [_seq(48, ends_with_eod=False)]]
    path = tmp_path / "bins.parquet"
    _write_parquet(path, bins)
    per_seq = _PackedBins(str(path))
    whole = _PackedBins(str(path), whole_bins=True)
    assert len(per_seq) == 3 and len(whole) == 2
    item = whole[0]
    assert item["seq_start_id"] == [0, 64] and len(item["input_ids"]) == 96
    assert np.array_equal(per_seq[1]["input_ids"], item["input_ids"][64:])


def test_packed_bin_sample_matches_per_sequence_path():
    seqs = [_seq(64), _seq(32), _seq(48, ends_with_eod=False)]
    sample = build_packed_bin_sample(seqs, EOD, PAD, MAX_LEN, GRAN)
    parts = []
    for ids, mask in seqs:
        targets = [t if m else IGNORE_INDEX for t, m in zip(ids, mask)]
        parts.append(_shift_and_pad_sequence(list(ids), targets, EOD, PAD, MAX_LEN, GRAN))
    lengths = [p[5] for p in parts]
    # boundaries and per-sequence tensors
    assert sample["cu_seqlens"].tolist() == [0] + list(np.cumsum(lengths))
    assert int(sample["max_seqlen"]) == max(lengths)
    for k, p in enumerate(parts):
        lo, hi = sample["cu_seqlens"][k].item(), sample["cu_seqlens"][k + 1].item()
        assert torch.equal(sample["tokens"][lo:hi], p[0])
        assert torch.equal(sample["labels"][lo:hi], p[1])
        assert torch.equal(sample["loss_mask"][lo:hi], p[2])
        assert torch.equal(sample["position_ids"][lo:hi], torch.arange(hi - lo))
    # a sequence that ends with EOD loses one token to the shift and is padded back to the
    # granularity; one without EOD gets it appended: 64 -> 64, 32 -> 32, 48 (+EOD) -> 48
    assert lengths == [64, 32, 48]
    # prompt tokens are loss-masked but still present as tokens
    assert sample["loss_mask"][: 64 // 3 - 1].sum() == 0 and sample["tokens"][:5].tolist() == [
        10,
        11,
        12,
        13,
        14,
    ]


def test_packed_bin_sample_unpacks_in_scheduler():
    from megatron.core.datasets.data_schedule_utils import _unpack_batch

    # 63 real tokens pad to 64: the scheduler must see original 63 / padded 64, not 64 / 64
    seqs = [_seq(64), _seq(32), _seq(63)]
    sample = build_packed_bin_sample(seqs, EOD, PAD, MAX_LEN, GRAN)
    assert sample["original_seq_lens"].tolist() == [63, 31, 62]
    sample = {k: (v.unsqueeze(0) if v.dim() == 1 else v) for k, v in sample.items()}  # collate dim
    sample["max_seqlen"] = sample["max_seqlen"].reshape(1)
    subs = _unpack_batch([sample])
    assert [int(s["padded_seq_len"]) for s in subs] == [64, 32, 64]
    assert [int(s["original_seq_len"]) for s in subs] == [63, 31, 62]
    assert torch.equal(subs[1]["position_ids"], torch.arange(32))
