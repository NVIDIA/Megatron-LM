# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for TeacherTarDataset's document-aware CP reslicing (CP+packing fix).

Covers the loader side of the fix: persisted cu_seqlens_padded flowing from
a v2 tar payload through TeacherTarDataset to document-aware
slice_tensor_for_cp_rank calls, and the guard that refuses to silently
reshard packed data under CP>1 when a payload predates the fix.
"""

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.training.distillation import cached_logits_loss
from megatron.training.distillation import utils as distillation_utils
from megatron.training.distillation.cached_logits_loss import make_teacher_tar_dataset
from megatron.training.distillation.utils import (
    LOGPROBS_TAR_MEMBER_SUFFIX,
    META_TAR_MEMBER,
    slice_tensor_for_cp_rank,
    v2_pack_indices,
)

zstandard = pytest.importorskip("zstandard")


def _torch_save_bytes(payload) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _write_v2_shard(
    root: Path,
    *,
    dp_rank: int = 0,
    num_samples: int,
    seq_len: int,
    cu_seqlens_padded,
    cp_size_save=None,
) -> None:
    """Write a single v2 tar shard: one saved iteration of num_samples
    samples, each with a real (seq_len,) sequence whose values encode
    global sequence position (values[s, b, 0] == s), for CP-reslicing
    correctness checks.
    """
    values = (
        torch.arange(seq_len)
        .float()
        .view(seq_len, 1, 1)
        .expand(seq_len, num_samples, 1)
        .contiguous()
    )
    indices = torch.zeros(seq_len, num_samples, 1, dtype=torch.long)
    indices_low, bit_17 = v2_pack_indices(indices)
    payload = {
        "values": values,
        "indices_low": indices_low,
        "bit_17": bit_17,
        "format_version": 2,
        "cu_seqlens_padded": cu_seqlens_padded,
    }
    metadata = {
        "saver": {"format_version": 2, "mbs_save": 1, "dp_size_save": 1, "gbs_save": num_samples}
    }
    if cp_size_save is not None:
        metadata["saver"]["cp_size_save"] = cp_size_save
    compressor = zstandard.ZstdCompressor(level=1)
    tar_path = root / f"dp{dp_rank}__0-{num_samples}.tar"
    with tarfile.open(tar_path, "w") as tar:
        meta_bytes = json.dumps(metadata).encode("utf-8")
        info = tarfile.TarInfo(META_TAR_MEMBER)
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

        member_name = f"0-{num_samples}{LOGPROBS_TAR_MEMBER_SUFFIX}"
        compressed = compressor.compress(_torch_save_bytes(payload))
        info = tarfile.TarInfo(member_name)
        info.size = len(compressed)
        tar.addfile(info, io.BytesIO(compressed))


def _load_args(monkeypatch, *, sft: bool, num_samples: int):
    args = SimpleNamespace(
        micro_batch_size=1,
        global_batch_size=num_samples,
        sft=sft,
        dataloader_inter_document_masking=False,
    )
    monkeypatch.setattr(cached_logits_loss, "get_args", lambda: args)
    # is_packed_sequence_run() reads get_args() bound inside utils.py itself,
    # not cached_logits_loss's binding -- patch both so TeacherTarDataset's
    # _requires_cu_seqlens sees the same args.
    monkeypatch.setattr(distillation_utils, "get_args", lambda: args)


# ---------------------------------------------------------------------------
# Document-aware CP reslicing round trip through the real tar/dataset stack
# ---------------------------------------------------------------------------


def test_teacher_tar_dataset_document_aware_cp_reslicing(tmp_path, monkeypatch):
    seq_len = 16
    cp_size = 2
    cu_seqlens_padded = torch.tensor([[0, seq_len]])  # one sample, single document
    _write_v2_shard(tmp_path, num_samples=1, seq_len=seq_len, cu_seqlens_padded=cu_seqlens_padded)
    _load_args(monkeypatch, sft=True, num_samples=1)

    full = torch.arange(seq_len).float().view(seq_len, 1, 1)
    for cp_rank in range(cp_size):
        dataset = make_teacher_tar_dataset(
            str(tmp_path), cp_rank=cp_rank, cp_size=cp_size, dp_rank=0, dp_size=1, ignore_hash=True
        )
        loaded = list(dataset)
        assert len(loaded) == 1
        _, values_list, _ = loaded[0]
        assert len(values_list) == 1

        expected = slice_tensor_for_cp_rank(full, cp_rank, cp_size, cu_seqlens=cu_seqlens_padded[0])
        assert torch.equal(values_list[0], expected)


def test_teacher_tar_dataset_document_aware_cp_reslicing_multi_document(tmp_path, monkeypatch):
    seq_len = 24
    cp_size = 2
    cu_seqlens_padded = torch.tensor([[0, 8, seq_len]])  # one sample, two documents
    _write_v2_shard(tmp_path, num_samples=1, seq_len=seq_len, cu_seqlens_padded=cu_seqlens_padded)
    _load_args(monkeypatch, sft=True, num_samples=1)

    full = torch.arange(seq_len).float().view(seq_len, 1, 1)
    all_positions = set()
    for cp_rank in range(cp_size):
        dataset = make_teacher_tar_dataset(
            str(tmp_path), cp_rank=cp_rank, cp_size=cp_size, dp_rank=0, dp_size=1, ignore_hash=True
        )
        _, values_list, _ = list(dataset)[0]

        expected = slice_tensor_for_cp_rank(full, cp_rank, cp_size, cu_seqlens=cu_seqlens_padded[0])
        assert torch.equal(values_list[0], expected)
        all_positions.update(values_list[0].squeeze().long().tolist())

    # Every CP rank's slice together must reconstruct the full sequence with
    # no overlap or gap -- the property that was broken by the original bug.
    assert all_positions == set(range(seq_len))


def test_teacher_tar_dataset_cp_size_one_ignores_cu_seqlens(tmp_path, monkeypatch):
    seq_len = 24
    cu_seqlens_padded = torch.tensor([[0, 8, seq_len]])
    _write_v2_shard(tmp_path, num_samples=1, seq_len=seq_len, cu_seqlens_padded=cu_seqlens_padded)
    _load_args(monkeypatch, sft=True, num_samples=1)

    dataset = make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=1, dp_rank=0, dp_size=1, ignore_hash=True
    )
    _, values_list, _ = list(dataset)[0]
    assert torch.equal(values_list[0], torch.arange(seq_len).float().view(seq_len, 1, 1))


# ---------------------------------------------------------------------------
# Backward-compat guard: pre-fix payloads (no cu_seqlens_padded) under CP>1
# ---------------------------------------------------------------------------


def test_missing_cu_seqlens_raises_when_load_run_needs_it(tmp_path, monkeypatch):
    seq_len = 16
    _write_v2_shard(tmp_path, num_samples=1, seq_len=seq_len, cu_seqlens_padded=None)
    _load_args(monkeypatch, sft=True, num_samples=1)

    dataset = make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=2, dp_rank=0, dp_size=1, ignore_hash=True
    )
    with pytest.raises(ValueError, match="cu_seqlens_padded"):
        list(dataset)


def test_missing_cu_seqlens_is_harmless_when_cp_size_one(tmp_path, monkeypatch):
    seq_len = 16
    _write_v2_shard(tmp_path, num_samples=1, seq_len=seq_len, cu_seqlens_padded=None)
    _load_args(monkeypatch, sft=True, num_samples=1)

    dataset = make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=1, dp_rank=0, dp_size=1, ignore_hash=True
    )
    loaded = list(dataset)  # should not raise
    assert len(loaded) == 1


def test_missing_cu_seqlens_is_harmless_when_load_run_not_packed(tmp_path, monkeypatch):
    seq_len = 16
    _write_v2_shard(tmp_path, num_samples=1, seq_len=seq_len, cu_seqlens_padded=None)
    _load_args(monkeypatch, sft=False, num_samples=1)

    dataset = make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=2, dp_rank=0, dp_size=1, ignore_hash=True
    )
    loaded = list(dataset)  # should not raise -- CP=2 but no packing intent
    assert len(loaded) == 1


# ---------------------------------------------------------------------------
# CP-size compatibility: saved CP vs. load CP for packed (--sft) sequences
# ---------------------------------------------------------------------------


def test_load_cp_larger_and_not_dividing_save_cp_raises_early(tmp_path, monkeypatch):
    """Saved at CP=4 (documents padded to multiples of 8), loading at CP=16
    (needs multiples of 32) is not guaranteed safe -- must raise at dataset
    construction time, before any per-document ValueError surfaces."""
    seq_len = 16
    cu_seqlens_padded = torch.tensor([[0, seq_len]])
    _write_v2_shard(
        tmp_path,
        num_samples=1,
        seq_len=seq_len,
        cu_seqlens_padded=cu_seqlens_padded,
        cp_size_save=4,
    )
    _load_args(monkeypatch, sft=True, num_samples=1)

    with pytest.raises(ValueError, match="context-parallel"):
        make_teacher_tar_dataset(
            str(tmp_path), cp_rank=0, cp_size=16, dp_rank=0, dp_size=1, ignore_hash=True
        )


def test_load_cp_dividing_save_cp_does_not_raise(tmp_path, monkeypatch):
    """Saved at CP=16, loading at CP=4 (a divisor of 16): mechanically safe,
    no error expected from the early check."""
    seq_len = 16
    cu_seqlens_padded = torch.tensor([[0, seq_len]])
    _write_v2_shard(
        tmp_path,
        num_samples=1,
        seq_len=seq_len,
        cu_seqlens_padded=cu_seqlens_padded,
        cp_size_save=16,
    )
    _load_args(monkeypatch, sft=True, num_samples=1)

    make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=4, dp_rank=0, dp_size=1, ignore_hash=True
    )  # should not raise


def test_load_cp_smaller_than_save_cp_warns_about_truncation(tmp_path, monkeypatch):
    seq_len = 16
    cu_seqlens_padded = torch.tensor([[0, seq_len]])
    _write_v2_shard(
        tmp_path,
        num_samples=1,
        seq_len=seq_len,
        cu_seqlens_padded=cu_seqlens_padded,
        cp_size_save=16,
    )
    _load_args(monkeypatch, sft=True, num_samples=1)

    with pytest.warns(UserWarning, match="truncat"):
        make_teacher_tar_dataset(
            str(tmp_path), cp_rank=0, cp_size=4, dp_rank=0, dp_size=1, ignore_hash=True
        )


def test_load_cp_equal_to_save_cp_does_not_warn_or_raise(tmp_path, monkeypatch, recwarn):
    seq_len = 16
    cu_seqlens_padded = torch.tensor([[0, seq_len]])
    _write_v2_shard(
        tmp_path,
        num_samples=1,
        seq_len=seq_len,
        cu_seqlens_padded=cu_seqlens_padded,
        cp_size_save=4,
    )
    _load_args(monkeypatch, sft=True, num_samples=1)

    make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=4, dp_rank=0, dp_size=1, ignore_hash=True
    )
    assert len(recwarn) == 0


def test_missing_cp_size_save_skips_the_early_check(tmp_path, monkeypatch):
    """Old-format metadata (no cp_size_save) can't be compared -- falls
    through to the missing-cu_seqlens_padded guard instead, not this one."""
    seq_len = 16
    cu_seqlens_padded = torch.tensor([[0, seq_len]])
    _write_v2_shard(
        tmp_path,
        num_samples=1,
        seq_len=seq_len,
        cu_seqlens_padded=cu_seqlens_padded,
        cp_size_save=None,
    )
    _load_args(monkeypatch, sft=True, num_samples=1)

    make_teacher_tar_dataset(
        str(tmp_path), cp_rank=0, cp_size=16, dp_rank=0, dp_size=1, ignore_hash=True
    )  # should not raise from the cp_size_save check (metadata doesn't have it)
