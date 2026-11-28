# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mock length configurations must describe valid tokens after the label shift."""

import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from megatron.core.datasets.data_schedule import DefaultDynamicCPScheduler
from megatron.training.arguments import parse_args, validate_args
from megatron.training.datasets.sft_dataset import MockSFTDataset, MockSFTLowLevelDataset
from megatron.training.datasets.varlen_dataset import MockVarlenDataset


def _distribution(length: int) -> dict[str, str | int | float]:
    return {
        "mode": "distribution",
        "type": "lognormal",
        "min_seq_len": length,
        "max_seq_len": length,
        "mean_seq_len": length,
        "lognormal_sigma": 1.1,
        "vocab_size": 65537,
    }


def _config(
    length: int,
    *,
    cp: int = 1,
    dp: int = 1,
    sp: int = 1,
    dynamic_cp: bool = False,
    pad: int | None = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=SimpleNamespace(eod=0, pad=pad, vocab_size=65537),
        sequence_length=length,
        dynamic_context_parallel=dynamic_cp,
        context_parallel_size=cp,
        data_parallel_size=dp,
        sequence_parallel_size=sp,
    )


def _dataset(
    dataset_type: type[MockSFTDataset], low_level: MockSFTLowLevelDataset, config: SimpleNamespace
) -> MockSFTDataset:
    # These CPU-only contracts do not need tokenizer/distributed initialization.
    dataset = dataset_type.__new__(dataset_type)
    dataset.dataset = low_level
    dataset.indices = np.array([0])
    dataset.config = config
    return dataset


def _packing_args(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    *,
    dp: int = 2,
    cp: int = 1,
    sp: int = 1,
    dynamic_cp: bool = True,
    capacity: int | None = None,
) -> Namespace:
    argv = [
        "test_mock_dataset_lengths.py",
        "--num-layers",
        "2",
        "--hidden-size",
        "16",
        "--num-attention-heads",
        "4",
        "--max-position-embeddings",
        str(length),
        "--seq-length",
        str(length),
        "--micro-batch-size",
        "1",
        "--context-parallel-size",
        str(cp),
        "--tensor-model-parallel-size",
        str(sp),
        "--use-varlen-dataset",
        "--mock-data",
        "--calculate-per-token-loss",
    ]
    if dynamic_cp:
        argv.append("--dynamic-context-parallel")
    if sp > 1:
        argv.append("--sequence-parallel")
    if capacity is not None:
        argv.extend(["--max-seqlen-per-dp-cp-rank", str(capacity)])
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", str(dp * cp * sp))
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    # These argument and scheduling contracts do not require physical GPUs.
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 9)
    monkeypatch.setattr("megatron.training.arguments._print_args", lambda *_args: None)
    return validate_args(parse_args())


@pytest.mark.parametrize(
    "ceiling,length,sp,padded_length,capacity",
    [(9, 9, 1, 12, 6), (8, 7, 1, 8, 4), (1, 1, 1, 4, 2), (9, 9, 2, 16, 8)],
)
def test_dynamic_cp_default_capacity_schedules_padded_mock_samples(
    monkeypatch: pytest.MonkeyPatch,
    ceiling: int,
    length: int,
    sp: int,
    padded_length: int,
    capacity: int,
) -> None:
    """CLI defaults must schedule physical samples while preserving every valid target."""
    args = _packing_args(monkeypatch, ceiling, sp=sp)
    assert args.data_parallel_size == 2
    assert args.max_seqlen_per_dp_cp_rank == capacity

    monkeypatch.setattr(MockSFTLowLevelDataset, "size", 2)
    low_level = MockSFTLowLevelDataset(**_distribution(length))
    config = _config(ceiling, dp=2, sp=sp, dynamic_cp=True)
    sample = _dataset(MockVarlenDataset, low_level, config)[0]
    assert sample["original_seq_len"].item() == length
    assert sample["padded_seq_len"].item() == padded_length
    assert sample["tokens"].numel() == sample["labels"].numel() == padded_length
    assert sample["loss_mask"].sum().item() == length
    assert sample["labels"][length - 1].item() == config.tokenizer.eod
    assert sample["loss_mask"][length - 1].item() == 1
    assert torch.count_nonzero(sample["loss_mask"][length:]).item() == 0

    scheduler = DefaultDynamicCPScheduler(
        args.max_seqlen_per_dp_cp_rank,
        cp_size=args.context_parallel_size,
        dp_size=args.data_parallel_size,
        microbatch_group_size_per_vp_stage=None,
    )
    assert scheduler.get_groups_and_subsamples([(0, padded_length), (1, padded_length)]) == [
        [[0], [0]],
        [[1], [1]],
    ]


@pytest.mark.parametrize("sp,capacity", [(1, 5), (2, 7)])
def test_dynamic_cp_rejects_capacity_smaller_than_padded_sample(
    monkeypatch: pytest.MonkeyPatch, sp: int, capacity: int
) -> None:
    """Insufficient explicit capacity must fail before data scheduling begins."""
    with pytest.raises(AssertionError, match="padded single sequence max length"):
        _packing_args(monkeypatch, 9, sp=sp, capacity=capacity)


@pytest.mark.parametrize("sp,capacity", [(1, 6), (2, 8)])
def test_dynamic_cp_accepts_capacity_that_fits_padded_sample(
    monkeypatch: pytest.MonkeyPatch, sp: int, capacity: int
) -> None:
    """A sufficient explicit capacity remains valid across CP and SP alignment."""
    args = _packing_args(monkeypatch, 9, sp=sp, capacity=capacity)
    assert args.max_seqlen_per_dp_cp_rank == capacity


def test_static_cp_capacity_includes_sequence_parallel_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static CP capacity must also cover the dataset's CP-times-SP alignment."""
    args = _packing_args(monkeypatch, 12, dp=1, cp=2, sp=2, dynamic_cp=False)
    assert args.sequence_packing_scheduler == "dp_balanced"
    assert args.max_seqlen_per_dp_cp_rank == 8


@pytest.mark.parametrize("length", [1, 7, 8, 65536])
def test_mock_distribution_returns_configured_content_length(
    monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    """The sampled length counts content tokens before EOD is appended."""
    monkeypatch.setattr(MockSFTLowLevelDataset, "size", 2)
    dataset = MockSFTLowLevelDataset(**_distribution(length))
    assert len(dataset[0]) == length
    assert dataset[0][0] == 1
    assert dataset[0][-1] == length


def test_mock_csv_lengths_are_content_lengths(tmp_path: Path) -> None:
    """CSV lengths follow the same convention as sampled distribution lengths."""
    path = tmp_path / "lengths.csv"
    path.write_text("length\n1\n7\n8\n")
    dataset = MockSFTLowLevelDataset(mode="file", path=str(path), vocab_size=65537)
    assert [len(dataset[i]) for i in range(len(dataset))] == [1, 7, 8]


@pytest.mark.parametrize("length", [1, 2, 7])
def test_mock_verification_returns_full_length_across_documents(
    monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    """Verification mode supplies every requested token, including document wrapping."""
    monkeypatch.setattr(MockSFTLowLevelDataset, "size", 2)
    documents = [np.array([11, 12]), np.array([21])]
    monkeypatch.setattr(
        "megatron.training.datasets.sft_dataset.IndexedDataset", lambda path: documents
    )
    settings = _distribution(length)
    settings.update(mode="verification", data_path="unused")
    dataset = MockSFTLowLevelDataset(**settings)
    expected = np.array([11, 12, 21, 11, 12, 21, 11])[:length]
    np.testing.assert_array_equal(dataset[0], expected)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 65536])
@pytest.mark.parametrize("cp", [1, 2])
def test_varlen_config_length_survives_shift_and_padding(
    monkeypatch: pytest.MonkeyPatch, length: int, cp: int
) -> None:
    """A JSON length n yields n valid targets and masks only the alignment padding."""
    monkeypatch.setattr(MockSFTLowLevelDataset, "size", 2)
    config = _config(length, cp=cp, pad=None)
    config.varlen_mock_dataset_config_json = json.dumps(_distribution(length))
    low_level = MockVarlenDataset.build_low_level_dataset(None, config)
    sample = _dataset(MockVarlenDataset, low_level, config)[0]
    divisor = cp * 2 if cp > 1 else 1
    expected_padded = ((length + divisor - 1) // divisor) * divisor

    assert sample["original_seq_len"].item() == length
    assert sample["padded_seq_len"].item() == expected_padded
    assert sample["tokens"].numel() == expected_padded
    assert sample["loss_mask"].sum().item() == length
    assert sample["labels"][length - 1].item() == config.tokenizer.eod
    assert sample["loss_mask"][length - 1].item() == 1
    assert torch.equal(sample["tokens"][:length], torch.arange(1, length + 1))
    assert torch.equal(sample["labels"][: length - 1], torch.arange(2, length + 1))


@pytest.mark.parametrize("length", [8, 9, 10, 16])
def test_varlen_truncation_preserves_maximum_valid_length(
    monkeypatch: pytest.MonkeyPatch, length: int
) -> None:
    """Clipping retains seq_length content tokens plus the final EOD target."""
    monkeypatch.setattr(MockSFTLowLevelDataset, "size", 2)
    low_level = MockSFTLowLevelDataset(**_distribution(length))
    sample = _dataset(MockVarlenDataset, low_level, _config(8, cp=2))[0]
    assert sample["original_seq_len"].item() == 8
    assert sample["padded_seq_len"].item() == 8
    assert sample["loss_mask"].sum().item() == 8
    assert sample["tokens"].tolist() == list(range(1, 9))
    assert sample["labels"].tolist() == list(range(2, 9)) + [0]


@pytest.mark.parametrize("format", ["thd", "sbhd"])
@pytest.mark.parametrize("length", [1, 7, 8, 9])
@pytest.mark.parametrize("pad", [0, -1])
def test_shared_sft_path_keeps_configured_valid_length(
    monkeypatch: pytest.MonkeyPatch, format: str, length: int, pad: int
) -> None:
    """Both SFT layouts preserve the valid length even when EOD is also padding."""
    monkeypatch.setattr(MockSFTLowLevelDataset, "size", 2)
    low_level = MockSFTLowLevelDataset(**_distribution(length), format=format)
    sample = _dataset(MockSFTDataset, low_level, _config(8, cp=2, pad=pad))[0]
    valid = min(length, 8)
    padded = 8 if format == "sbhd" else ((valid + 3) // 4) * 4
    assert sample["tokens"].numel() == padded
    assert sample["loss_mask"].sum().item() == valid
    assert sample["labels"][valid - 1].item() == 0
    assert sample["loss_mask"][valid - 1].item() == 1
    if format == "thd":
        assert sample["cu_seqlens"].tolist() == [0, padded]
