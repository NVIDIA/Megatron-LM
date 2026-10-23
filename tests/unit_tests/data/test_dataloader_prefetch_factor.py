# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for --dataloader-prefetch-factor in build_pretraining_data_loader.

The flag is an explicit override of the training loader's prefetch window. Unset, a
loader with a sequence-packing scheduler keeps one step in flight and every other
loader keeps the PyTorch default.
"""

import argparse
import logging
import sys
import time
from types import SimpleNamespace

import pytest
import torch

import megatron.core.num_microbatches_calculator as mb_calculator
from megatron.core.datasets.utils import Split
from megatron.training.arguments import dataloader_prefetch_factor_type, parse_args, validate_args
from megatron.training.datasets import data_samplers

# Keyword arguments build_pretraining_data_loader passed to the DataLoader before
# --dataloader-prefetch-factor existed (plain, non-packing path).
_LEGACY_LOADER_KWARGS = {
    "batch_sampler",
    "num_workers",
    "pin_memory",
    "persistent_workers",
    "worker_init_fn",
    "generator",
}


class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, split=Split.train):
        self.num_samples = num_samples
        if split is not None:
            self.split = split

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor([idx])


class _MarkingDataset(torch.utils.data.Dataset):
    """Index-pure dataset that records, in shared memory, which samples workers produced."""

    def __init__(self, num_samples):
        self.produced = torch.zeros(num_samples, dtype=torch.int32).share_memory_()

    def __len__(self):
        return self.produced.numel()

    def __getitem__(self, idx):
        self.produced[idx] = 1
        return torch.tensor([idx])


class _RecordingDataLoader:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.kwargs = kwargs


def _loader_args(**overrides):
    args = SimpleNamespace(
        dataloader_type='single',
        micro_batch_size=1,
        full_validation=False,
        num_workers=2,
        exit_signal_handler=False,
        use_varlen_dataset=False,
        varlen_sbhd_validation=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        dataloader_prefetch_factor=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _no_calculator():
    raise AssertionError("the microbatch calculator must only be read for 'auto'")


@pytest.fixture
def init_calculator():
    """Install a fresh global microbatch calculator; restore the previous one afterwards."""
    saved = mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR

    def _init(**kwargs):
        mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
        mb_calculator.init_num_microbatches_calculator(rank=0, **kwargs)

    yield _init
    mb_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR = saved


@pytest.fixture
def single_dp_rank(monkeypatch):
    """Build samplers for DP rank 0 of 1 without initializing torch.distributed."""
    monkeypatch.setattr(
        data_samplers,
        "mpu",
        SimpleNamespace(get_data_parallel_rank=lambda: 0, get_data_parallel_world_size=lambda: 1),
    )


@pytest.fixture
def recording_loader(monkeypatch, single_dp_rank):
    monkeypatch.setattr(torch.utils.data, "DataLoader", _RecordingDataLoader)


def _build(monkeypatch, args, dataset):
    monkeypatch.setattr(data_samplers, "get_args", lambda: args)
    return data_samplers.build_pretraining_data_loader(dataset, consumed_samples=0)


# ----------------------------------------------------------------------------
# get_dataloader_prefetch_factor
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("num_microbatches_per_step", [None, 2048])
def test_unset_prefetch_factor_keeps_torch_default(num_workers, num_microbatches_per_step):
    assert (
        data_samplers.get_dataloader_prefetch_factor(None, num_workers, num_microbatches_per_step)
        is None
    )


@pytest.mark.parametrize("num_microbatches_per_step", [None, 8, 2048])
def test_integer_prefetch_factor_is_passed_through(num_microbatches_per_step):
    assert data_samplers.get_dataloader_prefetch_factor(16, 2, num_microbatches_per_step) == 16


@pytest.mark.parametrize(
    ("num_microbatches_per_step", "num_workers", "expected"),
    [
        # GLM-5.2 on 256 GPUs (DP=1, MBS=1, 2 workers): 4K, 128K and 1M.
        (2048, 2, 1025),
        (64, 2, 33),
        (8, 2, 5),
        # Steps that do not divide evenly over the workers round up.
        (7, 2, 5),
        (8, 3, 4),
        (1, 8, 2),
    ],
)
def test_auto_prefetch_factor_sizes_one_step(num_microbatches_per_step, num_workers, expected):
    assert (
        data_samplers.get_dataloader_prefetch_factor('auto', num_workers, num_microbatches_per_step)
        == expected
    )


@pytest.mark.parametrize("num_workers", [1, 2, 3, 8])
@pytest.mark.parametrize("num_microbatches_per_step", range(1, 70))
def test_auto_prefetch_window_covers_a_whole_step(num_microbatches_per_step, num_workers):
    factor = data_samplers.get_dataloader_prefetch_factor(
        'auto', num_workers, num_microbatches_per_step
    )
    # torch keeps factor * num_workers batches outstanding: the window holds a full
    # step even without the extra batch per worker, and is no larger than that.
    assert (factor - 1) * num_workers >= num_microbatches_per_step
    assert (factor - 2) * num_workers < num_microbatches_per_step


def test_auto_prefetch_factor_without_step_size_keeps_torch_default():
    assert data_samplers.get_dataloader_prefetch_factor('auto', 2, None) is None


def test_prefetch_factor_without_workers():
    # 'auto' has nothing to size without worker processes.
    assert data_samplers.get_dataloader_prefetch_factor('auto', 0, 2048) is None
    # An explicit value cannot be honored; torch would reject it in the DataLoader.
    with pytest.raises(ValueError, match="requires --num-workers > 0"):
        data_samplers.get_dataloader_prefetch_factor(4, 0, 2048)


@pytest.mark.parametrize("prefetch_factor", [0, -1, True, 2.5, "8", "fast"])
def test_invalid_prefetch_factor_is_rejected(prefetch_factor):
    with pytest.raises(ValueError, match="positive integer or 'auto'"):
        data_samplers.get_dataloader_prefetch_factor(prefetch_factor, 2, 8)


# ----------------------------------------------------------------------------
# Training-step size from the microbatch calculator
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("global_batch_size", "micro_batch_size", "data_parallel_size", "expected"),
    [(2048, 1, 1, 2048), (64, 1, 1, 64), (8, 1, 1, 8), (64, 2, 4, 8)],
)
def test_train_step_size_matches_num_microbatches(
    init_calculator, global_batch_size, micro_batch_size, data_parallel_size, expected
):
    init_calculator(
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=data_parallel_size,
    )
    assert mb_calculator.get_num_microbatches() == expected
    assert data_samplers._get_train_num_microbatches_per_step() == expected


def test_train_step_size_uses_decreased_running_batch(init_calculator):
    # 32 samples over 8 x 3 rounds down to 24, i.e. one microbatch per step.
    init_calculator(
        global_batch_size=32,
        micro_batch_size=8,
        data_parallel_size=3,
        decrease_batch_size_if_needed=True,
    )
    assert data_samplers._get_train_num_microbatches_per_step() == 1


def test_train_step_size_covers_largest_scheduled_batch(init_calculator):
    # The schedule starts at 4 microbatches per step and grows to 16 later.
    init_calculator(
        global_batch_size=None,
        micro_batch_size=1,
        data_parallel_size=1,
        step_batch_size_schedule="0:4 100:16",
    )
    assert mb_calculator.get_num_microbatches() == 4
    assert data_samplers._get_train_num_microbatches_per_step() == 16


# ----------------------------------------------------------------------------
# DataLoader construction (DataLoader monkeypatched to record its arguments)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("has_attribute", [False, True])
def test_unset_flag_builds_the_legacy_dataloader(monkeypatch, recording_loader, has_attribute):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _loader_args()
    if not has_attribute:
        # Args from YAML configs or older checkpoints predate the flag.
        del args.dataloader_prefetch_factor
    loader = _build(monkeypatch, args, _SplitDataset(64))
    assert set(loader.kwargs) == _LEGACY_LOADER_KWARGS
    assert loader.kwargs["num_workers"] == 2
    assert loader.kwargs["persistent_workers"] is True


@pytest.mark.parametrize(
    ("global_batch_size", "micro_batch_size", "num_workers", "expected"),
    [(2048, 1, 2, 1025), (64, 1, 2, 33), (8, 1, 2, 5), (64, 2, 3, 12)],
)
def test_auto_sizes_the_packing_training_loader(
    monkeypatch,
    recording_loader,
    init_calculator,
    global_batch_size,
    micro_batch_size,
    num_workers,
    expected,
):
    init_calculator(
        global_batch_size=global_batch_size, micro_batch_size=micro_batch_size, data_parallel_size=1
    )
    args = _loader_args(
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
        dynamic_context_parallel=True,
        sequence_packing_scheduler='default_dynamic_cp',
        dataloader_prefetch_factor='auto',
    )
    loader = _build(monkeypatch, args, _SplitDataset(4 * global_batch_size))
    assert loader.kwargs["prefetch_factor"] == expected
    # The packing path keeps its identity collate next to the new argument.
    assert loader.kwargs["collate_fn"]([1, 2]) == [1, 2]
    assert set(loader.kwargs) == _LEGACY_LOADER_KWARGS | {"collate_fn", "prefetch_factor"}
    # One batch is one microbatch of micro_batch_size samples.
    assert len(next(iter(loader.kwargs["batch_sampler"]))) == micro_batch_size


def test_auto_treats_a_loader_without_split_as_training(
    monkeypatch, recording_loader, init_calculator
):
    init_calculator(global_batch_size=64, micro_batch_size=1, data_parallel_size=1)
    args = _loader_args(dataloader_prefetch_factor='auto')
    loader = _build(monkeypatch, args, _SplitDataset(256, split=None))
    assert loader.kwargs["prefetch_factor"] == 33


@pytest.mark.parametrize("prefetch_factor", ['auto', 16])
@pytest.mark.parametrize("full_validation", [False, True])
@pytest.mark.parametrize("split", [Split.valid, Split.test])
def test_eval_loaders_keep_torch_default(
    monkeypatch, recording_loader, split, full_validation, prefetch_factor
):
    # Persistent eval workers would fill the window at setup and hold it between
    # evaluations, so neither form of the flag reaches validation/test loaders.
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _loader_args(dataloader_prefetch_factor=prefetch_factor, full_validation=full_validation)
    loader = _build(monkeypatch, args, _SplitDataset(64, split=split))
    assert set(loader.kwargs) == _LEGACY_LOADER_KWARGS


@pytest.mark.parametrize("split", [Split.train, None])
def test_integer_prefetch_factor_applies_to_the_training_loader(
    monkeypatch, recording_loader, split
):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _loader_args(dataloader_prefetch_factor=16)
    loader = _build(monkeypatch, args, _SplitDataset(64, split=split))
    assert loader.kwargs["prefetch_factor"] == 16


def test_prefetch_factor_with_zero_workers(monkeypatch, recording_loader):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _loader_args(num_workers=0, dataloader_prefetch_factor='auto')
    loader = _build(monkeypatch, args, _SplitDataset(64))
    assert "prefetch_factor" not in loader.kwargs
    assert loader.kwargs["persistent_workers"] is False

    args = _loader_args(num_workers=0, dataloader_prefetch_factor=4)
    with pytest.raises(ValueError, match="requires --num-workers > 0"):
        _build(monkeypatch, args, _SplitDataset(64))


# ----------------------------------------------------------------------------
# The flag overrides the default window of sequence-packing schedulers
# ----------------------------------------------------------------------------


def _packing_args(**overrides):
    packing = dict(
        global_batch_size=2048,
        dynamic_context_parallel=True,
        sequence_packing_scheduler='default_dynamic_cp',
    )
    return _loader_args(**{**packing, **overrides})


@pytest.mark.parametrize(
    ("global_batch_size", "micro_batch_size", "num_workers", "expected"),
    [
        # max(2, ceil(items per step / num_workers)); GLM-5.2 4K/128K/1M on 256 GPUs (DP=1).
        (2048, 1, 2, 1024),
        (64, 1, 2, 32),
        (8, 1, 2, 4),
        # Steps that do not divide evenly over the workers round up.
        (64, 2, 3, 11),
        # Never below the PyTorch default.
        (2, 1, 2, 2),
    ],
)
def test_unset_flag_keeps_the_packing_default_window(
    monkeypatch,
    recording_loader,
    caplog,
    global_batch_size,
    micro_batch_size,
    num_workers,
    expected,
):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _packing_args(
        global_batch_size=global_batch_size,
        micro_batch_size=micro_batch_size,
        num_workers=num_workers,
    )
    with caplog.at_level(logging.INFO, logger=data_samplers.logger.name):
        loader = _build(monkeypatch, args, _SplitDataset(4 * global_batch_size))
    assert loader.kwargs["prefetch_factor"] == expected
    assert set(loader.kwargs) == _LEGACY_LOADER_KWARGS | {"collate_fn", "prefetch_factor"}
    # The resolved window is logged on the default path too.
    assert (
        f"dataloader prefetch_factor {expected} (sequence-packing scheduler default" in caplog.text
    )


@pytest.mark.parametrize("prefetch_factor", [1, 2, 16])
def test_integer_prefetch_factor_overrides_the_packing_default(
    monkeypatch, recording_loader, caplog, prefetch_factor
):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _packing_args(dataloader_prefetch_factor=prefetch_factor)
    with caplog.at_level(logging.INFO, logger=data_samplers.logger.name):
        loader = _build(monkeypatch, args, _SplitDataset(4 * 2048))
    # 2 restores the PyTorch window that the packing default (1024 here) replaces.
    assert loader.kwargs["prefetch_factor"] == prefetch_factor
    assert (
        f"dataloader prefetch_factor {prefetch_factor} "
        f"(--dataloader-prefetch-factor {prefetch_factor};" in caplog.text
    )


def test_auto_prefetch_factor_overrides_the_packing_default(
    monkeypatch, recording_loader, init_calculator
):
    init_calculator(global_batch_size=2048, micro_batch_size=1, data_parallel_size=1)
    loader = _build(
        monkeypatch, _packing_args(dataloader_prefetch_factor='auto'), _SplitDataset(4 * 2048)
    )
    # ceil(2048 / 2) + 1, not the packing default's 1024.
    assert loader.kwargs["prefetch_factor"] == 1025


@pytest.mark.parametrize("prefetch_factor", [None, 'auto', 16])
@pytest.mark.parametrize("split", [Split.valid, Split.test])
def test_packing_eval_loaders_keep_torch_default(
    monkeypatch, recording_loader, split, prefetch_factor
):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    args = _packing_args(dataloader_prefetch_factor=prefetch_factor)
    loader = _build(monkeypatch, args, _SplitDataset(64, split=split))
    assert "prefetch_factor" not in loader.kwargs


def test_packing_loader_without_workers_has_no_window(monkeypatch, recording_loader):
    monkeypatch.setattr(data_samplers, "get_num_microbatches", _no_calculator)
    loader = _build(monkeypatch, _packing_args(num_workers=0), _SplitDataset(64))
    assert "prefetch_factor" not in loader.kwargs
    assert loader.kwargs["persistent_workers"] is False


# ----------------------------------------------------------------------------
# Real DataLoader: 'auto' produces the whole next step ahead, in the same order.
# ----------------------------------------------------------------------------


def _wait_until(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


@pytest.mark.parametrize("prefetch_factor", [None, 'auto'])
def test_auto_prefetch_produces_the_next_step_ahead(
    monkeypatch, single_dp_rank, init_calculator, prefetch_factor
):
    num_microbatches, num_workers, num_steps = 8, 2, 3
    init_calculator(global_batch_size=num_microbatches, micro_batch_size=1, data_parallel_size=1)
    dataset = _MarkingDataset(num_steps * num_microbatches)
    args = _loader_args(num_workers=num_workers, dataloader_prefetch_factor=prefetch_factor)
    loader = _build(monkeypatch, args, dataset)
    data_iter = None
    try:
        # torch's default is 2; 'auto' is ceil(8 / 2) + 1.
        expected_factor = 2 if prefetch_factor is None else 5
        assert loader.prefetch_factor == expected_factor
        window = expected_factor * num_workers

        sampler_batches = list(loader.batch_sampler)
        assert len(sampler_batches) == num_steps * num_microbatches
        next_step = [
            idx
            for batch in sampler_batches[num_microbatches : 2 * num_microbatches]
            for idx in batch
        ]

        data_iter = iter(loader)
        consumed = [next(data_iter).flatten().tolist() for _ in range(num_microbatches)]

        def produced(indices):
            return all(int(dataset.produced[idx]) == 1 for idx in indices)

        if prefetch_factor == 'auto':
            # Every sample of the next step is produced before the step asks for it.
            assert _wait_until(lambda: produced(next_step))
        else:
            # The default window stops after 2 x num_workers batches of the next step.
            assert _wait_until(lambda: produced(next_step[:window]))
            time.sleep(0.5)
            assert not any(int(dataset.produced[idx]) for idx in next_step[window:])

        consumed += [next(data_iter).flatten().tolist() for _ in range(2 * num_microbatches)]
        # Prefetching does not change the sample order.
        assert consumed == sampler_batches
    finally:
        del data_iter, loader


# ----------------------------------------------------------------------------
# Argument parsing and validation
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), [("auto", "auto"), ("8", 8), (8, 8), (None, None)])
def test_prefetch_factor_type_accepts(value, expected):
    assert dataloader_prefetch_factor_type(value) == expected


@pytest.mark.parametrize("value", ["0", "-2", "2.5", "fast", "AUTO", 0, 2.5, True])
def test_prefetch_factor_type_rejects(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer or 'auto'"):
        dataloader_prefetch_factor_type(value)


@pytest.mark.parametrize(
    ("cli", "expected"),
    [
        ([], None),
        (["--dataloader-prefetch-factor", "auto"], "auto"),
        (["--dataloader-prefetch-factor", "64"], 64),
    ],
)
def test_parse_args_prefetch_factor(monkeypatch, cli, expected):
    monkeypatch.setattr(sys, "argv", ["test_data_samplers.py", *cli])
    assert parse_args().dataloader_prefetch_factor == expected


def test_parse_args_rejects_zero_prefetch_factor(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test_data_samplers.py", "--dataloader-prefetch-factor", "0"])
    with pytest.raises(SystemExit):
        parse_args()


def _parse_minimal_training_args(monkeypatch, cli):
    monkeypatch.setattr(sys, "argv", ["test_data_samplers.py", *cli])
    args = parse_args()
    args.num_layers = 2
    args.vocab_size = 256
    args.hidden_size = 64
    args.num_attention_heads = 4
    args.max_position_embeddings = 128
    args.seq_length = 128
    args.micro_batch_size = 1
    args.train_iters = 1
    args.lr = 1e-4
    args.tokenizer_type = 'NullTokenizer'
    args.calculate_per_token_loss = True
    return args


def test_validate_args_rejects_integer_prefetch_factor_without_workers(monkeypatch):
    args = _parse_minimal_training_args(
        monkeypatch, ["--num-workers", "0", "--dataloader-prefetch-factor", "4"]
    )
    with pytest.raises(ValueError, match="requires --num-workers > 0"):
        validate_args(args)


@pytest.mark.parametrize(
    ("cli", "expected"),
    [
        # 'auto' is a no-op without workers rather than an error.
        (["--num-workers", "0", "--dataloader-prefetch-factor", "auto"], "auto"),
        (["--num-workers", "2", "--dataloader-prefetch-factor", "4"], 4),
        (["--num-workers", "0"], None),
    ],
)
def test_validate_args_accepts_prefetch_factor(monkeypatch, cli, expected):
    args = _parse_minimal_training_args(monkeypatch, cli)
    assert validate_args(args).dataloader_prefetch_factor == expected


@pytest.mark.parametrize(
    ("cli", "warns"),
    [
        # Captures in torch's default capture_error_mode='global'.
        (["--dataloader-prefetch-factor", "auto", "--optimizer-cuda-graph"], True),
        (["--dataloader-prefetch-factor", "4", "--cuda-graph-impl", "local"], True),
        # The one-step default window of a sequence-packing scheduler widens it as well
        # once a step needs more than two items per worker: GBS 8 / 2 workers -> 4.
        (
            [
                "--sequence-packing-scheduler",
                "dp_balanced",
                "--global-batch-size",
                "8",
                "--optimizer-cuda-graph",
            ],
            True,
        ),
        # PyTorch's own window (2 per worker), with or without a packing scheduler.
        (["--dataloader-prefetch-factor", "2", "--optimizer-cuda-graph"], False),
        (
            [
                "--sequence-packing-scheduler",
                "dp_balanced",
                "--dataloader-prefetch-factor",
                "2",
                "--optimizer-cuda-graph",
            ],
            False,
        ),
        # Full-iteration graphs capture in 'thread_local' mode.
        (
            [
                "--dataloader-prefetch-factor",
                "auto",
                "--cuda-graph-impl",
                "full_iteration",
                "--no-check-for-nan-in-loss-and-grad",
            ],
            False,
        ),
        # No capture, the flag unset, or 'auto' without workers (no pin-memory thread).
        (["--dataloader-prefetch-factor", "auto"], False),
        (["--sequence-packing-scheduler", "dp_balanced"], False),
        (["--optimizer-cuda-graph"], False),
        (
            [
                "--num-workers",
                "0",
                "--dataloader-prefetch-factor",
                "auto",
                "--optimizer-cuda-graph",
            ],
            False,
        ),
        (
            [
                "--num-workers",
                "0",
                "--sequence-packing-scheduler",
                "dp_balanced",
                "--optimizer-cuda-graph",
            ],
            False,
        ),
        # The packing default equals PyTorch's window up to two items per worker per step
        # (GBS 4 / 2 workers), where the remedy would be a no-op.
        (
            [
                "--sequence-packing-scheduler",
                "dp_balanced",
                "--global-batch-size",
                "4",
                "--optimizer-cuda-graph",
            ],
            False,
        ),
    ],
)
def test_validate_args_warns_about_global_mode_cuda_graph_capture(monkeypatch, cli, warns):
    messages = []
    monkeypatch.setattr(
        "megatron.training.arguments.warn_rank_0",
        lambda message, rank=None: messages.append(message),
    )
    validate_args(_parse_minimal_training_args(monkeypatch, cli))
    capture_warnings = [message for message in messages if "pin-memory thread" in message]
    assert bool(capture_warnings) == warns
    # The warning names the remedy: PyTorch's window.
    assert all("--dataloader-prefetch-factor 2" in message for message in capture_warnings)
