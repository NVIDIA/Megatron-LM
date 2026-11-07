# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Corpus dumps: one whole source dataset, scored once by a frozen teacher.

A corpus source is generic, like the files it is derived from: complete, unweighted and
unsplit. Weights, train/validation choices and repetition belong to blends of sources.
A corpus dump is an ordinary V3 dump with three restrictions, so it reuses the saver,
files, parallel range jobs and resume unchanged:

- one unweighted source (one ``--data-path`` prefix or SFT file), written under
  ``<corpus>/<source>/``;
- all of it, exactly once, as Megatron builds it (GPT windows, SFT rows), with no
  ``--split`` partition;
- a seeded permutation, so every block is a random sample of the source.

The epoch is padded to a whole number of global batches by repeating its first
(permuted) samples; global positions at or beyond ``num_samples`` are filler, written
but never part of the corpus.
"""

import os
from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist

# Placeholder budget until the epoch length is known (set once the dataset is built).
_UNTIL_EPOCH_END = 1 << 40


def validate_corpus_dump(args: Any) -> None:
    """Check and apply ``--kd-corpus-dump`` settings (idempotent)."""
    name = getattr(args, "kd_corpus_dump", None)
    if not name:
        return
    if not args.logits_save_dir:
        raise ValueError("--kd-corpus-dump requires --logits-save-dir (the corpus directory)")
    if os.sep in name or name in ("", ".", ".."):
        raise ValueError(f"--kd-corpus-dump source name {name!r} must be a plain directory name")
    if not getattr(args, "freeze_all_layers", False):
        raise ValueError("Corpus dumps require a frozen teacher (--freeze-all-layers)")
    paths = getattr(args, "data_path", None) or []
    if not getattr(args, "mock_data", False) and len(paths) != 1:
        raise ValueError(
            "Corpus dumps take exactly one unweighted --data-path prefix (or SFT file); "
            "dump each source with its own job"
        )
    for flag in ("train_data_path", "valid_data_path", "test_data_path", "data_args_path"):
        if getattr(args, flag, None):
            raise ValueError(f"Corpus dumps do not support --{flag.replace('_', '-')}")
    if getattr(args, "per_split_data_args_path", None):
        raise ValueError("Corpus dumps do not support --per-split-data-args-path")
    if getattr(args, "phase_transition_iterations", None):
        raise ValueError("Corpus dumps do not support dataset phase transitions")
    if getattr(args, "eval_iters", 0):
        raise ValueError(
            "Corpus dumps score the whole source and no validation; use --eval-iters 0"
        )
    if getattr(args, "kd_corpus_root", None) is None:
        if getattr(args, "train_samples", None):
            raise ValueError("Corpus dumps size themselves to one epoch; omit --train-samples")
        args.kd_corpus_root = args.logits_save_dir
        args.logits_save_dir = os.path.join(args.logits_save_dir, name)
        args.train_iters = _UNTIL_EPOCH_END // args.global_batch_size
    # The whole source is the training part: a corpus source is never split.
    args.split = "100,0,0"
    args.logits_save_inputs = True


def validate_corpus_blend(args: Any) -> None:
    """Check and apply ``--kd-corpus-blend``: corpus loading reuses the replay path."""
    path = getattr(args, "kd_corpus_blend", None)
    if not path:
        return
    from .format.blend import load_config

    if getattr(args, "logits_save_inputs", False) or getattr(args, "kd_corpus_dump", None):
        raise ValueError("A run cannot both dump and load offline KD v3 data")
    if getattr(args, "logits_load_shuffle_shards", None) is not None:
        raise ValueError("A corpus blend is ordered by its plan; omit --logits-load-shuffle-shards")
    config = load_config(path)
    args.logits_load_inputs = True
    args.logits_load_dir = args.logits_load_dir or config["corpus"]


def epoch_length(dataset: Any) -> int:
    """Samples in one epoch of a Megatron dataset built with ``num_samples=None``.

    ``GPTDataset`` then builds exactly one epoch of windows; ``SFTDataset`` reports its
    requested count, so its length is set to the number of rows it indexes.
    """
    if getattr(dataset, "num_samples", 0) is None and hasattr(dataset, "indices"):
        dataset.num_samples = len(dataset.indices)
    return len(dataset)


class CorpusEpoch(torch.utils.data.Dataset):
    """One seeded permutation of a dataset's epoch, padded to whole global batches."""

    def __init__(self, dataset: Any, *, seed: int, global_batch_size: int):
        self.dataset = dataset
        self.num_samples = epoch_length(dataset)
        if self.num_samples < 1:
            raise ValueError("Corpus dump source has no samples")
        self.order = np.random.RandomState(seed).permutation(self.num_samples)
        self.padded = -(-self.num_samples // global_batch_size) * global_batch_size

    def __len__(self) -> int:
        return self.padded

    def __getitem__(self, index: int) -> Any:
        # Filler positions repeat the first samples; they are never part of the corpus.
        return self.dataset[int(self.order[index % self.num_samples])]


def build_corpus_dump_datasets(args: Any, provider: Callable) -> tuple:
    """Build the whole source for exactly one epoch and wrap it (data-building ranks)."""
    # A single unweighted prefix, all in the training part, with size None builds one
    # full epoch of the whole source.
    dataset = provider([None, None, None])[0]
    if dataset is None:
        raise ValueError("Corpus dump source has no samples")
    epoch = CorpusEpoch(dataset, seed=args.seed, global_batch_size=args.global_batch_size)
    args.kd_corpus_samples = epoch.num_samples
    return epoch, None, None


def agree_corpus_epoch(args: Any) -> None:
    """Share the epoch length with every rank and size the run to it (collective)."""
    length = torch.tensor(
        [getattr(args, "kd_corpus_samples", 0)],
        dtype=torch.long,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if dist.is_initialized():
        dist.all_reduce(length, op=dist.ReduceOp.MAX)
    args.kd_corpus_samples = int(length.item())
    args.train_iters = -(-args.kd_corpus_samples // args.global_batch_size)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"Offline KD corpus dump: {args.kd_corpus_samples} samples of "
            f"{args.kd_corpus_dump} in {args.train_iters} iterations",
            flush=True,
        )
