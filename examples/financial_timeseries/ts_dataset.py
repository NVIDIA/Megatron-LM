# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron-LM data layer for the financial time-series patch tensors from ``preprocessing.py``.

Pieces here
-----------
- ``FinancialWindowDataset`` : map-style dataset over one split's stacked window tensors.
- ``train_valid_test_datasets_provider`` : returns (train, val, test); Megatron builds loaders.
- ``get_batch`` : fetch one micro-batch and TP-broadcast it (mixed dtypes handled per group).

Tensor contract (must match ``preprocessing.py`` exactly)
---------------------------------------------------------
    patches         (B, L, P, F) float32   raw log-return / log-volume per time unit
    inst_id         (B,)         int64     instrument/symbol id
    patch_tod       (B, L)       int64     time-of-day patch index
    target_return   (B, L, H)    float32   future patch log-return at each horizon
    target_volshare (B, L, H)    float32   future log-volume target (legacy key name)
    target_mask     (B, L, H)    bool      generated on-the-fly in get_batch (not stored on disk)

"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset

# tensor_parallel.broadcast_data + the parallel-state helpers (mpu). If `mpu` is not exported
# in your version, use: `from megatron.core import parallel_state as mpu`.
from megatron.core import mpu, tensor_parallel
from megatron.core.datasets.utils import Split

# get_args / print_rank_0 live in megatron.training in current Megatron-LM.
from megatron.training import get_args, print_rank_0

# --------------------------------------------------------------------------- #
# Schema: the ONE place that knows the data layout.                           #
# Grouped by dtype because tensor_parallel.broadcast_data broadcasts a single  #
# dtype per call (it does NOT cast -- the source tensor must already be that    #
# dtype). preprocessing.py already emits float32 / int64, so only the bool      #
# mask needs special handling (shipped as uint8, cast back after broadcast).    #
# --------------------------------------------------------------------------- #
_FLOAT_KEYS = ["patches", "target_return", "target_volshare"]
_INT_KEYS = ["inst_id", "patch_tod"]

# Every key the dataset loads from disk per item.
# target_mask is generated from finite return/volume targets in get_batch.
ALL_KEYS = _FLOAT_KEYS + _INT_KEYS

# Split-file stem -> Megatron Split. Megatron's loader builder reads ``dataset.index_split`` to pick
# the eval micro-batch size and the full-validation sampler for valid/test (see
# megatron/training/datasets/data_samplers.py), exactly as it does for GPTDataset.
SPLIT_BY_NAME = {
    "train": Split.train,
    "val": Split.valid,
    "valid": Split.valid,
    "test": Split.test,
}


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #
class FinancialMarketDataset(Dataset):
    """One split (train|val|test) of pre-materialised, single-symbol windows."""

    def __init__(self, path: Path, name: str, mmap: bool = True) -> None:
        """Validate a split file and record its length without keeping it mapped.

        Args:
            path: Path to ``train.pt``, ``val.pt`` or ``test.pt`` written by ``preprocessing.py``.
            name: Split name (``train``, ``val``/``valid`` or ``test``). Sets ``index_split`` so
                Megatron treats the dataset as that split, and is used in log messages.
            mmap: Memory-map the tensors on first access instead of loading them into RAM.
        """
        self.name = name
        try:
            # Same attribute GPTDataset exposes via MegatronDataset: Megatron's
            # build_pretraining_data_loader uses it for eval batch sizing / full validation.
            self.index_split: Split = SPLIT_BY_NAME[name]
        except KeyError:
            raise ValueError(
                f"unknown split name {name!r}; expected one of {sorted(SPLIT_BY_NAME)}"
            ) from None
        self.path = Path(path)
        self.mmap = mmap
        self._data: dict[str, torch.Tensor] | None = None  # mapped lazily, once per worker

        # Metadata-only open: validate and learn N, then release the mapping.
        blob = self._open()
        missing = [k for k in ALL_KEYS if k not in blob]
        if missing:
            raise KeyError(
                f"{self.path} is missing keys {missing} (did preprocessing.py change?). "
                f"present={sorted(blob)}"
            )
        self._n = int(blob["inst_id"].shape[0])  # .shape is metadata, not data
        for k in ALL_KEYS:  # every tensor must agree on N
            n_k = int(blob[k].shape[0])
            if n_k != self._n:
                raise ValueError(f"{self.path}:{k} has N={n_k} != {self._n}")
        del blob  # drop the handle; workers remap

    def _open(self) -> dict[str, torch.Tensor]:
        try:
            return torch.load(self.path, map_location="cpu", mmap=self.mmap)
        except (TypeError, RuntimeError):  # old torch / legacy non-zip file
            return torch.load(self.path, map_location="cpu")

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._data is None:  # first access in this worker
            self._data = self._open()
        # .clone() copies just this row into private RAM, decoupled from the file mapping.
        return {k: self._data[k][idx].clone() for k in ALL_KEYS}


# --------------------------------------------------------------------------- #
# Dataset provider: Megatron calls this, then builds the DataLoaders itself.   #
# --------------------------------------------------------------------------- #
def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Return ``(train_ds, valid_ds, test_ds)``. Megatron builds the loaders from these.

    ``train_val_test_num_samples`` is how many samples Megatron *wants* (derived from
    ``--train-iters`` * ``--global-batch-size`` and the eval cadence). With
    ``--dataloader-type cyclic`` the sampler reshuffles and cycles your finite window set to
    supply that many, so we return the full datasets and ignore the requested counts.

    Returning ``None`` for a split makes Megatron skip building that loader -- handy because
    your one-month demo can produce empty val/test after the embargo. A missing or empty
    train split is an error when training (--train-iters > 0), so a wrong data path fails here.
    """
    args = get_args()
    root = Path(args.ts_data_path)
    use_mmap = not args.ts_no_mmap

    def load_split(name: str):
        """The split's dataset, or None if its file is missing or has no windows.

        Megatron skips building a loader for a None split.
        """
        path = root / f"{name}.pt"
        required = name == "train" and getattr(args, "train_iters", 0) > 0
        if not path.exists():
            if required:
                raise FileNotFoundError(
                    f"[ts-data] {path} not found: check ts.ts_data_path ({root})"
                )
            print_rank_0(f"[ts-data] {path} not found -> skipping '{name}' split")
            return None
        ds = FinancialMarketDataset(path, name, mmap=use_mmap)
        if len(ds) == 0:
            if required:
                raise ValueError(
                    f"[ts-data] train split {path} has no windows; nothing to train on"
                )
            print_rank_0(f"[ts-data] '{name}' split is empty -> skipping")
            return None
        actual_L = ds[0]["patches"].shape[0]
        if actual_L != args.seq_length:
            raise ValueError(
                f"[ts-data] '{name}' dataset has context_len={actual_L} patches but "
                f"--seq-length={args.seq_length}. Re-run preprocessing with the same "
                f"seq_length, or fix model.seq_length in your config."
            )
        print_rank_0(
            f"[ts-data] {name}: {len(ds)} windows | patches {tuple(ds[0]['patches'].shape)}"
        )
        return ds

    return load_split("train"), load_split("val"), load_split("test")


# Build the dataset independently on every rank (each just opens a .pt). This tells Megatron
# NOT to build-on-rank-0-then-broadcast. Set here so it travels with the function.
train_valid_test_datasets_provider.is_distributed = True


# --------------------------------------------------------------------------- #
# get_batch: fetch one micro-batch, make it identical across the TP group.     #
# --------------------------------------------------------------------------- #
def get_batch(data_iterator: Iterator[dict[str, torch.Tensor]] | None):
    """
    Generate a batch and make it available on the correct ranks.

    data_iterator: yields one CPU micro-batch per call, a dict keyed by ALL_KEYS (Megatron passes
    its
        cyclic DataLoader wrapped in a RerunDataIterator). Only tensor-parallel rank 0 reads it; on
        the
        other ranks it may be None, and they receive the batch through the broadcast.

    It generates the following input/output for the model:

    B: batch size
    L: Patch per window
    P: Time units per patch
    F: Features per timestep
    H: Number of horizons

    - Input:
        - patches   ([B, L, P, F]): patches
        - inst_id   ([B])         : instrument ids
        - patch_tod ([B, L])      : time of the day for each patch

    - Outputs (target):
        - return:   ([B, L, H])   : return values over H horizons
        - volume:   ([B, L, H])   : volume values over H horizons
        - target_mask ([B, L, H]) : same size as volume/return
    """
    if not (mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()):
        return (None,) * 5

    data = None
    if mpu.get_tensor_model_parallel_rank() == 0 and data_iterator is not None:
        data = next(data_iterator)
        for k in _FLOAT_KEYS:
            if k in data and data[k] is not None:
                data[k] = data[k].to(torch.float32)
        for k in _INT_KEYS:
            if k in data and data[k] is not None:
                data[k] = data[k].to(torch.int64)

    fb = tensor_parallel.broadcast_data(_FLOAT_KEYS, data, torch.float32)
    ib = tensor_parallel.broadcast_data(_INT_KEYS, data, torch.int64)

    patches = fb["patches"]
    inst_id = ib["inst_id"]
    patch_tod = ib["patch_tod"]
    targets = {"return": fb["target_return"], "volume": fb["target_volshare"]}
    # target_mask: True where the target exists. Time-unit-horizon datasets store NaN for targets
    # that would
    # cross the day close; older datasets have no NaN, so the mask is all-True as before.
    valid = torch.isfinite(targets["return"]) & torch.isfinite(targets["volume"])
    targets = {k: torch.nan_to_num(v, nan=0.0) for k, v in targets.items()}
    target_mask = valid
    args = get_args()
    patches = apply_market_recent(patches, getattr(args, "market_recent_minutes", 0))
    n_last = getattr(args, "loss_last_positions", 0)
    if n_last:  # only the last n positions count in the loss
        # n_last >= L means every position counts; without the clamp the slice bound
        # goes negative and wraps, masking the FIRST positions instead.
        keep_from = max(target_mask.shape[1] - n_last, 0)
        target_mask[:, :keep_from] = False
    return patches, inst_id, patch_tod, targets, target_mask


def load_target_stats(ts_data_path, forecast_horizons: str, log_tag: str = "ts-norm"):
    """Per-horizon target mean/std from meta.pt, as two [2, H] tensors (return row 0, volume row 1).

    Single loader shared by pretrain_ts.py and evaluate_megatron.py so training and
    evaluation always read/validate the stats identically. Returns (None, None) when
    the path/meta/stats are absent (callers fall back to legacy RevIN target scaling);
    raises on horizon or shape mismatches.
    """
    if not ts_data_path:
        return None, None
    path = Path(ts_data_path) / "meta.pt"
    if not path.exists():
        return None, None
    meta = torch.load(path, map_location="cpu")

    configured_horizons = [int(x) for x in str(forecast_horizons).split(",") if x.strip()]
    dataset_horizons = list(meta.get("horizons", []))
    if dataset_horizons and dataset_horizons != configured_horizons:
        raise ValueError(
            f"dataset horizons {dataset_horizons} do not match configured horizons "
            f"{configured_horizons}"
        )

    stats = meta.get("target_stats")
    if not stats:
        print_rank_0(
            f"[{log_tag}] meta.pt has no target_stats; using legacy per-patch RevIN target scaling"
        )
        return None, None
    mean = torch.stack([stats["return_mean"], stats["volume_mean"]]).float()
    std = torch.stack([stats["return_std"], stats["volume_std"]]).float()
    expected = (2, len(configured_horizons))
    if mean.shape != expected or std.shape != expected:
        raise ValueError(
            f"target stats have shape mean={tuple(mean.shape)}, std={tuple(std.shape)} "
            f"but {expected} is expected; check that preprocessing and training horizons match"
        )
    print_rank_0(
        f"[{log_tag}] loaded per-horizon target stats from {path}: "
        f"return_std={std[0].tolist()} volume_std={std[1].tolist()}"
    )
    return mean, std


def apply_market_recent(patches: torch.Tensor, k: int) -> torch.Tensor:
    """Zero the extra features (index >= 2, e.g. market return) except in the last k time units of
    each window.

    patches: [B, L, P, F]. k <= 0 leaves the batch unchanged (also used by evaluation scripts)."""
    if k <= 0 or patches.shape[-1] <= 2:
        return patches
    B, L, P, F = patches.shape
    flat = patches.reshape(B, L * P, F).clone()
    flat[:, : max(L * P - k, 0), 2:] = 0.0
    return flat.reshape(B, L, P, F)
