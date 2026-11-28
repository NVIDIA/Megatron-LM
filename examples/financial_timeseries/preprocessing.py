# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Preprocess a parquet store of N-time-unit bars into training-ready tensors.

Reads the hive-partitioned OHLCV parquet under ``<data-dir>``
and materialises the tensors the Qwen-style decoder-only TS transformer recipe asks for,
writing one ``.pt`` per split (train / val / test) plus a ``meta.pt`` describing them.

Input: parquet files (any number, any partitioning), one row per bar:
    Symbol     str            instrument id
    Date       date/datetime  trading day (rows are grouped by Symbol, Date)
    Timestamp  datetime       bar time; sorts rows within a day
    Close      float          last price of the bar (> 0)
    Volume     float          volume traded in the bar
    <feature>  float          optional extra columns named in --features (e.g. mkt_return), passed
    through as-is
Rows are steps: patches are P consecutive rows of one day, whatever their spacing (1-min, N-min,
trade
events). Missing bars are not filled, so forward-fill to a regular grid upstream if steps should be
equal in time.

What this implements from the recipe
------------------------------------
- Patching: non-overlapping patches of ``patch_len`` (P) time unit -> one token each.
- Per-instrument features: log-return + log-volume per time unit (recipe default, F=2).
  A patch token is the raw ``(P, F)`` slice; the patch-embedding MLP (§2.4) lives in the model.
- Targets: configurable point or cumulative forecasts of log-return and log-volume
  at horizons {1,5,15,60}, supervised at every context position.
- Each sample carries an integer ``inst_id`` for the learned instrument embedding.
- Splits: time-based train/val/test with a trading-day embargo gap; a window is
  kept only if every patch it touches (inputs *and* targets) lies in one split, so there is
  zero patch-level leakage across split boundaries.

Left to the model (not materialised here): the patch-embedding MLP (§2.4), RoPE /
instrument / calendar embeddings (§2.5), cross-instrument attention (§6). ``patch_tod``
(0..n_patch_per_day-1) is provided so the model can add a time-of-day calendar embedding.

Usage
-----
    /opt/venv/bin/python preprocessing.py --data-dir <data-dir>         # --data-dir is REQUIRED
    python examples/financial_timeseries/preprocessing.py --data-dir DATA/run \
        --patch-len 8 --context-len 64 --out-dir tensors

Each saved split ``.pt`` is a dict of stacked tensors with a leading sample axis ``N``:
    patches        (N, L, P, F) float32  raw log-return / log-volume per time unit
    inst_id        (N,)         int64    learned-embedding index (§5)
    patch_tod      (N, L)       int64    time-of-day patch index
    target_return  (N, L, H)    float32  future patch log-return at each horizon (§3.1)
    target_volshare(N, L, H)    float32  future log-volume target (legacy key name)
    (target_mask is generated on-the-fly in get_batch — not stored on disk)
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

HORIZON_UNIT_ALIASES = {
    "time-units": "time_units",
    "minutes": "time_units",
}  # accepted --horizon-unit spellings


def _ram_gb() -> str:
    used = psutil.Process().memory_info().rss / 1024**3
    total = psutil.virtual_memory().total / 1024**3
    return f"{used:.2f}/{total:.1f} GB"


@dataclass(frozen=True)
class PreprocConfig:
    """All knobs of one preprocessing run, built from the CLI in :func:`main`."""

    data_dir: Path  # dir parquet files (required)
    patch_len: int = 8  # P: time units per patch
    context_len: int = 32  # L: patches per training window
    horizons: tuple[int, ...] = (
        1,
        5,
        15,
        60,
    )  # forecast horizons, in horizon_unit (time units by default)
    window_stride: int = 8  # stride between windows, in patches
    features: tuple[str, ...] = ("log_return", "log_volume")  # per-time-unit features
    time_units_per_day: int = 390
    embargo_days: int = 2  # trading-day gap between splits
    split_fracs: tuple[float, float, float] = (0.6, 0.2, 0.2)  # train / val / test, by date
    within_day_context: bool = False  # if True, reject windows where context crosses a day boundary
    # "patches": target h = return / log-volume of the single patch h patches ahead.
    # "time_units" (default): target h = cumulative return (and log(1 + total volume)) over the next
    # h TIME_UNITS after the
    # current patch ends; targets that would cross the end of the trading day are NaN (masked in the
    # loss).
    horizon_unit: str = "time_units"
    # "point": target the single patch/time-unit at t+h (the historical behaviour).
    # "cumulative": target the complete interval t+1..t+h.
    target_mode: str = "point"
    # If False, any target whose endpoint is on another trading day is NaN/masked.
    # If True, cumulative returns include the close-to-next-session first-bar move.
    allow_overnight_targets: bool = True

    @property
    def partition_dir(self) -> Path:
        """Directory scanned for parquet files."""
        return Path(self.data_dir)

    @property
    def n_features(self) -> int:
        """Number of per-row input features (``F``)."""
        return len(self.features)

    @property
    def max_horizon(self) -> int:
        """Largest forecast horizon, in ``horizon_unit``."""
        return max(self.horizons)

    @property
    def lookahead_patches(self) -> int:
        """Patches needed after the context window to build every target."""
        if self.horizon_unit == "time_units":
            return -(-self.max_horizon // self.patch_len)  # ceil(max_h time units / P)
        return self.max_horizon

    @property
    def patches_per_day(self) -> int:
        """Whole patches that fit in one trading day; trailing rows are dropped."""
        return (
            self.time_units_per_day // self.patch_len
        )  # 390 // 8 = 48 (last 6 time units dropped)

    @property
    def num_tod_buckets(self) -> int:
        """Range of patch_tod values (the model's --num-tod-buckets must be at least this)."""
        return self.patches_per_day


# --------------------------------------------------------------------------- #
# Per-time-unit feature construction                                             #
# --------------------------------------------------------------------------- #
def _patch_tod(n_full: int) -> np.ndarray:
    """Time-of-day index per patch of one (symbol, day): the patch's position within the day (0 ..
    n_full-1).
    Assumes regular rows that start at the session open (forward-filled grid), so position = clock
    time.
    """
    return np.arange(n_full, dtype=np.int16)


def _time_unit_features(group: pd.DataFrame, features: tuple[str, ...]):
    """Per-time-unit (feature_matrix (T,F), log_return (T,), volume (T,)) for one (symbol, day)."""
    close = group["Close"].to_numpy(dtype=np.float64)
    volume = group["Volume"].to_numpy(dtype=np.float64)
    log_return = np.zeros_like(close)
    log_return[1:] = np.log(close[1:]) - np.log(close[:-1])  # intraday; first bar of day = 0
    log_volume = np.log1p(volume)  # log-volume; heavy-tailed -> log

    available = {"log_return": log_return, "log_volume": log_volume}
    for name in features:  # any other feature: a column of the same name
        if name not in available:
            if name not in group.columns:
                raise ValueError(
                    f"feature {name!r} is neither built in (log_return, log_volume) nor a column"
                )
            available[name] = group[name].to_numpy(dtype=np.float64)
    feat = np.stack([available[name] for name in features], axis=1)  # (T, F)
    return feat.astype(np.float32), log_return.astype(np.float32), volume.astype(np.float32)


def _date_split_map(dates: np.ndarray, fracs: tuple[float, float, float], embargo: int) -> dict:
    """Map each trading date -> 0(train)/1(val)/2(test)/-1(drop), with embargo gaps between
    train/val
    (we always embargo from trian).
    """
    n = len(dates)
    f_tr, f_val, _ = fracs
    i_tr, i_val = int(n * f_tr), int(n * (f_tr + f_val))
    split = np.empty(n, dtype=np.int8)
    split[:i_tr] = 0
    split[i_tr:i_val] = 1
    split[i_val:] = 2
    boundary = i_tr
    lo, hi = max(0, boundary - embargo), min(n, boundary)
    split[lo:hi] = -1

    return {pd.Timestamp(d): int(s) for d, s in zip(dates, split)}


# --------------------------------------------------------------------------- #
# Patch store: contiguous per-instrument patch sequences + per-patch targets   #
# --------------------------------------------------------------------------- #
@dataclass
class PatchStore:
    """Contiguous per-instrument patch sequences and per-patch targets, keyed by symbol."""

    symbols: list[str]
    sym_to_id: dict[str, int]
    feats: dict[str, np.ndarray]  # symbol -> (n_patch, P, F) float32  input features
    patch_return: dict[
        str, np.ndarray
    ]  # symbol -> (n_patch,)      float32  target patch log-return
    patch_volume: dict[str, np.ndarray]  # symbol -> (n_patch,)      float64  raw patch volume
    tod: dict[str, np.ndarray]  # symbol -> (n_patch,)      int16    time-of-day index
    split_id: dict[str, np.ndarray]  # symbol -> (n_patch,)      int8     0/1/2 or -1 (drop)
    patch_day_idx: dict[str, np.ndarray]  # symbol -> (n_patch,)      int32    trading-day index
    # minute-level series of the kept minutes (patch p = minutes p*P .. (p+1)*P-1), for minute
    # horizons
    unit_return: dict[str, np.ndarray] | None = (
        None  # symbol -> (n_patch*P,) float64 minute log-return
    )
    unit_volume: dict[str, np.ndarray] | None = None  # symbol -> (n_patch*P,) float64 minute volume
    unit_day_idx: dict[str, np.ndarray] | None = (
        None  # symbol -> (n_patch*P,) int32   trading-day index
    )


def build_store(cfg: PreprocConfig) -> PatchStore:
    """Read every parquet file under ``cfg.partition_dir`` and patch it into a :class:`PatchStore`.

    Files are processed in lexicographic path order; within a file, rows are grouped by
    (symbol, date), sorted by timestamp, featurised and cut into ``cfg.patch_len``-row patches.
    Trading days are also assigned to train/val/test splits here.

    Args:
        cfg: Preprocessing configuration.

    Returns:
        The populated store, with one entry per symbol that produced at least one patch.
    """
    t0 = time.perf_counter()
    P, F = cfg.patch_len, cfg.n_features
    cols = ["Symbol", "Date", "Timestamp", "Close", "Volume"]
    cols += [
        f for f in cfg.features if f not in ("log_return", "log_volume")
    ]  # pass-through feature columns

    dataset = ds.dataset(cfg.partition_dir, format="parquet", partitioning="hive")

    log.info("scanning Date column for split map  [RAM %s]", _ram_gb())
    t_scan = time.perf_counter()
    all_dates = np.sort(dataset.to_table(columns=["Date"]).column("Date").unique().to_pylist())
    date_split = _date_split_map(all_dates, cfg.split_fracs, cfg.embargo_days)
    split_counts = {0: 0, 1: 0, 2: 0, -1: 0}
    for v in date_split.values():
        split_counts[v] += 1
    log.info(
        "date scan done in %.1fs  |  %d dates  |  train/val/test/embargo: %d/%d/%d/%d  [RAM %s]",
        time.perf_counter() - t_scan,
        len(all_dates),
        split_counts[0],
        split_counts[1],
        split_counts[2],
        split_counts[-1],
        _ram_gb(),
    )

    sym_f: dict[str, list] = {}
    sym_ret: dict[str, list] = {}
    sym_vol: dict[str, list] = {}
    sym_tod: dict[str, list] = {}
    sym_split: dict[str, list] = {}
    sym_pday: dict[str, list] = {}
    sym_unit_return: dict[str, list] = {}
    sym_unit_volume: dict[str, list] = {}
    sym_unit_day_idx: dict[str, list] = {}
    previous_close: dict[str, float] = {}
    day_ord = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}

    fragments = sorted(
        dataset.get_fragments(), key=lambda frag: frag.path
    )  # each fragment will be one month
    log.info("processing %d fragments  [RAM %s]", len(fragments), _ram_gb())

    def register(symbol: str) -> None:
        if symbol not in sym_f:
            sym_f[symbol] = []
            sym_ret[symbol] = []
            sym_vol[symbol] = []
            sym_tod[symbol] = []
            sym_split[symbol] = []
            sym_pday[symbol] = []
            sym_unit_return[symbol] = []
            sym_unit_volume[symbol] = []
            sym_unit_day_idx[symbol] = []

    def add_fragment_arrays(table) -> bool:
        """Fast path: every (date, symbol) in the file has the same number of rows, so each date is
        a
        (symbols x rows) grid and all symbols are processed with whole-array operations. Gives
        exactly the
        same values as add_fragment_pandas (same float32 rounding, same sums). Returns False,
        without
        adding anything, if the rows are not a regular grid (e.g. trade events), so the caller falls
        back.
        """
        # Sort by the actual values: date, then symbol, then time.
        table = table.sort_by(
            [("Date", "ascending"), ("Symbol", "ascending"), ("Timestamp", "ascending")]
        )
        sym_enc = pc.dictionary_encode(table.column("Symbol")).combine_chunks()
        date_enc = pc.dictionary_encode(table.column("Date")).combine_chunks()
        sym_code = sym_enc.indices.to_numpy(zero_copy_only=False)
        date_code = date_enc.indices.to_numpy(zero_copy_only=False)
        n_rows = table.num_rows

        blocks = []  # (date code, start, end, n_symbols, rows per symbol)
        date_cuts = np.flatnonzero(np.diff(date_code)) + 1
        for a, b in zip(np.r_[0, date_cuts], np.r_[date_cuts, n_rows]):
            sym_cuts = np.flatnonzero(np.diff(sym_code[a:b])) + 1
            sizes = np.diff(np.r_[0, sym_cuts, b - a])
            if sizes.min() != sizes.max():  # irregular rows -> slow path
                return False
            blocks.append((int(date_code[a]), int(a), int(b), len(sizes), int(sizes[0])))

        sym_names = sym_enc.dictionary.to_pylist()
        date_values = date_enc.dictionary.to_pylist()
        close_all = np.asarray(table.column("Close").to_numpy(), dtype=np.float64)
        vol_all = np.asarray(table.column("Volume").to_numpy(), dtype=np.float64)
        extra_all = {
            name: np.asarray(table.column(name).to_numpy(), dtype=np.float64)
            for name in cfg.features
            if name not in ("log_return", "log_volume")
        }

        for dcode, a, b, S, T in blocks:
            names = [sym_names[c] for c in sym_code[a:b:T]]
            for symbol in names:
                register(symbol)
            n_full = T // P
            if n_full == 0:
                continue
            U = n_full * P  # rows kept per symbol (whole patches)
            close = close_all[a:b].reshape(S, T)[:, :U]
            volume = vol_all[a:b].reshape(S, T)[:, :U]
            log_return = np.zeros_like(close)
            log_return[:, 1:] = np.log(close[:, 1:]) - np.log(
                close[:, :-1]
            )  # intraday; first bar of day = 0
            available = {"log_return": log_return, "log_volume": np.log1p(volume)}
            for name, values in extra_all.items():
                available[name] = values[a:b].reshape(S, T)[:, :U]
            feat = np.stack([available[name] for name in cfg.features], axis=-1).astype(np.float32)
            feat = feat.reshape(S, n_full, P, F)

            # Same rounding as the per-symbol path: returns and volumes pass through float32 first.
            target_ret = log_return.astype(np.float32).astype(np.float64)
            if cfg.target_mode == "cumulative" and cfg.allow_overnight_targets:
                for i, symbol in enumerate(names):
                    if symbol in previous_close:
                        target_ret[i, 0] = np.log(float(close[i, 0])) - np.log(
                            previous_close[symbol]
                        )
            for i, symbol in enumerate(names):
                previous_close[symbol] = float(close[i, U - 1])
            vol32 = volume.astype(np.float32)
            ret_p = target_ret.reshape(S, n_full, P).sum(axis=-1).astype(np.float32)
            vol_p = vol32.reshape(S, n_full, P).sum(axis=-1).astype(np.float64)

            date = date_values[dcode]
            tod = _patch_tod(n_full)
            split = np.full(n_full, date_split[pd.Timestamp(date)], dtype=np.int8)
            pday = np.full(n_full, day_ord[pd.Timestamp(date)], dtype=np.int32)
            if cfg.horizon_unit == "time_units":
                vol64 = vol32.astype(np.float64)
                uday = np.full(U, day_ord[pd.Timestamp(date)], dtype=np.int32)
            for i, symbol in enumerate(names):  # views of the day's arrays, one per symbol
                sym_f[symbol].append(feat[i])
                sym_ret[symbol].append(ret_p[i])
                sym_vol[symbol].append(vol_p[i])
                sym_tod[symbol].append(tod)
                sym_split[symbol].append(split)
                sym_pday[symbol].append(pday)
                if cfg.horizon_unit == "time_units":
                    sym_unit_return[symbol].append(target_ret[i])
                    sym_unit_volume[symbol].append(vol64[i])
                    sym_unit_day_idx[symbol].append(uday)
        return True

    def add_fragment_pandas(df: pd.DataFrame) -> None:
        """Per-symbol, per-day path; works for any rows, including non-uniform trade events."""
        df = df.sort_values(["Symbol", "Date", "Timestamp"])
        for symbol, sym_df in df.groupby("Symbol", sort=True):  # one symbol
            register(symbol)
            for date, day in sym_df.groupby("Date", sort=True):  # one day minutes
                n_full = len(day) // P
                if n_full == 0:
                    continue
                feat, log_ret, vol = _time_unit_features(day.iloc[: n_full * P], cfg.features)
                # Inputs retain the historical intraday-return feature (first row = 0).
                # Targets optionally include the overnight close-to-first-bar move.
                target_ret = log_ret.astype(np.float64)
                if (
                    cfg.target_mode == "cumulative"
                    and cfg.allow_overnight_targets
                    and symbol in previous_close
                ):
                    first_close = float(day["Close"].iloc[0])
                    target_ret = target_ret.copy()
                    target_ret[0] = np.log(first_close) - np.log(previous_close[symbol])
                previous_close[symbol] = float(day["Close"].iloc[n_full * P - 1])

                sym_f[symbol].append(feat.reshape(n_full, P, F))
                ret_p = target_ret.reshape(n_full, P).sum(axis=1)
                vol_p = vol.reshape(n_full, P).sum(
                    axis=1
                )  # raw minute volume summed over the patch
                sym_ret[symbol].append(ret_p.astype(np.float32))
                sym_vol[symbol].append(vol_p.astype(np.float64))
                sym_tod[symbol].append(_patch_tod(n_full))
                sym_split[symbol].append(
                    np.full(n_full, date_split[pd.Timestamp(date)], dtype=np.int8)
                )
                sym_pday[symbol].append(
                    np.full(n_full, day_ord[pd.Timestamp(date)], dtype=np.int32)
                )
                if cfg.horizon_unit == "time_units":
                    sym_unit_return[symbol].append(target_ret.astype(np.float64))
                    sym_unit_volume[symbol].append(vol.astype(np.float64))
                    sym_unit_day_idx[symbol].append(
                        np.full(n_full * P, day_ord[pd.Timestamp(date)], dtype=np.int32)
                    )

    n_fast = 0
    for frag in tqdm(fragments, desc="fragments", unit="frag"):
        frag_label = Path(frag.path).parent.name  # e.g. "month=03"
        t_frag = time.perf_counter()
        table = frag.to_table(columns=cols)
        if add_fragment_arrays(table):
            n_fast += 1
            path = "arrays"
        else:
            add_fragment_pandas(table.to_pandas())
            path = "pandas"
        log.info(
            "  %-12s  %7.0fk rows  %5.1fs  %s  [RAM %s]",
            frag_label,
            table.num_rows / 1e3,
            time.perf_counter() - t_frag,
            path,
            _ram_gb(),
        )
    log.info(
        "fragments: %d via whole-day arrays, %d via the per-symbol pandas path",
        n_fast,
        len(fragments) - n_fast,
    )

    log.info("concatenating per-symbol arrays  [RAM %s]", _ram_gb())
    # A symbol with fewer than patch_len rows on every day yields no patches at all (possible for
    # thinly traded names with non-uniform rows, e.g. one row per price change); it can never form
    # a window, so leave it out of the store (and of sym_to_id) instead of failing on an empty
    # concat.
    empty = sorted(s for s, parts in sym_f.items() if not parts)
    if empty:
        log.info(
            "dropping %d symbols with no complete patch on any day (e.g. %s)",
            len(empty),
            ", ".join(empty[:5]),
        )
    symbols = sorted(s for s, parts in sym_f.items() if parts)
    sym_to_id = {s: i for i, s in enumerate(symbols)}
    feats = {s: np.concatenate(sym_f[s], axis=0) for s in symbols}
    patch_return = {s: np.concatenate(sym_ret[s]) for s in symbols}
    patch_volume = {s: np.concatenate(sym_vol[s]) for s in symbols}
    tod = {s: np.concatenate(sym_tod[s]) for s in symbols}
    split_id = {s: np.concatenate(sym_split[s]) for s in symbols}
    patch_day_idx = {s: np.concatenate(sym_pday[s]) for s in symbols}

    if not symbols:
        raise ValueError(
            f"no symbol has a complete patch: check --data-dir ({cfg.partition_dir}) and "
            "--patch-len "
            f"({cfg.patch_len}) against the rows per (symbol, day) in the data"
        )
    patches_per_sym = len(next(iter(feats.values())))
    log.info(
        "store built in %.1fs  |  %d instruments  |  ~%d patches/symbol  [RAM %s]",
        time.perf_counter() - t0,
        len(symbols),
        patches_per_sym,
        _ram_gb(),
    )
    store = PatchStore(
        symbols, sym_to_id, feats, patch_return, patch_volume, tod, split_id, patch_day_idx
    )
    if cfg.horizon_unit == "time_units":
        store.unit_return = {s: np.concatenate(sym_unit_return[s]) for s in symbols}
        store.unit_volume = {s: np.concatenate(sym_unit_volume[s]) for s in symbols}
        store.unit_day_idx = {s: np.concatenate(sym_unit_day_idx[s]) for s in symbols}
    return store


# --------------------------------------------------------------------------- #
# Windowing: enumerate leak-free windows, then build the stacked split tensors#
# --------------------------------------------------------------------------- #
def _enumerate_windows(store: PatchStore, cfg: PreprocConfig, split: int) -> list[tuple[str, int]]:
    L, stride = cfg.context_len, cfg.window_stride
    span = L + cfg.lookahead_patches  # patches touched by inputs + targets
    windows: list[tuple[str, int]] = []
    for symbol in store.symbols:
        sid = store.split_id.get(symbol)
        if sid is None or len(sid) < span:
            continue
        tod = store.tod[symbol] if cfg.within_day_context else None
        for s in range(0, len(sid) - span + 1, stride):
            seg = sid[s : s + span]  # keep iff fully inside this split
            if not (seg[0] == split and seg.min() == seg.max()):
                continue
            if cfg.within_day_context:
                ctx_tod = tod[s : s + L]
                if np.any(ctx_tod[1:] < ctx_tod[:-1]):  # tod resets = day boundary in context
                    continue
            windows.append((symbol, s))
    return windows


def _window_starts(
    store: PatchStore, cfg: PreprocConfig
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Every kept window of every split in one pass: symbol -> (start patch indices, split of each
    start).

    Same rule and order as _enumerate_windows (starts every window_stride patches; kept iff all
    L + lookahead patches lie in one split; with within_day_context, no day boundary inside the
    context), computed with array operations per symbol instead of a Python loop per start.
    """
    L, stride = cfg.context_len, cfg.window_stride
    span = L + cfg.lookahead_patches  # patches touched by inputs + targets
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for symbol in store.symbols:
        sid = store.split_id.get(symbol)
        if sid is None or len(sid) < span:
            continue
        n = len(sid)
        starts = np.arange(0, n - span + 1, stride, dtype=np.int64)
        # run_end[i]: first index after i where split_id changes (n if it never does)
        cuts = np.flatnonzero(np.diff(sid)) + 1
        run_end = np.r_[cuts, n][np.searchsorted(cuts, np.arange(n), side="right")]
        keep = (run_end[starts] >= starts + span) & (sid[starts] >= 0)
        if cfg.within_day_context:
            tod = store.tod[symbol]
            resets = np.r_[0, np.cumsum(tod[1:] < tod[:-1])]  # day boundaries up to each index
            keep &= resets[starts + L - 1] == resets[starts]
        starts = starts[keep]
        out[symbol] = (starts, sid[starts].astype(np.int8))
    return out


def _covered_patches(n: int, starts: np.ndarray, length: int) -> np.ndarray:
    """Boolean (n,): True for every patch inside at least one window [s, s + length)."""
    edges = np.zeros(n + 1, dtype=np.int64)
    np.add.at(edges, starts, 1)
    np.add.at(edges, starts + length, -1)
    return np.cumsum(edges[:n]) > 0


def _symbol_targets(
    store: PatchStore, cfg: PreprocConfig, symbol: str
) -> tuple[np.ndarray, np.ndarray]:
    """Raw return/volume targets for EVERY usable context position of one symbol.

    Returns two ``(n_patch - lookahead_patches, H)`` float64 arrays: row p holds the
    targets of context position p, so the targets of a window starting at ``s`` are
    the slice ``[s : s + context_len]``. A position's target depends only on
    (symbol, position, horizon) — never on the window — so this is computed once
    per symbol instead of once per window, with the exact formulas of the old
    per-window path (values are bit-identical; the float32 cast still happens at
    memmap assignment). Invalid intraday-only targets are NaN and become False in
    ``target_mask`` in get_batch().
    """
    horizons = np.asarray(cfg.horizons, dtype=np.int64)
    n_valid = max(len(store.patch_return[symbol]) - cfg.lookahead_patches, 0)
    base = np.arange(n_valid, dtype=np.int64)
    target_ret = np.empty((n_valid, len(horizons)), dtype=np.float64)
    target_vol = np.empty_like(target_ret)

    if cfg.horizon_unit == "time_units":
        ret, vol = store.unit_return[symbol], store.unit_volume[symbol]
    else:
        ret, vol = store.patch_return[symbol], store.patch_volume[symbol]
    cumulative_ret = np.concatenate([[0.0], np.cumsum(ret, dtype=np.float64)])
    cumulative_vol = np.concatenate([[0.0], np.cumsum(vol, dtype=np.float64)])

    if cfg.horizon_unit == "time_units":
        origin = (base + 1) * cfg.patch_len - 1  # last time unit of each context patch
        ret_series, vol_series = store.unit_return[symbol], store.unit_volume[symbol]
        day = store.unit_day_idx[symbol]
        for j, h in enumerate(horizons):
            endpoint = origin + h
            if cfg.target_mode == "point":
                ret = store.unit_return[symbol][endpoint]
                vol = np.log1p(store.unit_volume[symbol][endpoint])
            else:
                ret = cumulative_ret[endpoint + 1] - cumulative_ret[origin + 1]
                vol = np.log1p(cumulative_vol[endpoint + 1] - cumulative_vol[origin + 1])
            valid = np.ones(n_valid, dtype=bool)
            if not cfg.allow_overnight_targets:
                valid = day[endpoint] == day[origin]
            target_ret[:, j] = np.where(valid, ret, np.nan)
            target_vol[:, j] = np.where(valid, vol, np.nan)
        return target_ret, target_vol

    day = store.patch_day_idx[symbol]
    for j, h in enumerate(horizons):
        endpoint = base + h
        if cfg.target_mode == "point":
            ret = store.patch_return[symbol][endpoint]
            vol = np.log1p(store.patch_volume[symbol][endpoint])
        else:
            ret = cumulative_ret[endpoint + 1] - cumulative_ret[base + 1]
            vol = np.log1p(cumulative_vol[endpoint + 1] - cumulative_vol[base + 1])
        valid = np.ones(n_valid, dtype=bool)
        if not cfg.allow_overnight_targets:
            valid = day[endpoint] == day[base]
        target_ret[:, j] = np.where(valid, ret, np.nan)
        target_vol[:, j] = np.where(valid, vol, np.nan)
    return target_ret, target_vol


def _empty_target_stats(n_horizons: int) -> dict[str, np.ndarray]:
    return {
        "return_sum": np.zeros(n_horizons, dtype=np.float64),
        "return_sumsq": np.zeros(n_horizons, dtype=np.float64),
        "return_count": np.zeros(n_horizons, dtype=np.int64),
        "volume_sum": np.zeros(n_horizons, dtype=np.float64),
        "volume_sumsq": np.zeros(n_horizons, dtype=np.float64),
        "volume_count": np.zeros(n_horizons, dtype=np.int64),
    }


def _accumulate_target_stats(stats: dict[str, np.ndarray], name: str, values: np.ndarray) -> None:
    valid = np.isfinite(values)
    safe = np.where(valid, values, 0.0)
    stats[f"{name}_sum"] += safe.sum(axis=0, dtype=np.float64)
    stats[f"{name}_sumsq"] += np.square(safe).sum(axis=0, dtype=np.float64)
    stats[f"{name}_count"] += valid.sum(axis=0)


def _finalize_target_stats(stats: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name in ("return", "volume"):
        count = stats[f"{name}_count"]
        if np.any(count == 0):
            missing = np.flatnonzero(count == 0).tolist()
            raise ValueError(
                f"training split has no valid {name} targets for horizon indices {missing}; "
                "reduce the horizon or enable overnight targets"
            )
        mean = stats[f"{name}_sum"] / count
        variance = np.maximum(stats[f"{name}_sumsq"] / count - mean**2, 0.0)
        std = np.sqrt(variance)
        std = np.maximum(std, 1e-8)
        result[f"{name}_mean"] = torch.from_numpy(mean.astype(np.float32))
        result[f"{name}_std"] = torch.from_numpy(std.astype(np.float32))
        result[f"{name}_count"] = torch.from_numpy(count.copy())
    return result


def build_split_tensors(
    store: PatchStore,
    cfg: PreprocConfig,
    split: int,
    tmp_dir: Path,
    window_starts: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
    """Stack every leak-free window of one split into batched tensors (see module docstring).

    Uses memory-mapped numpy arrays backed by files in tmp_dir so that the full
    (N, L, P, F) output never has to fit in RAM. Windows are written to disk as
    they are filled; the OS pages in only what is needed at any moment.

    tmp_dir must live on a filesystem with enough free space for all five arrays
    (roughly the same size as the final .pt file). The caller is responsible for
    removing tmp_dir with shutil.rmtree() after torch.save() completes.

    Works one symbol at a time: _symbol_targets gives the targets of every position once, then all
    the symbol's windows are written with one indexed copy per array. Rows come out in the same
    order and with the same values as a per-window build (_enumerate_windows followed by slicing
    _symbol_targets).

    Args:
        store: Patch store from :func:`build_store`.
        cfg: Preprocessing configuration.
        split: Split id to materialise (0 = train, 1 = val, 2 = test).
        tmp_dir: Directory for the memory-mapped scratch arrays.
        window_starts: Output of ``_window_starts``; computed here when omitted. Pass it in to
            reuse the same window enumeration across splits.

    Returns:
        Tuple ``(tensors, target_stats)``: the split's tensors keyed by name, and the accumulated
        per-horizon target statistics that ``main`` writes to ``meta.pt``.
    """
    if window_starts is None:
        window_starts = _window_starts(store, cfg)
    per_symbol = [(sym, starts[labels == split]) for sym, (starts, labels) in window_starts.items()]
    per_symbol = [(sym, s) for sym, s in per_symbol if len(s)]
    L, H = cfg.context_len, len(cfg.horizons)
    P, F = cfg.patch_len, cfg.n_features

    if not per_symbol:
        return {
            "patches": torch.zeros(0, L, P, F),
            "inst_id": torch.zeros(0, dtype=torch.long),
            "patch_tod": torch.zeros(0, L, dtype=torch.long),
            "target_return": torch.zeros(0, L, H),
            "target_volshare": torch.zeros(0, L, H),
        }, _empty_target_stats(H)

    n = sum(len(s) for _, s in per_symbol)
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Estimate and log disk requirement so the user can abort early if needed.
    est_bytes = n * (L * P * F * 4 + 8 + L * 8 + L * H * 4 + L * H * 4)
    log.info("  %d windows  |  memmaps ~%.1f GiB  [RAM %s]", n, est_bytes / 1024**3, _ram_gb())

    # np.memmap creates a file-backed array; writes go to disk immediately.
    # torch.from_numpy() on a memmap shares the same storage, so torch.save()
    # streams bytes from the mapped file without loading everything into RAM.
    patches_mm = np.memmap(tmp_dir / "patches", mode="w+", dtype=np.float32, shape=(n, L, P, F))
    inst_mm = np.memmap(tmp_dir / "inst", mode="w+", dtype=np.int64, shape=(n,))
    tod_mm = np.memmap(tmp_dir / "tod", mode="w+", dtype=np.int64, shape=(n, L))
    tgt_ret_mm = np.memmap(tmp_dir / "tgt_ret", mode="w+", dtype=np.float32, shape=(n, L, H))
    tgt_shr_mm = np.memmap(tmp_dir / "tgt_shr", mode="w+", dtype=np.float32, shape=(n, L, H))

    target_stats = _empty_target_stats(H)
    offsets = np.arange(L, dtype=np.int64)
    i0 = 0
    for symbol, starts in tqdm(per_symbol, desc="symbols", unit="sym", leave=False):
        i1 = i0 + len(starts)
        # Per-symbol target panel (row p = targets of context position p), computed once; every kept
        # window satisfies s + L <= n_patch - lookahead, so all its rows exist in the panel.
        sym_ret, sym_vol = _symbol_targets(store, cfg, symbol)
        idx = starts[:, None] + offsets  # (windows, L) patch indices
        patches_mm[i0:i1] = store.feats[symbol][idx]
        inst_mm[i0:i1] = store.sym_to_id[symbol]
        tod_mm[i0:i1] = store.tod[symbol][idx]
        window_ret, window_vol = sym_ret[idx], sym_vol[idx]  # (windows, L, H)
        tgt_ret_mm[i0:i1] = window_ret
        tgt_shr_mm[i0:i1] = window_vol
        # Statistics over all windows' targets (a patch counts once per window it appears in).
        _accumulate_target_stats(target_stats, "return", window_ret.reshape(-1, H))
        _accumulate_target_stats(target_stats, "volume", window_vol.reshape(-1, H))
        i0 = i1

    for mm in (patches_mm, inst_mm, tod_mm, tgt_ret_mm, tgt_shr_mm):
        mm.flush()

    return {
        "patches": torch.from_numpy(patches_mm),
        "inst_id": torch.from_numpy(inst_mm),
        "patch_tod": torch.from_numpy(tod_mm),
        "target_return": torch.from_numpy(tgt_ret_mm),
        "target_volshare": torch.from_numpy(tgt_shr_mm),
    }, target_stats


def _leakage_check(
    store: PatchStore,
    cfg: PreprocConfig,
    window_starts: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[int, int, int]:
    """Return overlap counts of (symbol, patch-index) anchors between splits; expect (0,0,0).

    Counts the context patches covered by windows of two different splits, per symbol (overlap is
    only possible within one symbol's series), using boolean coverage arrays instead of sets.
    """
    if window_starts is None:
        window_starts = _window_starts(store, cfg)
    tv = tt = vt = 0
    for symbol, (starts, labels) in window_starts.items():
        n = len(store.split_id[symbol])
        tr, va, te = (_covered_patches(n, starts[labels == k], cfg.context_len) for k in (0, 1, 2))
        tv += int(np.count_nonzero(tr & va))
        tt += int(np.count_nonzero(tr & te))
        vt += int(np.count_nonzero(va & te))
    return tv, tt, vt


def main() -> None:
    """Parse the CLI, build the patch store and write the split tensors plus ``meta.pt``."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--data-dir", type=Path, required=True, help="REQUIRED: dir holding parquet files"
    )
    p.add_argument("--patch-len", type=int, default=PreprocConfig.patch_len)
    p.add_argument("--context-len", type=int, default=PreprocConfig.context_len)
    p.add_argument("--window-stride", type=int, default=PreprocConfig.window_stride)
    p.add_argument("--embargo-days", type=int, default=PreprocConfig.embargo_days)
    p.add_argument(
        "--time-units-per-day",
        type=int,
        default=PreprocConfig.time_units_per_day,
        help="rows (time steps) per trading day (default 390 = 1-min bars, 09:30–16:00; 78 for "
        "5-min "
        "bars). Sets patches per day and the intraday horizon check.",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in PreprocConfig.horizons),
        help="comma-separated forecast horizons in --horizon-unit units (default '1,5,15,60' time "
        "units)",
    )
    p.add_argument(
        "--split-fracs",
        type=str,
        default=",".join(str(f) for f in PreprocConfig.split_fracs),
        help="comma-separated train/val/test fractions summing to 1 (default '0.6,0.2,0.2')",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="output dir for the .pt tensors (default: <data-dir>/../tensors)",
    )
    p.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="dir for temporary memmap files (default: <out-dir>/.tmp_<split>). Use /tmp to save "
        "disk space.",
    )
    p.add_argument(
        "--features",
        type=str,
        default=",".join(PreprocConfig.features),
        help="comma-separated per-time-unit features (default 'log_return,log_volume'). Names "
        "other than "
        "these two are read from parquet columns of the same name. Keep log_return, log_volume "
        "first: the loss normalizes the return / volume targets with features 0 and 1.",
    )
    p.add_argument(
        "--horizon-unit",
        choices=["patches", "time_units", *HORIZON_UNIT_ALIASES],
        default=PreprocConfig.horizon_unit,
        help="unit used by --horizons: time_units (default; rows, i.e. minutes for 1-min bars; "
        "'time-units' and the "
        "older 'minutes' mean the same) or patches. Use with --target-mode to choose point versus "
        "cumulative targets.",
    )
    p.add_argument(
        "--target-mode",
        choices=["point", "cumulative"],
        default=PreprocConfig.target_mode,
        help="point (default): target only the patch/time-unit at t+h. cumulative: target the "
        "complete future interval t+1..t+h.",
    )
    p.add_argument(
        "--allow-overnight-targets",
        action=argparse.BooleanOptionalAction,
        default=PreprocConfig.allow_overnight_targets,
        help="allow targets to cross sessions; cumulative returns then include the overnight move. "
        "Use --no-allow-overnight-targets to mask targets crossing the close.",
    )
    p.add_argument(
        "--within-day-context",
        action="store_true",
        default=False,
        help="Reject windows where the context patches cross a day boundary. Horizons may still "
        "extend into future days.",
    )
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.data_dir.parent / "tensors"

    horizons = tuple(int(x) for x in args.horizons.split(","))
    split_fracs = tuple(float(x) for x in args.split_fracs.split(","))
    if len(split_fracs) != 3 or abs(sum(split_fracs) - 1.0) > 1e-6:
        p.error("--split-fracs must be three comma-separated values summing to 1.0")

    cfg = PreprocConfig(
        data_dir=args.data_dir,
        patch_len=args.patch_len,
        context_len=args.context_len,
        window_stride=args.window_stride,
        embargo_days=args.embargo_days,
        time_units_per_day=args.time_units_per_day,
        horizons=horizons,
        split_fracs=split_fracs,
        within_day_context=args.within_day_context,
        features=tuple(args.features.split(",")),
        horizon_unit=HORIZON_UNIT_ALIASES.get(args.horizon_unit, args.horizon_unit),
        target_mode=args.target_mode,
        allow_overnight_targets=args.allow_overnight_targets,
    )
    if cfg.features[:2] != ("log_return", "log_volume"):
        p.error("--features must start with log_return,log_volume (the loss uses features 0 and 1)")
    if not cfg.allow_overnight_targets:
        day_units = cfg.patches_per_day if cfg.horizon_unit == "patches" else cfg.time_units_per_day
        impossible = [h for h in cfg.horizons if h >= day_units]
        if impossible:
            p.error(
                "intraday-only targets require every horizon to be shorter than one session; "
                f"got {impossible} with {day_units} {cfg.horizon_unit} per session"
            )

    t_total = time.perf_counter()
    log.info("reading %s  [RAM %s]", cfg.partition_dir, _ram_gb())
    store = build_store(cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    t_win = time.perf_counter()
    window_starts = _window_starts(store, cfg)  # all splits' windows, found once
    log.info("window starts found in %.1fs  [RAM %s]", time.perf_counter() - t_win, _ram_gb())
    counts = {}
    split_target_stats: dict[str, dict[str, np.ndarray]] = {}
    for name, split in (("train", 0), ("val", 1), ("test", 2)):
        t_split = time.perf_counter()
        log.info("building %s split tensors  [RAM %s]", name, _ram_gb())
        tmp_base = args.tmp_dir if args.tmp_dir is not None else out_dir
        tmp = tmp_base / f".tmp_{name}"
        tensors, target_stats = build_split_tensors(store, cfg, split, tmp, window_starts)
        n_windows = int(tensors["inst_id"].shape[0])
        patch_shape = tuple(tensors["patches"].shape)
        out = out_dir / f"{name}.pt"
        counts[name] = n_windows
        split_target_stats[name] = target_stats
        if n_windows == 0:
            # No file for an empty split: the data loader skips a missing split (and only requires
            # train.pt when training), so a one-split dataset (e.g. --split-fracs 0,0,1) just works.
            shutil.rmtree(tmp, ignore_errors=True)
            # A rerun into the same out_dir must not leave an earlier run's file behind: the
            # loader would read it and silently mix datasets from different runs.
            if out.exists():
                out.unlink()
                log.warning("  %-5s -> removed stale %s from a previous run", name, out)
            log.info(
                "  %-5s -> no windows (split_fracs %s): %s not written",
                name,
                cfg.split_fracs,
                out.name,
            )
            continue
        torch.save(tensors, out)
        shutil.rmtree(tmp, ignore_errors=True)  # remove memmaps now that .pt is written
        log.info(
            "  %-5s -> %s  (%d windows, patches %s)  %.1fs  [RAM %s]",
            name,
            out,
            counts[name],
            patch_shape,
            time.perf_counter() - t_split,
            _ram_gb(),
        )
    if not any(counts.values()):
        raise ValueError(
            f"no windows in any split: {sum(len(v) for v in store.feats.values()):,} patches over "
            f"{len(store.symbols)} symbols, but a window needs context_len + lookahead = "
            f"{cfg.context_len + cfg.lookahead_patches} patches inside one split (split_fracs "
            f"{cfg.split_fracs}, "
            f"embargo {cfg.embargo_days} days). Use more data, or a shorter context / horizon."
        )

    log.info("running leakage check  [RAM %s]", _ram_gb())
    o_tv, o_tt, o_vt = _leakage_check(store, cfg, window_starts)
    log.info(
        "leakage (train/val, train/test, val/test): %d %d %d  (expect 0 0 0)", o_tv, o_tt, o_vt
    )

    # Per-horizon target mean / std from the train split only: pretrain_ts.py and
    # evaluate_megatron.py scale
    # the targets with these (without them they fall back to per-patch RevIN scaling).
    if counts["train"]:
        target_stats = _finalize_target_stats(split_target_stats["train"])
    else:
        target_stats = None
        log.warning(
            "train split is empty, so meta.pt has no target_stats. This dataset can be used for "
            "evaluation "
            "only; training / evaluation code will fall back to per-patch RevIN target scaling on "
            "it, which "
            "differs from a model trained with per-horizon target_stats."
        )
    valid_target_counts = {
        split_name: {
            "return": stats["return_count"].tolist(),
            "volume": stats["volume_count"].tolist(),
        }
        for split_name, stats in split_target_stats.items()
    }
    meta = {
        "n_instruments": len(store.symbols),
        "symbols": store.symbols,
        "sym_to_id": store.sym_to_id,
        "features": list(cfg.features),
        "patch_len": cfg.patch_len,
        "context_len": cfg.context_len,
        "horizons": list(cfg.horizons),
        "horizon_unit": cfg.horizon_unit,
        "horizon_minutes": [
            h * cfg.patch_len if cfg.horizon_unit == "patches" else h for h in cfg.horizons
        ],
        "target_mode": cfg.target_mode,
        "allow_overnight_targets": cfg.allow_overnight_targets,
        "target_stats": target_stats,
        "valid_target_counts": valid_target_counts,
        "input_token_shape": (cfg.patch_len, cfg.n_features),
        "time_units_per_day": cfg.time_units_per_day,
        "patches_per_day": cfg.patches_per_day,
        "num_tod_buckets": cfg.num_tod_buckets,
        "window_counts": counts,
        "split_fracs": cfg.split_fracs,
        "embargo_days": cfg.embargo_days,
    }
    torch.save(meta, out_dir / "meta.pt")
    log.info("  meta  -> %s", out_dir / "meta.pt")
    log.info("done in %.1fs  [RAM %s]", time.perf_counter() - t_total, _ram_gb())


if __name__ == "__main__":
    main()
