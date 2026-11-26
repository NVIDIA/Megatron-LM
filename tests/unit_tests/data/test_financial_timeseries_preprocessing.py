# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from examples.financial_timeseries.preprocessing import (
    PatchStore,
    PreprocConfig,
    _accumulate_target_stats,
    _empty_target_stats,
    _enumerate_windows,
    _finalize_target_stats,
    _leakage_check,
    _symbol_targets,
    _window_starts,
    build_split_tensors,
    build_store,
)
from megatron.core.datasets.utils import Split


def _store() -> PatchStore:
    symbol = "TEST"
    n = 6
    return PatchStore(
        symbols=[symbol],
        sym_to_id={symbol: 0},
        feats={symbol: np.zeros((n, 1, 2), dtype=np.float32)},
        # Index 3 represents a synthetic overnight move.
        patch_return={symbol: np.asarray([0.0, 1.0, 2.0, 10.0, 4.0, 5.0], dtype=np.float32)},
        patch_volume={symbol: np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float64)},
        tod={symbol: np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int16)},
        split_id={symbol: np.zeros(n, dtype=np.int8)},
        patch_day_idx={symbol: np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32)},
    )


def _window_targets(store: PatchStore, cfg: PreprocConfig, symbol: str, start: int):
    """Targets of one window = slice of the per-symbol panels (as the pipeline does)."""
    ret, vol = _symbol_targets(store, cfg, symbol)
    return ret[start : start + cfg.context_len], vol[start : start + cfg.context_len]


class WindowTargetTest(unittest.TestCase):
    def _cfg(self, **kwargs) -> PreprocConfig:
        values = dict(
            data_dir=Path("."),
            patch_len=1,
            context_len=2,
            horizons=(1, 2),
            window_stride=1,
            time_units_per_day=3,
            embargo_days=0,
            split_fracs=(1.0, 0.0, 0.0),
            horizon_unit="patches",  # the store above is patch-level only (no per-time-unit arrays)
            target_mode="cumulative",
            allow_overnight_targets=True,
        )
        values.update(kwargs)
        return PreprocConfig(**values)

    def test_point_mode_matches_single_future_patch(self):
        ret, vol = _window_targets(_store(), self._cfg(target_mode="point"), "TEST", 0)
        np.testing.assert_allclose(ret, [[1.0, 2.0], [2.0, 10.0]])
        np.testing.assert_allclose(vol, [[np.log1p(2), np.log1p(3)], [np.log1p(3), np.log1p(4)]])

    def test_cumulative_patch_targets(self):
        ret, vol = _window_targets(_store(), self._cfg(), "TEST", 0)
        np.testing.assert_allclose(ret, [[1.0, 3.0], [2.0, 12.0]])
        np.testing.assert_allclose(vol, [[np.log1p(2), np.log1p(5)], [np.log1p(3), np.log1p(7)]])

    def test_intraday_mode_masks_cross_close_targets(self):
        ret, vol = _window_targets(_store(), self._cfg(allow_overnight_targets=False), "TEST", 0)
        self.assertTrue(np.isfinite(ret[0, 1]))
        self.assertTrue(np.isnan(ret[1, 1]))
        self.assertTrue(np.isnan(vol[1, 1]))

    def test_split_containment_includes_target_lookahead(self):
        store = _store()
        store.split_id["TEST"] = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8)
        self.assertEqual(_enumerate_windows(store, self._cfg(), split=0), [])


class EndToEndTargetTest(unittest.TestCase):
    def test_overnight_return_and_normalization_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            parquet_dir = root / "bars"
            parquet_dir.mkdir()
            rows = []
            for date, closes in (("2026-01-05", [100.0, 101.0]), ("2026-01-06", [110.0, 111.0])):
                for minute, close in enumerate(closes):
                    rows.append(
                        {
                            "Symbol": "TEST",
                            "Date": pd.Timestamp(date).date(),
                            "Timestamp": pd.Timestamp(f"{date} 09:{30 + minute}:00"),
                            "Close": close,
                            "Volume": 10 + minute,
                        }
                    )
            pd.DataFrame(rows).to_parquet(parquet_dir / "bars.parquet", index=False)

            cfg = PreprocConfig(
                data_dir=parquet_dir,
                patch_len=1,
                context_len=2,
                horizons=(1, 2),
                window_stride=1,
                time_units_per_day=2,
                embargo_days=0,
                split_fracs=(1.0, 0.0, 0.0),
                target_mode="cumulative",
                allow_overnight_targets=True,
            )
            store = build_store(cfg)
            expected_overnight = np.log(110.0) - np.log(101.0)
            self.assertAlmostEqual(
                float(store.patch_return["TEST"][2]), expected_overnight, places=6
            )

            tensors, raw_stats = build_split_tensors(store, cfg, split=0, tmp_dir=root / "tmp")
            self.assertEqual(tuple(tensors["target_return"].shape), (1, 2, 2))
            stats = _finalize_target_stats(raw_stats)
            values = tensors["target_return"]
            z = (values - stats["return_mean"]) / stats["return_std"]
            restored = z * stats["return_std"] + stats["return_mean"]
            self.assertTrue(np.allclose(values.numpy(), restored.numpy(), equal_nan=True))


class SingleSplitTest(unittest.TestCase):
    """The CLI must cope with split fractions that leave some splits empty: no .pt for an empty
    split,
    target_stats only when train has windows, and a clear error when no split has any."""

    _REPO = Path(__file__).resolve().parents[3]
    _SCRIPT = _REPO / "examples" / "financial_timeseries" / "preprocessing.py"

    def _run(
        self,
        root: Path,
        split_fracs: str,
        days: int = 8,
        context_len: int = 2,
        out: Path | None = None,
    ):
        import subprocess
        import sys

        parquet_dir = root / "bars"
        parquet_dir.mkdir(exist_ok=True)
        rows = []
        for d in range(days):
            date = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
            for minute in range(4):
                rows.append(
                    {
                        "Symbol": "TEST",
                        "Date": date.date(),
                        "Timestamp": date + pd.Timedelta(hours=9, minutes=30 + minute),
                        "Close": 100.0 + d + 0.1 * minute,
                        "Volume": 10.0 + minute,
                    }
                )
        pd.DataFrame(rows).to_parquet(parquet_dir / "bars.parquet", index=False)
        out = out if out is not None else root / "tensors"
        cmd = [
            sys.executable,
            str(self._SCRIPT),
            "--data-dir",
            str(parquet_dir),
            "--out-dir",
            str(out),
            "--patch-len",
            "1",
            "--context-len",
            str(context_len),
            "--horizons",
            "1",
            "--window-stride",
            "1",
            "--time-units-per-day",
            "4",
            "--embargo-days",
            "0",
            "--split-fracs",
            split_fracs,
            "--horizon-unit",
            "patches",
            "--target-mode",
            "cumulative",
            "--no-allow-overnight-targets",
        ]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=self._REPO), out

    def test_test_only_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            res, out = self._run(Path(root), "0,0,1")
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            self.assertFalse((out / "train.pt").exists())
            self.assertFalse((out / "val.pt").exists())
            self.assertTrue((out / "test.pt").exists())
            meta = torch.load(out / "meta.pt")
            self.assertIsNone(meta["target_stats"])
            self.assertEqual(meta["window_counts"]["train"], 0)
            self.assertGreater(meta["window_counts"]["test"], 0)
            self.assertIn("train split is empty", res.stderr)

    def test_rerun_removes_stale_split_files(self):
        """A split that is empty on a rerun must not leave the previous run's .pt in out_dir."""
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "tensors"
            res, _ = self._run(Path(root), "1,0,0", out=out)
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            self.assertTrue((out / "train.pt").exists())
            res, _ = self._run(Path(root), "0,0,1", out=out)
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            self.assertFalse((out / "train.pt").exists(), "stale train.pt survived the rerun")
            self.assertFalse((out / "val.pt").exists())
            self.assertTrue((out / "test.pt").exists())
            self.assertIn("removed stale", res.stderr)

    def test_train_only_dataset(self):
        with tempfile.TemporaryDirectory() as root:
            res, out = self._run(Path(root), "1,0,0")
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            self.assertTrue((out / "train.pt").exists())
            self.assertFalse((out / "val.pt").exists())
            self.assertFalse((out / "test.pt").exists())
            meta = torch.load(out / "meta.pt")
            self.assertIsNotNone(meta["target_stats"])
            self.assertEqual(meta["window_counts"]["val"], 0)

    def test_no_windows_anywhere_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as root:
            # 1 day (4 patches) in val and in test, none in train; a window needs context 5 +
            # lookahead 1 = 6
            # patches inside one split, so no split can hold one.
            res, _ = self._run(Path(root), "0.4,0.3,0.3", days=2, context_len=5)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("no windows in any split", res.stderr)


class RowOrderTest(unittest.TestCase):
    """build_store must not depend on the row order inside a file. The whole-day array path sorts by
    the real
    Date / Symbol / Timestamp values; sorting on dictionary codes (first-appearance order) would put
    a later
    date's block first whenever a file starts with it, e.g. a symbol that only exists from day 2 on.
    """

    @staticmethod
    def _rows():
        rows = []
        for date, symbols in (
            ("2026-01-05", {"TEST": [100.0, 101.0]}),
            ("2026-01-06", {"AAPEL": [50.0, 51.0], "TEST": [110.0, 111.0]}),
        ):
            for symbol, closes in symbols.items():
                for minute, close in enumerate(closes):
                    rows.append(
                        {
                            "Symbol": symbol,
                            "Date": pd.Timestamp(date).date(),
                            "Timestamp": pd.Timestamp(f"{date} 09:{30 + minute}:00"),
                            "Close": close,
                            "Volume": 10.0 + minute,
                        }
                    )
        return rows

    def _build(self, root: Path, rows) -> PatchStore:
        parquet_dir = root / "bars"
        parquet_dir.mkdir()
        pd.DataFrame(rows).to_parquet(parquet_dir / "bars.parquet", index=False)
        cfg = PreprocConfig(
            data_dir=parquet_dir,
            patch_len=1,
            context_len=2,
            horizons=(1,),
            window_stride=1,
            time_units_per_day=2,
            embargo_days=0,
            split_fracs=(1.0, 0.0, 0.0),
            horizon_unit="patches",
            target_mode="cumulative",
            allow_overnight_targets=True,
        )
        return build_store(cfg)

    def test_scrambled_rows_give_the_same_store(self):
        rows = self._rows()
        # Day 2 first, and within it the day-2-only symbol first: its dictionary codes would both be
        # 0.
        scrambled = (
            [r for r in rows if str(r["Date"]) == "2026-01-06" and r["Symbol"] == "AAPEL"]
            + [r for r in rows if str(r["Date"]) == "2026-01-06" and r["Symbol"] == "TEST"]
            + [r for r in rows if str(r["Date"]) == "2026-01-05"]
        )
        self.assertNotEqual([r["Date"] for r in scrambled], [r["Date"] for r in rows])
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ordered, shuffled = self._build(Path(a), rows), self._build(Path(b), scrambled)
        self.assertEqual(ordered.symbols, shuffled.symbols)
        for symbol in ordered.symbols:
            self.assertTrue(
                np.all(np.diff(shuffled.patch_day_idx[symbol]) >= 0), symbol
            )  # chronological
            for name in (
                "feats",
                "patch_return",
                "patch_volume",
                "tod",
                "split_id",
                "patch_day_idx",
            ):
                np.testing.assert_array_equal(
                    getattr(shuffled, name)[symbol],
                    getattr(ordered, name)[symbol],
                    err_msg=f"{name} {symbol}",
                )
        # TEST's day-2 first patch carries the overnight move 101 -> 110, whatever the row order.
        self.assertAlmostEqual(
            float(shuffled.patch_return["TEST"][2]), np.log(110.0) - np.log(101.0), places=6
        )


def _random_store(
    seed: int = 0, patch_len: int = 2, days: int = 12, patches_per_day: int = 5
) -> PatchStore:
    """Two symbols with random returns and volumes, split by day into train/embargo/val/test.

    The time-of-day index resets daily.
    """
    rng = np.random.default_rng(seed)
    n = days * patches_per_day
    day = np.repeat(np.arange(days, dtype=np.int32), patches_per_day)
    split = np.select([day < 6, day < 7, day < 9], [0, -1, 1], 2).astype(np.int8)
    store = PatchStore(
        symbols=[],
        sym_to_id={},
        feats={},
        patch_return={},
        patch_volume={},
        tod={},
        split_id={},
        patch_day_idx={},
        unit_return={},
        unit_volume={},
        unit_day_idx={},
    )
    for k, symbol in enumerate(("AAA", "BBB")):
        m = n - 3 * k  # different lengths per symbol
        unit_ret = rng.normal(0, 1e-3, m * patch_len)
        unit_vol = rng.integers(0, 500, m * patch_len).astype(np.float64)
        store.symbols.append(symbol)
        store.sym_to_id[symbol] = k
        store.feats[symbol] = rng.normal(size=(m, patch_len, 2)).astype(np.float32)
        store.patch_return[symbol] = unit_ret.reshape(m, patch_len).sum(1).astype(np.float32)
        store.patch_volume[symbol] = unit_vol.reshape(m, patch_len).sum(1)
        store.tod[symbol] = np.tile(np.arange(patches_per_day, dtype=np.int16), days)[:m]
        store.split_id[symbol] = split[:m]
        store.patch_day_idx[symbol] = day[:m]
        store.unit_return[symbol] = unit_ret
        store.unit_volume[symbol] = unit_vol
        store.unit_day_idx[symbol] = np.repeat(day[:m], patch_len)
    return store


class PerSymbolMatchesPerWindowTest(unittest.TestCase):
    """The per-symbol pipeline (_window_starts, build_split_tensors, _leakage_check) must give
    exactly what a
    per-window build gives (_enumerate_windows + slicing _symbol_targets per window), for every
    option.
    """

    def _configs(self):
        for horizon_unit, horizons in (("patches", (1, 2)), ("time_units", (1, 3))):
            for target_mode in ("point", "cumulative"):
                for overnight in (True, False):
                    for within_day in (False, True):
                        yield PreprocConfig(
                            data_dir=Path("."),
                            patch_len=2,
                            context_len=3,
                            horizons=horizons,
                            window_stride=2,
                            embargo_days=0,
                            split_fracs=(1.0, 0.0, 0.0),
                            horizon_unit=horizon_unit,
                            target_mode=target_mode,
                            allow_overnight_targets=overnight,
                            within_day_context=within_day,
                        )

    def test_window_starts_match_enumerate_windows(self):
        store = _random_store()
        for cfg in self._configs():
            starts = _window_starts(store, cfg)
            for split in (0, 1, 2):
                new = [(sym, int(s)) for sym, (st, lab) in starts.items() for s in st[lab == split]]
                self.assertEqual(new, _enumerate_windows(store, cfg, split), (cfg, split))

    def test_split_tensors_match_per_window_build(self):
        store = _random_store()
        for cfg in self._configs():
            for split in (0, 1, 2):
                windows = _enumerate_windows(store, cfg, split)
                with tempfile.TemporaryDirectory() as tmp:
                    tensors, stats = build_split_tensors(store, cfg, split, Path(tmp))
                    self.assertEqual(int(tensors["inst_id"].shape[0]), len(windows))
                    ref_stats = _empty_target_stats(len(cfg.horizons))
                    L = cfg.context_len
                    for i, (symbol, s) in enumerate(windows):
                        ref_ret, ref_vol = _window_targets(store, cfg, symbol, s)
                        _accumulate_target_stats(ref_stats, "return", ref_ret)
                        _accumulate_target_stats(ref_stats, "volume", ref_vol)
                        np.testing.assert_array_equal(
                            tensors["patches"][i].numpy(), store.feats[symbol][s : s + L]
                        )
                        np.testing.assert_array_equal(
                            tensors["patch_tod"][i].numpy(), store.tod[symbol][s : s + L]
                        )
                        self.assertEqual(int(tensors["inst_id"][i]), store.sym_to_id[symbol])
                        np.testing.assert_array_equal(
                            tensors["target_return"][i].numpy(), ref_ret.astype(np.float32)
                        )
                        np.testing.assert_array_equal(
                            tensors["target_volshare"][i].numpy(), ref_vol.astype(np.float32)
                        )
                    for key in ref_stats:  # same terms, different summation order
                        if key.endswith("_count"):
                            np.testing.assert_array_equal(stats[key], ref_stats[key], err_msg=key)
                        else:
                            np.testing.assert_allclose(
                                stats[key], ref_stats[key], rtol=1e-12, err_msg=key
                            )

    def test_leakage_check_matches_set_version(self):
        store = _random_store()
        for cfg in self._configs():

            def anchors(split):
                return {
                    (sym, s + p)
                    for sym, s in _enumerate_windows(store, cfg, split)
                    for p in range(cfg.context_len)
                }

            tr, va, te = anchors(0), anchors(1), anchors(2)
            self.assertEqual(_leakage_check(store, cfg), (len(tr & va), len(tr & te), len(va & te)))


class DatasetSplitTest(unittest.TestCase):
    """``FinancialMarketDataset.index_split`` is the Megatron ``Split`` the loader builder expects."""

    @staticmethod
    def _write_split(path: Path, n: int = 3) -> None:
        torch.save(
            {
                "patches": torch.zeros(n, 2, 1, 2),
                "target_return": torch.zeros(n, 2, 1),
                "target_volshare": torch.zeros(n, 2, 1),
                "inst_id": torch.zeros(n, dtype=torch.int64),
                "patch_tod": torch.zeros(n, 2, dtype=torch.int64),
            },
            path,
        )

    def test_index_split_follows_split_name(self):
        from examples.financial_timeseries.ts_dataset import FinancialMarketDataset

        expected = {
            "train": Split.train,
            "val": Split.valid,
            "valid": Split.valid,
            "test": Split.test,
        }
        with tempfile.TemporaryDirectory() as d:
            for name, split in expected.items():
                path = Path(d) / f"{name}.pt"
                self._write_split(path)
                ds = FinancialMarketDataset(path, name, mmap=False)
                self.assertIs(ds.index_split, split)
                self.assertEqual(len(ds), 3)

    def test_unknown_split_name_is_rejected(self):
        from examples.financial_timeseries.ts_dataset import FinancialMarketDataset

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "holdout.pt"
            self._write_split(path)
            with self.assertRaises(ValueError):
                FinancialMarketDataset(path, "holdout", mmap=False)


if __name__ == "__main__":
    unittest.main()
