# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from pathlib import Path

import pytest

from megatron.training.distillation import utils as distillation_utils
from megatron.training.distillation.utils import (
    STALE_TAR_SUFFIX,
    detect_saved_dp_size,
    quarantine_contained_tars,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"")


# ---------------------------------------------------------------------------
# quarantine_contained_tars
# ---------------------------------------------------------------------------


def test_quarantine_contains_fully_contained_shard(tmp_path):
    old = tmp_path / "dp0__10-20.tar"
    new = tmp_path / "dp0__0-30.tar"
    _touch(old)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(old), str(new)])

    assert quarantined == [(str(old), str(old) + STALE_TAR_SUFFIX)]
    assert not old.exists()
    assert Path(str(old) + STALE_TAR_SUFFIX).exists()


def test_quarantine_skips_partial_overlap(tmp_path):
    old = tmp_path / "dp0__10-25.tar"
    new = tmp_path / "dp0__0-20.tar"
    _touch(old)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(old), str(new)])

    assert quarantined == []
    assert old.exists()


def test_quarantine_skips_exact_path_duplicate(tmp_path):
    new = tmp_path / "dp0__0-20.tar"
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(new)])

    assert quarantined == []
    assert new.exists()


def test_quarantine_ignores_non_v2_shards(tmp_path):
    legacy = tmp_path / "cp0_dp0__5.tar"
    new = tmp_path / "dp0__0-30.tar"
    _touch(legacy)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(legacy), str(new)])

    assert quarantined == []
    assert legacy.exists()


def test_quarantine_known_tars_is_not_dp_rank_scoped_by_the_function_itself(tmp_path):
    """Regression/documentation test: quarantine_contained_tars does not filter
    known_tars by DP rank -- per its docstring, the caller must pre-scope the
    listing to tar_path's own DP rank prefix. Passing an unscoped listing risks
    quarantining another DP rank's shard, as demonstrated here.
    """
    other_rank = tmp_path / "dp1__0-30.tar"  # different DP rank, same range
    new = tmp_path / "dp0__0-30.tar"
    _touch(other_rank)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(other_rank), str(new)])

    assert quarantined == [(str(other_rank), str(other_rank) + STALE_TAR_SUFFIX)]


def test_quarantine_boundary_adjacent_not_contained(tmp_path):
    # old ends exactly where new starts: not overlapping, and per the
    # containment predicate (old_start >= new_start and old_end <= new_end)
    # this is also not contained in new's range -- should be left alone.
    old = tmp_path / "dp0__0-10.tar"
    new = tmp_path / "dp0__10-20.tar"
    _touch(old)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(old), str(new)])

    assert quarantined == []
    assert old.exists()


def test_quarantine_boundary_touching_ranges_are_contained(tmp_path):
    # old's range touches new's range at both ends but is a strict subset
    # elsewhere: [5, 20) is contained in [0, 20) since old_end == new_end.
    old = tmp_path / "dp0__5-20.tar"
    new = tmp_path / "dp0__0-20.tar"
    _touch(old)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(old), str(new)])

    assert quarantined == [(str(old), str(old) + STALE_TAR_SUFFIX)]


def test_quarantine_uses_known_tars_without_globbing(tmp_path, monkeypatch):
    old = tmp_path / "dp0__10-20.tar"
    new = tmp_path / "dp0__0-30.tar"
    _touch(old)
    _touch(new)

    def _fail_glob(*args, **kwargs):
        raise AssertionError("storage_glob should not be called when known_tars is provided")

    monkeypatch.setattr(distillation_utils, "storage_glob", _fail_glob)

    quarantined = quarantine_contained_tars(str(new), known_tars=[str(old), str(new)])

    assert quarantined == [(str(old), str(old) + STALE_TAR_SUFFIX)]


def test_quarantine_globs_when_known_tars_omitted(tmp_path):
    old = tmp_path / "dp0__10-20.tar"
    new = tmp_path / "dp0__0-30.tar"
    _touch(old)
    _touch(new)

    quarantined = quarantine_contained_tars(str(new))

    assert quarantined == [(str(old), str(old) + STALE_TAR_SUFFIX)]


def test_quarantine_non_v2_tar_path_is_noop(tmp_path):
    legacy_path = tmp_path / "cp0_dp0__5.tar"
    _touch(legacy_path)

    assert quarantine_contained_tars(str(legacy_path), known_tars=[]) == []


# ---------------------------------------------------------------------------
# detect_saved_dp_size
# ---------------------------------------------------------------------------


def test_detect_saved_dp_size_from_v1_and_v2_names(tmp_path):
    _touch(tmp_path / "cp0_dp0__5.tar")
    _touch(tmp_path / "cp0_dp2__5.tar")
    _touch(tmp_path / "dp1__0-30.tar")

    assert detect_saved_dp_size(str(tmp_path)) == 3


def test_detect_saved_dp_size_rejects_nonzero_cp_legacy_shards(tmp_path):
    _touch(tmp_path / "cp1_dp0__5.tar")

    with pytest.raises(ValueError, match="CP rank"):
        detect_saved_dp_size(str(tmp_path))


def test_detect_saved_dp_size_empty_dir_returns_none(tmp_path):
    assert detect_saved_dp_size(str(tmp_path)) is None
