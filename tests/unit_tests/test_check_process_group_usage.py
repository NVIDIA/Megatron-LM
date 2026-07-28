# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the global process-group usage ratchet (``tools/check_process_group_usage.py``).

These exercise the checker's own logic only -- no torch, no distributed init. (The enclosing
``tests.unit_tests`` package imports torch, so they run as part of the normal unit-test suite;
the checker itself is stdlib-only and the CI workflow invokes it directly, without pytest.)
"""

import importlib.util
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "tools" / "check_process_group_usage.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_process_group_usage", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load()


def _hits(checker, tmp_path, source):
    f = tmp_path / "sample.py"
    f.write_text(source, encoding="utf-8")
    return [d for _, _, d in checker._violations_in(f)]


def test_flags_qualified_accessor(checker, tmp_path):
    hits = _hits(
        checker,
        tmp_path,
        "from megatron.core import parallel_state\n"
        "g = parallel_state.get_tensor_model_parallel_group()\n",
    )
    assert "parallel_state.get_tensor_model_parallel_group" in hits


def test_flags_directly_imported_accessor(checker, tmp_path):
    hits = _hits(
        checker,
        tmp_path,
        "from megatron.core.parallel_state import "
        "get_data_parallel_group\n"
        "g = get_data_parallel_group()\n",
    )
    assert "get_data_parallel_group" in hits


def test_flags_the_use_mpu_shim(checker, tmp_path):
    """The PR #5916 'lateral move' must not pass as a migration."""
    hits = _hits(checker, tmp_path, "pgs = ProcessGroupCollection.use_mpu_process_groups()\n")
    assert "use_mpu_process_groups" in hits


def test_flags_rank_and_world_size_accessors(checker, tmp_path):
    hits = _hits(
        checker,
        tmp_path,
        "from megatron.core import parallel_state\n"
        "r = parallel_state.get_tensor_model_parallel_rank()\n"
        "n = parallel_state.get_data_parallel_world_size()\n",
    )
    assert len(hits) == 2


def test_ignores_the_long_term_surface(checker, tmp_path):
    """initialize / destroy / is_initialized are not deprecated."""
    hits = _hits(
        checker,
        tmp_path,
        "from megatron.core import parallel_state\n"
        "parallel_state.initialize_model_parallel()\n"
        "parallel_state.destroy_model_parallel()\n"
        "parallel_state.is_initialized()\n",
    )
    assert hits == []


def test_ignores_tier4_globals_with_no_replacement(checker, tmp_path):
    """Virtual-pipeline and memory-buffer globals have no ProcessGroupCollection equivalent."""
    hits = _hits(
        checker,
        tmp_path,
        "from megatron.core import parallel_state\n"
        "parallel_state.get_virtual_pipeline_model_parallel_rank()\n"
        "parallel_state.get_global_memory_buffer()\n"
        "parallel_state.get_nccl_options()\n",
    )
    assert hits == []


def test_does_not_flag_unrelated_getters(checker, tmp_path):
    hits = _hits(checker, tmp_path, "x = config.get_thing_group()\n" "y = some_object.get_rank()\n")
    assert hits == []


def test_allowlist_matches_the_tree(checker):
    """The committed allowlist must describe reality, or the ratchet is meaningless."""
    found = checker.scan()
    allowed = checker._load_allowlist()
    new = {f: sorted(set(h) - set(allowed.get(f, []))) for f, h in found.items()}
    new = {f: h for f, h in new.items() if h}
    assert not new, (
        f"new global process-group reads not in the allowlist: {new}. "
        "megatron/core must not read process groups from parallel_state."
    )


def test_allowlist_has_no_stale_entries(checker):
    """The allowlist must only shrink; stale entries mean it was not refreshed after a removal."""
    found = checker.scan()
    allowed = checker._load_allowlist()
    stale = {f: sorted(set(h) - set(found.get(f, []))) for f, h in allowed.items()}
    stale = {f: h for f, h in stale.items() if h}
    assert not stale, (
        f"allowlist references sites that no longer exist: {stale}. "
        "Refresh with `python tools/check_process_group_usage.py --update`."
    )
