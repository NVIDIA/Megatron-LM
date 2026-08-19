# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for the Triton autotune policy.

These exercise selection, the tuned table and policy resolution directly, so
they need neither a GPU nor a working triton install.
"""

import json

import pytest

from megatron.core.tuning import selection
from megatron.core.tuning import table as table_mod
from megatron.core.tuning.policy import AutotunePolicy, set_deterministic_mode


class _Config:
    """Stand-in for ``triton.Config``."""

    def __init__(self, kwargs, num_warps=4, num_stages=1, pre_hook=None):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.pre_hook = pre_hook


class _Autotuner:
    """Stand-in for ``triton.runtime.autotuner.Autotuner``."""

    def __init__(self, configs, name="_fake_kernel", module="mamba_ssm.ops.triton.fake"):
        self.configs = configs
        self.arg_names = ("x", "seqlen")
        self.keys = ("seqlen",)
        self.base_fn = type("_Fn", (), {"__name__": name, "__module__": module})()


CHEAP = _Config({"BLOCK_SIZE_M": 32}, num_warps=4, num_stages=1)
FAST = _Config({"BLOCK_SIZE_M": 128}, num_warps=8, num_stages=2, pre_hook=lambda *_: None)


@pytest.fixture
def restore_state():
    saved = set(selection._untuned_kernels_warned)
    selection._untuned_kernels_warned.clear()
    yield
    selection._untuned_kernels_warned.clear()
    selection._untuned_kernels_warned.update(saved)


def _table(entries):
    return table_mod.TunedTable("sm100", entries)


def test_falls_back_to_cheapest_when_untuned(restore_state):
    """With no table entry the cheapest config wins: deterministic, never timed."""
    tuner = _Autotuner([FAST, CHEAP])
    with pytest.warns(UserWarning, match="No pre-tuned config"):
        chosen = selection.deterministic_choice(tuner, tuner.configs, (None, 8192), {})
    assert chosen is CHEAP


def test_prefers_tuned_config_and_preserves_identity(restore_state):
    """A table hit returns the ORIGINAL Config object, keeping pre_hook intact."""
    table = _table(
        {"_fake_kernel": {"*": {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 8, "num_stages": 2}}}
    )
    tuner = _Autotuner([FAST, CHEAP])
    chosen = selection.deterministic_choice(tuner, tuner.configs, (None, 8192), {}, table=table)
    assert chosen is FAST
    assert chosen.pre_hook is not None


def test_lookup_matches_against_live_candidates():
    """A recorded entry resolves to a live candidate, or misses; it never fabricates."""
    table = _table(
        {"_fake_kernel": {"*": {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 8, "num_stages": 2}}}
    )
    assert table.lookup("_fake_kernel", "seqlen=8192", [CHEAP, FAST]) is FAST
    assert table.lookup("_other_kernel", "seqlen=8192", [CHEAP, FAST]) is None
    # A stale entry that matches no live candidate must miss rather than fabricate
    # one, so a config list that changed upstream cannot produce a bad launch.
    assert table.lookup("_fake_kernel", "seqlen=8192", [CHEAP]) is None


def test_on_miss_error_refuses_to_guess(restore_state):
    tuner = _Autotuner([FAST, CHEAP])
    with pytest.raises(RuntimeError, match="No tuned config"):
        selection.deterministic_choice(tuner, tuner.configs, (None, 8192), {}, on_miss="error")


def test_tuning_key_is_stable_and_shape_sensitive():
    tuner = _Autotuner([CHEAP])
    assert selection.tuning_key(tuner, (None, 8192), {}) == selection.tuning_key(
        tuner, (None, 8192), {}
    )
    assert selection.tuning_key(tuner, (None, 8192), {}) != selection.tuning_key(
        tuner, (None, 4096), {}
    )


def test_chaos_choice_is_per_rank_but_reproducible(monkeypatch):
    """The positive control must differ across ranks and repeat within one."""
    tuner = _Autotuner([CHEAP, FAST])
    monkeypatch.setenv("RANK", "0")
    first = selection.chaos_choice(tuner, tuner.configs, (None, 8192), {})
    assert selection.chaos_choice(tuner, tuner.configs, (None, 8192), {}) is first
    picks = set()
    for rank in range(16):
        monkeypatch.setenv("RANK", str(rank))
        picks.add(id(selection.chaos_choice(tuner, tuner.configs, (None, 8192), {})))
    assert len(picks) > 1


def test_autotune_configs_pins_under_deterministic_mode():
    """The in-tree kernel path pins regardless of TRITON_CACHE_AUTOTUNING."""
    set_deterministic_mode(True)
    try:
        assert selection.autotune_configs([FAST, CHEAP]) == [CHEAP]
    finally:
        set_deterministic_mode(None)


def test_inert_outside_deterministic_mode():
    set_deterministic_mode(False)
    try:
        assert selection.autotune_configs([FAST, CHEAP]) == [FAST, CHEAP]
    finally:
        set_deterministic_mode(None)


def test_policy_from_env_modes(monkeypatch):
    """Deterministic mode implies pinned; an explicit mode or a record path wins."""
    for var in ("MCORE_AUTOTUNE_MODE", "MCORE_AUTOTUNE_RECORD", "MCORE_DET_TUNE_RECORD"):
        monkeypatch.delenv(var, raising=False)
    set_deterministic_mode(False)
    try:
        assert AutotunePolicy.from_env().mode == "auto"
        set_deterministic_mode(True)
        assert AutotunePolicy.from_env().mode == "pinned"
        monkeypatch.setenv("MCORE_AUTOTUNE_RECORD", "/tmp/rec")
        assert AutotunePolicy.from_env().mode == "record"
        monkeypatch.setenv("MCORE_AUTOTUNE_MODE", "auto")
        assert AutotunePolicy.from_env().mode == "auto"
    finally:
        set_deterministic_mode(None)


def test_policy_accepts_legacy_env_names(monkeypatch):
    """Old DET_AUTOTUNE_* names keep working so existing scripts do not break."""
    monkeypatch.delenv("MCORE_AUTOTUNE_MODULES", raising=False)
    monkeypatch.setenv("DET_AUTOTUNE_PIN_MODULES", "mamba_ssm")
    monkeypatch.setenv("DET_AUTOTUNE_ENUMERATE", "1")
    policy = AutotunePolicy.from_env()
    assert policy.modules == ("mamba_ssm",)
    assert policy.enumerate_autotuners
    assert policy.intercepts


def test_merge_records_uses_majority_vote(tmp_path):
    """Ranks disagree; that disagreement is the variance the table removes."""
    entry = {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 8, "num_stages": 2}
    odd = {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 99, "num_stages": 2}
    paths = []
    for rank, config in enumerate((entry, entry, odd)):
        path = tmp_path / f"rec.rank{rank}.json"
        path.write_text(json.dumps({"sm100": {"_fake_kernel": {"*": config}}}))
        paths.append(str(path))
    merged = table_mod.merge_records(paths)
    assert merged["sm100"]["_fake_kernel"]["*"]["num_warps"] == 8
    assert len(table_mod.disagreement_report(paths)) == 1


def test_table_write_and_load_round_trip(tmp_path):
    kernels = {
        "_fake_kernel": {"*": {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 8, "num_stages": 2}}
    }
    table_mod.write("sm100", kernels, tmp_path / "sm100.json", source="unit test")
    loaded = table_mod.load("sm100", table_path=[tmp_path])
    assert loaded.kernels == kernels
    assert loaded.provenance["source"] == "unit test"


def test_packaged_tables_are_loadable():
    """The shipped tables must parse and be non-empty for the architectures we ship."""
    for arch in ("sm100", "sm103"):
        table = table_mod.load(arch)
        assert table, f"packaged table for {arch} is empty"
        assert table.provenance["arch"] == arch


def test_verify_cadence_requires_the_interception(monkeypatch):
    """MCORE_AUTOTUNE_VERIFY alone installs the patch: the check reads its log."""
    for var in ("MCORE_AUTOTUNE_MODE", "MCORE_AUTOTUNE_RECORD", "MCORE_DET_TUNE_RECORD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MCORE_AUTOTUNE_VERIFY", "5")
    set_deterministic_mode(False)
    try:
        policy = AutotunePolicy.from_env()
        assert policy.verify_every == 5
        # auto mode leaves the choice to Triton, but the interception is still
        # what records it, so there is nothing to compare across ranks without it.
        assert policy.mode == "auto"
        assert policy.intercepts
    finally:
        set_deterministic_mode(None)


def test_maybe_verify_choices_honours_the_cadence(monkeypatch):
    """The training loop calls every step; the policy decides which ones check."""
    from megatron.core.tuning import interception

    calls = []

    def fake_verify(group=None):
        calls.append(group)
        return True

    monkeypatch.setattr(interception, "verify_choices", fake_verify)

    # No policy installed, and a policy that did not ask: None means "not checked",
    # which a caller must be able to tell apart from "checked and agreed".
    monkeypatch.setattr(interception, "_policy", None)
    assert interception.maybe_verify_choices(1) is None

    monkeypatch.setattr(interception, "_policy", AutotunePolicy(verify_every=0))
    assert interception.maybe_verify_choices(1) is None
    assert calls == []

    monkeypatch.setattr(interception, "_policy", AutotunePolicy(verify_every=3))
    assert interception.maybe_verify_choices(1) is None
    assert interception.maybe_verify_choices(2) is None
    assert interception.maybe_verify_choices(3) is True
    assert interception.maybe_verify_choices(6) is True
    assert calls == [None, None]


def test_verify_choices_sizes_the_gather_to_its_group(monkeypatch):
    """all_gather_object needs one slot per group member, not per world rank."""
    import torch

    from megatron.core.tuning import interception

    lengths = []

    def fake_world_size(group=None):
        return 8 if group is None else 2

    def fake_all_gather_object(object_list, obj, group=None):
        # This is the assertion torch itself makes; a subgroup used to fail it.
        assert len(object_list) == fake_world_size(group=group)
        lengths.append(len(object_list))
        object_list[:] = [obj] * len(object_list)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", fake_world_size)
    monkeypatch.setattr(torch.distributed, "all_gather_object", fake_all_gather_object)

    subgroup = object()
    assert interception.verify_choices(group=subgroup) is True
    assert lengths == [2]
    assert interception.verify_choices() is True
    assert lengths == [2, 8]
