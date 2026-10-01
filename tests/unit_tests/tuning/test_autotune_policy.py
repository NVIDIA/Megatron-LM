# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU tests for the Triton autotune policy.

These exercise selection, the tuned table and policy resolution directly, so
they need neither a GPU nor a working triton install.
"""

import importlib
import json
import logging
import os
from dataclasses import FrozenInstanceError

import pytest

from megatron.core import _rank_utils
from megatron.core.tuning import selection
from megatron.core.tuning import table as table_mod
from megatron.core.tuning.policy import (
    AutotunePolicy,
    set_deterministic_mode,
    use_deterministic_mode,
)


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


def test_falls_back_to_cheapest_when_untuned(restore_state, caplog):
    """With no table entry the cheapest config wins: deterministic, never timed."""
    tuner = _Autotuner([FAST, CHEAP])
    with caplog.at_level(logging.WARNING, logger=selection.__name__):
        chosen = selection.deterministic_choice(tuner, tuner.configs, (None, 8192), {})
        selection.deterministic_choice(tuner, tuner.configs, (None, 4096), {})
    assert chosen is CHEAP
    # Logged (not warnings.warn, which launchers silence off rank 0), once per kernel.
    assert caplog.text.count("No pre-tuned config") == 1
    assert "mamba_ssm.ops.triton.fake._fake_kernel" in caplog.text


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
    # The rank comes from the process group or the launcher, not only RANK.
    monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda: 0)
    first = selection.chaos_choice(tuner, tuner.configs, (None, 8192), {})
    assert selection.chaos_choice(tuner, tuner.configs, (None, 8192), {}) is first
    picks = set()
    for rank in range(16):
        monkeypatch.setattr(_rank_utils, "safe_get_rank", lambda rank=rank: rank)
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


@pytest.mark.parametrize("deterministic", [False, True])
def test_deterministic_mode_ignores_mamba_environment(monkeypatch, deterministic):
    import torch

    monkeypatch.setenv("MAMBA_DETERMINISTIC", "0" if deterministic else "1")
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: deterministic)
    set_deterministic_mode(None)
    assert use_deterministic_mode() is deterministic
    set_deterministic_mode(not deterministic)
    try:
        assert use_deterministic_mode() is (not deterministic)
    finally:
        set_deterministic_mode(None)


def test_policy_resolves_explicit_modes():
    """Deterministic mode implies pinned; an explicit mode or a record path wins."""
    set_deterministic_mode(False)
    try:
        assert AutotunePolicy().resolve().mode == "auto"
        assert AutotunePolicy().resolve(deterministic=True).mode == "pinned"
        assert AutotunePolicy(record_path="/tmp/rec").resolve(deterministic=True).mode == "record"
        assert AutotunePolicy(mode="auto").resolve(deterministic=True).mode == "auto"
        assert AutotunePolicy(mode="pinned", record_path="/tmp/rec").resolve().mode == "pinned"
        set_deterministic_mode(True)
        assert AutotunePolicy().resolve().mode == "pinned"
    finally:
        set_deterministic_mode(None)


def test_policy_ignores_tuning_environment(monkeypatch):
    """Tuning controls, including former aliases, are exclusively configuration values."""
    for name, value in {
        "MCORE_AUTOTUNE_MODE": "record",
        "MCORE_AUTOTUNE_RECORD": "/tmp/ignored",
        "MCORE_DET_TUNE_RECORD": "/tmp/ignored-legacy",
        "MCORE_AUTOTUNE_MODULES": "ignored",
        "DET_AUTOTUNE_PIN_MODULES": "ignored-legacy",
        "MCORE_AUTOTUNE_TABLE_PATH": "/tmp/ignored-table",
        "MCORE_AUTOTUNE_ON_MISS": "error",
        "MCORE_AUTOTUNE_VERIFY": "17",
        "DET_AUTOTUNE_VERIFY": "19",
        "MCORE_AUTOTUNE_VERIFY_STRICT": "1",
        "DET_AUTOTUNE_VERIFY_STRICT": "1",
        "MCORE_AUTOTUNE_ENUMERATE": "1",
        "DET_AUTOTUNE_ENUMERATE": "1",
        "MCORE_AUTOTUNE_CHAOS": "1",
        "DET_AUTOTUNE_CHAOS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    set_deterministic_mode(False)
    try:
        assert AutotunePolicy().resolve() == AutotunePolicy(mode="auto")
    finally:
        set_deterministic_mode(None)


def test_policy_normalizes_explicit_options_without_mutable_aliases(tmp_path):
    modules = ["custom.kernels"]
    paths = [tmp_path]
    blocks = {"BLOCK_SIZE_M": 128}
    policy = AutotunePolicy(
        modules=modules,
        table_path=paths,
        record_path=tmp_path / "record",
        on_miss="error",
        verify_every=5,
        verify_strict=True,
        enumerate_autotuners=True,
        chaos=True,
        block_sizes=blocks,
    ).resolve()
    assert policy == AutotunePolicy(
        mode="record",
        modules=("custom.kernels",),
        table_path=(str(tmp_path),),
        record_path=str(tmp_path / "record"),
        on_miss="error",
        verify_every=5,
        verify_strict=True,
        enumerate_autotuners=True,
        chaos=True,
        block_sizes=(("BLOCK_SIZE_M", 128),),
    )
    modules.append("other.kernels")
    paths.append(tmp_path / "other")
    blocks["BLOCK_SIZE_M"] = 32
    assert policy.modules == ("custom.kernels",)
    assert policy.table_path == (str(tmp_path),)
    assert policy.block_sizes == (("BLOCK_SIZE_M", 128),)
    with pytest.raises(FrozenInstanceError):
        policy.mode = "auto"


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "typo"},
        {"on_miss": "typo"},
        {"verify_every": -1},
        {"mode": "record"},
        {"block_sizes": (("NOT_A_BLOCK", 32),)},
        {"block_sizes": (("BLOCK_SIZE", 0),)},
        {"block_sizes": (("BLOCK_SIZE", -1),)},
        {"block_sizes": (("BLOCK_SIZE", "32"),)},
    ],
)
def test_policy_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        AutotunePolicy(**options)


def test_explicit_block_override_satisfies_strict_selection():
    tuner = _Autotuner([CHEAP, FAST])
    assert (
        selection.deterministic_choice(
            tuner, tuner.configs, (), {}, on_miss="error", block_sizes=(("BLOCK_SIZE_M", 128),)
        )
        is FAST
    )


def test_block_size_environment_is_ignored(monkeypatch, restore_state):
    monkeypatch.setenv("TRITON_AUTOTUNE_BLOCK_SIZE_M", "not-an-integer")
    tuner = _Autotuner([CHEAP, FAST])
    assert selection.deterministic_choice(tuner, tuner.configs, (), {}) is CHEAP
    set_deterministic_mode(True)
    try:
        assert selection.autotune_configs([FAST, CHEAP]) == [CHEAP]
    finally:
        set_deterministic_mode(None)


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


def test_table_search_uses_only_explicit_paths(monkeypatch, tmp_path):
    kernels = {"_fake_kernel": {"*": selection.config_data(FAST)}}
    table_mod.write("test_arch", kernels, tmp_path / "test_arch.json")
    monkeypatch.setenv("MCORE_AUTOTUNE_TABLE_PATH", str(tmp_path))
    assert not table_mod.load("test_arch")
    assert table_mod.load("test_arch", table_path=(tmp_path,)).kernels == kernels


def test_packaged_tables_are_loadable():
    """The shipped tables must parse and be non-empty for the architectures we ship."""
    for arch in ("sm100", "sm103"):
        table = table_mod.load(arch)
        assert table, f"packaged table for {arch} is empty"
        assert table.provenance["arch"] == arch


def test_packaged_entries_name_live_candidates():
    """Each packaged entry for an in-tree kernel must name one of its current candidates.

    A stale entry does not fail at run time: it misses, and the kernel quietly falls
    back to the cheapest candidate.
    """
    autotuner = pytest.importorskip("triton.runtime.autotuner")
    for arch in ("sm100", "sm103"):
        table = table_mod.load(arch)
        for kernel, entries in table.kernels.items():
            if not kernel.startswith("megatron.core."):
                continue
            module_name, name = kernel.rsplit(".", 1)
            tuner = getattr(importlib.import_module(module_name), name)
            if not isinstance(tuner, autotuner.Autotuner):
                pytest.skip(f"{kernel} is not built with this triton install")
            assert f"{selection.kernel_module(tuner)}.{selection.kernel_name(tuner)}" == kernel
            for key in entries:
                assert table.lookup(kernel, key, tuner.configs) is not None, (arch, kernel, key)


def test_verify_cadence_requires_the_interception():
    """Explicit verification cadence installs the adapter so choices are recorded."""
    set_deterministic_mode(False)
    try:
        policy = AutotunePolicy(verify_every=5).resolve()
        assert policy.verify_every == 5
        # auto mode leaves the choice to Triton, but the interception is still
        # what records it, so there is nothing to compare across ranks without it.
        assert policy.mode == "auto"
        assert policy.intercepts
    finally:
        set_deterministic_mode(None)


def test_maybe_verify_choices_honours_the_cadence(isolated_policy, monkeypatch):
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


def test_verify_choices_sizes_the_gather_to_its_group(isolated_policy, monkeypatch):
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


def test_default_scope_covers_in_tree_kernels_and_exempts_invariant_ones():
    policy = AutotunePolicy()
    assert "megatron.core" in policy.modules
    for kernel in (
        "transformer_engine.common.triton.permutation._permute_kernel",
        "transformer_engine.common.triton.permutation._unpermute_kernel",
        "transformer_engine.common.triton.permutation._sort_chunks_by_map_kernel",
    ):
        assert kernel in policy.config_invariant
    # Kernels whose config changes a reduction, or the rounding of arithmetic, stay pinned.
    for kernel in (
        "megatron.core.fusions.fused_mla_yarn_rope_apply._mla_rope_bwd_kv_split_kernel",
        "megatron.core.fusions.fused_mla_yarn_rope_apply._mla_rope_bwd_inplace_kernel",
        "megatron.core.fusions.fused_mhc_kernels._triton_hpb_bwd_g_hp_hr_kernel",
        "transformer_engine.common.triton.permutation._unpermute_bwd_with_merging_probs_kernel",
    ):
        assert kernel not in policy.config_invariant


def test_record_path_is_expanded_and_must_be_a_prefix(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert AutotunePolicy(record_path="~/rec").record_path == os.path.join(str(tmp_path), "rec")
    assert AutotunePolicy(table_path=("~/tables",)).table_path == (
        os.path.join(str(tmp_path), "tables"),
    )
    with pytest.raises(ValueError, match="file prefix"):
        AutotunePolicy(record_path=str(tmp_path) + "/")


@pytest.mark.parametrize(
    "options, error",
    [
        ({"chaos": "false"}, TypeError),
        ({"verify_strict": "no"}, TypeError),
        ({"enumerate_autotuners": 1}, TypeError),
        ({"verify_every": "10"}, TypeError),
        ({"verify_every": True}, TypeError),
        ({"modules": None}, TypeError),
        ({"table_path": [None]}, TypeError),
        ({"block_sizes": None}, TypeError),
        ({"block_sizes": {"BLOCK-C": 64}}, ValueError),
    ],
)
def test_policy_rejects_mistyped_options(options, error):
    with pytest.raises(error):
        AutotunePolicy(**options)


def test_policy_from_mapping_skips_nulls_and_rejects_unknown_keys():
    policy = AutotunePolicy.from_mapping(
        {"mode": "pinned", "table_path": None, "verify_every": None, "modules": ["pkg"]}
    )
    assert policy == AutotunePolicy(mode="pinned", modules=("pkg",))
    with pytest.raises(TypeError, match="Unknown AutotunePolicy option"):
        AutotunePolicy.from_mapping({"enumerate": True})


def test_table_load_skips_mislabelled_and_missing_locations(tmp_path, caplog):
    kernels = {"_fake_kernel": {"*": selection.config_data(FAST)}}
    table_mod.write("sm103", kernels, tmp_path / "sm100.json")
    with caplog.at_level(logging.WARNING, logger=table_mod.__name__):
        loaded = table_mod.load("sm100", table_path=[tmp_path, tmp_path / "missing"])
    assert loaded.provenance.get("arch") != "sm103"
    assert "holds a table for sm103, not sm100" in caplog.text
    assert "does not exist" in caplog.text


def _write_recording(path, arch="sm100", kernel="_fake_kernel", config=None):
    config = config or {"kwargs": {"BLOCK_SIZE_M": 128}, "num_warps": 8, "num_stages": 2}
    path.write_text(json.dumps({arch: {kernel: {"*": config}}}))
    return str(path)


def test_merge_cli_reports_unusable_inputs(tmp_path, capsys):
    from megatron.core.tuning.__main__ import main

    good = _write_recording(tmp_path / "rec.rank0.json")
    truncated = tmp_path / "rec.rank1.json"
    truncated.write_text('{"sm100": {"_fake_kernel": {"*": {"kwargs"')
    table = tmp_path / "sm100.json"
    table_mod.write("sm100", {"_fake_kernel": {}}, table)

    assert main(["merge", good, str(truncated), "-o", str(tmp_path / "out.json")]) == 1
    assert "rec.rank1.json" in capsys.readouterr().err
    assert main(["merge", str(table), good, "-o", str(tmp_path / "out.json")]) == 1
    assert "tuned table, not a per-rank recording" in capsys.readouterr().err
    assert main(["merge", good, "--arch", "sm103", "-o", str(tmp_path / "out.json")]) == 1
    assert "recordings cover ['sm100']" in capsys.readouterr().err
    assert main(["report", good]) == 0
    assert main(["merge", good, "-o", str(tmp_path / "tables" / "sm100.json")]) == 0
    recorded = json.loads((tmp_path / "rec.rank0.json").read_text())["sm100"]
    assert table_mod.load("sm100", table_path=[tmp_path / "tables"]).kernels == recorded


def test_mamba_environment_is_reported_when_ignored(monkeypatch, caplog):
    monkeypatch.setattr(selection, "_mamba_env_warned", False)
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
    with caplog.at_level(logging.WARNING, logger=selection.__name__):
        selection.warn_if_mamba_env_ignored(AutotunePolicy(mode="pinned"))
        assert "MAMBA_DETERMINISTIC" not in caplog.text
        selection.warn_if_mamba_env_ignored(AutotunePolicy(mode="auto"))
        selection.warn_if_mamba_env_ignored(AutotunePolicy(mode="auto"))
    assert caplog.text.count("MAMBA_DETERMINISTIC=1 only affects") == 1
