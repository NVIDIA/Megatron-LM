# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The policy applied to cudnn-frontend's cuTile launch autotune, against a stand-in for its hook module."""

import os
import subprocess
import sys
import types
from types import SimpleNamespace

import pytest

from megatron.core._rank_utils import safe_get_rank
from megatron.core.tuning import cutile, interception, selection, table
from megatron.core.tuning.policy import AutotunePolicy

COMMON = "cudnn.linear_attention.cutile.kernels.common"
GDN = "cudnn.linear_attention.cutile.kernels.gdn"


@pytest.fixture
def fake_common(monkeypatch):
    """A stand-in ``common`` module with the two hooks, reachable through the cudnn package path."""
    parent = None
    for name in (
        "cudnn",
        "cudnn.linear_attention",
        "cudnn.linear_attention.cutile",
        GDN.rsplit(".", 1)[0],
    ):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
        if parent is not None:
            setattr(parent, name.rsplit(".", 1)[1], package)
        parent = package
    common = types.ModuleType(COMMON)
    common.select_hints = None
    common.record_hint = None
    common.launch_hint_cache = {}
    monkeypatch.setitem(sys.modules, COMMON, common)
    parent.common = common
    monkeypatch.setattr(cutile, "_unhookable_warned", False)
    return common


def kernel(name="chunk_fwd_kernel_o", module=GDN):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = module
    return SimpleNamespace(_pyfunc=fn)


CONFIGS = [SimpleNamespace(occupancy=occ, num_worker_warps=nww) for occ in (1, 4) for nww in (4, 8)]
KEY = (16, 16, 128, 128, 64, 64, 1, "torch.bfloat16")


def launch(common, kern, key=KEY, configs=CONFIGS, timed=CONFIGS[-1]):
    """What ``autotuned_launch`` does with the hooks: ask, else time and report."""
    chosen = common.select_hints(kern, key, configs) if common.select_hints else None
    if chosen is None:
        chosen = timed
        if common.record_hint:
            common.record_hint(kern, key, chosen)
    return chosen


def test_auto_mode_leaves_the_hooks_unset(isolated_policy, fake_common):
    interception.install(AutotunePolicy(mode="auto"))
    assert fake_common.select_hints is None and fake_common.record_hint is None


def test_pinned_without_a_table_runs_the_first_candidate(isolated_policy, fake_common):
    interception.install(AutotunePolicy(mode="pinned"))
    assert launch(fake_common, kernel()) is CONFIGS[0]
    ((entry, value),) = interception.choice_log().items()
    assert entry == f"test_arch|{GDN}.chunk_fwd_kernel_o|" + "|".join(map(str, KEY))
    assert value == '{"num_worker_warps": 4, "occupancy": 1};pinned'


def test_pinned_runs_the_table_entry(isolated_policy, fake_common, tmp_path):
    table.write(
        "test_arch",
        {
            f"{GDN}.chunk_fwd_kernel_o": {
                "|".join(map(str, KEY)): {"occupancy": 4, "num_worker_warps": 8}
            }
        },
        tmp_path / "test_arch.json",
    )
    interception.install(
        AutotunePolicy(mode="pinned", table_path=(str(tmp_path),), on_miss="error")
    )
    assert launch(fake_common, kernel()) is CONFIGS[-1]
    with pytest.raises(RuntimeError, match="No tuned config for cuTile kernel"):
        launch(fake_common, kernel("chunk_bwd_kernel_dqkwg"))


def test_stale_table_entry_falls_back_to_the_first_candidate(
    isolated_policy, fake_common, tmp_path
):
    table.write(
        "test_arch",
        {f"{GDN}.chunk_fwd_kernel_o": {"*": {"occupancy": 2, "num_worker_warps": 16}}},
        tmp_path / "test_arch.json",
    )
    interception.install(AutotunePolicy(mode="pinned", table_path=(str(tmp_path),)))
    assert launch(fake_common, kernel()) is CONFIGS[0]


def test_record_captures_the_timed_winner_and_a_merged_table_pins_it(
    isolated_policy, fake_common, tmp_path
):
    interception.install(AutotunePolicy(mode="record", record_path=str(tmp_path / "rec")))
    assert launch(fake_common, kernel(), timed=CONFIGS[2]) is CONFIGS[2]
    interception._dump_records()
    recording_file = interception._record_file(str(tmp_path / "rec"), safe_get_rank())
    recording = table._read_recording(recording_file)
    assert recording == {
        "test_arch": {
            f"{GDN}.chunk_fwd_kernel_o": {
                "|".join(map(str, KEY)): {"num_worker_warps": 4, "occupancy": 4}
            }
        }
    }
    assert interception.choice_log()[
        f"test_arch|{GDN}.chunk_fwd_kernel_o|" + "|".join(map(str, KEY))
    ].endswith(";timed")

    merged = table.merge_records([recording_file])
    table.write("test_arch", merged["test_arch"], tmp_path / "tables" / "test_arch.json")
    interception.install(
        AutotunePolicy(mode="pinned", table_path=(str(tmp_path / "tables"),), on_miss="error")
    )
    assert launch(fake_common, kernel()) is CONFIGS[2]


def test_a_kernel_outside_the_scope_keeps_its_timed_pick(isolated_policy, fake_common):
    interception.install(AutotunePolicy(mode="pinned", modules=("mamba_ssm",)))
    assert launch(fake_common, kernel()) is CONFIGS[-1]
    assert interception.choice_log() == {}


def test_config_invariant_kernels_stay_timed_and_unlogged(isolated_policy, fake_common, tmp_path):
    """An invariant kernel times its pick, and that pick is neither recorded nor held to the agreement
    check: two ranks may time different winners by design."""
    invariant = (f"{GDN}.chunk_fwd_kernel_o",)
    interception.install(AutotunePolicy(mode="pinned", config_invariant=invariant))
    assert launch(fake_common, kernel()) is CONFIGS[-1]
    assert launch(fake_common, kernel("chunk_bwd_kernel_dqkwg")) is CONFIGS[0]
    assert [entry.split("|")[1] for entry in interception.choice_log()] == [
        f"{GDN}.chunk_bwd_kernel_dqkwg"
    ]
    interception.install(
        AutotunePolicy(mode="record", record_path=str(tmp_path / "rec"), config_invariant=invariant)
    )
    launch(fake_common, kernel())
    assert interception._tune_records == {}


def test_a_policy_change_drops_the_cached_launch_picks(isolated_policy, fake_common):
    """cudnn-frontend consults the hooks only on a launch-cache miss, so a pick timed under the previous
    policy must not outlive it on any rank."""
    fake_common.launch_hint_cache = {("chunk_fwd_kernel_o", KEY, 0): object()}
    interception.install(AutotunePolicy(mode="pinned"))
    assert fake_common.launch_hint_cache == {}
    fake_common.launch_hint_cache[("chunk_fwd_kernel_o", KEY, 0)] = object()
    interception.install(AutotunePolicy(mode="pinned"))  # the same policy: nothing to invalidate
    assert len(fake_common.launch_hint_cache) == 1
    interception.install(AutotunePolicy(mode="auto"))
    assert fake_common.launch_hint_cache == {}


def test_the_hooks_of_another_host_are_kept_and_restored(isolated_policy, fake_common):
    def host_select(kernel, cache_key, configs):
        return configs[1]

    fake_common.select_hints = host_select
    interception.install(AutotunePolicy(mode="auto"))
    assert fake_common.select_hints is host_select and fake_common.record_hint is None
    interception.install(AutotunePolicy(mode="pinned"))
    assert fake_common.select_hints is cutile.select and fake_common.record_hint is cutile.record
    interception.install(AutotunePolicy(mode="auto"))
    assert fake_common.select_hints is host_select and fake_common.record_hint is None


def test_a_hook_the_host_took_back_is_kept_and_the_other_restored(isolated_policy, fake_common):
    """Ownership is per hook: a host that replaces one hook under the policy keeps it, and
    reclaiming it later drops the launch picks that hook made."""

    def host_select(kernel, cache_key, configs):
        return configs[1]

    def host_record(kernel, cache_key, config):
        pass

    fake_common.select_hints, fake_common.record_hint = host_select, host_record
    interception.install(AutotunePolicy(mode="pinned"))
    assert (fake_common.select_hints, fake_common.record_hint) == (cutile.select, cutile.record)
    fake_common.select_hints = host_select  # the host takes its selector back
    fake_common.launch_hint_cache["picked-by-host"] = object()
    interception.install(AutotunePolicy(mode="pinned"))  # the same policy: reclaims the selector
    assert (fake_common.select_hints, fake_common.record_hint) == (cutile.select, cutile.record)
    assert fake_common.launch_hint_cache == {}
    fake_common.select_hints = host_select
    interception.install(AutotunePolicy(mode="auto"))
    assert (fake_common.select_hints, fake_common.record_hint) == (host_select, host_record)


def test_chaos_picks_differ_by_rank_and_repeat_per_rank(isolated_policy, fake_common, monkeypatch):
    interception.install(AutotunePolicy(mode="pinned", chaos=True))
    picks = {}
    for rank in range(8):
        monkeypatch.setattr("megatron.core._rank_utils.safe_get_rank", lambda rank=rank: rank)
        first = launch(fake_common, kernel())
        assert launch(fake_common, kernel()) is first
        assert first is selection.chaos_pick(
            CONFIGS, f"{GDN}.chunk_fwd_kernel_o", "|".join(map(str, KEY))
        )
        picks[rank] = CONFIGS.index(first)
    assert len(set(picks.values())) > 1, picks


def test_a_record_run_without_triton_writes_its_recording_at_exit(tmp_path):
    """The writer is registered at import, not by the Triton patch: a process that has no Triton
    and records only cuTile winners still writes its file when it exits normally."""
    script = f"""
import importlib, sys, types
from types import SimpleNamespace
# The real packages stay (flashinfer, imported by megatron.core, uses the real ``cudnn``);
# only a package that does not import, and the launch module itself, are stand-ins.
names = {list(COMMON.split(".")[:-1])!r}
for depth in range(1, len(names) + 1):
    name = ".".join(names[:depth])
    try:
        parent = importlib.import_module(name)
    except Exception:
        parent = sys.modules[name] = types.ModuleType(name)
        parent.__path__ = []
        if depth > 1:
            setattr(sys.modules[".".join(names[:depth - 1])], names[depth - 1], parent)
common = sys.modules[{COMMON!r}] = types.ModuleType({COMMON!r})
common.select_hints = common.record_hint = None
common.launch_hint_cache = {{}}
parent.common = common
from megatron.core.tuning import interception, selection
from megatron.core.tuning.policy import AutotunePolicy
interception._autotuner_class = lambda: None
selection.arch_tag = lambda: "test_arch"
interception.install(AutotunePolicy(mode="record", record_path={str(tmp_path / "rec")!r}))
fn = lambda: None
fn.__name__, fn.__module__ = "chunk_fwd_kernel_o", {GDN!r}
common.record_hint(SimpleNamespace(_pyfunc=fn), {KEY!r}, SimpleNamespace(occupancy=4, num_worker_warps=4))
"""
    # The child is rank 0 whatever rank this test process runs as under torchrun.
    env = {**os.environ, "RANK": "0"}
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120, env=env)
    recording = table._read_recording(interception._record_file(str(tmp_path / "rec"), 0))
    assert list(recording["test_arch"]) == [f"{GDN}.chunk_fwd_kernel_o"]


# Every cuTile kernel the GDN and KDA plans launch, by module under cudnn.linear_attention.cutile.kernels;
# ``common`` holds the Q/K normalization pair of ``use_qk_l2norm_in_kernel``, launched by both plans.
CUTILE_KERNELS = {
    "gdn": (
        "chunk_bwd_kernel_dqkwg",
        "chunk_bwd_kernel_dv_local",
        "chunk_fwd_kernel_o",
        "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
        "prepare_wy_repr_bwd_kernel",
        "recompute_w_u_fwd_kernel",
    ),
    "kda": (
        "chunk_gla_fwd_kernel_o",
        "chunk_kda_fwd_kernel_inter_diag_compute_solve",
        "chunk_kda_fwd_kernel_inter_solve_fused",
        "recompute_w_u_fwd_kda_kernel",
    ),
    "common": ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
}


def test_packaged_sm103_table_covers_every_cutile_kernel():
    """A pinned run on GB300 must never fall back to the first candidate: the packaged table carries a
    wildcard launch hint for every cuTile kernel the GDN and KDA plans launch, Q/K normalization included.
    """
    loaded = table.load("sm103", ())
    missing = []
    for module, names in CUTILE_KERNELS.items():
        for name in names:
            qualified = f"cudnn.linear_attention.cutile.kernels.{module}.{name}"
            hint = (loaded.kernels.get(qualified) or {}).get("*")
            if not hint:
                missing.append(qualified)
                continue
            assert set(hint) == {"num_worker_warps", "occupancy"}, (qualified, hint)
            assert all(isinstance(v, int) and v > 0 for v in hint.values()), (qualified, hint)
    assert not missing, f"sm103 table has no wildcard cuTile entry for: {missing}"
