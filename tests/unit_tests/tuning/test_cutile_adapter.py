# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The policy applied to cudnn-frontend's cuTile launch autotune, against a stand-in for its hook module."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from megatron.core.tuning import cutile, interception, table
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
    recording = json.loads((tmp_path / "rec.rank0.json").read_text())
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

    merged = table.merge_records([tmp_path / "rec.rank0.json"])
    table.write("test_arch", merged["test_arch"], tmp_path / "tables" / "test_arch.json")
    interception.install(
        AutotunePolicy(mode="pinned", table_path=(str(tmp_path / "tables"),), on_miss="error")
    )
    assert launch(fake_common, kernel()) is CONFIGS[2]


def test_a_kernel_outside_the_scope_keeps_its_timed_pick(isolated_policy, fake_common):
    interception.install(AutotunePolicy(mode="pinned", modules=("mamba_ssm",)))
    assert launch(fake_common, kernel()) is CONFIGS[-1]
    assert interception.choice_log() == {}


def test_config_invariant_kernels_stay_timed(isolated_policy, fake_common):
    interception.install(
        AutotunePolicy(mode="pinned", config_invariant=(f"{GDN}.chunk_fwd_kernel_o",))
    )
    assert launch(fake_common, kernel()) is CONFIGS[-1]
    assert launch(fake_common, kernel("chunk_bwd_kernel_dqkwg")) is CONFIGS[0]


def test_a_cudnn_frontend_without_the_hooks_is_left_alone(isolated_policy, fake_common, caplog):
    del fake_common.select_hints, fake_common.record_hint
    with caplog.at_level("WARNING"):
        interception.install(AutotunePolicy(mode="pinned"))
    assert not hasattr(fake_common, "select_hints")
    assert "no cuTile autotune hooks" in caplog.text


def test_reinstalling_auto_clears_the_hooks(isolated_policy, fake_common):
    interception.install(AutotunePolicy(mode="pinned"))
    assert fake_common.select_hints is cutile.select
    interception.install(AutotunePolicy(mode="auto"))
    assert fake_common.select_hints is None and fake_common.record_hint is None


CUTILE_KERNELS = (
    "chunk_bwd_kernel_dqkwg",
    "chunk_bwd_kernel_dv_local",
    "chunk_fwd_kernel_o",
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64",
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
    "prepare_wy_repr_bwd_kernel",
    "recompute_w_u_fwd_kernel",
    "chunk_gla_fwd_kernel_o",
    "chunk_kda_fwd_kernel_inter_diag_compute_solve",
    "chunk_kda_fwd_kernel_inter_solve_fused",
    "recompute_w_u_fwd_kda_kernel",
)


def test_packaged_sm103_table_covers_every_cutile_kernel():
    """A pinned run on GB300 must never fall back to the first candidate: the packaged table carries a
    wildcard launch hint for every cuTile GDN and KDA kernel the adapter can see."""
    loaded = table.load("sm103", ())
    missing = []
    for name in CUTILE_KERNELS:
        module = (
            "kda"
            if name.endswith("kda_kernel") or "kda" in name or name == "chunk_gla_fwd_kernel_o"
            else "gdn"
        )
        entries = loaded.kernels.get(f"cudnn.linear_attention.cutile.kernels.{module}.{name}") or {}
        hint = entries.get("*")
        if not hint:
            missing.append(name)
            continue
        assert set(hint) == {"num_worker_warps", "occupancy"}, (name, hint)
        assert all(isinstance(v, int) and v > 0 for v in hint.values()), (name, hint)
    assert not missing, f"sm103 table has no wildcard cuTile entry for: {missing}"
