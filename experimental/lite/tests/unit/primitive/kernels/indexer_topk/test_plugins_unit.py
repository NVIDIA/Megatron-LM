# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer top-k plugin loaders, their environment rules and the key cache.

The plugins here are pure-Python ABI v1 adapter trees (with fake CUDA files) built in tmp_path.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
import torch

from megatron.lite.primitive.kernels.indexer_topk import (
    ExactTopKConfig,
    IndexerTopKConfigError,
    IndexerTopKPluginError,
    IndexerTopKRuntimeError,
    LiteTopKPluginConfig,
    LiteTopKPluginSettings,
)
from megatron.lite.primitive.kernels.indexer_topk.plugins import abi, cache, env, loader

pytestmark = pytest.mark.mlite

_PREFIX = "SGLANG_LITETOPK"
_SCORE = "SGLANG_LITETOPK_H32_SCORE_POLICY"
_DUAL_CTA = "SGLANG_LITETOPK_FP8_CP_DUAL_CTA"
_GRAFT = "LITETOPK_GRAFT_TIGHTEN"
_KERNEL_FILES = ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh", "dense_topk_litetopk.cuh")

# Every key the fake adapter snapshots (a subset of what the real adapters read).
_READS = (
    "SGLANG_LITETOPK",
    "SGLANG_LITETOPK_CP_GLOBAL_CARRY",
    "SGLANG_LITETOPK_COLDSTART_IDENTITY",
    "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q",
    "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES",
    "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q",
    "SGLANG_LITETOPK_H32_SCORE_POLICY",
    "SGLANG_LITETOPK_H32_TIE_POLICY",
    "SGLANG_LITETOPK_MERGE_CAP",
    "SGLANG_LITETOPK_NB",
    "SGLANG_LITETOPK_PAGED_CANDIDATES",
    "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW",
    "SGLANG_LITETOPK_PATH_TIMING",
    "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK",
    "SGLANG_LITETOPK_TIERED_SEED_12K",
)
# A source that has no H32 policy keys (like the MXFP4-only source).
_READS_WITHOUT_POLICIES = tuple(key for key in _READS if "_H32_" not in key)

_FP8_ROUTE = {
    "name": "fp8_paged",
    "fmt": "fp8",
    "heads": [32],
    "head_dims": [128],
    "topk": [2048],
    "max_topk": 2048,
    "qualified_query_lengths": [1016, 1024, 2040, 2048],
    "admitted_max_query_len": 1776,
    "min_keys": 196608,
    "max_keys": 1 << 20,
    "hot_prefix": 12288,
    "exact": False,
    "tie_policies": ["logical-id-desc", "storage"],
    "score_policies": ["folded", "native-fp32"],
}
_FP4_ROUTE = {
    **_FP8_ROUTE,
    "name": "fp4_slab",
    "fmt": "mxfp4",
    "heads": [32, 64],
    "topk": None,
    "qualified_query_lengths": [4032, 4096],
    "admitted_max_query_len": 0,
    "min_keys": 65536,
    "tie_policies": ["storage"],
    "score_policies": ["folded"],
}

_COMMON = LiteTopKPluginSettings(
    paged_pool_pages_per_row=32,
    fp8_row_tiles=2,
    fp8_paged_admit_max_query_len=1776,
    coldstart_identity=True,
)
_POLICIES = dataclasses.replace(_COMMON, tie_policy="logical-id-desc", score_policy="native-fp32")

_FAKE_ADAPTER = '''
"""A pure-Python stand-in for an ABI v1 LiteTopK adapter (test fixture)."""
import hashlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "fake_plugin.json"), encoding="utf-8") as _handle:
    _CONFIG = json.load(_handle)


def _event(*record):
    with open(os.path.join(_HERE, "events.log"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\\n")


_event("import")
LITETOPK_ABI_VERSION = _CONFIG["abi"]
_IMPORT_ENV = {k: v for k, v in os.environ.items() if k.startswith("SGLANG_LITETOPK")}
_EFFECTIVE = {key: _IMPORT_ENV.get(key) for key in _CONFIG["reads"]}
_EFFECTIVE.update(_CONFIG["effective_override"])


def _source_id():
    digest = hashlib.sha256()
    for name in ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh", "dense_topk_litetopk.cuh"):
        digest.update(name.encode())
        with open(os.path.join(_HERE, "litetopk_kernels", name), "rb") as handle:
            digest.update(handle.read())
    return digest.hexdigest()[:12]


def plugin_info():
    info = {
        "abi": LITETOPK_ABI_VERSION,
        "source_id": _source_id(),
        "routes": _CONFIG["routes"],
        "effective_config": dict(_EFFECTIVE),
        "launch_time_env_keys": _CONFIG["launch_keys"],
        "tie_policy": _EFFECTIVE.get("SGLANG_LITETOPK_H32_TIE_POLICY") or "storage",
        "score_policy": _EFFECTIVE.get("SGLANG_LITETOPK_H32_SCORE_POLICY") or "folded",
    }
    info.update(_CONFIG["info_override"])
    return info


def load_extension(
    *, prebuilt_path=None, prebuilt_sha256=None, build_dir=None, deepgemm_include_dir=None,
    cuda_arch="10.0a",
):
    _event("load_extension", prebuilt_path, prebuilt_sha256, build_dir, deepgemm_include_dir)
    if _CONFIG["fail_load"]:
        raise RuntimeError("fake extension failed to load")


def production_min_s(use_fp4):
    return 65536 if use_fp4 else 196608


def carry_vote_rows():
    return 1536


def begin_call(device, hot_key, sequence_length):
    _event("begin_call", str(device), sequence_length)


def prepare_permuted_gather(
    kv_cache, dst_k, dst_scale, block_table, *, sequence_length, query_length, num_reqs,
    common_end, window_start, hot_key,
):
    return None


def try_large_exact_once_chunk(
    q, k, k_scale, weights, ks, ke, out_idx, topk, *, permuted_plan, num_reqs, ke_min_hint,
    cap=None, hot_key=None, ks_common_hint=0, carry_extent_hint=None,
    carry_recent_rows_hint=None, q_sf=None, carry_io=True, exact=False, status_out=None,
):
    return False


def stash_carry(hot_key, idx, S, min_index=0, *, recent_rows_hint=None):
    return None


def drop_carry(device, hot_key):
    return None


def release(device, *, release_scratch=True):
    _event("release", str(device), release_scratch)


for _name in _CONFIG["omit"]:
    del globals()[_name]
'''


def _make_plugin(
    root: Path,
    *,
    cuda_tag: str = "base",
    reads: tuple[str, ...] = _READS,
    launch_keys: tuple[str, ...] = (_SCORE, _DUAL_CTA),
    routes: tuple[dict, ...] = (_FP8_ROUTE,),
    abi_version: object = 1,
    effective_override: dict | None = None,
    info_override: dict | None = None,
    fail_load: bool = False,
    omit: tuple[str, ...] = (),
    extra_source: str = "",
) -> Path:
    kernels = root / "litetopk_kernels"
    kernels.mkdir(parents=True)
    for name in _KERNEL_FILES:
        (kernels / name).write_text(f"// {name} of fake source {cuda_tag}\n", encoding="utf-8")
    (root / "litetopk.py").write_text(_FAKE_ADAPTER + extra_source, encoding="utf-8")
    _write_fake_config(
        root,
        abi=abi_version,
        reads=list(reads),
        launch_keys=list(launch_keys),
        routes=list(routes),
        effective_override=effective_override or {},
        info_override=info_override or {},
        fail_load=fail_load,
        omit=list(omit),
    )
    return root


def _write_fake_config(root: Path, **config) -> None:
    (root / "fake_plugin.json").write_text(json.dumps(config), encoding="utf-8")


def _events(root: Path) -> list[list]:
    path = root / "events.log"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plugin_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_PREFIX) or key in env.LAUNCH_TIME_ENV_KEYS
    }


def _fake_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name.startswith(("megatron_lite_litetopk_", "megatron_lite_exact_topk_"))
    )


def _clear_plugin_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forget the loaded plugins and remove the keys and modules the loaders added."""
    for key in _plugin_env():
        os.environ.pop(key)
    for name in _fake_modules():
        sys.modules.pop(name)
    monkeypatch.setattr(env, "_RENDERED_BY", {})
    monkeypatch.setattr(env, "_LAUNCH_ENV", {})
    monkeypatch.setattr(loader, "_LOADED", {})
    monkeypatch.setattr(loader, "_EXACT_LOADED", {})


@pytest.fixture(autouse=True)
def _isolated_plugin_state(monkeypatch):
    """Give every test an empty plugin ledger and environment, and restore the process after."""
    for key in _plugin_env():
        monkeypatch.delenv(key)
    for name in _fake_modules():
        monkeypatch.delitem(sys.modules, name)
    _clear_plugin_state(monkeypatch)
    # The fakes need none of the runtime packages of real plugins.
    monkeypatch.setattr(loader, "_missing_packages", lambda: [])
    yield
    for key in _plugin_env():
        os.environ.pop(key)
    for name in _fake_modules():
        sys.modules.pop(name)


def _load(root: Path, settings: LiteTopKPluginSettings | None = None, **config):
    return loader.load_litetopk_plugin(LiteTopKPluginConfig(source=str(root), **config), settings)


def test_source_id_matches_external_algorithm(tmp_path):
    root = _make_plugin(tmp_path / "plugin")
    digest = hashlib.sha256()
    for name in _KERNEL_FILES:
        digest.update(name.encode())
        digest.update((root / "litetopk_kernels" / name).read_bytes())
    expected = digest.hexdigest()[:12]
    assert loader.compute_litetopk_source_id(root) == expected
    assert loader.compute_litetopk_source_id(str(root)) == expected

    plugin = _load(root)
    assert plugin.source_id == plugin.info.source_id == expected

    # The adapter is not covered by the id; every CUDA byte is.
    (root / "litetopk.py").write_text(_FAKE_ADAPTER + "\n# edited\n", encoding="utf-8")
    assert loader.compute_litetopk_source_id(root) == expected
    (root / "litetopk_kernels" / "dense_topk_litetopk.cuh").write_bytes(b"// changed\n")
    assert loader.compute_litetopk_source_id(root) != expected

    # Known answer: sha256 over (file name, file bytes) in the fixed file order, 12 hex digits.
    fixed = tmp_path / "fixed" / "litetopk_kernels"
    fixed.mkdir(parents=True)
    for name, body in zip(_KERNEL_FILES, (b"// dsa\n", b"// sm100\n", b"// dense\n")):
        (fixed / name).write_bytes(body)
    assert loader.compute_litetopk_source_id(fixed.parent) == "82648da0b428"

    (fixed / "sm100_dsa_litetopk.cuh").unlink()
    with pytest.raises(IndexerTopKPluginError, match="litetopk_kernels/sm100_dsa_litetopk.cuh"):
        loader.compute_litetopk_source_id(fixed.parent)


def test_pins_reject_mismatch(tmp_path, monkeypatch):
    root = _make_plugin(tmp_path / "plugin")
    prebuilt = tmp_path / "extension.so"
    prebuilt.write_bytes(b"not really an extension")
    pins = {
        "expected_source_id": loader.compute_litetopk_source_id(root),
        "expected_adapter_sha256": _sha256(root / "litetopk.py"),
        "prebuilt_extension": str(prebuilt),
        "prebuilt_extension_sha256": _sha256(prebuilt),
    }
    for name, what in (
        ("expected_source_id", "source id"),
        ("expected_adapter_sha256", "adapter sha256"),
        ("prebuilt_extension_sha256", "prebuilt extension sha256"),
    ):
        wrong = "0" * len(pins[name])
        with pytest.raises(
            IndexerTopKPluginError,
            match=re.escape(f"LiteTopK plugin at {root}: {what} is {pins[name]}, expected {wrong}"),
        ):
            _load(root, _COMMON, **{**pins, name: wrong})
    # A wrong pin stops the load before any plugin code runs or any key is written.
    assert _events(root) == []
    assert _fake_modules() == []
    assert _plugin_env() == {}

    # Every missing item is reported at once, again before anything runs.
    empty = tmp_path / "empty"
    (empty / "litetopk_kernels").mkdir(parents=True)
    monkeypatch.setattr(loader, "_missing_packages", lambda: ["deep_gemm"])
    with pytest.raises(IndexerTopKPluginError) as excinfo:
        _load(empty, prebuilt_extension=str(tmp_path / "absent.so"))
    message = str(excinfo.value)
    for part in (
        "litetopk.py is missing",
        "litetopk_kernels/dsa_litetopk.cu is missing",
        "litetopk_kernels/sm100_dsa_litetopk.cuh is missing",
        "litetopk_kernels/dense_topk_litetopk.cuh is missing",
        "Python package deep_gemm is not installed",
        f"prebuilt extension {tmp_path / 'absent.so'} does not exist",
    ):
        assert part in message
    monkeypatch.setattr(loader, "_missing_packages", lambda: [])

    # Matching pins load, and the verified extension hash reaches load_extension.
    plugin = _load(root, _COMMON, **pins)
    assert plugin.pinned == {
        "expected_source_id",
        "expected_adapter_sha256",
        "prebuilt_extension_sha256",
    }
    assert plugin.prebuilt_extension == str(prebuilt.resolve())
    assert _events(root) == [
        ["import"],
        ["load_extension", str(prebuilt.resolve()), pins["prebuilt_extension_sha256"], None, None],
    ]


def test_abi_version_required(tmp_path):
    cases = (
        ({"abi_version": 2}, "exposes ABI 2; Megatron Lite needs LITETOPK_ABI_VERSION == 1"),
        ({"abi_version": "1"}, "exposes ABI '1'"),
        ({"extra_source": "\ndel LITETOPK_ABI_VERSION\n"}, "exposes ABI None"),
        ({"omit": ("release", "drop_carry")}, "lacks the ABI v1 functions drop_carry, release"),
        (
            {"extra_source": "\ndef release(device):\n    return None\n"},
            r"release\(device\) cannot take the ABI v1 call release\(arg, release_scratch=\)",
        ),
        (
            {
                "extra_source": (
                    "\ndef try_large_exact_once_chunk(q, k, k_scale, weights, ks, ke, out_idx, "
                    "topk, *, permuted_plan, num_reqs, ke_min_hint, _carry_io=True):\n"
                    "    return False\n"
                )
            },
            "try_large_exact_once_chunk.* cannot take the ABI v1 call",
        ),
        ({"info_override": {"abi": 2}}, r"plugin_info\(\)\['abi'\] must be 1, got 2"),
        ({"info_override": {"extra": 1}}, r"has missing keys \[\] and unknown keys \['extra'\]"),
        (
            {"info_override": {"source_id": "0" * 12}},
            "reports source id 000000000000, but its CUDA sources hash to",
        ),
        ({"routes": ({**_FP8_ROUTE, "fmt": "mxfp4"},)}, r"\['fmt'\] must be 'fp8' for fp8_paged"),
        ({"routes": ()}, r"\['routes'\] must be a non-empty list"),
        ({"routes": (_FP8_ROUTE, _FP8_ROUTE)}, "declares a route twice"),
    )
    for index, (options, match) in enumerate(cases):
        root = _make_plugin(tmp_path / f"case{index}", **options)
        with pytest.raises(IndexerTopKPluginError, match=match):
            _load(root, _COMMON)
        # The module ran but was discarded, and its extension was never loaded.
        assert _events(root) == [["import"]]
        assert _fake_modules() == []
        assert _plugin_env() == {}


def test_env_conflict_raises(tmp_path, monkeypatch):
    root = _make_plugin(tmp_path / "plugin")
    monkeypatch.setenv("SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW", "13")
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape(
            "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW='13' is already set in the process but the "
            "plugin settings need '32'; unset it or change LiteTopKPluginSettings"
        ),
    ):
        _load(root, _COMMON)
    assert _events(root) == []
    assert _plugin_env() == {"SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW": "13"}

    # A pre-set key with the rendered value is not a conflict.
    monkeypatch.setenv("SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW", "32")
    plugin = _load(root, _COMMON)
    assert plugin.info.effective_config["SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW"] == "32"

    # A second source that needs another value of a key the first one rendered cannot load.
    other = _make_plugin(tmp_path / "other", cuda_tag="other")
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape(
            "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q='1776' is already set in the process but the "
            "plugin settings need '0'; unset it or change LiteTopKPluginSettings (it was rendered "
            f"for LiteTopK source {plugin.owner})"
        ),
    ):
        _load(other, dataclasses.replace(_COMMON, fp8_paged_admit_max_query_len=0))
    assert _events(other) == []
    assert os.environ["SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q"] == "1776"


def test_rendered_env_persists(tmp_path):
    root = _make_plugin(tmp_path / "plugin")
    plugin = _load(root, _POLICIES)
    rendered = env.render_plugin_env(_POLICIES)
    assert plugin.rendered_env == rendered
    # Written before the import, never restored afterwards.
    assert _plugin_env() == rendered
    assert {
        key: value for key, value in plugin.info.effective_config.items() if value is not None
    } == rendered

    # A second, compatible source leaves the keys in place, and so does a cached reload.
    other = _make_plugin(
        tmp_path / "other",
        cuda_tag="other",
        reads=_READS_WITHOUT_POLICIES,
        launch_keys=(_DUAL_CTA,),
        routes=(_FP4_ROUTE,),
    )
    _load(other, _COMMON)
    assert _load(root, _POLICIES) is plugin
    assert _plugin_env() == rendered

    provenance = plugin.provenance()
    assert json.loads(json.dumps(provenance)) == provenance
    assert provenance["rendered_env"] == rendered
    assert provenance["source_id"] == plugin.source_id
    assert provenance["plugin_info"]["routes"][0]["name"] == "fp8_paged"


def test_render_plugin_env_table():
    fixed = {
        "SGLANG_LITETOPK": "1",
        "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q": "1",
        "SGLANG_LITETOPK_PAGED_CANDIDATES": "1",
        "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK": "1",
    }
    assert env.render_plugin_env(None) == fixed
    assert env.render_plugin_env(LiteTopKPluginSettings()) == fixed
    every = LiteTopKPluginSettings(
        tie_policy="logical-id",
        score_policy="native-fp32",
        paged_pool_pages_per_row=32,
        fp8_row_tiles=2,
        fp8_paged_admit_max_query_len=0,
        tiered_seed_12k=False,
        coldstart_identity=True,
        merge_cap=262144,
        raw32_staging="u40x14",
    )
    assert all(getattr(every, field.name) is not None for field in dataclasses.fields(every))
    assert env.render_plugin_env(every) == {
        **fixed,
        "SGLANG_LITETOPK_COLDSTART_IDENTITY": "1",
        "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES": "2",
        "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q": "0",
        "SGLANG_LITETOPK_H32_SCORE_POLICY": "native-fp32",
        "SGLANG_LITETOPK_H32_TIE_POLICY": "logical-id",
        "SGLANG_LITETOPK_MERGE_CAP": "262144",
        "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW": "32",
        "SGLANG_LITETOPK_RAW32_STAGING": "u40x14",
        "SGLANG_LITETOPK_TIERED_SEED_12K": "0",
    }
    # The rendered launch-time key is on the known launch-time list; the diagnostic keys are not
    # rendered.
    assert _SCORE in env.LAUNCH_TIME_ENV_KEYS
    assert not env.DIAGNOSTIC_ENV_KEYS & set(env.render_plugin_env(every))


def test_render_reproduces_previous_integration_env():
    # The environment the previous LiteTopK integration imported its adapters with: its
    # production defaults (setdefault), the GLM-5.2 driver overrides, and the query-length
    # admission that replaced its module-attribute patch (1776-row tiles for FP8, none for FP4).
    fp8_historical = LiteTopKPluginSettings(
        tie_policy="logical-id-desc",
        score_policy="native-fp32",
        paged_pool_pages_per_row=13,
        fp8_row_tiles=2,
        fp8_paged_admit_max_query_len=1776,
        tiered_seed_12k=True,
        coldstart_identity=True,
    )
    assert env.render_plugin_env(fp8_historical) == {
        "SGLANG_LITETOPK": "1",
        "SGLANG_LITETOPK_COLDSTART_IDENTITY": "1",
        "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q": "1",
        "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES": "2",
        "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q": "1776",
        "SGLANG_LITETOPK_H32_SCORE_POLICY": "native-fp32",
        "SGLANG_LITETOPK_H32_TIE_POLICY": "logical-id-desc",
        "SGLANG_LITETOPK_PAGED_CANDIDATES": "1",
        "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW": "13",
        "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK": "1",
        "SGLANG_LITETOPK_TIERED_SEED_12K": "1",
    }
    mxfp4_historical = LiteTopKPluginSettings(
        paged_pool_pages_per_row=32,
        fp8_row_tiles=2,
        fp8_paged_admit_max_query_len=0,
        coldstart_identity=True,
    )
    assert env.render_plugin_env(mxfp4_historical) == {
        "SGLANG_LITETOPK": "1",
        "SGLANG_LITETOPK_COLDSTART_IDENTITY": "1",
        "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q": "1",
        "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES": "2",
        "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q": "0",
        "SGLANG_LITETOPK_PAGED_CANDIDATES": "1",
        "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW": "32",
        "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK": "1",
    }


def test_unrendered_litetopk_key_raises(tmp_path, monkeypatch):
    root = _make_plugin(tmp_path / "plugin")
    monkeypatch.setenv("SGLANG_LITETOPK_NB", "128")
    monkeypatch.setenv(_GRAFT, "1")
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape(
            f"the process sets {_GRAFT}, SGLANG_LITETOPK_NB, which the plugin settings do not "
            "render"
        ),
    ):
        _load(root, _COMMON)
    assert _events(root) == []
    assert _plugin_env() == {"SGLANG_LITETOPK_NB": "128", _GRAFT: "1"}
    monkeypatch.delenv("SGLANG_LITETOPK_NB")
    monkeypatch.delenv(_GRAFT)

    # Diagnostic keys pass through: the plugin snapshots those it reads, and the provenance
    # records all of them.
    monkeypatch.setenv("SGLANG_LITETOPK_PATH_TIMING", "1")
    monkeypatch.setenv("SGLANG_LITETOPK_OVF_LOG", "1")
    plugin = _load(root, _COMMON)
    diagnostics = {"SGLANG_LITETOPK_OVF_LOG": "1", "SGLANG_LITETOPK_PATH_TIMING": "1"}
    assert plugin.diagnostic_env == diagnostics
    assert plugin.info.effective_config["SGLANG_LITETOPK_PATH_TIMING"] == "1"
    assert "SGLANG_LITETOPK_OVF_LOG" not in plugin.info.effective_config
    assert plugin.provenance()["diagnostic_env"] == diagnostics


def test_launch_time_keys_rechecked(tmp_path, monkeypatch):
    root = _make_plugin(tmp_path / "plugin", launch_keys=(_SCORE, _DUAL_CTA, _GRAFT))
    plugin = _load(root, _POLICIES)
    assert plugin.launch_time_env == {_GRAFT: None, _DUAL_CTA: None, _SCORE: "native-fp32"}
    plugin.check_launch_time_env()

    # The fixture removes every key set here.
    os.environ[_SCORE] = "folded"
    with pytest.raises(
        IndexerTopKRuntimeError,
        match=re.escape(f"{_SCORE} changed from 'native-fp32' to 'folded' after LiteTopK source"),
    ):
        plugin.check_launch_time_env()
    del os.environ[_SCORE]
    with pytest.raises(IndexerTopKRuntimeError, match="from 'native-fp32' to None"):
        plugin.check_launch_time_env()
    os.environ[_SCORE] = "native-fp32"
    plugin.check_launch_time_env()

    os.environ[_DUAL_CTA] = "1"
    with pytest.raises(IndexerTopKRuntimeError, match=f"{_DUAL_CTA} changed from None to '1'"):
        plugin.check_launch_time_env()
    del os.environ[_DUAL_CTA]
    plugin.check_launch_time_env()

    # A launch-time key the plugin does not read is not its concern.
    os.environ["SGLANG_LITETOPK_FP8_CP_LOREG_CONTROL"] = "64"
    plugin.check_launch_time_env()


def test_effective_config_mismatch_raises(tmp_path):
    cases = (
        (
            {"effective_override": {"SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW": "16"}},
            _COMMON,
            "LiteTopK plugin effective_config[SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW]='16' "
            "differs from the requested '32'",
        ),
        (
            # A rendered key the module does not read.
            {"reads": _READS_WITHOUT_POLICIES},
            _POLICIES,
            "LiteTopK plugin effective_config[SGLANG_LITETOPK_H32_SCORE_POLICY]=None differs "
            "from the requested 'native-fp32'",
        ),
        (
            # An unrendered key that the module snapshotted with a value.
            {"effective_override": {"SGLANG_LITETOPK_NB": "64"}},
            _COMMON,
            "LiteTopK plugin effective_config[SGLANG_LITETOPK_NB]='64' differs from the "
            "requested None",
        ),
        (
            {"info_override": {"tie_policy": "storage"}},
            _POLICIES,
            "reports tie_policy 'storage'; the settings request 'logical-id-desc'",
        ),
    )
    for index, (options, settings, match) in enumerate(cases):
        root = _make_plugin(tmp_path / f"case{index}", **options)
        with pytest.raises(IndexerTopKPluginError, match=re.escape(match)):
            _load(root, settings)
        # Imported, rejected before its extension loaded; the written keys are removed again.
        assert _events(root) == [["import"]]
        assert _fake_modules() == []
        assert _plugin_env() == {}

    # A key rendered for a loaded plugin stays set; another source that reads it at import but
    # leaves it unrendered would snapshot the first plugin's value, so it is refused.
    seeded = dataclasses.replace(_COMMON, tiered_seed_12k=True)
    first = _load(_make_plugin(tmp_path / "first", cuda_tag="first"), seeded)
    second = _make_plugin(tmp_path / "second", cuda_tag="second")
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape(
            "LiteTopK plugin effective_config[SGLANG_LITETOPK_TIERED_SEED_12K]='1' differs from "
            "the requested None (SGLANG_LITETOPK_TIERED_SEED_12K was rendered for LiteTopK "
            f"source {first.owner} and this plugin reads it at import"
        ),
    ):
        _load(second, _COMMON)
    assert _events(second) == [["import"]]
    assert _plugin_env() == env.render_plugin_env(seeded)


def test_distinct_sources_coexist(tmp_path, monkeypatch):
    fp8 = _make_plugin(tmp_path / "fp8", cuda_tag="fp8", launch_keys=(_SCORE, _DUAL_CTA))
    fp4 = _make_plugin(
        tmp_path / "fp4",
        cuda_tag="fp4",
        reads=_READS_WITHOUT_POLICIES,
        launch_keys=(_DUAL_CTA, _GRAFT),
        routes=(_FP4_ROUTE,),
    )
    # Both orders work: the score-policy key is launch-time for one source only.
    for order in ((fp8, fp4), (fp4, fp8)):
        _clear_plugin_state(monkeypatch)
        loaded = {root: _load(root, _POLICIES if root == fp8 else _COMMON) for root in order}
        first, second = loaded[fp8], loaded[fp4]
        assert first.source_id != second.source_id
        assert first.module is not second.module
        assert {first.module.__name__, second.module.__name__} <= set(sys.modules)
        assert loader.loaded_litetopk_plugins() == tuple(loaded[root] for root in order)
        assert first.info.route("fp8_paged") is not None and first.info.route("fp4_slab") is None
        assert second.info.route("fp4_slab").heads == frozenset((32, 64))
        assert os.environ[_SCORE] == "native-fp32"
        assert _SCORE not in second.launch_time_env
        first.check_launch_time_env()
        second.check_launch_time_env()


def test_sources_conflicting_launch_keys_raise(tmp_path, monkeypatch):
    renders = _make_plugin(tmp_path / "renders", cuda_tag="renders", launch_keys=(_SCORE,))
    needs_unset = _make_plugin(
        tmp_path / "unset", cuda_tag="unset", reads=_READS_WITHOUT_POLICIES, launch_keys=(_SCORE,)
    )
    first = _load(renders, _POLICIES)
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape(
            f"reads {_SCORE} on every CUDA launch and needs it unset, but LiteTopK source "
            f"{first.owner} set {_SCORE}='native-fp32' in this process"
        ),
    ):
        _load(needs_unset, _COMMON)
    assert _events(needs_unset) == [["import"]]
    first.check_launch_time_env()

    # The other order fails before the second import: the loaded plugin requires the key unset.
    _clear_plugin_state(monkeypatch)
    (renders / "events.log").unlink()
    second = _load(needs_unset, _COMMON)
    assert second.launch_time_env == {_SCORE: None}
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape(
            f"LiteTopK source {second.owner} reads {_SCORE} on every CUDA launch and was loaded "
            f"with {_SCORE}=None, but the plugin settings need 'native-fp32'"
        ),
    ):
        _load(renders, _POLICIES)
    assert _events(renders) == []
    assert _SCORE not in os.environ
    second.check_launch_time_env()


def test_same_source_conflicting_settings_raise(tmp_path):
    root = _make_plugin(tmp_path / "plugin")
    first = _load(root, _POLICIES)
    message = re.escape(
        f"LiteTopK source {first.source_id} is already loaded in this process with different "
        "settings; ABI v1 snapshots settings at import. Use one settings profile per process."
    )
    with pytest.raises(IndexerTopKPluginError, match=message):
        _load(root, dataclasses.replace(_POLICIES, paged_pool_pages_per_row=13))

    # Another adapter of the same CUDA source shares its extension, so the rule still applies.
    twin = tmp_path / "twin"
    shutil.copytree(root, twin)
    (twin / "events.log").unlink()
    (twin / "litetopk.py").write_text(_FAKE_ADAPTER + "\n# twin\n", encoding="utf-8")
    with pytest.raises(IndexerTopKPluginError, match=message):
        _load(twin, _COMMON)
    assert _events(twin) == []
    assert _load(root, _POLICIES) is first
    assert _events(root).count(["import"]) == 1


def test_loader_idempotent(tmp_path, caplog):
    root = _make_plugin(tmp_path / "plugin")
    with caplog.at_level(logging.WARNING, logger=loader.__name__):
        first = _load(root, _POLICIES)
        second = loader.load_litetopk_plugin(
            LiteTopKPluginConfig(source=f"{tmp_path}/./plugin/"), _POLICIES
        )
    assert second is first
    assert copy.deepcopy(first) is first  # process state, shared by copies of its users
    assert [event[0] for event in _events(root)] == ["import", "load_extension"]
    warnings = [record.getMessage() for record in caplog.records if record.name == loader.__name__]
    assert len(warnings) == 1
    assert "without expected_source_id, expected_adapter_sha256" in warnings[0]
    assert first.source_id in warnings[0]

    with pytest.raises(TypeError, match="expected LiteTopKPluginSettings or None, got dict"):
        loader.load_litetopk_plugin(LiteTopKPluginConfig(source=str(root)), {"tie_policy": None})

    # The same plugin with another extension request is refused instead of silently reused.
    with pytest.raises(IndexerTopKPluginError, match=r"already loaded in this process with \("):
        _load(root, _POLICIES, build_dir=str(tmp_path / "build"))

    # A failed load is not cached and leaves nothing behind; a later load retries.
    flaky = _make_plugin(tmp_path / "flaky", cuda_tag="flaky", fail_load=True)
    with pytest.raises(
        IndexerTopKPluginError,
        match=re.escape("load_extension() failed: RuntimeError: fake extension failed to load"),
    ):
        _load(flaky, _POLICIES)
    assert _fake_modules() == [first.module.__name__]
    config = json.loads((flaky / "fake_plugin.json").read_text(encoding="utf-8"))
    _write_fake_config(flaky, **{**config, "fail_load": False})
    retried = _load(flaky, _POLICIES)
    assert [event[0] for event in _events(flaky)] == [
        "import",
        "load_extension",
        "import",
        "load_extension",
    ]
    assert loader.loaded_litetopk_plugins() == (first, retried)


def test_plugin_config_validation():
    sha = "a" * 64
    for kwargs, match in (
        ({"source": ""}, "indexer_topk.litetopk.source must be a non-empty path string"),
        ({"source": Path("/x")}, "indexer_topk.litetopk.source must be a non-empty path string"),
        ({"source": "/x", "expected_source_id": "996E735C52DF"}, "12 characters"),
        ({"source": "/x", "expected_adapter_sha256": "abc"}, "a SHA-256"),
        ({"source": "/x", "prebuilt_extension_sha256": sha}, "set prebuilt_extension as well"),
        (
            {"source": "/x", "prebuilt_extension": "/x.so", "build_dir": "/b"},
            "cannot be combined with prebuilt_extension",
        ),
    ):
        with pytest.raises(IndexerTopKConfigError, match=match):
            LiteTopKPluginConfig(**kwargs)
    LiteTopKPluginConfig(
        source="/x",
        expected_source_id="996e735c52df",
        expected_adapter_sha256=sha,
        prebuilt_extension="/x.so",
        prebuilt_extension_sha256=sha,
    )
    LiteTopKPluginConfig(source="/x", build_dir="/b", deepgemm_include_dir="/d")

    with pytest.raises(IndexerTopKConfigError, match="unknown file 'other.py'"):
        ExactTopKConfig(source="/x", expected_sha256={"other.py": sha})
    with pytest.raises(IndexerTopKConfigError, match="block_scan.py"):
        ExactTopKConfig(source="/x", expected_sha256={"block_scan.py": "0"})
    with pytest.raises(IndexerTopKConfigError, match="must map file names"):
        ExactTopKConfig(source="/x", expected_sha256=[("block_scan.py", sha)])

    for kwargs, match in (
        ({"tie_policy": "logical_id"}, "tie_policy must be one of"),
        ({"score_policy": "fp32"}, "score_policy must be one of"),
        ({"paged_pool_pages_per_row": 0}, "paged_pool_pages_per_row must be an integer >= 1"),
        ({"fp8_row_tiles": 3}, "fp8_row_tiles must be one of 1, 2, 4, 8"),
        ({"fp8_row_tiles": True}, "fp8_row_tiles must be an integer"),
        ({"fp8_paged_admit_max_query_len": -4}, "must be an integer >= 0"),
        ({"tiered_seed_12k": 1}, "tiered_seed_12k must be a bool"),
        ({"coldstart_identity": "1"}, "coldstart_identity must be a bool"),
        ({"merge_cap": 0}, "merge_cap must be an integer >= 1"),
    ):
        with pytest.raises(IndexerTopKConfigError, match=match):
            LiteTopKPluginSettings(**kwargs)
    # Settings are hashable: the loader caches plugins by them.
    assert hash(_POLICIES) == hash(dataclasses.replace(_POLICIES))


def test_route_capability_helpers():
    info = abi.parse_plugin_info(
        {
            "abi": 1,
            "source_id": "996e735c52df",
            "routes": [_FP8_ROUTE, _FP4_ROUTE],
            "effective_config": {"SGLANG_LITETOPK": "1", "SGLANG_LITETOPK_NB": None},
            "launch_time_env_keys": [_SCORE],
            "tie_policy": "storage",
            "score_policy": "folded",
        },
        origin="test",
    )
    fp8, fp4 = info.route("fp8_paged"), info.route("fp4_slab")
    assert fp8.supports_topk(2048) and not fp8.supports_topk(512)
    assert fp4.supports_topk(512) and fp4.supports_topk(2048) and not fp4.supports_topk(4096)
    # Qualified lengths, and multiples of four up to the admitted maximum.
    assert fp8.admits_query_length(2040) and fp8.admits_query_length(1776)
    assert fp8.admits_query_length(912) and not fp8.admits_query_length(1774)
    assert not fp8.admits_query_length(1780) and not fp8.admits_query_length(0)
    assert fp4.admits_query_length(4032) and not fp4.admits_query_length(2048)
    assert info.as_dict()["routes"] == [_FP8_ROUTE, _FP4_ROUTE]
    assert info.launch_time_env_keys == {_SCORE}

    bad_topk = {**_FP8_ROUTE, "topk": [4096]}
    with pytest.raises(IndexerTopKPluginError, match=r"\['topk'\] exceeds max_topk 2048"):
        abi.parse_plugin_info({**info.as_dict(), "routes": [bad_topk]}, origin="test")
    bad_heads = {**_FP8_ROUTE, "heads": [32, True]}
    with pytest.raises(IndexerTopKPluginError, match=r"\['heads'\] must be a non-empty list"):
        abi.parse_plugin_info({**info.as_dict(), "routes": [bad_heads]}, origin="test")


def _make_exact_package(root: Path, *, util_body: str = "") -> Path:
    root.mkdir(parents=True)
    (root / "block_scan.py").write_text("NAME = 'block_scan'\n", encoding="utf-8")
    (root / "indexer_top_k_varlen_util.py").write_text(
        f"from .block_scan import NAME as SCAN_NAME\n{util_body}\n", encoding="utf-8"
    )
    (root / "indexer_top_k_decode_varlen.py").write_text(
        "from .block_scan import NAME\n"
        "from .indexer_top_k_varlen_util import SCAN_NAME\n"
        "\n"
        "def cute_dsl_topk_wrapper(input_values, seq_lens, top_k, next_n, return_val=True):\n"
        "    raise AssertionError('the fake exact top-k is never called')\n",
        encoding="utf-8",
    )
    return root


def test_exact_topk_private_package(tmp_path):
    source = _make_exact_package(tmp_path / "exact")
    path_before = list(sys.path)
    loaded = loader.load_exact_topk(ExactTopKConfig(source=str(source)))
    assert sys.path == path_before
    assert re.fullmatch(r"megatron_lite_exact_topk_[0-9a-f]{8}", loaded.package)
    package = sys.modules[loaded.package]
    assert package.__path__ == [str(source.resolve())]
    decode = sys.modules[f"{loaded.package}.indexer_top_k_decode_varlen"]
    assert decode.cute_dsl_topk_wrapper is loaded.wrapper
    # The relative imports resolved inside the private package, not as top-level modules.
    assert decode.NAME == decode.SCAN_NAME == "block_scan"
    assert "block_scan" not in sys.modules and "indexer_top_k_varlen_util" not in sys.modules
    assert loaded.file_sha256 == {
        name: _sha256(source / name)
        for name in (
            "block_scan.py",
            "indexer_top_k_varlen_util.py",
            "indexer_top_k_decode_varlen.py",
        )
    }

    # Idempotent, and pinned loads of the same files return the same package.
    assert copy.deepcopy(loaded) is loaded
    pinned = ExactTopKConfig(source=str(source), expected_sha256=loaded.file_sha256)
    assert loader.load_exact_topk(pinned) is loaded
    wrong = ExactTopKConfig(source=str(source), expected_sha256={"block_scan.py": "0" * 64})
    with pytest.raises(
        IndexerTopKPluginError,
        match=f"sha256 of block_scan.py is {loaded.file_sha256['block_scan.py']}, expected 0+$",
    ):
        loader.load_exact_topk(wrong)

    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "block_scan.py").write_text("", encoding="utf-8")
    with pytest.raises(
        IndexerTopKPluginError,
        match="indexer_top_k_varlen_util.py, indexer_top_k_decode_varlen.py missing",
    ):
        loader.load_exact_topk(ExactTopKConfig(source=str(tmp_path / "partial")))

    # A failing import is reported and leaves no module of the private package behind.
    broken = _make_exact_package(
        tmp_path / "broken", util_body="raise ImportError('cutlass is not installed')"
    )
    modules_before = set(sys.modules)
    with pytest.raises(
        IndexerTopKPluginError, match="failed to import: ImportError: cutlass is not installed"
    ):
        loader.load_exact_topk(ExactTopKConfig(source=str(broken)))
    assert set(sys.modules) == modules_before

    # The wrapper checks its arguments from metadata before calling into the package.
    with pytest.raises(ValueError, match="CUDA float32 scores"):
        loaded(torch.zeros(2, 8), torch.zeros(2, dtype=torch.int32), top_k=4)


def _unpack(views: cache.KeyCacheViews, rows: int, value_bytes: int):
    """Read rows back through the block table (an independent oracle of the layout)."""
    values, scales = [], []
    for row in range(rows):
        block = views.cache[int(views.block_table[0, row // 64])].reshape(-1)
        offset = row % 64
        values.append(block[offset * value_bytes : (offset + 1) * value_bytes])
        scale_start = 64 * value_bytes + offset * 4
        scales.append(block[scale_start : scale_start + 4])
    return torch.stack(values), torch.stack(scales)


@pytest.mark.parametrize(
    "fmt, value_bytes, value_dtype", [("fp8", 128, torch.float8_e4m3fn), ("mxfp4", 64, torch.int8)]
)
def test_key_cache_pack_roundtrip_bytes(fmt, value_bytes, value_dtype):
    generator = torch.Generator().manual_seed(7)
    rows = 3 * 64 + 17
    raw = torch.randint(0, 256, (rows, value_bytes), dtype=torch.uint8, generator=generator)
    words = torch.randint(-(2**31), 2**31 - 1, (rows,), dtype=torch.int32, generator=generator)
    values = raw.view(value_dtype)
    scales = words.view(torch.float32) if fmt == "fp8" else words

    pool = cache.KeyCachePool()
    views = pool.acquire(fmt, rows, torch.device("cpu"))
    assert views.cache.shape == (4, 64, value_bytes + 4) and views.cache.dtype == torch.uint8
    assert views.keys.shape == (rows, value_bytes)
    assert views.keys.dtype == (torch.float8_e4m3fn if fmt == "fp8" else torch.uint8)
    assert views.scales.shape == (rows, 4) and views.scales.dtype == torch.uint8
    assert views.block_table.dtype == torch.int32 and views.block_table.tolist() == [[0, 1, 2, 3]]

    views.cache.fill_(0xAB)
    cache.pack_key_cache(views.cache, values, scales)
    unpacked_values, unpacked_scales = _unpack(views, rows, value_bytes)
    assert torch.equal(unpacked_values, raw)
    assert torch.equal(unpacked_scales, words.view(torch.uint8).view(rows, 4))
    # The partial last block is zero past the last row, in both regions.
    tail = views.cache[3].reshape(-1)
    assert not tail[17 * value_bytes : 64 * value_bytes].any()
    assert not tail[64 * value_bytes + 17 * 4 :].any()

    # Packing in aligned chunks writes the same bytes.
    chunked = cache.KeyCachePool().acquire(fmt, rows, torch.device("cpu"))
    chunked.cache.fill_(0x55)
    for start in range(0, rows, 128):
        cache.pack_key_cache(
            chunked.cache, values[start : start + 128], scales[start : start + 128], start_row=start
        )
    assert torch.equal(chunked.cache, views.cache)

    with pytest.raises(ValueError, match="align to 64 rows"):
        cache.pack_key_cache(views.cache, values[:8], scales[:8], start_row=8)
    with pytest.raises(ValueError, match="do not fit"):
        cache.pack_key_cache(views.cache, values[:128], scales[:128], start_row=192)
    with pytest.raises(ValueError, match="expected contiguous values"):
        cache.pack_key_cache(views.cache, values[:, :-1], scales)
    with pytest.raises(ValueError, match="expected contiguous values"):
        cache.pack_key_cache(views.cache, values, scales.view(torch.uint8))
    with pytest.raises(ValueError, match="contiguous uint8"):
        cache.block_major_cache_views(views.cache.view(torch.int8), value_bytes)


def test_key_cache_pool_reuse_and_release():
    pool = cache.KeyCachePool()
    cpu = torch.device("cpu")
    first = pool.acquire("fp8", 5000, cpu)
    assert first.cache.shape[0] == 79
    # Shorter sequences reuse the storage; the capacity grows in steps of 4096 rows.
    shorter = pool.acquire("fp8", 100, cpu)
    assert shorter.cache.data_ptr() == first.cache.data_ptr()
    assert shorter.keys.data_ptr() == first.keys.data_ptr()
    fp8_bytes = pool.allocated_bytes(cpu)
    assert fp8_bytes == 8192 * (132 + 128 + 4) + 128 * 4
    longer = pool.acquire("fp8", 9000, cpu)
    assert longer.cache.data_ptr() != first.cache.data_ptr()
    assert pool.allocated_bytes() == 12288 * (132 + 128 + 4) + 192 * 4
    # Formats get separate caches.
    fp4 = pool.acquire("mxfp4", 100, cpu)
    assert fp4.cache.shape == (2, 64, 68) and fp4.keys.dtype == torch.uint8
    assert pool.allocated_bytes(cpu) == 12288 * 264 + 192 * 4 + 4096 * (68 + 64 + 4) + 64 * 4
    pool.release(cpu)
    assert pool.allocated_bytes() == 0
    assert pool.acquire("fp8", 100, cpu).cache.data_ptr() != longer.cache.data_ptr()
    pool.release()
    assert pool.allocated_bytes() == 0

    with pytest.raises(ValueError, match="'fp8' or 'mxfp4'"):
        pool.acquire("bf16", 100, cpu)
    with pytest.raises(ValueError, match="head_dim=64"):
        pool.acquire("mxfp4", 100, cpu, head_dim=64)
    with pytest.raises(ValueError, match="at least one row"):
        pool.acquire("fp8", 0, cpu)
