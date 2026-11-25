# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Indexer top-k bindings with real LiteTopK plugins (optional, one Blackwell GPU).

The plugins and the exact-tie top-k are not part of Megatron Lite, so these tests skip unless
their locations are given as JSON, inline or as the path of a JSON file::

    LITETOPK_TEST_SELECTORS='[{"native_format": "fp8", "precision": "exact", "heads": 32,
        "topk": 2048, "litetopk": {"source": "/path/to/plugin", "prebuilt_extension": "..."},
        "plugin_settings": {"paged_pool_pages_per_row": 13}}]' \\
    LITETOPK_TEST_EXACT_TOPK='{"source": "/path/to/exact-tie", "pythonpath": ["/path/to/cudnn"]}' \\
    experimental/lite/tests/run_tests.sh \\
        experimental/lite/tests/smoke/primitive/indexer_topk/test_selector_gpu.py

A selector entry holds the operand format of the indexer (``fp8``), the precision, the indexer
heads and top-k, the ``LiteTopKPluginConfig`` fields under ``litetopk`` and optional
``plugin_settings`` (``LiteTopKPluginSettings`` fields). Every entry runs in a fresh interpreter,
because plugin settings are process-wide: it selects a 262144-token prompt of random operands
once with the reference backend and twice with the LiteTopK backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

_SELECTORS_VARIABLE = "LITETOPK_TEST_SELECTORS"
_EXACT_VARIABLE = "LITETOPK_TEST_EXACT_TOPK"
_TOKENS = 262144


def _read_json(variable: str):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return json.loads(raw if raw[0] in "[{" else Path(raw).read_text(encoding="utf-8"))


_SPECS = _read_json(_SELECTORS_VARIABLE) or []
_EXACT_SPEC = _read_json(_EXACT_VARIABLE)
_REPORTS: dict[str, dict] = {}


def _spec_id(spec) -> str:
    if spec is None:
        return "unset"
    return f"{Path(spec['litetopk']['source']).name}-{spec['native_format']}-{spec['precision']}"


def _specs(native_format: str | None = None, precision: str | None = None) -> list:
    selected = [
        spec
        for spec in _SPECS
        if native_format in (None, spec["native_format"]) and precision in (None, spec["precision"])
    ]
    return selected or [None]


def _report(spec) -> dict:
    """Run the entry in a child interpreter (once per test session)."""
    if spec is None or _EXACT_SPEC is None:
        pytest.skip(f"{_SELECTORS_VARIABLE} and {_EXACT_VARIABLE} are not both set")
    key = json.dumps(spec, sort_keys=True)
    if key not in _REPORTS:
        result = subprocess.run(
            [sys.executable, __file__, json.dumps({"selector": spec, "exact_topk": _EXACT_SPEC})],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
        _REPORTS[key] = json.loads(result.stdout.strip().splitlines()[-1])
    return _REPORTS[key]


def _check_common(report: dict) -> None:
    lite, reference = report["litetopk"], report["reference"]
    # Both arms return ascending rows with -1 last, and valid ids only.
    assert lite["rows_not_ascending"] == reference["rows_not_ascending"] == 0
    assert lite["ids_out_of_range"] == reference["ids_out_of_range"] == 0
    # The LiteTopK arm selected the rows of its plan with the plugin, the rest with the
    # reference selector, and nothing had to be recomputed.
    stats = lite["stats"]
    assert stats["rows"] == stats["litetopk_rows"] + stats["reference_rows"] == _TOKENS
    assert stats["litetopk_rows"] == report["planned_litetopk_rows"] > 0
    assert stats["tiles"] == report["planned_tiles"] and stats["plans"] == report["planned_groups"]
    assert (stats["recomputed_rows"], stats["recomputed_tiles"]) == (0, 0)
    assert stats["declined_tiles"] == {} and stats["status_rows"] == {}
    assert reference["stats"]["reference_rows"] == _TOKENS and reference["stats"]["tiles"] == 0
    assert report["carries_after_call"] == 0


@pytest.mark.parametrize("spec", _specs("fp8", "fast"), ids=_spec_id)
def test_fp8_paged_tiles(spec):
    report = _report(spec)
    _check_common(report)
    stats = report["litetopk"]["stats"]
    # Reference prefix, one stashed seed, then full tiles in groups of eight and a last tile.
    assert stats["carry_stashes"] == 1 and stats["bootstrap_rows"] == 0
    assert report["tile_rows"] == report["planned_tile_rows"]
    # The fast selector may differ from the exact reference only among nearly tied keys.
    assert report["recall_min"] >= 0.99 and report["recall_mean"] >= 0.999


@pytest.mark.parametrize("spec", _specs(precision="exact"), ids=_spec_id)
def test_exact_mode_bitwise(spec):
    report = _report(spec)
    _check_common(report)
    # Exact precision: every row of the LiteTopK arm equals the reference arm, bit for bit.
    assert report["rows_differ"] == 0 and report["recall_min"] == 1.0
    assert report["litetopk_repeat_rows_differ"] == 0


@pytest.mark.parametrize("spec", _specs(), ids=_spec_id)
def test_sync_debug_mode_no_extra_syncs(spec):
    report = _report(spec)
    # The second LiteTopK call (the first warms up the plugin's one-time host checks) ran under
    # torch.cuda.set_sync_debug_mode("error") except for its one status read.
    assert report["synchronizations_outside_status_read"] == 0, report.get("synchronization_error")
    assert report["status_reads"] == 1


def _child(spec: dict) -> dict:
    exact_spec = dict(spec["exact_topk"])
    for entry in reversed(exact_spec.pop("pythonpath", [])):
        sys.path.insert(0, entry)
    import torch

    from megatron.lite.primitive.kernels.indexer_topk import (
        ExactTopKConfig,
        IndexerGeometry,
        IndexerTopKConfig,
        IndexerTopKTuning,
        LiteTopKPluginConfig,
        LiteTopKPluginSettings,
        QueryLayout,
    )
    from megatron.lite.primitive.kernels.indexer_topk.planner import plan_segment
    from megatron.lite.primitive.modules.attention import indexer_topk as bindings

    selector = spec["selector"]
    fmt = selector["native_format"]
    heads, topk = selector["heads"], selector["topk"]
    geometry = IndexerGeometry(num_heads=heads, head_dim=128, topk=topk)

    class Consumer(torch.nn.Module):
        binding = None

        def indexer_geometry(self):
            return geometry

        def set_indexer_topk(self, binding):
            self.binding = binding

    exact_topk = ExactTopKConfig(**exact_spec)
    tuning = IndexerTopKTuning(
        required=True, plugin_settings=LiteTopKPluginSettings(**selector.get("plugin_settings", {}))
    )
    arms = {}
    for backend in ("reference", "litetopk"):
        consumer = Consumer()
        bindings.configure_indexer_topk(
            [consumer],
            IndexerTopKConfig(
                backend=backend,
                precision=selector["precision"],
                litetopk=LiteTopKPluginConfig(**selector["litetopk"]),
                exact_topk=exact_topk,
            ),
            native_format=fmt,
            tuning=tuning,
        )
        arms[backend] = consumer.binding

    device = torch.device("cuda", 0)
    generator = torch.Generator(device=device).manual_seed(20260930)
    keys = _TOKENS
    q = torch.randn((_TOKENS, heads, 128), generator=generator, device=device).to(torch.bfloat16)
    k = torch.randn((keys, 128), generator=generator, device=device).to(torch.bfloat16)
    weights = torch.rand((_TOKENS, heads), generator=generator, device=device) * heads**-0.5
    layout = QueryLayout.full(_TOKENS, keys=keys)

    def select(binding):
        with torch.no_grad():
            return binding.select(q, k, weights, layout=layout, topk=topk, softmax_scale=128**-0.5)

    def describe(binding, out):
        valid = out >= 0
        ascending = ((out[:, 1:] > out[:, :-1]) | ~valid[:, 1:]).all(1)
        ascending &= (valid[:, :-1] | ~valid[:, 1:]).all(1)
        return {
            "stats": binding.stats.as_dict(),
            "rows_not_ascending": int((~ascending).sum()),
            "ids_out_of_range": int((out >= keys).sum()),
        }

    reference_out = select(arms["reference"])
    report = {"reference": describe(arms["reference"], reference_out)}
    lite = arms["litetopk"]
    first = select(lite)  # also warms up the plugin's one-time host checks
    report["litetopk"] = describe(lite, first)

    # The second call must not synchronize with the device except for its one status read.
    status_reads = []
    original = bindings.IndexerTopKBinding._worst_status

    def read(self, status):
        torch.cuda.set_sync_debug_mode("default")
        try:
            status_reads.append(original(self, status))
        finally:
            torch.cuda.set_sync_debug_mode("error")
        return status_reads[-1]

    bindings.IndexerTopKBinding._worst_status = read
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        second = select(lite)
        synchronizations = 0
    except RuntimeError as exc:
        if "synchron" not in str(exc):
            raise
        synchronizations, second = 1, first
        report["synchronization_error"] = str(exc)[:500]
    finally:
        torch.cuda.set_sync_debug_mode("default")
        bindings.IndexerTopKBinding._worst_status = original
    report["synchronizations_outside_status_read"] = synchronizations
    report["status_reads"] = len(status_reads)
    report["litetopk_repeat_rows_differ"] = int((second != first).any(1).sum())

    # Rows that differ between the arms, and the recall of the LiteTopK rows.
    differ = (first != reference_out).any(1)
    report["rows_differ"] = int(differ.sum())
    hits = torch.zeros(_TOKENS, dtype=torch.int64, device=device)
    for start in range(0, _TOKENS, 8192):
        block = slice(start, start + 8192)
        table = torch.zeros((first[block].shape[0], keys + 1), dtype=torch.bool, device=device)
        table.scatter_(1, reference_out[block].clamp(min=0).long(), True)
        found = torch.gather(table, 1, first[block].clamp(min=0).long()) & (first[block] >= 0)
        hits[block] = found.sum(1)
    expected = (reference_out >= 0).sum(1)
    recall = hits.double() / expected.clamp(min=1).double()
    report["recall_min"] = float(recall[expected > 0].min())
    report["recall_mean"] = float(recall[expected > 0].mean())

    # The plan the binding must have followed, from the planner and the resolved tuning.
    resolved = lite.resolved_tuning(device)
    route = lite.plugin.info.route("fp8_paged")
    (plan,) = [
        plan_segment(
            segment,
            route=route,
            tuning=resolved,
            topk=topk,
            vote_rows=lite.plugin.module.carry_vote_rows(),
        )
        for segment in layout.segments
    ]
    report["planned_tiles"] = len(plan.tiles)
    report["planned_groups"] = len(plan.groups)
    report["planned_litetopk_rows"] = plan.litetopk_rows
    planned_tile_rows: dict[str, int] = {}
    for tile in plan.tiles:
        planned_tile_rows[str(tile.rows)] = planned_tile_rows.get(str(tile.rows), 0) + 1
    report["planned_tile_rows"] = planned_tile_rows
    report["tile_rows"] = report["litetopk"]["stats"]["tile_rows"]
    report["carries_after_call"] = sum(
        1 for key in getattr(lite.plugin.module, "_HOT_CARRY", {}) if key[0] == str(device)
    )
    report["resolved_tuning"] = resolved.as_dict()
    torch.cuda.synchronize()
    return report


if __name__ == "__main__":
    print(json.dumps(_child(json.loads(sys.argv[1])), sort_keys=True))
