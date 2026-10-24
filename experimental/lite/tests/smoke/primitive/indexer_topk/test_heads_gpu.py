# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Head-count negotiation and zero-head padding on a Blackwell GPU (optional).

Needs DeepGEMM and the exact-tie top-k package; the LiteTopK tests also need LiteTopK plugin
entries (FP8 with precision ``exact``, MXFP4 with precision ``fast``). Their locations are given
as JSON, inline or as the path of a JSON file, as for ``test_selector_gpu.py``::

    LITETOPK_TEST_SELECTORS='[{"native_format": "fp8", "precision": "exact", "heads": 32,
        "topk": 2048, "litetopk": {"source": "/path/to/glm-litetopk-raw32h64-abi1", ...}}]' \\
    LITETOPK_TEST_EXACT_TOPK='{"source": "/path/to/exact-tie", "pythonpath": ["/path/to/cudnn"]}' \\
    experimental/lite/tests/run_tests.sh \\
        experimental/lite/tests/smoke/primitive/indexer_topk/test_heads_gpu.py

The LiteTopK tests use the FP8 entries with precision ``exact`` and the MXFP4 entries with
precision ``fast`` (their ``heads`` field is not used: every test chooses its own head count) and
run in a fresh interpreter each, because plugin settings are process-wide:

* a 16-head FP8 layer padded to the route's 32-head kernels selects, on a 262144-token prompt,
  exactly what the reference selector selects on the same padded operands, and what it selects
  without padding;
* a 64-head FP8 layer, on a route with 64-head kernels: the default plan gives LiteTopK no row
  (measured slower than the reference selector), and with an explicit start position LiteTopK
  selects exactly what the reference selector selects;
* a 48-head MXFP4 layer padded to the route's 64-head kernels, on a 262144-token prompt: DeepGEMM
  scores 48 heads padded to 64 as well, so the default plan gives LiteTopK its rows from the MXFP4
  start position; LiteTopK selects every planned row without a status row or a recomputed row,
  and differs from the reference backend only among nearly tied keys (the recall bounds of
  ``test_selector_gpu.py``).

A LiteTopK test skips an entry whose route lacks the kernels it needs, as declared in the
plugin's ``plugin_info()`` (the 64-head test skips ``glm-litetopk-raw32-abi1``, which has 32-head
kernels only); which kernels a route has is known only once the plugin is loaded.

The reference tests show that padding preserves the reference selections: the DeepGEMM scores of
16 heads equal those of the same heads padded to 32, bit for bit, and so do the selections; 48
heads (which DeepGEMM 0.1.3 only scores padded to 64) select exactly the float64 top-k of inputs
whose float32 scores are exact.
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
_STARTUP = 188416
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _read_json(variable: str):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return json.loads(raw if raw[0] in "[{" else Path(raw).read_text(encoding="utf-8"))


_ENTRIES = _read_json(_SELECTORS_VARIABLE) or []
_SPECS = [
    spec for spec in _ENTRIES if spec["native_format"] == "fp8" and spec["precision"] == "exact"
] or [None]
_MXFP4_SPECS = [
    spec for spec in _ENTRIES if spec["native_format"] == "mxfp4" and spec["precision"] == "fast"
] or [None]
_EXACT_SPEC = _read_json(_EXACT_VARIABLE)


def _spec_id(spec) -> str:
    return "unset" if spec is None else Path(spec["litetopk"]["source"]).name


def _run(scenario: str, spec) -> dict:
    """Run one scenario in a child interpreter."""
    if _EXACT_SPEC is None or (spec is None and scenario != "reference"):
        pytest.skip(f"{_SELECTORS_VARIABLE} and {_EXACT_VARIABLE} are not both set")
    payload = {"scenario": scenario, "selector": spec, "exact_topk": _EXACT_SPEC}
    result = subprocess.run(
        [sys.executable, __file__, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("spec", _SPECS, ids=_spec_id)
def test_padded_litetopk_equals_padded_reference(spec):
    report = _run("padded", spec)
    if report.get("skip"):
        pytest.skip(report["skip"])
    heads = report["heads"]
    assert heads["num_heads"] == 16 and heads["litetopk_heads"] == heads["reference_heads"] == 32
    assert heads["baseline_heads"] == 16
    # Without an explicit start position the default plan gives LiteTopK no row: the reference
    # selector scores 16 heads natively, LiteTopK 32.
    assert report["default_startup"] is None
    stats = report["stats"]
    assert stats["litetopk_rows"] == report["planned_litetopk_rows"] > 0
    assert stats["padded_litetopk_rows"] == stats["litetopk_rows"]
    assert (
        stats["padded_reference_rows"]
        == stats["reference_rows"]
        == _TOKENS - stats["litetopk_rows"]
    )
    assert (stats["recomputed_rows"], stats["status_rows"]) == (0, {})
    # Exact: LiteTopK on the padded operands equals the reference selector on the same operands.
    assert report["rows_differ_vs_padded_reference"] == 0
    # Padding changes no selection of the reference selector either.
    assert report["rows_differ_padded_vs_unpadded_reference"] == 0


@pytest.mark.parametrize("spec", _SPECS, ids=_spec_id)
def test_h64_fp8_route_exact_vs_reference(spec):
    report = _run("h64", spec)
    if report.get("skip"):
        pytest.skip(report["skip"])
    assert report["heads"] == {
        "num_heads": 64,
        "litetopk_heads": 64,
        "reference_heads": 64,
        "baseline_heads": 64,
        "litetopk_padded": False,
        "reference_padded": False,
    }
    # The default plan keeps 64-head FP8 layers on the reference selector.
    assert report["default_startup"] is None
    assert report["default_stats"]["litetopk_rows"] == 0
    assert report["default_stats"]["reference_segments"] == {
        "no LiteTopK start position for the kernel heads": 1
    }
    # An explicit start position runs the route's 64-head (BLOCK_Q 2) kernels, exactly.
    stats = report["stats"]
    assert stats["litetopk_rows"] == report["planned_litetopk_rows"] > 0
    assert report["tile_rows"] == report["planned_tile_rows"]
    assert (stats["recomputed_rows"], stats["status_rows"], stats["padded_litetopk_rows"]) == (
        0,
        {},
        0,
    )
    assert report["rows_differ_vs_reference"] == 0


@pytest.mark.parametrize("spec", _MXFP4_SPECS, ids=_spec_id)
def test_mxfp4_padded_litetopk_default_plan(spec):
    report = _run("mxfp4-padded", spec)
    if report.get("skip"):
        pytest.skip(report["skip"])
    assert report["heads"] == {
        "num_heads": 48,
        "litetopk_heads": 64,
        "reference_heads": 64,
        "baseline_heads": 64,
        "litetopk_padded": True,
        "reference_padded": True,
    }
    # The reference backend pads 48 heads to 64 as well, so the default plan keeps the MXFP4 start.
    assert report["reference_backend_heads"]["reference_heads"] == 64
    assert report["default_startup"] is not None
    stats = report["stats"]
    assert stats["litetopk_rows"] == report["planned_litetopk_rows"] > 0
    assert stats["padded_litetopk_rows"] == stats["litetopk_rows"]
    assert (
        stats["padded_reference_rows"]
        == stats["reference_rows"]
        == _TOKENS - stats["litetopk_rows"]
    )
    assert (stats["recomputed_rows"], stats["status_rows"], stats["declined_tiles"]) == (0, {}, {})
    assert report["rows_not_ascending"] == report["ids_out_of_range"] == 0
    # Fast: the padded slab route may differ from the reference only among nearly tied keys.
    assert report["recall_min"] >= 0.99 and report["recall_mean"] >= 0.999


def test_padding_preserves_reference_sets():
    report = _run("reference", None)
    for case, result in report["cases"].items():
        # The scores of the unpadded and the padded heads are bitwise equal (no zero of either
        # sign differs either), and so are the selections.
        assert result["logits_bits_differ"] == 0, case
        assert result["rows_differ"] == 0, case
    for case, result in report["float64"].items():
        assert result["rows_differ"] == 0, case
        assert result["kernel_heads"] == 64, case
    assert sum(result["ties"] for result in report["float64"].values()) > 100


# ---------------------------------------------------------------------------------------------
# Child interpreter
# ---------------------------------------------------------------------------------------------


def _exact_inputs(torch, fmt: str, rows: int, keys: int, heads: int, seed: int):
    """Inputs whose quantized scores are exact in float32 (see test_reference_gpu.py)."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def draw(count: int, hot_column: int):
        if fmt == "fp8":
            values = torch.randint(-3, 4, (count, 128), generator=generator, device="cuda")
            values[:, hot_column] = 448
            return values.float()
        grid = torch.tensor(_E2M1, device="cuda")
        signs = torch.randint(0, 2, (count, 128), generator=generator, device="cuda") * 2 - 1
        values = grid[torch.randint(0, 5, (count, 128), generator=generator, device="cuda")]
        values = (values * signs).reshape(count, 4, 32)
        values[:, :, 7] = 6.0
        return values.reshape(count, 128)

    q = draw(rows * heads, 0).reshape(rows, heads, 128).to(torch.bfloat16)
    k = draw(keys, 1).to(torch.bfloat16)
    weights = torch.randint(-2, 5, (rows, heads), generator=generator, device="cuda").float()
    return q, k, weights


def _float64_topk(torch, fmt, q, k, weights, layout, topk, softmax_scale, ties):
    from megatron.lite.primitive.kernels.indexer_topk import (
        quantize_indexer_fp8_rows,
        quantize_indexer_mxfp4_rows,
        sort_topk_rows_,
    )

    def dequantize(value):
        if fmt == "fp8":
            data, scale = quantize_indexer_fp8_rows(value)
            return data.double() * scale.double()[..., None]
        packed, scales = quantize_indexer_mxfp4_rows(value)
        codes = packed.view(torch.uint8).long()
        code = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
        grid = torch.tensor(_E2M1, dtype=torch.float64, device=value.device)
        values = torch.where(code >= 8, -grid[code & 7], grid[code & 7])
        exponents = torch.stack([(scales >> shift) & 255 for shift in (0, 8, 16, 24)], -1)
        return (
            values.unflatten(-1, (4, 32)) * torch.pow(2.0, exponents.double() - 127)[..., None]
        ).flatten(-2)

    queries, keys = dequantize(q), dequantize(k)
    (segment,) = layout.segments
    out = torch.full((layout.rows, topk), -1, dtype=torch.int32, device=q.device)
    columns = torch.arange(segment.key_count, device=q.device)
    for first in range(0, layout.rows, 256):
        rows = torch.arange(first, min(first + 256, layout.rows), device=q.device)
        scores = torch.einsum("rhd,kd->rhk", queries[rows], keys).relu()
        scores = (scores * (weights[rows].double() * softmax_scale)[..., None]).sum(1)
        visible = ((rows + 1) // layout.key_ratio).clamp(max=segment.key_count)
        scores = scores.masked_fill(columns[None, :] >= visible[:, None], float("-inf"))
        ranked, order = torch.sort(-scores, dim=1, stable=True)
        if ranked.shape[1] > topk:
            ties.append(int(((visible > topk) & (ranked[:, topk - 1] == ranked[:, topk])).sum()))
        order = order[:, :topk]
        out[rows, : order.shape[1]] = torch.where(order < visible[:, None], order, -1).int()
    return sort_topk_rows_(out)


def _selector(torch, fmt, kernel_heads, exact_topk):
    from megatron.lite.primitive.kernels.indexer_topk import reference as reference_module

    return reference_module.ReferenceSelector(
        fmt=fmt,
        topk_kernel=reference_module.topk_kernel(exact_topk),
        kernel_heads=kernel_heads,
        budget_bytes=1 << 30,
        rows_per_call=None,
        num_sms=torch.cuda.get_device_properties(0).multi_processor_count,
    )


def _select_all(torch, selector, q, k, weights, layout, topk, softmax_scale):
    from megatron.lite.primitive.kernels.indexer_topk import sort_topk_rows_
    from megatron.lite.primitive.kernels.indexer_topk.reference import quantize_keys

    out = torch.empty((layout.rows, topk), dtype=torch.int32, device=q.device)
    keys = quantize_keys(k, selector.fmt)
    selector.select(
        q,
        weights,
        keys,
        layout=layout,
        row_ranges=((0, layout.rows),),
        topk=topk,
        softmax_scale=softmax_scale,
        out=out,
    )
    return sort_topk_rows_(out)


def _reference_child(torch, exact_topk) -> dict:
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout
    from megatron.lite.primitive.kernels.indexer_topk import reference as reference_module

    report = {"cases": {}, "float64": {}}
    generator = torch.Generator(device="cuda").manual_seed(20261001)
    for fmt, heads, padded in (("fp8", 16, 32), ("mxfp4", 16, 32), ("fp8", 16, 64)):
        ratio = 1 if fmt == "fp8" else 4
        rows, keys, topk = 8192, 8192 // ratio, 2048 if fmt == "fp8" else 512
        q = torch.randn((rows, heads, 128), generator=generator, device="cuda").to(torch.bfloat16)
        k = torch.randn((keys, 128), generator=generator, device="cuda").to(torch.bfloat16)
        weights = torch.randn((rows, heads), generator=generator, device="cuda")
        layout = QueryLayout.full(rows, keys=keys, key_ratio=ratio)
        scale = 128**-0.5
        # DeepGEMM scores of the same rows with their own heads and with zero heads appended.
        quantized = reference_module.quantize_keys(k, fmt)
        ends = ((torch.arange(rows, device="cuda") + 1) // ratio).clamp(max=keys).to(torch.int32)
        zeros = torch.zeros_like(ends)
        logits = []
        for kernel_heads in (heads, padded):
            data, scales, folded = reference_module.quantize_queries(
                q, weights, fmt, softmax_scale=scale, kernel_heads=kernel_heads
            )
            logits.append(
                reference_module._mqa_logits(
                    (data, scales),
                    (quantized.data, quantized.scale),
                    folded,
                    zeros,
                    ends,
                    max_seqlen_k=0,
                )
            )
        valid = torch.arange(keys, device="cuda")[None, :] < ends[:, None]
        bits = [value[valid].view(torch.int32) for value in logits]
        native = _select_all(
            torch, _selector(torch, fmt, heads, exact_topk), q, k, weights, layout, topk, scale
        )
        padded_out = _select_all(
            torch, _selector(torch, fmt, padded, exact_topk), q, k, weights, layout, topk, scale
        )
        report["cases"][f"{fmt}-{heads}-{padded}"] = {
            "logits_bits_differ": int((bits[0] != bits[1]).sum()),
            "logits_compared": int(bits[0].numel()),
            "rows_differ": int((native != padded_out).any(1).sum()),
        }
    # 48 heads: DeepGEMM scores them padded to 64; inputs with exact float32 scores, whose
    # float64 top-k is what an exact selector must return.
    for fmt in ("fp8", "mxfp4"):
        ratio = 1 if fmt == "fp8" else 4
        rows, keys, topk = 4096, 4096 // ratio, 700
        q, k, weights = _exact_inputs(torch, fmt, rows, keys, 48, seed=48)
        layout = QueryLayout.full(rows, keys=keys, key_ratio=ratio)
        kernel_heads = reference_module.score_kernel_heads(
            48, fmt=fmt, head_dim=128, device=q.device
        )
        selected = reference_module.reference_topk(
            q,
            k,
            weights,
            layout=layout,
            topk=topk,
            softmax_scale=0.5,
            fmt=fmt,
            exact_topk=exact_topk,
        )
        ties = []
        expected = _float64_topk(torch, fmt, q, k, weights, layout, topk, 0.5, ties)
        report["float64"][fmt] = {
            "kernel_heads": kernel_heads,
            "rows_differ": int((selected != expected).any(1).sum()),
            "ties": sum(ties),
        }
    return report


def _missing_kernels(scenario: str, route) -> str | None:
    """Why the FP8 route cannot run a LiteTopK scenario, or None when it can."""
    if scenario == "padded" and (16 in route.heads or route.padded_heads(16) != 32):
        return "the padding test needs 32-head kernels and none for 16 to 31 heads"
    if scenario == "h64" and 64 not in route.heads:
        return "the 64-head test needs 64-head kernels"
    return None


def _litetopk_child(torch, scenario: str, spec: dict, exact_topk) -> dict:
    from megatron.lite.primitive.kernels.indexer_topk import (
        IndexerGeometry,
        IndexerTopKConfig,
        IndexerTopKConfigError,
        IndexerTopKTuning,
        LiteTopKPluginConfig,
        LiteTopKPluginSettings,
        QueryLayout,
    )
    from megatron.lite.primitive.kernels.indexer_topk.planner import plan_segment
    from megatron.lite.primitive.kernels.indexer_topk.plugins.loader import loaded_litetopk_plugins
    from megatron.lite.primitive.modules.attention import indexer_topk as bindings

    heads = 16 if scenario == "padded" else 64
    geometry = IndexerGeometry(num_heads=heads, head_dim=128, topk=2048, key_ratio=1)

    class Consumer(torch.nn.Module):
        binding = None

        def indexer_geometry(self):
            return geometry

        def set_indexer_topk(self, binding):
            self.binding = binding

    plugin = LiteTopKPluginConfig(**spec["litetopk"])
    settings = dict(spec.get("plugin_settings", {}))

    def bind(backend, tuning):
        consumer = Consumer()
        bindings.configure_indexer_topk(
            [consumer],
            IndexerTopKConfig(
                backend=backend,
                precision="exact",
                litetopk=plugin,
                exact_topk=exact_topk,
                head_padding=scenario == "padded",
            ),
            native_format="fp8",
            tuning=tuning,
        )
        return consumer.binding

    device = torch.device("cuda", 0)
    failure = None
    try:
        default = bind(
            "litetopk", IndexerTopKTuning(plugin_settings=LiteTopKPluginSettings(**settings))
        )
    except IndexerTopKConfigError as error:
        failure = error
    # configure_indexer_topk loads the plugin before it negotiates the layer's head counts.
    plugins = loaded_litetopk_plugins()
    route = plugins[-1].info.route("fp8_paged") if plugins else None
    missing = None if route is None else _missing_kernels(scenario, route)
    if missing is not None:
        return {
            "skip": f"route fp8_paged of LiteTopK source {plugins[-1].source_id} has kernels for "
            f"{sorted(route.heads)} heads; {missing}"
        }
    if failure is not None:
        raise failure
    report = {
        "heads": default.heads.as_dict(),
        "default_startup": default.resolved_tuning(device).startup_position,
    }
    tile_rows = default.resolved_tuning(device).tile_rows
    admitted = default.plugin.settings.fp8_paged_admit_max_query_len
    explicit = bind(
        "litetopk",
        IndexerTopKTuning(
            required=True,
            startup_position=_STARTUP,
            tile_rows=tile_rows,
            # The same plugin settings as the default binding: one profile per process.
            plugin_settings=LiteTopKPluginSettings(
                **{**settings, "fp8_paged_admit_max_query_len": admitted}
            ),
        ),
    )
    generator = torch.Generator(device=device).manual_seed(20261001 + heads)
    q = torch.randn((_TOKENS, heads, 128), generator=generator, device=device).to(torch.bfloat16)
    k = torch.randn((_TOKENS, 128), generator=generator, device=device).to(torch.bfloat16)
    weights = torch.rand((_TOKENS, heads), generator=generator, device=device) * heads**-0.5
    layout = QueryLayout.full(_TOKENS, keys=_TOKENS)
    scale = 128**-0.5

    def select(binding):
        with torch.no_grad():
            return binding.select(q, k, weights, layout=layout, topk=2048, softmax_scale=scale)

    if scenario == "h64":
        out = select(default)
        report["default_stats"] = default.stats.as_dict()
        del out
    lite = select(explicit)
    report["stats"] = explicit.stats.as_dict()
    (plan,) = [
        plan_segment(
            segment,
            key_ratio=1,
            route=route,
            tuning=explicit.resolved_tuning(device),
            topk=2048,
            vote_rows=explicit.plugin.module.carry_vote_rows(),
        )
        for segment in layout.segments
    ]
    report["planned_litetopk_rows"] = plan.litetopk_rows
    planned: dict[str, int] = {}
    for tile in plan.tiles:
        planned[str(tile.rows)] = planned.get(str(tile.rows), 0) + 1
    report["planned_tile_rows"] = planned
    report["tile_rows"] = report["stats"]["tile_rows"]
    reference = bind("reference", None)
    unpadded = select(reference)
    if scenario == "padded":
        kernel_heads = explicit.heads.litetopk_heads
        padded = _select_all(
            torch,
            _selector(torch, "fp8", kernel_heads, exact_topk),
            q,
            k,
            weights,
            layout,
            2048,
            scale,
        )
        report["rows_differ_vs_padded_reference"] = int((lite != padded).any(1).sum())
        report["rows_differ_padded_vs_unpadded_reference"] = int((padded != unpadded).any(1).sum())
    else:
        report["rows_differ_vs_reference"] = int((lite != unpadded).any(1).sum())
    torch.cuda.synchronize()
    return report


def _mxfp4_padded_child(torch, spec: dict, exact_topk) -> dict:
    from megatron.lite.primitive.kernels.indexer_topk import (
        IndexerGeometry,
        IndexerTopKConfig,
        IndexerTopKConfigError,
        IndexerTopKTuning,
        LiteTopKPluginConfig,
        LiteTopKPluginSettings,
        QueryLayout,
    )
    from megatron.lite.primitive.kernels.indexer_topk.planner import plan_segment
    from megatron.lite.primitive.kernels.indexer_topk.plugins.loader import loaded_litetopk_plugins
    from megatron.lite.primitive.modules.attention import indexer_topk as bindings

    heads, ratio, topk = 48, 4, spec["topk"]
    geometry = IndexerGeometry(num_heads=heads, head_dim=128, topk=topk, key_ratio=ratio)

    class Consumer(torch.nn.Module):
        binding = None

        def indexer_geometry(self):
            return geometry

        def set_indexer_topk(self, binding):
            self.binding = binding

    def bind(backend, tuning):
        consumer = Consumer()
        bindings.configure_indexer_topk(
            [consumer],
            IndexerTopKConfig(
                backend=backend,
                precision="fast",
                litetopk=LiteTopKPluginConfig(**spec["litetopk"]),
                exact_topk=exact_topk,
                head_padding=backend == "litetopk",
            ),
            native_format="mxfp4",
            tuning=tuning,
        )
        return consumer.binding

    device = torch.device("cuda", 0)
    failure = None
    try:
        # The default plan; required: binding fails if it gives LiteTopK no row.
        lite = bind(
            "litetopk",
            IndexerTopKTuning(
                required=True,
                plugin_settings=LiteTopKPluginSettings(**spec.get("plugin_settings", {})),
            ),
        )
    except IndexerTopKConfigError as error:
        failure = error
    plugins = loaded_litetopk_plugins()
    route = plugins[-1].info.route("fp4_slab") if plugins else None
    if route is not None and (heads in route.heads or route.padded_heads(heads) is None):
        return {
            "skip": f"route fp4_slab of LiteTopK source {plugins[-1].source_id} has kernels for "
            f"{sorted(route.heads)} heads; the padding test needs kernels for more than 48 heads "
            "and none for 48"
        }
    if failure is not None:
        raise failure
    reference = bind("reference", None)
    generator = torch.Generator(device=device).manual_seed(20261001 + heads)
    keys = _TOKENS // ratio
    q = torch.randn((_TOKENS, heads, 128), generator=generator, device=device).to(torch.bfloat16)
    k = torch.randn((keys, 128), generator=generator, device=device).to(torch.bfloat16)
    weights = torch.rand((_TOKENS, heads), generator=generator, device=device) * heads**-0.5
    layout = QueryLayout.full(_TOKENS, keys=keys, key_ratio=ratio)

    def select(binding):
        with torch.no_grad():
            return binding.select(q, k, weights, layout=layout, topk=topk, softmax_scale=128**-0.5)

    out = select(lite)
    expected = select(reference)
    resolved = lite.resolved_tuning(device)
    (plan,) = [
        plan_segment(
            segment,
            key_ratio=ratio,
            route=route,
            tuning=resolved,
            topk=topk,
            vote_rows=lite.plugin.module.carry_vote_rows(),
        )
        for segment in layout.segments
    ]
    valid = out >= 0
    ascending = ((out[:, 1:] > out[:, :-1]) | ~valid[:, 1:]).all(1)
    ascending &= (valid[:, :-1] | ~valid[:, 1:]).all(1)
    hits = torch.zeros(_TOKENS, dtype=torch.int64, device=device)
    for start in range(0, _TOKENS, 8192):
        block = slice(start, start + 8192)
        table = torch.zeros((out[block].shape[0], keys + 1), dtype=torch.bool, device=device)
        table.scatter_(1, expected[block].clamp(min=0).long(), True)
        found = torch.gather(table, 1, out[block].clamp(min=0).long()) & (out[block] >= 0)
        hits[block] = found.sum(1)
    count = (expected >= 0).sum(1)
    recall = hits.double() / count.clamp(min=1).double()
    report = {
        "heads": lite.heads.as_dict(),
        "reference_backend_heads": reference.heads.as_dict(),
        "default_startup": resolved.startup_position,
        "stats": lite.stats.as_dict(),
        "planned_litetopk_rows": plan.litetopk_rows,
        "rows_not_ascending": int((~ascending).sum()),
        "ids_out_of_range": int((out >= keys).sum()),
        "rows_differ_vs_reference": int((out != expected).any(1).sum()),
        "recall_min": float(recall[count > 0].min()),
        "recall_mean": float(recall[count > 0].mean()),
    }
    torch.cuda.synchronize()
    return report


def _child(payload: dict) -> dict:
    exact_spec = dict(payload["exact_topk"])
    for entry in reversed(exact_spec.pop("pythonpath", [])):
        sys.path.insert(0, entry)
    import torch

    from megatron.lite.primitive.kernels.indexer_topk import ExactTopKConfig

    torch.cuda.set_device(0)
    exact_topk = ExactTopKConfig(**exact_spec)
    if payload["scenario"] == "reference":
        return _reference_child(torch, exact_topk)
    if payload["scenario"] == "mxfp4-padded":
        return _mxfp4_padded_child(torch, payload["selector"], exact_topk)
    return _litetopk_child(torch, payload["scenario"], payload["selector"], exact_topk)


if __name__ == "__main__":
    print(json.dumps(_child(json.loads(sys.argv[1])), sort_keys=True))
