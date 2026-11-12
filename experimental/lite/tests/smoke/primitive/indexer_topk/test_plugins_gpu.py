# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Load the external LiteTopK plugins and the exact-tie top-k (optional, one Blackwell GPU).

The plugins are not part of Megatron Lite, so these tests skip unless their locations are
given as JSON, inline or as the path of a JSON file::

    LITETOPK_TEST_PLUGINS='[{"source": "/path/to/plugin", "expected_source_id": "<12 hex>",
        "prebuilt_extension": "/path/to/extension.so", "prebuilt_extension_sha256": "<hex>",
        "settings": {"tie_policy": "logical-id", "score_policy": "native-fp32"}}]' \\
    LITETOPK_TEST_EXACT_TOPK='{"source": "/path/to/exact-tie", "pythonpath": ["/path/to/cudnn"]}' \\
    experimental/lite/tests/run_tests.sh \\
        experimental/lite/tests/smoke/primitive/indexer_topk/test_plugins_gpu.py

A plugin entry holds the ``LiteTopKPluginConfig`` fields plus optional ``settings``
(``LiteTopKPluginSettings`` fields). The exact top-k entry holds the ``ExactTopKConfig`` fields
plus optional ``pythonpath`` entries to prepend, for example a cuDNN frontend that provides the
DeepSeek sparse attention compiler. Every entry runs in a fresh interpreter, because plugin
settings are process-wide.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

_PLUGINS_VARIABLE = "LITETOPK_TEST_PLUGINS"
_EXACT_VARIABLE = "LITETOPK_TEST_EXACT_TOPK"
_GETENV = re.compile(r'getenv\(\s*"([A-Za-z0-9_]+)"\s*\)')


def _read_json(variable: str):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return json.loads(raw if raw[0] in "[{" else Path(raw).read_text(encoding="utf-8"))


_PLUGIN_SPECS = _read_json(_PLUGINS_VARIABLE) or []
_EXACT_SPEC = _read_json(_EXACT_VARIABLE)


def _spec_id(spec) -> str:
    return "unset" if spec is None else Path(spec["source"]).name


def _run_child(kind: str, spec: dict) -> dict:
    result = subprocess.run(
        [sys.executable, __file__, kind, json.dumps(spec)],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("spec", _PLUGIN_SPECS or [None], ids=_spec_id)
def test_plugin_loads_with_rendered_configuration(spec):
    if spec is None:
        pytest.skip(f"{_PLUGINS_VARIABLE} is not set")
    report = _run_child("plugin", spec)
    provenance = report["provenance"]
    info = provenance["plugin_info"]
    rendered = provenance["rendered_env"]

    # Pins and hashes: the source id is the plugin's own algorithm, reproduced before import.
    assert info["source_id"] == provenance["source_id"] == report["computed_source_id"]
    if spec.get("expected_source_id"):
        assert provenance["source_id"] == spec["expected_source_id"]
    if spec.get("prebuilt_extension_sha256"):
        assert provenance["prebuilt_extension_sha256"] == spec["prebuilt_extension_sha256"]
    assert report["idempotent"]

    # The module snapshotted exactly the rendered environment, which stays set after the load.
    effective = info["effective_config"]
    assert {key: effective.get(key) for key in rendered} == rendered
    assert {
        key: value
        for key, value in effective.items()
        if key not in rendered and value != provenance["diagnostic_env"].get(key)
    } == {}
    assert report["process_env"] == rendered
    assert info["routes"]

    # The launch-time keys are exactly the getenv reads of the CUDA sources; every one of them is
    # recorded at load (whether Lite's list of known keys has it or not), and a changed
    # launch-time key is detected before the next selection.
    assert sorted(info["launch_time_env_keys"]) == report["getenv_keys"]
    assert set(provenance["launch_time_env"]) == set(info["launch_time_env_keys"])
    assert report["launch_check"].startswith(f"{report['changed_key']} changed from ")
    assert "already loaded in this process with different settings" in report["conflict"]

    assert report["carry_vote_rows"] > 0
    assert report["production_min_s"][0] > 0 and report["production_min_s"][1] > 0


def test_exact_topk_selects_lowest_ids_on_ties():
    if _EXACT_SPEC is None:
        pytest.skip(f"{_EXACT_VARIABLE} is not set")
    report = _run_child("exact", _EXACT_SPEC)
    assert report["package"].startswith("megatron_lite_exact_topk_")
    if _EXACT_SPEC.get("expected_sha256"):
        assert report["file_sha256"] == {**report["file_sha256"], **_EXACT_SPEC["expected_sha256"]}
    assert report["mismatched_rows"] == []
    assert report["rows"] > 0


def _plugin_child(spec: dict) -> dict:
    import torch

    from megatron.lite.primitive.kernels.indexer_topk import (
        IndexerTopKPluginError,
        IndexerTopKRuntimeError,
        LiteTopKPluginConfig,
        LiteTopKPluginSettings,
    )
    from megatron.lite.primitive.kernels.indexer_topk.plugins import abi, loader

    spec = dict(spec)
    settings = LiteTopKPluginSettings(**spec.pop("settings", {}))
    config = LiteTopKPluginConfig(**spec)
    plugin = loader.load_litetopk_plugin(config, settings)
    report = {
        "provenance": plugin.provenance(),
        "idempotent": loader.load_litetopk_plugin(config, settings) is plugin,
        "process_env": {key: os.environ.get(key) for key in plugin.rendered_env},
        "computed_source_id": loader.compute_litetopk_source_id(config.source),
    }
    keys = set()
    for name in abi.LITETOPK_KERNEL_FILES:
        path = Path(config.source) / "litetopk_kernels" / name
        keys.update(_GETENV.findall(path.read_text(encoding="utf-8", errors="replace")))
    report["getenv_keys"] = sorted(keys)

    other = dataclasses.replace(
        settings, paged_pool_pages_per_row=(settings.paged_pool_pages_per_row or 16) + 1
    )
    try:
        loader.load_litetopk_plugin(config, other)
        report["conflict"] = ""
    except IndexerTopKPluginError as exc:
        report["conflict"] = str(exc)

    plugin.check_launch_time_env()
    key = min(plugin.launch_time_env)
    previous = os.environ.get(key)
    os.environ[key] = "changed-after-load"
    try:
        plugin.check_launch_time_env()
        report["launch_check"] = ""
    except IndexerTopKRuntimeError as exc:
        report["launch_check"] = str(exc)
    if previous is None:
        del os.environ[key]
    else:
        os.environ[key] = previous
    plugin.check_launch_time_env()
    report["changed_key"] = key

    # The ABI calls that need no selection work on the device right after the load.
    device = torch.device("cuda", torch.cuda.current_device())
    plugin.module.begin_call(device, ("indexer-topk-test",), 262144)
    plugin.module.drop_carry(device, ("indexer-topk-test",))
    plugin.module.release(device)
    torch.cuda.synchronize()
    report["carry_vote_rows"] = plugin.module.carry_vote_rows()
    report["production_min_s"] = [
        plugin.module.production_min_s(False),
        plugin.module.production_min_s(True),
    ]
    return report


def _exact_child(spec: dict) -> dict:
    spec = dict(spec)
    for entry in reversed(spec.pop("pythonpath", [])):
        sys.path.insert(0, entry)
    import torch

    from megatron.lite.primitive.kernels.indexer_topk import ExactTopKConfig
    from megatron.lite.primitive.kernels.indexer_topk.plugins import loader

    selector = loader.load_exact_topk(ExactTopKConfig(**spec))
    generator = torch.Generator(device="cuda").manual_seed(0)
    top_k, keys = 512, 6000
    lengths = [keys, 5999, 4096, 2048, 513, 512, 511, 100, 1, 0] * 4
    rows = len(lengths)
    # Few distinct values, so most cutoffs fall inside a run of equal scores. The rows are padded
    # (row stride > keys), as score kernels write them.
    padded = torch.randint(0, 48, (rows, keys + 128), generator=generator, device="cuda")
    scores = padded.float()[:, :keys]
    row_lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    indices, values = selector(scores, row_lengths, top_k=top_k, return_values=True)
    torch.cuda.synchronize()

    scores, indices, values = scores.cpu(), indices.cpu(), values.cpu()
    mismatched = []
    for row, length in enumerate(lengths):
        # Reference: score descending, lower column id first on equal scores (a stable sort).
        order = torch.sort(-scores[row, :length], stable=True).indices[:top_k]
        selected = indices[row][indices[row] >= 0]
        padding = indices[row][indices[row] < 0]
        if (
            sorted(selected.tolist()) != sorted(order.tolist())
            or padding.tolist() != [-1] * max(0, top_k - length)
            or not torch.equal(values[row][indices[row] >= 0], scores[row, selected.long()])
        ):
            mismatched.append(row)
    return {
        "package": selector.package,
        "file_sha256": selector.file_sha256,
        "rows": rows,
        "mismatched_rows": mismatched,
    }


if __name__ == "__main__":
    _kind, _spec = sys.argv[1], json.loads(sys.argv[2])
    _report = _plugin_child(_spec) if _kind == "plugin" else _exact_child(_spec)
    print(json.dumps(_report, sort_keys=True))
