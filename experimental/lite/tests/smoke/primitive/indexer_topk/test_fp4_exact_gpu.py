# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real exact FP4 dispatch and reference repair with an optional external plugin."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.optional, pytest.mark.gpus(1, min_architecture="blackwell")]


def _read(name):
    value = os.environ.get(name, "").strip()
    return json.loads(value if value[0] in "[{" else Path(value).read_text()) if value else None


_SPECS = [
    s
    for s in _read("LITETOPK_TEST_SELECTORS") or []
    if s["native_format"] == "mxfp4" and s["precision"] == "exact"
]


@pytest.mark.parametrize("spec", _SPECS or [None], ids=lambda s: str(s["heads"]) if s else "unset")
def test_fp4_exact_dispatch_and_repair(spec):
    exact = _read("LITETOPK_TEST_EXACT_TOPK")
    if spec is None or exact is None:
        pytest.skip("exact MXFP4 selector and exact top-k manifests are required")
    result = subprocess.run(
        [sys.executable, __file__, json.dumps({"selector": spec, "exact_topk": exact})],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-8000:] + result.stderr[-8000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert len(report) == 5


def _child(payload):
    import torch

    from megatron.lite.primitive.kernels.indexer_topk import (
        ExactTopKConfig,
        IndexerGeometry,
        IndexerTopKConfig,
        IndexerTopKTuning,
        LiteTopKPluginConfig,
        QueryLayout,
    )
    from megatron.lite.primitive.modules.attention.indexer_topk import configure_indexer_topk

    spec = payload["selector"]
    exact = dict(payload["exact_topk"])
    for path in reversed(exact.pop("pythonpath", [])):
        sys.path.insert(0, path)
    H, K, Q, S = spec["heads"], 512, 4096, 65536

    class Consumer(torch.nn.Module):
        def indexer_geometry(self):
            return IndexerGeometry(H, 128, K, 4)

        def set_indexer_topk(self, binding):
            self.binding = binding

    layout = QueryLayout.contiguous(Q, position=200000, keys=S, key_ratio=4)
    torch.manual_seed(918)
    q = torch.randn(Q, H, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(S, 128, device='cuda', dtype=torch.bfloat16)
    w = torch.rand(Q, H, device='cuda')
    results = []
    for case in ('random', 'zero-capacity', 'large-weights', 'large-q-scale', 'large-k-scale'):
        arms = {}
        for backend in ('reference', 'litetopk'):
            c = Consumer()
            install = configure_indexer_topk(
                [c],
                IndexerTopKConfig(
                    backend=backend,
                    precision='exact',
                    litetopk=LiteTopKPluginConfig(**spec['litetopk']),
                    exact_topk=ExactTopKConfig(**exact),
                ),
                native_format='mxfp4',
                tuning=IndexerTopKTuning(
                    required=case == "random", candidate_budget_bytes=4096 * 16384 * 8
                ),
            )
            arms[backend] = (c.binding, install)
        cq = q * (2.0**40 if case == 'large-q-scale' else 1.0)
        ck = k * (2.0**40 if case == 'large-k-scale' else 1.0)
        cw = w * (0.0 if case == 'zero-capacity' else 2.0**45 if case == 'large-weights' else 1.0)
        outs = {}
        for backend, (binding, _) in arms.items():
            with torch.no_grad():
                outs[backend] = binding.select(
                    cq, ck, cw, layout=layout, topk=K, softmax_scale=128**-0.5
                )
        diff = int((outs['reference'] != outs['litetopk']).any(1).sum())
        stats = arms['litetopk'][0].stats.as_dict()
        result = dict(case=case, rows_differ=diff, stats=stats)
        print(json.dumps(result), flush=True)
        results.append(result)
        assert diff == 0 and stats['litetopk_rows'] > 0, result
        assert (
            stats['recomputed_rows'] == 0 if case == 'random' else stats['recomputed_rows'] > 0
        ), result
        for binding, install in arms.values():
            install.release()
    return results


if __name__ == "__main__":
    print(json.dumps(_child(json.loads(sys.argv[1]))))
