# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Context-parallel shards selected one by one equal the whole prompt (optional GPU).

CP-sim on one Blackwell GPU: the rows of every rank of a contiguous P-rank split (P = 2, 4, 8) are
selected one rank at a time against all keys, with the layout that rank of Lite's context
parallelism builds, and the ranks together must equal the selection of the whole prompt:

* FP8 indexers (DSA native CP): one sequence, ``QueryLayout.contiguous(L, position=r * L,
  keys=S)``, and two packed sequences, ``QueryLayout.packed(cu, row_start=r * L, rows=L,
  absolute_ids=True)`` (ids into the gathered keys);
* MXFP4 indexers (CSA THD CP, four query tokens per compressed key): one sequence and two packed
  sequences, ``QueryLayout.packed(cu, row_start=r * L, rows=L, key_ratio=4,
  absolute_ids=False)`` (sequence-relative ids).

The matched-precision reference selector and plugin routes with exact selection must match the
whole prompt bit for bit. A fast route may differ where a rank's plan differs from the whole
prompt's (another reference bootstrap, tile grid or seed), among nearly tied keys only
(DESIGN-REVISIONS M7): every row of the CP-sim selection that differs from the exact reference
selection of the whole prompt is classified by the relative distance between the scores of the
keys it swaps and the row's cutoff (the smallest score among the reference's keys), with float64
scores of the quantized operands, and no row may be farther than the format's near-tie limit
(``_NEAR_TIE_LIMIT``, from the repeated runs of the previous integration). The whole prompt's own
fast selection is held to the same limit.

Plugins and the exact-tie top-k are given as for ``test_selector_gpu.py``
(``LITETOPK_TEST_SELECTORS``, ``LITETOPK_TEST_EXACT_TOPK``); every entry runs in a fresh
interpreter. ``LITETOPK_TEST_CP_INPUTS`` optionally adds real FP8 operands of one sequence: a JSON
list (inline or a file) of ``{"path": ..., "softmax_scale": ...}`` torch files holding
``{"tensors": {"q": [S, H, 128], "k": [S, 128], "weights": [S, H]}}``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpus(1, min_architecture="blackwell"), pytest.mark.optional]

_SELECTORS_VARIABLE = "LITETOPK_TEST_SELECTORS"
_EXACT_VARIABLE = "LITETOPK_TEST_EXACT_TOPK"
_INPUTS_VARIABLE = "LITETOPK_TEST_CP_INPUTS"
_PARTS = (2, 4, 8)
# Prompt tokens and the start of the second of two packed sequences, per format. The second
# sequence starts inside rank 0 of every split and is long enough for LiteTopK tiles: FP8 routes
# need 196608 keys per sequence, MXFP4 routes 65536 compressed keys (262144 tokens).
_TOKENS = {"fp8": 262144, "mxfp4": 270336}
_SECOND_SEQUENCE = {"fp8": 4100, "mxfp4": 6148}
_KEY_RATIO = {"fp8": 1, "mxfp4": 4}
# The farthest a fast route may swap a key from a row's cutoff score, relative to that score
# (DESIGN-REVISIONS M7: no row farther than the repeated runs of the previous integration on the
# same operands). FP8: the M7 "significant" line, 1e-3; the farthest swap of the previous
# integration's FP8 plugin on two GLM-5.2 layers was 4.0e-4 (4.3e-4 in CP-sim). MXFP4: the slab
# selector is not set-deterministic at near ties, and on the indexer operands of a DeepSeek-V4-
# sized C4 layer it swaps a key 1.097e-3 from the cutoff in 5 of 8 runs, in the previous
# integration as in this one (the same CUDA code): an inherent near tie above that line, which
# the limit (the largest distance of those repeated runs, rounded up) admits. On this file's
# random operands every swap of both formats' fast routes stays below 1e-4.
_NEAR_TIE_LIMIT = {"fp8": 1e-3, "mxfp4": 1.1e-3}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _read_json(variable: str):
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    return json.loads(raw if raw[0] in "[{" else Path(raw).read_text(encoding="utf-8"))


def _specs() -> list:
    return list(_read_json(_SELECTORS_VARIABLE) or []) or [None]


def _spec_id(spec) -> str:
    if spec is None:
        return "unset"
    name = Path(spec["litetopk"]["source"]).name
    return f"{name}-{spec['native_format']}-{spec['precision']}"


@pytest.mark.parametrize("spec", _specs(), ids=_spec_id)
def test_cp_sim_equals_whole_prompt(spec):
    exact = _read_json(_EXACT_VARIABLE)
    if spec is None or exact is None:
        pytest.skip(f"{_SELECTORS_VARIABLE} and {_EXACT_VARIABLE} are not both set")
    arguments = {"selector": spec, "exact_topk": exact, "inputs": _read_json(_INPUTS_VARIABLE)}
    result = subprocess.run(
        [sys.executable, __file__, json.dumps(arguments)],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    exact_route = spec["precision"] == "exact"
    limit = _NEAR_TIE_LIMIT[spec["native_format"]]
    for name, case in report["cases"].items():
        whole = case["whole"]
        assert whole["litetopk"]["tiles"] > 0, name
        if exact_route:
            assert whole["litetopk_vs_reference"]["rows"] == 0, (name, whole)
        else:
            assert whole["litetopk_vs_reference"]["max_distance"] <= limit, (name, whole)
        for parts, split in case["cp"].items():
            # The reference selector is exact and deterministic: bit for bit, every split.
            assert split["reference"]["rows_differ"] == 0, (name, parts)
            lite = split["litetopk"]
            # Every rank covered its rows once, LiteTopK where its plan allows.
            assert sum(rank["rows"] for rank in lite["ranks"]) == case["tokens"]
            assert all(rank["status_rows"] == {} for rank in lite["ranks"])
            if exact_route:
                assert lite["rows_differ"] == 0, (name, parts, lite)
            else:
                assert lite["vs_reference"]["max_distance"] <= limit, (name, parts, lite)


def cp_sim(binding, q, k, weights, *, parts: int, layout, topk: int, softmax_scale: float):
    """Select the rows of every rank of a contiguous ``parts``-rank split with
    ``layout(parts, rank)``; return the rows in order and the counters of every rank's call."""
    tokens = q.shape[0]
    local = tokens // parts
    outputs, ranks = [], []
    for rank in range(parts):
        rows = slice(rank * local, (rank + 1) * local)
        binding.stats.reset()
        with torch.no_grad():
            outputs.append(
                binding.select(
                    q[rows],
                    k,
                    weights[rows],
                    layout=layout(parts, rank),
                    topk=topk,
                    softmax_scale=softmax_scale,
                )
            )
        stats = binding.stats
        ranks.append(
            {
                "rows": stats.rows,
                "litetopk_rows": stats.litetopk_rows,
                "reference_rows": stats.reference_rows,
                "bootstrap_rows": stats.bootstrap_rows,
                "tiles": stats.tiles,
                "status_rows": dict(stats.status_rows),
                "recomputed_rows": stats.recomputed_rows,
            }
        )
    return torch.cat(outputs), ranks


def layouts(fmt: str, tokens: int, packed: list[int] | None):
    """The whole prompt's layout and the layout of rank ``r`` of a ``P``-rank split, as the
    format's context-parallel module builds them."""
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    ratio = _KEY_RATIO[fmt]
    absolute = fmt == "fp8"  # DSA ids index the gathered keys; CSA ids are sequence-relative
    if packed is None:
        whole = QueryLayout.full(tokens, keys=tokens // ratio, key_ratio=ratio)
    else:
        whole = QueryLayout.packed(
            packed, row_start=0, rows=tokens, key_ratio=ratio, absolute_ids=absolute
        )

    def rank_layout(parts: int, rank: int):
        local = tokens // parts
        if packed is None and fmt == "fp8":
            return QueryLayout.contiguous(local, position=rank * local, keys=tokens)
        # CSA THD describes one sequence as packed sequence offsets too.
        return QueryLayout.packed(
            [0, tokens] if packed is None else packed,
            row_start=rank * local,
            rows=local,
            key_ratio=ratio,
            absolute_ids=absolute,
        )

    return whole, rank_layout


def _dequantize_mxfp4(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Exact float64 values of packed E2M1 rows ``[..., 64]`` with UE8M0 group scales ``[...]``
    (four 32-value groups per 128-value row, group ``g`` in bits ``[8g, 8g + 8)``)."""
    nibbles = codes.view(torch.uint8).to(torch.int64)
    code = torch.stack((nibbles & 0xF, nibbles >> 4), dim=-1).flatten(-2)
    grid = torch.tensor(_E2M1, dtype=torch.float64, device=codes.device)
    value = torch.where((code & 0x8) != 0, -grid[code & 0x7], grid[code & 0x7])
    exponents = torch.stack(
        [(scales.to(torch.int64) >> shift) & 0xFF for shift in (0, 8, 16, 24)], dim=-1
    )
    group_scale = torch.exp2(exponents.to(torch.float64) - 127.0)
    return (value.unflatten(-1, (4, 32)) * group_scale[..., None]).flatten(-2)


class Float64Scorer:
    """float64 scores ``sum_h w[h] * relu(q[h] . k)`` of the quantized operands the selectors
    read (the queries quantized and their weights folded as for the score kernel)."""

    def __init__(self, fmt: str, q, k, weights, softmax_scale: float):
        from megatron.lite.primitive.kernels.indexer_topk.reference import quantize_keys

        self.fmt, self.q, self.weights, self.softmax_scale = fmt, q, weights, softmax_scale
        keys = quantize_keys(k, fmt)
        if fmt == "fp8":
            self.keys = keys.data.to(torch.float64) * keys.scale.to(torch.float64)[:, None]
        else:
            self.keys = _dequantize_mxfp4(keys.data, keys.scale)

    def scores(self, rows: torch.Tensor, key_rows: torch.Tensor) -> torch.Tensor:
        """float64 ``[len(rows), K]`` scores of ``key_rows[i]`` (``[len(rows), K]``) for query
        row ``rows[i]``."""
        from megatron.lite.primitive.kernels.indexer_topk.reference import quantize_queries

        data, scales, folded = quantize_queries(
            self.q[rows],
            self.weights[rows],
            self.fmt,
            softmax_scale=self.softmax_scale,
            kernel_heads=self.q.shape[1],
        )
        if self.fmt == "fp8":
            queries = data.to(torch.float64)
        else:
            queries = _dequantize_mxfp4(data, scales)
        dots = torch.einsum("rhd,rkd->rkh", queries, self.keys[key_rows])
        return (torch.relu(dots) * folded.to(torch.float64)[:, None, :]).sum(dim=-1)


def classify(scorer: Float64Scorer, layout, candidate, reference, chunk: int = 32) -> dict:
    """The rows whose candidate set differs from the reference set, and the largest relative
    distance from a row's cutoff score of a key in the symmetric difference of the two sets."""
    rows = torch.nonzero((candidate != reference).any(dim=1)).flatten()
    # Key row of an id: id - index_base + key_start of the row's segment.
    shift = torch.zeros(candidate.shape[0], dtype=torch.int64, device=candidate.device)
    for segment in layout.segments:
        shift[segment.row_start : segment.row_end] = segment.key_start - segment.index_base
    distances = []
    for first in range(0, rows.numel(), chunk):
        block = rows[first : first + chunk]
        ids = torch.cat((reference[block], candidate[block]), dim=1).to(torch.int64)
        valid = ids >= 0
        key_rows = torch.where(valid, ids + shift[block, None], torch.zeros_like(ids))
        scores = scorer.scores(block, key_rows)
        topk = reference.shape[1]
        ref_ids, cand_ids = ids[:, :topk], ids[:, topk:]
        in_ref = (cand_ids[:, :, None] == ref_ids[:, None, :]).any(-1) & valid[:, topk:]
        in_cand = (ref_ids[:, :, None] == cand_ids[:, None, :]).any(-1) & valid[:, :topk]
        ref_scores = scores[:, :topk].masked_fill(~valid[:, :topk], float("inf"))
        cutoff = ref_scores.min(dim=1).values
        swapped = torch.cat((valid[:, :topk] & ~in_cand, valid[:, topk:] & ~in_ref), dim=1)
        relative = (scores - cutoff[:, None]).abs() / cutoff.abs().clamp_min(1e-30)[:, None]
        distances.append(relative.masked_fill(~swapped, 0.0).amax(dim=1))
    distance = torch.cat(distances) if distances else torch.zeros(0, dtype=torch.float64)
    return {
        "rows": int(rows.numel()),
        "max_distance": float(distance.max()) if distance.numel() else 0.0,
        "rows_beyond_1e-3": int((distance > 1e-3).sum()),
    }


def bindings(spec: dict, heads: int, topk: int, fmt: str) -> dict:
    """The reference and LiteTopK bindings of a consumer with an indexer of ``fmt``."""
    from megatron.lite.primitive.kernels.indexer_topk import (
        ExactTopKConfig,
        IndexerGeometry,
        IndexerTopKConfig,
        IndexerTopKTuning,
        LiteTopKPluginConfig,
        LiteTopKPluginSettings,
    )
    from megatron.lite.primitive.modules.attention import indexer_topk

    selector = spec["selector"]
    exact_spec = dict(spec["exact_topk"])
    exact_spec.pop("pythonpath", None)
    geometry = IndexerGeometry(num_heads=heads, head_dim=128, topk=topk, key_ratio=_KEY_RATIO[fmt])

    class Consumer(torch.nn.Module):
        binding = None

        def indexer_geometry(self):
            return geometry

        def set_indexer_topk(self, binding):
            self.binding = binding

    tuning = IndexerTopKTuning(
        plugin_settings=LiteTopKPluginSettings(**selector.get("plugin_settings", {}))
    )
    arms = {}
    for backend in ("reference", "litetopk"):
        consumer = Consumer()
        indexer_topk.configure_indexer_topk(
            [consumer],
            IndexerTopKConfig(
                backend=backend,
                precision=selector["precision"],
                litetopk=LiteTopKPluginConfig(**selector["litetopk"]),
                exact_topk=ExactTopKConfig(**exact_spec),
            ),
            native_format=fmt,
            tuning=tuning,
        )
        arms[backend] = consumer.binding
    return arms


def operands(spec: dict, device) -> dict:
    """A random prompt (one sequence and two packed sequences) and the real operands."""
    selector = spec["selector"]
    fmt, heads = selector["native_format"], selector["heads"]
    tokens, ratio = _TOKENS[fmt], _KEY_RATIO[fmt]
    generator = torch.Generator(device=device).manual_seed(20260930)
    random = dict(
        q=torch.randn((tokens, heads, 128), generator=generator, device=device).to(torch.bfloat16),
        k=torch.randn((tokens // ratio, 128), generator=generator, device=device).to(
            torch.bfloat16
        ),
        weights=torch.rand((tokens, heads), generator=generator, device=device) * heads**-0.5,
        softmax_scale=128**-0.5,
    )
    cases = {
        "random": dict(random, packed=None),
        "random-packed": dict(random, packed=[0, _SECOND_SEQUENCE[fmt], tokens]),
    }
    if fmt == "fp8":
        for entry in spec.get("inputs") or []:
            payload = torch.load(entry["path"], map_location="cpu", weights_only=False, mmap=True)
            tensors = payload["tensors"]
            cases[Path(entry["path"]).stem] = dict(
                q=tensors["q"].to(device),
                k=tensors["k"].to(device),
                weights=tensors["weights"].to(device),
                softmax_scale=float(entry["softmax_scale"]),
                packed=None,
            )
    return cases


def _child(spec: dict) -> dict:
    exact_spec = dict(spec["exact_topk"])
    for entry in reversed(exact_spec.pop("pythonpath", [])):
        sys.path.insert(0, entry)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    selector = spec["selector"]
    fmt, heads, topk = selector["native_format"], selector["heads"], selector["topk"]
    arms = bindings(spec, heads, topk, fmt)
    report = {"format": fmt, "cases": {}}
    for name, case in operands(spec, device).items():
        q, k, weights, scale = case["q"], case["k"], case["weights"], case["softmax_scale"]
        tokens = q.shape[0]
        whole_layout, rank_layout = layouts(fmt, tokens, case["packed"])
        scorer = Float64Scorer(fmt, q, k, weights, scale)
        whole, entry = {}, {"tokens": tokens, "packed": case["packed"], "whole": {}, "cp": {}}
        for arm, binding in arms.items():
            binding.stats.reset()
            with torch.no_grad():
                whole[arm] = binding.select(
                    q, k, weights, layout=whole_layout, topk=topk, softmax_scale=scale
                )
            entry["whole"][arm] = {
                "tiles": binding.stats.tiles,
                "litetopk_rows": binding.stats.litetopk_rows,
            }
        entry["whole"]["litetopk_vs_reference"] = classify(
            scorer, whole_layout, whole["litetopk"], whole["reference"]
        )
        for parts in _PARTS:
            split = {}
            for arm, binding in arms.items():
                selected, ranks = cp_sim(
                    binding,
                    q,
                    k,
                    weights,
                    parts=parts,
                    layout=rank_layout,
                    topk=topk,
                    softmax_scale=scale,
                )
                differ = (selected != whole[arm]).any(1)
                local = tokens // parts
                for rank, counters in enumerate(ranks):
                    counters["rows_differ"] = int(differ[rank * local : (rank + 1) * local].sum())
                split[arm] = {"rows_differ": int(differ.sum()), "ranks": ranks}
                if arm == "litetopk":
                    split[arm]["vs_reference"] = classify(
                        scorer, whole_layout, selected, whole["reference"]
                    )
                del selected
            entry["cp"][str(parts)] = split
        report["cases"][name] = entry
        del whole, scorer
        torch.cuda.empty_cache()
    return report


if __name__ == "__main__":
    print(json.dumps(_child(json.loads(sys.argv[1])), sort_keys=True))
