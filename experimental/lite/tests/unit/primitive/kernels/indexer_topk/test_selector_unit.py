# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the LiteTopK selection engine and the per-layer indexer top-k bindings.

The LiteTopK plugin is a pure-Python ABI v1 adapter tree built in tmp_path and loaded through
the real loader. Its kernels are emulated on the CPU: the score kernel of the reference selector
and the plugin's tile selection compute the same float64 scores from the quantized operands they
receive, so every selected set is known exactly. The tests check what surrounds the kernels:
plans, call sequences, carry keys, fallbacks, status handling, ordering and counters.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

from megatron.lite.primitive.kernels.indexer_topk import (
    ExactTopKConfig,
    IndexerGeometry,
    IndexerTopKConfig,
    IndexerTopKConfigError,
    IndexerTopKPluginError,
    IndexerTopKRuntimeError,
    IndexerTopKStats,
    IndexerTopKTuning,
    LiteTopKPluginConfig,
    LiteTopKPluginSettings,
    QueryLayout,
    QuerySegment,
    normalize_indexer_topk_config,
)
from megatron.lite.primitive.kernels.indexer_topk import reference as reference_module
from megatron.lite.primitive.kernels.indexer_topk import (
    release_indexer_topk_workspaces,
    sort_topk_rows_,
)
from megatron.lite.primitive.kernels.indexer_topk.plugins import cache, env, loader
from megatron.lite.primitive.kernels.indexer_topk.reference import quantize_keys, quantize_queries

pytestmark = pytest.mark.mlite

# megatron.lite.primitive.modules.attention.indexer_topk. The fixture below imports it, after the
# Transformer Engine stub that the attention package needs on machines without Transformer Engine.
binding_module = None

_PREFIX = "SGLANG_LITETOPK"
_SCORE = "SGLANG_LITETOPK_H32_SCORE_POLICY"
_TIE = "SGLANG_LITETOPK_H32_TIE_POLICY"
_ADMIT = "SGLANG_LITETOPK_FP8_PAGED_ADMIT_MAX_Q"
_READS = (
    "SGLANG_LITETOPK",
    "SGLANG_LITETOPK_COLDSTART_IDENTITY",
    "SGLANG_LITETOPK_EXPERIMENTAL_CP_SMALL_Q",
    "SGLANG_LITETOPK_FP8_LARGE_Q_ROW_TILES",
    _ADMIT,
    _SCORE,
    _TIE,
    "SGLANG_LITETOPK_MERGE_CAP",
    "SGLANG_LITETOPK_PAGED_CANDIDATES",
    "SGLANG_LITETOPK_PAGED_POOL_PAGES_PER_ROW",
    "SGLANG_LITETOPK_RELEASE_SCRATCH_ON_ROLLBACK",
    "SGLANG_LITETOPK_TIERED_SEED_12K",
)
_KERNEL_FILES = ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh", "dense_topk_litetopk.cuh")

HEADS, HEAD_DIM, TOPK, VOTE_ROWS, HOT_PREFIX = 8, 128, 8, 12, 24
FP8_GEOMETRY = IndexerGeometry(num_heads=HEADS, head_dim=HEAD_DIM, topk=TOPK, key_ratio=1)
FP4_GEOMETRY = IndexerGeometry(num_heads=HEADS, head_dim=HEAD_DIM, topk=TOPK, key_ratio=4)
_FP8_ROUTE = {
    "name": "fp8_paged",
    "fmt": "fp8",
    "heads": [HEADS],
    "head_dims": [HEAD_DIM],
    "topk": [TOPK],
    "max_topk": TOPK,
    "qualified_query_lengths": [],
    "admitted_max_query_len": 0,  # replaced by the admission the adapter was imported with
    "min_keys": 96,
    "max_keys": 4096,
    "hot_prefix": HOT_PREFIX,
    "exact": False,
    "tie_policies": ["logical-id", "logical-id-desc", "storage"],
    "score_policies": ["folded", "native-fp32"],
}
_FP4_ROUTE = {
    **_FP8_ROUTE,
    "name": "fp4_slab",
    "fmt": "mxfp4",
    "topk": None,
    "qualified_query_lengths": [12, 16],
    "min_keys": 64,
    "tie_policies": ["storage"],
    "score_policies": ["folded"],
}
# A scaled-down wave: a reference prefix up to position 160, 16-row tiles in groups of three.
FP8_TUNING = IndexerTopKTuning(tile_rows=16, startup_position=160, group_tiles=3)
# MXFP4 tiles from the first row that sees the HOT prefix (the scaled-down sequences are far
# below the default start position).
FP4_TUNING = IndexerTopKTuning(tile_rows=16, startup_position=0)

# The adapter only forwards every ABI call to the FakeLiteTopK the test installs.
_FAKE_ADAPTER = '''
"""A pure-Python ABI v1 LiteTopK adapter that forwards its calls to a test double."""
import hashlib
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "fake_plugin.json"), encoding="utf-8") as _handle:
    _CONFIG = json.load(_handle)

LITETOPK_ABI_VERSION = 1
_EFFECTIVE = {key: os.environ.get(key) for key in _CONFIG["reads"]}
BEHAVIOR = None


def plugin_info():
    digest = hashlib.sha256()
    for name in ("dsa_litetopk.cu", "sm100_dsa_litetopk.cuh", "dense_topk_litetopk.cuh"):
        digest.update(name.encode())
        with open(os.path.join(_HERE, "litetopk_kernels", name), "rb") as handle:
            digest.update(handle.read())
    routes = [dict(route) for route in _CONFIG["routes"]]
    for route in routes:
        if route["name"] == "fp8_paged":
            route["admitted_max_query_len"] = int(_EFFECTIVE.get(_CONFIG["admit_key"]) or 0)
    return {
        "abi": 1,
        "source_id": digest.hexdigest()[:12],
        "routes": routes,
        "effective_config": dict(_EFFECTIVE),
        "launch_time_env_keys": _CONFIG["launch_keys"],
        "tie_policy": _EFFECTIVE.get(_CONFIG["tie_key"]) or "storage",
        "score_policy": _EFFECTIVE.get(_CONFIG["score_key"]) or "folded",
    }


def load_extension(
    *, prebuilt_path=None, prebuilt_sha256=None, build_dir=None, deepgemm_include_dir=None,
    cuda_arch="10.0a",
):
    return None


def production_min_s(use_fp4):
    return _CONFIG["min_keys"][1 if use_fp4 else 0]


def carry_vote_rows():
    return _CONFIG["vote_rows"]


def begin_call(device, hot_key, sequence_length):
    return BEHAVIOR.begin_call(device, hot_key, sequence_length)


def prepare_permuted_gather(
    kv_cache, dst_k, dst_scale, block_table, *, sequence_length, query_length, num_reqs,
    common_end, window_start, hot_key,
):
    return BEHAVIOR.prepare_permuted_gather(
        kv_cache, dst_k, dst_scale, block_table, sequence_length=sequence_length,
        query_length=query_length, num_reqs=num_reqs, common_end=common_end,
        window_start=window_start, hot_key=hot_key,
    )


def try_large_exact_once_chunk(
    q, k, k_scale, weights, ks, ke, out_idx, topk, *, permuted_plan, num_reqs, ke_min_hint,
    cap=None, hot_key=None, ks_common_hint=0, carry_extent_hint=None,
    carry_recent_rows_hint=None, q_sf=None, carry_io=True, exact=False, status_out=None,
):
    return BEHAVIOR.try_large_exact_once_chunk(
        q, k, k_scale, weights, ks, ke, out_idx, topk, permuted_plan=permuted_plan,
        num_reqs=num_reqs, ke_min_hint=ke_min_hint, cap=cap, hot_key=hot_key,
        ks_common_hint=ks_common_hint, carry_extent_hint=carry_extent_hint,
        carry_recent_rows_hint=carry_recent_rows_hint, q_sf=q_sf, carry_io=carry_io,
        exact=exact, status_out=status_out,
    )


def stash_carry(hot_key, idx, S, min_index=0, *, recent_rows_hint=None):
    return BEHAVIOR.stash_carry(hot_key, idx, S, min_index, recent_rows_hint=recent_rows_hint)


def drop_carry(device, hot_key):
    return BEHAVIOR.drop_carry(device, hot_key)


def release(device, *, release_scratch=True):
    return BEHAVIOR.release(device, release_scratch=release_scratch)
'''

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float64)


def _dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    codes = packed.view(torch.uint8).to(torch.int64)
    code = torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(*packed.shape[:-1], -1)
    value = torch.where((code & 8) != 0, -_E2M1[code & 7], _E2M1[code & 7])
    exponents = torch.stack([(scales >> shift) & 0xFF for shift in (0, 8, 16, 24)], dim=-1)
    group_scale = torch.pow(2.0, exponents.to(torch.float64) - 127.0)
    return (value.reshape(*value.shape[:-1], 4, 32) * group_scale[..., None]).flatten(-2)


def _row_scores(q_data, q_sf, weights, k_data, k_scale, row, start, end) -> torch.Tensor:
    """float32 scores of one quantized query row against the keys [start, end)."""
    if q_sf is None:  # FP8: the query scales are folded into the weights
        queries, keys = q_data[row].double(), k_data[start:end].double()
        key_scale = k_scale[start:end].double()
    else:
        queries = _dequantize_mxfp4(q_data[row], q_sf[row])
        keys = _dequantize_mxfp4(k_data[start:end], k_scale[start:end])
        key_scale = None
    scores = (torch.relu(queries @ keys.T) * weights[row].double()[:, None]).sum(0)
    if key_scale is not None:
        scores = scores * key_scale
    return scores.float()


def _best(scores: torch.Tensor, topk: int) -> torch.Tensor:
    """Score descending, lower id first on equal scores."""
    return torch.sort(-scores.double(), stable=True).indices[:topk].int()


class FakeScoreKernel:
    """CPU emulation of the reference score kernel (plain or row-relative float32 score rows)."""

    def __init__(self):
        self.calls = []

    def __call__(self, q, kv, weights, ks, ke, *, max_seqlen_k):
        rows = q[0].shape[0]
        self.calls.append({"rows": rows, "ks": ks.tolist(), "ke": ke.tolist()})
        # max_seqlen_k == 0: one column per key; otherwise columns relative to the row's window.
        logits = torch.full(
            (rows, max_seqlen_k or kv[0].shape[0]), float("nan"), dtype=torch.float32
        )
        for row in range(rows):
            start, end = int(ks[row]), int(ke[row])
            if end > start:
                first = start if max_seqlen_k == 0 else 0
                logits[row, first : first + end - start] = _row_scores(
                    *q, weights, *kv, row, start, end
                )
        return logits

    @property
    def rows(self) -> int:
        return sum(call["rows"] for call in self.calls)


def exact_topk_kernel(scores, lengths, top_k):
    ids = torch.full((scores.shape[0], top_k), -1, dtype=torch.int32)
    for row in range(scores.shape[0]):
        order = _best(scores[row, : int(lengths[row])], top_k)
        ids[row, : order.numel()] = order
    return ids


class FakeLiteTopK:
    """The test double behind the fake adapter: exact tile selection plus scripted faults.

    ``events`` records every ABI call. ``decline_plans`` / ``decline_tiles`` hold the indices
    of plan / tile calls that decline on the host; ``row_status`` maps a tile call index to
    ``{row offset: status code}``. A carry published by a tile with a failed row is invalid, and
    every tile of a group planned from an invalid carry fails (a poisoned seed).
    """

    def __init__(self):
        self.events = []
        self.carries = {}
        self.decline_plans = set()
        self.decline_tiles = set()
        self.row_status = {}
        self.raise_in_tile = None
        self.plans = 0
        self.tiles = 0
        self.gathered = None
        self.operands = []

    def begin_call(self, device, hot_key, sequence_length):
        self.events.append(("begin_call", hot_key[1:], sequence_length))
        self.carries.pop(hot_key, None)

    def prepare_permuted_gather(
        self,
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        *,
        sequence_length,
        query_length,
        num_reqs,
        common_end,
        window_start,
        hot_key,
    ):
        index, self.plans = self.plans, self.plans + 1
        carry = self.carries.get(hot_key)
        seed = None if carry is None else carry["kind"]
        assert num_reqs == 1 and window_start == 0
        assert carry is None or carry["extent"] <= common_end
        assert dst_k.shape[0] == sequence_length and dst_scale.shape == (sequence_length, 4)
        self.events.append(("plan", hot_key[1:], sequence_length, query_length, common_end, seed))
        if index in self.decline_plans:
            return None
        # Gather the block-major cache through the block table (identity order).
        value_bytes = dst_k.shape[1]
        values, scales = cache.block_major_cache_views(kv_cache, value_bytes)
        blocks = block_table[0].long()
        dst_k.view(torch.uint8).copy_(values[blocks].reshape(-1, value_bytes)[:sequence_length])
        dst_scale.copy_(scales[blocks].reshape(-1, 4)[:sequence_length])
        self.gathered = (dst_k.view(torch.uint8).clone(), dst_scale.clone())
        return {"poisoned": bool(carry and carry["poisoned"])}

    def try_large_exact_once_chunk(
        self,
        q,
        k,
        k_scale,
        weights,
        ks,
        ke,
        out_idx,
        topk,
        *,
        permuted_plan,
        num_reqs,
        ke_min_hint,
        cap,
        hot_key,
        ks_common_hint,
        carry_extent_hint,
        carry_recent_rows_hint,
        q_sf,
        carry_io,
        exact,
        status_out,
    ):
        index, self.tiles = self.tiles, self.tiles + 1
        rows = q.shape[0]
        self.events.append(
            ("tile", hot_key[1:], int(ke[0]), rows, carry_io, cap, exact, carry_extent_hint)
        )
        assert num_reqs == 1 and ks_common_hint == 0 and carry_recent_rows_hint == VOTE_ROWS
        assert int(ks.max()) == 0 and int(ke.min()) >= ke_min_hint >= HOT_PREFIX
        assert int(ke[-1]) == carry_extent_hint
        assert out_idx.shape == (rows, topk) and status_out.shape == (rows,)
        if index in self.decline_tiles:
            return False
        if index == self.raise_in_tile:
            raise RuntimeError("fake kernel fault")
        self.operands.append((q, q_sf, weights))
        codes = torch.zeros(rows, dtype=torch.int32)
        if permuted_plan["poisoned"]:
            codes.fill_(3)
        for offset, code in self.row_status.get(index, {}).items():
            codes[offset] = code
        for row in range(rows):
            if codes[row]:
                out_idx[row] = torch.arange(topk, dtype=torch.int32) + 1  # a wrong result
                continue
            scores = _row_scores(q, q_sf, weights, k, k_scale, row, 0, int(ke[row]))
            out_idx[row] = _best(scores, topk).flip(0)  # not ascending: the caller sorts
        status_out.copy_(codes)
        if carry_io:
            self.carries[hot_key] = {
                "kind": "tile",
                "extent": carry_extent_hint,
                "poisoned": bool((codes == 3).any()),
            }
        return True

    def stash_carry(self, hot_key, idx, S, min_index=0, *, recent_rows_hint=None):
        assert min_index == 0 and idx.shape[1] == TOPK and int(idx.max()) < S
        self.events.append(("stash", hot_key[1:], idx.shape[0], S, recent_rows_hint))
        self.carries[hot_key] = {"kind": "stash", "extent": S, "poisoned": False}
        self.stashed = idx.clone()

    def drop_carry(self, device, hot_key):
        self.events.append(("drop", hot_key[1:]))
        self.carries.pop(hot_key, None)

    def release(self, device, *, release_scratch=True):
        self.events.append(("release", str(device), release_scratch))

    def names(self) -> list[str]:
        return [event[0] for event in self.events]


def _plugin_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key.startswith(_PREFIX) or key in env.LAUNCH_TIME_ENV_KEYS
    }


def _fake_modules() -> list[str]:
    return sorted(name for name in sys.modules if name.startswith("megatron_lite_litetopk_"))


def _new_process() -> None:
    """Forget the loaded plugins and their environment, as a fresh process would."""
    for key in _plugin_env():
        os.environ.pop(key)
    for name in _fake_modules():
        sys.modules.pop(name)
    env._RENDERED_BY.clear()
    env._LAUNCH_ENV.clear()
    loader._LOADED.clear()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, transformer_engine_import_stub):
    """An empty plugin ledger and environment, CPU stand-ins for the kernels and the device."""
    global binding_module
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.attention import indexer_topk

    binding_module = indexer_topk
    for key in _plugin_env():
        monkeypatch.delenv(key)
    for name in _fake_modules():
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(env, "_RENDERED_BY", {})
    monkeypatch.setattr(env, "_LAUNCH_ENV", {})
    monkeypatch.setattr(loader, "_LOADED", {})
    monkeypatch.setattr(loader, "_missing_packages", lambda: [])
    monkeypatch.setattr(binding_module, "_num_sms", lambda device: 4)
    monkeypatch.setattr(binding_module, "_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(binding_module, "_module_device", lambda module: torch.device("cpu"))
    monkeypatch.setattr(binding_module, "topk_kernel", lambda exact_topk: exact_topk_kernel)
    monkeypatch.setattr(binding_module, "score_kernel_heads", lambda heads, **kwargs: heads)
    yield
    release_indexer_topk_workspaces()
    for key in _plugin_env():
        os.environ.pop(key)
    for name in _fake_modules():
        sys.modules.pop(name)


@pytest.fixture
def score_kernel(monkeypatch):
    kernel = FakeScoreKernel()
    monkeypatch.setattr(reference_module, "_mqa_logits", kernel)
    return kernel


def _make_plugin(
    root: Path, *, exact: bool = False, routes=None, launch_keys=(_SCORE,), reads=_READS
) -> Path:
    kernels = root / "litetopk_kernels"
    kernels.mkdir(parents=True)
    for name in _KERNEL_FILES:
        (kernels / name).write_text(f"// {name} of fake source {root.name}\n", encoding="utf-8")
    (root / "litetopk.py").write_text(_FAKE_ADAPTER, encoding="utf-8")
    if routes is None:
        routes = [{**_FP8_ROUTE, "exact": exact}, _FP4_ROUTE]
    config = {
        "reads": list(reads),
        "routes": routes,
        "launch_keys": list(launch_keys),
        "admit_key": _ADMIT,
        "tie_key": _TIE,
        "score_key": _SCORE,
        "min_keys": [96, 64],
        "vote_rows": VOTE_ROWS,
    }
    (root / "fake_plugin.json").write_text(json.dumps(config), encoding="utf-8")
    return root


class Consumer(nn.Module):
    """A module with an indexer, as the binding sees it."""

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.binding = "unset"

    def indexer_geometry(self):
        return self.geometry

    def set_indexer_topk(self, binding):
        self.binding = binding


def _bind(
    tmp_path,
    fmt="fp8",
    *,
    backend="litetopk",
    precision="fast",
    tuning=None,
    exact_route=False,
    name="plugin",
    geometry=None,
):
    """Configure one consumer and return (binding, fake plugin behind it or None)."""
    geometry = geometry or (FP8_GEOMETRY if fmt == "fp8" else FP4_GEOMETRY)
    fields = {"backend": backend, "precision": precision}
    if backend == "litetopk":
        root = tmp_path / name
        if not root.exists():
            _make_plugin(root, exact=exact_route)
        fields["litetopk"] = LiteTopKPluginConfig(source=str(root))
    if precision == "exact":
        fields["exact_topk"] = ExactTopKConfig(source=str(tmp_path))
    if tuning is None and backend == "litetopk":
        tuning = FP8_TUNING if fmt == "fp8" else FP4_TUNING
    consumer = Consumer(geometry)
    installation = binding_module.configure_indexer_topk(
        [consumer], IndexerTopKConfig(**fields), native_format=fmt, tuning=tuning
    )
    binding = consumer.binding
    assert installation.bindings == (binding,)
    fake = None
    if binding.plugin is not None:
        fake = binding.plugin.module.BEHAVIOR
        if fake is None:
            fake = binding.plugin.module.BEHAVIOR = FakeLiteTopK()
    return binding, fake


def _inputs(rows, keys, seed=0):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((rows, HEADS, HEAD_DIM), generator=generator).to(torch.bfloat16)
    k = torch.randn((keys, HEAD_DIM), generator=generator).to(torch.bfloat16)
    k[keys // 2 :] = k[: keys - keys // 2].clone()  # duplicate keys: exact ties at the cutoff
    weights = torch.randn((rows, HEADS), generator=generator).to(torch.bfloat16)
    return q, k, weights


def _expected(q, k, weights, layout, topk, softmax_scale, fmt, rows=None):
    """Exact top-k of every row over its visible keys, from the quantized operands."""
    data, scales, folded = quantize_queries(
        q, weights, fmt, softmax_scale=softmax_scale, kernel_heads=q.shape[1]
    )
    keys = quantize_keys(k, fmt)
    out = torch.full((layout.rows, topk), -1, dtype=torch.int32)
    for segment in layout.segments:
        for row in range(segment.row_start, segment.row_end):
            if rows is not None and row not in rows:
                continue
            start = segment.key_start
            end = start + layout.visible_keys(segment, row)
            if end == start:
                continue
            scores = _row_scores(data, scales, folded, keys.data, keys.scale, row, start, end)
            order = _best(scores, topk)
            out[row, : order.numel()] = order + segment.index_base
    return sort_topk_rows_(out)


def _select(binding, q, k, weights, layout, scale=0.5, **kwargs):
    with torch.no_grad():
        return binding.select(
            q, k, weights, layout=layout, topk=TOPK, softmax_scale=scale, **kwargs
        )


WAVE_ROWS = 250
# Reference rows [0, 162); tiles of the wave, in three groups.
WAVE_TILES = [(162, 178), (178, 194), (194, 210), (210, 226), (226, 242), (242, 250)]


def _wave(tmp_path, **kwargs):
    binding, fake = _bind(tmp_path, "fp8", **kwargs)
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    return binding, fake, (q, k, weights, layout)


def test_fp8_call_sequence_matches_a_wave(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))

    key = ("rolling", 0)
    assert fake.events == [
        ("begin_call", key, 250),
        # The last 12 reference rows before the first tile vote the first seed.
        ("stash", key, 12, 162, VOTE_ROWS),
        ("plan", key, 250, 16, 163, "stash"),
        ("tile", key, 163, 16, False, None, False, 178),
        ("tile", key, 179, 16, False, None, False, 194),
        ("tile", key, 195, 16, True, None, False, 210),  # publishes the next group's seed
        ("plan", key, 250, 16, 211, "tile"),
        ("tile", key, 211, 16, False, None, False, 226),
        ("tile", key, 227, 16, True, None, False, 242),
        ("plan", key, 250, 8, 243, "tile"),  # the shorter last tile is its own group
        ("tile", key, 243, 8, False, None, False, 250),  # no later group: nothing to publish
        ("drop", key),
    ]
    # The reference selector selected the prefix in one call before any tile ran.
    assert score_kernel.rows == 162
    assert torch.equal(sort_topk_rows_(fake.stashed), out[150:162])

    # Both selectors read the same operands: the keys the plugin gathered are the bytes of the
    # quantized keys, and its query tiles are those the reference selector would score.
    keys = quantize_keys(k, "fp8")
    assert torch.equal(fake.gathered[0], keys.data.view(torch.uint8))
    assert torch.equal(fake.gathered[1], keys.scale.view(torch.uint8).view(WAVE_ROWS, 4))
    for (start, end), (data, scales, folded) in zip(WAVE_TILES, fake.operands):
        expected = quantize_queries(
            q[start:end], weights[start:end], "fp8", softmax_scale=0.5, kernel_heads=HEADS
        )
        assert scales is None and expected[1] is None
        assert torch.equal(data.view(torch.uint8), expected[0].view(torch.uint8))
        assert torch.equal(folded, expected[2])

    stats = binding.stats
    assert (stats.calls, stats.rows, stats.litetopk_rows, stats.reference_rows) == (1, 250, 88, 162)
    assert (stats.tiles, stats.plans, stats.carry_stashes) == (6, 3, 1)
    assert dict(stats.tile_rows) == {16: 5, 8: 1}
    assert (stats.reference_calls, stats.bootstrap_rows, stats.padding_rows) == (1, 0, 0)
    assert stats.reference_score_calls == len(score_kernel.calls)
    assert (stats.recomputed_rows, stats.recomputed_tiles, stats.reseeded_groups) == (0, 0, 0)
    assert not stats.declined_tiles and not stats.status_rows


def test_fp8_bootstrap_tile_and_context_parallel_shard(tmp_path, score_kernel):
    # A shard that starts past the startup position has no reference prefix: its first tile is
    # selected by the reference selector and votes the first seed.
    binding, fake = _bind(tmp_path, "fp8")
    q, k, weights = _inputs(64, 400, seed=1)
    layout = QueryLayout.contiguous(64, position=300, keys=400)
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))
    key = ("rolling", 0)
    assert fake.events[:4] == [
        ("begin_call", key, 400),
        ("stash", key, 12, 316, VOTE_ROWS),
        ("plan", key, 400, 16, 317, "stash"),
        ("tile", key, 317, 16, False, None, False, 332),
    ]
    assert fake.names().count("tile") == 3 and score_kernel.rows == 16
    assert (binding.stats.bootstrap_rows, binding.stats.reference_rows) == (16, 16)


def test_fp4_identity_seed_sequence(tmp_path, score_kernel):
    binding, fake = _bind(tmp_path, "mxfp4")
    # 396 tokens over 96 compressed keys: the rows of the last tile see every key (the cap).
    rows, keys = 396, 96
    q, k, weights = _inputs(rows, keys, seed=2)
    layout = QueryLayout.full(rows, keys=keys, key_ratio=4)
    out = _select(binding, q, k, weights, layout, scale=HEAD_DIM**-0.5)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, HEAD_DIM**-0.5, "mxfp4"))

    key = ("rolling", 0)
    tiles = [(start, start + 16) for start in range(96, 384, 16)] + [(384, 396)]
    expected = [("begin_call", key, keys)]
    for index, (start, end) in enumerate(tiles):
        # One plan per tile; the first starts from the identity seed (no carry), and every
        # tile but the last publishes the seed of the next. The slab holds at least 16384
        # candidates.
        seed = None if index == 0 else "tile"
        publish = index + 1 < len(tiles)
        first_end, last_end = (start + 1) // 4, min(keys, end // 4)
        expected.append(("plan", key, keys, end - start, first_end, seed))
        expected.append(("tile", key, first_end, end - start, publish, 16384, False, last_end))
    expected.append(("drop", key))
    assert fake.events == expected
    assert "stash" not in fake.names() and score_kernel.rows == 96
    # MXFP4 tiles carry their group scales; the weights hold only the softmax scale.
    data, scales, folded = fake.operands[0]
    expected_operands = quantize_queries(
        q[96:112], weights[96:112], "mxfp4", softmax_scale=HEAD_DIM**-0.5, kernel_heads=HEADS
    )
    assert torch.equal(data, expected_operands[0]) and torch.equal(scales, expected_operands[1])
    assert torch.equal(folded, expected_operands[2])
    assert dict(binding.stats.tile_rows) == {16: 18, 12: 1}


def test_slab_capacity_rule(tmp_path, score_kernel):
    rows, keys = 396, 99
    q, k, weights = _inputs(rows, keys, seed=2)
    layout = QueryLayout.full(rows, keys=keys, key_ratio=4)
    caps = {}
    for label, tuning in {
        "clamped": dataclasses.replace(FP4_TUNING, candidate_capacity=60),
        "limit": dataclasses.replace(
            FP4_TUNING, plugin_settings=LiteTopKPluginSettings(merge_cap=70)
        ),
    }.items():
        binding, fake = _bind(tmp_path, "mxfp4", tuning=tuning, name=label)
        fake.events.clear()
        _select(binding, q, k, weights, layout)
        caps[label] = {event[5] for event in fake.events if event[0] == "tile"}
        if label == "clamped":
            # An explicit capacity bounds the slab of every tile.
            assert caps[label] == {60} and binding.stats.tiles == 19
        else:
            # A slab limit below the smallest slab a tile is given: no tile runs.
            assert caps[label] == set() and binding.stats.reference_rows == rows
        _new_process()
    engine = binding_module.LiteTopKEngine
    plugin = type("Plugin", (), {})()
    plugin.module = type("Module", (), {"carry_vote_rows": staticmethod(lambda: 12)})
    for merge_cap, key_count, topk, limit in (
        (None, 65536, 512, 196608),
        (None, 262144, 512, 262144),
        (None, 1 << 20, 512, 262144),
        (300000, 1 << 20, 512, 300000),
        (16383, 65536, 512, 0),
        (40000, 65536, 2048, 0),
    ):
        plugin.settings = LiteTopKPluginSettings(merge_cap=merge_cap)
        route = type("Route", (), {"name": "fp4_slab"})()
        built = engine(plugin, route, None, exact=False, kernel_heads=64, layer_key=object())
        assert built.max_tile_keys(key_count, topk) == limit
        route.name = "fp8_paged"
        assert built.max_tile_keys(key_count, topk) is None


@pytest.mark.parametrize("record_bytes", (6, 8))
def test_slab_capacity_follows_visible_keys_and_budget(tmp_path, score_kernel, record_bytes):
    # A context-parallel shard late in a long sequence: its rows see more keys than the smallest
    # slab (16384), so a tile's slab holds as many candidates as the segment's last tile sees
    # keys, bounded by the byte budget of a tile or by an explicit capacity.
    routes = [
        dict(_FP8_ROUTE),
        {**_FP4_ROUTE, "max_keys": 1 << 20, "candidate_record_bytes": record_bytes},
    ]
    keys, rows, position = 20000, 32, 67440
    q, k, weights = _inputs(rows, keys, seed=6)
    layout = QueryLayout.contiguous(rows, position=position, keys=keys, key_ratio=4)
    (segment,) = layout.segments
    visible = layout.visible_keys(segment, rows - 1)
    assert visible == (position + rows) // 4 == 16868 > 16384
    budget = 16 * record_bytes * 16500  # 16500 candidates of 6 bytes per row of a 16-row tile
    cases = {
        "visible": (FP4_TUNING, visible),
        "budget": (dataclasses.replace(FP4_TUNING, candidate_budget_bytes=budget), 16500),
        "smallest": (
            dataclasses.replace(FP4_TUNING, candidate_budget_bytes=16 * record_bytes * 16384),
            16384,
        ),
        "explicit": (dataclasses.replace(FP4_TUNING, candidate_capacity=16600), 16600),
    }
    for label, (tuning, capacity) in cases.items():
        _make_plugin(tmp_path / label, routes=routes)
        binding, fake = _bind(tmp_path, "mxfp4", tuning=tuning, name=label)
        out = _select(binding, q, k, weights, layout, scale=HEAD_DIM**-0.5)
        if label == "visible":
            expected = _expected(q, k, weights, layout, TOPK, HEAD_DIM**-0.5, "mxfp4")
            assert torch.equal(out, expected)
        tiles = [event for event in fake.events if event[0] == "tile"]
        assert len(tiles) == 2 and {event[5] for event in tiles} == {capacity}, label
        # The slab of a 16-row tile, 6 bytes per candidate.
        assert binding.stats.candidate_slab_bytes == 16 * capacity * record_bytes
        _new_process()


def test_rolling_keys_dropped_in_finally(tmp_path, score_kernel):
    # Two packed sequences: each segment has its own rolling key and seed, the ids of the second
    # are offset by its key start, and no carry survives the call.
    binding, fake = _bind(tmp_path, "fp8")
    cu_seqlens = [0, 200, 430]
    q, k, weights = _inputs(430, 430, seed=3)
    layout = QueryLayout.packed(cu_seqlens, row_start=0, rows=430, absolute_ids=True)
    out = _select(binding, q, k, weights, layout)
    expected = _expected(q, k, weights, layout, TOPK, 0.5, "fp8")
    assert torch.equal(out, expected)
    assert int(out[200:][out[200:] >= 0].min()) >= 200
    first = fake.names().index("drop")
    assert {event[1] for event in fake.events[: first + 1]} == {("rolling", 0)}
    assert {event[1] for event in fake.events[first + 1 :]} == {("rolling", 1)}
    assert fake.events[first + 1 : first + 4] == [
        ("begin_call", ("rolling", 1), 230),
        ("stash", ("rolling", 1), 12, 162, VOTE_ROWS),
        ("plan", ("rolling", 1), 230, 16, 163, "stash"),
    ]
    # The stashed ids are sequence-local although the output ids are key tensor rows.
    assert torch.equal(sort_topk_rows_(fake.stashed + 200), expected[350:362])
    assert fake.names().count("drop") == 2 and not fake.carries
    # The reference rows of both sequences were selected in one batched call.
    assert binding.stats.reference_calls == 1 and score_kernel.rows == 160 + 162

    # An error inside a tile still drops the carry of its segment.
    fake.events.clear()
    fake.raise_in_tile = fake.tiles + 4
    with pytest.raises(RuntimeError, match="fake kernel fault"):
        _select(binding, q, k, weights, layout)
    assert fake.events[-1][0] == "drop" and not fake.carries

    # Two layers never share a carry key.
    other, _ = _bind(tmp_path, "fp8")
    assert other.plugin is binding.plugin and other._layer_key is not binding._layer_key


def test_host_decline_falls_back_and_rebootstraps(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    expected = _expected(q, k, weights, layout, TOPK, 0.5, "fp8")
    key = ("rolling", 0)

    fake.decline_plans = {1}  # the second group gets no plan
    assert torch.equal(_select(binding, q, k, weights, layout), expected)
    assert fake.events[6:] == [
        ("plan", key, 250, 16, 211, "tile"),
        # The declined group went to the reference selector; its last rows re-seed the next.
        ("stash", key, 12, 242, VOTE_ROWS),
        ("plan", key, 250, 8, 243, "stash"),
        ("tile", key, 243, 8, False, None, False, 250),
        ("drop", key),
    ]
    stats = binding.stats
    assert dict(stats.declined_tiles) == {"plan declined": 2} and stats.reseeded_groups == 1
    assert (stats.litetopk_rows, stats.reference_rows, stats.tiles) == (56, 194, 4)
    assert (stats.reference_calls, score_kernel.rows) == (2, 162 + 32)

    # A tile declined inside a group: the rest of the group falls back.
    fake.events.clear()
    fake.decline_plans, fake.decline_tiles = set(), {fake.tiles + 1}
    binding.stats.reset()
    assert torch.equal(_select(binding, q, k, weights, layout), expected)
    assert fake.events[2:7] == [
        ("plan", key, 250, 16, 163, "stash"),
        ("tile", key, 163, 16, False, None, False, 178),
        ("tile", key, 179, 16, False, None, False, 194),  # declined
        ("stash", key, 12, 210, VOTE_ROWS),
        ("plan", key, 250, 16, 211, "stash"),
    ]
    assert dict(binding.stats.declined_tiles) == {"tile declined": 2}
    assert (binding.stats.litetopk_rows, binding.stats.reference_rows) == (56, 194)


def test_refine_overflow_recomputes_rows_only(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(
        tmp_path,
        tuning=IndexerTopKTuning(tile_rows=16, startup_position=160, group_tiles=3, required=True),
    )
    # Rows 165 and 169 of the first tile and row 245 of the last report a candidate overflow.
    fake.row_status = {0: {3: 2, 7: 1}, 5: {3: 2}}
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))
    # Only those three rows were scored again, in one reference call; nothing ran twice and a
    # required binding does not treat row-level codes as a failure.
    assert score_kernel.rows == 162 + 3 and score_kernel.calls[-1]["rows"] == 3
    assert [event[0] for event in fake.events].count("tile") == 6
    stats = binding.stats
    assert (stats.recomputed_rows, stats.recomputed_tiles, stats.reseeded_groups) == (3, 0, 0)
    assert dict(stats.status_rows) == {1: 1, 2: 2} and stats.litetopk_rows == 88


def test_producer_failure_rebootstraps_next_group(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    expected = _expected(q, k, weights, layout, TOPK, 0.5, "fp8")
    key = ("rolling", 0)

    # The last tile of the first group fails. Its carry is invalid, so the second group fails
    # too, and so does the third, seeded by the second.
    fake.row_status = {2: {5: 3}}
    assert torch.equal(_select(binding, q, k, weights, layout), expected)
    first_pass = fake.events.index(("drop", key))
    assert fake.events[first_pass + 1 :] == [
        # The failed tile was recomputed by the reference selector; the next group is seeded
        # from the repaired rows and runs again.
        ("begin_call", key, 250),
        ("stash", key, 12, 210, VOTE_ROWS),
        ("plan", key, 250, 16, 211, "stash"),
        ("tile", key, 211, 16, False, None, False, 226),
        ("tile", key, 227, 16, False, None, False, 242),
        ("drop", key),
        ("begin_call", key, 250),
        ("stash", key, 12, 242, VOTE_ROWS),
        ("plan", key, 250, 8, 243, "stash"),
        ("tile", key, 243, 8, False, None, False, 250),
        ("drop", key),
    ]
    stats = binding.stats
    assert (stats.recomputed_tiles, stats.recomputed_rows, stats.reseeded_groups) == (1, 16, 2)
    assert dict(stats.status_rows) == {3: 1 + 32 + 8}
    assert score_kernel.rows == 162 + 16 and stats.tiles == 6

    # A failed tile whose group does not seed a failing group: only that tile is recomputed.
    fake.events.clear()
    score_kernel.calls.clear()
    binding.stats.reset()
    fake.row_status = {fake.tiles + 3: {0: 3, 9: 2}}
    assert torch.equal(_select(binding, q, k, weights, layout), expected)
    assert fake.names().count("begin_call") == 1 and fake.names().count("tile") == 6
    assert (binding.stats.recomputed_tiles, binding.stats.reseeded_groups) == (1, 0)
    assert score_kernel.rows == 162 + 16

    # Failures in two groups that are unrelated (the first failed tile does not publish the
    # seed of the next group): both tiles are recomputed, no group runs again.
    fake.events.clear()
    score_kernel.calls.clear()
    binding.stats.reset()
    fake.row_status = {fake.tiles: {4: 3}, fake.tiles + 3: {0: 3}}
    assert torch.equal(_select(binding, q, k, weights, layout), expected)
    assert fake.names().count("begin_call") == 1 and fake.names().count("tile") == 6
    assert (binding.stats.recomputed_tiles, binding.stats.reseeded_groups) == (2, 0)
    assert score_kernel.rows == 162 + 32


def test_group_reruns_are_capped(tmp_path, score_kernel):
    # One tile per group, so a failed tile poisons the seed of every later group.
    tuning = IndexerTopKTuning(tile_rows=16, startup_position=160, group_tiles=1)
    binding, fake, (q, k, weights, layout) = _wave(tmp_path, tuning=tuning)
    expected = _expected(q, k, weights, layout, TOPK, 0.5, "fp8")
    stats = binding.stats

    def run(row_status):
        fake.events.clear()
        score_kernel.calls.clear()
        stats.reset()
        fake.row_status = {fake.tiles + index: {0: 3} for index in row_status}
        assert torch.equal(_select(binding, q, k, weights, layout), expected)
        return fake.names().count("tile")

    # The first group fails: the next two groups run again from seeds voted from repaired rows
    # (and succeed); the three after them go straight to the reference selector.
    assert run([0]) == 6 + 2
    assert (stats.rerun_groups, stats.reseeded_groups) == (2, 2)
    assert (stats.recomputed_tiles, stats.recomputed_rows) == (4, 16 + 16 + 16 + 8)
    assert score_kernel.rows == 162 + 56

    # The second run of the second group fails as well: its seed was valid, so seeds do not
    # explain the failures and no later group runs again.
    assert run([0, 6]) == 6 + 1
    assert (stats.rerun_groups, stats.recomputed_tiles, stats.recomputed_rows) == (1, 6, 88)

    # Every tile fails (an exhausted candidate pool, say): one second run, then every failed
    # tile goes to the reference selector.
    assert run(range(7)) == 6 + 1
    assert (stats.rerun_groups, stats.recomputed_tiles, stats.recomputed_rows) == (1, 6, 88)
    assert score_kernel.rows == 162 + 88


def test_status_tail_and_device_assert_options(tmp_path, score_kernel):
    tail = IndexerTopKTuning(
        tile_rows=16, startup_position=160, group_tiles=3, status_check="sync_recompute_tail"
    )
    binding, fake, (q, k, weights, layout) = _wave(tmp_path, tuning=tail)
    expected = _expected(q, k, weights, layout, TOPK, 0.5, "fp8")
    fake.row_status = {1: {0: 3}}
    assert torch.equal(_select(binding, q, k, weights, layout), expected)
    # Every tile from the failed one to the end of the segment is recomputed; nothing reruns.
    assert fake.names().count("begin_call") == 1
    assert (binding.stats.recomputed_tiles, binding.stats.recomputed_rows) == (5, 72)
    assert score_kernel.rows == 162 + 72


def test_device_assert_reads_no_status(tmp_path, score_kernel, monkeypatch):
    tuning = IndexerTopKTuning(
        tile_rows=16, startup_position=160, group_tiles=3, status_check="device_assert"
    )
    binding, fake, (q, k, weights, layout) = _wave(tmp_path, tuning=tuning)
    reads = []
    monkeypatch.setattr(
        binding_module.IndexerTopKBinding,
        "_worst_status",
        lambda self, status: reads.append(1) or 0,
    )
    _select(binding, q, k, weights, layout)
    assert not reads
    fake.row_status = {fake.tiles: {0: 2}}
    with pytest.raises(RuntimeError, match="candidate overflow or a selection failure"):
        _select(binding, q, k, weights, layout)


def test_single_status_read_per_call(tmp_path, score_kernel, monkeypatch):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    reads, original = [], binding_module.IndexerTopKBinding._worst_status

    def counted(self, status):
        reads.append(status.shape[0])
        return original(self, status)

    monkeypatch.setattr(binding_module.IndexerTopKBinding, "_worst_status", counted)
    monkeypatch.setattr(
        binding_module.IndexerTopKBinding,
        "_repair",
        lambda *args: pytest.fail("no repair without a fault"),
    )
    _select(binding, q, k, weights, layout)
    assert reads == [250]


def test_required_raises(tmp_path, score_kernel):
    required = IndexerTopKTuning(tile_rows=16, startup_position=160, group_tiles=3, required=True)
    binding, fake, (q, k, weights, layout) = _wave(tmp_path, tuning=required)
    fake.decline_plans = {1}
    with pytest.raises(IndexerTopKRuntimeError, match="required LiteTopK fell back .* 32 planned"):
        _select(binding, q, k, weights, layout)
    assert fake.events[-1][0] == "drop"
    fake.decline_plans = set()
    fake.row_status = {fake.tiles + 1: {2: 3}}
    with pytest.raises(IndexerTopKRuntimeError, match="16 planned rows .* reported a failure"):
        _select(binding, q, k, weights, layout)
    with pytest.raises(IndexerTopKRuntimeError, match="declined a call: batch>1"):
        binding.decline("batch>1")
    assert dict(binding.stats.declined_calls) == {"batch>1": 1}

    lenient, _ = _bind(tmp_path, "fp8")
    assert lenient.decline("batch>1") is None
    assert dict(lenient.stats.declined_calls) == {"batch>1": 1}


def test_exact_requires_exact_route_and_exact_topk(tmp_path, score_kernel):
    with pytest.raises(IndexerTopKConfigError, match="exact_topk.source is required"):
        IndexerTopKConfig(backend="reference", precision="exact")
    with pytest.raises(IndexerTopKConfigError, match="litetopk.source is required"):
        IndexerTopKConfig(backend="litetopk", precision="fast")
    assert IndexerTopKConfig().backend == "default" and IndexerTopKConfig().precision == "exact"

    # A route that does not advertise exact selection cannot serve precision="exact".
    with pytest.raises(IndexerTopKConfigError, match="advertises exact selection"):
        _bind(tmp_path, "fp8", precision="exact", name="fast-only")
    _new_process()

    binding, fake = _bind(tmp_path, "fp8", precision="exact", exact_route=True, name="exact")
    # Exact selection renders the policies it needs and asks every tile for an exact result.
    assert binding.plugin.settings.tie_policy == "logical-id"
    assert binding.plugin.settings.score_policy == "native-fp32"
    assert os.environ[_SCORE] == "native-fp32" and os.environ[_TIE] == "logical-id"
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))
    assert {event[6] for event in fake.events if event[0] == "tile"} == {True}


def test_launch_time_env_rechecked_at_select(tmp_path, score_kernel, monkeypatch):
    binding, fake = _bind(tmp_path, "fp8", precision="exact", exact_route=True)
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    _select(binding, q, k, weights, layout)
    # The CUDA extension reads this key on every launch: a change after the load would switch
    # the score kernel silently, so every selection checks it first.
    monkeypatch.setenv(_SCORE, "folded")
    fake.events.clear()
    with pytest.raises(IndexerTopKRuntimeError, match=f"{_SCORE} changed from 'native-fp32'"):
        _select(binding, q, k, weights, layout)
    assert not fake.events


def test_plugin_specific_launch_time_keys(tmp_path, score_kernel, monkeypatch):
    # A plugin may read launch-time keys of its own (a staging layout, say), with or without the
    # usual prefix. Every key it lists is recorded at load and rechecked at every selection.
    staging, knob = "SGLANG_LITETOPK_FAKE_STAGING", "FAKE_LITETOPK_LAUNCH_KNOB"
    assert staging not in env.LAUNCH_TIME_ENV_KEYS and knob not in env.LAUNCH_TIME_ENV_KEYS
    _make_plugin(tmp_path / "plugin", launch_keys=(_SCORE, staging, knob))
    binding, fake = _bind(tmp_path, "fp8")
    assert binding.plugin.launch_time_env == {_SCORE: None, knob: None, staging: None}
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    _select(binding, q, k, weights, layout)
    for key in (staging, knob):
        monkeypatch.setenv(key, "changed")
        with pytest.raises(IndexerTopKRuntimeError, match=f"{key} changed from None to 'changed'"):
            _select(binding, q, k, weights, layout)
        # Once the plugin is loaded its keys are known plugin keys: another plugin load refuses
        # them as stray settings.
        with pytest.raises(Exception, match=f"the process sets {key}"):
            _bind(tmp_path, "mxfp4", name=f"other-{key}")
        monkeypatch.delenv(key)
    _select(binding, q, k, weights, layout)

    # A plugin's own launch-time key that is set before its load is refused.
    _new_process()
    _make_plugin(tmp_path / "preset", launch_keys=(_SCORE, knob))
    monkeypatch.setenv(knob, "u40x14")
    with pytest.raises(Exception, match=f"{knob}='u40x14' is set in the process"):
        _bind(tmp_path, "fp8", name="preset")


def test_grad_enabled_inactive(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    with torch.enable_grad():
        assert not binding.active()
        assert binding.select(q, k, weights, layout=layout, topk=TOPK, softmax_scale=0.5) is None
    assert not fake.events and not score_kernel.calls and binding.stats.calls == 0
    with torch.no_grad():
        assert binding.active()


def test_capture_raises(tmp_path, score_kernel, monkeypatch):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    monkeypatch.setattr(binding_module, "_is_capturing", lambda device: True)
    with pytest.raises(IndexerTopKRuntimeError, match="cannot run under CUDA graph capture"):
        _select(binding, q, k, weights, layout)
    assert not fake.events and binding.stats.calls == 0


def test_output_sorted_invalid_last(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    out = _select(binding, q, k, weights, layout)
    valid = out >= 0
    assert torch.equal(valid.sum(1), torch.arange(1, 251).clamp(max=TOPK))
    assert (valid[:, :-1] | ~valid[:, 1:]).all()  # -1 only after the valid ids
    assert ((out[:, 1:] > out[:, :-1]) | ~valid[:, 1:]).all()  # ascending, no duplicates

    # index_order="selector" keeps each selector's slot order (the fake writes descending
    # scores, so its rows are not ascending); the sets are the same.
    unsorted = IndexerTopKTuning(
        tile_rows=16, startup_position=160, group_tiles=3, index_order="selector"
    )
    raw_binding, _ = _bind(tmp_path, "fp8", tuning=unsorted)
    raw = _select(raw_binding, q, k, weights, layout)
    assert not torch.equal(raw, out) and torch.equal(sort_topk_rows_(raw.clone()), out)


def test_uncovered_rows_minus_one(tmp_path, score_kernel):
    binding, fake = _bind(tmp_path, "fp8")
    q, k, weights = _inputs(300, 250, seed=4)
    # Local rows [0, 300) of a pack that ends at token 250 + 30: 20 padding rows at the end,
    # and the shard starts 30 tokens into the first sequence.
    layout = QueryLayout.packed([0, 280], row_start=30, rows=300, absolute_ids=False)
    assert layout.segments[0].row_end == 250
    q, k, weights = _inputs(300, 280, seed=4)
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))
    assert (out[250:] == -1).all() and binding.stats.padding_rows == 50
    stats = binding.stats
    assert stats.litetopk_rows + stats.reference_rows + stats.padding_rows == stats.rows == 300

    # Padding before and between segments as well.
    gaps = QueryLayout(
        rows=300,
        key_ratio=1,
        segments=(QuerySegment(6, 40, 0, 0, 34, 0), QuerySegment(44, 294, 0, 34, 246, 34)),
    )
    binding.stats.reset()
    out = _select(binding, q, k, weights, gaps)
    assert torch.equal(out, _expected(q, k, weights, gaps, TOPK, 0.5, "fp8"))
    assert (out[:6] == -1).all() and (out[40:44] == -1).all() and (out[294:] == -1).all()
    assert binding.stats.padding_rows == 16 and binding.stats.litetopk_rows > 0

    empty = QueryLayout.packed([0], row_start=0, rows=4, absolute_ids=False)
    out = _select(binding, q[:4], k, weights[:4], empty)
    assert (out == -1).all() and out.shape == (4, TOPK)


def test_reference_backend_needs_no_plugin(tmp_path, score_kernel):
    binding, fake = _bind(tmp_path, "fp8", backend="reference")
    assert fake is None and binding.plugin is None and not loader._LOADED and not _plugin_env()
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))
    assert score_kernel.rows == WAVE_ROWS
    stats = binding.stats
    assert (stats.reference_rows, stats.litetopk_rows, stats.tiles) == (WAVE_ROWS, 0, 0)
    # The reference backend runs on any device the score kernel supports.
    assert binding.resolved_tuning(torch.device("cpu")).reference_budget_bytes > 0

    # A smaller top-k than the geometry's (a short sequence) is selected by the reference.
    short = QueryLayout.full(6, keys=6)
    with torch.no_grad():
        out = binding.select(q[:6], k[:6], weights[:6], layout=short, topk=6, softmax_scale=0.5)
    assert torch.equal(out, _expected(q[:6], k[:6], weights[:6], short, 6, 0.5, "fp8"))


def test_stats(tmp_path, score_kernel):
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    _select(binding, q, k, weights, layout)
    _select(binding, q, k, weights, layout)
    stats = binding.stats
    assert (stats.calls, stats.rows, stats.tiles, stats.plans) == (2, 500, 12, 6)
    assert stats.litetopk_rows + stats.reference_rows + stats.padding_rows == stats.rows
    record = stats.as_dict()
    assert record["tile_rows"] == {"16": 10, "8": 2} and record["declined_tiles"] == {}
    assert (stats.litetopk_segments, record["reference_segments"]) == (2, {})
    assert json.loads(json.dumps(record)) == record
    stats.reset()
    assert stats == IndexerTopKStats()

    # Segments whose plan gives every row to the reference selector are counted by reason.
    q, k, weights = _inputs(430, 430, seed=3)
    packed = QueryLayout.packed([0, 50, 60, 430], row_start=0, rows=430, absolute_ids=True)
    _select(binding, q, k, weights, packed)
    assert stats.litetopk_segments == 1 and stats.reference_segments == {
        "fewer keys than the route minimum": 2
    }
    assert stats.candidate_slab_bytes == 0  # the paged route has no slab


def test_no_collectives_runtime(tmp_path, score_kernel, monkeypatch):
    import torch.distributed as dist

    def forbidden(*args, **kwargs):
        raise AssertionError("indexer top-k selection must not issue a collective")

    for name in ("all_reduce", "all_gather", "all_gather_into_tensor", "broadcast", "barrier"):
        monkeypatch.setattr(dist, name, forbidden)
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    fake.decline_plans = {1}
    fake.row_status = {0: {1: 2}, 3: {0: 3}}
    out = _select(binding, q, k, weights, layout)
    assert torch.equal(out, _expected(q, k, weights, layout, TOPK, 0.5, "fp8"))


def test_configure_default_unbinds_and_returns_none(tmp_path):
    consumers = [Consumer(FP8_GEOMETRY), Consumer(FP8_GEOMETRY)]
    model = nn.Sequential(*consumers)
    for config in (None, IndexerTopKConfig(), {"backend": "default"}):
        for consumer in consumers:
            consumer.binding = "stale"
        assert binding_module.configure_indexer_topk([model], config, native_format="fp8") is None
        assert [consumer.binding for consumer in consumers] == [None, None]
    assert not loader._LOADED and not _plugin_env()


def test_configure_skips_modules_without_geometry(tmp_path):
    model = nn.ModuleDict(
        {
            "full": Consumer(FP8_GEOMETRY),
            "shared": Consumer(None),  # reuses another layer's top-k
            "plain": nn.Linear(2, 2),
            "nested": nn.Sequential(Consumer(FP8_GEOMETRY)),
        }
    )
    installation = binding_module.configure_indexer_topk(
        [model], {"backend": "reference", "precision": "fast"}, native_format="fp8"
    )
    assert isinstance(installation, binding_module.IndexerTopKInstallation)
    assert [binding.name for binding in installation.bindings] == ["full", "nested.0"]
    assert model["shared"].binding is None
    assert model["full"].binding is installation.bindings[0]
    assert model["full"].binding is not model["nested"][0].binding
    assert installation.plugin_info is None and set(installation.stats()) == {"full", "nested.0"}
    # Several chunks: the names say which chunk a module is in.
    second = nn.Sequential(Consumer(FP8_GEOMETRY))
    installation = binding_module.configure_indexer_topk(
        [model, second], {"backend": "reference", "precision": "fast"}, native_format="fp8"
    )
    assert [binding.name for binding in installation.bindings] == [
        "chunk0.full",
        "chunk0.nested.0",
        "chunk1.0",
    ]
    with pytest.raises(IndexerTopKConfigError, match="native_format"):
        binding_module.configure_indexer_topk(
            [model], {"backend": "reference", "precision": "fast"}, native_format="bf16"
        )


def test_reconfigure_rebinds(tmp_path, score_kernel):
    from megatron.lite.primitive.kernels.indexer_topk import engine as engine_module

    root = _make_plugin(tmp_path / "plugin")
    consumer = Consumer(FP8_GEOMETRY)
    litetopk = {"backend": "litetopk", "precision": "fast", "litetopk": {"source": str(root)}}
    first = binding_module.configure_indexer_topk(
        [consumer], litetopk, native_format="fp8", tuning=FP8_TUNING
    )
    lite_binding = consumer.binding
    assert first.plugin_info["source_id"] == lite_binding.plugin.source_id
    assert first.plugin_info["plugin_info"]["routes"][0]["admitted_max_query_len"] == 16

    # Switching arms rebinds; a binding that was replaced is simply no longer installed.
    reference = {**litetopk, "backend": "reference"}
    second = binding_module.configure_indexer_topk(
        [consumer], reference, native_format="fp8", tuning=FP8_TUNING
    )
    assert consumer.binding is second.bindings[0] and consumer.binding.plugin is None
    third = binding_module.configure_indexer_topk(
        [consumer], litetopk, native_format="fp8", tuning=FP8_TUNING
    )
    assert consumer.binding is third.bindings[0] is not lite_binding
    assert consumer.binding.plugin is lite_binding.plugin  # the plugin load is cached

    # A failing configuration leaves the installed bindings alone.
    with pytest.raises(IndexerTopKConfigError, match="unknown keys"):
        binding_module.configure_indexer_topk(
            [consumer], {"backend": "reference", "mode": 1}, native_format="fp8"
        )
    other = IndexerTopKTuning(tile_rows=32, startup_position=160)
    with pytest.raises(Exception, match="already loaded in this process with different settings"):
        binding_module.configure_indexer_topk(
            [consumer], litetopk, native_format="fp8", tuning=other
        )
    assert consumer.binding is third.bindings[0]

    third.bindings[0].stats.calls = 3
    third.reset_stats()
    assert third.stats()[third.bindings[0].name].calls == 0
    # release() frees the pooled key caches and asks the loaded plugin to free its scratch.
    consumer.binding.plugin.module.BEHAVIOR = fake = FakeLiteTopK()
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    _select(consumer.binding, q, k, weights, QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS))
    assert engine_module._KEY_CACHES.allocated_bytes() > 0
    fake.events.clear()
    third.release(torch.device("cuda", 0))
    assert fake.events == [("release", "cuda:0", True)]
    third.release(torch.device("cpu"))
    assert engine_module._KEY_CACHES.allocated_bytes() == 0 and len(fake.events) == 1


def test_binding_deepcopy_shares_binding(tmp_path):
    binding, _ = _bind(tmp_path, "fp8")
    holder = Consumer(FP8_GEOMETRY)
    holder.binding = binding
    clone = copy.deepcopy(holder)
    assert clone.binding is binding and clone.binding.plugin is binding.plugin
    assert not list(holder.state_dict()) and not isinstance(binding, nn.Module)


def test_negotiation_errors(tmp_path, score_kernel, monkeypatch):
    fp4_only = _make_plugin(tmp_path / "fp4-only", routes=[_FP4_ROUTE])
    config = IndexerTopKConfig(
        backend="litetopk", precision="fast", litetopk=LiteTopKPluginConfig(source=str(fp4_only))
    )
    with pytest.raises(IndexerTopKConfigError, match="has no fp8_paged route"):
        binding_module.configure_indexer_topk([Consumer(FP8_GEOMETRY)], config, native_format="fp8")
    _new_process()  # the load above rendered FP8 settings for this source
    wide = IndexerGeometry(num_heads=16, head_dim=HEAD_DIM, topk=TOPK, key_ratio=4)
    consumers = [Consumer(FP4_GEOMETRY), Consumer(wide)]
    with pytest.raises(IndexerTopKConfigError, match=r"supports indexer heads \[8\].*H=16"):
        binding_module.configure_indexer_topk(
            [nn.Sequential(*consumers)], config, native_format="mxfp4", tuning=FP4_TUNING
        )
    # Nothing is bound when any layer fails: not even the layers before it.
    assert [consumer.binding for consumer in consumers] == ["unset", "unset"]

    # A top-k the route does not select goes to the reference selector, unless required.
    _new_process()
    small = IndexerGeometry(num_heads=HEADS, head_dim=HEAD_DIM, topk=4, key_ratio=1)
    required = IndexerTopKTuning(tile_rows=16, startup_position=160, required=True)
    with pytest.raises(IndexerTopKConfigError, match="does not select top-k 4"):
        _bind(tmp_path, "fp8", geometry=small, tuning=required)
    binding, fake = _bind(tmp_path, "fp8", geometry=small)
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    with torch.no_grad():
        out = binding.select(q, k, weights, layout=layout, topk=4, softmax_scale=0.5)
    assert torch.equal(out, _expected(q, k, weights, layout, 4, 0.5, "fp8"))
    assert not fake.events and binding.stats.reference_rows == WAVE_ROWS

    # Operand and device checks of a selection.
    binding, fake = _bind(tmp_path, "fp8")
    with pytest.raises(ValueError, match="expected q"):
        _select(binding, q[:, :4], k, weights[:, :4], layout)
    with pytest.raises(ValueError, match="key ratio"):
        _select(binding, q, k, weights, QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS, key_ratio=4))
    with pytest.raises(ValueError, match="topk must be"):
        with torch.no_grad():
            binding.select(q, k, weights, layout=layout, topk=TOPK + 1, softmax_scale=0.5)
    with pytest.raises(ValueError, match="need 400 keys"):
        _select(binding, q, k, weights, QueryLayout.full(WAVE_ROWS, keys=400))
    monkeypatch.setattr(binding_module, "_compute_capability", lambda device: (9, 0))
    fresh, _ = _bind(tmp_path, "fp8")
    with pytest.raises(IndexerTopKRuntimeError, match="requires an SM100 .* capability 9.0"):
        _select(fresh, q, k, weights, layout)
    # A device with another SM count needs other FP8 tiles, so other plugin settings than the
    # loaded ones (the admitted tile length is an import-time setting of the plugin).
    monkeypatch.setattr(binding_module, "_compute_capability", lambda device: (10, 0))
    _new_process()
    default_tiles, _ = _bind(tmp_path, "fp8", tuning=IndexerTopKTuning(), name="default-tiles")
    assert default_tiles.plugin.settings.fp8_paged_admit_max_query_len == 3 * 4 * (128 // HEADS)
    monkeypatch.setattr(binding_module, "_num_sms", lambda device: 8)
    with pytest.raises(IndexerTopKRuntimeError, match="plugin settings are fixed for the process"):
        _select(default_tiles, q, k, weights, layout)


def test_head_count_probed_at_configure(tmp_path, score_kernel, monkeypatch):
    probes = []

    def probe(heads, *, fmt, head_dim, device):
        probes.append((heads, fmt, head_dim, str(device)))
        if heads == 12:
            raise IndexerTopKConfigError(
                f"DeepGEMM fp8_fp4_mqa_logits supports no head count from {heads} to 128"
            )
        return heads

    monkeypatch.setattr(binding_module, "score_kernel_heads", probe)
    odd = IndexerGeometry(num_heads=12, head_dim=HEAD_DIM, topk=TOPK, key_ratio=1)
    consumers = [Consumer(FP8_GEOMETRY), Consumer(odd)]
    reference = {"backend": "reference", "precision": "fast"}
    # A head count the reference score kernel does not support fails when the model is built,
    # and no module is bound.
    with pytest.raises(IndexerTopKConfigError, match="supports no head count from 12"):
        binding_module.configure_indexer_topk(
            [nn.Sequential(*consumers)], reference, native_format="fp8"
        )
    assert probes == [(8, "fp8", HEAD_DIM, "cpu"), (12, "fp8", HEAD_DIM, "cpu")]
    assert [consumer.binding for consumer in consumers] == ["unset", "unset"]
    probes.clear()
    # The probe of a supported head count happens at configure, before any selection.
    binding, _ = _bind(tmp_path, "fp8")
    assert probes == [(8, "fp8", HEAD_DIM, "cpu")]


def test_select_runs_on_the_operands_device(tmp_path, score_kernel, monkeypatch):
    scopes, original = [], binding_module._device_scope
    monkeypatch.setattr(
        binding_module,
        "_device_scope",
        lambda device: scopes.append(device) or contextlib.nullcontext(),
    )
    binding, fake, (q, k, weights, layout) = _wave(tmp_path)
    _select(binding, q, k, weights, layout)
    assert scopes == [q.device]
    # A CUDA device becomes the current device for the selection (the plugins allocate their
    # scratch on the current device); nothing to do on the CPU.
    scope = original(torch.device("cuda", 1))
    assert isinstance(scope, torch.cuda.device) and scope.idx == 1
    assert isinstance(original(torch.device("cpu")), contextlib.nullcontext)


def test_raw32_staging_setting(tmp_path, score_kernel, monkeypatch):
    staging = "SGLANG_LITETOPK_RAW32_STAGING"
    reads = (*_READS, staging)
    q, k, weights = _inputs(WAVE_ROWS, WAVE_ROWS)
    layout = QueryLayout.full(WAVE_ROWS, keys=WAVE_ROWS)
    # Unset by default: not rendered, the plugin keeps its compiled layout.
    _make_plugin(tmp_path / "default", launch_keys=(_SCORE, staging), reads=reads)
    binding, _ = _bind(tmp_path, "fp8", name="default")
    assert binding.plugin.settings.raw32_staging is None and staging not in os.environ
    assert binding.plugin.launch_time_env[staging] is None
    _new_process()

    # A plugin that lists the key among its launch-time keys: rendered, recorded and rechecked.
    tuning = dataclasses.replace(
        FP8_TUNING, plugin_settings=LiteTopKPluginSettings(raw32_staging="u40x14")
    )
    _make_plugin(tmp_path / "raw32", launch_keys=(_SCORE, staging), reads=reads)
    binding, _ = _bind(tmp_path, "fp8", tuning=tuning, name="raw32")
    assert os.environ[staging] == "u40x14" and binding.plugin.rendered_env[staging] == "u40x14"
    assert binding.plugin.launch_time_env[staging] == "u40x14"
    _select(binding, q, k, weights, layout)
    monkeypatch.setenv(staging, "u40x18k3")
    with pytest.raises(IndexerTopKRuntimeError, match=f"{staging} changed from 'u40x14'"):
        _select(binding, q, k, weights, layout)
    monkeypatch.setenv(staging, "u40x14")
    _new_process()

    # A plugin that does not read it at launch refuses the setting, and the load leaves no key.
    _make_plugin(tmp_path / "other", launch_keys=(_SCORE,), reads=reads)
    with pytest.raises(IndexerTopKPluginError, match="does not list it among its launch-time"):
        _bind(tmp_path, "fp8", tuning=tuning, name="other")
    assert staging not in os.environ and not loader._LOADED
    for value in ("U40x14", "u40 x14", "", 14):
        with pytest.raises(IndexerTopKConfigError, match="raw32_staging"):
            LiteTopKPluginSettings(raw32_staging=value)


def test_normalize_indexer_topk_config(tmp_path):
    assert normalize_indexer_topk_config(None) is None
    config = IndexerTopKConfig(backend="reference", precision="fast")
    assert normalize_indexer_topk_config(config) is config
    normalized = normalize_indexer_topk_config(
        {
            "backend": "litetopk",
            "precision": "exact",
            "litetopk": {"source": "/plugins/x", "expected_source_id": "0123456789ab"},
            "exact_topk": {"source": "/deps/exact-tie"},
        }
    )
    assert normalized == IndexerTopKConfig(
        backend="litetopk",
        precision="exact",
        litetopk=LiteTopKPluginConfig(source="/plugins/x", expected_source_id="0123456789ab"),
        exact_topk=ExactTopKConfig(source="/deps/exact-tie"),
    )
    for value, match in (
        ({"backend": "fast"}, "backend must be one of 'default', 'reference', 'litetopk'"),
        ({"precision": "auto"}, "precision must be one of"),
        ({"backend": "reference", "required": True}, "unknown keys \\['required'\\]; valid keys"),
        ({"backend": "reference", "reference": "upstream"}, "unknown keys \\['reference'\\]"),
        ({"litetopk": {"path": "/x"}}, "indexer_topk.litetopk has unknown keys \\['path'\\]"),
        ({"litetopk": {"expected_source_id": "0123456789ab"}}, "litetopk is incomplete"),
        ({"litetopk": "/plugins/x"}, "must be LiteTopKPluginConfig, a mapping or None"),
        ({"exact_topk": {"source": ""}}, "exact_topk.source must be a non-empty path"),
        ("litetopk", "must be IndexerTopKConfig, a mapping or None"),
    ):
        with pytest.raises(IndexerTopKConfigError, match=match):
            normalize_indexer_topk_config(value)
    with pytest.raises(IndexerTopKConfigError, match="litetopk must be LiteTopKPluginConfig"):
        IndexerTopKConfig(backend="litetopk", litetopk={"source": "/x"})
