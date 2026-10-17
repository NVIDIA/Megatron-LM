# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU contracts of the indexer head-count negotiation and of zero-head padding.

* The capability matrix: indexer heads 4 to 128 against LiteTopK routes with kernels for 32 and
  64 heads and a reference score kernel that accepts 16, 32 and 64 heads (DeepGEMM 0.1.3), for
  both operand formats, with and without ``head_padding``: the exact kernel heads of every
  selector and the default plan, or the exact error text.
* The float32 head reduction of the score kernels (four FMA chains), simulated with exact
  rational arithmetic and one round-to-nearest-even per operation: appending zero heads leaves
  every chain, the head sum and the score bit for bit unchanged.
* A pure-Python ABI v1 plugin built in tmp_path and loaded through the real loader receives the
  padded operands (zero query codes and weights, MXFP4 group scales 127) and selects with them;
  the reference selector scores the same operands, and every selection equals the unpadded one.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import torch
from torch import nn

from megatron.lite.primitive.kernels.indexer_topk import (
    ExactTopKConfig,
    IndexerGeometry,
    IndexerHeads,
    IndexerTopKConfig,
    IndexerTopKConfigError,
    IndexerTopKTuning,
    LiteTopKPluginConfig,
    QueryLayout,
    normalize_indexer_topk_config,
)
from megatron.lite.primitive.kernels.indexer_topk import reference as reference_module
from megatron.lite.primitive.kernels.indexer_topk import sort_topk_rows_
from megatron.lite.primitive.kernels.indexer_topk.heads import negotiate_indexer_heads
from megatron.lite.primitive.kernels.indexer_topk.plugins import cache, env, loader
from megatron.lite.primitive.kernels.indexer_topk.plugins.abi import RouteCapability
from megatron.lite.primitive.kernels.indexer_topk.reference import quantize_keys, quantize_queries

pytestmark = pytest.mark.mlite

# megatron.lite.primitive.modules.attention.indexer_topk, imported by the fixture below after
# the Transformer Engine stub that the attention package needs on machines without it.
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
HEAD_DIM = 128
# The reference score kernel of DeepGEMM 0.1.3 (fp8_fp4_mqa_logits) accepts these head counts.
DEEPGEMM_HEADS = frozenset({16, 32, 64})
SMS = 4

# Production-like routes: kernels for 32 and 64 heads (the raw32h64 FP8 paged route, which
# selects exactly, and the slab route of the DeepSeek-V4 plugin).
_FP8_ROUTE = {
    "name": "fp8_paged",
    "fmt": "fp8",
    "heads": [32, 64],
    "head_dims": [HEAD_DIM],
    "topk": [2048],
    "max_topk": 2048,
    "qualified_query_lengths": [1016, 1024, 2040, 2048],
    "admitted_max_query_len": 0,  # replaced by the admission the adapter was imported with
    "min_keys": 196608,
    "max_keys": 1048576,
    "hot_prefix": 12288,
    "exact": True,
    "tie_policies": ["logical-id", "logical-id-desc", "storage"],
    "score_policies": ["folded", "native-fp32"],
}
_FP4_ROUTE = {
    **_FP8_ROUTE,
    "name": "fp4_slab",
    "fmt": "mxfp4",
    "topk": None,
    "qualified_query_lengths": [4032, 4096],
    "min_keys": 65536,
    "exact": False,
    "tie_policies": ["storage"],
    "score_policies": ["folded"],
}

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
        "launch_time_env_keys": [_CONFIG["score_key"]],
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
        q, k, k_scale, weights, ks, ke, out_idx, topk, q_sf=q_sf, status_out=status_out,
    )


def stash_carry(hot_key, idx, S, min_index=0, *, recent_rows_hint=None):
    return None


def drop_carry(device, hot_key):
    return None


def release(device, *, release_scratch=True):
    return None
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
    """float32 scores of one quantized query row (any head count) against the keys [start, end)."""
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
    """CPU stand-in of DeepGEMM fp8_fp4_mqa_logits for the head counts it is built for."""

    def __init__(self, supported):
        self.supported = frozenset(supported)
        self.calls = []

    def __call__(self, q, kv, weights, ks, ke, *, max_seqlen_k):
        heads = q[0].shape[1]
        if heads not in self.supported:
            raise RuntimeError(
                "Assertion error (attention.hpp:120): num_heads == "
                + " or num_heads == ".join(str(value) for value in sorted(self.supported))
            )
        rows = q[0].shape[0]
        self.calls.append({"q": q, "weights": weights, "heads": heads, "rows": rows})
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


def exact_topk_kernel(scores, lengths, top_k):
    ids = torch.full((scores.shape[0], top_k), -1, dtype=torch.int32)
    for row in range(scores.shape[0]):
        order = _best(scores[row, : int(lengths[row])], top_k)
        ids[row, : order.numel()] = order
    return ids


class FakeLiteTopK:
    """The test double behind the fake adapter: exact tile selection from the operands it gets."""

    def __init__(self):
        self.operands = []

    def begin_call(self, device, hot_key, sequence_length):
        pass

    def prepare_permuted_gather(self, kv_cache, dst_k, dst_scale, block_table, **kwargs):
        value_bytes = dst_k.shape[1]
        values, scales = cache.block_major_cache_views(kv_cache, value_bytes)
        blocks = block_table[0].long()
        length = kwargs["sequence_length"]
        dst_k.view(torch.uint8).copy_(values[blocks].reshape(-1, value_bytes)[:length])
        dst_scale.copy_(scales[blocks].reshape(-1, 4)[:length])
        return {}

    def try_large_exact_once_chunk(
        self, q, k, k_scale, weights, ks, ke, out_idx, topk, *, q_sf, status_out
    ):
        self.operands.append((q, q_sf, weights))
        for row in range(q.shape[0]):
            scores = _row_scores(q, q_sf, weights, k, k_scale, row, 0, int(ke[row]))
            out_idx[row] = _best(scores, topk)
        status_out.zero_()
        return True


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
    """An empty plugin ledger and environment, and CPU stand-ins for the kernels and device."""
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
    monkeypatch.setattr(binding_module, "_num_sms", lambda device: SMS)
    monkeypatch.setattr(binding_module, "_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(binding_module, "_module_device", lambda module: torch.device("cpu"))
    monkeypatch.setattr(binding_module, "topk_kernel", lambda exact_topk: exact_topk_kernel)
    # The real head-count probe of the reference selector, against a CPU score kernel.
    monkeypatch.setattr(reference_module, "_HEAD_SUPPORT", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 0))
    yield
    for key in _plugin_env():
        os.environ.pop(key)
    for name in _fake_modules():
        sys.modules.pop(name)


def _score_kernel(monkeypatch, supported=DEEPGEMM_HEADS) -> FakeScoreKernel:
    kernel = FakeScoreKernel(supported)
    monkeypatch.setattr(reference_module, "_mqa_logits", kernel)
    return kernel


def _make_plugin(root: Path, routes, *, min_keys=(196608, 65536), vote_rows=1536) -> Path:
    kernels = root / "litetopk_kernels"
    kernels.mkdir(parents=True)
    for name in _KERNEL_FILES:
        (kernels / name).write_text(f"// {name} of fake source {root.name}\n", encoding="utf-8")
    (root / "litetopk.py").write_text(_FAKE_ADAPTER, encoding="utf-8")
    config = {
        "reads": list(_READS),
        "routes": routes,
        "admit_key": _ADMIT,
        "tie_key": _TIE,
        "score_key": _SCORE,
        "min_keys": list(min_keys),
        "vote_rows": vote_rows,
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


def _geometry(heads: int, fmt: str) -> IndexerGeometry:
    if fmt == "fp8":
        return IndexerGeometry(num_heads=heads, head_dim=HEAD_DIM, topk=2048, key_ratio=1)
    return IndexerGeometry(num_heads=heads, head_dim=HEAD_DIM, topk=512, key_ratio=4)


def _configure(tmp_path, geometry, fmt, *, backend="litetopk", precision, padding, tuning=None):
    """Bind one consumer; return its binding, or the IndexerTopKConfigError text."""
    _new_process()
    fields = {"backend": backend, "precision": precision, "head_padding": padding}
    if backend == "litetopk":
        root = tmp_path / "plugin"
        if not root.exists():
            _make_plugin(root, [_FP8_ROUTE, _FP4_ROUTE])
        fields["litetopk"] = LiteTopKPluginConfig(source=str(root))
    if precision == "exact":
        fields["exact_topk"] = ExactTopKConfig(source=str(tmp_path))
    consumer = Consumer(geometry)
    try:
        binding_module.configure_indexer_topk(
            [consumer], IndexerTopKConfig(**fields), native_format=fmt, tuning=tuning
        )
    except IndexerTopKConfigError as error:
        assert consumer.binding == "unset"  # nothing is bound when a layer cannot be served
        return str(error)
    return consumer.binding


# The expected negotiation against routes with kernels for 32 and 64 heads and a reference score
# kernel for 16, 32 and 64 heads: the heads LiteTopK pads to, and the heads the reference score
# kernel needs on its own.
_PADDED = {4: 32, 8: 32, 12: 32, 16: 32, 20: 32, 48: 64}
_BASELINE = {4: 16, 8: 16, 12: 16, 16: 16, 20: 32, 32: 32, 48: 64, 64: 64}
# Default first LiteTopK position of a layer whose reference selector scores as many heads as
# LiteTopK: FP8 8192 before the 196608-key route minimum with 32 kernel heads, none (LiteTopK is
# slower than the reference selector) with 64; MXFP4 the row that sees 45056 compressed keys.
_STARTUP = {("fp8", 32): 188416, ("fp8", 64): None, ("mxfp4", 32): 180224, ("mxfp4", 64): 180224}
_MATRIX_HEADS = (4, 8, 12, 16, 20, 32, 48, 64, 96, 128)


def _route_text(fmt: str) -> tuple[str, str]:
    if fmt == "fp8":
        return "fp8_paged", "[2048]"
    return "fp4_slab", "<= 2048"


@pytest.mark.parametrize("heads", _MATRIX_HEADS)
@pytest.mark.parametrize(
    "fmt, precision", [("fp8", "exact"), ("fp8", "fast"), ("mxfp4", "fast")], ids=str
)
def test_capability_matrix(tmp_path, monkeypatch, fmt, precision, heads):
    _score_kernel(monkeypatch)
    geometry = _geometry(heads, fmt)
    route, topk_text = _route_text(fmt)
    for padding in (False, True):
        result = _configure(tmp_path, geometry, fmt, precision=precision, padding=padding)
        source = next(iter(loader._LOADED.values())).source_id
        prefix = (
            f"LiteTopK route {route} (source {source}) supports indexer heads [32, 64], "
            f"head_dim [128], topk {topk_text}; layer Consumer has H={heads}, D=128, "
            f"K={geometry.topk}."
        )
        if heads in (96, 128):
            # Padding cannot reach a route head count, and the reference selector cannot score
            # the layer either.
            assert result == (
                f"{prefix} The reference score kernel cannot score {heads} heads either; keep "
                "backend='default'."
            )
            continue
        if heads in _PADDED and not padding:
            kernel = _PADDED[heads]
            assert result == (
                f"{prefix} Set indexer_topk.head_padding=True to pad heads to {kernel} "
                f"({kernel / heads:.2f}x scoring work) or use backend='reference'."
            )
            continue
        litetopk = _PADDED.get(heads, heads)
        baseline = _BASELINE[heads]
        # The reference selector scores the plugin's padded operands, except with precision
        # fast when DeepGEMM supports the layer's own head count.
        reference = heads if precision == "fast" and baseline == heads else litetopk
        assert result.heads == IndexerHeads(heads, litetopk, reference, baseline), padding
        state = result._state(torch.device("cpu"))
        assert state.engine.kernel_heads == litetopk and state.heads == result.heads
        startup = _STARTUP[(fmt, litetopk)] if baseline >= litetopk else None
        assert state.tuning.startup_position == startup
        # Without LiteTopK rows the reference selector scores like the reference backend.
        assert state.reference_heads == (reference if startup is not None else baseline)
        if fmt == "fp8":
            # Three waves of the kernel heads' score kernel, at least 12 rows per SM.
            assert state.tuning.tile_rows == SMS * max(12, 3 * (128 // litetopk))
            # The plugin settings admit the tiles of the layer's own head count.
            admitted = state.tuning.plugin_settings.fp8_paged_admit_max_query_len
            assert admitted == SMS * max(12, 3 * (128 // heads)) >= state.tuning.tile_rows

    # The reference backend pads only where the reference score kernel has no kernel for the
    # head count, with or without head_padding.
    for padding in (False, True):
        result = _configure(
            tmp_path, geometry, fmt, backend="reference", precision=precision, padding=padding
        )
        if heads in (96, 128):
            assert result == (
                f"DeepGEMM fp8_fp4_mqa_logits supports no head count from {heads} to 128 for "
                f"{fmt} operands with head_dim 128; the indexer top-k reference selector "
                f"cannot score {heads} heads"
            )
            continue
        baseline = _BASELINE[heads]
        assert result.heads == IndexerHeads(heads, None, baseline, baseline)
        assert result._state(torch.device("cpu")).reference_heads == baseline


def test_negotiation_error_texts_and_order():
    route = RouteCapability(
        name="fp8_paged",
        fmt="fp8",
        heads=frozenset({48}),
        head_dims=frozenset({128}),
        topk=frozenset({2048}),
        max_topk=2048,
        qualified_query_lengths=frozenset(),
        admitted_max_query_len=0,
        min_keys=196608,
        max_keys=1048576,
        hot_prefix=12288,
        exact=True,
        tie_policies=frozenset({"logical-id"}),
        score_policies=frozenset({"native-fp32"}),
    )
    probes = []

    def deepgemm(heads):
        probes.append(heads)
        supported = [value for value in sorted(DEEPGEMM_HEADS) if value >= heads]
        if not supported:
            raise IndexerTopKConfigError(f"no head count from {heads}")
        return supported[0]

    def negotiate(heads, **overrides):
        arguments = dict(
            name="L",
            precision="exact",
            head_padding=True,
            route=route,
            source_id="0123456789ab",
            reference_heads=deepgemm,
        )
        arguments.update(overrides)
        return negotiate_indexer_heads(_geometry(heads, "fp8"), **arguments)

    # A route head count DeepGEMM has no kernel for: precision exact needs the reference
    # selector to score the plugin's operands (48 heads) and it would pad them to 64.
    with pytest.raises(IndexerTopKConfigError) as error:
        negotiate(40)
    assert str(error.value) == (
        "precision='exact' scores the rows LiteTopK does not cover on the plugin's operands, "
        "but the reference score kernel cannot score 48 heads (layer L, H=40; it would pad "
        "them to 64); use precision='fast'"
    )
    assert negotiate(40, precision="fast") == IndexerHeads(40, 48, 64, 64)
    assert negotiate(48, precision="fast") == IndexerHeads(48, 48, 64, 64)
    # Every head count is probed once per negotiation.
    probes.clear()
    negotiate(40, precision="fast")
    assert probes == [40, 48]
    # Padding needs a multiple of four.
    with pytest.raises(IndexerTopKConfigError, match=r"has H=42, D=128, K=2048\. Use backend"):
        negotiate(42)
    # Padding cannot help another head dimension; the advice depends on the reference selector.
    wide = IndexerGeometry(num_heads=48, head_dim=64, topk=2048, key_ratio=1)
    with pytest.raises(IndexerTopKConfigError, match=r"has H=48, D=64, K=2048\. Use backend"):
        negotiate_indexer_heads(
            wide,
            name="L",
            precision="exact",
            head_padding=True,
            route=route,
            source_id="0123456789ab",
            reference_heads=deepgemm,
        )
    # Without a reference score kernel for the layer, padding is the only remedy offered.
    with pytest.raises(IndexerTopKConfigError) as error:
        negotiate(32, head_padding=False, reference_heads=lambda heads: deepgemm(heads + 64))
    assert str(error.value).endswith(
        "Set indexer_topk.head_padding=True to pad heads to 48 (1.50x scoring work)."
    )
    # The upstream selector selects the reference rows: no probe, no reference heads.
    probes.clear()
    assert negotiate(32, reference_heads=None) == IndexerHeads(32, 48, None, None)
    assert probes == []
    # Without a route only the reference selector negotiates.
    assert negotiate(20, route=None, source_id=None) == IndexerHeads(20, None, 32, 32)
    with pytest.raises(IndexerTopKConfigError, match="no head count from 96"):
        negotiate(96, route=None, source_id=None)


def test_padded_heads_of_a_route():
    route = RouteCapability.__new__(RouteCapability)
    object.__setattr__(route, "heads", frozenset({32, 64}))
    assert [route.padded_heads(heads) for heads in (4, 16, 20, 28, 32, 36, 48, 60, 64, 96)] == [
        32,
        32,
        32,
        32,
        64,
        64,
        64,
        64,
        None,
        None,
    ]
    assert route.padded_heads(30) is None and route.padded_heads(0) is None


def test_head_padding_config_field():
    assert IndexerTopKConfig().head_padding is False
    normalized = normalize_indexer_topk_config(
        {"backend": "reference", "precision": "fast", "head_padding": True}
    )
    assert normalized == IndexerTopKConfig(backend="reference", precision="fast", head_padding=True)
    for value in ("yes", 1, None):
        with pytest.raises(
            IndexerTopKConfigError, match=r"indexer_topk.head_padding must be a bool, got"
        ):
            normalize_indexer_topk_config({"backend": "default", "head_padding": value})


def test_head_records(tmp_path, monkeypatch):
    _score_kernel(monkeypatch)
    heads = IndexerHeads(16, 32, 32, 16)
    assert heads.litetopk_padded and heads.reference_padded
    assert heads.as_dict() == {
        "num_heads": 16,
        "litetopk_heads": 32,
        "reference_heads": 32,
        "baseline_heads": 16,
        "litetopk_padded": True,
        "reference_padded": True,
    }
    assert not IndexerHeads(32, 32, 32, 32).litetopk_padded
    assert not IndexerHeads(16, None, None, None).reference_padded
    for bad in ({"num_heads": 0}, {"num_heads": 16, "litetopk_heads": 8}):
        with pytest.raises(ValueError, match="IndexerHeads"):
            IndexerHeads(**bad)
    # The installation records the head counts of every binding.
    _new_process()
    root = _make_plugin(tmp_path / "plugin", [_FP8_ROUTE, _FP4_ROUTE])
    consumer = Consumer(_geometry(16, "fp8"))
    installation = binding_module.configure_indexer_topk(
        [consumer],
        {
            "backend": "litetopk",
            "precision": "exact",
            "head_padding": True,
            "litetopk": {"source": str(root)},
            "exact_topk": {"source": str(tmp_path)},
        },
        native_format="fp8",
    )
    assert installation.heads() == {"Consumer": heads.as_dict()}


def test_required_needs_a_default_litetopk_start(tmp_path, monkeypatch):
    _score_kernel(monkeypatch)
    required = IndexerTopKTuning(required=True)
    result = _configure(
        tmp_path, _geometry(16, "fp8"), "fp8", precision="exact", padding=True, tuning=required
    )
    assert result == (
        "IndexerTopKTuning.required needs LiteTopK rows, but the default plan gives LiteTopK "
        "none for fp8 operands with 32 kernel heads (LiteTopK runs 32 zero-padded heads, the "
        "reference selector 16); set IndexerTopKTuning.startup_position to select with LiteTopK "
        "anyway"
    )
    result = _configure(
        tmp_path, _geometry(64, "fp8"), "fp8", precision="exact", padding=False, tuning=required
    )
    assert result.startswith("IndexerTopKTuning.required needs LiteTopK rows") and (
        "with 64 kernel heads (no start was faster than the reference selector at every "
        "measured length)" in result
    )
    # An explicit start position enables LiteTopK.
    explicit = IndexerTopKTuning(required=True, startup_position=188416)
    binding = _configure(
        tmp_path, _geometry(64, "fp8"), "fp8", precision="exact", padding=False, tuning=explicit
    )
    assert binding.resolved_tuning(torch.device("cpu")).startup_position == 188416


# ---------------------------------------------------------------------------------------------
# The head reduction of the score kernels, simulated exactly
# ---------------------------------------------------------------------------------------------


def _float32(value: float) -> float:
    """``value`` rounded to the nearest float32 (as a Python float)."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _round_float32(value: Fraction) -> float:
    """The float32 nearest to the nonzero rational ``value``, ties to even (one rounding)."""
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1  # now 2**exponent <= magnitude < 2**(exponent + 1)
    quantum = Fraction(2) ** (max(exponent, -126) - 23)  # subnormals below 2**-126
    scaled = magnitude / quantum
    whole = scaled.numerator // scaled.denominator
    rest = scaled - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2):
        whole += 1
    result = whole * quantum
    assert result < Fraction(2) ** 128, "the simulation stays in the finite range"
    return math.copysign(float(result), value)


def _negative(value: float) -> bool:
    return math.copysign(1.0, value) < 0


def _fma(a: float, b: float, c: float) -> float:
    """float32 fused multiply-add: ``a * b + c`` rounded once, IEEE signs of zero."""
    exact = Fraction(a) * Fraction(b) + Fraction(c)
    if exact != 0:
        return _round_float32(exact)
    if a * b == 0 and c == 0:  # a sum of two zeros is -0 only when both are -0
        return -0.0 if _negative(a) != _negative(b) and _negative(c) else 0.0
    return 0.0  # exact cancellation rounds to +0


def _add(a: float, b: float) -> float:
    return _fma(1.0, a, b)


def _mul(a: float, b: float) -> float:
    exact = Fraction(a) * Fraction(b)
    if exact != 0:
        return _round_float32(exact)
    return -0.0 if _negative(a) != _negative(b) else 0.0


def _relu(value: float, zero: float) -> float:
    """``fmaxf(value, 0)``; ``zero`` is what it returns for -0 (either sign is allowed)."""
    if value > 0:
        return value
    return zero if value == 0 and _negative(value) else 0.0


def _score(scores, weights, scale, relu_zero):
    """The epilogue of the score kernels for one (row, key): returns (chains, head sum, score).

    Heads go four at a time into two float2 FMA accumulators (``sum_0`` takes heads j and j + 1,
    ``sum_1`` heads j + 2 and j + 3), both starting at +0; then ``__fadd2_rn(sum_0, sum_1)``,
    the sum of its two lanes and the key scale.
    """
    chains = [0.0, 0.0, 0.0, 0.0]  # sum_0.x, sum_0.y, sum_1.x, sum_1.y
    for head in range(0, len(scores), 4):
        for lane in range(4):
            relu = _relu(scores[head + lane], relu_zero)
            chains[lane] = _fma(relu, weights[head + lane], chains[lane])
    head_sum = _add(_add(chains[0], chains[2]), _add(chains[1], chains[3]))
    return chains, head_sum, _mul(scale, head_sum)


def _random_value(rng: random.Random, zeros: float, exponents: tuple[int, int]) -> float:
    if rng.random() < zeros:
        return rng.choice((0.0, -0.0))
    sign = rng.choice((1.0, -1.0))
    return _float32(sign * rng.uniform(1.0, 2.0) * 2.0 ** rng.randint(*exponents))


def _rows(rng: random.Random, heads: int, count: int, exponents: tuple[int, int]):
    """Random rows, then rows built to hit the corner cases of the reduction."""
    for _ in range(count):
        scores = [_random_value(rng, 0.1, exponents) for _ in range(heads)]
        weights = [_random_value(rng, 0.05, exponents) for _ in range(heads)]
        yield scores, weights, _float32(rng.uniform(0.5, 2.0))
    # Every score negative or zero: every product is a signed zero, every chain stays +0.
    yield [-1.0] * (heads // 2) + [-0.0] * (heads - heads // 2), [-2.0] * heads, 1.0
    # Exact cancellation: each chain returns to +0 after the first two of its heads.
    if heads >= 8:
        value = _float32(1.0 + 2.0**-20)
        weights = [value] * 4 + [-value] * 4 + [3.0] * (heads - 8)
        yield [1.0] * heads, weights, 1.0
    # Subnormal products (multiples of the smallest subnormal) and a term absorbing them.
    yield [2.0**-75] * heads, [2.0**-74] * (heads - 1) + [2.0**100], 2.0**-20


def _padded(scores, weights, scale, relu_zero, pad, padded_score):
    return _score(scores + [padded_score] * pad, weights + [0.0] * pad, scale, relu_zero)


@pytest.mark.parametrize("relu_zero", [0.0, -0.0], ids=["relu(-0)=+0", "relu(-0)=-0"])
@pytest.mark.parametrize(
    "heads, kernel_heads",
    [(4, 32), (8, 32), (12, 32), (16, 32), (20, 32), (48, 64), (16, 64), (32, 64)],
)
def test_zero_head_padding_ffma2_simulation(heads, kernel_heads, relu_zero):
    pad = kernel_heads - heads
    # Operands of at least 2**-50: every product is a multiple of the smallest float32
    # subnormal, so no sum underflows and no chain ever holds -0. Appending zero heads (whose
    # score is a signed zero: a zero-code head's accumulator) then leaves every chain, the head
    # sum and the score bit for bit unchanged.
    rng = random.Random(heads * 1000 + kernel_heads)
    checked = 0
    for scores, weights, scale in _rows(rng, heads, 200, (-50, 40)):
        native = _score(scores, weights, scale, relu_zero)
        assert not any(value == 0 and _negative(value) for value in native[0])
        for padded_score in (0.0, -0.0):
            padded = _padded(scores, weights, scale, relu_zero, pad, padded_score)
            assert [_bits(value) for value in padded[0]] == [_bits(value) for value in native[0]]
            assert _bits(padded[1]) == _bits(native[1])
            assert _bits(padded[2]) == _bits(native[2])
            checked += 1
    assert checked >= 400

    # Operands down to 2**-140: a negative product (or partial sum) below half the smallest
    # subnormal rounds to -0, and a zero head adding +0 turns that chain into +0. Then only the
    # sign of a zero changes; every nonzero chain, head sum and score keeps its bits, and a
    # score changes only from -0 to +0, when every chain was -0.
    flips = {"chain": 0, "score": 0}
    tiny = _rows(random.Random(heads * 1000 + kernel_heads + 1), heads, 200, (-140, -60))
    underflow = [_float32(2.0**-80)] * heads, [_float32(-(2.0**-80))] * heads, 1.0
    for scores, weights, scale in [*tiny, underflow]:
        native = _score(scores, weights, scale, relu_zero)
        for padded_score in (0.0, -0.0):
            padded = _padded(scores, weights, scale, relu_zero, pad, padded_score)
            pairs = [*zip(padded[0], native[0]), (padded[1], native[1]), (padded[2], native[2])]
            for after, before in pairs:
                assert after == before  # equal values (a zero of either sign equals a zero)
                if _bits(after) != _bits(before):
                    assert before == 0 and _negative(before) and not _negative(after)
            flips["chain"] += sum(_bits(a) != _bits(b) for a, b in zip(padded[0], native[0]))
            flips["score"] += _bits(padded[2]) != _bits(native[2])
    # The all-underflowing row: four -0 chains, a -0 score; a padded +0 head makes it +0.
    assert flips["chain"] > 0 and flips["score"] > 0

    # Negative control: heads that are not zero (a subnormal weight on a unit score) do change
    # scores, so the comparisons above can fail.
    changed = 0
    for scores, weights, scale in _rows(random.Random(7), heads, 50, (-50, 40)):
        native = _score(scores, weights, scale, relu_zero)
        nonzero = _score(scores + [1.0] * pad, weights + [2.0**-149] * pad, scale, relu_zero)
        changed += _bits(nonzero[1]) != _bits(native[1])
    assert changed > 0


def test_float32_rounding_of_the_simulation():
    # Ties go to even, values are rounded once, subnormals have a fixed quantum.
    assert _round_float32(Fraction(1) + Fraction(1, 2**24)) == 1.0
    assert _round_float32(Fraction(1) + Fraction(3, 2**24)) == 1.0 + 2.0**-22
    assert _round_float32(Fraction(3, 2**150)) == 2.0**-148
    assert _round_float32(Fraction(-1, 2**150)) == -0.0 and _negative(
        _round_float32(Fraction(-1, 2**150))
    )
    # fma rounds once where a separate multiply and add round twice.
    a = _float32(1.0 + 2.0**-12)
    assert _fma(a, a, -1.0) == 2.0**-11 + 2.0**-24
    assert _add(_mul(a, a), -1.0) == 2.0**-11
    # Signs of zero.
    assert _negative(_fma(-0.0, 1.0, -0.0)) and not _negative(_fma(-0.0, 1.0, 0.0))
    assert not _negative(_fma(1.0, 1.0, -1.0)) and _negative(_mul(-2.0, 0.0))


# ---------------------------------------------------------------------------------------------
# Padded operands reach the plugin and the reference selector
# ---------------------------------------------------------------------------------------------

_SMALL_FP8 = {
    **_FP8_ROUTE,
    "heads": [16],
    "topk": [8],
    "max_topk": 8,
    "qualified_query_lengths": [],
    "min_keys": 96,
    "max_keys": 4096,
    "hot_prefix": 24,
}
_SMALL_FP4 = {
    **_SMALL_FP8,
    "name": "fp4_slab",
    "fmt": "mxfp4",
    "topk": None,
    "exact": False,
    "qualified_query_lengths": [12, 16],
    "min_keys": 64,
    "tie_policies": ["storage"],
    "score_policies": ["folded"],
}
_SMALL_TUNING = {
    "fp8": IndexerTopKTuning(tile_rows=16, startup_position=160, group_tiles=3),
    "mxfp4": IndexerTopKTuning(tile_rows=16, startup_position=0),
}
_UE8M0_ONE = 0x7F7F7F7F  # four group scales of 127, the UE8M0 code of 1.0


def _inputs(rows, keys, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn((rows, heads, HEAD_DIM), generator=generator).to(torch.bfloat16)
    k = torch.randn((keys, HEAD_DIM), generator=generator).to(torch.bfloat16)
    weights = torch.randn((rows, heads), generator=generator).to(torch.bfloat16)
    return q, k, weights


def _expected(q, k, weights, layout, topk, softmax_scale, fmt):
    """Exact top-k of every row over its visible keys, from the unpadded quantized operands."""
    data, scales, folded = quantize_queries(
        q, weights, fmt, softmax_scale=softmax_scale, kernel_heads=q.shape[1]
    )
    keys = quantize_keys(k, fmt)
    out = torch.full((layout.rows, topk), -1, dtype=torch.int32)
    for segment in layout.segments:
        for row in range(segment.row_start, segment.row_end):
            end = layout.visible_keys(segment, row)
            if end:
                scores = _row_scores(data, scales, folded, keys.data, keys.scale, row, 0, end)
                order = _best(scores, topk)
                out[row, : order.numel()] = order
    return sort_topk_rows_(out)


@pytest.mark.parametrize(
    "fmt, precision, heads, reference_heads, baseline_heads",
    [
        # The reference selector scores the plugin's padded operands.
        ("fp8", "exact", 4, 16, 8),
        # Precision fast and a head count DeepGEMM supports: no reference padding.
        ("fp8", "fast", 8, 8, 8),
        # Precision fast, but DeepGEMM pads the head count anyway: the plugin's operands.
        ("fp8", "fast", 4, 16, 8),
        ("mxfp4", "fast", 12, 16, 16),
        # The route's own head count: head_padding pads nothing and counts no padded row.
        ("fp8", "exact", 16, 16, 16),
        ("mxfp4", "fast", 16, 16, 16),
    ],
)
def test_padded_operands_passed_to_plugin(
    tmp_path, monkeypatch, fmt, precision, heads, reference_heads, baseline_heads
):
    kernel = _score_kernel(monkeypatch, supported={8, 16})
    root = _make_plugin(
        tmp_path / "plugin", [_SMALL_FP8, _SMALL_FP4], min_keys=(96, 64), vote_rows=12
    )
    geometry = IndexerGeometry(
        num_heads=heads, head_dim=HEAD_DIM, topk=8, key_ratio=1 if fmt == "fp8" else 4
    )
    fields = {
        "backend": "litetopk",
        "precision": precision,
        "head_padding": True,
        "litetopk": {"source": str(root)},
    }
    if precision == "exact":
        fields["exact_topk"] = {"source": str(tmp_path)}
    consumer = Consumer(geometry)
    binding_module.configure_indexer_topk(
        [consumer], fields, native_format=fmt, tuning=_SMALL_TUNING[fmt]
    )
    binding = consumer.binding
    assert binding.heads == IndexerHeads(heads, 16, reference_heads, baseline_heads)
    fake = binding.plugin.module.BEHAVIOR = FakeLiteTopK()
    kernel.calls.clear()  # the head-count probes of the configuration

    if fmt == "fp8":
        rows, keys, scale = 250, 250, 0.5
        layout = QueryLayout.full(rows, keys=keys)
    else:
        rows, keys, scale = 396, 96, HEAD_DIM**-0.5
        layout = QueryLayout.full(rows, keys=keys, key_ratio=4)
    q, k, weights = _inputs(rows, keys, heads, seed=heads)
    with torch.no_grad():
        out = binding.select(q, k, weights, layout=layout, topk=8, softmax_scale=scale)
    # Zero heads change no selection: both selectors agree with the unpadded exact top-k.
    assert torch.equal(out, _expected(q, k, weights, layout, 8, scale, fmt))

    # The plugin got 16 heads per row: the layer's quantized heads, then (below 16 heads) zero
    # codes and zero weights (and MXFP4 group scales 127).
    assert fake.operands
    for data, scales, folded in fake.operands:
        assert data.shape[1:] == (16, data.shape[2]) and folded.shape[1] == 16
        assert not data[:, heads:].view(torch.uint8).any()
        assert not folded[:, heads:].any() and not folded[:, heads:].signbit().any()
        if fmt == "fp8":
            assert scales is None
        else:
            assert bool((scales[:, heads:] == _UE8M0_ONE).all())
            assert bool((scales[:, heads:].view(torch.uint8) == 127).all())
    # The first tile's own heads are the layer's quantized heads.
    tile_data, tile_scales, tile_weights = fake.operands[0]
    tile_rows = tile_data.shape[0]
    tile_start = 162 if fmt == "fp8" else 96
    expected = quantize_queries(
        q[tile_start : tile_start + tile_rows],
        weights[tile_start : tile_start + tile_rows],
        fmt,
        softmax_scale=scale,
        kernel_heads=heads,
    )
    assert torch.equal(tile_data[:, :heads].view(torch.uint8), expected[0].view(torch.uint8))
    assert torch.equal(tile_weights[:, :heads], expected[2])
    if fmt == "mxfp4":
        assert torch.equal(tile_scales[:, :heads], expected[1])

    # The reference selector scored the head count the negotiation chose.
    assert kernel.calls and {call["heads"] for call in kernel.calls} == {reference_heads}
    stats = binding.stats
    padded_litetopk = stats.litetopk_rows if heads < 16 else 0
    assert stats.litetopk_rows > 0 and stats.padded_litetopk_rows == padded_litetopk
    padded_reference = stats.reference_rows if reference_heads > heads else 0
    assert stats.padded_reference_rows == padded_reference
    assert stats.as_dict()["padded_litetopk_rows"] == padded_litetopk
    assert stats.as_dict()["padded_reference_rows"] == padded_reference


def test_padding_default_plan_leaves_rows_to_the_reference(tmp_path, monkeypatch):
    # LiteTopK pads 8 heads to 16 while DeepGEMM scores 8 heads natively: no crossover was
    # measured for that, so the default plan gives LiteTopK no row and the reference selector
    # keeps the layer's own head count.
    kernel = _score_kernel(monkeypatch, supported={8, 16})
    root = _make_plugin(
        tmp_path / "plugin", [_SMALL_FP8, _SMALL_FP4], min_keys=(96, 64), vote_rows=12
    )
    geometry = IndexerGeometry(num_heads=8, head_dim=HEAD_DIM, topk=8, key_ratio=1)
    consumer = Consumer(geometry)
    binding_module.configure_indexer_topk(
        [consumer],
        {
            "backend": "litetopk",
            "precision": "exact",
            "head_padding": True,
            "litetopk": {"source": str(root)},
            "exact_topk": {"source": str(tmp_path)},
        },
        native_format="fp8",
        tuning=IndexerTopKTuning(tile_rows=16),
    )
    binding = consumer.binding
    assert binding.heads == IndexerHeads(8, 16, 16, 8)
    binding.plugin.module.BEHAVIOR = fake = FakeLiteTopK()
    kernel.calls.clear()  # the head-count probes of the configuration
    q, k, weights = _inputs(250, 250, 8, seed=3)
    layout = QueryLayout.full(250, keys=250)
    with torch.no_grad():
        out = binding.select(q, k, weights, layout=layout, topk=8, softmax_scale=0.5)
    assert torch.equal(out, _expected(q, k, weights, layout, 8, 0.5, "fp8"))
    assert not fake.operands and binding.stats.litetopk_rows == 0
    assert dict(binding.stats.reference_segments) == {
        "no LiteTopK start position for the kernel heads": 1
    }
    assert {call["heads"] for call in kernel.calls} == {8}
    assert binding.stats.padded_reference_rows == 0
