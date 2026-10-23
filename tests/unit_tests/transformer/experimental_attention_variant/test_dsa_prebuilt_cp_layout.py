# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU tests for DSA packed-CP layout tables prebuilt at batch construction.

The prebuilt tables must be bit-identical to the ones ``DSAttention.forward`` memoizes
lazily, DSA must consume them without running any layout builder, fall back to the
lazy path when they do not answer the request, and refresh in place for graphs.
"""

import itertools
import logging
import os
import random
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.context_parallel_layout import prebuild_thd_cp_partition_routes
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa as dsa_module
from megatron.core.transformer.experimental_attention_variant import dsa_layout, dsa_masking
from megatron.core.transformer.experimental_attention_variant.dsa import (
    DSAttention,
    DSAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.dsa_layout import (
    PackedCPLayoutGeometry,
    PrebuiltPackedCPLayout,
    build_packed_allgather_cp_local_positions_from_host,
    build_packed_allgather_cp_query_positions_and_key_reorder_from_host,
    build_packed_cp_layout_from_host,
    copy_prebuilt_packed_cp_layout_,
    get_prebuilt_packed_cp_layout,
    prebuild_packed_cp_layout,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.training.utils import packed_seq_utils

_CPU = torch.device("cpu")
_HIDDEN = 64
_HEADS = 4
_HEAD_DIM = 8
_Q_LORA = 16
_COVER_COMBOS = list(itertools.product((False, True), repeat=2))


class _FakeGroup:
    def __init__(self, size: int, rank: int = 0):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


def _compact(boundaries):
    compact = [boundaries[0]]
    for value in boundaries[1:]:
        if value != compact[-1]:
            compact.append(value)
    return compact


def _random_boundaries(rng, cp_size, num_seqs, *, total_units=None, zero_prob=0.25):
    """Packed boundaries whose sequence lengths are multiples of ``2 * cp_size``."""
    unit = 2 * cp_size
    if total_units is None:
        units = [0 if rng.random() < zero_prob else rng.randint(1, 4) for _ in range(num_seqs)]
    else:
        cuts = sorted(rng.randint(0, total_units) for _ in range(num_seqs - 1))
        units = [b - a for a, b in zip([0] + cuts, cuts + [total_units])]
    boundaries = [0]
    for count in units:
        boundaries.append(boundaries[-1] + count * unit)
    return boundaries


def _memo_path_tables(host_q, host_kv, cp_size, cp_rank, local_rows, q_cover, k_cover):
    """The exact builder calls of DSAttention.forward's two memo entries."""
    positions, reorder, inverse = (
        build_packed_allgather_cp_query_positions_and_key_reorder_from_host(
            host_q,
            host_kv,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=_CPU,
            local_output_size=local_rows,
            key_local_output_size=local_rows,
            global_output_size=local_rows * cp_size,
            query_cu_seqlens_cover_output=q_cover,
            key_cu_seqlens_cover_output=k_cover,
            return_inverse_key_reorder=True,
        )
    )
    sp_positions = build_packed_allgather_cp_local_positions_from_host(
        host_q, cp_size, cp_rank, _CPU, output_size=local_rows
    )
    return positions, reorder, inverse, sp_positions


def _random_layout_cases():
    rng = random.Random(6206)
    cases = []
    for cp_size in (2, 4, 8):
        for trial in range(6):
            num_seqs = rng.randint(1, 5)
            host_q = _compact(_random_boundaries(rng, cp_size, num_seqs))
            if trial % 3 == 1 and host_q[-1] > 0:
                # Distinct K boundaries over the same packed tokens.
                units = host_q[-1] // (2 * cp_size)
                host_kv = _compact(
                    _random_boundaries(rng, cp_size, rng.randint(1, 4), total_units=units)
                )
            else:
                host_kv = host_q
            # Padding rows not covered by cu_seqlens on every other trial.
            pad_rows = 2 * rng.randint(1, 3) if trial % 2 else 0
            local_rows = host_q[-1] // cp_size + pad_rows
            cases.append((cp_size, host_q, host_kv, local_rows))
        # All-empty packs, with and without uncovered padding rows.
        cases.append((cp_size, [0], [0], 0))
        cases.append((cp_size, [0], [0], 4))
    return cases


@pytest.mark.parametrize(("cp_size", "host_q", "host_kv", "local_rows"), _random_layout_cases())
def test_prebuilt_tables_bit_identical_to_memo_builders(cp_size, host_q, host_kv, local_rows):
    """Every rank and cover-flag combination, including the SP positions-only request."""
    for cp_rank in range(cp_size):
        for q_cover, k_cover in _COVER_COMBOS:
            geometry = PackedCPLayoutGeometry(cp_size, cp_rank, local_rows, q_cover, k_cover)
            layout = build_packed_cp_layout_from_host(host_q, host_kv, geometry, _CPU)
            positions, reorder, inverse, sp_positions = _memo_path_tables(
                host_q, host_kv, cp_size, cp_rank, local_rows, q_cover, k_cover
            )
            assert layout.geometry == geometry
            assert torch.equal(layout.query_positions, positions)
            assert torch.equal(layout.kv_reorder_idx, reorder)
            assert torch.equal(layout.kv_reorder_inv_idx, inverse)
            assert torch.equal(
                layout.kv_reorder_idx[layout.kv_reorder_inv_idx],
                torch.arange(local_rows * cp_size, dtype=torch.int64),
            )
            assert torch.equal(layout.query_positions, sp_positions)
            assert layout.nbytes == 8 * local_rows * (1 + 2 * cp_size)
            assert set(layout.named_tables()) == {
                "query_positions",
                "kv_reorder_idx",
                "kv_reorder_inv_idx",
            }


@pytest.fixture(scope="module")
def _single_process_group(tmp_path_factory):
    """DSA reads CP sizes through get_pg_size, which needs an initialized default group."""
    if dist.is_initialized():
        yield
        return
    store_path = os.path.join(tmp_path_factory.mktemp("dsa_prebuilt_store"), "store")
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 1), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


class _FakeIndexer(torch.nn.Module):
    """Indexer stand-in returning ``rows`` CP-local (TP-gathered) rows."""

    index_kpool = 1
    _kpool_gate_score = None
    rows = None

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward_before_topk(self, x, qr, packed_seq_params):
        rows = self.rows if self.rows is not None else x.size(0)
        generator = torch.Generator().manual_seed(rows)
        return (
            torch.randn(rows, x.size(1), 2, 4, generator=generator),
            torch.randn(rows, x.size(1), 4, generator=generator),
            torch.ones(rows, x.size(1), 2),
        )

    def backward_dw(self):
        pass


def _make_dsa(cp_group, tp_group):
    config = MLATransformerConfig(
        num_layers=2,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        use_cpu_initialization=True,
        q_lora_rank=_Q_LORA,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        rope_type="rope",
        rotary_base=10000,
        rotary_percent=1.0,
        dsa_indexer_n_heads=2,
        dsa_indexer_head_dim=4,
        dsa_indexer_topk=8,
        dsa_indexer_loss_coeff=0.0,
        dsa_indexer_use_sparse_loss=False,
    )
    config.attention_backend = "auto"
    config.dsa_kernel_backend = "cudnn"
    attention = DSAttention(
        config=config,
        submodules=DSAttentionSubmodules(indexer=ModuleSpec(module=_FakeIndexer)),
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        cp_comm_type="allgather",
        pg_collection=SimpleNamespace(tp=tp_group, cp=cp_group),
    )
    return attention


class _ForwardHarness:
    """Run DSAttention.forward on CPU with fake collectives and a recording attention."""

    def __init__(self, monkeypatch, cp_size, cp_rank, *, tp_size=1, tp_rank=0):
        self.cp_group = _FakeGroup(cp_size, cp_rank)
        self.tp_group = _FakeGroup(tp_size, tp_rank)
        self.tp_size = tp_size
        self.attention = _make_dsa(self.cp_group, self.tp_group)
        self.attention.config.sequence_parallel = tp_size > 1
        self.seen = {}
        self.inputs = {}
        self.builder_calls = []

        def fake_gather(tensor, group):
            return torch.cat([tensor] * group.size(), dim=0)

        def fake_attention(**kwargs):
            self.seen["key"] = kwargs["key"].detach().clone()
            self.seen["k_indexer"] = kwargs["k_indexer"].detach().clone()
            query = kwargs["query"]
            return torch.zeros(query.size(0), query.size(1), _HIDDEN), None

        real_mask = dsa_masking.build_dsattention_forward_mask

        def recording_mask(**kwargs):
            self.seen["packed_query_positions"] = kwargs["packed_query_positions"]
            return real_mask(**kwargs)

        monkeypatch.setattr(dsa_module, "gather_from_sequence_parallel_region", fake_gather)
        monkeypatch.setattr(dsa_module.dsa_kernels, "run_fused_dsa_attention", fake_attention)
        monkeypatch.setattr(dsa_masking, "build_dsattention_forward_mask", recording_mask)
        for name in ("_cp_reorder_from_host", "_build_cp_positions_from_host"):
            monkeypatch.setattr(dsa_module, name, self._counting(name, getattr(dsa_module, name)))
        for name in (
            "build_packed_allgather_cp_query_positions_and_key_reorder",
            "build_packed_allgather_cp_local_positions",
        ):
            monkeypatch.setattr(dsa_layout, name, self._counting(name, getattr(dsa_layout, name)))

    def set_cp_group(self, cp_size, cp_rank):
        self.cp_group = _FakeGroup(cp_size, cp_rank)
        self.attention.pg_collection.cp = self.cp_group
        return self.cp_group

    def _counting(self, name, builder):
        def wrapped(*args, **kwargs):
            self.builder_calls.append(name)
            return builder(*args, **kwargs)

        return wrapped

    def forward(self, packed_seq_params, local_rows, seed=0, requires_grad=False):
        self.seen.clear()
        self.attention.indexer.rows = local_rows
        sp_rows = local_rows // self.tp_size
        generator = torch.Generator().manual_seed(seed)

        def randn(*shape):
            return torch.randn(*shape, generator=generator)

        query_rows = sp_rows if self.tp_size > 1 else local_rows
        query = randn(query_rows, 1, _HEADS, _HEAD_DIM)
        self.inputs = {
            "key": randn(sp_rows, 1, _HEADS, _HEAD_DIM).requires_grad_(requires_grad),
            "value": randn(sp_rows, 1, _HEADS, _HEAD_DIM).requires_grad_(requires_grad),
        }
        return self.attention(
            query=query,
            key=self.inputs["key"],
            value=self.inputs["value"],
            x=randn(sp_rows, 1, _HIDDEN),
            qr=randn(sp_rows, 1, _Q_LORA),
            attention_mask=None,
            attn_mask_type=AttnMaskType.causal,
            packed_seq_params=packed_seq_params,
        )


def _make_packed_seq_params(boundaries_q, boundaries_kv, max_seqlen_q, max_seqlen_kv):
    cu_q = torch.tensor(boundaries_q, dtype=torch.int32)
    cu_kv = (
        cu_q if boundaries_kv is boundaries_q else torch.tensor(boundaries_kv, dtype=torch.int32)
    )
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
    )


def _dsa_config():
    return SimpleNamespace(experimental_attention_variant="dsa")


def _prepare(monkeypatch, packed_seq_params, cp_group, config=None, **kwargs):
    """prepare_packed_seq_params with the real THD route prebuild for ``cp_group``."""

    def finalize(params):
        params.cp_group = cp_group
        prebuild_thd_cp_partition_routes(params, cp_group)
        return params

    monkeypatch.setattr(packed_seq_utils, "finalize_packed_seq_params", finalize)
    return packed_seq_utils.prepare_packed_seq_params(
        packed_seq_params, config if config is not None else _dsa_config(), **kwargs
    )


def _forward_cases():
    rng = random.Random(2026)
    cases = []
    for cp_size in (2, 4, 8):
        unit = 2 * cp_size
        multi = _random_boundaries(rng, cp_size, 4, zero_prob=0.0)
        multi_with_zero = multi[:2] + [multi[1]] + multi[2:]
        cases.append((cp_size, multi_with_zero, None, None, None, 0, "multi-zero-entry"))
        single = [0, 2 * unit]
        for q_cover, k_cover in _COVER_COMBOS:
            cases.append(
                (
                    cp_size,
                    single,
                    None,
                    single[-1] if q_cover else single[-1] - 1,
                    single[-1] if k_cover else single[-1] - 1,
                    0,
                    f"single-cover-q{int(q_cover)}-k{int(k_cover)}",
                )
            )
        distinct_kv = [0, unit, 3 * unit]
        cases.append((cp_size, [0, 2 * unit, 3 * unit], distinct_kv, None, None, 0, "distinct-qk"))
        cases.append((cp_size, [0, unit, 3 * unit], None, None, None, 4, "uncovered-padding"))
        cases.append((cp_size, [0, 0, 0], None, None, None, 2, "all-empty-padding"))
    return cases


def _case_id(case):
    return f"cp{case[0]}-{case[-1]}"


@pytest.mark.parametrize("sequence_parallel", [False, True], ids=["no-sp", "sp-tp2"])
@pytest.mark.parametrize(
    ("cp_size", "boundaries_q", "boundaries_kv", "max_q", "max_kv", "pad_rows", "label"),
    _forward_cases(),
    ids=[_case_id(case) for case in _forward_cases()],
)
def test_dsa_forward_consumes_prebuilt_layout_bit_identically(
    _single_process_group,
    monkeypatch,
    sequence_parallel,
    cp_size,
    boundaries_q,
    boundaries_kv,
    max_q,
    max_kv,
    pad_rows,
    label,
):
    """Prebuilt at batch construction == lazy memo path, and no builder runs in the forward."""
    boundaries_kv = boundaries_q if boundaries_kv is None else boundaries_kv
    local_rows = boundaries_q[-1] // cp_size + pad_rows
    if max_q is None:
        max_q = max(end - start for start, end in zip(boundaries_q[:-1], boundaries_q[1:]))
    max_kv = max_q if max_kv is None else max_kv
    tp_size = 2 if sequence_parallel else 1
    harness = _ForwardHarness(monkeypatch, cp_size, 0, tp_size=tp_size, tp_rank=1)
    for cp_rank in range(cp_size):
        group = harness.set_cp_group(cp_size, cp_rank)
        harness.builder_calls.clear()

        lazy_params = _make_packed_seq_params(boundaries_q, boundaries_kv, max_q, max_kv)
        lazy_params.cp_group = group
        prebuild_thd_cp_partition_routes(lazy_params, group)
        lazy_output = harness.forward(lazy_params, local_rows)
        lazy_seen = dict(harness.seen)
        lazy_memo = getattr(lazy_params, DSAttention._LAYOUT_HOLDER_ATTR)
        assert harness.builder_calls, "the lazy path must build the layout in the forward"
        assert get_prebuilt_packed_cp_layout(lazy_params) is None

        prebuilt_params = _make_packed_seq_params(boundaries_q, boundaries_kv, max_q, max_kv)
        if pad_rows:
            # Rows the boundaries do not cover: the caller states the physical row count.
            prebuilt_params.cp_group = group
            prebuild_thd_cp_partition_routes(prebuilt_params, group)
            layout = prebuild_packed_cp_layout(prebuilt_params, local_rows=local_rows)
        else:
            _prepare(monkeypatch, prebuilt_params, group)
            layout = get_prebuilt_packed_cp_layout(prebuilt_params)
        assert layout is not None and layout.geometry.local_rows == local_rows

        harness.builder_calls.clear()
        prebuilt_output = harness.forward(prebuilt_params, local_rows)
        assert harness.builder_calls == []
        assert getattr(prebuilt_params, DSAttention._LAYOUT_HOLDER_ATTR) == {}

        # Same memo key as the lazy path -> the same tables, bit for bit.
        reorder_key = (
            "positions_and_reorder",
            cp_size,
            cp_rank,
            local_rows,
            local_rows,
            local_rows * cp_size,
            layout.geometry.query_cu_seqlens_cover_output,
            layout.geometry.key_cu_seqlens_cover_output,
        )
        assert reorder_key in lazy_memo, sorted(lazy_memo)
        assert torch.equal(lazy_memo[reorder_key][1], layout.kv_reorder_idx)
        assert torch.equal(lazy_memo[reorder_key][2], layout.kv_reorder_inv_idx)
        if sequence_parallel:
            positions_key = ("local_positions", cp_size, cp_rank, local_rows)
            assert torch.equal(lazy_memo[positions_key], layout.query_positions)
        else:
            assert torch.equal(lazy_memo[reorder_key][0], layout.query_positions)

        positions = harness.seen["packed_query_positions"]
        assert positions.untyped_storage().data_ptr() == (
            layout.query_positions.untyped_storage().data_ptr()
        )
        assert torch.equal(positions, lazy_seen["packed_query_positions"])
        assert torch.equal(harness.seen["key"], lazy_seen["key"])
        assert torch.equal(harness.seen["k_indexer"], lazy_seen["k_indexer"])
        assert torch.equal(prebuilt_output, lazy_output)


def _prebuilt_harness(monkeypatch, boundaries, cp_size=2, cp_rank=1):
    harness = _ForwardHarness(monkeypatch, cp_size, cp_rank)
    params = _make_packed_seq_params(boundaries, boundaries, boundaries[-1], boundaries[-1])
    _prepare(monkeypatch, params, harness.cp_group)
    return harness, params, boundaries[-1] // cp_size


@pytest.mark.parametrize(
    "mutation", ["cp_size", "cp_rank", "local_rows", "query_cover", "key_cover", "device"]
)
def test_dsa_falls_back_to_lazy_path_for_mismatched_prebuilt_layout(
    _single_process_group, monkeypatch, caplog, mutation
):
    boundaries = [0, 8, 24]
    harness, params, local_rows = _prebuilt_harness(monkeypatch, boundaries)
    geometry = get_prebuilt_packed_cp_layout(params).geometry
    other = {
        "cp_size": PackedCPLayoutGeometry(4, 1, local_rows // 2, False, False),
        "cp_rank": geometry._replace(cp_rank=0),
        "local_rows": geometry._replace(local_rows=local_rows + 2),
        "query_cover": geometry._replace(query_cu_seqlens_cover_output=True),
        "key_cover": geometry._replace(key_cu_seqlens_cover_output=True),
        "device": geometry,
    }[mutation]
    mismatched = build_packed_cp_layout_from_host(boundaries, boundaries, other, _CPU)
    if mutation == "device":
        mismatched = PrebuiltPackedCPLayout(
            geometry=other,
            **{name: table.to("meta") for name, table in mismatched.named_tables().items()},
        )
    dsa_layout.attach_prebuilt_packed_cp_layout(params, mismatched)
    monkeypatch.setattr(dsa_module, "_DSA_PREBUILT_LAYOUT_MISMATCH_WARNED", set())

    with caplog.at_level(logging.WARNING, logger=dsa_module.logger.name):
        harness.forward(params, local_rows)
        harness.forward(params, local_rows)

    assert "_cp_reorder_from_host" in harness.builder_calls
    memo = getattr(params, DSAttention._LAYOUT_HOLDER_ATTR)
    expected = build_packed_cp_layout_from_host(boundaries, boundaries, geometry, _CPU)
    ((positions, reorder, inverse),) = memo.values()
    assert torch.equal(positions, expected.query_positions)
    assert torch.equal(reorder, expected.kv_reorder_idx)
    assert torch.equal(inverse, expected.kv_reorder_inv_idx)
    warnings = [r for r in caplog.records if "prebuilt" in r.getMessage()]
    assert len(warnings) == 1


def test_dsa_without_prebuilt_layout_keeps_lazy_memo_path(_single_process_group, monkeypatch):
    harness = _ForwardHarness(monkeypatch, 2, 0)
    params = _make_packed_seq_params([0, 8, 24], [0, 8, 24], 16, 16)
    params.cp_group = harness.cp_group
    prebuild_thd_cp_partition_routes(params, harness.cp_group)
    harness.forward(params, 12)
    harness.forward(params, 12)
    assert harness.builder_calls == ["_cp_reorder_from_host"]


def test_dsa_rejects_inconsistent_prebuilt_layout(_single_process_group, monkeypatch):
    harness, params, local_rows = _prebuilt_harness(monkeypatch, [0, 8, 24])
    layout = get_prebuilt_packed_cp_layout(params)
    layout.kv_reorder_idx = layout.kv_reorder_idx[:-1]
    with pytest.raises(RuntimeError, match="inconsistent"):
        harness.forward(params, local_rows)


def test_mismatched_prebuilt_layout_raises_during_graph_capture(monkeypatch):
    params = _make_packed_seq_params([0, 8, 24], [0, 8, 24], 16, 16)
    geometry = PackedCPLayoutGeometry(2, 0, 12, False, False)
    dsa_layout.attach_prebuilt_packed_cp_layout(
        params, build_packed_cp_layout_from_host([0, 8, 24], [0, 8, 24], geometry, _CPU)
    )
    match = DSAttention._match_prebuilt_packed_cp_layout
    assert match(params, geometry, _CPU) is get_prebuilt_packed_cp_layout(params)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert match(params, geometry, _CPU) is get_prebuilt_packed_cp_layout(params)
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        match(params, geometry._replace(local_rows=14), _CPU)


def test_refresh_in_place_keeps_storage_and_updates_values():
    cp_size, cp_rank, local_rows = 4, 2, 12
    geometry = PackedCPLayoutGeometry(cp_size, cp_rank, local_rows, False, False)
    first = build_packed_cp_layout_from_host([0, 8, 48], [0, 8, 48], geometry, _CPU)
    second = build_packed_cp_layout_from_host([0, 40, 48], [0, 16, 48], geometry, _CPU)
    assert not torch.equal(first.query_positions, second.query_positions)
    assert not torch.equal(first.kv_reorder_idx, second.kv_reorder_idx)
    assert not torch.equal(first.kv_reorder_inv_idx, second.kv_reorder_inv_idx)
    tables_before = first.named_tables()
    pointers_before = {name: table.data_ptr() for name, table in tables_before.items()}

    assert first.copy_(second) is first

    for name, table in first.named_tables().items():
        assert table is tables_before[name]
        assert table.data_ptr() == pointers_before[name]
        assert torch.equal(table, second.named_tables()[name])
    assert first.geometry == geometry

    # Every geometry field must match, the cover flags included: DSA matches the
    # geometry it derives from the static metadata, so a refresh must not relabel it.
    for other in (
        geometry._replace(cp_rank=1),
        PackedCPLayoutGeometry(2, 1, 24, False, False),
        geometry._replace(local_rows=14),
        geometry._replace(query_cu_seqlens_cover_output=True),
        geometry._replace(key_cu_seqlens_cover_output=True),
    ):
        with pytest.raises(ValueError, match="across geometries"):
            first.copy_(build_packed_cp_layout_from_host([0, 48], [0, 48], other, _CPU))
    assert first.geometry == geometry
    with pytest.raises(TypeError):
        first.copy_(object())


def test_refresh_before_backward_is_reported_by_eager_autograd():
    """Pins the copy_ contract: the reorders save a prebuilt table for the backward.

    The K/V and indexer-K reorders (``_permute_rows``) save the inverse reorder. The
    KPool gate reorder runs without grad (KPool rejects the indexer loss, so its indexer
    block runs under ``no_grad``) and saves nothing.
    """
    geometry = PackedCPLayoutGeometry(2, 0, 12, False, False)
    layout_a = build_packed_cp_layout_from_host([0, 8, 24], [0, 8, 24], geometry, _CPU)
    layout_b = build_packed_cp_layout_from_host([0, 16, 24], [0, 16, 24], geometry, _CPU)
    key = torch.randn(24, 3, requires_grad=True)
    out = dsa_module._permute_rows(key, layout_a.kv_reorder_idx, layout_a.kv_reorder_inv_idx)
    layout_a.copy_(layout_b)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


def _record_permute_rows(monkeypatch):
    """Route DSA's ``_permute_rows`` through a recording ``_PermuteRows``.

    The returned list receives ``("forward", perm, inv_perm)`` for every recorded
    reorder and ``("backward", inv_perm)`` with the tensor its backward unpacks.
    """
    records = []
    base = dsa_module._PermuteRows

    class _RecordingPermuteRows(base):
        @staticmethod
        def forward(ctx, x, perm, inv_perm):
            records.append(("forward", perm, inv_perm))
            return base.forward(ctx, x, perm, inv_perm)

        @staticmethod
        def backward(ctx, grad_output):
            (inv_perm,) = ctx.saved_tensors
            records.append(("backward", inv_perm))
            return base.backward(ctx, grad_output)

    monkeypatch.setattr(dsa_module, "_PermuteRows", _RecordingPermuteRows)
    return records


@pytest.mark.parametrize(
    ("tp_size", "tp_rank"), [(1, 0), (2, 0), (2, 1)], ids=["no-sp", "sp-tp2-r0", "sp-tp2-r1"]
)
@pytest.mark.parametrize(
    ("cp_size", "cp_rank", "boundaries", "pad_rows"),
    [(2, 1, [0, 8, 24, 24], 0), (4, 2, [0, 16, 48], 4), (8, 5, [0, 32, 32, 96], 0)],
    ids=["cp2", "cp4-uncovered-padding", "cp8-zero-entry"],
)
def test_permute_rows_backward_gathers_with_the_prebuilt_inverse(
    _single_process_group, monkeypatch, tp_size, tp_rank, cp_size, cp_rank, boundaries, pad_rows
):
    """The K/V and indexer-K reorders save the prebuilt inverse and gather with it.

    With a prebuilt layout every ``_PermuteRows`` gets the layout's own
    ``kv_reorder_idx`` and ``kv_reorder_inv_idx``, and its backward reads that inverse
    back. The gradients are bit-identical to the lazy memo path's and to autograd's
    ``index_select`` backward for the same incoming gradient. Without sequence
    parallelism the reorders take the tables from the positions + reorder site; with it
    the query positions come from the positions-only site and the K/V reorders from
    ``_build_kv_reorder_idx``.
    """
    records = _record_permute_rows(monkeypatch)
    harness = _ForwardHarness(monkeypatch, cp_size, cp_rank, tp_size=tp_size, tp_rank=tp_rank)
    # Sequence-local TP query shards reject the indexer loss, so with SP the indexer-K
    # reorder runs without grad and only K and V record a backward. Without SP a
    # positive coefficient records the indexer-K reorder for the backward as well.
    with_indexer_loss = tp_size == 1
    names = ("key", "value", "k_indexer") if with_indexer_loss else ("key", "value")
    loss_weights = {}

    def weighted_sum(name, tensor):
        if name not in loss_weights:
            generator = torch.Generator().manual_seed(len(loss_weights) + 1)
            loss_weights[name] = torch.randn(tensor.shape, generator=generator)
        return (tensor * loss_weights[name]).sum()

    live = {}

    def fake_attention(**kwargs):
        # Output (and indexer loss) depend on the reordered K/V (and indexer K).
        for name in names:
            live[name] = kwargs[name]
            live[name].retain_grad()
        query = kwargs["query"]
        attention_loss = weighted_sum("key", kwargs["key"]) + weighted_sum("value", kwargs["value"])
        output = torch.zeros(query.size(0), query.size(1), _HIDDEN) + attention_loss
        if not with_indexer_loss:
            return output, None
        return output, weighted_sum("k_indexer", kwargs["k_indexer"])

    monkeypatch.setattr(dsa_module.dsa_kernels, "run_fused_dsa_attention", fake_attention)
    if with_indexer_loss:
        harness.attention.config.dsa_indexer_loss_coeff = 1.0
        monkeypatch.setattr(
            dsa_module.DSAIndexerLossLoggingHelper,
            "save_loss_to_tracker",
            staticmethod(lambda **kwargs: None),
        )
        indexer = harness.attention.indexer
        fake_before_topk = indexer.forward_before_topk

        def before_topk_with_grad(x, qr, packed_seq_params):
            q, k, weights = fake_before_topk(x, qr, packed_seq_params)
            harness.inputs["k_indexer"] = k.requires_grad_()
            return q, k, weights

        monkeypatch.setattr(indexer, "forward_before_topk", before_topk_with_grad)

    local_rows = boundaries[-1] // cp_size + pad_rows
    max_seqlen = max(end - start for start, end in zip(boundaries[:-1], boundaries[1:]))
    grads, inverses = {}, {}
    for path in ("lazy", "prebuilt"):
        params = _routes_only(boundaries, harness.cp_group, max_seqlen=max_seqlen)
        layout = None
        if path == "prebuilt":
            layout = prebuild_packed_cp_layout(params, local_rows=local_rows)
        records.clear()
        harness.builder_calls.clear()
        output = harness.forward(params, local_rows, requires_grad=True)
        if layout is None:
            memo = getattr(params, DSAttention._LAYOUT_HOLDER_ATTR)
            ((_, reorder, inverse),) = (
                entry for key, entry in memo.items() if key[0] == "positions_and_reorder"
            )
            assert harness.builder_calls.count("_cp_reorder_from_host") == 1
        else:
            reorder, inverse = layout.kv_reorder_idx, layout.kv_reorder_inv_idx
            assert harness.builder_calls == []
            assert getattr(params, DSAttention._LAYOUT_HOLDER_ATTR) == {}
        # Key, value (and indexer key): each reorder gets this path's very tables.
        assert [record[0] for record in records] == ["forward"] * len(names)
        assert all(perm is reorder and inv is inverse for _, perm, inv in records)

        records.clear()
        output.sum().backward()
        assert [record[0] for record in records] == ["backward"] * len(names)
        for _, saved in records:
            assert saved.data_ptr() == inverse.data_ptr()
            assert torch.equal(saved, inverse)

        # Autograd's generic index_select backward with the same incoming gradient, after
        # the same fake gathers: over TP for K and V with SP, then over CP.
        assert sorted(harness.inputs) == sorted(names)
        for name, leaf in harness.inputs.items():
            reference_leaf = leaf.detach().requires_grad_()
            gathered = reference_leaf
            if tp_size > 1:
                gathered = torch.cat([gathered] * tp_size, dim=0)
            gathered = torch.cat([gathered] * cp_size, dim=0).index_select(0, reorder)
            assert torch.equal(gathered.detach(), live[name].detach())
            (reference_grad,) = torch.autograd.grad(gathered, reference_leaf, live[name].grad)
            assert torch.equal(leaf.grad, reference_grad)
        grads[path] = {name: leaf.grad for name, leaf in harness.inputs.items()}
        inverses[path] = inverse

    assert torch.equal(inverses["prebuilt"], inverses["lazy"])
    for name, grad in grads["prebuilt"].items():
        assert torch.equal(grad, grads["lazy"][name])


def test_refresh_through_packed_seq_params_feeds_the_next_forward(
    _single_process_group, monkeypatch
):
    """A static PackedSeqParams refreshed in place serves a different packing."""
    harness = _ForwardHarness(monkeypatch, 2, 1)
    group = harness.cp_group
    static_boundaries = [0, 8, 24, 24]
    static_params = _make_packed_seq_params(static_boundaries, static_boundaries, 16, 16)
    _prepare(monkeypatch, static_params, group)
    static_layout = get_prebuilt_packed_cp_layout(static_params)
    static_tables = static_layout.named_tables()
    # A memo entry for this geometry (e.g. from an eager fallback) describes pack A.
    memo = harness.attention._get_packed_cp_layout_cache(static_params)
    memo["stale"] = object()

    new_boundaries = [0, 16, 20, 24]
    new_params = _make_packed_seq_params(new_boundaries, new_boundaries, 16, 16)
    with pytest.raises(RuntimeError):
        copy_prebuilt_packed_cp_layout_(static_params, new_params)
    _prepare(monkeypatch, new_params, group)

    # What a graph integration does before replay: refresh cu_seqlens and the DSA state.
    static_params.cu_seqlens_q.copy_(new_params.cu_seqlens_q)
    assert copy_prebuilt_packed_cp_layout_(static_params, new_params) is static_layout
    assert get_prebuilt_packed_cp_layout(static_params) is static_layout
    for name, table in static_layout.named_tables().items():
        assert table is static_tables[name]
    for name in ("thd_cp_host_cu_seqlens_q", "thd_cp_host_cu_seqlens_kv"):
        assert getattr(static_params, name) is getattr(new_params, name)
    assert not hasattr(static_params, DSAttention._LAYOUT_HOLDER_ATTR)

    harness.builder_calls.clear()
    refreshed_output = harness.forward(static_params, 12)
    refreshed_seen = dict(harness.seen)
    assert harness.builder_calls == []

    lazy_params = _make_packed_seq_params(new_boundaries, new_boundaries, 16, 16)
    lazy_params.cp_group = group
    prebuild_thd_cp_partition_routes(lazy_params, group)
    lazy_output = harness.forward(lazy_params, 12)
    assert torch.equal(refreshed_output, lazy_output)
    assert torch.equal(refreshed_seen["key"], harness.seen["key"])
    assert torch.equal(
        refreshed_seen["packed_query_positions"], harness.seen["packed_query_positions"]
    )

    # A refreshed static PackedSeqParams also falls back correctly: its lazy path now
    # builds from the new pack's host boundaries instead of serving pack A's memo.
    dsa_layout.attach_prebuilt_packed_cp_layout(static_params, None)
    harness.builder_calls.clear()
    assert torch.equal(harness.forward(static_params, 12), lazy_output)
    assert harness.builder_calls == ["_cp_reorder_from_host"]


def test_dynamic_cp_tables_follow_the_realized_group(_single_process_group, monkeypatch):
    """Under dynamic CP each microbatch's tables are built for its own realized group."""
    harness = _ForwardHarness(monkeypatch, 2, 0)
    microbatches = [
        ([0, 16, 48], _FakeGroup(2, 1)),
        ([0, 64], _FakeGroup(4, 3)),
        ([0, 32, 32, 96], _FakeGroup(8, 5)),
        ([0, 16], _FakeGroup(1, 0)),
    ]
    layouts = []
    for boundaries, group in microbatches:
        params = _make_packed_seq_params(boundaries, boundaries, boundaries[-1], boundaries[-1])
        params.local_cp_size = group.size()
        _prepare(monkeypatch, params, group)
        layout = get_prebuilt_packed_cp_layout(params)
        if group.size() == 1:
            assert layout is None
            continue
        local_rows = boundaries[-1] // group.size()
        assert layout.geometry[:3] == (group.size(), group.rank(), local_rows)
        positions, reorder, inverse, _ = _memo_path_tables(
            _compact(boundaries),
            _compact(boundaries),
            group.size(),
            group.rank(),
            local_rows,
            *layout.geometry[3:],
        )
        assert torch.equal(layout.query_positions, positions)
        assert torch.equal(layout.kv_reorder_idx, reorder)
        assert torch.equal(layout.kv_reorder_inv_idx, inverse)

        # The attention module switches to the microbatch's group (absorbed MLA does
        # this for dynamic CP); DSA then consumes the matching tables.
        harness.attention.pg_collection.cp = group
        harness.builder_calls.clear()
        harness.forward(params, local_rows)
        assert harness.builder_calls == []
        layouts.append(layout)

    with pytest.raises(ValueError, match="across geometries"):
        layouts[0].copy_(layouts[1])


def test_prepare_packed_seq_params_prebuilds_only_for_dsa(monkeypatch):
    group = _FakeGroup(2, 0)

    def fresh():
        return _make_packed_seq_params([0, 8, 24], [0, 8, 24], 16, 16)

    params = _prepare(monkeypatch, fresh(), group, config=SimpleNamespace())
    assert get_prebuilt_packed_cp_layout(params) is None

    params = _prepare(monkeypatch, fresh(), group)
    layout = get_prebuilt_packed_cp_layout(params)
    assert layout.geometry == PackedCPLayoutGeometry(2, 0, 12, False, False)

    # Raw middle-stage boundaries: the physical rows are unknown, so DSA stays lazy.
    params = _prepare(monkeypatch, fresh(), group, capacity=32)
    assert get_prebuilt_packed_cp_layout(params) is None
    params = _prepare(monkeypatch, fresh(), group, capacity=24)
    assert get_prebuilt_packed_cp_layout(params) is not None

    params = _prepare(monkeypatch, fresh(), _FakeGroup(1, 0))
    assert get_prebuilt_packed_cp_layout(params) is None


def test_prebuild_requires_routes_and_detaches_stale_layouts():
    params = _make_packed_seq_params([0, 8, 24], [0, 8, 24], 16, 16)
    params.cp_group = _FakeGroup(2, 0)
    assert prebuild_packed_cp_layout(params) is None  # no host boundaries recorded

    prebuild_thd_cp_partition_routes(params, params.cp_group)
    layout = prebuild_packed_cp_layout(params)
    assert get_prebuilt_packed_cp_layout(params) is layout
    assert layout.device == _CPU

    params.cp_group = _FakeGroup(1, 0)
    assert prebuild_packed_cp_layout(params) is None
    assert get_prebuilt_packed_cp_layout(params) is None
    assert prebuild_packed_cp_layout(None) is None
    assert prebuild_packed_cp_layout(PackedSeqParams(qkv_format="sbhd")) is None


def test_layout_validation_rejects_inconsistent_tables():
    geometry = PackedCPLayoutGeometry(2, 0, 4, False, False)
    good = build_packed_cp_layout_from_host([0, 8], [0, 8], geometry, _CPU)
    bad_tables = [
        {"query_positions": good.query_positions.to(torch.int32)},
        {"kv_reorder_idx": good.kv_reorder_idx[:-1]},
        {"query_positions": good.query_positions.view(2, 2)},
        {"kv_reorder_idx": torch.arange(16, dtype=torch.int64)[::2]},
        {"kv_reorder_inv_idx": good.kv_reorder_inv_idx[:-1]},
        {"kv_reorder_inv_idx": good.kv_reorder_inv_idx.to(torch.int32)},
        {"kv_reorder_inv_idx": good.kv_reorder_inv_idx.to("meta")},
    ]
    for overrides in bad_tables:
        tables = {**good.named_tables(), **overrides}
        with pytest.raises(RuntimeError, match="inconsistent"):
            PrebuiltPackedCPLayout(geometry=geometry, **tables)
    # The inverse is required: M4's memo entry, which the layout replaces, always has it.
    with pytest.raises(TypeError):
        PrebuiltPackedCPLayout(
            geometry=geometry,
            query_positions=good.query_positions,
            kv_reorder_idx=good.kv_reorder_idx,
        )
    missing_inverse = build_packed_cp_layout_from_host([0, 8], [0, 8], geometry, _CPU)
    missing_inverse.kv_reorder_inv_idx = None
    with pytest.raises(RuntimeError, match="missing tables"):
        missing_inverse.validate()


def test_layout_validation_checks_the_inverse_of_host_tables():
    """A host layout whose inverse does not invert its reorder is rejected.

    Such a layout would give a correct forward but wrong K/V gradients. The check reads
    table values, so it covers host-resident tables only.
    """
    geometry = PackedCPLayoutGeometry(4, 2, 12, False, False)
    good = build_packed_cp_layout_from_host([0, 8, 48], [0, 8, 48], geometry, _CPU)
    reorder, inverse = good.kv_reorder_idx, good.kv_reorder_inv_idx
    assert not torch.equal(reorder, inverse)
    bad_inverses = {
        "reorder-as-inverse": reorder.clone(),
        "rolled": inverse.roll(1),
        "not-a-permutation": torch.zeros_like(inverse),
        "out-of-range": inverse + reorder.numel(),
        "negative": inverse - reorder.numel(),
    }
    for bad_inverse in bad_inverses.values():
        with pytest.raises(RuntimeError, match="does not invert"):
            PrebuiltPackedCPLayout(
                geometry=geometry,
                query_positions=good.query_positions,
                kv_reorder_idx=reorder,
                kv_reorder_inv_idx=bad_inverse,
            )
    # A partial refresh (one table rewritten outside copy_) is rejected as well, both by
    # DSA's match (validate) and by a later copy_.
    other = build_packed_cp_layout_from_host([0, 40, 48], [0, 16, 48], geometry, _CPU)
    assert not torch.equal(other.kv_reorder_idx, reorder)
    good.kv_reorder_idx.copy_(other.kv_reorder_idx)
    with pytest.raises(RuntimeError, match="does not invert"):
        good.validate()
    with pytest.raises(RuntimeError, match="does not invert"):
        other.copy_(good)
    # Empty tables are trivially consistent.
    empty = PackedCPLayoutGeometry(2, 0, 0, False, False)
    assert build_packed_cp_layout_from_host([0, 0], [0, 0], empty, _CPU).nbytes == 0
    with pytest.raises(RuntimeError, match="invalid CP geometry"):
        PrebuiltPackedCPLayout(geometry=geometry._replace(cp_size=1), **good.named_tables())
    with pytest.raises(TypeError):
        dsa_layout.attach_prebuilt_packed_cp_layout(PackedSeqParams(qkv_format="thd"), object())


def _simulate_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


def _routes_only(boundaries, group, max_seqlen=16):
    """Prepared metadata (routes + host boundaries) without a prebuilt DSA layout."""
    params = _make_packed_seq_params(boundaries, boundaries, max_seqlen, max_seqlen)
    params.cp_group = group
    prebuild_thd_cp_partition_routes(params, group)
    return params


@pytest.mark.parametrize("warm_memo", [False, True], ids=["empty-memo", "warm-memo"])
def test_dsa_without_prebuilt_layout_raises_during_graph_capture(
    _single_process_group, monkeypatch, warm_memo
):
    """A capture must bake neither a lazily built table nor an eager warmup's memo entry."""
    harness = _ForwardHarness(monkeypatch, 2, 1)
    params = _routes_only([0, 8, 24, 24], harness.cp_group)
    if warm_memo:
        harness.forward(params, 12)
        assert getattr(params, DSAttention._LAYOUT_HOLDER_ATTR)
    harness.builder_calls.clear()
    _simulate_capture(monkeypatch)
    with pytest.raises(RuntimeError, match="no current prebuilt layout"):
        harness.forward(params, 12)
    assert harness.builder_calls == []


def test_dsa_consumes_prebuilt_layout_during_graph_capture(_single_process_group, monkeypatch):
    harness, params, local_rows = _prebuilt_harness(monkeypatch, [0, 8, 24, 24])
    eager_output = harness.forward(params, local_rows)
    _simulate_capture(monkeypatch)
    harness.builder_calls.clear()
    captured_output = harness.forward(params, local_rows)
    assert harness.builder_calls == []
    assert torch.equal(captured_output, eager_output)


def test_rerouted_metadata_never_serves_a_stale_prebuilt_layout(_single_process_group, monkeypatch):
    """New boundaries recorded on the same object invalidate the attached layout."""
    harness = _ForwardHarness(monkeypatch, 2, 1)
    group = harness.cp_group
    old, new = [0, 8, 24], [0, 16, 24]  # same geometry: cp 2, rank 1, 12 rows

    fresh = _routes_only(new, group)
    expected_output = harness.forward(fresh, 12)
    expected_positions = harness.seen["packed_query_positions"]

    def reprepared(reroute):
        params = _make_packed_seq_params(old, old, 16, 16)
        _prepare(monkeypatch, params, group)
        stale = get_prebuilt_packed_cp_layout(params)
        assert stale is not None
        params.cu_seqlens_q.copy_(torch.tensor(new, dtype=torch.int32))
        reroute(params)
        assert get_prebuilt_packed_cp_layout(params) is None
        assert not torch.equal(stale.query_positions, expected_positions)
        return params

    # A raw-boundary (capacity) re-prepare detaches the layout it cannot rebuild ...
    params = reprepared(lambda p: _prepare(monkeypatch, p, group, capacity=32))
    assert not hasattr(params, dsa_layout._PREBUILT_PACKED_CP_LAYOUT_ATTR)
    # ... and re-recording the routes alone leaves it attached but stale.
    params = reprepared(lambda p: prebuild_thd_cp_partition_routes(p, group))

    harness.builder_calls.clear()
    assert torch.equal(harness.forward(params, 12), expected_output)
    assert torch.equal(harness.seen["packed_query_positions"], expected_positions)
    assert harness.builder_calls == ["_cp_reorder_from_host"]

    _simulate_capture(monkeypatch)
    with pytest.raises(RuntimeError, match="no current prebuilt layout"):
        harness.forward(_routes_only(new, group), 12)


def test_prebuild_is_idempotent_and_builds_nothing_during_capture(monkeypatch):
    builds = []
    real_build = dsa_layout.build_packed_cp_layout_from_host

    def counting_build(*args, **kwargs):
        builds.append(args[2])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(dsa_layout, "build_packed_cp_layout_from_host", counting_build)
    group = _FakeGroup(2, 1)
    params = _routes_only([0, 8, 24], group)
    layout = prebuild_packed_cp_layout(params)
    assert prebuild_packed_cp_layout(params) is layout
    # prepare_packed_seq_params on metadata whose routes are kept (a static slot that is
    # prepared again in every step) reuses the attached layout as well.
    monkeypatch.setattr(packed_seq_utils, "finalize_packed_seq_params", lambda p: p)
    packed_seq_utils.prepare_packed_seq_params(params, _dsa_config())
    assert get_prebuilt_packed_cp_layout(params) is layout
    assert len(builds) == 1
    # Another row count is another geometry: built again.
    assert prebuild_packed_cp_layout(params, local_rows=14).geometry.local_rows == 14
    assert len(builds) == 2
    layout = prebuild_packed_cp_layout(params)
    assert len(builds) == 3

    _simulate_capture(monkeypatch)
    assert prebuild_packed_cp_layout(params) is layout
    assert len(builds) == 3
    # Inside a capture nothing is built (no H2D copy from a transient staging buffer);
    # a layout that does not answer the request is detached, and DSA reports it.
    assert prebuild_packed_cp_layout(params, local_rows=14) is None
    assert get_prebuilt_packed_cp_layout(params) is None
    assert len(builds) == 3


def test_prebuild_is_idempotent_for_a_device_spelled_without_its_tensor_form(monkeypatch):
    """``device`` is compared as tables report it, so another spelling is not a rebuild.

    Tables report ``cuda:<current>`` for ``"cuda"`` and ``cpu`` for ``"cpu:0"``; comparing
    the raw request would rebuild on every call and, during a capture, detach the
    current layout and attach nothing.
    """
    builds = []
    real_build = dsa_layout.build_packed_cp_layout_from_host

    def counting_build(*args, **kwargs):
        builds.append(args[3])
        return real_build(*args, **kwargs)

    monkeypatch.setattr(dsa_layout, "build_packed_cp_layout_from_host", counting_build)
    params = _routes_only([0, 8, 24], _FakeGroup(2, 1))
    layout = prebuild_packed_cp_layout(params, device="cpu:0")
    assert layout.device == _CPU
    assert prebuild_packed_cp_layout(params, device="cpu:0") is layout
    assert prebuild_packed_cp_layout(params, device=_CPU) is layout
    assert prebuild_packed_cp_layout(params) is layout
    assert builds == [_CPU]

    _simulate_capture(monkeypatch)
    assert prebuild_packed_cp_layout(params, device="cpu:0") is layout
    assert get_prebuilt_packed_cp_layout(params) is layout
    assert len(builds) == 1

    # An index-less CUDA device means the current one, as for tensor creation.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert dsa_layout._table_device("cuda") == torch.device("cuda", 3)
    assert dsa_layout._table_device(torch.device("cuda", 1)) == torch.device("cuda", 1)
    assert dsa_layout._table_device("cpu:0") == _CPU
    assert dsa_layout._table_device("meta") == torch.device("meta")


def test_mismatch_warning_is_reported_once_per_kind(_single_process_group, monkeypatch, caplog):
    boundaries = [0, 8, 24]
    harness, params, local_rows = _prebuilt_harness(monkeypatch, boundaries)
    geometry = get_prebuilt_packed_cp_layout(params).geometry
    monkeypatch.setattr(dsa_module, "_DSA_PREBUILT_LAYOUT_MISMATCH_WARNED", set())
    with caplog.at_level(logging.WARNING, logger=dsa_module.logger.name):
        for other in (
            geometry._replace(cp_rank=0),
            geometry._replace(cp_rank=0),
            geometry._replace(local_rows=local_rows + 2),
            geometry._replace(local_rows=local_rows + 4),
        ):
            dsa_layout.attach_prebuilt_packed_cp_layout(
                params, build_packed_cp_layout_from_host(boundaries, boundaries, other, _CPU)
            )
            harness.forward(params, local_rows)
    warnings = [r.getMessage() for r in caplog.records if "prebuilt" in r.getMessage()]
    assert len(warnings) == 2, warnings
    assert all("twice" in message for message in warnings)


def test_resolve_packed_cp_layout_geometry():
    resolve = dsa_layout.resolve_packed_cp_layout_geometry
    params = _make_packed_seq_params([0, 8, 24], [0, 8, 24], 16, 16)
    assert resolve(params) is None  # no CP group
    params.cp_group = _FakeGroup(2, 1)
    assert resolve(params) is None  # rows unknown without host boundaries
    assert resolve(params, local_rows=14) == PackedCPLayoutGeometry(2, 1, 14, False, False)
    prebuild_thd_cp_partition_routes(params, params.cp_group)
    assert resolve(params) == PackedCPLayoutGeometry(2, 1, 12, False, False)
    # An explicit group, e.g. the layer's, for metadata rebuilt without one.
    assert resolve(params, cp_group=_FakeGroup(4, 3)) == PackedCPLayoutGeometry(
        4, 3, 6, False, False
    )
    assert resolve(params, cp_group=_FakeGroup(1, 0)) is None
    assert resolve(params, cp_group=_FakeGroup(5, 0)) is None  # 24 rows do not divide
    single = _routes_only([0, 24], _FakeGroup(2, 1), max_seqlen=24)
    assert resolve(single) == PackedCPLayoutGeometry(2, 1, 12, True, True)
    assert resolve(PackedSeqParams(qkv_format="sbhd")) is None
    assert resolve(None) is None
