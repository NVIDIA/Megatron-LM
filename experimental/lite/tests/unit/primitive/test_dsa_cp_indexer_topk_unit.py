# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""DSA native context parallelism selecting through indexer top-k bindings (CPU).

A bound layer that selects a top-k builds no dense [local query, global key] causal mask: its
rank's contiguous rows are described by a ``QueryLayout`` (one dense sequence, or the packed
sequences of ``cu_seqlens`` with ids into the gathered keys) and selected by the binding. A
binding that does not select (inactive) gets the mask built with the caller's expression, and
unbound layers, IndexShare shared layers, training forwards (also without autograd, as Lite's
reentrant activation recompute runs them) and forwards with autograd run the upstream code.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.mlite

# The upstream statements, verbatim.
_UPSTREAM_SELECT = """
_scores, topk_indices = _index_scores_and_topk(
    q_indexer,
    k_indexer,
    weights_indexer,
    mask=mask,
    topk=self.index_topk,
    scale=self.indexer_softmax_scale,
)
"""
_UPSTREAM_DENSE_MASK = """
mask = _build_cp_causal_mask(
    query_pos,
    torch.arange(kv.shape[0], device=x.device),
)
"""
_UPSTREAM_PACKED_MASK = """
mask = _build_cp_causal_mask(
    query_pos,
    key_pos,
    cu_seqlens=cu_seqlens,
)
"""
_UPSTREAM_PACKED_KEY_POS = """
key_pos = torch.arange(kv.shape[0], device=x.device, dtype=torch.long)
"""


@pytest.fixture(autouse=True)
def _te_import_stub(transformer_engine_import_stub):
    transformer_engine_import_stub()


def _dsa():
    from megatron.lite.primitive.modules.attention import dsa

    return dsa


def _assignments(function, callee: str) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return [
        ast.dump(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == callee
    ]


def _dump(statement: str) -> str:
    return ast.dump(ast.parse(textwrap.dedent(statement)).body[0])


class FakeBinding:
    """Records what a layer asks of its binding; ``select`` returns ``result(call)``."""

    def __init__(self, result=None, *, required=False, active=True):
        self.result = result
        self.required = required
        self.is_active = active
        self.selects: list[dict] = []
        self.declines: list[str] = []

    def active(self) -> bool:
        return self.is_active and not torch.is_grad_enabled()

    def decline(self, reason: str) -> None:
        self.declines.append(reason)
        if self.required:
            raise RuntimeError(f"required binding declined: {reason}")

    def select(self, q, k, weights, *, layout, topk, softmax_scale):
        call = dict(
            q=q, k=k, weights=weights, layout=layout, topk=topk, softmax_scale=softmax_scale
        )
        self.selects.append(call)
        return None if self.result is None else self.result(call)


def _oracle(call: dict) -> torch.Tensor:
    """The exact top-k of every row of a call, as a binding returns it (float64 scores)."""
    layout, topk = call["layout"], call["topk"]
    q, k, weights = call["q"].double(), call["k"].double(), call["weights"].double()
    out = torch.full((layout.rows, topk), -1, dtype=torch.int32)
    for segment in layout.segments:
        keys = k[segment.key_start : segment.key_start + segment.key_count]
        for row in range(segment.row_start, segment.row_end):
            visible = layout.visible_keys(segment, row)
            if not visible:
                continue
            scores = torch.relu(q[row] @ keys[:visible].T) * weights[row][:, None]
            scores = scores.sum(dim=0) * call["softmax_scale"]
            chosen = scores.topk(min(topk, visible)).indices.sort().values
            out[row, : chosen.numel()] = (chosen + segment.index_base).to(torch.int32)
    return out


# ---------------------------------------------------------------------------
# The CP forwards with test doubles (as test_dsa_cp_native_unit.py)
# ---------------------------------------------------------------------------


class _CudaLike(torch.Tensor):
    """A CPU tensor that reports ``is_cuda``, so the CP forwards pad the gathered keys."""

    @property
    def is_cuda(self):
        return True


def _fake_attention(*, local: int, cp_rank: int = 1, skip_topk: bool = False, **extra):
    """A CP=2 layer double: local rows project to fixed shapes, gathers duplicate them."""
    record = SimpleNamespace(sparse=[], gathers=0)

    def project_inputs(x, cos, sin, position_ids):
        del x, cos, sin, position_ids
        return (
            torch.randn(local, 1, 2, 8),
            torch.randn(local, 1, 8),
            torch.randn(2, 2, 4),
            None if skip_topk else torch.randn(local, 1, 2, 4),
            None if skip_topk else torch.randn(local, 1, 4),
            None if skip_topk else torch.randn(local, 1, 2),
        )

    def gather_projected(tensor, reorder, *, contiguous=False):
        del contiguous
        record.gathers += 1
        return torch.cat([tensor, tensor], dim=0).index_select(0, reorder)

    def run_sparse(query, kv, q_idx, k_idx, weights, mask, **kwargs):
        record.sparse.append(dict(kv=kv, q_idx=q_idx, k_idx=k_idx, mask=mask, **kwargs))
        return torch.randn(local, 1, 8)

    fake = SimpleNamespace(
        cp_size=2,
        cp_rank=cp_rank,
        skip_topk=skip_topk,
        _project_cp_inputs=project_inputs,
        _gather_projected_cp=gather_projected,
        _run_cp_sparse_segment=run_sparse,
        _project_cp_output=lambda out, weight: out,
        _packed_cu_seqlens=lambda params, device: params.cu_seqlens_q.to(device),
    )
    for name, value in extra.items():
        setattr(fake, name, value)
    return fake, record


def _run_dense(fake, *, batch: int = 1, local: int = 4):
    return _dsa().DynamicSparseAttention._forward_dense_cp_native(
        fake,
        torch.randn(batch, local, 64),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        index_share_state=None,
    )


def _run_packed(fake, cu_seqlens: list[int]):
    params = SimpleNamespace(
        cp_layout="contiguous", cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32)
    )
    local = cu_seqlens[-1] // 2
    return _dsa().DynamicSparseAttention._forward_packed_cp_native(
        fake,
        torch.randn(1, local, 64),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        params,
        index_share_state=None,
    )


def test_dense_cp_skips_mask_when_bound(monkeypatch):
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    dsa = _dsa()
    build_mask = Mock(side_effect=AssertionError("no mask for a bound layer"))
    monkeypatch.setattr(dsa, "_build_cp_causal_mask", build_mask)
    binding = FakeBinding()
    fake, record = _fake_attention(local=4, _indexer_topk=binding)
    with torch.no_grad():
        _run_dense(fake)
    build_mask.assert_not_called()
    (sparse,) = record.sparse
    assert sparse["mask"] is None and record.gathers == 2
    query = sparse["cp_query"]
    assert query.binding is binding and query.cu_seqlens is None
    assert query.layout == QueryLayout.contiguous(4, position=4, keys=8)
    assert query.query_positions.tolist() == [4, 5, 6, 7]
    # The segment double does not select; the binding declined nothing.
    assert binding.selects == [] and binding.declines == []


def test_dense_cp_layout_uses_logical_rows(monkeypatch):
    """The layout sees the gathered keys before their 512-row alignment padding."""
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    dsa = _dsa()
    monkeypatch.setattr(dsa, "_build_cp_causal_mask", Mock(side_effect=AssertionError))
    fake, record = _fake_attention(local=260, cp_rank=0, _indexer_topk=FakeBinding())
    fake._gather_projected_cp = _cuda_like_gather(fake._gather_projected_cp)
    with torch.no_grad():
        _run_dense(fake, local=260)
    (sparse,) = record.sparse
    assert sparse["kv"].shape[0] == sparse["k_idx"].shape[0] == 1024  # padded
    assert sparse["cp_query"].layout == QueryLayout.contiguous(260, position=0, keys=520)


def _cuda_like_gather(gather):
    def wrapped(tensor, reorder, *, contiguous=False):
        return gather(tensor, reorder, contiguous=contiguous).as_subclass(_CudaLike)

    return wrapped


@pytest.mark.parametrize("cp_rank", [0, 1])
def test_packed_cp_layout_absolute_ids(monkeypatch, cp_rank):
    from megatron.lite.primitive.kernels.indexer_topk import QueryLayout

    dsa = _dsa()
    monkeypatch.setattr(dsa, "_build_cp_causal_mask", Mock(side_effect=AssertionError))
    offsets = [0, 3, 8]
    reads = []
    tolist = torch.Tensor.tolist

    def counting_tolist(tensor):
        reads.append(tuple(tensor.shape))
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counting_tolist)
    fake, record = _fake_attention(local=4, cp_rank=cp_rank, _indexer_topk=FakeBinding())
    with torch.no_grad():
        _run_packed(fake, offsets)
    assert reads == [(3,)]  # the offsets are read once
    (sparse,) = record.sparse
    query = sparse["cp_query"]
    assert sparse["mask"] is None
    assert torch.equal(query.cu_seqlens, torch.tensor(offsets, dtype=torch.int32))
    expected = QueryLayout.packed(offsets, row_start=4 * cp_rank, rows=4, absolute_ids=True)
    assert query.layout == expected
    # Absolute ids: a sequence's ids start at its first packed token.
    assert [segment.index_base for segment in expected.segments] == ([0, 3], [3])[cp_rank]


def test_unbound_cp_path_unchanged(monkeypatch):
    dsa = _dsa()
    layer = dsa.DynamicSparseAttention
    assert _assignments(layer._run_cp_sparse_segment, "_index_scores_and_topk") == [
        _dump(_UPSTREAM_SELECT)
    ]
    assert _assignments(layer._forward_dense_cp_native, "_build_cp_causal_mask") == [
        _dump(_UPSTREAM_DENSE_MASK)
    ]
    assert _assignments(layer._forward_packed_cp_native, "_build_cp_causal_mask") == [
        _dump(_UPSTREAM_PACKED_MASK)
    ]
    assert _assignments(layer._forward_packed_cp_native, "torch.arange") == [
        _dump(_UPSTREAM_PACKED_KEY_POS)
    ]
    mask = torch.zeros(4, 8)
    build_mask = Mock(return_value=mask)
    monkeypatch.setattr(dsa, "_build_cp_causal_mask", build_mask)
    # No binding, an inactive binding, and an active binding while autograd is enabled.
    for binding, grad in ((None, False), (FakeBinding(active=False), False), (FakeBinding(), True)):
        fake, record = _fake_attention(local=4, _indexer_topk=binding)
        with torch.set_grad_enabled(grad):
            _run_dense(fake)
        (sparse,) = record.sparse
        assert sparse["mask"] is mask and sparse["cp_query"] is None
        query_pos, key_pos = build_mask.call_args.args
        assert query_pos.tolist() == [4, 5, 6, 7] and key_pos.tolist() == list(range(8))
        assert binding is None or binding.selects == binding.declines == []


def test_cp_batch_above_one_declines(monkeypatch):
    dsa = _dsa()
    mask = torch.zeros(4, 8)
    monkeypatch.setattr(dsa, "_build_cp_causal_mask", Mock(return_value=mask))
    binding = FakeBinding()
    fake, record = _fake_attention(local=4, _indexer_topk=binding)
    with torch.no_grad():
        _run_dense(fake, batch=2)
    assert binding.declines == ["batch>1"]
    assert record.sparse[0]["mask"] is mask and record.sparse[0]["cp_query"] is None
    fake, _ = _fake_attention(local=4, _indexer_topk=FakeBinding(required=True))
    with torch.no_grad(), pytest.raises(RuntimeError, match="batch>1"):
        _run_dense(fake, batch=2)


def test_cp_shared_layer_never_selects(monkeypatch):
    dsa = _dsa()
    build_mask = Mock(side_effect=AssertionError("shared layers build no mask"))
    monkeypatch.setattr(dsa, "_build_cp_causal_mask", build_mask)
    binding = FakeBinding()  # set directly: set_indexer_topk refuses shared layers
    fake, record = _fake_attention(local=4, skip_topk=True, _indexer_topk=binding)
    with torch.no_grad():
        _run_dense(fake)
    assert record.sparse[0]["mask"] is None and record.sparse[0]["cp_query"] is None
    assert binding.selects == binding.declines == []


# ---------------------------------------------------------------------------
# A real layer on the CPU (torch RMSNorm), the other rank's keys simulated
# ---------------------------------------------------------------------------

KWARGS = dict(
    hidden_size=32,
    num_attention_heads=2,
    q_lora_rank=16,
    kv_lora_rank=8,
    qk_nope_head_dim=8,
    qk_rope_head_dim=8,
    v_head_dim=8,
    index_n_heads=4,
    index_head_dim=16,
    index_topk=4,
    rms_norm_eps=1e-5,
)


class CpLayer:
    """Rank ``cp_rank`` of a CP=2 DSA layer; the gathers return both ranks' projections."""

    def __init__(self, monkeypatch, *, cp_rank: int, tokens: int = 16):
        dsa = _dsa()
        monkeypatch.setattr(dsa, "RMSNorm", lambda size, eps: nn.RMSNorm(size, eps=eps))
        torch.manual_seed(0)
        self.layer = dsa.DynamicSparseAttention(**KWARGS, cp_size=2, cp_rank=cp_rank).eval()
        self.cp_rank, self.tokens, self.local = cp_rank, tokens, tokens // 2
        generator = torch.Generator().manual_seed(1)
        self.x = torch.randn(1, tokens, KWARGS["hidden_size"], generator=generator)
        self.cos, self.sin = dsa.build_rope_cache(
            dim=KWARGS["qk_rope_head_dim"], max_position_embeddings=tokens, rope_theta=1e4
        )
        self.selected: list[torch.Tensor] = []
        self.masks: list[tuple] = []
        self.losses: list[dict] = []
        with torch.no_grad():
            projections = [self._project(rank) for rank in range(2)]
        # Gathered in rank order: kv first, then the indexer keys.
        self.gathered = [torch.cat([p[i] for p in projections]) for i in (1, 4)]
        kernels = dsa._dsa_kernels
        build_flat = kernels.build_flat_topk_idxs
        build_mask = dsa._build_cp_causal_mask

        def flat_topk(topk_indices, **kwargs):
            self.selected.append(topk_indices)
            return build_flat(topk_indices, **kwargs)

        def mask(*args, **kwargs):
            self.masks.append((args, kwargs))
            return build_mask(*args, **kwargs)

        def sparse(query, kv, sink, idxs, scale, topk_length=None, value_dim=None):
            return query.new_zeros(query.shape[0], query.shape[1], query.shape[2] * value_dim)

        def loss(*args, mask, **kwargs):
            self.losses.append(dict(mask=mask))
            return torch.zeros(())

        monkeypatch.setattr(kernels, "build_flat_topk_idxs", flat_topk)
        monkeypatch.setattr(kernels, "dsa_sparse_attn", sparse)
        monkeypatch.setattr(dsa, "_build_cp_causal_mask", mask)
        monkeypatch.setattr(dsa, "_cp_indexer_loss", loss)
        calls = iter(range(10**6))
        self.layer._gather_projected_cp = lambda tensor, reorder, contiguous=False: (
            self.gathered[next(calls) % 2]
        )

    def _project(self, rank: int):
        rows = slice(rank * self.local, (rank + 1) * self.local)
        positions = torch.arange(self.tokens)[rows].unsqueeze(0)
        return self.layer._project_cp_inputs(self.x[:, rows], self.cos, self.sin, positions)

    def forward(self, packed: list[int] | None = None, *, requires_grad: bool = False):
        """The layer's ``forward`` (the native CP path) on this rank's rows."""
        rows = slice(self.cp_rank * self.local, (self.cp_rank + 1) * self.local)
        positions = torch.arange(self.tokens)[rows].unsqueeze(0)
        params = None
        if packed is not None:
            params = SimpleNamespace(
                cp_layout="contiguous", cu_seqlens_q=torch.tensor(packed, dtype=torch.int32)
            )
        x = self.x[:, rows].detach().requires_grad_(requires_grad)
        return self.layer(
            x, cos=self.cos, sin=self.sin, position_ids=positions, packed_seq_params=params
        )

    def run(self, packed: list[int] | None = None):
        rows = slice(self.cp_rank * self.local, (self.cp_rank + 1) * self.local)
        positions = torch.arange(self.tokens)[rows].unsqueeze(0)
        if packed is None:
            return self.layer._forward_dense_cp_native(
                self.x[:, rows], self.cos, self.sin, positions, index_share_state=None
            )
        params = SimpleNamespace(
            cp_layout="contiguous", cu_seqlens_q=torch.tensor(packed, dtype=torch.int32)
        )
        return self.layer._forward_packed_cp_native(
            self.x[:, rows], self.cos, self.sin, positions, params, index_share_state=None
        )


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("packed", [None, [0, 5, 16], [0, 9, 11, 16]])
def test_bound_cp_selection_equals_upstream(monkeypatch, cp_rank, packed):
    """End to end: the binding's ids and visibility equal the upstream masked selector's."""
    cp = CpLayer(monkeypatch, cp_rank=cp_rank)
    with torch.no_grad():
        upstream_out = cp.run(packed)
    (upstream,) = cp.selected
    assert len(cp.masks) == 1
    binding = FakeBinding(_oracle)
    cp.layer.set_indexer_topk(binding)
    with torch.no_grad():
        bound_out = cp.run(packed)
    assert len(cp.masks) == 1  # no mask for the bound layer
    (call,) = binding.selects
    assert call["topk"] == KWARGS["index_topk"]
    assert call["softmax_scale"] == cp.layer.indexer_softmax_scale
    assert call["k"].shape[0] == cp.tokens  # every gathered key (no padding on the CPU)
    selected = cp.selected[1]
    assert selected.shape == upstream.shape == (1, cp.local, KWARGS["index_topk"])
    assert torch.equal(selected.sort(dim=-1).values, upstream.sort(dim=-1).values)
    assert bound_out.shape == upstream_out.shape


def test_declined_selection_builds_identical_mask(monkeypatch):
    """A binding that does not select gets the mask of the unbound caller."""
    for packed in (None, [0, 5, 16]):
        cp = CpLayer(monkeypatch, cp_rank=1)
        with torch.no_grad():
            cp.run(packed)  # unbound: the caller builds the mask
            bound = FakeBinding(result=lambda call: None)  # selects nothing, like inactive
            cp.layer.set_indexer_topk(bound)
            cp.run(packed)
        assert len(bound.selects) == 1 and len(cp.masks) == 2
        (caller_args, caller_kwargs), (built_args, built_kwargs) = cp.masks
        assert all(torch.equal(a, b) for a, b in zip(caller_args, built_args, strict=True))
        assert caller_kwargs.keys() == built_kwargs.keys()
        for name in caller_kwargs:
            assert torch.equal(caller_kwargs[name], built_kwargs[name])
        assert built_args[1].dtype == caller_args[1].dtype and built_args[1].shape[0] == 16
        assert torch.equal(cp.selected[0], cp.selected[1])


def test_grad_enabled_cp_keeps_the_mask_and_the_loss_path(monkeypatch):
    """Training with autograd: the binding is inactive, the loss sees the caller's mask."""
    from megatron.lite.primitive.kernels.indexer_topk import IndexerTopKConfig
    from megatron.lite.primitive.modules.attention.indexer_topk import IndexerTopKBinding

    cp = CpLayer(monkeypatch, cp_rank=1)
    binding = IndexerTopKBinding(
        name="layer",
        geometry=cp.layer.indexer_geometry(),
        config=IndexerTopKConfig(backend="reference", precision="fast"),
        fmt="fp8",
        tuning=None,
        kernel=None,
        plugin=None,
        route=None,
    )
    monkeypatch.setattr(binding, "select", Mock(side_effect=AssertionError("inactive")))
    cp.layer.set_indexer_topk(binding)
    cp.layer.train()
    cp.run()
    assert len(cp.masks) == 1 and len(cp.losses) == 1
    assert cp.losses[0]["mask"] is not None
    assert cp.losses[0]["mask"].shape == (cp.local, cp.tokens)


@pytest.mark.parametrize("packed", [None, [0, 5, 16]])
def test_training_recompute_cp_never_selects(monkeypatch, packed):
    """Lite's reentrant recompute runs a training forward without autograd and recomputes it
    with gradients in the backward pass: both runs build the mask and select upstream."""
    from megatron.lite.primitive.recompute import wrap_checkpoint

    cp = CpLayer(monkeypatch, cp_rank=1)
    binding = FakeBinding(_oracle)
    cp.layer.set_indexer_topk(binding)
    cp.layer.train()
    wrap_checkpoint(cp.layer, preserve_rng_state=False)
    out = cp.forward(packed, requires_grad=True)
    # The checkpointed forward: the caller's mask and the upstream selector, no indexer loss.
    assert len(cp.masks) == 1 and cp.losses == [] and len(cp.selected) == 1
    out.sum().backward()
    # The recompute in the backward pass: the mask again, and the indexer loss reads it.
    assert len(cp.masks) == 2 and len(cp.losses) == 1 and len(cp.selected) == 2
    assert cp.losses[0]["mask"] is not None
    assert torch.equal(cp.selected[0], cp.selected[1])
    assert binding.selects == [] and binding.declines == []
    # The same layer in eval mode without autograd selects through the binding, without a mask.
    cp.layer.eval()
    with torch.no_grad():
        cp.forward(packed)
    assert len(binding.selects) == 1 and len(cp.masks) == 2
