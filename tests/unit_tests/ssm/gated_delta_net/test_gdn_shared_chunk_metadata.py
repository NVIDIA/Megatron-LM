# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for sharing the varlen ``chunk_indices`` table across the GDN layers of a stack.

The table depends only on ``cu_seqlens``, so one build per forward pass serves every GDN
layer. What must hold: the shared table is the one each layer would have built itself, it
is never reused for another ``cu_seqlens`` tensor or another forward pass, and a CUDA graph
replay sees the packing of the replayed step rather than the captured one.
"""

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net.common import (
    _FLA_CHUNK_SIZE,
    GDNSharedChunkMetadata,
    _GDNBase,
    get_shared_chunk_metadata,
)
from megatron.core.transformer.enums import CudaGraphModule

try:
    from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

MAX_PACKED_SEQUENCES = 8
MAX_SEQLEN = 4096


def _config(**overrides):
    config = SimpleNamespace(
        cuda_graph_thd_max_packed_sequences=MAX_PACKED_SEQUENCES,
        max_seqlen_per_dp_cp_rank=MAX_SEQLEN,
        context_parallel_size=1,
        cuda_graph_impl="none",
        cuda_graph_modules=[],
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


class _Layer(torch.nn.Module):
    """The chunk-metadata methods of ``_GDNBase`` without the rest of a GDN layer."""

    consumes_shared_chunk_metadata = True
    attach_shared_packed_seq_metadata = _GDNBase.attach_shared_packed_seq_metadata
    _attention_in_per_layer_cuda_graph = staticmethod(_GDNBase._attention_in_per_layer_cuda_graph)
    _fla_varlen_kwargs = _GDNBase._fla_varlen_kwargs
    _chunk_slot_arange = _GDNBase._chunk_slot_arange
    _cu_seqlens_cpu_mirror = _GDNBase._cu_seqlens_cpu_mirror

    def __init__(self, config=None, backend="cudnn", chunkwise_context_parallel=False):
        super().__init__()
        self.config = config if config is not None else _config()
        self.gdn_backend = backend
        self.chunkwise_context_parallel = chunkwise_context_parallel
        self._chunk_slot_cache = {}
        self._cu_seqlens_cpu_cache = None
        self.num_builds = 0

    def _fixed_shape_chunk_indices(self, *args, **kwargs):
        self.num_builds += 1
        return _GDNBase._fixed_shape_chunk_indices(self, *args, **kwargs)

    def varlen_table(self, packed_seq_params):
        """What a GDN layer forwards to its kernels for ``packed_seq_params``."""
        cu_seqlens = (
            packed_seq_params.cu_seqlens_q_padded
            if packed_seq_params.cu_seqlens_q_padded is not None
            else packed_seq_params.cu_seqlens_q
        )
        kwargs = self._fla_varlen_kwargs(
            cu_seqlens, shared_chunk_metadata=get_shared_chunk_metadata(packed_seq_params)
        )
        return kwargs["chunk_indices"]


def _make_cu_seqlens(lens):
    """Statically shaped cu_seqlens, padded with zero-length sequences."""
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    while len(cu) < MAX_PACKED_SEQUENCES + 1:
        cu.append(cu[-1])
    return torch.tensor(cu, device="cuda", dtype=torch.int32)


def _packed_seq_params(cu_seqlens, padded=None):
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
    )


PACKINGS = [[4096], [2048, 2048], [100, 200, 300, 3496], [64, 64, 3968], [512] * 8]


@pytest.mark.parametrize("lens", PACKINGS)
def test_layers_share_one_build_per_forward(lens):
    layers = [_Layer(), _Layer(), _Layer()]
    cu_seqlens = _make_cu_seqlens(lens)
    packed_seq_params = _packed_seq_params(cu_seqlens)
    reference = _Layer()._fixed_shape_chunk_indices(cu_seqlens)

    cache = {}
    tables = []
    for layer in layers:
        shared = layer.attach_shared_packed_seq_metadata(packed_seq_params, cache)
        # The caller's object can outlive the forward pass and must stay untouched.
        assert shared is not packed_seq_params
        assert get_shared_chunk_metadata(packed_seq_params) is None
        assert shared.cu_seqlens_q is cu_seqlens
        tables.append(layer.varlen_table(shared))

    assert [layer.num_builds for layer in layers] == [1, 0, 0]
    assert all(table is tables[0] for table in tables)
    assert torch.equal(tables[0], reference)

    # The next forward pass starts from an empty cache and builds again.
    shared = layers[1].attach_shared_packed_seq_metadata(packed_seq_params, {})
    assert layers[1].varlen_table(shared) is not tables[0]
    assert layers[1].num_builds == 1


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.parametrize("lens", PACKINGS)
def test_shared_table_matches_fla(lens):
    """Same contract as the per-layer table: fla's rows, and fla's chunk_offsets invariant."""
    layer = _Layer()
    cu_seqlens = _make_cu_seqlens(lens)
    shared = layer.attach_shared_packed_seq_metadata(_packed_seq_params(cu_seqlens), {})
    table = _Layer().varlen_table(shared)

    reference = prepare_chunk_indices(cu_seqlens, _FLA_CHUNK_SIZE)
    num_real = reference.shape[0]
    assert torch.equal(table[:num_real], reference)
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, _FLA_CHUNK_SIZE)
    rows = torch.arange(num_real, device=cu_seqlens.device, dtype=torch.int64)
    assert torch.equal(chunk_offsets[table[:num_real, 0].long()] + table[:num_real, 1], rows)


def test_refilled_cu_seqlens_is_rebuilt_in_the_next_forward():
    """Static THD packing refills one cu_seqlens buffer in place for every microbatch."""
    layers = [_Layer(), _Layer()]
    cu_seqlens = _make_cu_seqlens([4096])
    packed_seq_params = _packed_seq_params(cu_seqlens)

    for lens in ([4096], [100, 200, 300, 3496], [2048, 2048]):
        cu_seqlens.copy_(_make_cu_seqlens(lens))
        cache = {}
        expected = _Layer()._fixed_shape_chunk_indices(cu_seqlens)
        for layer in layers:
            shared = layer.attach_shared_packed_seq_metadata(packed_seq_params, cache)
            assert torch.equal(layer.varlen_table(shared), expected)


def test_table_is_only_used_for_its_own_cu_seqlens():
    layer = _Layer()
    cu_seqlens = _make_cu_seqlens([2048, 2048])
    other = _make_cu_seqlens([100, 200, 300, 3496])
    shared = layer.attach_shared_packed_seq_metadata(_packed_seq_params(cu_seqlens), {})
    metadata = get_shared_chunk_metadata(shared)
    assert isinstance(metadata, GDNSharedChunkMetadata) and metadata.cu_seqlens is cu_seqlens

    # Another tensor, even with equal contents, does not match: the check is by identity
    # because comparing contents would be a host sync.
    for cu in (other, cu_seqlens.clone()):
        table = layer._fla_varlen_kwargs(cu, shared_chunk_metadata=metadata)["chunk_indices"]
        assert table is not metadata.chunk_indices
        assert torch.equal(table, _Layer()._fixed_shape_chunk_indices(cu))

    # Conv alignment padding changes the table, so the unpadded one is not used.
    padded = cu_seqlens.clone()
    padded[-1] += 64
    kwargs = layer._fla_varlen_kwargs(
        padded, extra_tokens=64, cu_seqlens_unpadded=cu_seqlens, shared_chunk_metadata=metadata
    )
    assert kwargs["chunk_indices"] is not metadata.chunk_indices
    # Chunkwise CP hands the table back to fla whatever the stack attached.
    assert (
        layer._fla_varlen_kwargs(cu_seqlens, cp_context=object(), shared_chunk_metadata=metadata)
        == {}
    )


def test_padded_boundaries_take_precedence():
    layer = _Layer()
    cu_seqlens = _make_cu_seqlens([1000, 1000])
    padded = _make_cu_seqlens([1024, 1024])
    shared = layer.attach_shared_packed_seq_metadata(_packed_seq_params(cu_seqlens, padded), {})
    assert get_shared_chunk_metadata(shared).cu_seqlens is padded
    assert torch.equal(layer.varlen_table(shared), _Layer()._fixed_shape_chunk_indices(padded))


def test_rebuilt_packed_seq_params_drop_the_table():
    """dataclasses.replace and copies of the fields fall back to the per-layer build."""
    layer = _Layer()
    cu_seqlens = _make_cu_seqlens([2048, 2048])
    shared = layer.attach_shared_packed_seq_metadata(_packed_seq_params(cu_seqlens), {})
    assert get_shared_chunk_metadata(shared) is not None
    assert get_shared_chunk_metadata(copy.copy(shared)) is get_shared_chunk_metadata(shared)
    assert get_shared_chunk_metadata(dataclasses.replace(shared)) is None
    assert get_shared_chunk_metadata(None) is None


@pytest.mark.parametrize(
    "layer_kwargs, packed_kwargs",
    [
        (dict(backend="internal"), {}),
        (dict(backend="torch"), {}),
        (dict(chunkwise_context_parallel=True), {}),
        (dict(config=_config(cuda_graph_thd_max_packed_sequences=None)), {}),
        # Per-layer graphs that capture attention only take cu_seqlens as inputs.
        (dict(config=_config(cuda_graph_impl="transformer_engine")), {}),
        (
            dict(
                config=_config(
                    cuda_graph_impl="transformer_engine", cuda_graph_modules=[CudaGraphModule.attn]
                )
            ),
            {},
        ),
        (dict(config=_config(cuda_graph_impl="local")), {}),
        ({}, dict(qkv_format="sbhd")),
    ],
)
def test_not_shared(layer_kwargs, packed_kwargs):
    layer = _Layer(**layer_kwargs)
    packed_seq_params = _packed_seq_params(_make_cu_seqlens([2048, 2048]))
    for name, value in packed_kwargs.items():
        setattr(packed_seq_params, name, value)
    assert layer.attach_shared_packed_seq_metadata(packed_seq_params, {}) is packed_seq_params
    assert layer.attach_shared_packed_seq_metadata(None, {}) is None


def test_variant_without_varlen_kwargs_does_not_share():
    layer = _Layer()
    layer.consumes_shared_chunk_metadata = False
    packed_seq_params = _packed_seq_params(_make_cu_seqlens([4096]))
    assert layer.attach_shared_packed_seq_metadata(packed_seq_params, {}) is packed_seq_params
    assert layer.num_builds == 0


@pytest.mark.parametrize(
    "config",
    [
        _config(cuda_graph_impl="full_iteration"),
        # Per-layer graphs that leave attention eager.
        _config(
            cuda_graph_impl="transformer_engine", cuda_graph_modules=[CudaGraphModule.moe_router]
        ),
    ],
)
def test_shared_when_attention_is_not_in_a_per_layer_graph(config):
    layer = _Layer(config=config)
    packed_seq_params = _packed_seq_params(_make_cu_seqlens([2048, 2048]))
    shared = layer.attach_shared_packed_seq_metadata(packed_seq_params, {})
    assert get_shared_chunk_metadata(shared) is not None


def _stack_forward(layers, packed_seq_params):
    """What HybridStack does per forward pass: one cache, attach before every layer."""
    cache = {}
    tables = []
    for layer in layers:
        shared = layer.attach_shared_packed_seq_metadata(packed_seq_params, cache)
        tables.append(layer.varlen_table(shared))
    return tables


@pytest.mark.parametrize("cuda_graph_impl", ["full_iteration", "transformer_engine"])
def test_cuda_graph_replay_sees_current_packing(cuda_graph_impl):
    """Capture with one packing, replay with others.

    ``full_iteration`` captures the whole stack forward, so the shared build is in the
    graph. With per-layer graphs nothing is shared and every layer's own build is in its
    graph. Either way the replayed tables must follow the refilled ``cu_seqlens``.
    """
    config = _config(cuda_graph_impl=cuda_graph_impl)
    layers = [_Layer(config=config), _Layer(config=config), _Layer(config=config)]
    cu_seqlens = _make_cu_seqlens([4096])
    packed_seq_params = _packed_seq_params(cu_seqlens)
    # Warm up outside the capture, as the training loop does (fills the slot-axis cache).
    _stack_forward(layers, packed_seq_params)
    for layer in layers:
        layer.num_builds = 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tables = _stack_forward(layers, packed_seq_params)
        # Each "layer" consumes its table inside the graph.
        consumed = [table.clone() for table in tables]

    expected_builds = [1, 0, 0] if cuda_graph_impl == "full_iteration" else [1, 1, 1]
    assert [layer.num_builds for layer in layers] == expected_builds

    for lens in ([100, 200, 300, 3496], [2048, 2048], [512] * 8, [4096]):
        cu_seqlens.copy_(_make_cu_seqlens(lens))
        graph.replay()
        expected = _Layer()._fixed_shape_chunk_indices(cu_seqlens)
        for table in consumed:
            assert torch.equal(table, expected)
        if HAVE_FLA:
            # fla caches this helper by tensor identity unless FLA_DISABLE_TENSOR_CACHE=1,
            # and cu_seqlens is one buffer refilled in place: ask with a fresh tensor so
            # the reference is the table of the current contents.
            reference = prepare_chunk_indices(cu_seqlens.clone(), _FLA_CHUNK_SIZE)
            assert torch.equal(consumed[-1][: reference.shape[0]], reference)


def test_hybrid_stack_attaches_per_layer_with_one_cache():
    """HybridStack's hook finds the opted-in submodules and threads one cache through."""
    from megatron.core.models.hybrid.hybrid_block import HybridStack

    gdn_a, gdn_b = _Layer(), _Layer()
    stack = SimpleNamespace(
        layers=[
            torch.nn.Sequential(torch.nn.Identity(), gdn_a),
            torch.nn.Linear(2, 2),  # a layer without packed-sequence metadata to share
            torch.nn.Sequential(gdn_b),
        ],
        _shared_packed_seq_metadata_consumers=None,
    )
    packed_seq_params = _packed_seq_params(_make_cu_seqlens([100, 200, 300, 3496]))

    cache = {}
    per_layer = [
        HybridStack._attach_shared_packed_seq_metadata(stack, idx, packed_seq_params, cache)
        for idx in range(3)
    ]
    assert per_layer[1] is packed_seq_params
    assert get_shared_chunk_metadata(per_layer[0]) is get_shared_chunk_metadata(per_layer[2])
    assert (gdn_a.num_builds, gdn_b.num_builds) == (1, 0)
    assert gdn_b.varlen_table(per_layer[2]) is gdn_a.varlen_table(per_layer[0])
    assert HybridStack._attach_shared_packed_seq_metadata(stack, 0, None, cache) is None
