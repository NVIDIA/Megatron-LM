# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the fixed-shape ``chunk_indices`` table used for THD CUDA graph capture.

The table is what makes fla's varlen kernels capturable: only its *shape* is a
capture-time constant, its values are recomputed on every replay. It must agree
element-for-element with fla's own host-built table, and it must preserve fla's
``row == chunk_offsets[seq] + intra`` invariant, which the ``h``/``dh`` reader
kernels rely on to index into the chunk-state buffer.
"""

import copy

import pytest
import torch

import megatron.core.ssm.gated_delta_net.common as common
from megatron.core.ssm.gated_delta_net.common import _FLA_CHUNK_SIZE, _GDNBase
from megatron.core.transformer.enums import CudaGraphModule

try:
    from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


class _Config:
    thd_max_packed_sequences = 8
    max_seqlen_per_dp_cp_rank = 4096
    context_parallel_size = 1


class _Stub:
    """Minimal stand-in exposing just the pure table builder off ``_GDNBase``."""

    _fixed_shape_chunk_indices = _GDNBase._fixed_shape_chunk_indices
    _chunk_slot_arange = _GDNBase._chunk_slot_arange

    def __init__(self):
        self.config = _Config()
        self._chunk_slot_cache = {}


def _make_cu_seqlens(lens, extra_tokens=0):
    """Build a statically-shaped cu_seqlens, padded with zero-length sequences."""
    cu = [0]
    for length in lens:
        cu.append(cu[-1] + length)
    cu[-1] += extra_tokens
    while len(cu) < _Config.thd_max_packed_sequences + 1:
        cu.append(cu[-1])
    return torch.tensor(cu, device="cuda", dtype=torch.int32)


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "lens",
    [
        [4096],
        [2048, 2048],
        [1024, 512, 1536, 1024],
        [64, 64, 3968],
        [100, 200, 300, 3496],  # every sequence ends mid-chunk
        [512] * 8,  # cu_seqlens fully occupied, no padding sequences
        [_FLA_CHUNK_SIZE * 7 + 1] * 8,  # worst case for the NT_max bound
    ],
)
@pytest.mark.parametrize("extra_tokens", [0, 16, 63, 127])
def test_matches_fla_prepare_chunk_indices(lens, extra_tokens):
    stub = _Stub()
    cu_seqlens = _make_cu_seqlens(lens, extra_tokens=extra_tokens)

    table = stub._fixed_shape_chunk_indices(cu_seqlens, extra_tokens=extra_tokens)
    reference = prepare_chunk_indices(cu_seqlens, _FLA_CHUNK_SIZE)
    num_real = reference.shape[0]

    # The shape is a fixed upper bound; under-sizing it would silently drop chunks,
    # since NT is the Triton grid dimension.
    assert table.shape[0] >= num_real
    assert table.dtype == cu_seqlens.dtype
    # Real rows are bit-identical to fla's own table, and stay contiguous from row 0.
    assert torch.equal(table[:num_real], reference)

    # fla's h/dh reader kernels index by row, the writer by chunk_offsets[seq] + intra.
    chunk_offsets = prepare_chunk_offsets(cu_seqlens, _FLA_CHUNK_SIZE)
    rows = torch.arange(num_real, device=cu_seqlens.device, dtype=torch.int64)
    assert torch.equal(chunk_offsets[table[:num_real, 0].long()] + table[:num_real, 1], rows)

    # Trailing pad rows point at sequence 0 with an out-of-range intra index, so every
    # masked load/store in fla's kernels (all gated on o_t < T) no-ops.
    assert torch.all(table[num_real:, 0] == 0)
    assert torch.all(table[num_real:, 1] >= table.shape[0])


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_shape_is_constant_across_packings():
    """The whole point: shape frozen at capture, values recomputed on replay."""
    stub = _Stub()
    shapes = {
        stub._fixed_shape_chunk_indices(_make_cu_seqlens(lens)).shape
        for lens in ([4096], [2048, 2048], [100, 200, 300, 3496], [512] * 8)
    }
    assert len(shapes) == 1


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_returns_none_without_static_bounds():
    """Eager fallback: without the static bounds the caller lets fla build the table."""
    stub = _Stub()
    stub.config.thd_max_packed_sequences = None
    assert stub._fixed_shape_chunk_indices(_make_cu_seqlens([2048, 2048])) is None


class _KwargsStub(_Stub):
    _fla_varlen_kwargs = _GDNBase._fla_varlen_kwargs
    _cu_seqlens_cpu_mirror = _GDNBase._cu_seqlens_cpu_mirror

    def __init__(self):
        super().__init__()
        self._cu_seqlens_cpu_cache = None


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_varlen_kwargs_selects_static_table_or_cpu_mirror():
    stub = _KwargsStub()
    cu_seqlens = _make_cu_seqlens([2048, 2048])

    # Static bounds configured: the capture-safe table, and no redundant CPU mirror
    # (fla ignores cu_seqlens_cpu whenever chunk_indices is supplied).
    kwargs = stub._fla_varlen_kwargs(cu_seqlens)
    assert set(kwargs) == {"chunk_indices"}

    # Without them, fall back to the eager-only CPU mirror.
    stub.config.thd_max_packed_sequences = None
    kwargs = stub._fla_varlen_kwargs(cu_seqlens)
    assert set(kwargs) == {"cu_seqlens_cpu"}
    assert torch.equal(kwargs["cu_seqlens_cpu"], cu_seqlens.cpu())

    # SBHD supplies no cu_seqlens at all; fla never builds the table.
    assert stub._fla_varlen_kwargs(None) == {}


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_conv_padding_reuses_the_unpadded_mirror():
    """The padded mirror is host arithmetic on a cache hit, not a second D2H sync."""
    stub = _KwargsStub()
    stub.config.thd_max_packed_sequences = None  # force the cu_seqlens_cpu fallback
    unpadded = _make_cu_seqlens([1024, 512, 1536, 1024])
    padded = unpadded.clone()
    padded[-1] += 64

    conv = stub._fla_varlen_kwargs(padded, extra_tokens=64, cu_seqlens_unpadded=unpadded)
    # The cache now holds the *unpadded* buffer, so the second call site hits it.
    assert stub._cu_seqlens_cpu_cache[0] is unpadded
    gdr = stub._fla_varlen_kwargs(unpadded)
    assert stub._cu_seqlens_cpu_cache[0] is unpadded

    assert torch.equal(conv["cu_seqlens_cpu"], padded.cpu())
    assert torch.equal(gdr["cu_seqlens_cpu"], unpadded.cpu())
    # The cached mirror must not have been mutated by the padded variant.
    assert gdr["cu_seqlens_cpu"] is stub._cu_seqlens_cpu_cache[1]

    with pytest.raises(AssertionError):
        stub._fla_varlen_kwargs(padded, extra_tokens=64)


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_varlen_kwargs_defers_to_fla_under_chunkwise_cp():
    """fla overrides cu_seqlens with the CP-local partition but not chunk_indices.

    A table built here from the global cu_seqlens would describe the wrong tensor, so
    the CP path must hand the job back to fla entirely.
    """
    stub = _KwargsStub()
    cu_seqlens = _make_cu_seqlens([2048, 2048])
    assert stub._fla_varlen_kwargs(cu_seqlens, cp_context=object()) == {}


@pytest.mark.skipif(not HAVE_FLA, reason="fla is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_mirror_cache_holds_a_reference_to_its_source():
    """Keying on bare id() would let a freed tensor's address alias a new one."""
    stub = _KwargsStub()
    first = _make_cu_seqlens([2048, 2048])
    mirror = stub._cu_seqlens_cpu_mirror(first)
    assert stub._cu_seqlens_cpu_mirror(first) is mirror
    assert stub._cu_seqlens_cpu_cache[0] is first

    second = _make_cu_seqlens([1024, 512, 1536, 1024])
    assert not torch.equal(stub._cu_seqlens_cpu_mirror(second), mirror)


class _CGConfig:
    def __init__(
        self, impl="none", modules=None, packing_scheduler="dp_balanced", dynamic_cp=False
    ):
        self.cuda_graph_impl = impl
        self.cuda_graph_modules = modules
        self.sequence_packing_scheduler = packing_scheduler
        self.dynamic_context_parallel = dynamic_cp


@pytest.mark.parametrize(
    "impl,modules,should_raise",
    [
        ("none", None, False),
        ("none", [CudaGraphModule.attn], False),
        ("transformer_engine", [CudaGraphModule.attn], True),
        ("transformer_engine", [CudaGraphModule.moe_router], False),
        ("transformer_engine", None, True),  # unset == every module
        ("full_iteration", None, True),
        ("local", [CudaGraphModule.attn], True),
    ],
)
def test_tensor_cache_check_fires_only_when_thd_attention_is_captured(
    monkeypatch, impl, modules, should_raise
):
    """fla's @tensor_cache freezes chunk_offsets across replays; GDN must refuse capture.

    The failure it guards against is a silently diverging loss, so the check must be an
    error, and it must fire at construction rather than inside a live capture region.
    """
    monkeypatch.setattr(common, "FLA_DISABLE_TENSOR_CACHE", False)
    config = _CGConfig(impl, modules)
    if should_raise:
        with pytest.raises(RuntimeError, match="FLA_DISABLE_TENSOR_CACHE"):
            _GDNBase._check_fla_tensor_cache_disabled(config)
    else:
        _GDNBase._check_fla_tensor_cache_disabled(config)

    # With the cache disabled the check is a no-op for every configuration.
    monkeypatch.setattr(common, "FLA_DISABLE_TENSOR_CACHE", True)
    _GDNBase._check_fla_tensor_cache_disabled(config)


@pytest.mark.parametrize("impl", ["transformer_engine", "local", "full_iteration"])
def test_sbhd_capture_does_not_need_the_env_var(monkeypatch, impl):
    """SBHD capture was always safe and must not be blocked.

    With cu_seqlens None, fla takes the ``chunk_offsets = None`` branch in
    ``chunk_delta_h.py`` and never calls the ``@tensor_cache``'d
    ``prepare_chunk_offsets``, so there is nothing for a stale cache to freeze and no
    ``chunk_indices`` table to desynchronize from. Requiring FLA_DISABLE_TENSOR_CACHE
    here would break configurations that work today.
    """
    monkeypatch.setattr(common, "FLA_DISABLE_TENSOR_CACHE", False)
    config = _CGConfig(impl, [CudaGraphModule.attn], packing_scheduler=None, dynamic_cp=False)
    _GDNBase._check_fla_tensor_cache_disabled(config)


@pytest.mark.parametrize(
    "packing_scheduler,dynamic_cp",
    [("dp_balanced", False), ("default_dynamic_cp", False), (None, True)],
)
def test_thd_is_detected_by_either_packing_signal(monkeypatch, packing_scheduler, dynamic_cp):
    """Mirrors TransformerConfig.__post_init__'s THD-CUDA-graph predicate."""
    monkeypatch.setattr(common, "FLA_DISABLE_TENSOR_CACHE", False)
    config = _CGConfig(
        "transformer_engine",
        [CudaGraphModule.attn],
        packing_scheduler=packing_scheduler,
        dynamic_cp=dynamic_cp,
    )
    with pytest.raises(RuntimeError, match="FLA_DISABLE_TENSOR_CACHE"):
        _GDNBase._check_fla_tensor_cache_disabled(config)
