# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed chunkwise CP metadata lifetime and forward-routing regressions."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.gated_delta_net import gdn, kda
from megatron.core.ssm.gated_delta_net.common import _GDNBase
from megatron.core.ssm.gated_delta_net.packed_cp_metadata import _cpu_cu_seqlens, _packed_cp_context


class _Group:
    def __init__(self, size=2, rank=0):
        self._size = size
        self._rank = rank

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def _builder(calls):
    def build(*, cu_seqlens, cu_seqlens_cpu=None, group, conv1d_kernel_size):
        if cu_seqlens_cpu is None:
            cu_seqlens_cpu = cu_seqlens.cpu()
        assert cu_seqlens_cpu.device.type == 'cpu'
        torch.testing.assert_close(cu_seqlens_cpu, cu_seqlens.cpu(), rtol=0, atol=0)
        calls.append(cu_seqlens_cpu.clone())
        boundaries = cu_seqlens_cpu.tolist()
        part = boundaries[-1] // group.size()
        start, end = group.rank() * part, (group.rank() + 1) * part
        local = sorted({0, part, *(x - start for x in boundaries if start < x < end)})
        return SimpleNamespace(
            group=group,
            conv1d_kernel_size=conv1d_kernel_size,
            cu_seqlens_cpu=torch.tensor(local, dtype=torch.int32),
        )

    return build


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('lengths', [[4, 8, 4], [2, 6, 8], [16], [8, 8]])
def test_context_matches_fresh_partition_and_reuses_metadata(rank, lengths):
    offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    packed = PackedSeqParams()
    group = _Group(rank=rank)
    calls = []
    builder = _builder(calls)
    first = _packed_cp_context(packed, offsets, group, 4, builder)
    assert _packed_cp_context(packed, offsets, group, 4, builder) is first
    assert len(calls) == 1
    fresh = builder(
        cu_seqlens=offsets, cu_seqlens_cpu=offsets.clone(), group=group, conv1d_kernel_size=4
    )
    torch.testing.assert_close(first.cu_seqlens_cpu, fresh.cu_seqlens_cpu, rtol=0, atol=0)


@pytest.mark.parametrize('mutation', ['copy', 'view', 'replace', 'new_microbatch'])
def test_same_shape_different_boundaries_rebuild_context(mutation):
    offsets = torch.tensor([0, 4, 12, 16])
    packed = PackedSeqParams()
    group, calls = _Group(), []
    builder = _builder(calls)
    first = _packed_cp_context(packed, offsets, group, 4, builder)
    if mutation == 'replace':
        offsets = torch.tensor([0, 2, 8, 16])
    elif mutation == 'new_microbatch':
        packed = PackedSeqParams()
    elif mutation == 'view':
        offsets[1:3].copy_(torch.tensor([2, 8]))
    else:
        offsets.copy_(torch.tensor([0, 2, 8, 16]))
    second = _packed_cp_context(packed, offsets, group, 4, builder)
    assert second is not first
    assert len(calls) == 2
    if mutation != 'new_microbatch':
        assert calls[1].tolist() == [0, 2, 8, 16]
        assert first.cu_seqlens_cpu.tolist() == [0, 4, 8]


def test_group_and_convolution_width_have_separate_contexts():
    packed, offsets = PackedSeqParams(), torch.tensor([0, 4, 12, 16])
    group_a, group_b, calls = _Group(rank=0), _Group(rank=1), []
    builder = _builder(calls)
    first = _packed_cp_context(packed, offsets, group_a, 4, builder)
    other_rank = _packed_cp_context(packed, offsets, group_b, 4, builder)
    other_width = _packed_cp_context(packed, offsets, group_a, 3, builder)
    assert first is not other_rank and first is not other_width
    assert other_rank.group is group_b and other_width.conv1d_kernel_size == 3
    assert _packed_cp_context(packed, offsets, group_a, 4, builder) is first
    assert len(calls) == 3


def test_inference_tensors_are_not_cached():
    packed, group, calls = PackedSeqParams(), _Group(), []
    builder = _builder(calls)
    with torch.inference_mode():
        offsets = torch.tensor([0, 4, 12, 16])
        first = _packed_cp_context(packed, offsets, group, 4, builder)
        offsets.copy_(torch.tensor([0, 2, 8, 16]))
        second = _packed_cp_context(packed, offsets, group, 4, builder)
        assert first is not second
        assert calls[1].tolist() == [0, 2, 8, 16]


def test_metadata_cache_is_bounded():
    packed, group, calls = PackedSeqParams(), _Group(), []
    builder = _builder(calls)
    for _ in range(20):
        _packed_cp_context(packed, torch.tensor([0, 4, 12, 16]), group, 4, builder)
    assert len(packed._gdn_cpu_offsets_cache) == 4
    assert len(packed._gdn_cp_context_cache) == 8


@pytest.mark.parametrize('variant', [gdn, kda])
def test_forward_uses_microbatch_context_across_layers(monkeypatch, variant):
    group, calls = _Group(), []
    builder = _builder(calls)
    monkeypatch.setattr(variant, 'build_cp_context', builder)
    monkeypatch.setattr(
        variant,
        'convert_module_input_tensors_cp_partition_mode',
        lambda **kwargs: (kwargs['hidden_states'], None),
    )
    offsets = torch.tensor([0, 4, 12, 16], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format='thd', cu_seqlens_q=offsets, cu_seqlens_kv=offsets)
    observed = []

    def compute(*args):
        observed.append(args[-1])
        return args[0], None

    layer = SimpleNamespace(
        pg_collection=SimpleNamespace(cp=group),
        tp_group=None,
        sp_size=1,
        config=SimpleNamespace(
            linear_cp_mode='chunkwise',
            sequence_parallel=False,
            deterministic_mode=False,
            cp_partition_mode='contiguous',
        ),
        conv_kernel_dim=4,
        recompute_gdn=False,
        training=True,
        _forward_compute=compute,
        _chunkwise_cp_context_cache={},
        _resolve_cu_seqlens=lambda *args, **kwargs: _GDNBase._resolve_cu_seqlens(
            None, *args, **kwargs
        ),
        _validate_packed_cu_seqlens=kda.KimiDeltaAttention._validate_packed_cu_seqlens,
    )
    cls = gdn.GatedDeltaNet if variant is gdn else kda.KimiDeltaAttention
    for _ in range(3):
        cls.forward(layer, torch.zeros(8, 1, 2), attention_mask=None, packed_seq_params=packed)
    assert len(calls) == 1
    assert all(context is observed[0] for context in observed)
    offsets.copy_(torch.tensor([0, 2, 8, 16]))
    cls.forward(layer, torch.zeros(8, 1, 2), attention_mask=None, packed_seq_params=packed)
    assert len(calls) == 2 and observed[-1] is not observed[0]


def test_cpu_validation_keeps_padding_and_divisibility_checks():
    packed = PackedSeqParams()
    actual, padded = torch.tensor([0, 3, 11, 15]), torch.tensor([0, 4, 12, 16])
    mirror = _cpu_cu_seqlens(packed, padded)
    assert (
        _GDNBase._resolve_cu_seqlens(
            None, padded, actual, 16, 'q', cp_size=2, cu_seqlens_cpu=mirror
        )
        is padded
    )
    with pytest.raises(ValueError, match='total_sequence_length'):
        _GDNBase._resolve_cu_seqlens(
            None, padded, actual, 18, 'q', cp_size=2, cu_seqlens_cpu=mirror
        )
    with pytest.raises(ValueError, match='divisible'):
        _GDNBase._resolve_cu_seqlens(
            None, None, actual, 15, 'q', cp_size=2, cu_seqlens_cpu=_cpu_cu_seqlens(packed, actual)
        )
