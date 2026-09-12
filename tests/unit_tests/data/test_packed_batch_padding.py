# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.layout import _build_thd_zigzag_metadata
from megatron.core.datasets.data_schedule_utils import (
    _unpack_batch,
    pad_packed_batch_before_cp_slice,
    reroute_samples_to_dcp_ranks,
)
from megatron.core.model_parallel_config import ModelParallelConfig


def _batch():
    # Two sequences: three valid tokens plus one gap, then five valid tokens.
    return {
        'tokens': torch.tensor([11, 12, 13, 0, 21, 22, 23, 24, 25]),
        'labels': torch.tensor([12, 13, 14, 0, 22, 23, 24, 25, 26]),
        'position_ids': torch.tensor([0, 1, 2, 0, 0, 1, 2, 3, 4]),
        'loss_mask': torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        'padding_mask': torch.tensor(
            [False, False, False, True, False, False, False, False, False]
        ),
        'cu_seqlens': torch.tensor([0, 3, 8], dtype=torch.int32),
        'cu_seqlens_padded': torch.tensor([0, 4, 9], dtype=torch.int32),
        'max_seqlen': torch.tensor(5, dtype=torch.int32),
    }


@pytest.mark.parametrize('cp_size', [1, 2, 4])
@pytest.mark.parametrize('tp_size,sequence_parallel', [(1, False), (4, False), (2, True)])
@pytest.mark.parametrize('requested', [None, 48, 'max'])
def test_mxfp8_padding_preserves_tokens_and_aligns_both_cp_layouts(
    cp_size, tp_size, sequence_parallel, requested
):
    batch = _batch()
    config = SimpleNamespace(
        fp8='hybrid',
        fp8_recipe='mxfp8',
        sequence_parallel=sequence_parallel,
        pad_packed_seq_alignment=requested,
        max_seqlen_per_dp_cp_rank=192,
    )
    original_tokens = batch['tokens'][~batch['padding_mask']].clone()
    pad_packed_batch_before_cp_slice(batch, config, cp_size, tp_size)
    sp_size = tp_size if sequence_parallel else 1
    local_length = batch['tokens'].numel() // cp_size
    assert local_length % (32 * sp_size) == 0
    if requested == 'max':
        assert local_length == 192
    elif requested:
        assert local_length % requested == 0
    torch.testing.assert_close(batch['tokens'][~batch['padding_mask']], original_tokens)
    assert batch['loss_mask'].sum() == 8
    assert torch.all(batch['loss_mask'][batch['padding_mask']] == 0)
    assert torch.equal(batch['cu_seqlens'][:3], torch.tensor([0, 3, 8], dtype=torch.int32))
    assert batch['cu_seqlens_padded'][-1] == batch['tokens'].numel()
    assert batch['max_seqlen'] == batch['cu_seqlens_padded'].diff().max()
    if cp_size > 1:
        zigzag = _build_thd_zigzag_metadata(
            batch['cu_seqlens'], batch['cu_seqlens_padded'], cp_size, sp_size
        )
        assert zigzag.rank_order_indices.numel() == batch['tokens'].numel()
        torch.testing.assert_close(zigzag.cu_seqlens_padded, batch['cu_seqlens_padded'])


def test_padding_works_on_metadata_only_pipeline_stage():
    batch = _batch()
    for key in ('tokens', 'labels', 'loss_mask', 'position_ids'):
        del batch[key]
    config = SimpleNamespace(pad_packed_seq_alignment=32, max_seqlen_per_dp_cp_rank=128)
    pad_packed_batch_before_cp_slice(batch, config, 2, 1)
    assert batch['padding_mask'].numel() == 64
    assert (~batch['padding_mask']).sum() == 8
    assert batch['cu_seqlens_padded'][-1] == 64


def test_padding_rejects_capacity_overflow_without_truncation():
    batch = _batch()
    original = batch['tokens'].clone()
    config = SimpleNamespace(pad_packed_seq_alignment=32, max_seqlen_per_dp_cp_rank=16)
    with pytest.raises(ValueError, match='exceeds'):
        pad_packed_batch_before_cp_slice(batch, config, 1, 1)
    torch.testing.assert_close(batch['tokens'], original)


def test_padding_disabled_preserves_batch():
    batch = _batch()
    original = dict(batch)
    pad_packed_batch_before_cp_slice(batch, SimpleNamespace(), 2, 1)
    assert all(batch[key] is value for key, value in original.items())


@pytest.mark.parametrize('alignment', [0, -1, 'invalid'])
def test_packed_padding_config_rejects_invalid_alignment(alignment):
    with pytest.raises(ValueError, match='positive integer'):
        ModelParallelConfig(pad_packed_seq_alignment=alignment, max_seqlen_per_dp_cp_rank=128)


def test_unpack_drops_prompt_only_truncated_tail():
    batch = _batch()
    batch.pop('padding_mask')
    batch.pop('cu_seqlens_padded')
    batch['cu_seqlens'] = torch.tensor([0, 4, 9, 9], dtype=torch.int32)
    batch['loss_mask'][4:] = 0
    unpacked = _unpack_batch([batch])
    assert len(unpacked) == 1
    torch.testing.assert_close(unpacked[0]['tokens'], torch.tensor([11, 12, 13, 0]))


def test_unpack_all_prompt_only_samples_leaves_empty_local_batch():
    batch = _batch()
    batch['loss_mask'].zero_()
    assert _unpack_batch([batch]) == []


def test_empty_source_rank_receives_rerouted_samples(monkeypatch):
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    remote_fields = iter(
        [
            torch.tensor([11, 12]),
            torch.tensor([12, 13]),
            torch.ones(2),
            torch.tensor([0, 1]),
            torch.tensor([2]),
            torch.tensor([2]),
        ]
    )

    def gather(output, local, group):
        assert torch.count_nonzero(local) == 0
        remote = next(remote_fields)
        assert local.dtype == remote.dtype
        output.copy_(torch.cat((local, remote)))

    monkeypatch.setattr(torch.cuda, 'current_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(torch.distributed, 'all_gather_into_tensor', gather)
    received = reroute_samples_to_dcp_ranks(
        [],
        torch.empty(0, dtype=torch.int32),
        [(0, 2)],
        [[[0], [0]]],
        torch.tensor([0, 0, 1]),
        group,
        group,
    )
    torch.testing.assert_close(received[0]['tokens'], torch.tensor([11, 12]))
