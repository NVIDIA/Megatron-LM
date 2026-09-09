# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Exercise native per-chunk data streams and TP broadcasts with real process groups."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

import pretrain_hybrid
from megatron.core import parallel_state
from megatron.training.datasets.data_samplers import MegatronPretrainingSampler
from tests.unit_tests.test_utilities import Utils


class _IdentifiedSamples(Dataset):
    """Encode sample identity and source position in each token."""

    def __len__(self):
        return 128

    def __getitem__(self, index):
        tokens = index * 100 + torch.arange(16)
        return {
            'tokens': tokens,
            'labels': tokens + 1,
            'position_ids': torch.arange(16),
            'loss_mask': torch.ones(16),
        }


def _iterator():
    return iter(
        DataLoader(
            _IdentifiedSamples(),
            batch_sampler=MegatronPretrainingSampler(
                total_samples=128,
                consumed_samples=8,
                micro_batch_size=2,
                data_parallel_rank=parallel_state.get_data_parallel_rank(),
                data_parallel_size=parallel_state.get_data_parallel_world_size(),
            ),
            num_workers=0,
        )
    )


def _configure(monkeypatch, tp, pp, cp, vp, mtp):
    args = SimpleNamespace(
        sequence_packing_scheduler=None,
        context_parallel_size=cp,
        sft=False,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        hybrid_context_parallel=False,
        micro_batch_size=2,
        seq_length=16,
        pipeline_model_parallel_size=pp,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_layout=None,
        mtp_num_layers=1 if mtp else None,
        virtual_pipeline_model_parallel_size=vp,
        linear_cp_layout='contiguous',
        attention_cp_layout='zigzag',
        sequence_parallel=tp > 1,
        tensor_model_parallel_size=tp,
    )
    monkeypatch.setattr(pretrain_hybrid, 'get_args', lambda: args)
    monkeypatch.setattr(pretrain_hybrid, 'core_transformer_config_from_args', lambda _: config)


def _check_tokens(cp_batch, microbatch):
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    cp_size = parallel_state.get_context_parallel_world_size()
    sample_ids = 8 + microbatch * 2 * dp_size + dp_rank * 2 + torch.arange(2, device='cuda')
    positions = torch.arange(16, device='cuda')
    expected = sample_ids[:, None] * 100 + positions[None, :]
    contiguous = expected.chunk(cp_size, dim=1)[cp_rank]
    torch.testing.assert_close(cp_batch.get_batch()['tokens'], contiguous, rtol=0, atol=0)
    if cp_size > 1:
        segments = expected.chunk(cp_size * 2, dim=1)
        zigzag = torch.cat((segments[cp_rank], segments[2 * cp_size - cp_rank - 1]), dim=1)
        torch.testing.assert_close(cp_batch.get_batch('zigzag')['tokens'], zigzag, rtol=0, atol=0)


@pytest.mark.parametrize(
    ('tp', 'pp', 'cp', 'vp', 'ep'),
    [(1, 2, 1, None, 1), (2, 2, 1, 2, 1), (1, 2, 2, 2, 1), (2, 2, 2, 2, 1), (1, 2, 1, 2, 2)],
)
@pytest.mark.parametrize('mtp', [False, True])
def test_chunk_data_streams_match_across_pipeline_and_context_ranks(
    monkeypatch, tp, pp, cp, vp, mtp, ep
):
    if Utils.world_size != tp * pp * cp * ep:
        pytest.skip('requires the topology-specific world size')
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        virtual_pipeline_model_parallel_size=vp,
        expert_model_parallel_size=ep,
    )
    try:
        _configure(monkeypatch, tp, pp, cp, vp, mtp)
        # Every chunk owns an independent native sampler. Rank/chunk scheduling
        # must not affect the source sample consumed by its nth forward.
        chunks = list(range(vp)) if vp is not None else [None]
        iterators = {chunk: _iterator() for chunk in chunks}
        calls = []
        original_broadcast = torch.distributed.broadcast

        def record_broadcast(tensor, src, group, **kwargs):
            calls.append(group)
            return original_broadcast(tensor, src, group=group, **kwargs)

        monkeypatch.setattr(torch.distributed, 'broadcast', record_broadcast)
        for microbatch in range(3):
            # Opposite PP ranks deliberately visit their local VPP chunks in
            # different orders; no PP forward collective can be inserted here.
            local_chunks = (
                chunks if parallel_state.get_pipeline_model_parallel_rank() == 0 else chunks[::-1]
            )
            for chunk in local_chunks:
                assert pretrain_hybrid.is_dataset_built_on_rank(
                    vp_stage=chunk, requires_token_ids=True
                ) == (parallel_state.get_tensor_model_parallel_rank() == 0)
                batch = pretrain_hybrid.get_batch(
                    iterators[chunk], vp_stage=chunk, requires_token_ids=True
                )
                _check_tokens(batch, microbatch)
                last = parallel_state.is_pipeline_last_stage(
                    ignore_virtual=chunk is None, vp_stage=chunk
                )
                assert (batch.get_batch()['labels'] is not None) == last
        assert calls and all(
            group is parallel_state.get_tensor_model_parallel_group() for group in calls
        )
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('mtp', [False, True])
def test_feature_off_middle_chunk_does_not_advance_data(monkeypatch, mtp):
    if Utils.world_size != 2:
        pytest.skip('requires two pipeline ranks')
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2
    )
    try:
        _configure(monkeypatch, 1, 2, 1, 2, mtp)
        rank = parallel_state.get_pipeline_model_parallel_rank()
        # Rank zero's second chunk and rank one's first chunk are both interior.
        vp_stage = 1 - rank
        assert not pretrain_hybrid.is_dataset_built_on_rank(vp_stage=vp_stage)
        iterator = _iterator()
        batch = pretrain_hybrid.get_batch(iterator, vp_stage=vp_stage)
        assert all(value is None for value in batch.get_batch().values())
        expected = next(_iterator())
        actual = next(iterator)
        torch.testing.assert_close(actual['tokens'], expected['tokens'], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize('mtp', [False, True])
def test_feature_off_endpoints_keep_original_batch_schema(monkeypatch, mtp):
    if Utils.world_size != 2:
        pytest.skip('requires two pipeline ranks')
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2, virtual_pipeline_model_parallel_size=2
    )
    try:
        _configure(monkeypatch, 1, 2, 1, 2, mtp)
        rank = parallel_state.get_pipeline_model_parallel_rank()
        # These are the global embedding and output chunks respectively.
        assert pretrain_hybrid.is_dataset_built_on_rank(vp_stage=rank)
        batch = pretrain_hybrid.get_batch(_iterator(), vp_stage=rank).get_batch()
        assert (batch['tokens'] is not None) == (rank == 0 or mtp)
        assert (batch['position_ids'] is not None) == (rank == 0 or mtp)
        assert (batch['labels'] is not None) == (rank == 1)
        assert (batch['loss_mask'] is not None) == (rank == 1)
        expected = next(_iterator())
        for key in ('tokens', 'position_ids', 'labels', 'loss_mask'):
            if batch[key] is not None:
                torch.testing.assert_close(batch[key].cpu(), expected[key], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
