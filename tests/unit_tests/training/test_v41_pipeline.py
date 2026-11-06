# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed metadata-only stages must plan the same buffers as data-owning stages."""

import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.models.deepseek_v41.model import DeepSeekV41Model
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.training.datasets.v41_pipeline import (
    batch_shape,
    normalize_v41_batch,
    prepare_v41_pipeline_inputs,
)


@pytest.mark.parametrize("scheduled", [False, True])
def test_packed_plans_preserve_logical_lengths_and_consume_batches_once(scheduled):
    logical = torch.tensor([0, 5, 11], dtype=torch.int32)
    padded = torch.tensor([0, 8, 16], dtype=torch.int32)
    params = PackedSeqParams(
        qkv_format='thd',
        cu_seqlens_q=padded,
        cu_seqlens_kv=padded,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
        max_seqlen_q=8,
        max_seqlen_kv=8,
        total_tokens=16,
    )
    args = SimpleNamespace(
        context_parallel_size=2,
        sft=not scheduled,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler='dp_balanced' if scheduled else None,
    )
    batches = []
    for owns_data in (True, False):
        batch = ContextParallelBatch.from_single_layout(
            'contiguous',
            dict(
                cu_seqlens=logical, tokens=torch.ones(1, 8, dtype=torch.long) if owns_data else None
            ),
            params,
        )
        batches.append(normalize_v41_batch(batch))
    assert batch_shape(batches[0], args) == batch_shape(batches[1], args)
    assert torch.equal(batches[0].get_packed_seq_params().cu_seqlens_q, logical)
    model = SimpleNamespace(
        config=SimpleNamespace(engram_config=None),
        vp_stage=0,
        pipeline_payload_spec=lambda shape: (shape, shape),
    )
    reads = []

    def get_batch(iterator, stage):
        reads.append(stage)
        return next(iterator)

    prepared = prepare_v41_pipeline_inputs(iter(batches), model, 2, args=args, get_batch=get_batch)
    assert reads == [0, 0]
    assert prepared.pipeline_payload_plan.incoming[0] == prepared.pipeline_payload_plan.incoming[1]
    assert list(prepared) == batches
    assert reads == [0, 0]


def test_pp1_without_prefetch_keeps_batch_loading_lazy():
    iterator = iter([object()])
    model = SimpleNamespace(
        config=SimpleNamespace(engram_config=None, pipeline_model_parallel_size=1)
    )
    prepared = prepare_v41_pipeline_inputs(iterator, model, 1, args=None, get_batch=None)
    assert prepared is iterator


def test_pipeline_lookahead_keeps_ticket_with_its_microbatch():
    from megatron.core.pipeline_parallel.pipeline_payload import (
        PipelineDataIterator,
        PipelinePayloadPlan,
    )

    prepared = []

    def prepare(index):
        prepared.append(index)
        return {'lookup_ticket': object()}

    iterator = PipelineDataIterator(
        ['a', 'b', 'c'], PipelinePayloadPlan((None,) * 3, (None,) * 3), prepare_batch=prepare
    )
    assert prepared == [0]
    ticket = iterator._next_kwargs['lookup_ticket']
    batch, kwargs = iterator.next_with_model_kwargs()
    assert batch == 'a' and kwargs['lookup_ticket'] is ticket
    assert prepared == [0, 1]
    batch, second = iterator.next_with_model_kwargs()
    assert batch == 'b' and second['lookup_ticket'] is not ticket
    assert prepared == [0, 1, 2]
    assert iterator.next_with_model_kwargs()[0] == 'c'
    with pytest.raises(StopIteration):
        iterator.next_with_model_kwargs()


def _engram_pipeline_inputs_worker(rank, path):
    dist.init_process_group(
        'gloo', init_method='file://' + path, rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        pp = cp = None
        for members in ([0, 2], [1, 3], [0, 1], [2, 3]):
            group = dist.new_group(members, backend='gloo')
            if rank in members:
                if members[1] - members[0] == 2:
                    pp = group
                else:
                    cp = group
        logical = torch.tensor([0, 3, 6], dtype=torch.int32)
        padded = torch.tensor([0, 4, 8], dtype=torch.int32)
        packed = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=logical,
            cu_seqlens_kv=logical,
            cu_seqlens_q_padded=padded,
            cu_seqlens_kv_padded=padded,
            max_seqlen_q=3,
            max_seqlen_kv=3,
            cp_group=cp,
        )
        tokens = torch.tensor([[1, 2, 3, 0]]) + cp.rank() * 10
        batches = [
            ContextParallelBatch.from_single_layout(
                'contiguous',
                {'tokens': tokens + index if pp.rank() == 0 else None},
                packed if pp.rank() == 0 else None,
            )
            for index in range(2)
        ]
        calls, cache, received = [], {}, []
        embed = object()

        def hasher(local_tokens, valid, sequence, *, cp_group):
            assert cp_group is cp
            torch.testing.assert_close(valid, torch.tensor([[True, True, True, False]]))
            torch.testing.assert_close(sequence, torch.full((1, 4), cp.rank(), dtype=torch.long))
            return local_tokens[..., None, None]

        def prefetch(module, hashes):
            assert module is embed
            received.append(hashes)
            return object()

        runtime = SimpleNamespace(reset=lambda: calls.append('reset'), prefetch=prefetch)
        model = SimpleNamespace(
            vp_stage=0,
            pg_collection=ProcessGroupCollection(pp=pp, cp=cp),
            pipeline_control_group=pp,
            parameters=lambda: iter([torch.zeros(1)]),
            engram_input_cache=cache,
            engram_runtime=runtime,
            engram_hash=hasher,
            decoder=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        layer_number=3, engram=SimpleNamespace(hash_index=0, embed=embed)
                    )
                ]
            ),
        )
        DeepSeekV41Model.stage_engram_inputs(model, batches)
        assert calls == ['reset']
        for index in range(2):
            torch.testing.assert_close(cache[index][0], tokens + index)
            torch.testing.assert_close(cache[index][1].cu_seqlens_q, logical)
            torch.testing.assert_close(cache[index][1].cu_seqlens_q_padded, padded)
        # A later virtual chunk reuses the first chunk's PP broadcasts and buffers.
        shared_tokens = cache[0][0]
        model.vp_stage = 1
        DeepSeekV41Model.stage_engram_inputs(model, batches)
        assert cache[0][0] is shared_tokens and calls == ['reset']
        kwargs = DeepSeekV41Model.prefetch_engram_batch(model, 1)
        assert set(kwargs['engram_tickets']) == {2}
        torch.testing.assert_close(received[0], (tokens + 1)[..., None, None])
    finally:
        dist.destroy_process_group()


def test_engram_lookahead_pp2_cp2_and_shared_vpp_metadata():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_engram_pipeline_inputs_worker, args=(directory + '/rendezvous',), nprocs=4)
