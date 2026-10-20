# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed metadata-only stages must plan the same buffers as data-owning stages."""

from types import SimpleNamespace

import torch

from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training.datasets.v41_pipeline import (
    batch_shape,
    normalize_v41_batch,
    prepare_v41_pipeline_inputs,
)


def test_packed_plans_preserve_logical_lengths_and_consume_batches_once():
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
        context_parallel_size=2, sft=True, dataloader_inter_document_masking=False
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
    model = SimpleNamespace(vp_stage=0, pipeline_payload_spec=lambda shape: (shape, shape))
    reads = []

    def get_batch(iterator, stage):
        reads.append(stage)
        return next(iterator)

    prepared = prepare_v41_pipeline_inputs(iter(batches), model, 2, args=args, get_batch=get_batch)
    assert reads == [0, 0]
    assert prepared.pipeline_payload_plan.incoming[0] == prepared.pipeline_payload_plan.incoming[1]
    assert list(prepared) == batches
    assert reads == [0, 0]
