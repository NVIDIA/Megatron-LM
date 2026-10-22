# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare V4.1 typed-pipeline plans from main's ContextParallelBatch contract."""

from dataclasses import replace
from functools import partial

from megatron.core.models.deepseek_v41.state import BatchShape
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayloadPlan,
)
from megatron.core.utils import get_attr_wrapped_model


def normalize_v41_batch(cp_batch):
    """Keep logical lengths distinct from storage padding for attention and Engram."""
    batch = cp_batch.get_batch()
    packed = cp_batch.get_packed_seq_params()
    logical = batch.get('cu_seqlens')
    if packed is not None and logical is not None:
        packed = replace(
            packed, cu_seqlens_q=logical.reshape(-1).int(), cu_seqlens_kv=logical.reshape(-1).int()
        )
        cp_batch.packed_seq_params_by_layout[cp_batch.boundary_layout] = packed
    # Token padding is represented by packed logical/physical lengths. Dense masks
    # are not needed, and are not silently substituted for the packed contract.
    if batch.get('padding_mask') is not None:
        if packed is None or logical is None:
            raise ValueError('V4.1 padded batches require original logical sequence prefixes')
        batch['padding_mask'] = None
    return cp_batch


def batch_shape(cp_batch, args):
    """Derive the same capacity on data-owning and metadata-only PP stages."""
    batch, packed = cp_batch.get_batch(), cp_batch.get_packed_seq_params()
    data = next((batch.get(k) for k in ('tokens', 'labels') if batch.get(k) is not None), None)
    if data is not None:
        return BatchShape.from_batch(data, packed, args.context_parallel_size)
    if packed is not None:
        if (
            not isinstance(packed.total_tokens, int)
            or packed.total_tokens % args.context_parallel_size
        ):
            raise ValueError('Packed PP metadata must expose an aligned host token capacity')
        return BatchShape(
            packed.total_tokens // args.context_parallel_size,
            1,
            packed.max_seqlen_q,
            packed.cu_seqlens_q.numel() - 1,
            args.context_parallel_size,
        )
    return BatchShape(args.seq_length, args.micro_batch_size)


def prepare_v41_pipeline_inputs(
    iterator, model, num_microbatches, *, args, get_batch, forward_only=False
):
    """Stage variable shapes before P2P; keep fixed SBHD batches lazy."""
    describe = get_attr_wrapped_model(model, 'pipeline_payload_spec')
    vp_stage = get_attr_wrapped_model(model, 'vp_stage')
    load = partial(get_batch, iterator, vp_stage)
    if not (
        args.sft
        or args.dataloader_inter_document_masking
        or getattr(args, 'sequence_packing_scheduler', None) is not None
    ):
        incoming, outgoing = describe(BatchShape(args.seq_length, args.micro_batch_size))
        return PipelineDataIterator(
            load,
            PipelinePayloadPlan((incoming,) * num_microbatches, (outgoing,) * num_microbatches),
        )
    batches, incoming, outgoing = [], [], []
    for _ in range(num_microbatches):
        batch = normalize_v41_batch(load())
        recv, send = describe(batch_shape(batch, args))
        batches.append(batch)
        incoming.append(recv)
        outgoing.append(send)
    return PipelineDataIterator(batches, PipelinePayloadPlan(tuple(incoming), tuple(outgoing)))
