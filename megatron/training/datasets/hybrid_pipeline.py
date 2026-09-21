# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare Hybrid input batches and host descriptors before pipeline communication."""

from contextlib import AbstractContextManager
from functools import partial
from typing import Any, Callable, NamedTuple, Sequence

import torch

from megatron.core.context_parallel_layout import finalize_packed_seq_params
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    PipelinePayloadPlan,
)
from megatron.core.timers import Timers
from megatron.core.utils import divide, get_attr_wrapped_model


class _HybridBatch(NamedTuple):
    attention_mask: Any
    cu_seqlens: Any
    cu_seqlens_padded: Any
    hybrid_cp_group: Any
    labels: Any
    local_cp_size: Any
    loss_mask: Any
    max_seqlen: Any
    position_ids: Any
    tokens: Any
    padding_mask: Any
    packed_seq_params: Any


def get_hybrid_packed_seq_params(
    batch: Sequence[Any], *, tokens_per_sample: int
) -> PackedSeqParams | None:
    """Reuse packing metadata or resolve raw-prefix host scalars once per batch.

    Args:
        batch: The twelve fields returned by the Hybrid batch loader.
        tokens_per_sample: Configured sample length for the raw-prefix path.

    Returns:
        Existing or finalized packed metadata, or None for an unpacked batch.
    """
    batch = _HybridBatch(*batch)
    if batch.packed_seq_params is not None or batch.cu_seqlens is None:
        return batch.packed_seq_params
    cu_seqlens = batch.cu_seqlens.squeeze(0)
    cu_padded = batch.cu_seqlens_padded
    cu_padded = cu_padded.squeeze(0) if cu_padded is not None else None
    cu_for_params = cu_padded if cu_padded is not None else cu_seqlens
    max_seqlen = int(batch.max_seqlen.item())
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_for_params,
        cu_seqlens_kv=cu_for_params,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        local_cp_size=int(batch.local_cp_size.item()) if batch.local_cp_size is not None else None,
        cp_group=batch.hybrid_cp_group,
        total_tokens=int(cu_for_params[-1].item()),
        tokens_per_sample=tokens_per_sample,
    )
    finalize_packed_seq_params(params)
    return params


def prepare_hybrid_pipeline_inputs(
    data_iterator: Any,
    model: torch.nn.Module,
    num_microbatches: int,
    *,
    args: Any,
    get_batch: Callable[[Any, int | None], Sequence[Any]],
    get_timers: Callable[[], Timers],
    batch_context: Callable[[], AbstractContextManager],
    forward_only: bool = False,
) -> PipelineDataIterator:
    """Stage batches and payload plans before entering the PP/VPP schedule.

    Args:
        data_iterator: The original loader or rerun iterator for this virtual chunk.
        model: A model exposing pipeline_payload_spec and vp_stage, possibly wrapped.
        num_microbatches: Number of batches in this schedule invocation.
        args: Training arguments for TP, sequence packing and batch dimensions.
        get_batch: Loader callback that returns the twelve Hybrid batch fields.
        get_timers: Training timer getter, used only when staging packed batches.
        batch_context: Context factory for batch-generation instrumentation.
        forward_only: Whether receive tensors can omit gradient slots.

    Returns:
        An iterator with host-only plans and batches consumed exactly once.
        Fixed SBHD batches remain lazy; packed batches retain their original
        global prefixes and already CP-partitioned tensors. The input iterator
        remains authoritative for checkpointing and reruns.
    """
    if args.tensor_model_parallel_size != 1:
        raise ValueError("Prepared Hybrid pipeline payloads currently require TP=1")
    describe = get_attr_wrapped_model(model, "pipeline_payload_spec")
    vp_stage = get_attr_wrapped_model(model, "vp_stage")
    if not (args.sft or args.dataloader_inter_document_masking or args.sequence_packing_scheduler):
        incoming, outgoing = describe(
            divide(args.seq_length, args.context_parallel_size),
            args.micro_batch_size,
            requires_grad=not forward_only,
        )
        return PipelineDataIterator(
            partial(get_batch, data_iterator, vp_stage),
            PipelinePayloadPlan((incoming,) * num_microbatches, (outgoing,) * num_microbatches),
        )
    batches, incoming, outgoing = [], [], []
    timer = get_timers()("batch-generator", log_level=2)
    timer.start()
    try:
        with batch_context():
            for _ in range(num_microbatches):
                batch = _HybridBatch(*get_batch(data_iterator, vp_stage))
                raw_prefixes = batch.packed_seq_params is None and batch.cu_seqlens is not None
                params = get_hybrid_packed_seq_params(batch, tokens_per_sample=args.seq_length)
                batch = batch._replace(packed_seq_params=params)
                # Prefer actual CP-local physical buffers, including scheduler
                # tail padding. Metadata-only stages must interpret capacities
                # in the coordinate system used by their loader below.
                shape_source = next(
                    (t for t in (batch.tokens, batch.labels, batch.padding_mask) if t is not None),
                    None,
                )
                if shape_source is not None:
                    batch_size, seq_length = shape_source.shape
                elif params is not None:
                    seq_length, batch_size = params.total_tokens, 1
                    if type(seq_length) is not int:
                        raise ValueError("Packed pipeline batches require a host token capacity")
                    if raw_prefixes:
                        # Raw SFT prefixes describe the global physical pack. The
                        # scheduler instead supplies an already local capacity,
                        # which can include tail padding beyond the prefix end.
                        cp_size = (
                            params.cp_group.size()
                            if params.cp_group is not None
                            else params.local_cp_size or args.context_parallel_size
                        )
                        seq_length = divide(seq_length, cp_size)
                else:
                    seq_length = divide(args.seq_length, args.context_parallel_size)
                    batch_size = args.micro_batch_size
                recv_spec, send_spec = describe(
                    seq_length, batch_size, params, requires_grad=not forward_only
                )
                incoming.append(recv_spec)
                outgoing.append(send_spec)
                batches.append(tuple(batch))
    finally:
        timer.stop()
    return PipelineDataIterator(batches, PipelinePayloadPlan(tuple(incoming), tuple(outgoing)))
