# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepared Hybrid pipeline batches preserve packing and the original rerun iterator."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import pretrain_hybrid as entry
from megatron.core.pipeline_parallel.pipeline_payload import PipelineDataIterator
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)
from megatron.training.datasets import hybrid_pipeline
from tests.unit_tests.pipeline_parallel.test_typed_pipeline import _Timers
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import _packed
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _config,
    _pattern,
)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("layout", ["sbhd", "packed", "raw-prefixes"])
@pytest.mark.parametrize("forward_only", [False, True])
def test_training_prepares_each_batch_once_and_rebuilds_plans_on_rerun(
    monkeypatch, stage, layout, forward_only
):
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        seq_length=16,
        micro_batch_size=2,
        sft=layout == "raw-prefixes",
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler="dp_balanced" if layout == "packed" else None,
    )
    timers = _Timers()
    monkeypatch.setattr(entry, "get_args", lambda: args)
    monkeypatch.setattr(entry, "get_timers", lambda: timers)
    monkeypatch.setattr(entry, "stimer", lambda **kwargs: nullcontext())
    # CP=1 route initialization needs MPU globals but does not change the
    # batch descriptors exercised here. Packing tensors and host scalars are real.
    monkeypatch.setattr(hybrid_pipeline, "finalize_packed_seq_params", lambda params: params)
    monkeypatch.setattr(
        "megatron.core.rerun_state_machine.get_rerun_state_machine",
        lambda: SimpleNamespace(get_mode=lambda: RerunMode.VALIDATE_RESULTS),
    )
    config = replace(
        _config(),
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )
    pattern = _pattern((7, 8, 10))
    offsets = (0, 7, 8, 10)
    adapter = CSA2HybridAdapter(
        config,
        layer_type_list=list(pattern.split("|")[stage]),
        pp_layer_offset=offsets[stage],
        pre_process=stage == 0,
        post_process=stage == 3,
        is_mtp_layer=False,
    )
    adapter.configure_distributed_pipeline(
        pattern, SimpleNamespace(size=lambda: 2, rank=lambda: stage % 2), vp_stage=stage // 2
    )
    model = SimpleNamespace(
        pipeline_payload_spec=adapter.pipeline_payload_spec, vp_stage=stage // 2
    )
    batches, shapes = [], []
    for lengths in ([1], [3, 0, 2], [2, 4]):
        batch = [None] * 12
        shape = (2, 16)
        if layout != "sbhd":
            params, _, valid = _packed(lengths, lengths, tail=3 if layout == "packed" else 0)
            shape = (1, valid.numel())
            if layout == "packed":
                # The packing scheduler provides a mask even on metadata-only
                # stages. Its shape includes tail capacity beyond the endpoint.
                batch[-1] = params
                batch[10] = (~valid).reshape(shape)
            else:
                batch[1] = params.cu_seqlens_q.unsqueeze(0)
                batch[2] = params.cu_seqlens_q_padded.unsqueeze(0)
                batch[7] = torch.tensor(params.max_seqlen_q, dtype=torch.int32)
        if stage in (0, 3):
            batch[9 if stage == 0 else 4] = torch.zeros(shape, dtype=torch.int64)
        batches.append(tuple(batch))
        shapes.append(shape)

    original_get_batch = entry.get_batch
    fetched = []

    def get_batch(iterator, vp_stage=None):
        if isinstance(iterator, PipelineDataIterator):
            return original_get_batch(iterator, vp_stage)
        assert vp_stage == model.vp_stage
        fetched.append(vp_stage)
        return next(iterator)

    monkeypatch.setattr(entry, "get_batch", get_batch)
    iterator = RerunDataIterator(iter(batches))
    previous_plan = None
    for run in range(2):
        prepared = entry.prepare_pipeline_inputs(
            iterator, model, len(batches), forward_only=forward_only
        )
        assert len(fetched) == (run + (layout != "sbhd")) * len(batches)
        plan = prepared.pipeline_payload_plan
        if previous_plan is not None:
            assert plan == previous_plan and plan is not previous_plan
        previous_plan = plan
        for index, (batch_size, tokens) in enumerate(shapes):
            # Communication/forward reuses the resolved metadata. No tensor
            # scalar can be read while consuming the prepared batch queue.
            with monkeypatch.context() as context:

                def no_item(*args, **kwargs):
                    pytest.fail("Prepared input consumption must not read device scalars")

                context.setattr(torch.Tensor, "item", no_item)
                batch = entry.get_batch(prepared, model.vp_stage)
                assert (
                    hybrid_pipeline.get_hybrid_packed_seq_params(
                        batch, tokens_per_sample=args.seq_length
                    )
                    is batch[-1]
                )
            for spec in (plan.incoming[index], plan.outgoing[index]):
                if spec is not None:
                    assert spec.tensor_specs[0].shape[:2] == (tokens, batch_size)
                    assert spec.tensor_specs[0].requires_grad == (not forward_only)
                    if layout != "sbhd":
                        assert spec.metadata[1] == batch[-1].max_seqlen_q
            if layout != "sbhd":
                assert len(prepared._batches) == len(batches) - index - 1
            else:
                assert len(fetched) == run * len(batches) + index + 1
        with pytest.raises(StopIteration):
            entry.get_batch(prepared, model.vp_stage)
        assert len(fetched) == (run + 1) * len(batches)
        iterator.rewind()
    assert all(timer.started == timer.stopped for timer in timers.values())


def test_static_middle_stage_keeps_none_data_iterator(monkeypatch):
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        seq_length=16,
        micro_batch_size=2,
        sft=False,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler=None,
        create_attention_mask_in_dataloader=False,
        dynamic_context_parallel=False,
    )
    config = _config()
    monkeypatch.setattr(entry, "get_args", lambda: args)
    monkeypatch.setattr(entry, "core_transformer_config_from_args", lambda args: config)
    monkeypatch.setattr(entry.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(entry, "mtp_on_this_rank_func", lambda **kwargs: False)
    monkeypatch.setattr(entry, "is_first_or_last_pipeline_stage", lambda vp: False)
    adapter = CSA2HybridAdapter(
        config,
        layer_type_list=["E"],
        pp_layer_offset=7,
        pre_process=False,
        post_process=False,
        is_mtp_layer=False,
    )
    adapter.configure_distributed_pipeline(
        _pattern((7, 8, 10)), SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    )
    model = SimpleNamespace(pipeline_payload_spec=adapter.pipeline_payload_spec, vp_stage=None)
    prepared = entry.prepare_pipeline_inputs(None, model, 2)
    assert all(spec is not None for spec in prepared.pipeline_payload_plan.incoming)
    # Use production get_batch, including the early return for middle SBHD
    # stages. No loader, CUDA transfer, or iterator.next() is needed on this rank.
    assert entry.get_batch(prepared) == [None] * 12
    assert entry.get_batch(prepared) == [None] * 12
    with pytest.raises(StopIteration):
        entry.get_batch(prepared)
