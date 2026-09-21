# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepared Hybrid pipeline batches preserve packing and the original rerun iterator."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import pretrain_hybrid as entry
from megatron.core.models.hybrid.hybrid_stack_adapter import HybridStateAdapter
from megatron.core.pipeline_parallel.pipeline_payload import PipelineDataIterator
from megatron.core.rerun_state_machine import RerunDataIterator, RerunMode
from megatron.core.transformer.experimental_attention_variant.csa_utils.csa2_hybrid_adapter import (
    CSA2HybridAdapter,
)
from megatron.core.transformer.hyper_connection import SinglePassMHCBoundary
from megatron.training.datasets import hybrid_pipeline
from tests.unit_tests.pipeline_parallel.test_typed_pipeline import _Timers
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2 import _packed
from tests.unit_tests.transformer.experimental_attention_variant.test_csa2_pipeline import (
    _config,
    _pattern,
)


def _adapter(config, **kwargs):
    return HybridStateAdapter(
        config,
        components=(
            SinglePassMHCBoundary(config, kwargs["pp_layer_offset"]),
            CSA2HybridAdapter(config, **kwargs),
        ),
        hidden_size=config.hidden_size * config.num_residual_streams,
        hidden_dtype=config.params_dtype,
        pg_collection=SimpleNamespace(cp=None),
        **kwargs,
    )


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("forward_only", [False, True])
def test_fixed_sbhd_pp_plan_matches_cp_local_mhc_payload(cp_rank, forward_only):
    from megatron.core.models.hybrid.hybrid_state import build_hybrid_state_pipeline_plan
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _config, _stack

    config = _config(
        context_parallel_size=2, pipeline_model_parallel_size=2, pipeline_dtype=torch.float32
    )
    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
    plan = build_hybrid_state_pipeline_plan(config, "*-*|-*-", pp_size=2)
    first, second = [_stack(config, chunk) for chunk in plan]
    for rank, stack in enumerate((first, second)):
        stack.forward_adapter.cp_group = cp_group
        stack.forward_adapter.configure_distributed_pipeline(
            "*-*|-*-", SimpleNamespace(size=lambda: 2, rank=lambda: rank)
        )
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        seq_length=16,
        micro_batch_size=1,
        sft=False,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler=None,
    )
    plans = []
    for stack in (first, second):
        model = SimpleNamespace(
            pipeline_payload_spec=stack.forward_adapter.pipeline_payload_spec, vp_stage=None
        )
        prepared = hybrid_pipeline.prepare_hybrid_pipeline_inputs(
            None,
            model,
            1,
            args=args,
            get_batch=lambda *_: None,
            get_timers=lambda: _Timers(),
            batch_context=nullcontext,
            forward_only=forward_only,
        )
        plans.append(prepared.pipeline_payload_plan)
    with torch.set_grad_enabled(not forward_only):
        payload = first(torch.randn(8, 1, config.hidden_size), None)
        assert payload.tensors[0].shape == (8, 1, 32)
        assert payload.descriptor == plans[0].outgoing[0] == plans[1].incoming[0]
        received = second.forward_adapter.make_pipeline_payload(payload.tensors, payload.descriptor)
        second.set_input_tensor(received)
        assert second(None, None).shape == (8, 1, config.hidden_size)


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
    adapter = _adapter(
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


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("layout", ["raw-prefixes", "scheduled-tail"])
@pytest.mark.parametrize("forward_only", [False, True])
def test_packed_pp_cp_middle_stage_uses_local_physical_capacity(
    monkeypatch, cp_rank, layout, forward_only
):
    from megatron.core.models.hybrid.hybrid_state import build_hybrid_state_pipeline_plan
    from megatron.core.packed_seq_params import PackedSeqParams
    from tests.unit_tests.ssm.test_hybrid_state_adapter import _config, _stack

    config = _config(
        context_parallel_size=2, pipeline_model_parallel_size=3, pipeline_dtype=torch.float32
    )
    cp_group = SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
    monkeypatch.setattr(
        hybrid_pipeline,
        "finalize_packed_seq_params",
        lambda params: setattr(params, "cp_group", cp_group),
    )
    args = SimpleNamespace(
        tensor_model_parallel_size=1,
        context_parallel_size=2,
        seq_length=16,
        micro_batch_size=1,
        sft=True,
        dataloader_inter_document_masking=False,
        sequence_packing_scheduler=None if layout == "raw-prefixes" else "pack_by_seq",
    )
    local_capacity = 8 if layout == "raw-prefixes" else 12
    prefix = torch.tensor([0, 8, 16], dtype=torch.int32)
    chunks = build_hybrid_state_pipeline_plan(config, "*-|*-|*-", pp_size=3, qkv_format="thd")
    stacks, plans, params = [], [], []
    for rank, chunk in enumerate(chunks):
        stack = _stack(config, chunk)
        stack.forward_adapter.cp_group = cp_group
        stack.forward_adapter.configure_distributed_pipeline(
            "*-|*-|*-", SimpleNamespace(size=lambda: 3, rank=lambda: rank)
        )
        batch = [None] * 12
        if layout == "raw-prefixes":
            batch[1], batch[2], batch[7] = prefix.unsqueeze(0), prefix.unsqueeze(0), torch.tensor(8)
        else:
            batch[-1] = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=prefix,
                cu_seqlens_kv=prefix,
                cu_seqlens_q_padded=prefix,
                cu_seqlens_kv_padded=prefix,
                max_seqlen_q=8,
                max_seqlen_kv=8,
                total_tokens=local_capacity,
                cp_group=cp_group,
                local_cp_size=2,
            )
        if rank in (0, 2):
            batch[9 if rank == 0 else 4] = torch.zeros(1, local_capacity, dtype=torch.long)
        prepared = hybrid_pipeline.prepare_hybrid_pipeline_inputs(
            None,
            SimpleNamespace(
                pipeline_payload_spec=stack.forward_adapter.pipeline_payload_spec, vp_stage=None
            ),
            1,
            args=args,
            get_batch=lambda *_: tuple(batch),
            get_timers=lambda: _Timers(),
            batch_context=nullcontext,
            forward_only=forward_only,
        )
        stacks.append(stack)
        plans.append(prepared.pipeline_payload_plan)
        params.append(next(prepared)[-1])
    for sender, receiver in zip(plans, plans[1:]):
        assert sender.outgoing[0] == receiver.incoming[0]
        assert sender.outgoing[0].tensor_specs[0].shape == (local_capacity, 1, 32)
    # Consume all three stages with the same declared local capacity. The CPU
    # attention fixture exercises payload/state flow, not distributed CP math.
    with torch.set_grad_enabled(not forward_only):
        output = torch.randn(local_capacity, 1, config.hidden_size)
        for rank, stack in enumerate(stacks):
            if rank:
                stack.set_input_tensor(output)
            output = stack(output if rank == 0 else None, None, packed_seq_params=params[rank])
            if rank < 2:
                assert output.descriptor == plans[rank].outgoing[0]
        assert output.shape == (local_capacity, 1, config.hidden_size)


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
    adapter = _adapter(
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
