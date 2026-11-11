# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real pipeline replay with DCP banks and graph-scoped activation offload."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
    reconfigure_num_microbatches_calculator,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel import fine_grained_activation_offload as offloading
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.fine_grained_activation_offload import PipelineOffloadManager
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.dcp_graph_test_utils import requires_te_graph_memory_slots


@pytest.mark.parametrize("slot_capture", [False, True])
def test_offload_graph_boundary_preserves_events_and_only_joins_used_streams(
    slot_capture, monkeypatch
):
    events, waits = [], []
    stream = SimpleNamespace(record_event=events.append, wait_stream=waits.append)
    manager = SimpleNamespace(cuda_graph_event=object(), d2h_stream=object())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(offloading, "_is_te_slot_graph_capturing", lambda: slot_capture)
    monkeypatch.setattr(PipelineOffloadManager, "get_instance", staticmethod(lambda: manager))
    off_interface.forward_record()
    assert events == [manager.cuda_graph_event]
    assert waits == ([] if slot_capture else [manager.d2h_stream])


def test_graph_offload_policy_keeps_warmup_retention_decisions():
    manager = PipelineOffloadManager.__new__(PipelineOffloadManager)
    manager._is_warmup = False
    manager._cached_chunks_forward = [
        SimpleNamespace(
            offload_groups=[
                SimpleNamespace(graph_owner=(owner, "core_attn"), offload=enabled)
                for owner, enabled in ((1, True), (2, keep))
            ],
            tensor_need_offloading_checker=lambda tensor: tensor == "eligible",
        )
        for keep in (True, False)
    ]
    assert manager.graph_offload_checker(1, "core_attn")("eligible")
    assert not manager.graph_offload_checker(1, "core_attn")("small")
    assert not manager.graph_offload_checker(2, "core_attn")("eligible")
    with pytest.raises(RuntimeError, match="completed eager offload warmup"):
        manager.graph_offload_checker(3, "core_attn")


def test_offload_interface_drops_previous_graph_context_on_reentry():
    tensor = torch.zeros(1)
    interface = off_interface(False, tensor, "core_attn")
    interface._graph_context = object()
    with interface as result:
        assert result is tensor
    assert interface._graph_context is None


def _iteration(models, microbatches):
    reconfigure_num_microbatches_calculator(
        rank=torch.distributed.get_rank(),
        global_batch_size=microbatches * parallel_state.get_data_parallel_world_size(),
        micro_batch_size=1,
        data_parallel_size=parallel_state.get_data_parallel_world_size(),
    )

    def forward_step(data_iterator, model):
        microbatch = next(data_iterator)
        size = 1 if microbatch == 0 else 2
        group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=size)
        cu = torch.tensor([0, 128 * size, 128 * size], dtype=torch.int32, device="cuda")
        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cu,
            cu_seqlens_kv_padded=cu,
            max_seqlen_q=128 * size,
            max_seqlen_kv=128 * size,
            local_cp_size=size,
            cp_group=group if size > 1 else None,
            cp_singleton_group=group if size == 1 else None,
            pad_between_seqs=True,
        )
        tokens = (torch.arange(128, device="cuda") + microbatch).remainder(128).view(1, 128)
        positions = torch.arange(128, device="cuda").view(1, 128)
        output = model(
            tokens,
            positions,
            None,
            labels=tokens,
            packed_seq_params=params,
            padding_mask=torch.zeros(1, 128, dtype=torch.bool, device="cuda"),
        )

        def loss_func(value):
            loss = value.float().mean()
            return loss, {"lm loss": loss.detach()}

        return output, loss_func

    for model in models:
        model.zero_grad(set_to_none=False)
    return get_forward_backward_func()(
        forward_step_func=forward_step,
        data_iterator=[iter(range(microbatches)) for _ in models],
        model=models,
        num_microbatches=microbatches,
        seq_length=256,
        micro_batch_size=1,
        forward_only=False,
    )


def _snapshot_fp8_state(models):
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
    from transformer_engine.pytorch.ops.op import BasicOperation

    states = []
    for model in models:
        for module in model.modules():
            if isinstance(module, TransformerEngineBaseModule):
                state = module.get_fp8_meta_tensors()
                restore = module.reset_fp8_meta_tensors
            elif isinstance(module, BasicOperation):
                state = module._save_fp8_metas()
                restore = module._load_fp8_metas
            else:
                continue
            if state is not None:
                states.append((restore, state))
    return states


def _restore_fp8_state(states):
    for restore, state in states:
        restore(state)


@requires_te_graph_memory_slots
@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 4, reason="requires PP2 x DP2")
@pytest.mark.parametrize("vpp", [None, 2])
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_dcp_partial_graph_offload_pipeline_matches_eager(vpp, moe, fp8, monkeypatch):
    from transformer_engine.pytorch import cpu_offload
    from transformer_engine.pytorch.quantization import FP8GlobalStateManager

    monkeypatch.setenv("NVTE_CPU_OFFLOAD_V1", "1")
    monkeypatch.setattr(cpu_offload, "NVTE_CPU_OFFLOAD_V1", True)
    if fp8 and (
        torch.backends.cudnn.version() < 92300 or torch.cuda.get_device_capability()[0] < 10
    ):
        pytest.skip("FP8 THD backward requires cuDNN >= 9.23 and SM100+")
    FP8GlobalStateManager.reset()
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=1,
        dynamic_context_parallel=True,
    )
    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    off_interface.reset_instance()
    destroy_num_microbatches_calculator()
    groups = ProcessGroupCollection.use_mpu_process_groups()
    microbatches = 12
    init_num_microbatches_calculator(
        rank=torch.distributed.get_rank(),
        rampup_batch_size=None,
        global_batch_size=microbatches * groups.dp.size(),
        micro_batch_size=1,
        data_parallel_size=groups.dp.size(),
        decrease_batch_size_if_needed=False,
    )
    helper = None
    original_stream = torch.cuda.current_stream()
    graph_stream = torch.cuda.Stream()
    graph_stream.wait_stream(original_stream)
    torch.cuda.set_stream(graph_stream)
    try:
        config = TransformerConfig(
            num_layers=4 if vpp is None else 8,
            hidden_size=1024 if fp8 else 256,
            num_attention_heads=8,
            ffn_hidden_size=512,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=vpp,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            fp8="hybrid" if fp8 else None,
            fp8_recipe="delayed",
            fp8_dot_product_attention=fp8,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            gradient_accumulation_fusion=False,
            attention_backend=AttnBackend.fused,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            num_moe_experts=4 if moe else None,
            moe_grouped_gemm=moe,
            moe_token_dispatcher_type="alltoall",
            dynamic_context_parallel=True,
            calculate_per_token_loss=True,
            sequence_packing_scheduler="default_dynamic_cp",
            max_seqlen_per_dp_cp_rank=128,
            pad_packed_seq_alignment=128,
            thd_max_packed_sequences=2,
            fine_grained_activation_offloading=True,
            offload_modules=["core_attn", "attn_proj"] + (["moe_act"] if moe else []),
            min_offloaded_tensor_size=1,
            cuda_graph_impl="transformer_engine",
            cuda_graph_dynamic_microbatches=True,
            cuda_graph_modules=["attn"],
            cuda_graph_warmup_steps=3,
            use_te_rng_tracker=True,
        )
        models = []
        for vp_stage in range(vpp or 1):
            model = (
                GPTModel(
                    config=config,
                    transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                        num_experts=4 if moe else None, moe_grouped_gemm=moe
                    ),
                    vocab_size=128,
                    max_sequence_length=256,
                    pre_process=parallel_state.is_pipeline_first_stage(
                        ignore_virtual=False, vp_stage=vp_stage
                    ),
                    post_process=parallel_state.is_pipeline_last_stage(
                        ignore_virtual=False, vp_stage=vp_stage
                    ),
                    position_embedding_type="rope",
                    share_embeddings_and_output_weights=False,
                    vp_stage=vp_stage if vpp else None,
                    pg_collection=groups,
                )
                .cuda()
                .train()
            )
            model.model_type = ModelType.encoder_or_decoder
            model.zero_grad_buffer = lambda: None
            models.append(model)
        for _ in range(3):
            eager_losses = _iteration(models, microbatches)
        references = {}
        fp8_states = {}
        for count in (8, microbatches):
            if fp8:
                fp8_states[count] = _snapshot_fp8_state(models)
            losses = _iteration(models, count)
            grads = [
                {
                    name: p.grad.detach().clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                }
                for model in models
            ]
            references[count] = losses, grads
        if fp8:
            _restore_fp8_state(fp8_states[microbatches])
            _iteration(models, microbatches)
            for model, expected in zip(models, references[microbatches][1]):
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        torch.testing.assert_close(
                            parameter.grad,
                            expected[name],
                            rtol=1e-2,
                            atol=1e-6,
                            msg=lambda detail: f"Eager reference changed for {name}: {detail}",
                        )
        helper = TECudaGraphHelper(
            models,
            config,
            seq_length=256,
            micro_batch_size=1,
            pg_collection=groups,
            dynamic_cp_group_getter=parallel_state.get_dynamic_data_context_parallel_groups,
        )
        helper.create_cudagraphs()
        assert helper.graphs_created()
        assert all(
            not getattr(module, "_cuda_graph_rotary_cache_active", False)
            for model in models
            for module in model.modules()
        )
        assert microbatches > helper.num_microbatches, "test must exercise physical-slot wrap"
        graphs = [
            graph
            for layer in helper.flattened_callables
            for bank in layer.cuda_graphs_by_dynamic_cp_size.values()
            for graph in bank
        ]
        assert any(graph._te_cuda_graph_offload_arenas for graph in graphs)
        replays = {}
        for layer in helper.flattened_callables:
            replays[id(layer)] = []
            original_replay = layer._te_cuda_graph_replay

            def record_replay(*args, _layer=layer, _original=original_replay, **kwargs):
                replays[id(_layer)].append(
                    (
                        _layer.current_microbatch % helper.num_microbatches,
                        kwargs["packed_seq_params"].local_cp_size,
                    )
                )
                return _original(*args, **kwargs)

            monkeypatch.setattr(layer, "_te_cuda_graph_replay", record_replay)
        max_grad_error = 0.0
        for count in (microbatches, 8, microbatches):
            eager_losses, eager_grads = references[count]
            if fp8:
                _restore_fp8_state(fp8_states[count])
            graph_losses = _iteration(models, count)
            for model, expected in zip(models, eager_grads):
                actual = {
                    name: p.grad for name, p in model.named_parameters() if p.grad is not None
                }
                assert actual.keys() == expected.keys()
                for name, grad in actual.items():
                    max_grad_error = max(
                        max_grad_error, (grad.float() - expected[name].float()).abs().max().item()
                    )
                    torch.testing.assert_close(
                        grad,
                        expected[name],
                        rtol=1e-2,
                        atol=1e-6,
                        msg=lambda detail: f"{name}: {detail}",
                    )
            if parallel_state.is_pipeline_last_stage():
                for actual, expected in zip(graph_losses, eager_losses):
                    torch.testing.assert_close(
                        actual["lm loss"], expected["lm loss"], rtol=1e-3, atol=1e-3
                    )
        for calls in replays.values():
            assert {size for _, size in calls} == {1, 2}
            assert {size for slot, size in calls if slot == 0} == {1, 2}
        print(
            f"DCP_OFFLOAD_REPLAY_PASS vpp={vpp} moe={moe} fp8={fp8} cp_sizes=[1,2] "
            f"physical_slots={helper.num_microbatches} microbatches=[12,8,12] "
            f"max_grad_abs={max_grad_error}"
        )
    finally:
        if helper is not None:
            helper.delete_cuda_graphs()
        original_stream.wait_stream(graph_stream)
        torch.cuda.set_stream(original_stream)
        destroy_num_microbatches_calculator()
        off_interface.reset_instance()
        # A failed rank must surface its exception while peers may still be in the schedule.
        if sys.exception() is None:
            Utils.destroy_model_parallel()
        FP8GlobalStateManager.reset()
