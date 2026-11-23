# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.context_parallel.layout import ContextParallelLayoutManager
from megatron.core.models.common.model_chunk_schedule_plan import (
    LayerScheduleGroup,
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.hybrid.fine_grained_callables import validate_overlap_schedule
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.models.hybrid.external_layer import ExternalGainConfig
from tests.unit_tests.test_utilities import Utils


def _overlap_model_stub(
    *, static_cp_size=1, layer_cp_layouts=("contiguous", "contiguous"), **config_overrides
):
    config = dict(
        cuda_graph_impl="none",
        enable_mhc_connections=False,
        moe_shortcut_connection=False,
        wide_residual=None,
        recompute_granularity=None,
        dynamic_context_parallel=False,
        sequence_parallel=False,
    )
    config.update(config_overrides)
    config = SimpleNamespace(**config)
    decoder = SimpleNamespace(
        config=config,
        layer_config_list=[],
        boundary_layout="contiguous",
        layer_cp_layouts=layer_cp_layouts,
        _cp_layout_manager=None,
        _has_linear_layer_with_chunkwise_cp=False,
        cp_group=SimpleNamespace(size=lambda: static_cp_size),
        tp_group=None,
        tp_cp_group=None,
    )
    if static_cp_size > 1:
        decoder._cp_layout_manager = ContextParallelLayoutManager(
            layer_layouts=decoder.layer_cp_layouts,
            boundary_layout=decoder.boundary_layout,
            sequence_parallel=config.sequence_parallel,
            cp_group=decoder.cp_group,
            tp_group=decoder.tp_group,
            tp_cp_group=decoder.tp_cp_group,
        )
    return SimpleNamespace(config=config, _has_custom_layers=False, decoder=decoder)


@pytest.mark.parametrize(
    "static_cp_size, dynamic_cp, reject",
    [(1, False, False), (2, False, True), (1, True, True)],
    ids=["static_cp1", "static_cp2", "dynamic_from_cp1"],
)
@pytest.mark.parametrize(
    "layer_cp_layouts, chunkwise, message",
    [
        (("zigzag", "contiguous"), False, "mixed context-parallel layouts"),
        (("contiguous", "contiguous"), True, "linear_cp_mode='chunkwise'"),
    ],
    ids=["mixed_layout", "chunkwise"],
)
def test_hybrid_ep_overlap_cp_compatibility(
    static_cp_size, dynamic_cp, reject, layer_cp_layouts, chunkwise, message
):
    """Physical layout conversion and chunkwise metadata are needed only when CP may be active."""
    model = _overlap_model_stub(
        static_cp_size=static_cp_size,
        layer_cp_layouts=layer_cp_layouts,
        dynamic_context_parallel=dynamic_cp,
    )
    model.decoder._has_linear_layer_with_chunkwise_cp = chunkwise
    with pytest.raises(ValueError, match=message) if reject else nullcontext():
        validate_overlap_schedule(model)


@pytest.mark.parametrize(
    "static_cp_size, runtime_cp_size",
    [(2, None), (1, 1), (1, 2)],
    ids=["static_cp2", "dynamic_runtime_cp1", "dynamic_runtime_cp2"],
)
def test_hybrid_ep_overlap_accepts_uniform_headwise_cp(static_cp_size, runtime_cp_size):
    model = _overlap_model_stub(
        static_cp_size=static_cp_size, dynamic_context_parallel=runtime_cp_size is not None
    )
    validate_overlap_schedule(model)

    packed = None
    if runtime_cp_size is not None:
        packed = PackedSeqParams(
            local_cp_size=runtime_cp_size, cp_group=SimpleNamespace(size=lambda: runtime_cp_size)
        )
    group, manager = HybridStack._resolve_runtime_cp_layout(model.decoder, packed)
    assert group.size() == (runtime_cp_size or static_cp_size)
    assert manager is None or not manager.requires_conversion


@pytest.mark.parametrize(
    "config_overrides, message",
    [
        (dict(cuda_graph_impl="full_iteration"), "does not support CUDA graphs"),
        (dict(enable_mhc_connections=True), "does not support enable_mhc_connections"),
        (dict(moe_shortcut_connection=True), "moe_shortcut_connection"),
        (dict(recompute_granularity="full"), "recompute_granularity='full'"),
    ],
)
def test_hybrid_ep_overlap_rejects_unsupported_features(config_overrides, message):
    """Direct overlap callables must not bypass CUDA graphs or residual wrappers."""
    with pytest.raises(ValueError, match=message):
        validate_overlap_schedule(_overlap_model_stub(**config_overrides))


@pytest.mark.parametrize("wide_on_model", [False, True])
def test_hybrid_ep_overlap_rejects_wide_residual(wide_on_model):
    """First-class layer configs can be constructed separately from the model's overlap config."""
    model = _overlap_model_stub(wide_residual=object() if wide_on_model else None)
    model.decoder.layer_config_list = [
        SimpleNamespace(wide_residual=None if wide_on_model else object())
    ]
    with pytest.raises(ValueError, match="overlap_moe_expert_parallel_comm.*wide_residual"):
        validate_overlap_schedule(model)


def test_hybrid_ep_overlap_rejects_custom_layers_at_construction():
    """An external layer can override forward without implementing the overlap slots."""
    if Utils.world_size < 2:
        pytest.skip("Expert-parallel overlap requires at least two ranks")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            bf16=True,
            num_moe_experts=4,
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=True,
        )
        with pytest.raises(ValueError, match="Custom hybrid layers.*overlap_moe"):
            HybridModel(
                config=config,
                hybrid_layer_config_list=[ExternalGainConfig.from_config(config)],
                vocab_size=32,
                max_sequence_length=8,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
    finally:
        Utils.destroy_model_parallel()


def test_empty_intermediate_overlap_chunk_preserves_received_input():
    """Pseudo-deallocating a sent activation must preserve its received input and gradient."""
    if Utils.world_size < 4:
        pytest.skip("PP=2 with EP=2 requires four ranks")
    Utils.initialize_model_parallel(
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
    )
    try:
        config = MLATransformerConfig(
            num_layers=4,
            hidden_size=16,
            num_attention_heads=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            pipeline_dtype=torch.bfloat16,
            bf16=True,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=2,
            num_moe_experts=2,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=True,
            deallocate_pipeline_outputs=True,
        )
        groups = ProcessGroupCollection.use_mpu_process_groups()
        # Each PP rank owns one empty intermediate chunk in this VPP-major pattern.
        model = HybridModel(
            config=config,
            hybrid_layer_pattern="[*E]|||[*E]",
            vp_stage=1 - groups.pp.rank(),
            pg_collection=groups,
            pre_process=False,
            post_process=False,
            vocab_size=32,
            max_sequence_length=4,
        )
        expected = torch.arange(
            4 * 2 * config.hidden_size, device="cuda", dtype=config.pipeline_dtype
        ).reshape(4, 2, config.hidden_size)
        received = expected.clone().requires_grad_()
        model.set_input_tensor(received)
        set_streams()
        plan = model.build_schedule_plan(input_ids=None, position_ids=None, attention_mask=None)
        assert plan.num_layers() == 0
        output = TransformerModelChunkSchedulePlan.run(plan, None)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

        deallocate_output_tensor(output, config.deallocate_pipeline_outputs)
        assert output.numel() == 1
        assert output is not received
        torch.testing.assert_close(received, expected, rtol=0, atol=0)

        upstream_grad = expected.flip(0) - 5
        TransformerModelChunkSchedulePlan.run(None, plan, b_grad=upstream_grad)
        torch.testing.assert_close(received.grad, upstream_grad, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("pattern, num_layers", [("[M-][*E]", 4), ("[+E]", 2), ("*E", 2)])
def test_grouped_overlap_matches_eager_outputs_and_gradients(pattern, num_layers, monkeypatch):
    """Compare the grouped-pattern and flat config-list 1F1B paths with eager training."""
    if Utils.world_size < 2:
        pytest.skip("Expert-parallel overlap requires at least two ranks")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = MLATransformerConfig(
            num_layers=num_layers,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=256,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=False,
            num_moe_experts=4,
            moe_grouped_gemm=True,
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_shared_expert_intermediate_size=256,
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=True,
            multi_latent_attention="+" in pattern,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
        )
        layers = (
            {"hybrid_layer_pattern": pattern}
            if "[" in pattern
            else {"hybrid_layer_config_list": validate_segment_layers(pattern, config)}
        )
        model = HybridModel(
            config=config,
            **layers,
            vocab_size=128,
            max_sequence_length=16,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        positions = torch.arange(16, device="cuda").unsqueeze(0)
        microbatches = [
            dict(
                input_ids=torch.randint(128, (1, 16), device="cuda"),
                position_ids=positions,
                attention_mask=None,
                labels=torch.randint(128, (1, 16), device="cuda"),
            )
            for _ in range(3)
        ]
        if pattern == "*E":

            def scaled_loss(
                *,
                hidden_states,
                output_layer,
                output_weight,
                compute_language_model_loss,
                scale_logits,
                labels,
                context,
                **kwargs,
            ):
                logits, _ = output_layer(hidden_states, weight=output_weight)
                return compute_language_model_loss(labels, scale_logits(logits)) * context

            for index, data in enumerate(microbatches):
                data.update(output_processor=scaled_loss, output_processor_context=index + 2)

        references = []
        for data in microbatches:
            output = model(**data)
            references.append(output.detach().clone())
            output.backward(torch.ones_like(output))
        reference_grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
        model.zero_grad(set_to_none=True)

        # Planning each microbatch must not reconstruct the legacy full-stack symbol list.
        monkeypatch.delattr(HybridStack, "layer_type_list")
        set_streams()
        outputs = []
        previous = None
        for data in microbatches:
            plan = model.build_schedule_plan(**data)
            groups = [plan.get_layer(i).layer.layers for i in range(plan.num_layers())]
            assert [layer for group in groups for layer in group] == list(model.decoder.layers)
            assert [len(group) for group in groups] == (
                [2] * pattern.count("[") if "[" in pattern else [1] * num_layers
            )
            output = TransformerModelChunkSchedulePlan.run(
                plan, previous, b_grad=None if previous is None else torch.ones_like(outputs[-1])
            )
            outputs.append(output.detach().clone())
            previous = plan
        TransformerModelChunkSchedulePlan.run(None, previous, b_grad=torch.ones_like(outputs[-1]))

        for output, reference in zip(outputs, references):
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
        for name, param in model.named_parameters():
            if name in reference_grads:
                assert param.grad is not None, name
                torch.testing.assert_close(param.grad, reference_grads[name], rtol=2e-2, atol=2e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("include_nested_units", [False, True])
def test_fsdp_release_hooks_target_each_physical_layer(monkeypatch, include_nested_units):
    """The bracket has no parameters. Release the physical units, including v2 expert units."""

    class Unit(torch.nn.Module):
        pass

    monkeypatch.setattr("megatron.core.models.common.model_chunk_schedule_plan.FsdpModule", Unit)
    layers = (Unit(), Unit())
    layers[1].experts = Unit()
    plan = TransformerLayerSchedulePlan.__new__(TransformerLayerSchedulePlan)
    plan.layer = LayerScheduleGroup(layers, [], {}, True, 1)
    plan.pre_dispatch_computation = Mock()
    plan.moe_combine = Mock()
    releases = []
    plan.set_fsdp_reshard_hooks(
        lambda layer: releases.append(("forward", layer)),
        lambda layer: releases.append(("backward", layer)),
        include_nested_units=include_nested_units,
    )
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    plan.moe_combine.set_post_forward_hook.call_args.args[0]()
    plan.pre_dispatch_computation.set_post_backward_hook.call_args.args[0]()
    forward_units = (*layers, layers[1].experts) if include_nested_units else layers
    assert releases == [("forward", unit) for unit in forward_units] + [
        ("backward", layers[1]),
        ("backward", layers[0]),
    ]


def test_packed_moe_overlap_routes_per_sample_like_eager(monkeypatch):
    """A packed batch flattened to [mbs*s, 1, h] reaches the router per sample on both paths.

    The per-sample load-balancing loss reaches the router weight gradient, so routing the
    flattened [mbs*s, 1, h] tensor as one sequence would change that gradient.
    """
    if Utils.world_size < 2:
        pytest.skip("Expert-parallel overlap requires at least two ranks")
    # Direct backward initializes a scalar aux-loss scale outside the training loop.
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", None)
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        micro_batch_size, seq_length, hidden_size = 2, 8, 256
        config = MLATransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            ffn_hidden_size=256,
            bf16=True,
            params_dtype=torch.bfloat16,
            use_cpu_initialization=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            add_bias_linear=False,
            num_moe_experts=4,
            moe_grouped_gemm=True,
            moe_router_topk=2,
            moe_router_dtype="fp32",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=True,
        )
        model = HybridModel(
            config=config,
            hybrid_layer_pattern="E",
            vocab_size=128,
            max_sequence_length=seq_length,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()

        def decoder_output(*, hidden_states, **kwargs):
            # Compare the normalized packed activations on both model paths.
            return hidden_states.float()

        kwargs = dict(
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            packed_seq_params=PackedSeqParams(tokens_per_sample=seq_length),
            output_processor=decoder_output,
        )
        inputs = [
            torch.randn(
                micro_batch_size * seq_length, 1, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            for _ in range(2)
        ]
        references = []
        for hidden_states in inputs:
            output = model(decoder_input=hidden_states.clone().requires_grad_(), **kwargs)
            references.append(output.detach().clone())
            output.backward(torch.ones_like(output))
        reference_grads = {
            name: param.grad.detach().clone()
            for name, param in model.decoder.named_parameters()
            if param.grad is not None
        }
        model.zero_grad(set_to_none=True)

        set_streams()
        outputs = []
        previous = None
        for hidden_states in inputs:
            plan = model.build_schedule_plan(
                decoder_input=hidden_states.clone().requires_grad_(), **kwargs
            )
            output = TransformerModelChunkSchedulePlan.run(
                plan, previous, b_grad=None if previous is None else torch.ones_like(outputs[-1])
            )
            outputs.append(output.detach().clone())
            previous = plan
        TransformerModelChunkSchedulePlan.run(None, previous, b_grad=torch.ones_like(outputs[-1]))

        for output, reference in zip(outputs, references):
            assert output.shape == reference.shape
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
        router_grads = [name for name in reference_grads if name.endswith("router.weight")]
        assert router_grads
        assert "final_norm.weight" in reference_grads
        for name, param in model.decoder.named_parameters():
            if name in reference_grads:
                assert param.grad is not None, name
                torch.testing.assert_close(param.grad, reference_grads[name], rtol=2e-2, atol=2e-2)
    finally:
        Utils.destroy_model_parallel()
