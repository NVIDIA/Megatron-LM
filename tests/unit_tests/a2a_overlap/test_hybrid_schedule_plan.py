# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.layout import ContextParallelLayoutManager
from megatron.core.models.common.model_chunk_schedule_plan import (
    TransformerLayerSchedulePlan,
    TransformerModelChunkSchedulePlan,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.hybrid.model_chunk_schedule_plan import HybridStackSchedulePlan
from megatron.core.pipeline_parallel.utils import get_comm_stream, get_comp_stream, set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.a2a_overlap.utils import DummyState
from tests.unit_tests.test_utilities import Utils


def _hybrid_stack_stub(cp_layout_manager=None):
    stack = HybridStack.__new__(HybridStack)
    torch.nn.Module.__init__(stack)
    stack._cp_layout_manager = cp_layout_manager
    stack._has_linear_layer_with_chunkwise_cp = False
    return stack


def _overlap_model_stub(**config_overrides):
    config = dict(
        cuda_graph_impl="none", enable_mhc_connections=False, moe_shortcut_connection=False
    )
    config.update(config_overrides)
    model = torch.nn.Module()
    model.config = SimpleNamespace(**config)
    model.decoder = _hybrid_stack_stub()
    return model


@pytest.mark.parametrize("location", ["decoder", "group", "mtp"])
@pytest.mark.parametrize("needs_conversion", [True, False])
def test_hybrid_ep_overlap_checks_nested_cp_layouts(location, needs_conversion):
    """An outer group's boundary layout can hide an inner attention conversion."""
    config = AttentionLayerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
    config.attention_cp_layout = "zigzag" if needs_conversion else "contiguous"
    cp_group = SimpleNamespace(size=lambda: 2)

    def layout_manager(layer_configs):
        return ContextParallelLayoutManager(
            layer_layouts=tuple(
                HybridStack._get_layer_cp_layout(layer_config, "contiguous")
                for layer_config in layer_configs
            ),
            boundary_layout="contiguous",
            sequence_parallel=False,
            cp_group=cp_group,
            tp_group=None,
            tp_cp_group=None,
        )

    model = _overlap_model_stub()
    model.decoder._cp_layout_manager = layout_manager([(config,)])
    assert not model.decoder._cp_layout_manager.requires_conversion
    if location == "group":
        model.decoder.group = _hybrid_stack_stub()
        target = model.decoder.group
    elif location == "mtp":
        model.mtp = _hybrid_stack_stub()
        target = model.mtp
    else:
        target = model.decoder
    target._cp_layout_manager = layout_manager([config])

    if needs_conversion:
        with pytest.raises(ValueError, match="mixed context-parallel layouts"):
            HybridModel._validate_ep_overlap_support(model)
    else:
        HybridModel._validate_ep_overlap_support(model)


@pytest.mark.parametrize("location", ["decoder", "group", "mtp"])
def test_hybrid_ep_overlap_rejects_chunkwise_linear_cp(location):
    """Chunkwise linear CP needs packed-sequence metadata that only HybridStack.forward builds."""
    model = _overlap_model_stub()
    HybridModel._validate_ep_overlap_support(model)
    if location == "group":
        model.decoder.group = _hybrid_stack_stub()
        target = model.decoder.group
    elif location == "mtp":
        model.mtp = _hybrid_stack_stub()
        target = model.mtp
    else:
        target = model.decoder
    target._has_linear_layer_with_chunkwise_cp = True

    with pytest.raises(ValueError, match="linear_cp_mode='chunkwise'"):
        HybridModel._validate_ep_overlap_support(model)


@pytest.mark.parametrize(
    "config_overrides, message",
    [
        (dict(cuda_graph_impl="full_iteration"), "does not support CUDA graphs"),
        (dict(enable_mhc_connections=True), "does not support enable_mhc_connections"),
        (dict(moe_shortcut_connection=True), "moe_shortcut_connection"),
    ],
)
def test_hybrid_ep_overlap_rejects_unsupported_features(config_overrides, message):
    """Direct overlap callables must not bypass CUDA graphs or residual wrappers."""
    with pytest.raises(ValueError, match=message):
        HybridModel._validate_ep_overlap_support(_overlap_model_stub(**config_overrides))


@pytest.mark.parametrize("pattern", ["[*E]", "[+E]"])
def test_grouped_overlap_matches_eager_outputs_and_gradients(pattern):
    """Exercise grouped attention, MLA and shared experts through real A2A nodes."""
    if Utils.world_size < 2:
        pytest.skip("Expert-parallel overlap requires at least two ranks")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = MLATransformerConfig(
            num_layers=2,
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
        block = HybridStack(
            config,
            hybrid_stack_spec.submodules,
            layer_config_list=validate_segment_layers(pattern, config),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        inputs = [torch.randn(16, 1, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        references = []
        for hidden_states in inputs:
            output = block(hidden_states.clone().requires_grad_(), attention_mask=None)
            references.append(output.detach().clone())
            output.backward(torch.ones_like(output))
        reference_grads = {
            name: param.grad.detach().clone()
            for name, param in block.named_parameters()
            if param.grad is not None
        }
        block.zero_grad(set_to_none=True)

        set_streams()
        plans = []
        for _ in inputs:
            state = DummyState()
            state.model = SimpleNamespace(decoder=block)
            plans.append(
                HybridStackSchedulePlan(
                    block.layers[0],
                    torch.cuda.Event(),
                    state,
                    get_comp_stream,
                    get_comm_stream,
                    extra_args={"layer_type": block.layer_type_list[0], "is_last_layer": True},
                )
            )

        outputs = []
        previous = None
        for index, plan in enumerate(plans):
            output, _ = TransformerLayerSchedulePlan.run(
                plan,
                previous,
                f_input=inputs[index].clone().requires_grad_(),
                b_grad=None if previous is None else torch.ones_like(outputs[-1]),
            )
            torch.cuda.synchronize()
            outputs.append(output.detach().clone())
            previous = plan
        TransformerLayerSchedulePlan.run(None, previous, b_grad=torch.ones_like(outputs[-1]))
        torch.cuda.synchronize()

        for output, reference in zip(outputs, references):
            torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)
        for name, param in block.named_parameters():
            if name in reference_grads:
                assert param.grad is not None, name
                torch.testing.assert_close(param.grad, reference_grads[name], rtol=2e-2, atol=2e-2)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("num_depths", [1, 2, 4])
@pytest.mark.parametrize("mtp_pattern", ["[*E]", "[*E][*-]", "*E"])
def test_hybrid_mtp_overlap_matches_eager_outputs_and_gradients(num_depths, mtp_pattern):
    """Expand complete depths, including dense tails, and accumulate shared-head gradients."""
    if Utils.world_size < 2:
        pytest.skip("Expert-parallel overlap requires at least two ranks")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    try:
        model_parallel_cuda_manual_seed(123)
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=128,
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
            expert_model_parallel_size=2,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=True,
            mtp_num_layers=num_depths,
            mtp_loss_scaling_factor=0.7,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=128,
            max_sequence_length=16,
            hybrid_layer_pattern="[*E]" + f"/{mtp_pattern}" * num_depths,
            share_embeddings_and_output_weights=True,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        ).cuda()
        batches = []
        for _ in range(3):
            input_ids = torch.randint(1, 128, (2, 16), device="cuda")
            padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            padding_mask[0, -2:] = True
            padding_mask[1, -4:] = True
            input_mask = ~padding_mask
            input_mask[0, 5] = False
            batches.append(
                dict(
                    input_ids=input_ids,
                    position_ids=torch.arange(16, device="cuda").expand_as(input_ids),
                    labels=torch.randint(1, 128, input_ids.shape, device="cuda"),
                    attention_mask=None,
                    padding_mask=padding_mask,
                    loss_mask=(~padding_mask).float(),
                    mtp_input_mask=input_mask,
                )
            )

        set_streams()
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            references = []
            for batch in batches:
                output = model(**batch)
                references.append(output.detach().float().clone())
                output.sum().backward()
            reference_grads = {
                name: None if param.grad is None else param.grad.detach().float().clone()
                for name, param in model.named_parameters()
            }
            model.zero_grad(set_to_none=True)

            previous_plan = None
            previous_output = None
            for batch, reference in zip(batches, references):
                plan = model.build_schedule_plan(**batch)
                output = TransformerModelChunkSchedulePlan.run(
                    plan,
                    previous_plan,
                    b_grad=None if previous_output is None else torch.ones_like(previous_output),
                )
                torch.testing.assert_close(output, reference, rtol=1e-3, atol=1e-3)
                previous_plan, previous_output = plan, output
            TransformerModelChunkSchedulePlan.run(
                None, previous_plan, b_grad=torch.ones_like(previous_output)
            )
            torch.cuda.synchronize()

            for name, param in model.named_parameters():
                expected = reference_grads[name]
                assert (param.grad is None) == (expected is None), name
                if expected is not None:
                    actual = param.grad.float()
                    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=2e-4, msg=name)
                    relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
                    assert relative_error < 2e-2, (name, relative_error.item())
    finally:
        Utils.destroy_model_parallel()
