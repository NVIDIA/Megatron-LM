# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import patch

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
from megatron.core.models.hybrid.model_chunk_schedule_plan import (
    HybridStackModelChunkSchedulePlan,
    HybridStackSchedulePlan,
)
from megatron.core.pipeline_parallel.utils import get_comm_stream, get_comp_stream, set_streams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig
from tests.unit_tests.a2a_overlap.utils import DummyState
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("location", ["decoder", "group", "mtp"])
@pytest.mark.parametrize("needs_conversion", [True, False])
def test_hybrid_schedule_checks_nested_cp_layouts(location, needs_conversion):
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

    model = torch.nn.Module()
    model.config = SimpleNamespace(cuda_graph_impl="none")
    model.decoder = torch.nn.Module()
    model.decoder._cp_layout_manager = layout_manager([(config,)])
    assert not model.decoder._cp_layout_manager.requires_conversion
    if location == "group":
        model.decoder.group = torch.nn.Module()
        target = model.decoder.group
    elif location == "mtp":
        model.mtp = torch.nn.Module()
        target = model.mtp
    else:
        target = model.decoder
    target._cp_layout_manager = layout_manager([config])

    with patch.object(TransformerModelChunkSchedulePlan, "__init__", return_value=None) as build:
        if needs_conversion:
            with pytest.raises(AssertionError, match="mixed context-parallel layouts"):
                HybridStackModelChunkSchedulePlan(model)
            build.assert_not_called()
        else:
            HybridStackModelChunkSchedulePlan(model)
            build.assert_called_once()


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
