# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay capped Shortcut MoE expert GEMMs with changing live tokens."""

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import (
    wide_residual_gated_delta_product_inference_stack_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NVLSAllGatherVDispatcher,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.wide_residual_config import WideResidualConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.fixture
def wide_inference_layers():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_pre_softmax=True,
            moe_router_dtype="fp32",
            moe_shortcut_connection=True,
            expert_tensor_parallel_size=1,
            transformer_impl="inference_optimized",
            inference_shortcut_moe_overlap=True,
            inference_grouped_gemm_backend="vllm",
            inference_moe_token_dispatcher_type="nvls",
            normalization="RMSNorm",
            add_bias_linear=False,
            activation_func=squared_relu,
            params_dtype=torch.bfloat16,
            bf16=True,
            flash_attention_version=2,
            attention_output_gate=True,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            wide_residual=WideResidualConfig(num_streams=3, learned_retention=True),
        )
        submodules = wide_residual_gated_delta_product_inference_stack_spec.submodules

        def layer(name, number):
            return (
                build_module(
                    getattr(submodules, name),
                    config=config,
                    layer_number=number,
                    add_layer_offset=False,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                )
                .cuda()
                .eval()
            )

        yield config, layer
    finally:
        Utils.destroy_model_parallel()


def test_capped_expert_graph_covers_live_tiles(wide_inference_layers, monkeypatch):
    """Capped experts match uncapped serial as a graph's live-token prefix changes."""
    config, build_layer = wide_inference_layers
    experts = build_layer("moe_layer", 2).mlp.experts
    tokens = 64
    hidden = torch.randn(tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    probs = torch.rand(tokens, 2, device="cuda")
    # Include non-local experts and more tiles than the two-CTA launch cap.
    routes = torch.arange(tokens * 2, device="cuda").reshape(tokens, 2) % 6
    valid = torch.tensor(tokens, device="cuda", dtype=torch.int32)
    output = torch.empty(tokens, config.hidden_size, device="cuda", dtype=torch.float32)
    monkeypatch.setattr(InferenceAllGatherDispatcherBase, "_valid_tokens", lambda: valid)
    monkeypatch.setattr(
        InferenceAllGatherDispatcherBase, "_get_host_valid_tokens_estimate", lambda: 4
    )
    monkeypatch.setattr(NVLSAllGatherVDispatcher, "_get_rsv_tensor", lambda: output)

    def run(hidden, probs):
        result, bias = experts(hidden, None, probs, routing_map=routes, inference_max_blocks=2)
        assert bias is None
        return result

    with torch.inference_mode(), InferenceMode.active():
        assert_replays_bit_exact(
            run, (hidden, probs), backward=False, contention=True, what="Shortcut capped experts"
        )
        valid.fill_(9)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(hidden, probs)
        # The captured launch keeps its two-CTA cap; eager calls use ordinary grids.
        for count in (9, tokens, 0, 33):
            valid.fill_(count)
            output.fill_(float("nan"))
            graph.replay()
            actual = captured.clone()
            expected, bias = experts(hidden, None, probs, routing_map=routes)
            assert bias is None
            torch.testing.assert_close(actual[:count], expected[:count], atol=0, rtol=0)
            assert torch.isnan(actual[count:]).all()
