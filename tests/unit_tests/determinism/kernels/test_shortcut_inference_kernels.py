# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the cache-aware GDP, attention and expert adapters used by Shortcut MoE."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.batch_dimensions_utils import InferenceBatchDimensions
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
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
from tests.unit_tests.ssm.test_gdp_dynamic_inference import _build_model, requires_gdp_model
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.internal,
    *requires_gdp_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


class _DecodeContext:
    """The dynamic-context fields read by a one-token GDP decode step."""

    def __init__(self, conv_state, ssm_state, batch_indices, layer_number):
        batch_size = batch_indices.numel()
        self.padded_batch_dimensions = InferenceBatchDimensions(
            token_count=batch_size, decode_req_count=batch_size
        )
        self.num_speculative_tokens = 0
        self.mamba_metadata = SimpleNamespace(batch_indices_decode=batch_indices)
        self.conv_state = conv_state
        self.ssm_state = ssm_state
        self.layer_number = layer_number

    def is_dynamic_batching(self):
        return True

    def is_decode_only(self):
        return self.padded_batch_dimensions.prefill_req_count == 0

    def mamba_states_cache(self, layer_number, intermediate=False):
        assert layer_number == self.layer_number
        assert not intermediate
        return self.conv_state, self.ssm_state


@pytest.fixture
def decode_fixture():
    Utils.initialize_model_parallel(1, 1)
    try:
        model = _build_model(tp=1)
        layer = next(layer for layer in model.decoder.layers if hasattr(layer, "mixer"))
        # The fourth row is a graph-padding request; slot 1 is deliberately unused.
        indices = torch.tensor([2, 0, 3, -1], dtype=torch.int64, device="cuda")
        conv, ssm = layer.mixer.allocate_inference_cache(batch_size=4, max_seqlen=128)
        torch.manual_seed(1234)
        conv.normal_(std=0.05)
        ssm.normal_(std=0.05)
        hidden = torch.randn((4, 1, model.config.hidden_size), device="cuda", dtype=torch.bfloat16)
        layer_number = layer.mixer.layer_number - layer.mixer.pp_layer_offset

        def context():
            return _DecodeContext(conv.clone(), ssm.clone(), indices, layer_number)

        yield layer, hidden, context
    finally:
        Utils.destroy_model_parallel()


def test_staged_gdp_output_and_cache_replay(decode_fixture):
    """Fresh cache snapshots must give identical output and state under stream contention."""
    layer, hidden, make_context = decode_fixture

    def run(hidden):
        context = make_context()
        projection = layer.inference_project(hidden, context)
        output = layer.inference_post(projection, layer.inference_core(projection, context))
        return output, context.conv_state, context.ssm_state

    with torch.inference_mode(), InferenceMode.active():
        serial_context = make_context()
        expected = layer(hidden, inference_context=serial_context)
        actual, conv_state, ssm_state = run(hidden)
        for staged, serial in (
            (actual, expected),
            (conv_state, serial_context.conv_state),
            (ssm_state, serial_context.ssm_state),
        ):
            torch.testing.assert_close(staged, serial, atol=5e-3, rtol=5e-3)
        assert_replays_bit_exact(
            run, (hidden,), backward=False, contention=True, what="Shortcut GDP decode"
        )


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


def test_staged_attention_output_and_kv_cache_replay(wide_inference_layers):
    """Preserve RoPE, KV writes and wide residuals in prefill and padded decode replay."""
    config, build_layer = wide_inference_layers
    layer = build_layer("attention_layer", 1)
    real_requests = 3
    rope = RotaryEmbedding(
        kv_channels=config.kv_channels, rotary_percent=1.0, use_cpu_initialization=True
    )(64)

    def make_context():
        context = DynamicInferenceContext(
            model_config=config,
            inference_config=InferenceConfig(
                max_sequence_length=64,
                buffer_size_gb=0.0625,
                block_size_tokens=256,
                max_requests=4,
                max_tokens=64,
                num_cuda_graphs=1,
                cuda_graph_mixed_prefill_count=0,
                materialize_only_last_token_logits=False,
                use_flashinfer_fused_rope=False,
            ),
        )
        for request_id in range(real_requests):
            context.add_request(
                DynamicInferenceRequest(
                    request_id=request_id,
                    prompt_tokens=torch.arange(8, device="cuda"),
                    sampling_params=SamplingParams(num_tokens_to_generate=2),
                )
            )
        context.initialize_attention_state()
        return context

    serial_context, staged_context = make_context(), make_context()
    blocks = {
        context: context.request_to_kv_block_ids[:real_requests, 0].tolist()
        for context in (serial_context, staged_context)
    }

    def cache(context, length):
        return torch.stack(
            [context.memory_buffer[:, 0, block, :length] for block in blocks[context]], dim=1
        )

    with torch.inference_mode(), InferenceMode.active():
        for length in (8, 9):
            hidden = torch.randn(
                staged_context.padded_active_token_count,
                1,
                384,
                device="cuda",
                dtype=torch.bfloat16,
            )
            serial, _ = layer(hidden, None, inference_context=serial_context, rotary_pos_emb=rope)
            expected_cache = cache(serial_context, length)

            def run(hidden):
                projected = layer.forward_inference_project(
                    hidden, None, inference_context=staged_context, rotary_pos_emb=rope
                )
                output, _ = layer.forward_inference_post_core(
                    layer.forward_inference_core(projected)
                )
                return output[: staged_context.active_token_count], cache(staged_context, length)

            output, kv_cache = run(hidden)
            torch.testing.assert_close(
                output, serial[: staged_context.active_token_count], atol=0, rtol=0
            )
            torch.testing.assert_close(kv_cache, expected_cache, atol=0, rtol=0)
            assert_replays_bit_exact(
                run,
                (hidden,),
                backward=False,
                contention=True,
                what="Shortcut attention and KV cache",
            )
            if length == 8:
                for context in (serial_context, staged_context):
                    context.update_requests(
                        active_requests_mask=torch.ones(real_requests, dtype=torch.int32),
                        new_tokens=torch.full((real_requests,), 11, dtype=torch.long),
                    )
                    context.initialize_attention_state()
            else:
                assert staged_context.padded_batch_dimensions.decode_req_count == 4
                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    for _ in range(2):
                        run(hidden)
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured, captured_cache = run(hidden)
                graph.replay()
                torch.testing.assert_close(captured, output, atol=0, rtol=0)
                torch.testing.assert_close(captured_cache, expected_cache, atol=0, rtol=0)


def test_capped_expert_adapter_graph_covers_live_tiles(wide_inference_layers, monkeypatch):
    """Capped expert stages match serial as a graph's live-token prefix changes."""
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
        prepared = experts.prepare_shortcut_overlap(hidden, probs, routes, max_blocks=2)
        assert prepared.grid_size_fc1 == prepared.grid_size_fc2 == 2
        result, bias = experts.finish_shortcut_overlap(experts.compute_shortcut_overlap(prepared))
        assert bias is None
        return result

    with torch.inference_mode(), InferenceMode.active():
        uncapped = experts.prepare_shortcut_overlap(hidden, probs, routes, max_blocks=0)
        assert uncapped.grid_size_fc1 > 2 and uncapped.grid_size_fc2 > 2
        assert_replays_bit_exact(
            run, (hidden, probs), backward=False, contention=True, what="Shortcut capped experts"
        )
        valid.fill_(9)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(hidden, probs)
        for count in (9, tokens, 0, 33):
            valid.fill_(count)
            output.fill_(float("nan"))
            graph.replay()
            actual = captured.clone()
            expected, bias = experts(hidden, None, probs, routing_map=routes)
            assert bias is None
            torch.testing.assert_close(actual[:count], expected[:count], atol=0, rtol=0)
            assert torch.isnan(actual[count:]).all()
