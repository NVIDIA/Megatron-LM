# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exercise the automatic hybrid boundary fusion with native branch modules."""

import os
from copy import deepcopy

import pytest
import torch

from megatron.core.activations import squared_relu
from megatron.core.inference.config import InferenceConfig, MambaInferenceStateConfig
from megatron.core.inference.contexts import DynamicInferenceContext, StaticInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.ops.normalization.backends import select_bf16_add_rmsnorm
from megatron.core.ops.normalization.bf16_add_rmsnorm import fused_bf16_add_rmsnorm
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.inference_layers import InferenceLayerNormColumnParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.moe.token_dispatcher_inference import (
    InferenceAllGatherDispatcherBase,
    NCCLAllGatherDispatcher,
)
from tests.unit_tests.test_utilities import Utils, reset_cuda_graph_global_state


@pytest.fixture
def model_parallel(request):
    """Keep graph and inference state isolated from the rest of the test suite."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if select_bf16_add_rmsnorm(2688, torch.bfloat16, False) is None:
        pytest.skip("requires the validated native TE BF16 RMSNorm backend")
    Utils.initialize_model_parallel(getattr(request, "param", 1), 1)
    model_parallel_cuda_manual_seed(123, inference_rng_tracker=True, force_reset_rng=True)
    previous_valid_tokens = InferenceAllGatherDispatcherBase._valid_tokens_tensor
    previous_host_tokens = InferenceAllGatherDispatcherBase._host_valid_tokens_estimate
    NCCLAllGatherDispatcher.allocate_buffers()
    yield
    InferenceMode.unset_active()
    reset_cuda_graph_global_state()
    InferenceAllGatherDispatcherBase._valid_tokens_tensor = previous_valid_tokens
    InferenceAllGatherDispatcherBase._host_valid_tokens_estimate = previous_host_tokens
    Utils.destroy_model_parallel()


def build_model(pattern, graphs, flash_decode=True, hidden=2688):
    """Use production layer classes with small attention/SSM/expert inner widths."""
    config = TransformerConfig(
        num_layers=len(pattern),
        hidden_size=hidden,
        num_attention_heads=16,
        kv_channels=64,
        ffn_hidden_size=256,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        transformer_impl="inference_optimized",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        add_bias_linear=False,
        add_qkv_bias=False,
        activation_func=squared_relu,
        num_moe_experts=4,
        moe_ffn_hidden_size=256,
        moe_shared_expert_intermediate_size=256,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        inference_moe_token_dispatcher_type="nccl",
        mamba_num_heads=8,
        mamba_num_groups=4,
        mamba_head_dim=64,
        mamba_state_dim=16,
        cuda_graph_impl="local" if graphs else "none",
        flash_decode=flash_decode,
        use_cpu_initialization=True,
    )
    model = (
        HybridModel(
            config,
            hybrid_inference_stack_spec,
            vocab_size=128,
            max_sequence_length=32,
            hybrid_layer_pattern=pattern,
        )
        .cuda()
        .eval()
    )
    assert model.decoder._inference_residual_norm_plan is not None
    return model


def assert_bits(actual, expected):
    """The optimized path must preserve the native inference output bits."""
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("pattern", ["M*E", "EME", "EE"])
@pytest.mark.parametrize("hidden", [2688, 8192])
@torch.no_grad()
def test_default_hybrid_matches_native(model_parallel, pattern, hidden):
    """Cover Mamba/attention/MoE handoffs, consecutive MoEs and the final norm."""
    model = build_model(pattern, graphs=False, hidden=hidden)
    stack = model.decoder
    plan = stack._inference_residual_norm_plan
    keys = tuple(model.state_dict())
    x = torch.randn(8, 1, hidden, device="cuda", dtype=torch.bfloat16)
    native_context = StaticInferenceContext(max_batch_size=1, max_sequence_length=32)
    fused_context = StaticInferenceContext(max_batch_size=1, max_sequence_length=32)
    with InferenceMode.active():
        stack._inference_residual_norm_plan = None
        expected = stack(x.clone(), attention_mask=None, inference_context=native_context).clone()
        stack._inference_residual_norm_plan = plan
        assert plan.can_run(
            x, training=False, inference_context=fused_context, packed_seq_params=None
        )
        actual = stack(x.clone(), attention_mask=None, inference_context=fused_context)
    assert_bits(actual, expected)
    assert tuple(model.state_dict()) == keys
    assert not plan.can_run(
        x, training=True, inference_context=fused_context, packed_seq_params=None
    )
    assert not plan.can_run(
        x, training=False, inference_context=fused_context, packed_seq_params=None
    )


@torch.no_grad()
def test_dynamic_prefill_and_decode(model_parallel):
    """Compare actual dynamic attention/SSM metadata and caches through decode."""
    model = build_model("M*E", graphs=False, flash_decode=False)
    stack = model.decoder
    plan = stack._inference_residual_norm_plan
    contexts = [
        DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=256,
                buffer_size_gb=0.1,
                max_requests=4,
                max_tokens=256,
                block_size_tokens=256,
                num_cuda_graphs=0,
                mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
            ),
        )
        for _ in range(2)
    ]
    for context in contexts:
        context.add_request(
            DynamicInferenceRequest(
                request_id=0,
                prompt_tokens=torch.arange(8, device="cuda"),
                sampling_params=SamplingParams(num_tokens_to_generate=4),
            )
        )
    with InferenceMode.active():
        for step in range(4):
            outputs = []
            x = None
            for fused, context in zip((False, True), contexts):
                context.initialize_attention_state()
                if x is None:
                    x = torch.randn(
                        context.padded_active_token_count,
                        1,
                        2688,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                stack._inference_residual_norm_plan = plan if fused else None
                outputs.append(
                    stack(x.clone(), attention_mask=None, inference_context=context).clone()
                )
            # Prefill kernels need not initialize dummy token outputs. Compare
            # real tokens, then advance both caches through the same decodes.
            active_tokens = contexts[0].active_token_count
            assert active_tokens == contexts[1].active_token_count > 0
            assert_bits(outputs[1][:active_tokens], outputs[0][:active_tokens])
            for context in contexts:
                context.update_requests(
                    active_requests_mask=torch.tensor([1]), new_tokens=torch.tensor([step + 8])
                )
    stack._inference_residual_norm_plan = plan


def restore_cache(context, snapshot):
    """Restore tensor contents without replacing addresses captured by a graph."""
    for key, values in snapshot.items():
        for target, value in zip(context.key_value_memory_dict[key], values):
            target.copy_(value)


@pytest.mark.parametrize(
    "pattern,hidden", [("M", 2688), ("*", 2688), ("E", 2688), ("M*E", 2688), ("M*E", 8192)]
)
@torch.no_grad()
def test_local_graph_handoffs_and_weight_refit(model_parallel, pattern, hidden):
    """Replay all three native layer types after input changes and norm refits."""
    reference = build_model(pattern, graphs=False, hidden=hidden)
    reference.decoder._inference_residual_norm_plan = None
    fused = build_model(pattern, graphs=True, hidden=hidden)
    fused.load_state_dict(reference.state_dict())
    # Loading weights after eval must refresh Mamba's derived decode cache.
    fused.train().eval()
    contexts = [StaticInferenceContext(max_batch_size=1, max_sequence_length=32) for _ in range(2)]
    x = torch.randn(8, 1, hidden, device="cuda", dtype=torch.bfloat16)
    with InferenceMode.active():
        outputs = [
            model.decoder(x.clone(), attention_mask=None, inference_context=context).clone()
            for model, context in zip((reference, fused), contexts)
        ]
        assert_bits(outputs[1], outputs[0])
        for context in contexts:
            context.sequence_len_offset = 8
            context.enable_decode_mode()
        snapshot = deepcopy(contexts[1].key_value_memory_dict)
        decode = torch.randn(1, 1, hidden, device="cuda", dtype=torch.bfloat16)
        # Capture/warmup mutates the caches. Restore contents before the matched
        # comparison, retaining the captured cache pointers.
        fused.decoder(decode.clone(), attention_mask=None, inference_context=contexts[1])
        restore_cache(contexts[1], snapshot)
        for step in range(3):
            decode = torch.randn_like(decode)
            if step == 1:
                for name, weight in reference.named_parameters():
                    if "norm" in name and name.endswith("weight"):
                        weight.add_(0.125)
                fused.load_state_dict(reference.state_dict())
            outputs = [
                model.decoder(
                    decode.clone(), attention_mask=None, inference_context=context
                ).clone()
                for model, context in zip((reference, fused), contexts)
            ]
            assert_bits(outputs[1], outputs[0])
            for context in contexts:
                context.sequence_len_offset += 1


@pytest.mark.parametrize("model_parallel", [4], indirect=True)
@pytest.mark.parametrize("disable_nvls", [True, False])
@torch.no_grad()
def test_normalized_projection_preserves_tp_gather(model_parallel, disable_nvls):
    """Compare native and fused boundaries through an actual four-rank gather."""
    groups = ProcessGroupCollection.use_mpu_process_groups()
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2688,
        num_attention_heads=16,
        tensor_model_parallel_size=4,
        sequence_parallel=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        normalization="RMSNorm",
        add_bias_linear=False,
        transformer_impl="inference_optimized",
        activation_func=squared_relu,
        inference_disable_triton_nvls_kernels=disable_nvls,
    )
    projection = (
        InferenceLayerNormColumnParallelLinear(
            2688,
            512,
            config=config,
            init_method=config.init_method,
            gather_output=False,
            bias=False,
            skip_bias_add=True,
            is_expert=False,
            tp_group=groups.tp,
            pg_collection=groups,
        )
        .cuda()
        .bfloat16()
        .eval()
    )
    projection.set_barrier_before_all_gather()
    x = torch.randn(8, 1, 2688, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    expected, _ = projection(x + residual)
    expected = expected.clone()

    def run():
        normalized, _ = fused_bf16_add_rmsnorm(x, residual, projection.layer_norm_weight)
        return projection.forward_normalized(normalized)[0]

    assert_bits(run(), expected)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(3):
        x.normal_()
        residual.normal_()
        expected, _ = projection(x + residual)
        expected = expected.clone()
        graph.replay()
        assert_bits(actual, expected)
