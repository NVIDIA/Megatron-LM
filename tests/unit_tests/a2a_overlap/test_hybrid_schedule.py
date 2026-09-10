# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import pytest
import torch

from megatron.core.enums import Fp8Recipe
from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_dsv4_stack_spec, hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import (
    _get_all_rng_states,
    _set_all_rng_states,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import set_current_microbatch
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_logging import destroy_moe_metrics_tracker
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _make_config,
)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mhc", [False, True])
def test_hybrid_dsv4_schedule_loss_and_grad_parity(mhc):
    """Compare C/E/H/E/W/- ordinary forward with three overlapping invocations.

    Covers indexer, attention, dense, routed/shared expert, embedding, output and
    mHC gradients with asymmetric padding across a two-sequence batch.
    Two forwards are live on the same model; the third reuses the schedule after
    the first backward. This catches residual and dispatcher state aliasing.
    """
    if Utils.world_size < 2:
        pytest.skip("requires torchrun with at least two GPUs for real EP")
    Utils.initialize_model_parallel(expert_model_parallel_size=2)
    set_streams()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = _make_config(
        num_layers=6,
        csa_compress_ratios=[4, 0, 128, 0, 0, 0],
        dsa_indexer_loss_coeff=0.1,
        dsa_indexer_rotate_activation=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=128,
        moe_shared_expert_intermediate_size=128,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.1,
        moe_z_loss_coeff=0.01,
        overlap_moe_expert_parallel_comm=True,
        enable_hyper_connections=mhc,
        normalization="RMSNorm",
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_dsv4_stack_spec(config),
        vocab_size=256,
        max_sequence_length=256,
        hybrid_layer_pattern="CEHEW-",
    ).cuda()
    data = [
        dict(
            input_ids=torch.randint(0, 256, (2, 256), device="cuda"),
            position_ids=torch.arange(256, device="cuda").expand(2, -1),
            attention_mask=None,
            labels=torch.randint(0, 256, (2, 256), device="cuda"),
            padding_mask=torch.stack(
                [
                    torch.arange(256, device="cuda") >= 176 - 16 * microbatch,
                    torch.arange(256, device="cuda") < 40 + 8 * microbatch,
                ]
            ),
        )
        for microbatch in range(3)
    ]
    router_calls = {}

    def check_router_padding(module, inputs):
        microbatch = router_calls.get(module, 0) % len(data)
        torch.testing.assert_close(inputs[1], data[microbatch]["padding_mask"].transpose(0, 1))
        router_calls[module] = router_calls.get(module, 0) + 1

    router_hooks = [
        module.router.register_forward_pre_hook(check_router_padding)
        for module in model.modules()
        if isinstance(module, MoELayer)
    ]
    try:
        reference_outputs = []
        for batch in data:
            output = model(**batch).float()
            reference_outputs.append(output.detach().clone())
            output.sum().backward()
        reference_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
        }
        model.zero_grad(set_to_none=True)

        previous_plan = None
        previous_output = None
        for microbatch, batch in enumerate(data):
            set_current_microbatch(model, microbatch)
            plan = model.build_schedule_plan(**batch)
            output = TransformerModelChunkSchedulePlan.run(
                plan,
                previous_plan,
                b_grad=None if previous_output is None else torch.ones_like(previous_output),
            )
            torch.testing.assert_close(output, reference_outputs[microbatch], rtol=0, atol=0)
            previous_plan, previous_output = plan, output
        TransformerModelChunkSchedulePlan.run(
            None, previous_plan, b_grad=torch.ones_like(previous_output)
        )
        torch.cuda.synchronize()
        for name, parameter in model.named_parameters():
            reference = reference_grads[name]
            assert (parameter.grad is None) == (reference is None), name
            if reference is not None:
                # The combined backward changes BF16 accumulation order.
                torch.testing.assert_close(
                    parameter.grad, reference, rtol=0.02, atol=0.02, msg=name
                )
        assert any("indexer" in name and grad is not None for name, grad in reference_grads.items())
        assert any(
            "shared_experts" in name and grad is not None for name, grad in reference_grads.items()
        )
        assert len(router_calls) == 2
        assert all(count == 2 * len(data) for count in router_calls.values())
        if mhc:
            assert any(
                "hyper_connection" in name and grad is not None
                for name, grad in reference_grads.items()
            )
    finally:
        for hook in router_hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        del model
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()


def _assert_three_invocation_parity(model, data):
    """Replay identical default/tracker RNG states through actual overlap nodes."""
    assert len(data) == 3
    rng_states = _get_all_rng_states()
    model.set_is_first_microbatch()
    reference_outputs = []
    for batch in data:
        output = model(**batch).float()
        reference_outputs.append(output.detach().clone())
        output.sum().backward()
    reference_grads = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    model.zero_grad(set_to_none=True)
    _set_all_rng_states(*rng_states)
    model.set_is_first_microbatch()

    previous_plan, previous_output = None, None
    for microbatch, batch in enumerate(data):
        set_current_microbatch(model, microbatch)
        plan = model.build_schedule_plan(**batch)
        output = TransformerModelChunkSchedulePlan.run(
            plan,
            previous_plan,
            b_grad=None if previous_output is None else torch.ones_like(previous_output),
        )
        torch.testing.assert_close(output, reference_outputs[microbatch], rtol=0, atol=0)
        previous_plan, previous_output = plan, output
    TransformerModelChunkSchedulePlan.run(
        None, previous_plan, b_grad=torch.ones_like(previous_output)
    )
    torch.cuda.synchronize()
    for name, parameter in model.named_parameters():
        reference = reference_grads[name]
        assert (parameter.grad is None) == (reference is None), name
        if reference is not None:
            # Match the existing Hybrid parity test: split backward changes the
            # accumulation order of BF16 gradients across microbatches.
            torch.testing.assert_close(parameter.grad, reference, rtol=0.02, atol=0.02, msg=name)
    assert any("experts" in name and grad is not None for name, grad in reference_grads.items())
    if model.config.enable_hyper_connections:
        assert any(
            "hyper_connection" in name and grad is not None
            for name, grad in reference_grads.items()
        )


def _make_three_batches():
    return [
        dict(
            input_ids=torch.randint(0, 256, (2, 128), device="cuda"),
            position_ids=torch.arange(128, device="cuda").expand(2, -1),
            attention_mask=None,
            labels=torch.randint(0, 256, (2, 128), device="cuda"),
        )
        for _ in range(3)
    ]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mhc", [False, True])
def test_hybrid_sequence_parallel_rng_parity(mhc):
    """TP2/SP + EP2 dropout and jitter must use ordinary Hybrid's RNG streams.

    Dense/MoE/dense layers exercise all three Hybrid-owned compute callables.
    Use a plain TransformerConfig: DSv4 attention itself requires TP1.
    Three invocations overlap forward
    and backward on one model; both default and tracked RNG states are restored.
    """
    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("requires a world size divisible by four for TP2/EP2")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=2, expert_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    set_streams()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        hidden_dropout=0.1,
        attention_dropout=0.0,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=128,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_input_jitter_eps=0.1,
        gradient_accumulation_fusion=False,
        overlap_moe_expert_parallel_comm=True,
        enable_hyper_connections=mhc,
        normalization="RMSNorm",
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=256,
        max_sequence_length=128,
        hybrid_layer_pattern="-E-",
    ).cuda()
    try:
        _assert_three_invocation_parity(model, _make_three_batches())
    finally:
        model.zero_grad(set_to_none=True)
        del model
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mhc", [False, True])
def test_hybrid_fp8_bf16_boundary_input_lifetime(mhc):
    """BF16 boundary experts must retain an input bypassing both padding copies.

    One expert per EP rank skips dispatch-postprocess permutation; router padding
    skips expert-side FP8 padding. BF16 FC1 therefore saves the dispatch buffer
    itself. The middle dense layer actually runs in MXFP8.
    """
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires an even world size for EP2")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 requires Blackwell or newer")
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    Utils.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    set_streams()
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=128,
        num_moe_experts=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_router_dtype="fp32",
        moe_router_padding_for_quantization=True,
        fp8="e4m3",
        fp8_recipe=Fp8Recipe.mxfp8,
        first_last_layers_bf16=True,
        num_layers_at_start_in_bf16=1,
        num_layers_at_end_in_bf16=1,
        gradient_accumulation_fusion=False,
        overlap_moe_expert_parallel_comm=True,
        enable_hyper_connections=mhc,
        normalization="RMSNorm",
    )
    model = HybridModel(
        config=config,
        hybrid_stack_spec=hybrid_stack_spec,
        vocab_size=256,
        max_sequence_length=128,
        hybrid_layer_pattern="E-E",
    ).cuda()
    quantization_calls = {}
    quantization_expected = {}

    def check_quantization_context(module, inputs):
        assert FP8GlobalStateManager.is_fp8_enabled() == quantization_expected[module]
        quantization_calls[module] = quantization_calls.get(module, 0) + 1

    hooks = []
    for index, layer in enumerate(model.decoder.layers):
        inner = layer.inner_layer if mhc else layer
        mlp = inner.mlp.experts if isinstance(inner.mlp, MoELayer) else inner.mlp
        quantization_expected[mlp.linear_fc1] = index == 1
        hooks.append(mlp.linear_fc1.register_forward_pre_hook(check_quantization_context))
    try:
        _assert_three_invocation_parity(model, _make_three_batches())
        assert len(quantization_calls) == 3
        assert all(count == 6 for count in quantization_calls.values())
    finally:
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
        del model
        destroy_moe_metrics_tracker()
        Utils.destroy_model_parallel()
        gc.collect()
        torch.cuda.empty_cache()
