# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Precision regressions learned from the Nemotron vLLM parity audit."""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.mark.parametrize("preallocate", [False, True])
def test_fc2_preserves_sub_bf16_cancellation(preallocate):
    """Neither round-before-weighting nor round-after-weighting preserves this sum."""
    from megatron.core.inference.moe.fused_moe import ActivationType
    from megatron.core.inference.moe.vllm_fused_moe import VllmFusedMoeBuffers, vllm_fused_moe

    tokens, width, experts, topk = 4, 128, 2, 2
    x = torch.ones(tokens, width, device="cuda", dtype=torch.bfloat16)
    w1 = torch.zeros(experts, width, width, device="cuda", dtype=torch.bfloat16)
    w1[:, 0, 0] = 1
    w1[:, 1, 0] = 1 / 16
    w2 = torch.zeros_like(w1)
    w2[0, :, 0] = 1
    w2[1, :, 0] = -1
    w2[:, :, 1] = 1 / 2
    probs = torch.full((tokens, topk), 0.5, device="cuda")
    ids = torch.tensor([[0, 1]] * tokens, device="cuda")
    valid = torch.tensor([tokens - 1], device="cuda", dtype=torch.int32)

    # ReLU² outputs [1, 1/256, 0, ...]. FC2 gives +/-1 + 1/512.
    # Their equally weighted sum is exactly 1/512, including in FP32.
    expected = torch.full_like(x, 1 / 512, dtype=torch.float32)
    expected[-1].zero_()
    VllmFusedMoeBuffers._delete_buffers()
    try:
        if preallocate:
            VllmFusedMoeBuffers.allocate_buffers(tokens, topk, width, width, experts)

        def forward(x, probs):
            return vllm_fused_moe(
                x, probs, w1, w2, ActivationType.SQUARED_RELU, experts, 0, valid, ids
            )

        result = forward(x, probs)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert_replays_bit_exact(
            forward, (x, probs), backward=False, replays=3, what="FP32 MoE FC2 cancellation"
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward(x, probs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = forward(x, probs)
        for probability in (0.25, 0.5):
            probs.fill_(probability)
            graph.replay()
            torch.testing.assert_close(captured, expected * (2 * probability), rtol=0, atol=0)
    finally:
        VllmFusedMoeBuffers._delete_buffers()


@pytest.mark.parametrize("zero_centered_gamma", [False, True])
def test_fp32_residual_norm_preserves_input_and_weight_multiply(zero_centered_gamma):
    from megatron.core.fusions.fused_inference_rms_norm import fp32_residual_rms_norm

    generator = torch.Generator(device="cuda").manual_seed(2026)
    x = torch.randn(128, 512, device="cuda", generator=generator)
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16, generator=generator)
    eps = 1e-5
    scale = weight.double() + int(zero_centered_gamma)

    # Independent FP64 oracle; only the public output is quantized.
    def reference(value):
        value = value.double()
        return value / (value.square().sum(-1, keepdim=True) / value.shape[-1] + eps).sqrt() * scale

    oracle = reference(x)
    result = fp32_residual_rms_norm(x, weight, eps, zero_centered_gamma)
    rounded_input = reference(x.bfloat16()).bfloat16()
    ideal = oracle.bfloat16()
    assert result.dtype == weight.dtype
    # FP32 roundoff may affect a final BF16 tie; it must not reproduce the
    # systematic error from rounding the incoming residual before the norm.
    assert (result != ideal).sum() < (rounded_input != ideal).sum() // 100
    assert (result.double() - oracle).square().mean() < (
        rounded_input.double() - oracle
    ).square().mean()
    assert_replays_bit_exact(
        lambda value: fp32_residual_rms_norm(value, weight, eps, zero_centered_gamma),
        (x,),
        backward=False,
        what="FP32 residual RMSNorm",
    )


@pytest.mark.parametrize("training,fp32", [(False, True), (False, False), (True, True)])
def test_mamba_preserves_opted_in_inference_norm_input(training, fp32):
    from megatron.core.ssm.mamba_layer import MambaLayer

    x = torch.full((1, 1, 128), 1 + 1 / 512, device="cuda")
    layer = SimpleNamespace(
        training=training,
        config=SimpleNamespace(
            params_dtype=torch.bfloat16,
            fp32_residual_connection=fp32,
            transformer_impl="inference_optimized",
        ),
        norm=torch.nn.Identity(),
    )
    result = MambaLayer._prepare_mixer_input(layer, x)
    if fp32 and not training:
        assert result is x
    else:
        assert result.dtype == torch.bfloat16
        torch.testing.assert_close(result, x.bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "dispatcher_name,overlap",
    [
        ("NCCLAllGatherDispatcher", False),
        ("NVLSAllGatherVDispatcher", False),
        ("NVLSAllGatherVDispatcher", True),
    ],
)
@pytest.mark.parametrize("fp32_residual", [False, True])
def test_shared_expert_add_precedes_output_cast(
    dispatcher_name, overlap, fp32_residual, monkeypatch
):
    from megatron.core.transformer.moe import token_dispatcher_inference as dispatchers
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.shared_experts import SharedExpertMLP

    # Exercise the real EP=1 combine and layer postprocess without allocating
    # unrelated router/experts. The reduction for EP>1 uses the same finish hook.
    dispatcher = object.__new__(getattr(dispatchers, dispatcher_name))
    dispatcher.ep_size = 1
    dispatcher.hidden_shape = (2, 1, 128)
    dispatcher._shared_expert_output = None
    layer = SimpleNamespace(
        training=False,
        _latent_shared_expert_output=None,
        token_dispatcher=dispatcher,
        config=SimpleNamespace(
            moe_latent_size=None,
            params_dtype=torch.bfloat16,
            fp32_residual_connection=fp32_residual,
        ),
    )
    routed = torch.full((2, 128), 1 + 1 / 512, device="cuda")
    shared = torch.full(dispatcher.hidden_shape, -1, device="cuda", dtype=torch.bfloat16)
    if overlap:
        monkeypatch.setattr(SharedExpertMLP, "stream", torch.cuda.current_stream())
        dispatcher._shared_expert_output = shared
        shared = None
    reduced = dispatcher.token_combine(routed)
    result = MoELayer.postprocess(layer, reduced, shared)
    assert result.dtype == (torch.float32 if fp32_residual else torch.bfloat16)
    torch.testing.assert_close(result, torch.full_like(result, 1 / 512), rtol=0, atol=0)
