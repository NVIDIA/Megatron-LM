# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical comparisons between SonicMoE and Megatron grouped-GEMM MoE."""

import pytest
import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import clear_aux_losses_tracker
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.sonic_moe_layer import SonicMoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("sonicmoe")

NUM_EXPERTS = 4
TOPK = 2
HIDDEN_SIZE = 128
MOE_FFN_HIDDEN_SIZE = 256
TOKENS = 64
RTOL = 5.0e-3
ATOL = 5.0e-3


def _make_config(ep_size: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=4 * HIDDEN_SIZE,
        num_moe_experts=NUM_EXPERTS,
        moe_ffn_hidden_size=MOE_FFN_HIDDEN_SIZE,
        moe_router_topk=TOPK,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_score_function="softmax",
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        bias_activation_fusion=False,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
    )


def _stack_grouped_weights(grouped_linear: torch.nn.Module, count: int) -> torch.Tensor:
    return torch.stack([getattr(grouped_linear, f"weight{idx}") for idx in range(count)])


def _stack_grouped_weight_grads(grouped_linear: torch.nn.Module, count: int) -> torch.Tensor:
    grads = [getattr(grouped_linear, f"weight{idx}").grad for idx in range(count)]
    assert all(grad is not None for grad in grads)
    return torch.stack(grads)


def _to_sonic_layout(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape == target.shape:
        return source
    if source.ndim == 3 and source.shape[1] == target.shape[2]:
        return source.transpose(1, 2).contiguous()
    raise AssertionError(f"Unsupported grouped/Sonic layouts: {source.shape} and {target.shape}")


def _copy_reference_parameters(reference: MoELayer, sonic: SonicMoELayer) -> None:
    with torch.no_grad():
        sonic._router_weight().copy_(reference.router.weight)
        fc1 = _stack_grouped_weights(reference.experts.linear_fc1, sonic.num_local_experts)
        fc2 = _stack_grouped_weights(reference.experts.linear_fc2, sonic.num_local_experts)
        sonic.sonic_moe.c_fc.weight.copy_(_to_sonic_layout(fc1, sonic.sonic_moe.c_fc.weight))
        sonic.sonic_moe.c_proj.weight.copy_(_to_sonic_layout(fc2, sonic.sonic_moe.c_proj.weight))


def _set_balanced_router_weights(layer: MoELayer) -> None:
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[:, :NUM_EXPERTS].copy_(
            torch.eye(NUM_EXPERTS, device="cuda", dtype=layer.router.weight.dtype)
        )


def _make_hidden_states(rank: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(7200 + rank)
    hidden_states = torch.randn(
        TOKENS, 1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16, generator=generator
    ).mul_(0.1)

    scores = torch.linspace(-2.0, 2.0, NUM_EXPERTS, device="cuda")
    token_offsets = torch.arange(TOKENS, device="cuda").unsqueeze(1)
    expert_offsets = torch.arange(NUM_EXPERTS, device="cuda").unsqueeze(0)
    rotated_experts = (expert_offsets + token_offsets + rank) % NUM_EXPERTS
    hidden_states[:, 0, :NUM_EXPERTS] = scores[rotated_experts].to(hidden_states.dtype)
    return hidden_states


def _assert_close(candidate: torch.Tensor, reference: torch.Tensor) -> None:
    assert torch.isfinite(candidate).all()
    assert torch.isfinite(reference).all()
    torch.testing.assert_close(candidate, reference, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("ep_size", [1, 2], ids=["ep1", "ep2"])
def test_sonic_matches_topk_router_grouped_gemm(ep_size):
    """Compare full forward/backward numerics for EP=1 and EP>1."""
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=1,
    )
    try:
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        model_parallel_cuda_manual_seed(1234)
        config = _make_config(ep_size)

        grouped_moe_spec = get_moe_module_spec(
            use_te=True, num_experts=NUM_EXPERTS, moe_grouped_gemm=True
        )
        reference = grouped_moe_spec(config=config).cuda()
        sonic = SonicMoELayer(config=config).cuda()
        reference.set_layer_number(1)
        sonic.set_layer_number(1)
        reference.train()
        sonic.train()

        assert isinstance(reference.router, TopKRouter)
        assert isinstance(reference.experts, TEGroupedMLP)
        assert reference.config.moe_grouped_gemm
        assert parallel_state.get_expert_model_parallel_world_size() == ep_size
        assert parallel_state.get_expert_tensor_parallel_world_size() == 1

        _set_balanced_router_weights(reference)
        _copy_reference_parameters(reference, sonic)

        hidden_states = _make_hidden_states(torch.distributed.get_rank())
        reference_input = hidden_states.detach().clone().requires_grad_(True)
        sonic_input = hidden_states.detach().clone().requires_grad_(True)
        clear_aux_losses_tracker()

        reference_output, reference_bias = reference(reference_input)
        sonic_output, sonic_bias = sonic(sonic_input)
        assert reference_bias is None
        assert sonic_bias is None
        _assert_close(sonic_output, reference_output)

        generator = torch.Generator(device="cuda")
        generator.manual_seed(7300 + torch.distributed.get_rank())
        output_grad = torch.randn(
            reference_output.shape, device="cuda", dtype=reference_output.dtype, generator=generator
        )
        reference_output.backward(output_grad)
        sonic_output.backward(output_grad)

        _assert_close(sonic_input.grad, reference_input.grad)
        _assert_close(sonic._router_weight().grad, reference.router.weight.grad)

        reference_fc1_grad = _stack_grouped_weight_grads(
            reference.experts.linear_fc1, sonic.num_local_experts
        )
        reference_fc2_grad = _stack_grouped_weight_grads(
            reference.experts.linear_fc2, sonic.num_local_experts
        )
        _assert_close(
            sonic.sonic_moe.c_fc.weight.grad,
            _to_sonic_layout(reference_fc1_grad, sonic.sonic_moe.c_fc.weight.grad),
        )
        _assert_close(
            sonic.sonic_moe.c_proj.weight.grad,
            _to_sonic_layout(reference_fc2_grad, sonic.sonic_moe.c_proj.weight.grad),
        )
    finally:
        Utils.destroy_model_parallel()
