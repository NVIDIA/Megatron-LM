# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Router hooks used by shared-prefix execution: routing kwargs, logical expert counts,
fixed-row router GEMM blocks and the router GEMM backward with frozen inputs."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.transformer.moe import moe_utils
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import router_gating_linear, router_gating_token_blocks
from megatron.core.transformer.moe.router import (
    InferenceTopKRouter,
    TopKRouter,
    _expert_bias_token_counts,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _router_with_routing(routing):
    return SimpleNamespace(
        _maintain_float32_expert_bias=Mock(),
        apply_input_jitter=lambda value: value,
        gating=lambda value: value,
        config=SimpleNamespace(moe_router_force_load_balancing=False, moe_router_force_biased=None),
        routing=routing,
    )


def test_router_forward_keeps_upstream_routing_signature_without_multiplicities():
    hidden_states = torch.randn(3, 1, 4)
    expected = (torch.ones(3, 2), torch.ones(3, 4, dtype=torch.bool))
    calls = []

    def routing(logits, padding_mask=None, input_ids=None, packed_seq_params=None):
        calls.append((logits, padding_mask, input_ids, packed_seq_params))
        return expected

    router = _router_with_routing(routing)
    with patch("megatron.core.transformer.moe.router.is_observing_tensor", return_value=False):
        actual = TopKRouter.forward(router, hidden_states)
    assert actual == expected
    assert len(calls) == 1 and calls[0][0] is hidden_states


def test_router_forward_passes_multiplicities_only_when_set():
    hidden_states = torch.randn(3, 1, 4)
    multiplicities = torch.tensor([2, 1, 1])
    router = _router_with_routing(Mock(return_value=(None, None)))
    with patch("megatron.core.transformer.moe.router.is_observing_tensor", return_value=False):
        TopKRouter.forward(router, hidden_states)
        TopKRouter.forward(router, hidden_states, token_multiplicities=multiplicities)
    first, second = router.routing.call_args_list
    assert "token_multiplicities" not in first.kwargs
    assert second.kwargs["token_multiplicities"] is multiplicities


def test_inference_router_fallback_forwards_multiplicities():
    router = InferenceTopKRouter.__new__(InferenceTopKRouter)
    hidden_states = torch.randn(3, 1, 4)
    multiplicities = torch.tensor([2, 1, 1])
    parent_forward = Mock(return_value=("probs", "routes"))
    with (
        patch.object(TopKRouter, "forward", parent_forward),
        patch("megatron.core.transformer.moe.router.InferenceMode.is_active", return_value=False),
    ):
        actual = InferenceTopKRouter.forward(
            router, hidden_states, token_multiplicities=multiplicities
        )
    assert actual == ("probs", "routes")
    assert parent_forward.call_args.kwargs["token_multiplicities"] is multiplicities


@pytest.mark.parametrize("dense_indices", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_expert_bias_counts_are_exact_beyond_float32_integer_range(dense_indices, deterministic):
    # Shared-prefix layouts carry float32 multiplicities; counts must still be exact int64.
    multiplicities = torch.tensor([float(1 << 24), 1.0], dtype=torch.float32)
    routes = torch.tensor([[0, 1], [0, -1]])
    if not dense_indices:
        routes = torch.tensor([[True, True], [True, False]])
    router = SimpleNamespace(
        enable_expert_bias=True,
        local_tokens_per_expert=torch.zeros(2, dtype=torch.int64),
        config=SimpleNamespace(num_moe_experts=2),
    )
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        with torch.enable_grad():
            TopKRouter._apply_expert_bias(router, routes, token_multiplicities=multiplicities)
    finally:
        torch.use_deterministic_algorithms(prev)
    expected = torch.tensor([(1 << 24) + 1, 1 << 24], dtype=torch.int64)
    assert torch.equal(router.local_tokens_per_expert, expected)


@pytest.mark.parametrize("dense_indices", [False, True])
@pytest.mark.parametrize("padding", [False, True])
def test_expert_bias_counts_accumulate_in_place(dense_indices, padding):
    routes = torch.tensor([[0, 2], [1, 3], [2, 3], [0, -1]])
    bool_routes = torch.zeros(4, 4, dtype=torch.bool)
    for row, experts in enumerate(routes.tolist()):
        for expert in experts:
            if expert >= 0:
                bool_routes[row, expert] = True
    padding_mask = torch.tensor([False, True, False, False]) if padding else None
    multiplicities = torch.tensor([3, 1, 2, 5])
    route_map = routes if dense_indices else bool_routes
    keep = torch.ones(4, dtype=torch.bool) if padding_mask is None else ~padding_mask

    counts = torch.full((4,), 7, dtype=torch.int64)
    returned = _expert_bias_token_counts(route_map, padding_mask, out=counts)
    assert returned is counts
    assert torch.equal(counts, 7 + (bool_routes & keep.unsqueeze(-1)).sum(dim=0))

    counts = torch.full((4,), 7, dtype=torch.int64)
    _expert_bias_token_counts(route_map, padding_mask, multiplicities, out=counts)
    weights = (multiplicities * keep).unsqueeze(-1)
    assert torch.equal(counts, 7 + (bool_routes * weights).sum(dim=0))


def test_invalid_expert_count_metadata_rejected():
    routes = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="one value per routed token"):
        _expert_bias_token_counts(routes, token_multiplicities=torch.ones(3), num_experts=3)
    with pytest.raises(ValueError, match="num_experts"):
        _expert_bias_token_counts(routes, token_multiplicities=torch.ones(2))
    with pytest.raises(ValueError, match="padding mask"):
        _expert_bias_token_counts(routes, padding_mask=torch.zeros(3, dtype=torch.bool))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("with_padding", [False, True])
def test_topk_router_expert_bias_counts_logical_tokens(with_padding):
    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=8,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="none",
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
            moe_router_topk=2,
            add_bias_linear=False,
        )
        submodules = get_submodules(
            get_gpt_layer_local_submodules(num_experts=8, moe_grouped_gemm=False).mlp
        )
        router = MoELayer(config, submodules).router.cuda()
        hidden_states = torch.randn(6, 2, 12, device="cuda")
        padding_mask = None
        if with_padding:
            padding_mask = torch.zeros(6, 2, dtype=torch.bool, device="cuda")
            padding_mask[4:, 1] = True
        # Float32, as produced by SharedPrefixLayout.padded_token_multiplicities.
        multiplicities = torch.tensor(
            [4, 1, 1, 3, 1, 1, 2, 1, 1, 1, 1, 1], dtype=torch.float32, device="cuda"
        )
        router.local_tokens_per_expert.zero_()
        with torch.enable_grad():
            _, routing_map = router(
                hidden_states, padding_mask, token_multiplicities=multiplicities
            )
        keep = torch.ones(12, dtype=torch.bool, device="cuda")
        if padding_mask is not None:
            keep = ~padding_mask.reshape(-1)
        weights = (multiplicities.long() * keep).unsqueeze(-1)
        expected = (routing_map.bool() * weights).sum(dim=0)
        assert router.local_tokens_per_expert.dtype == torch.int64
        assert torch.equal(router.local_tokens_per_expert, expected)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "requires",
    [(True, False, False), (False, True, False), (False, False, True), (True, True, True)],
)
@pytest.mark.parametrize("block_size", [None, 3])
def test_router_frozen_parameters_preserve_requested_gradients(requires, block_size):
    """The router GEMM backward computes only the gradients its inputs require."""
    generator = torch.Generator().manual_seed(901)
    values = [
        torch.randn(shape, generator=generator, dtype=torch.float64)
        for shape in [(7, 1, 5), (4, 5), (4,)]
    ]
    actual_inputs = [value.clone().requires_grad_(need) for value, need in zip(values, requires)]
    reference_inputs = [value.clone().requires_grad_(need) for value, need in zip(values, requires)]
    if block_size is None:
        actual = router_gating_linear(*actual_inputs, torch.float64)
    else:
        with router_gating_token_blocks(block_size):
            actual = router_gating_linear(*actual_inputs, torch.float64)
    reference = torch.nn.functional.linear(*reference_inputs)
    cotangent = torch.randn(actual.shape, generator=generator, dtype=torch.float64)
    actual.backward(cotangent)
    reference.backward(cotangent)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    for value, reference_value, need in zip(actual_inputs, reference_inputs, requires):
        if need:
            torch.testing.assert_close(value.grad, reference_value.grad, rtol=1e-12, atol=1e-12)
        else:
            assert value.grad is None


def _blocked_reference(inp, weight, bias, block_size):
    # Previous implementation: pad and run every block as a separate copy.
    outputs = []
    for start in range(0, inp.shape[0], block_size):
        rows = inp[start : start + block_size]
        padded = torch.nn.functional.pad(rows, (0, 0, 0, block_size - rows.shape[0]))
        outputs.append(moe_utils._router_gating_gemm(padded, weight, bias, torch.float32))
    return torch.cat(outputs)[: inp.shape[0]]


@pytest.mark.parametrize("num_rows", [1, 7, 8, 9, 24, 29])
def test_router_gemm_blocks_use_views_of_one_padded_input(num_rows):
    block_size = 8
    inp = torch.randn(num_rows, 1, 5, dtype=torch.float64)
    weight = torch.randn(3, 5, dtype=torch.float64)
    seen = []
    gemm = moe_utils._router_gating_gemm

    def recording_gemm(rows, *args):
        seen.append((rows.shape, rows.untyped_storage().data_ptr()))
        return gemm(rows, *args)

    with patch.object(moe_utils, "_router_gating_gemm", recording_gemm):
        with router_gating_token_blocks(block_size):
            actual = router_gating_linear(inp, weight, None, torch.float64)
    torch.testing.assert_close(actual, inp @ weight.t(), rtol=0, atol=1e-12)
    assert actual.shape == (num_rows, 1, 3)
    assert len(seen) == -(-num_rows // block_size)
    assert all(shape == (block_size, 5) for shape, _ in seen)
    assert len({storage for _, storage in seen}) == 1


def test_router_gemm_default_path_skips_blocks():
    inp = torch.randn(9, 1, 5, dtype=torch.float64)
    weight = torch.randn(3, 5, dtype=torch.float64)
    with patch.object(moe_utils, "_router_gating_gemm_blocks", side_effect=AssertionError):
        actual = router_gating_linear(inp, weight, None, torch.float64)
    torch.testing.assert_close(actual, inp @ weight.t(), rtol=0, atol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the TE/cuBLAS router GEMM")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("num_rows", [1, 333, 1024, 2048, 2500])
def test_router_gemm_blocks_bitwise_match_per_block_copies(input_dtype, with_bias, num_rows):
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    inp = torch.randn(num_rows, 1, 256, device="cuda", generator=generator).to(input_dtype)
    weight = torch.randn(64, 256, device="cuda", generator=generator).to(input_dtype)
    bias = torch.randn(64, device="cuda", generator=generator).to(input_dtype)
    bias = bias if with_bias else None
    with router_gating_token_blocks(1024):
        actual = router_gating_linear(inp, weight, bias, torch.float32)
    expected = _blocked_reference(inp.view(num_rows, -1), weight, bias, 1024)
    assert torch.equal(actual.view(num_rows, -1), expected)
