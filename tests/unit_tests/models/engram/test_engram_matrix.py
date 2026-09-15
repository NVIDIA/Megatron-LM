# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.engram.hybrid_adapter import EngramAttentionLayer
from tests.unit_tests.models.engram.test_integration import (
    make_hybrid,
    make_transformer_config,
    model_inputs,
)
from tests.unit_tests.test_utilities import Utils

CASES = [
    ("local", "*-*-*-", {}),
    ("row_a2a", "*-*-*-", {"engram_table_backend": "row_a2a"}),
    ("moe", "*E*E*E", {"num_moe_experts": 4, "moe_ffn_hidden_size": 32, "moe_router_topk": 2}),
    ("mixed", "*-*E*E", {"num_moe_experts": 4, "moe_ffn_hidden_size": 32, "moe_router_topk": 2}),
    (
        "recompute",
        "*-*-*-",
        {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 2},
    ),
]


@pytest.mark.parametrize("name,pattern,overrides", CASES)
def test_engram_hybrid_configuration_matrix(name, pattern, overrides):
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        model = make_hybrid(make_transformer_config(**overrides), pattern)
        selected = [
            i
            for i, layer in enumerate(model.decoder.layers)
            if isinstance(layer, EngramAttentionLayer)
        ]
        assert selected == [0, 4]
        output = model(*model_inputs())
        assert torch.isfinite(output).all(), name
        output.float().square().mean().backward()
        for index in selected:
            assert any(p.grad is not None for p in model.decoder.layers[index].engram.parameters())
    finally:
        Utils.destroy_model_parallel()


def test_hybrid_full_recompute_preserves_output_and_gradients():
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        direct = make_hybrid(make_transformer_config())
        recomputed = make_hybrid(
            make_transformer_config(
                recompute_granularity="full", recompute_method="uniform", recompute_num_layers=2
            )
        )
        recomputed.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
        expected = direct(*model_inputs())
        actual = recomputed(*model_inputs())
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_grads = {name: p.grad for name, p in direct.named_parameters()}
        for name, p in recomputed.named_parameters():
            grad = expected_grads[name]
            assert (p.grad is None) == (grad is None), name
            if grad is not None:
                torch.testing.assert_close(p.grad, grad, rtol=1e-5, atol=1e-7, msg=name)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("method,num_layers", [("uniform", 2), ("block", 5)])
def test_row_a2a_hybrid_recompute_preserves_interleaved_microbatches(method, num_layers):
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        direct = make_hybrid(make_transformer_config(engram_table_backend="row_a2a"))
        recomputed = make_hybrid(
            make_transformer_config(
                engram_table_backend="row_a2a",
                recompute_granularity="full",
                recompute_method=method,
                recompute_num_layers=num_layers,
            )
        )
        recomputed.load_state_dict(copy.deepcopy(direct.state_dict()), strict=True)
        tokens, positions, mask = model_inputs()
        microbatches = [(tokens, positions, mask), (tokens.flip(-1) + 8, positions, mask)]

        # Both forwards precede either backward. Replaying the first microbatch
        # must use its own tokens even after the provider has prepared the second.
        expected = [direct(*inputs) for inputs in microbatches]
        actual = [recomputed(*inputs) for inputs in microbatches]
        expected_parameters = dict(direct.named_parameters())
        actual_parameters = dict(recomputed.named_parameters())
        assert expected_parameters.keys() == actual_parameters.keys()

        for microbatch, (actual_output, expected_output) in enumerate(zip(actual, expected)):
            torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
            expected_output.float().square().mean().backward()
            actual_output.float().square().mean().backward()

            # Check both the first gradient and the accumulated gradient without
            # clearing either model between microbatches. Keep COO indices exact
            # so looking up the wrong token rows cannot hide under a tolerance.
            sparse_gradients = 0
            dense_gradients = 0
            for name, parameter in actual_parameters.items():
                actual_grad = parameter.grad
                expected_grad = expected_parameters[name].grad
                message = f"{method}, microbatch {microbatch}, {name}"
                assert (actual_grad is None) == (expected_grad is None), message
                if expected_grad is None:
                    continue
                assert actual_grad.is_sparse == expected_grad.is_sparse, message
                if expected_grad.is_sparse:
                    sparse_gradients += 1
                    actual_grad = actual_grad.coalesce()
                    expected_grad = expected_grad.coalesce()
                    torch.testing.assert_close(
                        actual_grad.indices(), expected_grad.indices(), rtol=0, atol=0, msg=message
                    )
                    actual_grad = actual_grad.values()
                    expected_grad = expected_grad.values()
                else:
                    dense_gradients += 1
                torch.testing.assert_close(
                    actual_grad, expected_grad, rtol=1e-5, atol=1e-7, msg=message
                )
            assert sparse_gradients == len(direct.config.engram_layer_ids)
            assert dense_gradients > 0
    finally:
        Utils.destroy_model_parallel()
