from __future__ import annotations

import torch
import pytest

from megatron.lite.model.deepseek_v4.vllm.primitive.moe.communication import (
    _ordered_route_backward,
    _validate_and_order_route_preserving_outputs,
)


def test_ordered_route_backward_ignores_padded_slots() -> None:
    route_values = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    topk_weights = torch.tensor([[0.25, 9.0], [0.5, 11.0]])
    output_index = torch.tensor([[0, -1], [1, -1]])
    grad_output = torch.tensor([[13.0, 17.0], [19.0, 23.0]])
    grad_routes = torch.zeros_like(route_values)
    grad_weights = torch.zeros_like(topk_weights)

    _ordered_route_backward(
        route_values=route_values,
        topk_weights=topk_weights,
        output_index=output_index,
        grad_output=grad_output,
        grad_routes=grad_routes,
        grad_weights=grad_weights,
        static_mapping_valid=False,
    )

    torch.testing.assert_close(
        grad_routes,
        torch.stack((grad_output[0] * 0.25, grad_output[1] * 0.5)),
        rtol=0,
        atol=0,
    )
    assert torch.equal(grad_weights[:, 1], torch.zeros(2))
    assert grad_weights[0, 0] == torch.dot(grad_output[0], route_values[0])
    assert grad_weights[1, 0] == torch.dot(grad_output[1], route_values[1])


def test_route_metadata_requires_slime_receive_order() -> None:
    received_tokens = torch.arange(4 * 16, dtype=torch.bfloat16).reshape(4, 16)
    received_indices = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    received_weights = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    output_index = torch.tensor([[0], [2], [1], [3]], dtype=torch.int64)
    expert_outputs = torch.tensor([[10.0], [30.0], [20.0], [40.0]])
    route_fingerprints = received_tokens.clone()
    route_indices = received_indices.reshape(-1).clone()
    route_weights = received_weights.reshape(-1).clone()

    route_rows = _validate_and_order_route_preserving_outputs(
        expert_outputs,
        received_tokens,
        received_indices,
        received_weights,
        output_index,
        route_fingerprints,
        route_indices,
        route_weights,
        return_route_rows=True,
    )
    assert torch.equal(route_rows, torch.tensor([0, 2, 1, 3]))
    ordered = _validate_and_order_route_preserving_outputs(
        expert_outputs,
        received_tokens,
        received_indices,
        received_weights,
        output_index,
        route_fingerprints,
        route_indices,
        route_weights,
    )
    assert torch.equal(ordered, torch.tensor([10.0, 20.0, 30.0, 40.0]).unsqueeze(1))


@pytest.mark.parametrize("field", ["order", "expert", "weight", "fingerprint"])
def test_route_metadata_rejects_reordered_or_changed_identity(field: str) -> None:
    received_tokens = torch.arange(4 * 16, dtype=torch.bfloat16).reshape(4, 16)
    received_indices = torch.tensor([[0], [1], [0], [1]], dtype=torch.int64)
    received_weights = torch.tensor([[0.1], [0.2], [0.3], [0.4]])
    output_index = torch.tensor([[0], [2], [1], [3]], dtype=torch.int64)
    expert_outputs = torch.tensor([[10.0], [30.0], [20.0], [40.0]])
    route_fingerprints = received_tokens.clone()
    route_indices = received_indices.reshape(-1).clone()
    route_weights = received_weights.reshape(-1).clone()
    if field == "order":
        permutation = torch.tensor([2, 0, 3, 1])
        route_fingerprints = route_fingerprints.index_select(0, permutation)
        route_indices = route_indices.index_select(0, permutation)
        route_weights = route_weights.index_select(0, permutation)
    elif field == "expert":
        route_indices[0] = 1
    elif field == "weight":
        route_weights[0] += 0.01
    else:
        route_fingerprints[0, 0] += 1

    with pytest.raises(RuntimeError):
        _validate_and_order_route_preserving_outputs(
            expert_outputs,
            received_tokens,
            received_indices,
            received_weights,
            output_index,
            route_fingerprints,
            route_indices,
            route_weights,
        )
