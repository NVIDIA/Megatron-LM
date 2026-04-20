# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.attention_residuals import AttnResState, FullAttnRes


def test_full_attn_res_single_value_returns_input():
    module = FullAttnRes(hidden_size=4)
    value = torch.randn(3, 2, 4)

    output = module([value])

    assert output.shape == value.shape
    assert output.dtype == value.dtype
    torch.testing.assert_close(output, value)


def test_full_attn_res_zero_query_is_uniform_average():
    module = FullAttnRes(hidden_size=4)
    values = [torch.randn(3, 2, 4) for _ in range(3)]

    output = module(values)
    expected = torch.stack(values, dim=0).mean(dim=0)

    assert output.shape == values[0].shape
    torch.testing.assert_close(output, expected)


def test_full_attn_res_backward_reaches_values_query_and_norm_weight():
    module = FullAttnRes(hidden_size=4)
    with torch.no_grad():
        module.query.copy_(torch.tensor([0.25, -0.5, 0.75, 1.0]))

    values = [torch.randn(3, 2, 4, requires_grad=True) for _ in range(3)]
    output = module(values)
    loss = output.pow(2).mean()

    loss.backward()

    assert module.query.grad is not None
    assert torch.count_nonzero(module.query.grad).item() > 0
    assert module.weight.grad is not None
    assert torch.count_nonzero(module.weight.grad).item() > 0
    for value in values:
        assert value.grad is not None
        assert torch.count_nonzero(value.grad).item() > 0


@pytest.mark.parametrize("implementation", ["checkpointed", "triton", "triton_bwd"])
def test_full_attn_res_custom_implementation_matches_torch_forward_and_backward(implementation):
    torch.manual_seed(123)
    eager = FullAttnRes(hidden_size=4, implementation='torch')
    custom = FullAttnRes(hidden_size=4, implementation=implementation)
    with torch.no_grad():
        eager.query.copy_(torch.tensor([0.25, -0.5, 0.75, 1.0]))
        custom.query.copy_(eager.query)
        custom.weight.copy_(eager.weight)

    eager_values = [torch.randn(3, 2, 4, requires_grad=True) for _ in range(3)]
    custom_values = [value.detach().clone().requires_grad_(True) for value in eager_values]

    eager_output = eager(eager_values)
    custom_output = custom(custom_values)
    torch.testing.assert_close(custom_output, eager_output)

    eager_loss = eager_output.pow(2).mean()
    custom_loss = custom_output.pow(2).mean()
    eager_loss.backward()
    custom_loss.backward()

    torch.testing.assert_close(custom.query.grad, eager.query.grad)
    torch.testing.assert_close(custom.weight.grad, eager.weight.grad)
    for custom_value, eager_value in zip(custom_values, eager_values):
        torch.testing.assert_close(custom_value.grad, eager_value.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_full_attn_res_preserves_value_dtype(dtype):
    module = FullAttnRes(hidden_size=4)
    values = [torch.randn(3, 2, 4).to(dtype=dtype) for _ in range(2)]

    output = module(values)

    assert output.dtype == dtype


def test_full_attn_res_rejects_empty_history():
    module = FullAttnRes(hidden_size=4)

    with pytest.raises(ValueError, match="at least one value"):
        module([])


def test_block_attn_res_state_groups_sublayers_into_partial_blocks():
    initial = torch.full((2, 1, 4), 10.0)
    state = AttnResState.block(initial, num_sublayers=4, num_blocks=2)

    assert state.block_size == 2
    values = state.values()
    assert len(values) == 1
    torch.testing.assert_close(values[0], initial)

    first = torch.ones_like(initial)
    second = torch.full_like(initial, 2.0)
    third = torch.full_like(initial, 3.0)

    state.append(first)
    values = state.values()
    assert len(values) == 2
    torch.testing.assert_close(values[0], initial)
    torch.testing.assert_close(values[1], first)

    state.append(second)
    values = state.values()
    assert len(values) == 2
    torch.testing.assert_close(values[0], initial)
    torch.testing.assert_close(values[1], first + second)

    state.append(third)
    values = state.values()
    assert len(values) == 3
    torch.testing.assert_close(values[0], initial)
    torch.testing.assert_close(values[1], first + second)
    torch.testing.assert_close(values[2], third)
