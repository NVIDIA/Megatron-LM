# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.extensions.transformer_engine import te_general_gemm
from megatron.core.tensor_parallel.layers import (
    linear_with_frozen_weight,
    linear_with_grad_accumulation_and_async_allreduce,
)
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("tensor_parallel,allreduce_dgrad", [(1, False), (8, True)])
def test_LinearWithFrozenWeight(tensor_parallel, allreduce_dgrad):
    Utils.initialize_model_parallel(tensor_parallel, 1)

    size_per_partition = int(8 / tensor_parallel)

    # Input is an 8x8 identity matrix.
    input_data = torch.eye(8).cuda()
    input_data.requires_grad = True

    # Weight is an 8x8 matrix of all ones. If tensor parallelism > 1, the weight is partitioned evenly across GPUs.
    weight = torch.ones((size_per_partition, 8)).cuda()

    # Bias is a vector of length 8 of all zeros. If tensor parallelism > 1, the bias is partitioned evenly across GPUs
    bias = torch.zeros((size_per_partition)).cuda()

    gradient_accumulation_fusion = False
    sequence_parallel = False
    grad_output_buffer = None
    wgrad_deferral_limit = None

    output_parallel = linear_with_frozen_weight(
        input_data,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
    )
    output = gather_from_tensor_model_parallel_region(
        output_parallel
    )  # no-op if tensor_parallel == 1.
    output.sum().backward()

    expected_output = torch.ones(8).cuda()
    expected_grad = 8 * torch.ones(8).cuda()

    assert torch.allclose(output, expected_output)
    assert torch.allclose(input_data.grad, expected_grad)

    Utils.destroy_model_parallel()


def test_LinearWithFrozenWeight_3d_input_matches_torch_linear():
    Utils.initialize_model_parallel(1, 1)

    input_data = torch.randn(4, 3, 8, device="cuda", requires_grad=True)
    weight = torch.randn(6, 8, device="cuda")
    bias = torch.randn(6, device="cuda")

    expected_input = input_data.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.linear(expected_input, weight, bias)
    expected.sum().backward()

    actual = linear_with_frozen_weight(input_data, weight, bias, False, False, False, None, None)
    actual.sum().backward()

    assert torch.allclose(actual, expected)
    assert torch.allclose(input_data.grad, expected_input.grad)

    Utils.destroy_model_parallel()


@pytest.mark.skipif(
    te_general_gemm is None, reason="Transformer Engine general_gemm is not available"
)
def test_linear_with_grad_accumulation_supports_fp32_output_and_bf16_backward():
    Utils.initialize_model_parallel(1, 1)

    input_data = torch.randn(4, 3, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(32, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_input = input_data.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    output = linear_with_grad_accumulation_and_async_allreduce(
        input_data,
        weight,
        None,
        False,
        False,
        False,
        tp_group=None,
        output_dtype=torch.float32,
    )
    output.sum().backward()

    reference_output = torch.nn.functional.linear(reference_input, reference_weight)
    reference_output.sum().backward()
    fp32_reference_output = torch.nn.functional.linear(input_data.detach().float(), weight.detach().float())

    assert output.dtype == torch.float32
    assert input_data.grad.dtype == torch.bfloat16
    assert weight.grad.dtype == torch.bfloat16
    assert torch.allclose(output, fp32_reference_output, atol=1e-4, rtol=1e-4)
    assert torch.allclose(input_data.grad, reference_input.grad)
    assert torch.allclose(weight.grad, reference_weight.grad)

    Utils.destroy_model_parallel()
