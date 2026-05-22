# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import pytest
import torch

from megatron.core.tensor_parallel.layers import linear_with_frozen_weight
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


def test_LinearWithFrozenWeight_3d_non_contiguous_grad_output():
    """Backward must handle a 3D non-contiguous grad_output without
    crashing in the batched-GEMM dispatch."""
    Utils.initialize_model_parallel(1, 1)

    seq_length, batch_size, in_features, out_features = 4, 2, 8, 6

    # [B, S, K] contig leaf; transpose downstream so backward produces a
    # non-contiguous [S, B, K] grad_output into the linear.
    input_bsk = torch.arange(
        batch_size * seq_length * in_features, dtype=torch.float32
    ).reshape(batch_size, seq_length, in_features).cuda()
    input_bsk.requires_grad = True
    input_sbk = input_bsk.transpose(0, 1)
    assert not input_sbk.is_contiguous()

    weight = torch.ones((out_features, in_features)).cuda()
    bias = torch.zeros(out_features).cuda()

    output_parallel = linear_with_frozen_weight(
        input_sbk,
        weight,
        bias,
        False,  # gradient_accumulation_fusion
        False,  # allreduce_dgrad
        False,  # sequence_parallel
        None,  # grad_output_buffer
        None,  # wgrad_deferral_limit
    )
    output = gather_from_tensor_model_parallel_region(output_parallel)
    output.transpose(0, 1).sum().backward()

    # weight=ones means each input element contributes once per output
    # column, so its grad equals the number of output columns.
    expected_grad = torch.full_like(input_bsk, float(out_features))
    assert input_bsk.grad is not None
    assert torch.allclose(input_bsk.grad, expected_grad)

    Utils.destroy_model_parallel()
