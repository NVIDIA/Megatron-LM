import contextlib
import math
import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig, ParamAndGradBuffer
from tests.unit_tests.test_utilities import Utils, TestModel


def get_model_and_buffers(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    bias: bool,
    bucket_size: int,
    use_distributed_optimizer: bool,
    overlap_grad_reduce: bool,
):
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
    )
    model = TestModel(input_dim=input_dim, output_dim=output_dim, num_layers=num_layers, bias=bias)
    params = list(model.parameters())
    param_to_name = {}
    for name, param in model.named_parameters():
        param_to_name[param] = name

    param_and_grad_buffer = ParamAndGradBuffer(
        ddp_config,
        param_dtype=torch.bfloat16,
        grad_dtype=torch.float32,
        params=params,
        data_parallel_group=parallel_state.get_data_parallel_group(),
        bucket_size=bucket_size,
        param_to_name=param_to_name,
        gradient_scaling_factor=1.0,
    )

    return model, param_and_grad_buffer


@pytest.mark.parametrize("bucket_size", [None, 9999, 10000, 10001, 19999, 20000])
@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_bucket_sizes(bucket_size: int, use_distributed_optimizer: bool, bias: bool):
    Utils.initialize_model_parallel()

    input_dim = 100
    output_dim = 100
    num_layers = 10
    _, param_and_grad_buffer = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=bias,
        bucket_size=bucket_size,
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=False,
    )

    actual_numel_in_each_bucket = [
        bucket.numel_unpadded for bucket in param_and_grad_buffer.buckets
    ]
    actual_numel_padded_in_each_bucket = [
        bucket.grad_data.numel() for bucket in param_and_grad_buffer.buckets
    ]

    def _pad_if_needed(numel_unpadded):
        # Want 128-byte alignment for distributed optimizer.
        divisor = math.lcm(parallel_state.get_data_parallel_world_size(), 128)
        if use_distributed_optimizer:
            return math.ceil(numel_unpadded / divisor) * divisor
        return numel_unpadded

    if bucket_size is None:
        # If bucket_size is infinite (None), number of buckets should be 1.
        assert len(param_and_grad_buffer.buckets) == 1
    else:
        # Else, compute number of buckets.
        numel_in_each_bucket = []
        numel_padded_in_each_bucket = []
        numel_in_last_bucket = 0
        for _ in range(num_layers):
            numel_in_last_bucket += input_dim * output_dim
            if bias:
                numel_in_last_bucket += output_dim  # Include bias term.
            if numel_in_last_bucket >= bucket_size:
                numel_in_each_bucket.append(numel_in_last_bucket)
                numel_padded_in_each_bucket.append(_pad_if_needed(numel_in_last_bucket))
                numel_in_last_bucket = 0
        if numel_in_last_bucket > 0:
            numel_in_each_bucket.append(numel_in_last_bucket)
            numel_padded_in_each_bucket.append(_pad_if_needed(numel_in_last_bucket))

        assert len(param_and_grad_buffer.buckets) == len(numel_in_each_bucket)
        assert actual_numel_in_each_bucket == numel_in_each_bucket, (
            f"Number of parameters in each bucket should be {numel_in_each_bucket}, "
            f"but is {actual_numel_in_each_bucket}"
        )
        assert actual_numel_padded_in_each_bucket == numel_padded_in_each_bucket, (
            f"Number of parameters in each padded bucket should be {numel_padded_in_each_bucket}, "
            f"but is {actual_numel_padded_in_each_bucket}"
        )

    Utils.destroy_model_parallel()


@pytest.mark.parametrize("use_distributed_optimizer", [False, True])
@pytest.mark.parametrize("overlap_grad_reduce", [False, True])
def test_grad_sync(use_distributed_optimizer: bool, overlap_grad_reduce: bool):
    Utils.initialize_model_parallel()

    input_dim = 100
    output_dim = 100
    num_layers = 10
    model, param_and_grad_buffer = get_model_and_buffers(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bias=True,
        bucket_size=None,  # Group all params into single bucket.
        use_distributed_optimizer=use_distributed_optimizer,
        overlap_grad_reduce=overlap_grad_reduce,
    )

    param_and_grad_buffer.grad_data.data.fill_(1.0)
    expected_grad_data_value_after_collective = 1
    if torch.distributed.get_rank() == 0 or not use_distributed_optimizer:
        expected_grad_data_value_after_collective = parallel_state.get_data_parallel_world_size()

    params = list(model.parameters())
    for i, param in enumerate(params):
        register_grad_sync_context = (
            contextlib.nullcontext() if overlap_grad_reduce else pytest.raises(AssertionError)
        )
        finish_grad_sync_context = contextlib.nullcontext()
        if i < (len(params) - 1) and overlap_grad_reduce:
            # Can't finish grad sync until all params have been registered ready.
            finish_grad_sync_context = pytest.raises(AssertionError)

        with register_grad_sync_context:
            param_and_grad_buffer.register_grad_ready(param)
        with finish_grad_sync_context:
            # When overlap_grad_reduce is True, this should throw an assertion error until all
            # params in the model have registered their grad above.
            # When overlap_grad_reduce is False, the collective is forced through.
            param_and_grad_buffer.finish_grad_sync()

        expected_grad_data_value = expected_grad_data_value_after_collective
        if overlap_grad_reduce and i < (len(params) - 1):
            expected_grad_data_value = 1
        assert int(param_and_grad_buffer.grad_data[0]) == expected_grad_data_value

        if not overlap_grad_reduce:
            # Reset grad_data for subsequent collectives.
            param_and_grad_buffer.grad_data.data.fill_(1.0)

    Utils.destroy_model_parallel()
